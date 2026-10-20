"""
Web front end for the Bee assistant: serves static/index.html and wraps one
BeeSession (bee.py) behind a small JSON API.

    python server.py                       # http://127.0.0.1:8000
    python server.py --port 9000
    python server.py --memory-dir /tmp/scratch-memory --log-dir /tmp/scratch-logs

The agent is not thread-safe, so every call into the session (start, ask, end)
runs on one worker thread. A chat message becomes a job the page polls; when a
write tool needs approval the job waits until the page answers Allow/Decline.
"""
import argparse
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from strands.hooks import AfterModelCallEvent, AfterToolCallEvent, HookProvider, HookRegistry

from bee import LOG_DIR, BeeSession
from memory import MEMORY_DIR, _read_json
from topic_screen import NOT_RELATED, TopicScreen

STATIC_DIR = Path(__file__).parent / "static"
CONFIRM_TIMEOUT = 600  # seconds to wait for Allow/Decline before declining


class Job:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex
        self.status = "running"  # running | confirm | done | error
        self.reply = ""
        self.error = ""
        self.off_topic = False
        self.tools: list[dict] = []
        self.recalled: list[dict] = []
        self.tokens: dict | None = None
        self.pending: dict | None = None
        self.answer = ""
        self.answered = threading.Event()

    def view(self) -> dict:
        return {k: getattr(self, k) for k in ("id", "status", "reply", "error", "off_topic", "tools", "recalled", "tokens", "pending")}


class JobRecorder(HookProvider):
    """Record tool calls and token usage on the job that is running."""

    def __init__(self, app: "BeeApp") -> None:
        self.app = app

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolCallEvent, self.tool)
        registry.add_callback(AfterModelCallEvent, self.model)

    def tool(self, event: AfterToolCallEvent) -> None:
        if job := self.app.current:
            status = "declined" if event.cancel_message else event.result["status"]
            job.tools.append({"name": event.tool_use["name"], "status": status})

    def model(self, event: AfterModelCallEvent) -> None:
        if (job := self.app.current) and event.stop_response is not None:
            usage = event.stop_response.message.get("metadata", {}).get("usage", {})
            job.tokens = {"input": usage.get("inputTokens"), "output": usage.get("outputTokens")}


class BeeApp:
    def __init__(self, memory_dir: Path, log_dir: Path) -> None:
        self.memory_dir, self.log_dir = memory_dir, log_dir
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None
        self.screen: TopicScreen | None = None
        self.session: BeeSession | None = None
        self.status = "loading"  # loading | ready | busy | error
        self.error = ""

    # ---- runs on the worker thread -----------------------------------------

    def _confirm(self, prompt: str) -> str:
        job = self.current
        if job is None:
            return "n"
        job.pending = self.session.confirm_writes.pending if self.session else None
        job.answered.clear()
        job.status = "confirm"
        answered = job.answered.wait(CONFIRM_TIMEOUT)
        job.status, job.pending = "running", None
        return job.answer if answered else "n"

    def _start_session(self) -> None:
        self.status = "loading"
        try:
            if self.screen is None:
                print("Loading topic screen (strands-decider)...")
                self.screen = TopicScreen()
            self.session = BeeSession(self.screen, self.memory_dir, self.log_dir, confirm=self._confirm)
            self.session.agent.hooks.add_hook(JobRecorder(self))
            print(f"Logging conversation to {self.session.logger.path}")
            self.status, self.error = "ready", ""
        except Exception as exc:  # e.g. Ollama or the Bee MCP server not running
            self.session = None
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"

    def _ask(self, job: Job, message: str) -> None:
        self.current = job
        try:
            self.session.last_recalled = []
            job.reply = self.session.ask(message)
            job.recalled = [{"topic": n, "p": round(p, 2)} for n, p in self.session.last_recalled]
            job.off_topic = job.reply == NOT_RELATED
            job.status = "done"
        except Exception as exc:
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            self.current = None

    def _end_session(self, job: Job) -> None:
        self.current = job
        try:
            if self.session:
                self.session.close()
            self.session = None
            self._start_session()
            job.status = "done" if self.status == "ready" else "error"
            job.error = self.error
        finally:
            self.current = None

    # ---- called from request handlers --------------------------------------

    def submit(self, fn, *args) -> Job:
        job = Job()
        self.jobs[job.id] = job
        self.worker.submit(fn, job, *args)
        return job

    def state(self) -> dict:
        memory = _read_json(self.memory_dir / "persistent.json", {})
        session = self.session.memory.session if self.session else {}
        return {
            "status": self.status,
            "error": self.error,
            "busy": self.current is not None,
            "memory": memory,
            "session": {
                "started_at": session.get("started_at"),
                "notes": session.get("notes", []),
                "topics": {k: v.get("summary", "") for k, v in session.get("topics", {}).items()},
                "turns": len(session.get("transcript", [])),
            },
            "log": str(self.session.logger.path) if self.session else None,
        }

    def shutdown(self) -> None:
        for job in self.jobs.values():  # unblock a job waiting for Allow/Decline
            job.answer = "n"
            job.answered.set()
        if self.session:
            self.worker.submit(self.session.close).result()
        self.worker.shutdown()


class ChatIn(BaseModel):
    message: str


class ConfirmIn(BaseModel):
    allow: bool


def create_app(memory_dir: Path = MEMORY_DIR, log_dir: Path = LOG_DIR) -> FastAPI:
    bee = BeeApp(memory_dir, log_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        bee.worker.submit(bee._start_session)
        yield
        bee.shutdown()

    app = FastAPI(title="Bee Assistant", lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    def state():
        return bee.state()

    @app.post("/api/chat")
    def chat(body: ChatIn):
        message = body.message.strip()
        if not message:
            raise HTTPException(400, "Empty message")
        if bee.session is None:
            raise HTTPException(503, bee.error or "Assistant is still loading")
        return bee.submit(bee._ask, message).view()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        if job_id not in bee.jobs:
            raise HTTPException(404, "Unknown job")
        return bee.jobs[job_id].view()

    @app.post("/api/jobs/{job_id}/confirm")
    def confirm(job_id: str, body: ConfirmIn):
        job = bee.jobs.get(job_id)
        if job is None or job.status != "confirm":
            raise HTTPException(409, "Nothing is waiting for confirmation")
        job.answer = "y" if body.allow else "n"
        job.answered.set()
        return {"ok": True}

    @app.post("/api/session/end")
    def end_session():
        """Save this session into long-term memory and start a fresh one."""
        return bee.submit(bee._end_session).view()

    @app.post("/api/retry")
    def retry():
        if bee.status == "error":
            bee.worker.submit(bee._start_session)
        return {"ok": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--memory-dir", type=Path, default=MEMORY_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    args = parser.parse_args()
    uvicorn.run(create_app(args.memory_dir, args.log_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
