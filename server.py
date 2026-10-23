"""
Web front end for the Bee assistant: serves static/index.html and wraps one
BeeSession (bee.py) behind a small JSON API.

    python server.py                       # http://127.0.0.1:8000
    python server.py --port 9000
    python server.py --memory-dir /tmp/scratch-memory --log-dir /tmp/scratch-logs
    BEE_PASSWORD=... python server.py --host 0.0.0.0   # reachable from other devices

Off localhost the server needs a password (BEE_PASSWORD): the browser asks for
it once (HTTP Basic auth, any user name). Without one it refuses to start.

Every few minutes the server copies new Bee conversations, with transcripts,
to bee_conversation/ (conversation_store.py); the assistant answers from them.

The mobile page (static/mobile.html, at /mobile) uses the /api/m/... routes:
Bee account data through the bee CLI (bee_data.py), plus edits to persistent
and intermediate memory.

The agent is not thread-safe, so every call into the session (start, ask, end)
runs on one worker thread. A chat message becomes a job the page polls; when a
write tool needs approval the job waits until the page answers Allow/Decline.
"""
import argparse
import base64
import os
import secrets
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from strands.hooks import AfterModelCallEvent, AfterToolCallEvent, HookProvider, HookRegistry

from bee import CAREGIVER, CAREGIVER_PHONE, LOG_DIR, BeeSession, bedrock_model
from bee_data import BeeData, BeeError
from conversation_store import CONVERSATION_DIR, ConversationStore
from likes_to_know import answers_for
from memory import (CATEGORIES, MEMORY_DIR, PROMOTE_AFTER, QUESTION_PROMOTE_AFTER, QUESTION_WINDOW_DAYS, Memory,
                    _read_json, _write_json, load_intermediate, load_questions, settle_card)
from topic_screen import NOT_RELATED, TopicScreen

STATIC_DIR = Path(__file__).parent / "static"
CONFIRM_TIMEOUT = 600  # seconds to wait for Allow/Decline before declining
SYNC_INTERVAL = 300    # seconds between background copies of Bee conversations
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


def password_ok(header: str, password: str) -> bool:
    """Check an HTTP Basic Authorization header; the user name is ignored."""
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        given = base64.b64decode(encoded).decode().partition(":")[2]
    except ValueError:
        return False
    return secrets.compare_digest(given.encode(), password.encode())


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
    def __init__(self, memory_dir: Path, log_dir: Path, conversation_dir: Path = CONVERSATION_DIR) -> None:
        self.memory_dir, self.log_dir = memory_dir, log_dir
        self.conversations = ConversationStore(conversation_dir)
        self.stopping = threading.Event()
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.jobs: dict[str, Job] = {}
        self.current: Job | None = None
        self.screen: TopicScreen | None = None
        self.session: BeeSession | None = None
        self.status = "loading"  # loading | ready | busy | error
        self.error = ""
        # Guards persistent.json between the mobile page's edits and the
        # end-of-session merge (which re-reads the file, so edits survive it).
        self.memory_lock = threading.Lock()

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
            with self.memory_lock:  # start_session merges a session left over from a crash
                self.session = BeeSession(self.screen, self.memory_dir, self.log_dir, confirm=self._confirm,
                                          conversations=self.conversations)
            self.session.agent.hooks.add_hook(JobRecorder(self))
            print(f"Logging conversation to {self.session.logger.path}")
            self.status, self.error = "ready", ""
        except Exception as exc:  # e.g. Bedrock unreachable or the Bee MCP server not running
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
                with self.memory_lock:
                    self.session.close()
            self.session = None
            self._start_session()
            job.status = "done" if self.status == "ready" else "error"
            job.error = self.error
        finally:
            self.current = None

    def _delete_data(self, job: Job) -> None:
        """Delete every local copy of the user's data and start an empty session.
        The current session is dropped, not saved, so nothing is merged back.
        Data in the Bee account is untouched (deleted in the Bee app)."""
        self.current = job
        try:
            if self.session:
                self.session.logger.close()
                self.session = None
            with self.memory_lock:
                memory_files = [p for p in self.memory_dir.glob("*.json") if p.is_file()]
                logs = [p for p in self.log_dir.glob("*.log") if p.is_file()]
                copies = list(self.conversations.dir.glob("*.md"))
                for path in memory_files + logs + copies:
                    path.unlink()
            # Finished jobs still hold old replies in server memory.
            self.jobs = {k: j for k, j in self.jobs.items() if j.status not in ("done", "error")}
            self._start_session()
            job.reply = (f"Deleted {len(memory_files)} memory files, {len(logs)} conversation logs "
                         f"and {len(copies)} saved Bee transcripts.")
            job.status = "done" if self.status == "ready" else "error"
            job.error = self.error
        except Exception as exc:
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
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

    def sync_loop(self) -> None:
        """Keep the local copy of Bee conversations current while the server runs."""
        while not self.stopping.is_set():
            print(f"Bee conversations: {self.conversations.sync(force=True)}")
            self.stopping.wait(SYNC_INTERVAL)

    def shutdown(self) -> None:
        self.stopping.set()
        for job in self.jobs.values():  # unblock a job waiting for Allow/Decline
            job.answer = "n"
            job.answered.set()
        if self.session:
            self.worker.submit(self._close_locked).result()
        self.worker.shutdown()

    def _close_locked(self) -> None:
        with self.memory_lock:
            self.session.close()

    def edit_memory(self, edit: "MemoryEditIn") -> dict:
        """Add, change or delete one persistent-memory entry. Backs up the old file first."""
        path = self.memory_dir / "persistent.json"
        with self.memory_lock:
            data = _read_json(path, {})
            data.pop("updated_at", None)
            mem = Memory(**data)
            if edit.category == "profile":
                key, value = edit.key.strip(), edit.value.strip()
                if edit.action != "add" and edit.old not in mem.profile:
                    raise HTTPException(404, f"No profile entry {edit.old!r}")
                if edit.action == "delete":
                    del mem.profile[edit.old]
                else:
                    if not key or not value:
                        raise HTTPException(400, "Profile entries need a name and a value")
                    if edit.action == "edit" and edit.old != key:
                        del mem.profile[edit.old]
                    mem.profile[key] = value
            else:
                items, value = getattr(mem, edit.category), edit.value.strip()
                if edit.action != "add" and edit.old not in items:
                    raise HTTPException(404, "That entry is no longer in memory")
                if edit.action != "delete" and not value:
                    raise HTTPException(400, "Empty entry")
                if edit.action == "add":
                    items.append(value)
                elif edit.action == "edit":
                    items[items.index(edit.old)] = value
                else:
                    items.remove(edit.old)
            if path.exists():
                shutil.copy(path, path.with_suffix(".prev.json"))
            _write_json(path, {"updated_at": datetime.now().isoformat(timespec="seconds"), **mem.model_dump()})
            return _read_json(path, {})


class ChatIn(BaseModel):
    message: str


class ConfirmIn(BaseModel):
    allow: bool


class TodoIn(BaseModel):
    text: str
    alarm_at: str | None = None  # ISO 8601


class TodoPatch(BaseModel):
    text: str | None = None
    completed: bool | None = None
    alarm_at: str | None = None
    clear_alarm: bool = False


class SuggestionIn(BaseModel):
    accept: bool


class FactIn(BaseModel):
    text: str


class FactPatch(BaseModel):
    text: str | None = None
    confirmed: bool | None = None


class CardIn(BaseModel):
    keep: bool  # True: move to persistent memory now; False: forget it


class MemoryEditIn(BaseModel):
    category: Literal[CATEGORIES + ("likes_to_know",)]
    action: Literal["add", "edit", "delete"]
    old: str = ""    # the entry being edited or deleted (the key, for profile)
    key: str = ""    # profile only
    value: str = ""


def create_app(memory_dir: Path = MEMORY_DIR, log_dir: Path = LOG_DIR,
               conversation_dir: Path = CONVERSATION_DIR, password: str = "") -> FastAPI:
    bee = BeeApp(memory_dir, log_dir, conversation_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        bee.worker.submit(bee._start_session)
        threading.Thread(target=bee.sync_loop, daemon=True).start()
        yield
        bee.shutdown()

    app = FastAPI(title="Bee Assistant", lifespan=lifespan)

    if password:
        @app.middleware("http")
        async def require_password(request: Request, call_next):
            if password_ok(request.headers.get("authorization", ""), password):
                return await call_next(request)
            return Response("Password required", status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Bee Assistant"'})

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

    @app.post("/api/data/delete")
    def delete_data():
        """Delete all local memory, conversation logs and saved Bee transcripts (not the Bee account's data)."""
        return bee.submit(bee._delete_data).view()

    @app.post("/api/retry")
    def retry():
        if bee.status == "error":
            bee.worker.submit(bee._start_session)
        return {"ok": True}

    # ---- mobile page --------------------------------------------------------

    data = BeeData()

    def call(fn, *args):
        try:
            return fn(*args)
        except BeeError as exc:
            raise HTTPException(502, f"Bee: {exc}")

    @app.get("/mobile")
    def mobile():
        return FileResponse(STATIC_DIR / "mobile.html")

    @app.get("/api/m/highlights")
    def highlights():
        return call(data.highlights)

    @app.get("/api/m/conversations")
    def conversations(cursor: str | None = None):
        return call(data.conversations, cursor)

    @app.get("/api/m/conversations/{cid}")
    def conversation(cid: int):
        return call(data.conversation, cid)

    @app.get("/api/m/todos")
    def todos():
        return call(data.todos)

    @app.post("/api/m/todos")
    def todo_create(body: TodoIn):
        if not body.text.strip():
            raise HTTPException(400, "Empty to-do")
        return call(data.todo_create, body.text.strip(), body.alarm_at)

    @app.patch("/api/m/todos/{tid}")
    def todo_update(tid: int, body: TodoPatch):
        return call(data.todo_update, tid, body.text, body.completed, body.alarm_at, body.clear_alarm)

    @app.delete("/api/m/todos/{tid}")
    def todo_delete(tid: int):
        return call(data.todo_delete, tid)

    @app.post("/api/m/suggestions/{sid}")
    def suggestion(sid: int, body: SuggestionIn):
        return call(data.suggestion, sid, body.accept)

    @app.get("/api/m/profile")
    def profile():
        return {"memory": _read_json(memory_dir / "persistent.json", {}), "facts": call(data.facts)}

    @app.post("/api/m/memory")
    def memory_edit(body: MemoryEditIn):
        return bee.edit_memory(body)

    @app.get("/api/m/learning")
    def learning():
        return {"items": load_intermediate(memory_dir), "promote_after": PROMOTE_AFTER,
                "questions": load_questions(memory_dir), "question_promote_after": QUESTION_PROMOTE_AFTER,
                "question_window_days": QUESTION_WINDOW_DAYS}

    @app.get("/api/m/likes-to-know")
    def likes_to_know(refresh: bool = False):
        """Questions the user asks often, with today's prepared answers."""
        try:
            daily = data.highlights()["daily"]
            today = f"{daily[0].get('short_summary', '')}\n{(daily[0].get('summary') or '')[:2000]}" if daily else ""
        except BeeError:
            today = ""  # answer from memory and the date alone
        try:
            return answers_for(memory_dir, bedrock_model, today, CAREGIVER, CAREGIVER_PHONE, refresh)
        except Exception as exc:
            raise HTTPException(502, f"Couldn't prepare answers: {exc}")

    @app.post("/api/m/learning/{card_id}")
    def learning_settle(card_id: str, body: CardIn):
        with bee.memory_lock:
            try:
                items = settle_card(memory_dir, card_id, body.keep)
            except KeyError:
                raise HTTPException(404, "That card is no longer in intermediate memory")
        return {"items": items, "promote_after": PROMOTE_AFTER}

    @app.post("/api/m/facts")
    def fact_create(body: FactIn):
        if not body.text.strip():
            raise HTTPException(400, "Empty fact")
        return call(data.fact_create, body.text.strip())

    @app.patch("/api/m/facts/{fid}")
    def fact_update(fid: int, body: FactPatch):
        return call(data.fact_update, fid, body.text, body.confirmed)

    @app.delete("/api/m/facts/{fid}")
    def fact_delete(fid: int):
        return call(data.fact_delete, fid)

    @app.get("/api/m/search")
    def search(q: str):
        q = q.strip()
        if not q:
            return {"bee": [], "memory": [], "todos": []}
        words = q.lower().split()
        hit = lambda text: all(w in text.lower() for w in words)
        mem = _read_json(memory_dir / "persistent.json", {})
        memory = [{"category": "profile", "text": f"{k}: {v}"} for k, v in mem.get("profile", {}).items() if hit(f"{k} {v}")]
        memory += [{"category": c, "text": x} for c in CATEGORIES[1:] for x in mem.get(c, []) if hit(x)]
        todos = [t for t in call(data.todos)["todos"] if hit(t.get("text", ""))]
        return {"bee": call(data.search, q), "memory": memory, "todos": todos}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--memory-dir", type=Path, default=MEMORY_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--conversation-dir", type=Path, default=CONVERSATION_DIR)
    args = parser.parse_args()
    if not os.environ.get("BEE_MCP_TOKEN"):
        parser.error("set BEE_MCP_TOKEN to the token the Bee MCP server was started with (see README.md)")
    password = os.environ.get("BEE_PASSWORD", "")
    if args.host not in LOCAL_HOSTS and not password:
        parser.error(f"--host {args.host} makes the app reachable from other devices; set BEE_PASSWORD first")
    uvicorn.run(create_app(args.memory_dir, args.log_dir, args.conversation_dir, password),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
