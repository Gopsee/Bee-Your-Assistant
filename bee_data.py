"""
Read and write Bee account data through the authenticated `bee` CLI, for the
mobile page (static/mobile.html). The CLI holds the token, so it never reaches
the browser. Reads are cached briefly; any write clears the cache.
"""
import json
import re
import subprocess
import threading
import time

CACHE_SECONDS = 60
CLI_TIMEOUT = 60


class BeeError(RuntimeError):
    pass


class BeeData:
    def __init__(self) -> None:
        self._cache: dict[tuple, tuple[float, object]] = {}
        self._lock = threading.Lock()

    # ---- plumbing -----------------------------------------------------------

    def _run(self, *args: str):
        try:
            out = subprocess.run(["bee", *args, "--json"], capture_output=True, text=True, timeout=CLI_TIMEOUT)
        except FileNotFoundError:
            raise BeeError("The bee CLI is not installed or not on PATH")
        except subprocess.TimeoutExpired:
            raise BeeError(f"bee {args[0]} timed out")
        if out.returncode != 0:
            raise BeeError((out.stderr or out.stdout).strip()[:300] or f"bee {args[0]} failed")
        return json.loads(out.stdout) if out.stdout.strip() else {}

    def _cached(self, *args: str):
        with self._lock:
            hit = self._cache.get(args)
            if hit and time.time() - hit[0] < CACHE_SECONDS:
                return hit[1]
        data = self._run(*args)
        with self._lock:
            self._cache[args] = (time.time(), data)
        return data

    def _write(self, *args: str):
        data = self._run(*args)
        with self._lock:
            self._cache.clear()
        return data

    def _all_pages(self, *args: str, key: str, limit: int = 100) -> list:
        items, cursor = [], None
        while True:
            page = self._cached(*args, "--limit", str(limit), *(["--cursor", cursor] if cursor else []))
            items += page.get(key, [])
            cursor = page.get("next_cursor")
            if not cursor or not page.get(key):
                return items

    # ---- reads --------------------------------------------------------------

    def highlights(self) -> dict:
        daily = self._cached("daily", "list", "--limit", "7").get("daily_summaries", [])
        insights = self._cached("insights", "list", "--limit", "10").get("insights", [])
        convs = self._cached("conversations", "list", "--limit", "10").get("conversations", [])
        takeaways, actions = [], []
        for c in convs:
            title = c.get("short_summary") or f"Conversation {c['id']}"
            takeaways += [{"text": t, "conversation": c["id"], "title": title} for t in _section(c.get("summary"), "key take")]
            actions += [{"text": t, "conversation": c["id"], "title": title} for t in _section(c.get("summary"), "action item")]
        return {"daily": daily, "insights": insights, "conversations": convs,
                "takeaways": takeaways[:12], "actions": actions[:12]}

    def conversations(self, cursor: str | None = None) -> dict:
        args = ["conversations", "list", "--limit", "20"] + (["--cursor", cursor] if cursor else [])
        page = self._cached(*args)
        return {"conversations": page.get("conversations", []), "next_cursor": page.get("next_cursor")}

    def conversation(self, cid: int) -> dict:
        conv = self._cached("conversations", "get", str(cid)).get("conversation", {})
        conv.pop("transcriptions", None)  # the transcript call below is the sorted, complete version
        transcript = self._cached("conversations", "transcript", str(cid)).get("transcript", [])
        transcript.sort(key=lambda u: (u.get("spoken_at") or 0, u.get("start") or 0))
        return {"conversation": conv, "transcript": transcript}

    def todos(self) -> dict:
        todos = self._all_pages("todos", "list", key="todos")
        suggestions = self._cached("todos", "suggestions", "--limit", "10").get("todoSuggestions", [])
        return {"todos": todos, "suggestions": suggestions}

    def facts(self) -> list:
        return self._all_pages("facts", "list", key="facts")

    def search(self, query: str) -> list:
        return self._run("search", "--query", query, "--filter", "all", "--limit", "20").get("results", [])

    # ---- writes -------------------------------------------------------------

    def todo_create(self, text: str, alarm_at: str | None) -> dict:
        return self._write("todos", "create", "--text", text, *(["--alarm-at", alarm_at] if alarm_at else []))

    def todo_update(self, tid: int, text: str | None, completed: bool | None, alarm_at: str | None, clear_alarm: bool) -> dict:
        args = ["todos", "update", str(tid)]
        if text is not None:
            args += ["--text", text]
        if completed is not None:
            args += ["--completed", "true" if completed else "false"]
        if clear_alarm:
            args.append("--clear-alarm")
        elif alarm_at:
            args += ["--alarm-at", alarm_at]
        return self._write(*args)

    def todo_delete(self, tid: int) -> dict:
        return self._write("todos", "delete", str(tid))

    def suggestion(self, sid: int, accept: bool) -> dict:
        return self._write("todos", "accept-suggestion" if accept else "dismiss-suggestion", str(sid))

    def fact_create(self, text: str) -> dict:
        return self._write("facts", "create", "--text", text)

    def fact_update(self, fid: int, text: str | None, confirmed: bool | None) -> dict:
        args = ["facts", "update", str(fid)]
        if text is not None:
            args += ["--text", text]
        if confirmed is not None:
            args += ["--confirmed", "true" if confirmed else "false"]
        return self._write(*args)

    def fact_delete(self, fid: int) -> dict:
        return self._write("facts", "delete", str(fid))


def _section(summary: str | None, heading: str) -> list[str]:
    """Bullets under the first markdown heading containing `heading` (case-insensitive)."""
    if not summary:
        return []
    items, inside = [], False
    for line in summary.splitlines():
        if line.lstrip().startswith("#"):
            inside = heading in line.lower()
        elif inside and (m := re.match(r"\s*[-*]\s+(.*)", line)):
            items.append(m.group(1).strip())
    return items
