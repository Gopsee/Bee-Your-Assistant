"""
Local copy of every Bee conversation, with its full transcript, in
bee_conversation/ (one markdown file per conversation, <start>_<id>.md).

sync() pulls new conversations from the Bee account through the authenticated
`bee` CLI and refreshes ones that were still being recorded. Local files are
never deleted by a sync, so the copy outlives data cleared from the Bee app.
The agent answers questions about the user's day from these files.
"""
import json
import re
import subprocess
import threading
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CONVERSATION_DIR = Path(__file__).parent / "bee_conversation"
SYNC_EVERY = 60          # seconds between syncs unless forced
CLI_TIMEOUT = 60
TRANSCRIPT_CHARS = 4000  # per conversation handed to the model
INDEX_SUMMARY_CHARS = 700  # of each conversation's summary in the per-turn list


def bee(*args):
    out = subprocess.run(["bee", *args, "--json"], capture_output=True, text=True, check=True, timeout=CLI_TIMEOUT)
    return json.loads(out.stdout) if out.stdout.strip() else {}


def fmt(ms, tz, pattern="%Y-%m-%d %H:%M:%S"):
    return datetime.fromtimestamp(int(ms) / 1000, tz).strftime(pattern) if ms else "-"


def render(conv, transcript, tz):
    lines = [
        f"# Conversation {conv['id']}",
        "",
        f"- Start: {fmt(conv.get('start_time'), tz)}",
        f"- End: {fmt(conv.get('end_time'), tz)}",
        f"- State: {conv.get('state')}",
        f"- Updated: {conv.get('updated_at') or '-'}",
        f"- Utterances: {len(transcript)}",
        "",
    ]
    if conv.get("short_summary"):
        lines += [f"**Short summary:** {conv['short_summary']}", ""]
    if conv.get("summary"):
        lines += [conv["summary"].strip(), ""]
    lines += ["## Transcript", ""]
    for u in sorted(transcript, key=lambda u: (u.get("spoken_at") or 0, u.get("start") or 0)):
        lines.append(f"[{fmt(u.get('spoken_at'), tz, '%H:%M:%S')}] **{u.get('speaker') or 'Unknown'}:** {u.get('text', '').strip()}")
    return "\n".join(lines) + "\n"


def _field(text: str, name: str) -> str:
    m = re.search(rf"^- {name}: (.*)$", text, re.M)
    return m.group(1).strip() if m else ""


class ConversationStore:
    def __init__(self, directory: Path = CONVERSATION_DIR) -> None:
        self.dir = Path(directory)
        self._lock = threading.Lock()
        self._last_sync = 0.0

    # ---- sync from Bee -----------------------------------------------------

    def _list(self):
        convs, cursor, tz = [], None, "UTC"
        while True:
            page = bee("conversations", "list", "--limit", "100", *(["--cursor", cursor] if cursor else []))
            tz = page.get("timezone") or tz
            convs += page.get("conversations", [])
            cursor = page.get("next_cursor")
            if not cursor or not page.get("conversations"):
                return convs, ZoneInfo(tz)

    def _local(self, cid) -> Path | None:
        return next(self.dir.glob(f"*_{cid}.md"), None)

    def _up_to_date(self, conv, path: Path | None) -> bool:
        if path is None:
            return False
        text = path.read_text(encoding="utf-8")
        # Older exports have no Updated line; a completed one is final anyway.
        return ((_field(text, "State") == "COMPLETED" and conv.get("state") == "COMPLETED")
                or _field(text, "Updated") == str(conv.get("updated_at")))

    def sync(self, force: bool = False, log=print) -> str:
        """Copy new or changed conversations from Bee. Returns a one-line report;
        never raises, so a missing CLI or no network leaves the local copy usable."""
        with self._lock:
            if not force and time.time() - self._last_sync < SYNC_EVERY:
                return "recently synced"
            self._last_sync = time.time()
            try:
                convs, tz = self._list()
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                return f"sync failed: {type(exc).__name__}"
            self.dir.mkdir(parents=True, exist_ok=True)
            written = failed = 0
            for conv in convs:
                cid, old = conv["id"], self._local(conv["id"])
                if not force and self._up_to_date(conv, old):
                    continue
                try:
                    transcript = bee("conversations", "transcript", str(cid)).get("transcript", [])
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    failed += 1
                    log(f"  {cid}: failed ({type(exc).__name__})")
                    continue
                name = f"{fmt(conv.get('start_time'), tz, '%Y-%m-%d_%H%M')}_{cid}.md"
                (self.dir / name).write_text(render(conv, transcript, tz), encoding="utf-8")
                if old and old.name != name:
                    old.unlink()
                written += 1
            return f"{len(convs)} in Bee, {written} written, {failed} failed"

    # ---- read the local copy -----------------------------------------------

    def files_on(self, day: date) -> list[Path]:
        return sorted(self.dir.glob(f"{day.isoformat()}_*.md"))

    def day_index(self, day: date) -> str:
        """Each conversation on that day with its summary, for the opening message.
        The title alone is too vague ("Family Care During Memory Decline" for a
        call from Daniel), and a small model rarely calls a tool to look further."""
        lines = []
        for path in self.files_on(day):
            text = path.read_text(encoding="utf-8")
            start, end = _field(text, "Start")[11:16], _field(text, "End")[11:16]
            m = re.search(r"^\*\*Short summary:\*\* (.*)$", text, re.M)
            what = m.group(1) if m else "no summary yet"
            live = "" if _field(text, "State") == "COMPLETED" else " (still being recorded)"
            lines.append(f"- {start}{'-' + end if end else ''}: {what}{live}")
            m = re.search(r"^#+ Summary\s*\n(.*?)(?=^#|\Z)", text, re.M | re.S)
            if m and m.group(1).strip():
                summary = " ".join(m.group(1).split())
                if len(summary) > INDEX_SUMMARY_CHARS:
                    summary = summary[:INDEX_SUMMARY_CHARS].rsplit(" ", 1)[0] + " ..."
                lines.append(f"  {summary}")
        return "\n".join(lines)

    def read(self, day: date, person: str = "") -> str:
        """Summaries and transcripts for a day, optionally only those mentioning a person."""
        parts = []
        for path in self.files_on(day):
            text = path.read_text(encoding="utf-8")
            if person and person.lower() not in text.lower():
                continue
            head, _, transcript = text.partition("## Transcript")
            # Start the excerpt at the line where the person first comes up.
            at = transcript.lower().find(person.lower()) if person else -1
            if at > 0:
                transcript = "\n[... earlier transcript cut ...]\n" + transcript[transcript.rfind("\n", 0, at) + 1:]
            if len(transcript) > TRANSCRIPT_CHARS:
                transcript = transcript[:TRANSCRIPT_CHARS] + "\n[... transcript cut ...]"
            parts.append(head.strip() + "\n## Transcript" + transcript)
        if not parts:
            who = f" mentioning {person}" if person else ""
            return f"No conversations{who} saved for {day.isoformat()}."
        return "\n\n".join(parts)
