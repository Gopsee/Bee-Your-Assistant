"""Export every Bee conversation, with its full transcript, to bee_conversation/.

Uses the authenticated `bee` CLI (run `bee login` first). Writes one markdown
file per conversation, named <start date>_<id>.md. Re-running overwrites files,
so conversations that were still capturing get refreshed.

    .venv/bin/python export_conversations.py [--out DIR]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def bee(*args):
    out = subprocess.run(["bee", *args, "--json"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def list_conversations():
    convs, cursor, tz = [], None, "UTC"
    while True:
        args = ["conversations", "list", "--limit", "100"]
        if cursor:
            args += ["--cursor", cursor]
        page = bee(*args)
        tz = page.get("timezone") or tz
        convs += page["conversations"]
        cursor = page.get("next_cursor")
        if not cursor or not page["conversations"]:
            return convs, ZoneInfo(tz)


def fmt(ms, tz, pattern="%Y-%m-%d %H:%M:%S"):
    return datetime.fromtimestamp(ms / 1000, tz).strftime(pattern) if ms else "-"


def render(conv, transcript, tz):
    lines = [
        f"# Conversation {conv['id']}",
        "",
        f"- Start: {fmt(conv.get('start_time'), tz)}",
        f"- End: {fmt(conv.get('end_time'), tz)}",
        f"- State: {conv.get('state')}",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "bee_conversation"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    convs, tz = list_conversations()
    print(f"Found {len(convs)} conversations")
    failed = 0
    for i, conv in enumerate(convs, 1):
        cid = conv["id"]
        try:
            transcript = bee("conversations", "transcript", str(cid)).get("transcript", [])
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"  [{i}/{len(convs)}] {cid}: failed ({e.stderr.strip()[:200]})", file=sys.stderr)
            continue
        name = f"{fmt(conv.get('start_time'), tz, '%Y-%m-%d_%H%M')}_{cid}.md"
        (out / name).write_text(render(conv, transcript, tz), encoding="utf-8")
        print(f"  [{i}/{len(convs)}] {name} ({len(transcript)} utterances)")
    print(f"Done: {len(convs) - failed} written to {out}, {failed} failed")


if __name__ == "__main__":
    main()
