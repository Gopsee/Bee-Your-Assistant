"""Export every Bee conversation, with its full transcript, to bee_conversation/.

Uses the authenticated `bee` CLI (run `bee login` first). Writes one markdown
file per conversation, named <start date>_<id>.md, and rewrites them all.
The server and the assistant keep this copy current on their own (see
conversation_store.py); run this for a full refresh.

    .venv/bin/python export_conversations.py [--out DIR]
"""

import argparse

from conversation_store import CONVERSATION_DIR, ConversationStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CONVERSATION_DIR))
    store = ConversationStore(ap.parse_args().out)
    print(f"Done: {store.sync(force=True)} ({store.dir})")


if __name__ == "__main__":
    main()
