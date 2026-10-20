"""
Two-tier memory for the Bee agent.

- memory/session.json:    notes + transcript for the current session only.
- memory/persistent.json: organized long-term memory, shared across sessions.

At the end of a session the session memory is merged into persistent memory,
then cleared. Existing memory is only changed explicitly (the remember tool's
``replaces``); the LLM can only add new, grounded facts. Long chats are kept
small by offloading old exchanges into session memory by topic.
"""
import asyncio
import json
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from strands import tool
from strands.agent.conversation_manager import ConversationManager

MEMORY_DIR = Path(__file__).parent / "memory"
CATEGORIES = ("profile", "preferences", "people", "routines", "facts")
RECALL_CHARS = 4000  # max characters of one topic preloaded into the context


class Memory(BaseModel):
    """Organized long-term memory about the user."""

    profile: dict[str, str] = Field(
        default_factory=dict,
        description="Stable personal details as key/value, e.g. name, location, job.",
    )
    preferences: list[str] = Field(default_factory=list, description="Likes, dislikes, how the user wants answers.")
    people: list[str] = Field(default_factory=list, description="People in the user's life and how they relate.")
    routines: list[str] = Field(default_factory=list, description="Habits, schedules, recurring events.")
    facts: list[str] = Field(default_factory=list, description="Other durable facts worth remembering.")


class MemoryItem(BaseModel):
    category: str = Field(description="One of preferences, people, routines, facts.")
    item: str


class MemoryUpdate(BaseModel):
    """New facts found in a session. Additions only: never edits existing memory."""

    profile: dict[str, str] = Field(default_factory=dict)
    add: list[MemoryItem] = Field(default_factory=list)


_STOPWORDS = {"that", "this", "with", "from", "have", "will", "they", "their", "there", "about",
              "user", "user's", "likes", "wants", "would", "into", "when", "what", "also", "been",
              "the", "and", "for", "has", "his", "her", "she", "are", "was", "who",
              "now", "today", "anymore", "still", "actually", "really", "instead"}


def _content_words(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOPWORDS}


# Framing words in memory items ("interested in X", "lives near Y") that say nothing
# about the subject itself, so they don't count towards a memory match.
_FRAMING = {w[:5] for w in ("interested", "prefers", "lives", "near", "moving", "likes", "enjoys",
                             "loves", "dislikes", "hates", "staying", "favourite", "favorite")}


def _mentions(message: str, item: str) -> bool:
    """True if ``message`` names the subject of a memory item: at least two of the
    item's subject words (or its only one), so "notebook" alone doesn't match
    "interested in Notebook LLM" but "what is notebook LLM?" does."""
    subject = _content_words(item) - _FRAMING
    return bool(subject) and len(subject & _content_words(message)) >= min(2, len(subject))


def _sentences(text: str) -> list[str]:
    text = text.replace("\u2019", "'")  # curly apostrophes, as typed by phones and some models
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]


def _statements(said: list[str]) -> list[str]:
    """What the user stated, one entry per message, with questions removed.
    Questions say nothing about the user: "Long or short answers?" must not
    become "interested in long answers"."""
    kept = (" ".join(s for s in _sentences(text) if not s.endswith("?")) for text in said)
    return [k for k in kept if k]


def _grounded(text: str, statements: list[str]) -> bool:
    """True if most content words in ``text`` appear in one message the user
    sent. Stops the model inventing facts or stitching words from several turns."""
    words = _content_words(text)
    return not words or any(len(words & _content_words(s)) / len(words) >= 0.5 for s in statements)


def _known(mem: Memory) -> set[str]:
    return {x.strip().lower() for c in CATEGORIES[1:] for x in getattr(mem, c)}


def _add_item(mem: Memory, category: str, item: str) -> bool:
    item = item.strip()
    if not item or item.lower() in _known(mem):
        return False
    getattr(mem, category if category in CATEGORIES[1:] else "facts").append(item)
    return True


def _apply_notes(mem: Memory, notes: list[dict]) -> None:
    """Apply what the agent saved with the remember tool. These are explicit,
    so they may set profile values and replace existing items."""
    for n in notes:
        note, category, old = n["note"].strip(), n["category"], (n.get("replaces") or "").strip().lower()
        if category == "profile" and ":" in note:
            key, value = (part.strip() for part in note.split(":", 1))
            if key and value:
                mem.profile[key.lower()] = value
            continue
        if category == "profile":
            category = "facts"
        if old:
            for c in CATEGORIES[1:]:
                items = getattr(mem, c)
                hits = [i for i, x in enumerate(items) if x.strip().lower() == old]
                if hits:
                    del items[hits[0]]
                    break
        _add_item(mem, category, note)


def _overlapping(mem: Memory, item: str) -> str | None:
    """An existing item about the same subject as ``item``, if any: a duplicate
    ("sister Priya is staying in Bangalore" vs "sister Priya lives in Bangalore")
    or a contradiction ("long answers" vs "prefers short answers"). Extraction
    only adds; changing what is known needs an explicit remember(replaces=...)."""
    subject = _content_words(item) - _FRAMING
    if not subject:
        return None
    for c in CATEGORIES[1:]:
        for x in getattr(mem, c):
            other = _content_words(x) - _FRAMING
            # Relative to the shorter one, so extra words ("long answers from now on")
            # don't hide that it's about the same thing as "prefers short answers".
            if other and len(subject & other) / min(len(subject), len(other)) >= 0.5:
                return x
    return None


def _apply_update(mem: Memory, update: MemoryUpdate, statements: list[str]) -> list[str]:
    """Add LLM-extracted facts that are new and grounded in a user statement.
    Returns what was added."""
    added, source = [], "\n".join(statements).lower()
    for key, value in update.profile.items():
        key, value = key.strip().lower(), value.strip()
        if key and value and key not in mem.profile and value.lower() in source:
            mem.profile[key] = value
            added.append(f"{key}: {value}")
    for a in update.add:
        if not _grounded(a.item, statements):
            continue
        clash = _overlapping(mem, a.item)
        if clash:
            print(f"(not saving \"{a.item}\": overlaps \"{clash}\")")
        elif _add_item(mem, a.category, a.item):
            added.append(a.item)
    return added


class TopicGroup(BaseModel):
    topic: str = Field(description="Short lowercase topic name, 1-4 words.")
    summary: str = Field(description="1-3 sentences: key facts, answers and decisions.")
    exchanges: list[int] = Field(description="Numbers of the exchanges about this topic.")


class OffloadPlan(BaseModel):
    """How to file offloaded exchanges into topics."""

    topics: list[TopicGroup]


def _match_topic(query: str, topics: dict) -> str | None:
    """Find a topic by exact name, then substring, then shared words."""
    q = query.strip().lower()
    if q in topics:
        return q
    for name in topics:
        if q and (q in name or name in q):
            return name
    words = set(q.split())
    best = max(topics, key=lambda n: len(words & set(n.split())), default=None)
    return best if best and words & set(best.split()) else None


def _render_exchange(messages: list[dict]) -> str:
    """Flatten one exchange (user turn, tool calls, replies) into readable text."""
    lines = []
    for msg in messages:
        for block in msg["content"]:
            if "text" in block and block["text"].strip():
                lines.append(f"{'User' if msg['role'] == 'user' else 'Assistant'}: {block['text'].strip()}")
            elif "toolUse" in block:
                lines.append(f"Tool call: {block['toolUse']['name']} {json.dumps(block['toolUse']['input'], ensure_ascii=False)}")
            elif "toolResult" in block:
                result = " ".join(c.get("text", "") for c in block["toolResult"].get("content", []))
                lines.append(f"Tool result: {result[:500]}{'...' if len(result) > 500 else ''}")
    return "\n".join(lines)


def _structured(model, output_model: type[BaseModel], prompt: str) -> BaseModel:
    """Ask the model for JSON matching ``output_model``.

    Calls the model's structured_output directly: for Ollama this uses
    schema-constrained decoding, which llama3.1 follows far more reliably than
    the tool-call route used by Agent(structured_output_model=...). Runs in a
    worker thread because it may be called from inside the agent's event loop.
    """

    async def run():
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        async for event in model.structured_output(output_model, messages):
            if "output" in event:
                return event["output"]
        raise ValueError("model returned no structured output")

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, run()).result()


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # atomic, so a crash never leaves a half-written file


class MemoryStore:
    EXTRACT_CHUNK = 8  # user messages per fact-extraction call at the end of a session

    def __init__(self, model, memory_dir: Path = MEMORY_DIR) -> None:
        self.model = model
        memory_dir.mkdir(exist_ok=True)
        self.session_path = memory_dir / "session.json"
        self.persistent_path = memory_dir / "persistent.json"
        # The agent may run several tool calls (e.g. two remembers) in parallel threads.
        self._lock = threading.Lock()
        self.preloaded: list[str] = []  # topics recalled into the opening message this turn

    # ---- persistent memory -------------------------------------------------

    def load_persistent(self) -> Memory:
        data = _read_json(self.persistent_path, {})
        data.pop("updated_at", None)
        return Memory(**data)

    def persistent_prompt(self) -> str:
        """Render persistent memory as a compact block for the system prompt."""
        mem = self.load_persistent()
        lines = [f"- {k}: {v}" for k, v in mem.profile.items()]
        for category in CATEGORIES[1:]:
            items = getattr(mem, category)
            if items:
                lines.append(f"{category.capitalize()}:")
                lines += [f"  - {item}" for item in items]
        if not lines:
            return "Long-term memory about the user: (empty)"
        return "Long-term memory about the user (from earlier sessions):\n" + "\n".join(lines)

    def mentions_memory(self, message: str) -> bool:
        """True if the message is about something already stored in memory."""
        items = [x for c in CATEGORIES[1:] for x in getattr(self.load_persistent(), c)]
        items += [n["note"] for n in getattr(self, "session", {}).get("notes", [])]
        return any(_mentions(message, item) for item in items)

    def screen_context(self) -> str:
        """Persistent memory plus this session's notes, for the topic screen."""
        notes = [f"  - {n['note']}" for n in getattr(self, "session", {}).get("notes", [])]
        if not notes:
            return self.persistent_prompt()
        return self.persistent_prompt() + "\nNoted this session:\n" + "\n".join(notes)

    def topic_index_prompt(self) -> str:
        """One line per offloaded topic, so the model knows what it can recall."""
        topics = getattr(self, "session", {}).get("topics", {})
        if not topics:
            return ""
        lines = [f"- {name}: {t['summary']}" for name, t in topics.items()]
        return (
            "\n\nEarlier in this session we discussed these topics (details offloaded; "
            "call the recall tool with the topic name if you need them):\n" + "\n".join(lines)
        )

    def topic_text(self, name: str, max_chars: int | None = None) -> str:
        """An offloaded topic's summary and exchanges. With ``max_chars``, keeps
        the newest exchanges that fit."""
        t = self.session.get("topics", {})[name]
        head = f"Topic: {name}\nSummary: {t['summary']}"
        exchanges = [e["text"] for e in t["exchanges"]]
        if max_chars is not None:
            kept, size = [], len(head)
            for text in reversed(exchanges):
                if kept and size + len(text) > max_chars:
                    break
                kept.insert(0, text if kept else text[:max(0, max_chars - size)])
                size += len(text)
            exchanges = kept
        return head + "\n\n" + "\n\n".join(exchanges)

    def recalled_prompt(self, names: list[str]) -> str:
        """Offloaded topics picked for the next message (see TopicScreen.topics_to_recall)."""
        topics = getattr(self, "session", {}).get("topics", {})
        blocks = [self.topic_text(n, RECALL_CHARS) for n in names if n in topics]
        if not blocks:
            return ""
        return (
            "\n\nDetails from earlier in this session that may help with my next message "
            "(already recalled, no need to call recall for these):\n\n" + "\n\n".join(blocks)
        )

    def seed_messages(self, recalled: list[str] = ()) -> list[dict]:
        """Opening exchange that hands persistent memory (and the offloaded-topic
        index, plus any topics recalled for the next message) to the model.
        llama3.1 under-weights the system prompt when tools are attached, but
        reliably uses conversation turns."""
        self.preloaded = [n for n in recalled if n in getattr(self, "session", {}).get("topics", {})]
        text = (self.persistent_prompt() + self.topic_index_prompt() + self.recalled_prompt(self.preloaded)
                + "\nUse this when answering me.")
        return [
            {"role": "user", "content": [{"text": text}]},
            {"role": "assistant", "content": [{"text": "Got it, I'll use this memory when answering you."}]},
        ]

    # ---- session memory ----------------------------------------------------

    def start_session(self) -> None:
        """Consolidate a session left over from a crash, then start a fresh one."""
        if self.session_path.exists():
            print("Found unsaved memory from a previous session, consolidating it...")
            self.end_session()
        self.session = {"started_at": datetime.now().isoformat(timespec="seconds"), "notes": [], "transcript": [], "topics": {}}
        self._save_session()

    def _save_session(self) -> None:
        with self._lock:
            _write_json(self.session_path, self.session)

    def add_turn(self, role: str, text: str) -> None:
        with self._lock:
            self.session["transcript"].append({"role": role, "text": text})
        self._save_session()

    def add_note(self, category: str, note: str, replaces: str = "") -> None:
        with self._lock:
            self.session["notes"].append({"category": category, "note": note, "replaces": replaces})
        self._save_session()

    def remember_tool(self):
        store = self

        @tool
        def remember(note: str, category: str = "facts", replaces: str = "") -> str:
            """Save something the user told you about themselves so it is remembered
            in future sessions. Call it when the user shares their name, preferences,
            people they know, routines, or other lasting facts, or corrects one.

            Args:
                note: The fact as a short standalone sentence with names (not "she").
                    For category profile use "key: value", e.g. "name: Gowtham".
                category: One of profile, preferences, people, routines, facts.
                replaces: If this corrects a fact in long-term memory, that old fact
                    copied exactly; otherwise leave empty.
            """
            if category not in CATEGORIES:
                category = "facts"
            store.add_note(category, note, replaces)
            return f"Saved to session memory ({category}): {note}" + (f" (replaces: {replaces})" if replaces else "")

        return remember

    def recall_tool(self):
        store = self

        @tool
        def recall(topic: str) -> str:
            """Fetch the full details of an earlier part of THIS session that was
            offloaded from the conversation. Only use it for a topic listed under
            "Earlier in this session we discussed these topics"; never for long-term
            memory or to save anything (use remember for that).

            Args:
                topic: The topic name from the offloaded-topics list.
            """
            topics = store.session.get("topics", {})
            name = _match_topic(topic, topics)
            if name is None:
                return f"No topic matching '{topic}'. Available topics: {', '.join(topics) or 'none'}."
            return store.topic_text(name)

        return recall

    # ---- offloading --------------------------------------------------------

    def offload(self, exchanges: list[str]) -> list[str]:
        """Group exchanges by topic into session memory. Returns the topics touched."""
        topics = self.session.setdefault("topics", {})
        try:
            plan = self._classify(exchanges, topics)
            groups = [g for g in plan.topics if g.topic.strip()]
            assigned = {n: g.topic.strip().lower() for g in groups for n in g.exchanges}
            summaries = {g.topic.strip().lower(): g.summary.strip() for g in groups}
        except Exception as exc:  # LLM failed: still offload, under a generic topic
            print(f"(topic grouping failed: {exc})")
            assigned, summaries = {}, {}

        touched = []
        now = datetime.now().isoformat(timespec="seconds")
        for i, text in enumerate(exchanges, start=1):
            name = _match_topic(assigned.get(i, "general"), topics) or assigned.get(i, "general")
            topic = topics.setdefault(name, {"summary": "", "exchanges": []})
            topic["exchanges"].append({"offloaded_at": now, "text": text})
            if name not in touched:
                touched.append(name)
        for name in touched:
            new = summaries.get(name) or next((v for k, v in summaries.items() if _match_topic(k, {name: 0})), "")
            if new:
                topics[name]["summary"] = new
            elif not topics[name]["summary"]:
                topics[name]["summary"] = topics[name]["exchanges"][0]["text"].splitlines()[0][:120]
        self._save_session()
        return touched

    def _classify(self, exchanges: list[str], topics: dict) -> "OffloadPlan":
        existing = "\n".join(f"- {n}: {t['summary']}" for n, t in topics.items()) or "(none yet)"
        # Only the conversation text: tool calls/results are noise for topic grouping.
        brief = ["\n".join(l for l in text.splitlines() if l.startswith(("User:", "Assistant:")))[:600] for text in exchanges]
        numbered = "\n\n".join(f"### Exchange {i}\n{text}" for i, text in enumerate(brief, start=1))
        prompt = f"""Group these chat exchanges by topic so they can be looked up later.

EXISTING TOPICS:
{existing}

EXCHANGES:
{numbered}

List the topics these exchanges are about. For each topic give:
- topic: short lowercase name, 1-4 words (e.g. "daily summary", "family", "workout plan").
  Reuse an existing topic name exactly when it fits; otherwise make a new one.
- summary: 1-3 sentences covering its existing summary plus the new exchanges,
  keeping key facts, answers and decisions the user gave.
- exchanges: the numbers of the exchanges that belong to it.
Every exchange number from 1 to {len(exchanges)} must appear in exactly one topic."""
        return _structured(self.model, OffloadPlan, prompt)

    # ---- consolidation -----------------------------------------------------

    def end_session(self) -> None:
        """Merge session memory into persistent memory, then clear the session."""
        session = _read_json(self.session_path, {})
        if not session.get("notes") and not session.get("transcript"):
            self.session_path.unlink(missing_ok=True)
            return

        merged = self.load_persistent()
        _apply_notes(merged, session.get("notes", []))
        # Facts the agent didn't save with remember: extracted from the user's own
        # words only (assistant replies and the Bee data they quote are not facts
        # about the user), added only if new and grounded in what the user said.
        said = _statements([t["text"] for t in session.get("transcript", []) if t["role"] == "user"])
        # In chunks: given a long chat at once, small models pick out only a fact or two.
        for start in range(0, len(said), self.EXTRACT_CHUNK):
            chunk = said[start:start + self.EXTRACT_CHUNK]
            try:
                added = _apply_update(merged, self._extract_update(merged, chunk), chunk)
                if added:
                    print("Added to long-term memory: " + "; ".join(added))
            except Exception as exc:
                print(f"(fact extraction failed: {exc}; saved notes only)")

        if self.persistent_path.exists():  # one-step undo if a merge goes wrong
            shutil.copy(self.persistent_path, self.persistent_path.with_suffix(".prev.json"))
        _write_json(self.persistent_path, {"updated_at": datetime.now().isoformat(timespec="seconds"), **merged.model_dump()})
        self.session_path.unlink(missing_ok=True)

    def _extract_update(self, current: Memory, said: list[str]) -> MemoryUpdate:
        listing = "\n".join(f"- {t}" for t in said)
        prompt = f"""You maintain long-term memory about a user.

EXISTING MEMORY (JSON):
{current.model_dump_json(indent=2)}

WHAT THE USER SAID THIS SESSION:
{listing}

List NEW lasting facts the user stated about themselves that are not already in memory:
- profile: personal details as key/value (e.g. "city": "Pune"), only for keys not in memory.
- add: other facts, each with a category (preferences, people, routines, facts).
  Write each as a standalone fact with names, not "she" or "he".
Only use facts the user stated plainly. Skip requests, small talk, and things that
only matter today. Never infer a preference from a question: "long or short answers?"
says nothing about what the user prefers.
Skip anything that changes or contradicts existing memory.
If there is nothing new, return empty lists."""
        return _structured(self.model, MemoryUpdate, prompt)


class TopicOffloadManager(ConversationManager):
    """Keeps the context window small by moving old exchanges into session memory.

    After a turn, if the conversation has more than ``max_exchanges`` exchanges,
    the oldest ones are filed by topic into session.json and dropped from the
    context, keeping the last ``keep_exchanges``. The model sees only a one-line
    summary per topic (in the pinned opening message) and can call ``recall``.
    """

    PINNED = 2  # the seed user/assistant messages carrying memory

    def __init__(self, store: MemoryStore, max_exchanges: int = 6, keep_exchanges: int = 2) -> None:
        super().__init__()
        self.store = store
        self.max_exchanges = max_exchanges
        self.keep_exchanges = keep_exchanges

    def _exchange_starts(self, messages: list) -> list[int]:
        """Indexes of user messages that start an exchange (text, not a tool result)."""
        return [
            i for i in range(self.PINNED, len(messages))
            if messages[i]["role"] == "user"
            and not any("toolResult" in b or b.get("text", "").startswith("[steering] ") for b in messages[i]["content"])
        ]

    def _offload(self, agent, keep: int) -> bool:
        starts = self._exchange_starts(agent.messages)
        if len(starts) <= keep:
            return False
        cut = starts[-keep] if keep else len(agent.messages)
        old = agent.messages[self.PINNED:cut]
        bounds = [s for s in starts if s < cut] + [cut]
        exchanges = [_render_exchange(agent.messages[a:b]) for a, b in zip(bounds, bounds[1:])]
        touched = self.store.offload(exchanges)
        agent.messages[:] = self.store.seed_messages() + agent.messages[cut:]
        self.removed_message_count += len(old)
        print(f"\n(offloaded {len(exchanges)} earlier exchanges to session memory: {', '.join(touched)})")
        return True

    def apply_management(self, agent, **kwargs) -> None:
        if len(self._exchange_starts(agent.messages)) > self.max_exchanges:
            self._offload(agent, self.keep_exchanges)

    def reduce_context(self, agent, e: Exception | None = None, **kwargs) -> None:
        # Context overflow: offload everything but the current exchange.
        if not self._offload(agent, keep=1) and e is not None:
            raise e
