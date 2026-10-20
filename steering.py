"""
Steering for the Bee agent: checks that the agent follows the memory steps in
BASE_PROMPT before a tool runs and before a reply is accepted.

Rules are checked in code, not by another LLM, because the small models used
here are the ones that skip the steps in the first place. A broken rule cancels
the tool call (or discards the reply) and tells the model what to do instead.

- remember: the note must come from something the user stated (not a question),
  profile notes must be "key: value", ``replaces`` must name an existing item,
  and a note that overlaps existing memory must say what it replaces.
- recall: only for a topic offloaded this session, and not one already preloaded
  into the opening message by the topic screen.
- reply: if the user stated a lasting fact, remember must have been called; a
  reply must not claim it saved something it didn't.
"""
import re
from typing import Callable

from strands.vended_plugins.steering import Guide, Proceed, SteeringHandler

from memory import CATEGORIES, MemoryStore, _grounded, _match_topic, _overlapping, _sentences, _statements

STEERING_PREFIX = "[steering] "  # marks guidance messages so they don't count as user turns
MAX_GUIDES_PER_TURN = 3  # stop steering a turn after this, so a stubborn model can't loop

# First-person statements anywhere in a sentence, which may carry a lasting fact.
# Requests ("I want...", "I need...") and pleasantries ("I appreciate it") are
# left out: they are about this turn, not the user.
_STATES_FACT = re.compile(
    r"(^|[\s,;:(])(remember\b|call me\b|my (\w+ ){1,3}(is|are|was|lives|works|has|loves|likes)\b|"
    r"my (name|sister|brother|mom|mother|dad|father|wife|husband|partner|friend|son|daughter|kid|cat|dog|"
    r"pet|job|boss|hobby)\b|"
    r"i(\s+|'m\b|'ve\b|'d\b)\s*(?!want|need|wonder|think|would|could|can|was wondering|just|appreciate|"
    r"hope|guess|see|mean|know|understand|agree|thank)\w+)",
    re.IGNORECASE,
)
_CLAIMS_SAVED = re.compile(
    r"\b(i'll remember|i will remember|i've (saved|noted)|i have (saved|noted)|saved (that|this|it))\b",
    re.IGNORECASE,
)


def states_fact(text: str) -> bool:
    """True if ``text`` has a first-person statement ("I collect...", "my sister is...")."""
    return any(_STATES_FACT.search(x) for x in _sentences(text) if not x.endswith("?"))


def is_steering(message: dict) -> bool:
    return any(b.get("text", "").startswith(STEERING_PREFIX) for b in message["content"])


class BeeSteering(SteeringHandler):
    name = "bee-steering"

    def __init__(self, store: MemoryStore, log: Callable[[str, str], None] = lambda role, text: None) -> None:
        super().__init__()
        self.store = store
        self.log = log
        self._guides: dict[int, int] = {}  # turn number -> guides given in that turn

    # ---- current turn ------------------------------------------------------

    def _turn(self, agent) -> tuple[int, str, list[dict], bool]:
        """This turn's number (bee.py logs each user turn to the transcript first),
        the user message that started it, the tool calls since, and whether a
        reply was already steered this turn."""
        turn = len(self.store.session["transcript"])
        for i in range(len(agent.messages) - 1, -1, -1):
            msg = agent.messages[i]
            if msg["role"] == "user" and not is_steering(msg) and not any("toolResult" in b for b in msg["content"]):
                text = " ".join(b["text"] for b in msg["content"] if "text" in b)
                since = agent.messages[i + 1:]
                calls = [b["toolUse"] for m in since for b in m["content"] if "toolUse" in b]
                return turn, text, calls, any(is_steering(m) for m in since)
        return turn, "", [], False

    def _guide(self, turn: int, reason: str) -> Guide | Proceed:
        self._guides[turn] = self._guides.get(turn, 0) + 1
        if self._guides[turn] > MAX_GUIDES_PER_TURN:
            self.log("Steering", f"giving up on this turn: {reason}")
            return Proceed(reason="too many corrections this turn")
        self.log("Steering", reason)
        return Guide(reason=reason)

    def _user_statements(self) -> list[str]:
        return _statements([t["text"] for t in self.store.session["transcript"] if t["role"] == "user"])

    # ---- tool calls --------------------------------------------------------

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        turn, *_ = self._turn(agent)
        args = tool_use.get("input") if isinstance(tool_use.get("input"), dict) else {}
        if tool_use["name"] == "remember":
            problem = self._check_remember(args)
        elif tool_use["name"] == "recall":
            problem = self._check_recall(args)
        else:
            problem = None
        return self._guide(turn, problem) if problem else Proceed(reason="follows the steps")

    def _check_remember(self, args: dict) -> str | None:
        note, category = str(args.get("note", "")).strip(), str(args.get("category", "facts"))
        replaces = str(args.get("replaces", "") or "").strip()
        if not note:
            return "remember needs a note."
        if not _grounded(note, self._user_statements()):
            return (f'The user never stated "{note}". Only save facts the user said about themselves; '
                    "a question is not a fact. Answer the user without saving.")
        if category == "profile" and ":" not in note:
            return 'Profile notes must be "key: value", e.g. "name: Gowtham". Call remember again.'
        mem = self.store.load_persistent()
        items = [x for c in CATEGORIES[1:] for x in getattr(mem, c)]
        if replaces and replaces.lower() not in {x.lower() for x in items}:
            return (f'"{replaces}" is not in long-term memory. replaces must copy an existing item exactly: '
                    + "; ".join(f'"{x}"' for x in items))
        clash = _overlapping(mem, note)
        if clash and not replaces and category != "profile":
            return (f'Long-term memory already has "{clash}". If the user is correcting it, call remember '
                    f'again with replaces="{clash}". If it is the same fact, do not save it.')
        return None

    def _check_recall(self, args: dict) -> str | None:
        topics = self.store.session.get("topics", {})
        name = _match_topic(str(args.get("topic", "")), topics)
        if name is None:
            listed = ", ".join(topics) or "none"
            return (f"recall is only for topics offloaded this session ({listed}). "
                    "Answer from long-term memory or the conversation instead.")
        if name in self.store.preloaded:
            return (f'The details of "{name}" are already in the opening message, under "Details from '
                    'earlier in this session". Answer from them without calling recall.')
        return None

    # ---- replies -----------------------------------------------------------

    async def steer_after_model(self, *, agent, message, stop_reason, **kwargs):
        if stop_reason != "end_turn":
            return Proceed(reason="not a final reply")
        turn, said, calls, steered = self._turn(agent)
        remembered = any(c["name"] == "remember" for c in calls)
        reply = " ".join(b.get("text", "") for b in message["content"])
        if not remembered and _CLAIMS_SAVED.search(reply):
            return self._guide(turn, STEERING_PREFIX + "You said you saved or will remember something, but you "
                               "did not call remember. Call remember now, or don't claim it was saved.")
        # Nudge once: after that the model may decide the fact only matters today.
        facts = [x for x in _sentences(said) if not x.endswith("?") and _STATES_FACT.search(x)]
        if not remembered and not steered and facts:
            mem = self.store.load_persistent()
            clash = next((c for c in (_overlapping(mem, x) for x in facts) if c), None)
            if clash:
                return self._guide(turn, STEERING_PREFIX + f'Call the remember tool now, before replying. The '
                                   f'user\'s message may change "{clash}" in long-term memory: if so, call remember '
                                   f'with the new fact and replaces="{clash}"; if it is a separate new fact, call '
                                   'remember without replaces.')
            return self._guide(turn, STEERING_PREFIX + "Call the remember tool now, before replying, if the user "
                               "stated a lasting fact about themselves (name, preference, person, routine, hobby). "
                               "If it changes or contradicts an item already in long-term memory, set replaces to "
                               "that item, copied exactly. If it only matters today, reply without saving.")
        return Proceed(reason="follows the steps")
