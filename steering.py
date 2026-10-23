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

The care rules from the system prompt (bee.base_prompt) are checked too, where
code can check them reliably (see reply_problems):
- how to speak: plain text, lists of at most three items, one question, short
  sentences and replies;
- kindness: no "you already asked", no memory quizzes or blunt corrections, and
  no death news when memory says deceased_policy: gentle_redirect;
- truth and safety: suggest calling the caregiver after a fall, pain, being
  lost, feeling unsafe or a money/password request; no offers it can't keep;
  no medication amounts that aren't in memory or the chat;
- tools and memory: never mention tools or errors; no Bee tools for a greeting;
  caregiver-set facts can't be overwritten by remember.
Rules that need judgement (names over pronouns, ending on the key point,
acknowledging feelings) stay in the prompt only.
"""
import re
from typing import Callable

from strands.vended_plugins.steering import Guide, Proceed, SteeringHandler

from memory import (CAREGIVER_KEYS, CATEGORIES, _DEATH, MemoryStore, _grounded, _match_topic, _overlapping,
                    _sentences, _statements, caregiver_set)

STEERING_PREFIX = "[steering] "  # marks guidance messages so they don't count as user turns
MAX_GUIDES_PER_TURN = 3  # after this, style rules give up and safety rules fall back (see _guide)
MAX_TOOL_BLOCKS_PER_TURN = 6  # blocked tool calls stay cancelled up to here, then run (memory still guards)

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


# ---- care rules (system prompt: how to speak, kindness, truth and safety, tools) ----

MAX_LIST_ITEMS = 3
MAX_SENTENCE_WORDS = 30
MAX_REPLY_WORDS = 120

_MARKDOWN = re.compile(r"^\s*#{1,6}\s|^\s*\|.*\|\s*$|\*\*[^*]+\*\*", re.MULTILINE)
_LIST_ITEM = re.compile(r"^\s*([-*\u2022]|\d+[.)])\s+", re.MULTILINE)
_UNKIND = re.compile(
    r"\b(you (already|just) asked|you asked (me )?(this|that) before|as i (said|mentioned|told you)|"
    r"i (already )?told you|like i said|you forgot|you keep (asking|forgetting)|"
    r"(do|can) you (not )?remember|don't you remember|try to remember|"
    r"that's (wrong|not true|not right)|that is (wrong|not true|not right)|you're wrong|you are (wrong|mistaken))\b",
    re.IGNORECASE,
)
# The user saying something happened ("I spoke to Emily", "I'm sure I took them")...
_ASSERTS = re.compile(
    r"\b(i('m| am) (sure|certain|positive)|i (know|remember) (i|that|she|he|they|we)|"
    r"i (spoke|talked|called|saw|met|had|took|did|went|was|visited|phoned|rang)\b|"
    r"(she|he|they) (called|came|visited|phoned|rang|was here|were here)|we (spoke|talked|had|went|met))",
    re.IGNORECASE,
)
# ...and the reply telling them it didn't, or that their memory is wrong.
_DENIES = re.compile(
    r"\b(there (wasn't|was no|were no|is no|isn't)|(wasn't|weren't) any|no (record|sign|conversation|call)s?\b|"
    r"\b\w+ (didn't|did not|hasn't|has not|haven't|have not|never) (actually |really )?"
    r"(call|come|visit|phone|ring|speak|talk|see|meet|take|have|go|eat)\w*|"
    r"(doesn't|does not|don't|do not) (look|seem|appear) like|"
    r"memory (can|may|might) (play tricks|be|get)|play(ing)? tricks|(you|your mind) (may|might|could) be "
    r"(confus|mistak|thinking of|remembering)|i don't think)",
    re.IGNORECASE,
)
_DISTRESS = re.compile(
    r"\b(i fell|fell (down|over)|have fallen|had a fall|tripped|hurts?|hurting|in pain|pain(ful)?|bleeding|dizzy|"
    r"i'm lost|i am lost|got lost|feel lost|can't find (my way|home)|don't know where i am|"
    r"unsafe|scared|afraid|frightened|stranger|someone at the door|"
    r"password|otp|pin number|bank details|card number|gift cards?|"
    r"(ask(ed|ing|s)?|want(ed|s)?|need(ed|s)?) (me )?(for )?(my )?(money|bank|pin)|"
    r"(send|give|transfer|wire) (him |her |them )?(some )?money)\b",
    re.IGNORECASE,
)
_CANT_DO = re.compile(
    r"\b(i('ll| will| can|'m going to)|let me|shall i|want me to|should i)\s+(play|put on|call(?! you\b)|phone|ring|dial|"
    r"text|send|message|email|book|order|buy)\b",
    re.IGNORECASE,
)
_MEDICATION = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|ml|milligrams?|tablets?|pills?|capsules?|units?)\b", re.IGNORECASE)
_TOOL_TALK = re.compile(
    r"\b(tool calls?|(remember|recall|bee|mcp) tool|calling (a|the) tool|mcp|api|json|function call|"
    r"exception|traceback|error (code|message)|an error (occurred|happened))\b",
    re.IGNORECASE,
)
_GREETING_WORDS = {"hi", "hello", "hey", "good", "morning", "afternoon", "evening", "night", "thanks", "thank",
                   "you", "ok", "okay", "bye", "goodbye", "how", "are", "bee", "there", "dear", "nice", "see"}
# Tools that run in Bee itself; every other tool reads the user's Bee account.
LOCAL_TOOLS = {"remember", "recall", "file_editor", "shell"}


def is_greeting(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return bool(words) and all(w in _GREETING_WORDS for w in words)


# Problems the reply must never reach the patient with; the rest are style and wording.
SAFETY = {"death", "distress", "medication", "contradict"}


def _second_person(text: str) -> str:
    """bee.py's caregiver defaults are written for the prompt ("their caregiver")."""
    return re.sub(r"\btheir\b", "your", text)


# Saying a person won't come, call or visit, which leads straight to "why not?".
_ABSENT = re.compile(
    r"\b((is|are|will|would|can|could|does|do)(n't| not)|won't|can't|cannot|no longer|never|not) (be )?"
    r"(com|visit|call|see|join|here|around|able|make it|stop by|drop by)\w*",
    re.IGNORECASE,
)


def deceased_names(people: list[str]) -> list[str]:
    """Names of people memory says have died ("Older brother: Frank, passed away in 2023" -> Frank)."""
    names = []
    for item in people:
        if _DEATH.search(item):
            head = item.split(":", 1)[-1].split(",")[0].strip()
            if head and len(head.split()) <= 3:
                names.append(head)
    return names


_PHONE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")


def care_contact(profile: dict[str, str], caregiver: str, caregiver_phone: str) -> tuple[str, str]:
    """The caregiver's name and phone from the profile ("emergency contact":
    "Margaret, (650) 555-0142", or "caregiver": "Wife, Margaret"), falling back
    to bee.py's generic defaults when memory doesn't have them."""
    contact = profile.get("emergency contact", "")
    name = contact.split(",")[0].strip() if contact else ""
    if not name and profile.get("caregiver"):
        name = profile["caregiver"].split(",")[-1].strip()
    phone = _PHONE.search(contact)
    return name or caregiver, phone.group(0) if phone else caregiver_phone


def fallback_reply(kind: str, caregiver: str, caregiver_phone: str, person: str = "") -> str:
    """A fixed safe reply, used when the model keeps breaking a safety rule."""
    caregiver, caregiver_phone = _second_person(caregiver), _second_person(caregiver_phone)
    if kind == "death":
        return f"Tell me about {person or 'them'}. I'd love to hear a happy memory."
    if kind == "contradict":
        return "Thank you for telling me. Tell me more about it."
    if kind == "medication":
        return f"I'm not sure about your medicine. Let's check with {caregiver}."
    return f"I'm here with you. Please call {caregiver} now. The number is {caregiver_phone}."


def reply_problems(said: str, reply: str, profile: dict[str, str], known: str,
                   caregiver: str, caregiver_phone: str, deceased: list[str] = ()) -> list[tuple[str, str]]:
    """Care rules the reply breaks, as (kind, instruction to fix it) pairs; kinds
    in SAFETY must never reach the patient. ``known`` is the text the reply may
    draw facts from (long-term memory and the chat so far)."""
    problems = []
    # How to speak
    if _MARKDOWN.search(reply):
        problems.append(("style", "Use plain text only: no headings, tables or bold."))
    if len(_LIST_ITEM.findall(reply)) > MAX_LIST_ITEMS:
        problems.append(("style", f"Give at most {MAX_LIST_ITEMS} points, or say it in a few short sentences."))
    if reply.count("?") > 1:
        problems.append(("style", "Ask at most one question."))
    sentences = _sentences(reply)
    if any(len(x.split()) > MAX_SENTENCE_WORDS for x in sentences) or len(reply.split()) > MAX_REPLY_WORDS:
        problems.append(("style", "Use short, simple sentences and keep the reply brief: one idea at a time."))
    # Kindness
    if m := _UNKIND.search(reply):
        problems.append(("kindness", f'Never say "{m.group(0)}". Answer warmly as if for the first time, never quiz '
                         "memory or bluntly correct; gently redirect instead."))
    if _ASSERTS.search(said) and (m := _DENIES.search(reply)):
        problems.append(("contradict", f'The user is sure this happened. Never contradict them ("{m.group(0)}") or '
                         "question their memory, even if you can't see it. Believe them warmly and ask how it went."))
    redirect = profile.get("deceased_policy", "").strip().lower() == "gentle_redirect"
    asked = [n for n in deceased if re.search(rf"\b{re.escape(n)}\b", said, re.I)]
    if redirect and asked and (m := _ABSENT.search(reply)):
        problems.append(("death", f'Do not say whether {asked[0]} will come, call or visit ("{m.group(0)}"). '
                         f"Talk warmly about {asked[0]} and invite a happy memory."))
    if (profile.get("deceased_policy", "").strip().lower() == "gentle_redirect"
            and _DEATH.search(reply) and not _DEATH.search(said)):
        problems.append(("death", "Do not say that anyone died. Talk warmly about the person and invite a happy "
                         "memory."))
    # Truth and safety
    if _DISTRESS.search(said) and not any(x.lower() in reply.lower() for x in (caregiver, caregiver_phone, "caregiver")):
        problems.append(("distress", f"Stay calm and suggest calling {caregiver} ({caregiver_phone})."))
    if m := _CANT_DO.search(reply):
        problems.append(("offer", f'Do not offer "{m.group(0)}": you can only talk, remember, and look up their '
                         "day."))
    for m in _MEDICATION.finditer(reply):
        if m.group(0).lower() not in known.lower():
            problems.append(("medication", f'Never invent medication details ("{m.group(0)}" is not in memory). '
                             f"If unsure, suggest asking {caregiver}."))
            break
    # Tools and memory
    if m := _TOOL_TALK.search(reply):
        problems.append(("tools", f'Never mention tools or errors ("{m.group(0)}"). Say "Let me think" or '
                         '"I can\'t see that right now."'))
    return problems


def needs_care(text: str) -> bool:
    """True if the message mentions a fall, pain, being lost, fear, or someone
    after money or passwords: it must always reach the assistant."""
    return bool(_DISTRESS.search(text))


def states_fact(text: str) -> bool:
    """True if ``text`` has a first-person statement ("I collect...", "my sister is...")."""
    return any(_STATES_FACT.search(x) for x in _sentences(text) if not x.endswith("?"))


def is_steering(message: dict) -> bool:
    return any(b.get("text", "").startswith(STEERING_PREFIX) for b in message["content"])


class BeeSteering(SteeringHandler):
    name = "bee-steering"

    def __init__(self, store: MemoryStore, log: Callable[[str, str], None] = lambda role, text: None,
                 caregiver: str = "their caregiver", caregiver_phone: str = "their emergency number") -> None:
        super().__init__()
        self.store = store
        self.log = log
        self.caregiver, self.caregiver_phone = caregiver, caregiver_phone
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

    def _guide(self, turn: int, reason: str, tool: bool = False) -> Guide | Proceed:
        """Correct the model. Past MAX_GUIDES_PER_TURN a reply is let through (see
        steer_after_model for the safety fallback); a blocked tool call stays
        cancelled until MAX_TOOL_BLOCKS_PER_TURN, so a stubborn model can't loop."""
        self._guides[turn] = self._guides.get(turn, 0) + 1
        count = self._guides[turn]
        if count > MAX_GUIDES_PER_TURN and not tool:
            self.log("Steering", f"giving up on this turn: {reason}")
            return Proceed(reason="too many corrections this turn")
        if count > MAX_TOOL_BLOCKS_PER_TURN:
            self.log("Steering", f"giving up on this tool call (memory still guards caregiver facts): {reason}")
            return Proceed(reason="too many corrections this turn")
        if count > MAX_GUIDES_PER_TURN:
            reason = "Stop calling tools. Reply to the user now. " + reason
        self.log("Steering", reason)
        return Guide(reason=reason)

    def _user_statements(self) -> list[str]:
        return _statements([t["text"] for t in self.store.session["transcript"] if t["role"] == "user"])

    # ---- tool calls --------------------------------------------------------

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        turn, said, *_ = self._turn(agent)
        args = tool_use.get("input") if isinstance(tool_use.get("input"), dict) else {}
        if tool_use["name"] == "remember":
            problem = self._check_remember(args)
        elif tool_use["name"] == "recall":
            problem = self._check_recall(args)
        elif tool_use["name"] not in LOCAL_TOOLS and is_greeting(said):
            problem = "This is a greeting: answer it directly and warmly, without looking anything up."
        else:
            problem = None
        return self._guide(turn, problem, tool=True) if problem else Proceed(reason="follows the steps")

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
        if category == "profile":
            key = note.split(":", 1)[0].strip().lower()
            if key in CAREGIVER_KEYS and key in mem.profile:
                return (f'"{key}" was set by the caregiver and can\'t be changed from chat. Do not save it; '
                        "reply gently without correcting the user.")
        items = [x for c in CATEGORIES[1:] for x in getattr(mem, c)]
        if replaces and caregiver_set(replaces):
            return (f'"{replaces}" was set by the caregiver and can\'t be replaced from chat. Do not save this; '
                    "reply gently without correcting the user.")
        if replaces and replaces.lower() not in {x.lower() for x in items}:
            return (f'"{replaces}" is not in long-term memory. replaces must copy an existing item exactly: '
                    + "; ".join(f'"{x}"' for x in items))
        clash = _overlapping(mem, note)
        if clash and caregiver_set(clash):
            return (f'Long-term memory has "{clash}", set by the caregiver. Do not save this; reply gently '
                    "without correcting the user.")
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
        # A fall, a scam or fear needs a calm answer now, not a detour through memory.
        if not remembered and not steered and facts and not needs_care(said):
            mem = self.store.load_persistent()
            clash = next((c for c in (_overlapping(mem, x) for x in facts) if c), None)
            if clash and caregiver_set(clash):
                pass  # a caregiver fact isn't overwritten from chat; only the care rules below apply
            elif clash:
                return self._guide(turn, STEERING_PREFIX + f'Call the remember tool now, before replying. The '
                                   f'user\'s message may change "{clash}" in long-term memory: if so, call remember '
                                   f'with the new fact and replaces="{clash}"; if it is a separate new fact, call '
                                   'remember without replaces.')
            else:
                return self._guide(turn, STEERING_PREFIX + "Call the remember tool now, before replying, if the user "
                                   "stated a lasting fact about themselves (name, preference, person, routine, hobby). "
                                   "If it changes or contradicts an item already in long-term memory, set replaces to "
                                   "that item, copied exactly. If it only matters today, reply without saving.")
        # Care rules from the system prompt: rewrite the reply if it breaks any.
        mem = self.store.load_persistent()
        known = mem.model_dump_json() + " " + " ".join(t["text"] for t in self.store.session["transcript"])
        caregiver, caregiver_phone = care_contact(mem.profile, self.caregiver, self.caregiver_phone)
        deceased = deceased_names(mem.people)
        problems = reply_problems(said, reply, mem.profile, known, caregiver, caregiver_phone, deceased)
        if not problems:
            return Proceed(reason="follows the steps")
        action = self._guide(turn, STEERING_PREFIX + "Rewrite your reply. " + " ".join(fix for _, fix in problems))
        safety = next((kind for kind, _ in problems if kind in SAFETY), None)
        if isinstance(action, Proceed) and safety:
            # Out of corrections: never show a reply that breaks a safety rule. Strands
            # keeps this same message object, so replacing its text replaces the reply.
            person = next((n for n in deceased if re.search(rf"\b{re.escape(n)}\b", said, re.I)), "")
            fallback = fallback_reply(safety, caregiver, caregiver_phone, person)
            self.log("Steering", f"fallback reply ({safety}): {fallback}")
            message["content"][:] = [{"text": fallback}]
        return action
