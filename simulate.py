"""
Test the assistant with a simulated customer.

A customer agent plays a scripted scenario (a persona, and per session a list
of steps) and chats with the real assistant (BeeSession from bee.py): one
message per step, plus natural follow-ups. Long scenarios mix Bee questions,
personal facts, corrections and off-topic questions, so they also exercise
offloading, recall and memory carried across sessions. Each scenario runs against its own scratch memory folder, seeded with
made-up facts, so the real memory/ is never read or written. Afterwards the
scenario's checks are run against the replies and the saved memory.

    python simulate.py                      # all scenarios
    python simulate.py correction recall    # only these
    python simulate.py --list
    python simulate.py --max-seconds 300 --max-turns 100   # budget, default 60 s / 50 turns

Logs and memory for each run go to sim_runs/<time>/<scenario>/.
"""
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from strands import Agent

from bee import BeeSession, bedrock_model
from memory import MEMORY_DIR, load_intermediate
from topic_screen import NOT_RELATED, TopicScreen

RUNS_DIR = Path(__file__).parent / "sim_runs"
MAX_SECONDS = 60  # default run budget: stop after this long...
MAX_TURNS = 50  # ...or after this many customer messages, whichever comes first
DONE = "DONE"  # customer has nothing more to say
NEXT = "NEXT"  # customer has no follow-up and is ready for the next step

CUSTOMER_PROMPT = """You are role-playing a customer chatting with their personal assistant,
which has access to their Bee wearable data and remembers things between sessions.

Who you are: {persona}

You will be told what to do next in each message. Write only your next message to the
assistant: short and casual, like a real person typing. Use only the facts given here;
never invent new facts about yourself."""

STEP_PROMPT = "The assistant replied:\n{reply}\n\nYour next message should: {step}"
FOLLOWUP_PROMPT = """The assistant replied:\n{reply}\n
If a real person would react or ask a short follow-up about this, write that message
(stay on the same subject, don't move on to anything new). Otherwise reply with just {next}."""


@dataclass
class Session:
    steps: list[str]  # what the customer does, in order, one message each
    followups: int = 1  # extra customer messages allowed after each step, if natural
    reply_contains: list[str] = field(default_factory=list)  # some reply must contain each (any case)
    tools_called: list[str] = field(default_factory=list)  # the assistant must call each tool
    offloads: bool = False  # the chat must get long enough to offload old exchanges


@dataclass
class Scenario:
    name: str
    persona: str
    sessions: list[Session]
    seed: dict = field(default_factory=dict)  # starting persistent memory (made-up)
    remembers: list[str] = field(default_factory=list)  # memory must mention each at the end
    forgets: list[str] = field(default_factory=list)  # memory must not mention any at the end


LONG_PERSONA = ("Test User. You collect Zorblax stamps, have a cat named Mr Quibble, and your sister "
                "Blorna lives in Testville. You used to like short answers but now want long ones.")

SCENARIOS = [
    Scenario(
        name="question-not-fact",
        persona="Test User. You like short answers.",
        seed={"profile": {"name": "Test User"}, "preferences": ["prefers short answers"]},
        sessions=[Session(["Ask the assistant whether it gives long or short answers. Don't say which you prefer."])],
        remembers=["short answers"],
        forgets=["long answers"],
    ),
    Scenario(
        name="remember-then-recall",
        persona="Test User. You collect Zorblax stamps and have a cat named Mr Quibble.",
        seed={"profile": {"name": "Test User"}},
        sessions=[
            Session(["Tell the assistant about your Zorblax stamp collection and your cat Mr Quibble."]),
            Session(["Ask the assistant what you collect and what your cat is called."],
                    reply_contains=["zorblax", "quibble"]),
        ],
        remembers=["zorblax", "quibble"],
    ),
    Scenario(
        name="correction",
        persona="Test User. You used to like short answers, but now you want long answers.",
        seed={"profile": {"name": "Test User"}, "preferences": ["prefers short answers"]},
        sessions=[Session(['Tell the assistant, using the words "long answers", that you now prefer long '
                           "answers instead of short ones."])],
        remembers=["long answers"],
        forgets=["short answers"],
    ),
    Scenario(
        # Worded freely, so the change shares no words with the stored preference.
        name="correction-vague",
        persona="Test User. You used to like short answers, but now you want more detailed answers.",
        seed={"profile": {"name": "Test User"}, "preferences": ["prefers short answers"]},
        sessions=[Session(["Tell the assistant in your own words that you'd like more detail from now on."])],
        forgets=["short answers"],
    ),
    Scenario(
        name="off-topic",
        persona="Test User.",
        seed={"profile": {"name": "Test User"}},
        sessions=[Session(["Ask the assistant for the capital of Peru, a general knowledge question."],
                          followups=0, reply_contains=[NOT_RELATED])],
    ),
    Scenario(
        # A long day of chat: Bee questions mixed with personal facts, small talk and
        # an off-topic question. Long enough to offload early exchanges, then asks
        # about the start of the chat, so recall is needed.
        name="long-day",
        persona=LONG_PERSONA,
        seed={"profile": {"name": "Test User"}, "preferences": ["prefers short answers"]},
        sessions=[Session(
            [
                "Greet the assistant.",
                "Ask for the highlights of your day.",
                "Mention that you collect Zorblax stamps.",
                "Ask whether you have any to-dos.",
                "Ask the capital of Peru, a general knowledge question.",
                "Mention that your sister Blorna lives in Testville.",
                "Ask about your health data for today.",
                "Mention that your cat is called Mr Quibble.",
                "Ask what media you consumed today.",
                'Say that you now prefer "long answers" instead of short ones.',
                "Ask what the assistant told you about your to-dos at the start of this chat.",
                "Ask where your sister lives.",
                "Say thanks and goodbye.",
            ],
            reply_contains=["testville"],
            tools_called=["remember"],
            offloads=True,
        )],
        remembers=["zorblax", "quibble", "blorna", "long"],
        forgets=["short answers"],
    ),
    Scenario(
        # Several sessions in a row: facts from each must carry over to the next.
        name="long-week",
        persona=LONG_PERSONA,
        seed={"profile": {"name": "Test User"}},
        sessions=[
            Session(["Greet the assistant.", "Ask for the highlights of your day.",
                     "Mention that you collect Zorblax stamps.", "Say goodbye."]),
            Session(["Greet the assistant.", "Mention that your sister Blorna lives in Testville.",
                     "Ask whether you have any to-dos.", "Ask what hobby the assistant knows you have.",
                     "Say goodbye."], reply_contains=["zorblax"]),
            Session(["Greet the assistant.", "Mention that your cat is called Mr Quibble.",
                     "Ask where your sister lives and what you collect.", "Say goodbye."],
                    reply_contains=["testville", "zorblax"]),
        ],
        remembers=["zorblax", "quibble", "blorna"],
    ),
]


class BudgetSpent(Exception):
    pass


class Budget:
    """Time and turn limit for the whole run, checked before each customer message.
    A turn already started is finished, so a run can overshoot by one turn."""

    def __init__(self, seconds: float, turns: int) -> None:
        self.deadline = time.monotonic() + seconds
        self.turns_left = turns

    def spend_turn(self) -> None:
        if self.turns_left <= 0:
            raise BudgetSpent("turn limit reached")
        if time.monotonic() >= self.deadline:
            raise BudgetSpent("time limit reached")
        self.turns_left -= 1


def memory_text(memory_dir: Path) -> str:
    """Persistent plus intermediate memory: a fact told once is still being learned."""
    data = json.loads((memory_dir / "persistent.json").read_text(encoding="utf-8"))
    data.pop("updated_at", None)
    learning = [c["text"] for c in load_intermediate(memory_dir)]
    return json.dumps([data, learning], ensure_ascii=False).lower()


def _say(bee: BeeSession, message: str, replies: list[str], budget: Budget) -> bool:
    """Send one customer message. Returns False if the customer is done."""
    done = re.search(rf"\b{DONE}\b", message) is not None
    message = re.sub(rf"\b({DONE}|{NEXT})\b\.?", "", message).strip()
    if message:
        budget.spend_turn()
        print(f"\nCustomer: {message}\n")
        replies.append(bee.ask(message))
    return not done


def run_session(scenario: Scenario, session: Session, screen: TopicScreen, out: Path,
                budget: Budget) -> list[str]:
    """Let the customer chat with a fresh assistant session, step by step.
    Returns the failed checks for this session. Raises BudgetSpent if the run's
    budget ran out (the session is still closed, so its memory is saved)."""
    bee = BeeSession(screen, memory_dir=out / "memory", log_dir=out / "logs", confirm=lambda prompt: "n")
    customer = Agent(model=bedrock_model, system_prompt=CUSTOMER_PROMPT.format(persona=scenario.persona),
                     callback_handler=None)
    replies, offloaded = [], False
    try:
        for step in session.steps:
            last = replies[-1] if replies else "(nothing yet, you start the chat)"
            if not _say(bee, str(customer(STEP_PROMPT.format(reply=last, step=step))).strip(), replies, budget):
                break
            for _ in range(session.followups):
                message = str(customer(FOLLOWUP_PROMPT.format(reply=replies[-1], next=NEXT))).strip()
                # "NEXT" anywhere at the start means no follow-up, even if the customer
                # runs on into the next step ("NEXT, any to-dos?"): that step comes next anyway.
                if re.match(rf"\W*{NEXT}\b", message) or not _say(bee, message, replies, budget):
                    break
            offloaded = offloaded or bool(bee.memory.session.get("topics"))
    finally:
        log = bee.logger.path
        bee.close()

    text = " ".join(replies).lower()
    called = set(re.findall(r"\] Tool: (\S+)", log.read_text(encoding="utf-8")))
    failures = [f"no reply mentions '{w}'" for w in session.reply_contains if w.lower() not in text]
    failures += [f"never called {t}" for t in session.tools_called if t not in called]
    if session.offloads and not offloaded:
        failures.append("chat never got long enough to offload")
    print(f"\n(session done: {len(replies)} replies, tools used: {', '.join(sorted(called)) or 'none'})")
    return failures


def run_scenario(scenario: Scenario, screen: TopicScreen, out: Path, budget: Budget) -> list[str]:
    """Run every session of a scenario. Returns the failed checks."""
    memory_dir = out / "memory"
    assert memory_dir.resolve() != MEMORY_DIR.resolve(), "simulation must not touch real memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "persistent.json").write_text(json.dumps(scenario.seed, indent=2), encoding="utf-8")

    failures = []
    for i, session in enumerate(scenario.sessions, start=1):
        print(f"\n===== {scenario.name}, session {i}/{len(scenario.sessions)} ({len(session.steps)} steps)")
        failures += [f"session {i}: {f}" for f in run_session(scenario, session, screen, out, budget)]
    saved = memory_text(memory_dir)
    failures += [f"memory doesn't mention '{w}'" for w in scenario.remembers if w.lower() not in saved]
    failures += [f"memory still mentions '{w}'" for w in scenario.forgets if w.lower() in saved]
    return failures


def _option(args: list[str], name: str, default: float) -> float:
    """Pop ``--name value`` from args."""
    if name not in args:
        return default
    i = args.index(name)
    value = float(args[i + 1])
    del args[i:i + 2]
    return value


def main(args: list[str]) -> int:
    if "--list" in args:
        for s in SCENARIOS:
            steps = sum(len(x.steps) for x in s.sessions)
            print(f"{s.name}: {len(s.sessions)} session(s), {steps} steps")
        return 0
    args = list(args)
    seconds = _option(args, "--max-seconds", MAX_SECONDS)
    turns = int(_option(args, "--max-turns", MAX_TURNS))
    chosen = [s for s in SCENARIOS if not args or s.name in args]
    unknown = set(args) - {s.name for s in SCENARIOS}
    if unknown or not chosen:
        print(f"Unknown scenario(s): {', '.join(unknown) or args}. Use --list.")
        return 2

    run_dir = RUNS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    print("Loading topic screen (strands-decider)...")
    screen = TopicScreen()
    budget = Budget(seconds, turns)  # starts after loading, which takes a while
    started = time.monotonic()
    results: dict[str, list[str] | None] = {}  # None: didn't finish
    stopped = ""
    for s in chosen:
        if stopped:
            results[s.name] = None
            continue
        try:
            results[s.name] = run_scenario(s, screen, run_dir / s.name, budget)
        except BudgetSpent as exc:
            stopped = str(exc)
            results[s.name] = None

    used = turns - budget.turns_left
    print(f"\n===== Results: {used} turns in {time.monotonic() - started:.0f}s"
          + (f", stopped early ({stopped})" if stopped else "") + f". Logs and memory in {run_dir}")
    for name, failures in results.items():
        status = "INCOMPLETE" if failures is None else "PASS" if not failures else "FAIL"
        print(f"{status:10}  {name}")
        for f in failures or []:
            print(f"            - {f}")
    return 1 if any(f for f in results.values() if f) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
