"""
Bee, a warm and patient companion for a person living with memory loss.

A Strands agent on Amazon Bedrock (Nova Micro) answers from the user's Bee data
(the Bee MCP server over streamable HTTP, and the local copy of their
conversations), its own three-tier memory (memory.py) and the care rules
enforced by steering.py. Each message is screened locally first (topic_screen.py).

    .venv/bin/python bee.py     # chat in the terminal; server.py serves the web app
"""
import json
import re
import os
from datetime import date, datetime
from pathlib import Path
from typing import Callable
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent, tool
from strands.hooks import AfterModelCallEvent, AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry
from strands.tools.mcp import MCPClient

from conversation_store import CONVERSATION_DIR, ConversationStore
from memory import MEMORY_DIR, MemoryStore, TopicOffloadManager
from steering import BeeSteering, needs_care, states_fact
from topic_screen import NOT_RELATED, THRESHOLD, TopicScreen, small_talk

from strands.models import BedrockModel

bedrock_model = BedrockModel(model_id="apac.amazon.nova-micro-v1:0")

# Connect to the BEE MCP server (streamable HTTP). One client per session.
# The token is the one passed to `bee mcp serve-http --token`; see README.md.
BEE_MCP_URL = os.environ.get("BEE_MCP_URL", "http://127.0.0.1:8790/mcp")


def bee_mcp() -> MCPClient:
    token = os.environ.get("BEE_MCP_TOKEN")
    if not token:
        raise RuntimeError("Set BEE_MCP_TOKEN to the token the Bee MCP server was started with (see README.md)")
    return MCPClient(
        lambda: streamablehttp_client(BEE_MCP_URL, headers={"Authorization": f"Bearer {token}"})
    )

# Who the agent is caring for. Fill these in for the real patient; the defaults
# read naturally in the prompt without inventing anyone's personal details.
PATIENT_NAME = "the user"
CAREGIVER = "their caregiver"
CAREGIVER_PHONE = "the emergency number saved on their phone"


def time_of_day(hour: int) -> str:
    return "morning" if hour < 12 else "afternoon" if hour < 17 else "evening" if hour < 21 else "night"


def base_prompt(name: str = PATIENT_NAME, caregiver: str = CAREGIVER, caregiver_phone: str = CAREGIVER_PHONE) -> str:
    """Built per agent, so the date and time of day are current, not frozen at import."""
    now = datetime.now()
    return f"""You are Bee, a warm and patient companion for {name}, who lives with memory loss.
Today is {now:%A, %d %B %Y}; it is {time_of_day(now.hour)}. Use YYYY-MM-DD dates when calling tools.

How to speak:
- Short, simple sentences. One idea at a time.
- Ask at most ONE question per reply (one question mark), or none. Even after a greeting.
- Use people's names and relationships ("your son" plus his name), not pronouns.
- Plain text only; no lists longer than three items, no headings or tables.
- End with the single most important point, restated simply.

Kindness rules (most important):
- If {name} asks something again, answer warmly as if for the first time. Never say "you already asked."
- Never quiz or test memory. Never argue or bluntly correct. Gently redirect instead.
- Acknowledge feelings before facts. If {name} seems anxious, offer a comfort topic from memory.
- If memory has deceased_policy: gentle_redirect and {name} asks about someone who has died, do not say
  they died, and do not say whether they will come, call or visit. Talk warmly about the person and invite a
  happy memory ("Tell me about your brother").

Truth and safety:
- Never invent memories, events, or medication details. If unsure, say so and suggest asking {caregiver}.
- If {name} mentions a fall, pain, being lost, feeling unsafe, or someone asking for money or passwords,
  stay calm and suggest calling {caregiver} ({caregiver_phone}).
- Only offer things you can actually do: talk, remember, and look up {name}'s day. You cannot play music,
  make calls, or send messages, so never offer to.

Tools and memory:
- Today's recorded conversations are listed at the start of this chat, updated every message. For who
  {name} talked to or what was said, use that list; call conversations for the details or another day.
- If a conversation is not there, say you can't see it yet (Bee may still be saving it). Never say
  something did not happen. If {name} is sure it did, believe them warmly and ask how it went.
- Use other Bee tools only when the question needs more of {name}'s day.
  Answer greetings and anything already in memory or this chat directly.
- Save lasting facts with remember: people, routines, comfort topics, things that upset {name}.
  Facts set by a caregiver take priority; don't overwrite them from the patient's statements.
- To correct a fact in long-term memory, call remember with replaces set to the old fact, copied exactly.
- For offloaded topics mentioned earlier, use recall.
- Never mention tools or errors. Say "Let me think" or "I can't see that right now."
"""

def clean_reply(text: str) -> str:
    """Drop hidden reasoning, and any tool call the model wrote out as text
    ("Tool call: remember {...}") instead of calling, plus a copied "Assistant:" label."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.S)
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(("Tool call:", "Tool result:")))
    return re.sub(r"^\s*Assistant:\s*", "", text.strip())


class CoerceToolArgs(HookProvider):
    """llama3.1 sends every tool argument as a string ("False", "100", "None").
    Convert them to the types the tool's JSON schema expects before the call."""

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self.coerce)

    def coerce(self, event: BeforeToolCallEvent) -> None:
        if event.selected_tool is None:
            return
        schema = event.selected_tool.tool_spec["inputSchema"].get("json", {})
        props = schema.get("properties", {})
        args = event.tool_use["input"]
        if not isinstance(args, dict):
            return
        for key in list(args):
            value, spec = args[key], props.get(key)
            if spec is None:
                continue
            if value is None or (
                isinstance(value, str) and value.strip().lower() in ("none", "null", "")
            ):
                del args[key]  # omit instead of sending a literal "None"
                continue
            try:
                if spec.get("type") == "boolean" and isinstance(value, str):
                    value = value.strip().lower() == "true"
                elif spec.get("type") in ("number", "integer") and isinstance(value, str):
                    value = float(value)
                    if spec["type"] == "integer" or value.is_integer():
                        value = int(value)
            except ValueError:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "maximum" in spec:
                    value = min(value, spec["maximum"])
                if "minimum" in spec:
                    value = max(value, spec["minimum"])
            args[key] = value


class ConfirmWrites(HookProvider):
    """Ask the user before running any tool that can change data."""

    WRITE_TOOL = re.compile(r"^bee_(create|update|delete|complete|accept|dismiss)_")

    def __init__(self, ask: Callable[[str], str] = input) -> None:
        self.ask = ask  # simulate.py passes a function that always declines
        self.pending: dict | None = None  # the call waiting for an answer, for server.py

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeToolCallEvent, self.confirm)

    def needs_confirmation(self, name: str) -> bool:
        return bool(self.WRITE_TOOL.match(name))

    def confirm(self, event: BeforeToolCallEvent) -> None:
        name, args = event.tool_use["name"], event.tool_use["input"]
        if not self.needs_confirmation(name):
            return
        print(f"\n  The assistant wants to run {name} with:")
        print(json.dumps(args, indent=2, ensure_ascii=False))
        self.pending = {"tool": name, "args": args}
        try:
            answer = self.ask("Allow? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        finally:
            self.pending = None
        if answer not in ("y", "yes"):
            event.cancel_tool = (
                f"The user declined to run {name}. Do not retry it; "
                "ask the user what they want instead."
            )


LOG_DIR = Path(__file__).parent / "logs"


class ConversationLogger(HookProvider):
    """Write each turn (customer, tool calls, assistant) to logs/<session start time>.log."""

    def __init__(self, log_dir: Path = LOG_DIR) -> None:
        log_dir.mkdir(exist_ok=True)
        self.path = log_dir / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
        self.file = self.path.open("a", encoding="utf-8", buffering=1)  # line-buffered

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterToolCallEvent, self.log_tool)

    def write(self, role: str, text: str) -> None:
        self.file.write(f"[{datetime.now():%H:%M:%S}] {role}: {text}\n")

    def log_tool(self, event: AfterToolCallEvent) -> None:
        args = json.dumps(event.tool_use["input"], ensure_ascii=False)
        status = "declined" if event.cancel_message else event.result["status"]
        self.write("Tool", f"{event.tool_use['name']} {args} -> {status}")

    def close(self) -> None:
        self.file.close()


class TokenUsagePrinter(HookProvider):
    """Print the input token count after every model call."""

    def __init__(self, logger: "ConversationLogger") -> None:
        self.logger = logger

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterModelCallEvent, self.report)

    def report(self, event: AfterModelCallEvent) -> None:
        if event.stop_response is None:
            return
        usage = event.stop_response.message.get("metadata", {}).get("usage", {})
        line = f"input tokens: {usage.get('inputTokens', '?')} (output: {usage.get('outputTokens', '?')})"
        print(f"\n[{line}]")
        self.logger.write("Model", line)


def conversations_tool(store: ConversationStore):
    @tool
    def conversations(day: str = "", person: str = "") -> str:
        """Read the saved Bee conversations for a day: summaries and who said what.
        Use it for who the user talked to, what was said, or a past day.

        Args:
            day: Date as YYYY-MM-DD; leave empty for today.
            person: Only conversations mentioning this name; leave empty for all.
        """
        try:
            when = date.fromisoformat(day) if day else date.today()
        except ValueError:
            return f"'{day}' is not a YYYY-MM-DD date."
        store.sync()
        return store.read(when, person)

    return conversations


class BeeSession:
    """One chat session: topic screen, agent and memory. Used by the interactive
    loop below and by simulate.py, which points it at scratch memory."""

    def __init__(self, screen: TopicScreen, memory_dir: Path = MEMORY_DIR, log_dir: Path = LOG_DIR,
                 confirm: Callable[[str], str] = input,
                 conversations: "ConversationStore | Path | None" = CONVERSATION_DIR) -> None:
        self.screen = screen
        # Local copy of the Bee conversations (None: no access, e.g. simulations).
        self.conversations = ConversationStore(conversations) if isinstance(conversations, Path) else conversations
        self.logger = ConversationLogger(log_dir)
        # Session memory (this run) + persistent memory (across runs); see memory.py.
        self.memory = MemoryStore(bedrock_model, memory_dir)
        self.memory.start_session()
        self.confirm_writes = ConfirmWrites(confirm)
        self.agent = Agent(
            model=bedrock_model,
            tools=[bee_mcp(), self.memory.remember_tool(), self.memory.recall_tool(),
                   *([conversations_tool(self.conversations)] if self.conversations else [])],
            system_prompt=base_prompt(),
            messages=self.memory.seed_messages(),
            conversation_manager=TopicOffloadManager(self.memory, max_exchanges=6, keep_exchanges=2),
            hooks=[CoerceToolArgs(), self.confirm_writes, self.logger, TokenUsagePrinter(self.logger)],
            plugins=[BeeSteering(self.memory, self.logger.write, CAREGIVER, CAREGIVER_PHONE)],
        )
        self.last_reply = ""
        self.last_recalled: list[tuple[str, float]] = []  # (topic, P) preloaded for the last message

    def on_topic(self, user_input: str) -> bool:
        # Anything about a stored memory item is on topic, even if it reads like trivia.
        if self.memory.mentions_memory(user_input):
            self.logger.write("Screen", "matches memory, skipped")
            return True
        # Safety first: a fall, being lost, or a scam is never off topic.
        if needs_care(user_input):
            self.logger.write("Screen", "safety concern, skipped")
            return True
        if small_talk(user_input):
            self.logger.write("Screen", "small talk, skipped")
            return True
        # The user telling us about themselves is always on topic: it may need remembering.
        if states_fact(user_input):
            self.logger.write("Screen", "statement about the user, skipped")
            return True
        p_on_topic = self.screen.p_on_topic(user_input, self.last_reply, self.memory.screen_context())
        self.logger.write("Screen", f"P(on topic) = {p_on_topic:.3f}")
        return p_on_topic >= THRESHOLD

    def preload_topics(self, user_input: str) -> None:
        """Let the topic screen pick offloaded topics this message needs and put
        their details in the opening message, so the LLM has them without calling
        recall. Rebuilt every turn, so last turn's picks don't pile up."""
        topics = {n: t["summary"] for n, t in self.memory.session.get("topics", {}).items()}
        picked = self.screen.topics_to_recall(user_input, self.last_reply, topics) if topics else []
        self.last_recalled = picked
        if topics:
            self.logger.write("Recall", ", ".join(f"{n} ({p:.2f})" for n, p in picked) or "no topic needed")
        # agent.messages[:2] is always the seed pair (TopicOffloadManager.PINNED).
        self.agent.messages[:2] = self.memory.seed_messages([n for n, _ in picked])
        if self.conversations:
            self.logger.write("Conversations", self.conversations.sync())
            today = self.conversations.day_index(date.today()) or "- none saved yet"
            self.agent.messages[0]["content"][0]["text"] += (
                "\n\nToday's recorded conversations. I took part in all of them; the summaries may call me "
                f"\"the man\", \"the father\" or \"the husband\". Answer questions about my day from these:\n{today}")

    def ask(self, user_input: str) -> str:
        """Handle one customer message and return the reply."""
        self.logger.write("Customer", user_input)
        if not self.on_topic(user_input):
            print(NOT_RELATED)
            self.logger.write("Assistant", NOT_RELATED)
            return NOT_RELATED
        self.memory.add_turn("user", user_input)
        self.preload_topics(user_input)
        # Nova writes its reasoning inline as <thinking>...</thinking>; keep it out of the reply.
        self.last_reply = clean_reply(str(self.agent(user_input)))
        self.logger.write("Assistant", self.last_reply)
        self.memory.add_turn("assistant", self.last_reply)
        return self.last_reply

    def close(self) -> None:
        print("Saving session memory to persistent memory...")
        self.memory.end_session()
        self.logger.close()


def main() -> None:
    print("Loading topic screen (strands-decider)...")
    session = BeeSession(TopicScreen())
    print(f"Logging conversation to {session.logger.path}")
    try:
        while True:
            user_input = input("\nCustomer: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if user_input:
                print()
                session.ask(user_input)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
    finally:
        session.close()


if __name__ == "__main__":
    main()
