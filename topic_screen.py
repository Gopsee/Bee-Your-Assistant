"""
Screen each user message with strands-decider before it reaches the agent.
A yes/no (noul) question asks whether the message is on topic; anything below
the threshold is answered with NOT_RELATED instead of calling the LLM.

The same engine also picks which offloaded topics (memory.py) the message needs:
one yes/no question per topic, all evaluated in a single call, so bee.py can put
those details in the context before the LLM runs.
"""
from strands_decider.infer import load_engine
from strands_decider.schema import NoulQuestion

CHECKPOINT = "StrandsAgents/strands-decider-2B-hobson-v19"
DEVICE = "cpu"  # the 4 GB laptop GPU is too small for the 2B model
THRESHOLD = 0.5  # minimum P(on topic) to let a message through
NOT_RELATED = "Not related work"
RECALL_THRESHOLD = 0.5  # minimum P(needs topic) to preload an offloaded topic
MAX_RECALLED = 2  # preload at most this many topics per message

RECALL_CRITERIA = {
    "true": "The user's message refers back to this earlier topic (a person, plan, fact, "
    "question or answer discussed there), is a follow-up to it, or can only be answered "
    "correctly with what was said there.",
    "false": "The message is about something else, is a new question that can be answered "
    "without that earlier discussion, or is a greeting or thanks.",
}

ON_TOPIC = NoulQuestion(
    instructions="Is the user's message something a personal assistant for their Bee "
    "wearable should handle?",
    criteria={
        "true": "Questions or requests about the user's own day, conversations, "
        "to-dos, health, media, facts, preferences or memories; asking the assistant "
        "to remember or recall something; anything about a person, place, interest or "
        "fact listed in user_memory, even if it looks like general knowledge; the user "
        "telling the assistant something about themselves (hobbies, likes, people, plans); greetings, "
        "thanks or small talk with the assistant; or a follow-up to the assistant's "
        "previous reply.",
        "false": "General knowledge, trivia, homework, coding or writing tasks, news, "
        "or anything else unrelated to the user's personal Bee data, their stored "
        "memory, or this chat.",
    },
)


class TopicScreen:
    def __init__(self, checkpoint: str = CHECKPOINT, device: str = DEVICE) -> None:
        self.engine = load_engine(checkpoint, device=device)

    def p_on_topic(self, message: str, previous_reply: str = "", user_memory: str = "") -> float:
        # Include the last reply so short follow-ups ("and yesterday?") keep their context,
        # and the user's memory so questions about their known interests count as on topic.
        state = {"user_memory": user_memory, "assistant_previous_reply": previous_reply, "user_message": message}
        response = self.engine.ask(state, {"on_topic": ON_TOPIC})
        return response.answers["on_topic"].noul

    def topics_to_recall(self, message: str, previous_reply: str, topics: dict[str, str]) -> list[tuple[str, float]]:
        """Offloaded topics (name -> summary) the message needs, most likely first,
        as (name, P(needs topic)) pairs at or above RECALL_THRESHOLD."""
        if not topics:
            return []
        names = list(topics)
        questions = {
            f"topic_{i}": NoulQuestion(
                instructions=f'Does the user\'s message need details from the earlier part of this chat '
                f'about "{name}"? Summary of that part: {topics[name]}',
                criteria=RECALL_CRITERIA,
            )
            for i, name in enumerate(names)
        }
        state = {"assistant_previous_reply": previous_reply, "user_message": message}
        answers = self.engine.ask(state, questions).answers
        scored = sorted(((n, answers[f"topic_{i}"].noul) for i, n in enumerate(names)), key=lambda x: -x[1])
        return [(n, p) for n, p in scored if p >= RECALL_THRESHOLD][:MAX_RECALLED]

    def is_related(self, message: str, previous_reply: str = "", user_memory: str = "") -> bool:
        return self.p_on_topic(message, previous_reply, user_memory) >= THRESHOLD
