"""
Answers to the questions the user keeps asking ("likes to know" in persistent
memory), prepared ahead of time for the home page.

Generated once a day, all in one model call, from today's date, long-term
memory and today's Bee summary, then cached in <memory dir>/likes_to_know.json.
Each answer passes the same safety rules as a chat reply (steering.reply_problems);
one that breaks a safety rule is replaced with the fixed safe reply.
"""
import threading
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from memory import Memory, _read_json, _structured, _write_json
from steering import SAFETY, _second_person, fallback_reply, reply_problems

CACHE_FILE = "likes_to_know.json"
_lock = threading.Lock()  # the web page and its embedded mobile screen may ask at once


class Answer(BaseModel):
    question: str = Field(description="The question, copied exactly.")
    answer: str = Field(description="1-2 short, warm sentences spoken to the user.")


class Answers(BaseModel):
    answers: list[Answer]


def _time_of_day(hour: int) -> str:
    return "morning" if hour < 12 else "afternoon" if hour < 17 else "evening" if hour < 21 else "night"


def _generate(model, mem: Memory, questions: list[str], today_summary: str, caregiver: str) -> dict[str, str]:
    now = datetime.now()
    listing = "\n".join(f"- {q}" for q in questions)
    result = _structured(model, Answers, f"""You are Bee, a warm and patient companion for a person living with memory loss.
Today is {now:%A, %d %B %Y}; it is {_time_of_day(now.hour)}.

LONG-TERM MEMORY ABOUT THEM (JSON):
{mem.model_dump_json(indent=2)}

TODAY'S BEE SUMMARY:
{today_summary or "(none)"}

They often ask these questions. Write the answer to show on their home screen:
{listing}

Rules for every answer:
- 1-2 short, simple sentences, spoken to them ("you"). Plain text, no lists or markdown.
- Use people's names and relationships ("your wife Margaret"), not pronouns.
- Only use facts from the memory, the date, or today's summary. If the answer isn't there, say
  "Let's ask {_second_person(caregiver)}." Never invent events or medication details.
- If memory has deceased_policy: gentle_redirect and the question is about someone who has died,
  do not say they died; speak warmly about them and invite a happy memory.
Return one answer per question, with the question copied exactly.""")
    answers = {a.question.strip(): a.answer.strip() for a in result.answers}
    # The model may reword a question; fall back to matching by position.
    return {q: answers.get(q) or (result.answers[i].answer.strip() if i < len(result.answers) else "")
            for i, q in enumerate(questions)}


def answers_for(memory_dir: Path, model, today_summary: str, caregiver: str, caregiver_phone: str,
                refresh: bool = False) -> dict:
    """The questions in "likes to know" with today's answers, generating any that
    are missing (all of them on a new day or with ``refresh``)."""
    with _lock:
        data = _read_json(memory_dir / "persistent.json", {})
        data.pop("updated_at", None)
        mem = Memory(**data)
        path = memory_dir / CACHE_FILE
        today = datetime.now().date().isoformat()
        cache = _read_json(path, {})
        if refresh or cache.get("date") != today:
            cache = {"date": today, "answers": {}}
        missing = [q for q in mem.likes_to_know if not cache["answers"].get(q)]
        if missing:
            fresh = _generate(model, mem, missing, today_summary, caregiver)
            known = mem.model_dump_json() + " " + today_summary
            for q, answer in fresh.items():
                problems = reply_problems(q, answer, mem.profile, known, caregiver, caregiver_phone)
                safety = next((kind for kind, _ in problems if kind in SAFETY), None)
                cache["answers"][q] = fallback_reply(safety, caregiver, caregiver_phone) if safety else answer
            cache["generated_at"] = datetime.now().isoformat(timespec="seconds")
            _write_json(path, cache)
        return {
            "date": cache["date"],
            "generated_at": cache.get("generated_at"),
            "items": [{"question": q, "answer": cache["answers"].get(q, "")} for q in mem.likes_to_know],
        }
