"""
An LLM stand-in for the hard-question tests. It replays replies recorded from the real model, so the tests are deterministic and cost nothing;
only when told to (live=True) does it fetch a reply that is not recorded yet from OpenRouter, and record it.

A reply is found by the exact system prompt and user prompt (a hash of both). If the prompts change - the analysis or scoring wording, the
stories, or which candidates the databases return - the old reply is not found and the test says to record again.

Answers are never recorded or paid for: these tests measure which stories are found, not how an answer is worded.
Every real call is counted against `max_calls`, and the budget is checked before the call is made. The count is kept in the file and
survives discarding replies whose prompts changed: calls already paid for stay spent.
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ask_stories as ask  # noqa: E402
import intent as it  # noqa: E402

ANSWER_STUB = "(answer not recorded: these tests measure which stories are found)"
ROLES = {it.ANALYZE_SYSTEM: "analyze", it.SCORE_SYSTEM: "score", ask.JUDGE_SYSTEM: "judge"}


class CassetteMiss(RuntimeError):
    """A prompt with no recorded reply, outside live mode."""


class BudgetExceeded(RuntimeError):
    """Recording another reply would pass `max_calls`."""


class Cassette:
    def __init__(self, path, live: bool = False, max_calls: int = 100, model: str = ask.DEFAULT_MODELS["openrouter"],
                 fetch: Optional[Callable[[str, str], str]] = None):
        self.path, self.live, self.max_calls, self.model = Path(path), live, max_calls, model
        self.fetch = fetch or (lambda system, user: ask.chat("openrouter", model, system, user))
        saved = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"meta": {}, "replies": {}}
        self.entries: Dict[str, dict] = saved["replies"]
        self.spent: int = saved["meta"].get("calls_made", len(self.entries))      # real calls ever made for this file
        self.fetched = 0                                   # real calls made through this object

    @staticmethod
    def key(system: str, user: str) -> str:
        return hashlib.sha256((system + "\x00" + user).encode("utf-8")).hexdigest()[:20]

    def __call__(self, system: str, user: str) -> str:
        if system == ask.ANSWER_SYSTEM:
            return ANSWER_STUB
        key, role = self.key(system, user), ROLES.get(system, "other")
        if key in self.entries:
            return self.entries[key]["reply"]
        if not self.live:
            raise CassetteMiss(f"no recorded reply for this {role} prompt (it starts {user[:70]!r}); record it with: python test/record_hard_questions.py")
        if self.spent >= self.max_calls:
            raise BudgetExceeded(f"{self.spent} API calls have been made for this file already and the limit is {self.max_calls}")
        reply = self.fetch(system, user)
        if ask.failed(reply):
            raise RuntimeError(f"the API call failed and nothing was recorded: {reply[:200]}")
        self.fetched += 1
        self.spent += 1
        self.entries[key] = {"role": role, "asked": user[:120], "reply": reply}
        self.save()
        return reply

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {"meta": {"model": self.model, "temperature": 0, "calls_made": self.spent, "replies": len(self.entries)}, "replies": dict(sorted(self.entries.items()))}
        self.path.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
