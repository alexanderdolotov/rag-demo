"""
What the LLM makes of a question before the databases are searched, and how it scores what came back. Prompts and parsers only: no
services, and the LLM itself is passed in by the caller.

    analysis   the question -> concepts, most important first (each with core words and close variants), conditions a passage must meet,
               words that only narrow the question, and conditions no story can show; spelling is corrected on the way
    scoring    candidate passages -> a 0-3 score for each, against one concept and the conditions, keyed by story id

Both parsers refuse anything they cannot read exactly and say so, so a sloppy reply degrades to searching the question as typed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

MAX_CONCEPTS = 3
MAX_CORE, MAX_RELATED, MAX_MUST, MAX_CONTEXT, MAX_UNVERIFIABLE = 4, 8, 4, 6, 4

ANALYZE_SYSTEM = (
    "You analyse a user's question for a text search over a small collection of stories. Reply with JSON only, in exactly this shape: "
    '{"concepts": [{"name": "short phrase", "core": ["word"], "related": ["word"]}], "must": ["short condition"], "context": ["word"], '
    '"unverifiable": ["short condition"]}. Rules. Fix spelling mistakes. concepts: the separate subjects the question asks about, MOST '
    "IMPORTANT FIRST; most questions have exactly one, so use several only when the question really asks about different things. core: "
    "1-3 words that name the subject itself, the words a text on that subject would use; never a generic word such as 'company', 'story' "
    "or 'person'. related: up to 6 synonyms, close variants and typical associated words for the same subject. must: conditions that "
    "decide whether a passage is the right one and can be checked from its meaning, such as a direction ('benefited, not hurt'), a cause "
    "('shipping delays, not government policy') or a kind of thing; short phrases; empty if there are none. context: words that only "
    "narrow or colour the question (place, time, manner); weak evidence. unverifiable: conditions no story can show, such as where the "
    "user is; short phrases. Lowercase, no explanations.")
SCORE_SYSTEM = (
    "You judge how well passages match what a user is looking for. You get the question, the subject to judge against, conditions a "
    "passage must meet, and the passages, each labelled with an id in square brackets. Score every passage from 0 to 3 for how well it "
    "matches the SUBJECT while meeting the conditions: 3 = it is about the subject and meets every condition; 2 = it is about the subject "
    "but a condition is unclear, or it is about a very close subject; 1 = it only touches the subject, or it is about the subject but "
    "breaks a condition (for example the opposite outcome); 0 = unrelated. Judge by meaning, not by shared words: a company called "
    "'Foods' is not about food. Reply with a JSON object that maps every id to its score, for example {\"bread\": 3, \"dog\": 0}, and "
    "nothing else.")


@dataclass(frozen=True)
class Concept:
    name: str
    core: Tuple[str, ...]
    related: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Intent:
    concepts: Tuple[Concept, ...]            # most important first
    must: Tuple[str, ...] = ()               # conditions a passage must meet, judged on its meaning
    context: Tuple[str, ...] = ()            # words that only narrow the question
    unverifiable: Tuple[str, ...] = ()       # conditions no story can show
    understood: bool = True                  # False: the reply could not be read, and the question itself is the one concept


def flat_intent(question: str) -> Intent:
    """The question as typed, as its own single concept."""
    return Intent((Concept(question, (question,)),), understood=False)


def _json_object(reply: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _words(value, limit: int, exclude: Sequence[str] = ()) -> Tuple[str, ...]:
    """The distinct, lowercased, non-empty strings of a JSON list (nothing else is accepted), without those in `exclude`, at most `limit`."""
    out: List[str] = []
    for item in value if isinstance(value, list) else []:
        word = " ".join(item.lower().split()) if isinstance(item, str) else ""
        if word and word not in out and word not in exclude:
            out.append(word)
    return tuple(out[:limit])


def parse_intent(reply: str, question: str) -> Intent:
    """The analysis in `reply`; flat_intent(question) when it cannot be read or names no subject. A word is kept in its most important tier only."""
    data = _json_object(reply)
    concepts: List[Concept] = []
    for raw in (data.get("concepts") if data and isinstance(data.get("concepts"), list) else []):
        core = _words(raw.get("core") if isinstance(raw, dict) else None, MAX_CORE)
        if core:
            name = raw.get("name") if isinstance(raw.get("name"), str) and raw["name"].strip() else " ".join(core)
            concepts.append(Concept(" ".join(name.split())[:60], core, _words(raw.get("related"), MAX_RELATED, core)))
    if not concepts:
        return flat_intent(question)
    concepts = concepts[:MAX_CONCEPTS]
    used = {w for c in concepts for w in c.core + c.related}
    return Intent(tuple(concepts), _words(data.get("must"), MAX_MUST), _words(data.get("context"), MAX_CONTEXT, used),
                  _words(data.get("unverifiable"), MAX_UNVERIFIABLE))


def score_prompt(question: str, intent: Intent, concept: Concept, passages: Sequence[Tuple[str, str]]) -> str:
    """The user message for scoring: `passages` are (id, text)."""
    subject = f"{concept.name} (core words: {', '.join(concept.core)}" + (f"; close variants: {', '.join(concept.related)}" if concept.related else "") + ")"
    parts = [f"Question: {question}", f"Subject to judge the passages against: {subject}"]
    others = [c.name for c in intent.concepts if c != concept]
    if others:                                      # the conditions are the question's, not one subject's: say which ones are meant here
        parts.append("The question has other subjects too, judged separately and not counted here: " + "; ".join(others))
    if intent.must:
        parts.append(("Conditions from the question; each one applies only to the part of the question it is about, so use only the ones that "
                      "concern this subject: " if others else "Conditions a passage must meet: ") + "; ".join(intent.must))
    if intent.unverifiable:
        parts.append("Ignore these, a story cannot show them: " + "; ".join(intent.unverifiable))
    parts.append("Passages:\n" + "\n".join(f"[{pid}] {text}" for pid, text in passages))
    return "\n\n".join(parts)


def parse_scores(reply: str, ids: Sequence[str]) -> Optional[Dict[str, int]]:
    """{id: score 0-3} from a JSON object in `reply`; None unless every id has an integer score from 0 to 3 (extra ids are ignored)."""
    data = _json_object(reply)
    if data is None or not all(type(data.get(i)) is int and 0 <= data[i] <= 3 for i in ids):
        return None
    return {i: data[i] for i in ids}
