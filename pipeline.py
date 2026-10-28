"""
The one-shot search with concept priority: what to ask the databases, how to score what comes back and what to show. No services in here;
the caller passes in how to search (`search(db, query) -> rows`) and the LLM, so the whole logic can be tried with fakes.

    1  one search per concept per database, all in one round (a concept: its core words, its close variants and the question's narrowing
       words, each tier weighing less; see retrieval.Query). Nothing depends on an earlier result.
    2  one LLM call per concept scores every story either database found for any concept, 0-3, against that concept and the question's
       conditions (so the opposite outcome scores low), keyed by story id. How well a story fits does not depend on who found it.
    3  per database, per concept: the stories that scored STRONG or more, best first. The concepts take turns, the most important concept
       first, so a second subject is not crowded out by a first with many answers; a story counts once.
    4  nothing strong anywhere: the weak ones (score 1), so there is something to look at; nothing at all: nothing.

Without scores (no LLM, or its reply cannot be read) the databases' own order is kept, the concepts taking turns all the same.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

from intent import Concept, Intent, SCORE_SYSTEM, parse_scores, score_prompt
from retrieval import Query

STRONG = 2        # a score at or above this makes a passage a real answer

Rows = List[Dict]
Search = Callable[[str, Union[str, Query]], Rows]   # (database, query) -> rows, one per story, each with at least "topic" and "text"
LLM = Callable[[str, str], str]


def concept_query(concept: Concept, intent: Intent) -> Query:
    return Query(core=concept.core, related=concept.related, context=intent.context)


def search_all(question: str, intent: Intent, dbs: Sequence[str], search: Search) -> Dict[str, List[Rows]]:
    """{database: [rows for each concept, most important concept first]}. An analysis that was not understood means one search with the question."""
    queries: List[Union[str, Query]] = [concept_query(c, intent) for c in intent.concepts] if intent.understood else [question]
    return {db: [search(db, q) for q in queries] for db in dbs}


def score(llm: LLM, question: str, intent: Intent, concept: Concept, rows: Rows) -> Optional[Dict[str, int]]:
    """{story: score 0-3} for `rows` against one concept, from one LLM call; None when the reply does not score every story.
    The passages are listed by story name, not in the order the databases ranked them: the judge should see a set of candidates, not a
    ranking it could be swayed by."""
    rows = sorted(rows, key=lambda r: r["topic"])
    return parse_scores(llm(SCORE_SYSTEM, score_prompt(question, intent, concept, [(r["topic"], r["text"]) for r in rows])), [r["topic"] for r in rows])


def score_pool(llm: LLM, question: str, intent: Intent, found: Dict[str, List[Rows]]) -> Optional[List[Dict[str, int]]]:
    """For each concept, {story: score} over every story any database found for any concept. None when any reply cannot be used."""
    distinct = {r["topic"]: r for lists in found.values() for rows in lists for r in rows}
    if not distinct:
        return [{} for _ in intent.concepts]
    scores = []
    for concept in intent.concepts:
        got = score(llm, question, intent, concept, list(distinct.values()))
        if got is None:
            return None
        scores.append(got)
    return scores


def union_rows(per_concept: Sequence[Rows], top: int) -> Rows:
    """The rows of several concepts taking turns, the first concept first, no story twice, at most `top`."""
    queues = [list(rows) for rows in per_concept]
    out: Rows = []
    seen = set()
    while len(out) < top and any(queues):
        for queue in queues:
            while queue and queue[0]["topic"] in seen:
                queue.pop(0)
            if queue and len(out) < top:
                row = queue.pop(0)
                seen.add(row["topic"])
                out.append(row)
    return out


def rank(concepts: Sequence[Concept], found: List[Rows], scores: Optional[List[Dict[str, int]]], top: int) -> Rows:
    """What to show for one database, see the module docstring. `found` holds its rows for each concept; rows get "rel" and, when there is
    more than one concept, "concept"."""
    if scores is None:
        return union_rows([[dict(r, concept=c.name) for r in rows] for c, rows in zip(concepts, found)] if len(concepts) > 1 else found, top)
    pool: Dict[str, Dict] = {}
    for rows in found:
        for r in rows:
            pool.setdefault(r["topic"], r)                                 # in the order they were found, the first concept's rows first

    def candidates(keep: Callable[[int], bool]) -> List[Rows]:
        per_concept = []
        for concept, table in zip(concepts, scores):
            rows = [dict(r, rel=table[t], concept=concept.name) for t, r in pool.items() if keep(table[t])]
            per_concept.append(sorted(rows, key=lambda r: -r["rel"]))      # the sort is stable: equal scores stay in the order found
        return per_concept

    shown = union_rows(candidates(lambda s: s >= STRONG), top) or union_rows(candidates(lambda s: s == 1), top)
    if len(concepts) == 1:
        for r in shown:
            r.pop("concept")                                               # the tag only matters when there is more than one
    return shown
