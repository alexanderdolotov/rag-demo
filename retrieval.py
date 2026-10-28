"""
What the OpenSearch and the Neo4j side of the story experiments share, so neither has to import the other: the sentence-embedding
model, reciprocal-rank fusion, collapsing versions to stories, running a chosen subset of search arms, and the scoring
(rank of the first correct story, hit@3, mean reciprocal rank).

Every search arm returns ranked rows shaped {id, topic, version, kind, score}; everything here works on that shape.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def load_embedder(model: str = MODEL):
    """The sentence-transformers model, or None when the package is not installed."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    embedder = SentenceTransformer(model)
    embedder.max_seq_length = 512   # the default 256 tokens would silently cut a 250-word story short
    return embedder


def embed(embedder, texts: Sequence[str]) -> List[List[float]]:
    return embedder.encode(list(texts), normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Combining rankings
# ---------------------------------------------------------------------------
def rrf(rankings: Sequence[Sequence[Dict]], k: int = 60) -> List[Dict]:
    """Reciprocal-rank fusion: a document's score is the sum of 1/(k + rank) over the lists it appears in."""
    score: Dict[str, float] = {}
    hit: Dict[str, Dict] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            score[h["id"]] = score.get(h["id"], 0.0) + 1.0 / (k + rank)
            hit[h["id"]] = h
    return [dict(hit[i], score=score[i]) for i in sorted(score, key=score.get, reverse=True)]


def story_level(hits: Sequence[Dict]) -> List[Dict]:
    """One row per story, best first: the score of its best version (a tie goes to the original); `via` = the version that matched.

    The scores are used as they are. Neighbouring ranks differ by about 1% in a fused (RRF) score, so any multiplier, such as a discount
    for rewrites, would push a top-ranked rewrite below originals that sit far down the list."""
    best: Dict[str, Tuple[Tuple[float, bool], Dict]] = {}
    for h in hits:
        key = (h["score"], h["kind"] == "original")
        if h["topic"] not in best or key > best[h["topic"]][0]:
            best[h["topic"]] = (key, {"id": h["topic"], "topic": h["topic"], "version": "story",
                                      "kind": "story", "score": h["score"], "via": h["version"]})
    return sorted((row for _, row in best.values()), key=lambda row: row["score"], reverse=True)


# ---------------------------------------------------------------------------
# Running arms
# ---------------------------------------------------------------------------
Hits = List[Dict]


def run_selected(builders: Dict[str, Callable[[Callable[[str], Hits]], Hits]], only: Optional[Sequence[str]] = None) -> Dict[str, Hits]:
    """Run search arms. `builders` maps an arm name to a function that takes `get` and returns that arm's ranked rows; `get(name)` hands
    over another arm's rows (the one `hybrid` is fused from, say), running it at most once. With `only`, just those arms are returned and
    an arm nobody asked for or needs is never run, so no request is sent for it. Default: every arm, in the order of `builders`."""
    done: Dict[str, Hits] = {}

    def get(name: str) -> Hits:
        if name not in done:
            done[name] = builders[name](get)
        return done[name]

    wanted = list(builders) if only is None else list(only)
    unknown = [name for name in wanted if name not in builders]
    if unknown:
        raise ValueError(f"no such search arm: {', '.join(unknown)} (available here: {', '.join(builders)})")
    return {name: get(name) for name in wanted}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def first_relevant_rank(hits: Sequence[Dict], expect: Sequence[str]) -> Optional[int]:
    """1-based position of the first row about one of the `expect`ed stories, or None."""
    for rank, h in enumerate(hits, start=1):
        if h["topic"] in expect:
            return rank
    return None


def summarize(ranks: Sequence[Optional[int]], top: int = 3) -> Tuple[int, float]:
    """(queries whose first correct row is within the top `top`, mean reciprocal rank) for ranks from first_relevant_rank (None = not found)."""
    return sum(1 for r in ranks if r and r <= top), sum(1.0 / r for r in ranks if r) / len(ranks)
