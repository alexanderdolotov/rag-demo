"""
The OpenSearch side of the story experiments: index, loader, search arms and scoring, for trying retrieval ideas before
pointing them at real filings. The corpus itself is the files in stories/ (read by corpus.py), as are the test queries.

Every story has an original and a few LLM-style rewrites that say the same thing in different words
(toad hopped / amphibian leapt / import duties / customs levies ...), plus traps: polysemy (stock prices
"jumped", river "bank" vs central "bank"), negation ("did not jump"), and unrelated distractors.

Searches compared (each can be restricted to original texts only, to see what the index itself understands
without any rewriting):
  plain   BM25 on the raw words
  stem    BM25 with stop-word removal and stemming
  syn     stem + a synonym list applied to the query
  dense   k-NN on sentence embeddings (needs sentence-transformers)
  hybrid  syn + dense merged by reciprocal-rank fusion
  story   hybrid collapsed to one row per story (the best version's score)

Talks to OpenSearch over plain HTTP (OPENSEARCH_URL, default http://localhost:9200) with `requests` only.
The model, rank fusion and scoring it shares with the Neo4j side are in retrieval.py.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import requests

from retrieval import embed, first_relevant_rank, rrf, run_selected, story_level, summarize

URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")

SYNONYMS = [
    "frog, frogs, toad, toads, amphibian, amphibians",
    "jump, jumps, jumped, jumping, hop, hops, hopped, hopping, leap, leaps, leaped, leapt, leaping, vault, vaulted, bounded",
    "log, logs, trunk, trunks, timber, driftwood, wood",
    "tariff, tariffs, duty, duties, levy, levies, customs",
    "refund, refunds, refunded, rebate, recovered",
    "margin, margins, profitability",
]


# ---------------------------------------------------------------------------
# OpenSearch plumbing
# ---------------------------------------------------------------------------
def reachable() -> bool:
    try:
        return requests.get(URL, timeout=3).ok
    except requests.RequestException:
        return False


TRACE: Optional[Callable[[str, str, Optional[dict]], None]] = None   # set to log every JSON request: (method, path, body)


def call(method: str, path: str, body: Optional[dict] = None, ndjson: Optional[str] = None) -> dict:
    if TRACE is not None and ndjson is None:
        TRACE(method, path, body)
    headers = {"Content-Type": "application/x-ndjson" if ndjson is not None else "application/json"}
    data = ndjson if ndjson is not None else (json.dumps(body) if body is not None else None)
    resp = requests.request(method, URL + path, data=data, headers=headers, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {resp.status_code}: {resp.text[:600]}")
    return resp.json() if resp.text else {}


_INDEX_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def index_name(name: str) -> str:
    """`name` if it is exactly one ordinary index; ValueError for a wildcard, a comma list, `_all`, a hidden or system index and the like.
    Deleting goes by this name, so a slip in --index must never reach other indices. Also usable as an argparse `type`."""
    if not isinstance(name, str) or not _INDEX_NAME.match(name):
        raise ValueError(f"{name!r} is not a plain index name (lowercase letters, digits, - and _, not starting with - or _)")
    return name


def index_body(dim: Optional[int]) -> dict:
    properties = {
        "topic": {"type": "keyword"}, "version": {"type": "keyword"}, "kind": {"type": "keyword"},
        # `text` = plain standard analyzer; `.stem` = stop words + stemming; `.syn` = same at index time,
        # plus the synonym list on the query side only.
        "text": {"type": "text", "fields": {
            "stem": {"type": "text", "analyzer": "story_stem"},
            "syn": {"type": "text", "analyzer": "story_stem", "search_analyzer": "story_syn"},
        }},
    }
    settings: dict = {"analysis": {
        "filter": {"syn": {"type": "synonym_graph", "synonyms": SYNONYMS},
                   "stem": {"type": "stemmer", "language": "english"},
                   # "watchman's" must match "watchman", as in Lucene's own English analyzer (which Neo4j's full-text index uses)
                   "possessive": {"type": "stemmer", "language": "possessive_english"}},
        "analyzer": {
            "story_stem": {"tokenizer": "standard", "filter": ["possessive", "lowercase", "stop", "stem"]},
            "story_syn": {"tokenizer": "standard", "filter": ["possessive", "lowercase", "stop", "syn", "stem"]},
        }}}
    if dim:
        settings["index"] = {"knn": True}
        properties["embedding"] = {"type": "knn_vector", "dimension": dim, "method": {
            "name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}}
    return {"settings": settings, "mappings": {"properties": properties}}


def build_index(index: str, embedder, docs: List[Dict], meta: Optional[dict] = None) -> int:
    """(Re)create `index` and load `docs` (from corpus.load_corpus; `embedder` may be None for keyword arms only). Only `index` is touched.
    `meta` (e.g. the fingerprint of the corpus) is stored in the mapping once every document is in, so an index whose load stopped half
    way never carries it and never looks complete. Returns the document count."""
    index_name(index)
    vectors = embed(embedder, [d["text"] for d in docs]) if embedder else None
    requests.delete(f"{URL}/{index}", timeout=30)   # a 404 on the first run is fine
    call("PUT", f"/{index}", index_body(len(vectors[0]) if vectors else None))
    lines: List[str] = []
    for i, d in enumerate(docs):
        source = {k: d[k] for k in ("topic", "version", "kind", "text")}
        if vectors:
            source["embedding"] = vectors[i]
        lines += [json.dumps({"index": {"_index": index, "_id": d["id"]}}), json.dumps(source)]
    result = call("POST", "/_bulk?refresh=true", ndjson="\n".join(lines) + "\n")
    if result.get("errors"):
        raise RuntimeError("bulk load had errors: " + json.dumps(result)[:600])
    if meta:
        call("PUT", f"/{index}/_mapping", {"_meta": meta})
    return len(docs)


def drop_index(index: str) -> None:
    requests.delete(f"{URL}/{index_name(index)}", timeout=30)


def index_state(index: str) -> Optional[dict]:
    """None when the index does not exist, else {"docs": document count, "meta": the _meta stored once its load finished}."""
    index_name(index)
    resp = requests.get(f"{URL}/{index}/_mapping", timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return {"docs": call("GET", f"/{index}/_count")["count"], "meta": resp.json()[index]["mappings"].get("_meta", {})}


def analyze(index: str, analyzer: str, text: str) -> List[str]:
    return [t["token"] for t in call("POST", f"/{index}/_analyze", {"analyzer": analyzer, "text": text})["tokens"]]


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------
_FIELDS = ["topic", "version", "kind"]


def _search(index: str, body: dict) -> List[Dict]:
    hits = call("POST", f"/{index}/_search", {**body, "_source": _FIELDS})["hits"]["hits"]
    return [{"id": h["_id"], **h["_source"], "score": h["_score"]} for h in hits]


def bm25(index: str, field: str, text: str, size: int = 10, kind: Optional[str] = None) -> List[Dict]:
    query = {"bool": {"must": [{"match": {field: text}}], "filter": [{"term": {"kind": kind}}] if kind else []}}
    return _search(index, {"size": size, "query": query})


def knn(index: str, vector: List[float], size: int = 10, kind: Optional[str] = None) -> List[Dict]:
    inner: dict = {"vector": vector, "k": size}
    if kind:
        inner["filter"] = {"term": {"kind": kind}}
    return _search(index, {"size": size, "query": {"knn": {"embedding": inner}}})


def run_arms(index: str, query: str, embedder=None, kind: Optional[str] = None, size: int = 10,
             only: Optional[Sequence[str]] = None) -> Dict[str, List[Dict]]:
    """What each search arm returns for `query` (every arm this index and embedder allow, or just those named in `only`; see retrieval.run_selected)."""
    builders: Dict[str, Callable] = {
        "plain": lambda get: bm25(index, "text", query, size, kind),
        "stem": lambda get: bm25(index, "text.stem", query, size, kind),
        "syn": lambda get: bm25(index, "text.syn", query, size, kind),
    }
    if embedder:
        builders["dense"] = lambda get: knn(index, embed(embedder, [query])[0], size, kind)
        builders["hybrid"] = lambda get: rrf([get("syn"), get("dense")])
    if kind is None:
        builders["story"] = lambda get: story_level(get("hybrid") if embedder else rrf([get("syn")]))
    return run_selected(builders, only)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def scoreboard(index: str, queries: Sequence[Tuple[str, str, Tuple[str, ...]]], embedder=None, kind: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """{arm: {"hit3": queries with a correct answer in the top 3, "mrr": mean reciprocal rank, "n": queries}} over `queries`
    (id, text, correct story ids)."""
    ranks: Dict[str, List[Optional[int]]] = {}
    for _, text, expect in queries:
        for arm, hits in run_arms(index, text, embedder, kind).items():
            ranks.setdefault(arm, []).append(first_relevant_rank(hits, expect))
    return {arm: {"hit3": hit3, "mrr": round(mrr, 3), "n": len(r)} for arm, r in ranks.items() for hit3, mrr in [summarize(r)]}
