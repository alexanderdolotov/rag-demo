"""
A small dummy "story" corpus for trying retrieval ideas on OpenSearch before pointing them at real filings.

Every story topic has an original and a few LLM-style rewrites that say the same thing in different words
(toad hopped / amphibian leapt / import duties / customs levies ...), plus traps: polysemy (stock prices
"jumped", river "bank" vs central "bank"), negation ("did not jump"), and unrelated distractors.

Searches compared (each can be restricted to original texts only, to see what the index itself understands
without any rewriting):
  plain   BM25 on the raw words
  stem    BM25 with stop-word removal and stemming
  syn     stem + a synonym list applied to the query
  dense   k-NN on sentence embeddings (needs sentence-transformers)
  hybrid  syn + dense merged by reciprocal-rank fusion
  story   hybrid collapsed to one row per story, originals weighted above rewrites

Talks to OpenSearch over plain HTTP (OPENSEARCH_URL, default http://localhost:9200) with `requests` only.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import requests

URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
WEIGHT = {"original": 1.0, "rewrite": 0.6}   # how much a rewrite counts next to the original

# (topic, version, text) - "orig" is the original, "rwN" are rewrites of it
STORIES: List[Tuple[str, str, str]] = [
    ("frog_log", "orig", "A small green frog sat beside the pond at dawn. It jumped onto a mossy log and croaked at the rising sun. Then it leaped into the water and swam away."),
    ("frog_log", "rw1", "A tiny toad rested by the marsh at sunrise. It hopped up onto a fallen tree trunk and called out to the morning light before springing into the shallows."),
    ("frog_log", "rw2", "At first light an amphibian climbed from the water's edge, bounded onto driftwood, sang to the sun, and dived back in."),
    ("frog_log", "rw3", "At daybreak a green frog perched near the pond, vaulted atop a damp timber log, vocalized toward the sunrise, and plunged into the pond."),

    ("frog_still", "orig", "The frog did not jump that morning. A heron stood nearby, so it stayed motionless in the reeds until dark."),
    ("frog_still", "rw1", "The toad refused to hop all day, frozen among the cattails while a large bird watched the pond."),

    ("stock_jump", "orig", "Shares of the retailer jumped 12 percent after earnings beat expectations, and analysts raised their price targets."),
    ("stock_jump", "rw1", "The retailer's stock surged twelve percent following a strong earnings report, and analysts lifted their targets."),

    ("tariff_margin", "orig", "Gross margin was 50.1 percent, including a favorable impact of about two points from tariff refunds."),
    ("tariff_margin", "rw1", "Profit margins benefited by roughly two percentage points because the company recovered import duties it had paid."),
    ("tariff_margin", "rw2", "Margins improved as customs levies on imported goods were refunded."),

    ("tariff_hurt", "orig", "New tariffs on imported components raised our costs and squeezed gross margin by about three points."),
    ("tariff_hurt", "rw1", "Higher import duties on parts increased expenses and compressed profitability."),

    ("bank_rates", "orig", "The central bank raised interest rates by a quarter point to slow inflation."),
    ("bank_rates", "rw1", "The monetary authority lifted its benchmark rate by twenty-five basis points to cool rising prices."),

    ("river_bank", "orig", "The frog sat on the muddy bank of the river as the water rose."),
    ("river_bank", "rw1", "A toad rested on the soggy shore of the stream while the current climbed."),

    ("rocket", "orig", "The rocket lifted off at noon and reached orbit nine minutes later, carrying a weather satellite."),
    ("rocket", "rw1", "A launch vehicle left the pad at midday and arrived in space in under ten minutes with a climate observation craft."),
    ("bread", "orig", "She kneaded the dough, let it rise for an hour, and baked two golden loaves of bread."),
    ("bread", "rw1", "After working the dough and letting it proof, she baked a pair of brown loaves."),
    ("lost_dog", "orig", "The lost dog wandered through the neighborhood until a child found it and returned it to its owner."),
    ("lost_dog", "rw1", "A missing puppy roamed the streets until a boy discovered it and brought it home."),
    ("mountain_hike", "orig", "They hiked up the steep mountain trail and rested at a stone hut near the summit."),
]

# query id, text, topics that count as a correct answer
QUERIES: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("q1", "frog jump", ("frog_log",)),
    ("q2", "amphibian leaping onto wood", ("frog_log",)),
    ("q3", "the toad hopped by a tree trunk", ("frog_log",)),
    ("q4", "frog refused to jump", ("frog_still",)),
    ("q5", "prices jumped after earnings", ("stock_jump",)),
    ("q6", "central bank raises interest rates", ("bank_rates",)),
    ("q7", "Is the company being hurt by import duties?", ("tariff_margin", "tariff_hurt")),
    ("q8", "customs levies refunded", ("tariff_margin",)),
]

# A second query set, written after the first with less overlap in wording (words missing from the whole corpus,
# wrong-polarity and wrong-sense distractors, a story that has no rewrites). No parameter was tuned on it, but it
# was written knowing the corpus, so read it as "harder", not as an independent benchmark.
HELDOUT: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("h1", "a small green animal climbed onto a piece of timber at sunrise", ("frog_log",)),
    ("h2", "the creature hid from the bird and would not move", ("frog_still",)),
    ("h3", "costs went up because of new levies on foreign goods", ("tariff_hurt",)),
    ("h4", "did the firm get money back from customs authorities", ("tariff_margin",)),
    ("h5", "lender raises borrowing costs to fight rising prices", ("bank_rates",)),
    ("h6", "something took off and went into space", ("rocket",)),
    ("h7", "baking a loaf at home", ("bread",)),
    ("h8", "pet went missing and a kid brought it back", ("lost_dog",)),
    ("h9", "walking to the top of a hill and sheltering in a cabin", ("mountain_hike",)),
    ("h10", "an amphibian stayed still because a predator was nearby", ("frog_still",)),
    ("h11", "toad resting at the edge of a stream", ("river_bank",)),
]

# Queries for the nine stories that exist only in the on-disk corpus (stories); near-misses are the point
# (shipping costs are not import duties, a night watchman's 'duties' are not import duties, village 'customs' are not customs levies).
NEW: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("n1", "did the company suffer a cyberattack?", ("cyber_incident",)),
    ("n2", "who will replace the chief executive?", ("ceo_departure",)),
    ("n3", "ships stuck at port pushing up freight costs", ("supply_chain",)),
    ("n4", "management lowered its outlook for the year", ("guidance_cut",)),
    ("n5", "how much was the patent lawsuit settled for?", ("litigation",)),
    ("n6", "the lender set aside money for bad loans", ("regional_bank",)),
    ("n7", "what were the night watchman's responsibilities?", ("job_duties",)),
    ("n8", "what traditions does the village keep at its harvest festival?", ("village_customs",)),
    ("n9", "the company will buy back its own stock", ("buyback",)),
]

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


def make_docs() -> List[Dict]:
    return [
        {"id": f"{topic}:{version}", "topic": topic, "version": version,
         "kind": "original" if version == "orig" else "rewrite", "text": text}
        for topic, version, text in STORIES
    ]


def index_body(dim: Optional[int], meta: Optional[dict] = None) -> dict:
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
                   "stem": {"type": "stemmer", "language": "english"}},
        "analyzer": {
            "story_stem": {"tokenizer": "standard", "filter": ["lowercase", "stop", "stem"]},
            "story_syn": {"tokenizer": "standard", "filter": ["lowercase", "stop", "syn", "stem"]},
        }}}
    if dim:
        settings["index"] = {"knn": True}
        properties["embedding"] = {"type": "knn_vector", "dimension": dim, "method": {
            "name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}}
    mappings: dict = {"properties": properties}
    if meta:
        mappings["_meta"] = meta           # e.g. the fingerprint of the corpus this index was loaded from
    return {"settings": settings, "mappings": mappings}


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


def build_index(index: str, embedder=None, docs: Optional[List[Dict]] = None, meta: Optional[dict] = None) -> int:
    """(Re)create `index` and load the documents (default: the small in-code test stories). Only `index` is touched. Returns the document count."""
    docs = make_docs() if docs is None else docs
    vectors = embed(embedder, [d["text"] for d in docs]) if embedder else None
    requests.delete(f"{URL}/{index}", timeout=30)   # a 404 on the first run is fine
    call("PUT", f"/{index}", index_body(len(vectors[0]) if vectors else None, meta))
    lines: List[str] = []
    for i, d in enumerate(docs):
        source = {k: d[k] for k in ("topic", "version", "kind", "text")}
        if vectors:
            source["embedding"] = vectors[i]
        lines += [json.dumps({"index": {"_index": index, "_id": d["id"]}}), json.dumps(source)]
    result = call("POST", "/_bulk?refresh=true", ndjson="\n".join(lines) + "\n")
    if result.get("errors"):
        raise RuntimeError("bulk load had errors: " + json.dumps(result)[:600])
    return len(docs)


def drop_index(index: str) -> None:
    requests.delete(f"{URL}/{index}", timeout=30)


def index_state(index: str) -> Optional[dict]:
    """None when the index does not exist, else {"docs": document count, "meta": the _meta it was created with}."""
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


def rrf(rankings: Sequence[Sequence[Dict]], k: int = 60) -> List[Dict]:
    """Reciprocal-rank fusion: a document's score is the sum of 1/(k + rank) over the lists it appears in."""
    score: Dict[str, float] = {}
    hit: Dict[str, Dict] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, start=1):
            score[h["id"]] = score.get(h["id"], 0.0) + 1.0 / (k + rank)
            hit[h["id"]] = h
    return [dict(hit[i], score=score[i]) for i in sorted(score, key=score.get, reverse=True)]


def story_level(fused: Sequence[Dict]) -> List[Dict]:
    """One row per story: the best version's score, rewrites weighted below originals; `via` = version that matched."""
    best: Dict[str, Dict] = {}
    for h in fused:
        score = h["score"] * WEIGHT[h["kind"]]
        if h["topic"] not in best or score > best[h["topic"]]["score"]:
            best[h["topic"]] = {"id": h["topic"], "topic": h["topic"], "version": "story",
                                "kind": "story", "score": score, "via": h["version"]}
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def run_arms(index: str, query: str, embedder=None, kind: Optional[str] = None, size: int = 10) -> Dict[str, List[Dict]]:
    arms = {
        "plain": bm25(index, "text", query, size, kind),
        "stem": bm25(index, "text.stem", query, size, kind),
        "syn": bm25(index, "text.syn", query, size, kind),
    }
    if embedder:
        arms["dense"] = knn(index, embed(embedder, [query])[0], size, kind)
        arms["hybrid"] = rrf([arms["syn"], arms["dense"]])
    if kind is None:
        arms["story"] = story_level(arms.get("hybrid") or rrf([arms["syn"]]))
    return arms


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def first_relevant_rank(hits: Sequence[Dict], expect: Sequence[str]) -> Optional[int]:
    for rank, h in enumerate(hits, start=1):
        if h["topic"] in expect:
            return rank
    return None


def scoreboard(index: str, embedder=None, kind: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """{arm: {"hit3": queries with a correct answer in the top 3, "mrr": mean reciprocal rank, "n": queries}}."""
    totals: Dict[str, Dict[str, float]] = {}
    for _, text, expect in QUERIES:
        for arm, hits in run_arms(index, text, embedder, kind).items():
            rank = first_relevant_rank(hits, expect)
            t = totals.setdefault(arm, {"hit3": 0, "mrr": 0.0, "n": 0})
            t["n"] += 1
            t["hit3"] += 1 if rank and rank <= 3 else 0
            t["mrr"] += 1.0 / rank if rank else 0.0
    for t in totals.values():
        t["mrr"] = round(t["mrr"] / t["n"], 3)
    return totals
