"""
Graph-database side of the story experiments (see corpus.py for the corpus, opensearch_stories.py for the search-engine arms and
retrieval.py for what the two share).

The same story texts go into Neo4j as a small graph:

    (:StoryVersion {id, topic, version, kind, text, embedding})-[:OF]->(:Story {id})
    (:StoryVersion)-[:HAS {tf}]->(:StoryTerm {stem, idf, embedding})
    (:StoryTerm)-[:CO {n, npmi}]-(:StoryTerm)      terms that appear in the same story (any of its versions)

The rewrites are what makes the graph useful: "toad" and "frog" never occur in the same sentence, but they do occur
in versions of the same story, so they get a CO edge. A query is expanded along CO edges (a thesaurus learned from the
corpus instead of a hand-written synonym list) and then scored BM25-style over the HAS edges.

Arms (all return rows shaped like the OpenSearch ones: id, topic, version, kind, score):
  n_ft     Neo4j full-text index (Lucene "english" analyzer)               ~ OpenSearch `stem`
  n_vec    Neo4j vector index (cosine) on the sentence embeddings          ~ OpenSearch `dense`
  n_co     query terms + CO neighbours (npmi >= MIN_NPMI), BM25 over HAS   needs rewrites to learn from
  n_term   query words + nearest vocabulary terms by word embedding        a learned thesaurus from the model
  n_hyb    reciprocal-rank fusion of n_co, n_term and n_vec
  n_story  n_hyb collapsed to one row per story (only when rewrites are retrievable)

A graph is built for one scenario: which versions can be retrieved (`targets`) and which versions the CO edges are
learned from (`learn_from`): ("original", "original") = nothing but the originals; ("original", "all") = rewrites are
only a thesaurus and the answer is always an original; ("all", "all") = rewrites are retrievable too.

Needs `pip install neo4j nltk` and a Neo4j with the SEARCH clause for vector / full-text indexes (2026.x); env NEO4J_URI (default bolt://localhost:7687),
NEO4J_USER/NEO4J_PASSWORD (default: no auth, as in the local docker container). Only nodes labelled Story,
StoryVersion and StoryTerm (and the indexes starting with `story_`/`term_`) are created or deleted.
"""
from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from neo4j import GraphDatabase
from nltk.stem.snowball import SnowballStemmer

from retrieval import TIER_WEIGHT, Query, Searchable, embed, rrf, run_selected, story_level

# the driver reports every statement that mentions a label the database does not have yet (normal before the first load)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]) if os.environ.get("NEO4J_PASSWORD") else None

# Fixed before looking at any result; the set 2 queries (stories/queries.json) were never used to pick them.
MIN_NPMI = 0.6       # CO edges weaker than this are not used for expansion
ALPHA = 0.5          # the expansion terms together weigh at most ALPHA x the original query terms
TERM_K = 3           # nearest vocabulary terms per query word (n_term)
TERM_MIN_COS = 0.45  # ... and how similar they must be
K1, B = 1.2, 0.75    # BM25 constants for the graph scorer
STORE_NPMI = 0.3     # CO edges below this are not even stored

# Lucene's default English stop set, the same list OpenSearch's `stop` filter uses
STOP = frozenset("a an and are as at be but by for if in into is it no not of on or such that the their then there "
                 "these they this to was will with".split())
_STEMMER = SnowballStemmer("english")


# a word as Lucene's standard tokenizer sees it: letters and digits of any alphabet, an apostrophe allowed inside ("don't")
_WORD = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*")
_POSSESSIVE = re.compile(r"['\u2019]s$")


def token_spans(text: str) -> List[Tuple[str, str, int, int]]:
    """[(stem, surface word, start, end)] without stop words; start and end are where the word stands in `text`.
    "watchman's" is "watchman" and "naïve" stays whole, as in Lucene's English analyzer."""
    out = []
    for m in _WORD.finditer(text):
        word = _POSSESSIVE.sub("", m.group().lower())
        if word not in STOP:
            out.append((_STEMMER.stem(word), word, m.start(), m.end()))
    return out


def tokens(text: str) -> List[Tuple[str, str]]:
    """[(stem, surface word)] without stop words."""
    return [(stem, word) for stem, word, _, _ in token_spans(text)]


# ---------------------------------------------------------------------------
# The corpus, prepared once (tokens and vectors do not depend on the scenario)
# ---------------------------------------------------------------------------
@dataclass
class Prepared:
    docs: List[dict]              # id, topic, version, kind, text, tf (Counter of stems), length, vec
    vocab: Dict[str, dict]        # stem -> {"surface": most common word, "vec": embedding or None}
    has_vectors: bool


def prepare(embedder, docs: List[dict]) -> Prepared:
    """Tokens and vectors for `docs` (from corpus.load_corpus; `embedder` may be None for keyword arms only). The dicts given are copied, not changed."""
    docs = [dict(d) for d in docs]
    surfaces: Dict[str, Counter] = defaultdict(Counter)
    for d in docs:
        toks = tokens(d["text"])
        d["tf"] = Counter(stem for stem, _ in toks)
        d["length"] = len(toks)
        for stem, word in toks:
            surfaces[stem][word] += 1
    vocab = {stem: {"surface": c.most_common(1)[0][0], "vec": None} for stem, c in surfaces.items()}
    for d in docs:
        d["vec"] = None
    if embedder:
        for d, v in zip(docs, embed(embedder, [d["text"] for d in docs])):
            d["vec"] = v
        stems = sorted(vocab)
        for stem, v in zip(stems, embed(embedder, [vocab[s]["surface"] for s in stems])):
            vocab[stem]["vec"] = v
    return Prepared(docs, vocab, embedder is not None)


# ---------------------------------------------------------------------------
# Connection and building
# ---------------------------------------------------------------------------
def connect():
    return GraphDatabase.driver(URI, auth=AUTH)


def reachable() -> bool:
    try:
        driver = connect()
        try:
            driver.verify_connectivity()
            return True
        finally:
            driver.close()
    except Exception:
        return False


TRACE: Optional[Callable[[str, dict], None]] = None   # set to log every statement: (cypher, parameters)

# Every label, constraint and index this module creates starts with PREFIX, so the tests and compare_stories.py can build their own
# graphs in the same Neo4j without touching the loaded corpus (PREFIX "" = the real one). Community Neo4j has one database only.
PREFIX = ""
_NAMES = re.compile(r"\b(StoryVersion|StoryTerm|StoryCorpus|Story|story_version_id|story_term_stem|story_id|story_ft|story_vec|term_vec)\b")


def _ns(cypher: str) -> str:
    return _NAMES.sub(lambda m: PREFIX + m.group(1), cypher) if PREFIX else cypher


def _run(driver, cypher: str, params: Optional[dict] = None) -> List[dict]:
    if TRACE is not None:
        TRACE(cypher, params or {})            # as written here, without the namespace prefix
    with driver.session() as session:
        return session.run(_ns(cypher), params or {}).data()


def wipe(driver) -> None:
    _run(driver, "MATCH (n) WHERE n:Story OR n:StoryVersion OR n:StoryTerm OR n:StoryCorpus DETACH DELETE n")


def drop_namespace(driver) -> None:
    """Delete everything a non-empty PREFIX created: nodes, indexes and constraints. Refuses to do that to the real (unprefixed) data."""
    if not PREFIX:
        raise RuntimeError("drop_namespace is only for a PREFIXed namespace; use data_load.py --reload for the loaded corpus")
    wipe(driver)
    for name in ("story_vec", "term_vec", "story_ft"):
        _run(driver, f"DROP INDEX {name} IF EXISTS")
    for name in ("story_id", "story_version_id", "story_term_stem"):
        _run(driver, f"DROP CONSTRAINT {name} IF EXISTS")


def _co_edges(learn_docs: Sequence[dict]) -> List[dict]:
    """Story-level co-occurrence with normalised pointwise mutual information (1.0 = the two always appear together)."""
    stems_of: Dict[str, set] = defaultdict(set)
    for d in learn_docs:
        stems_of[d["topic"]].update(d["tf"])
    n = len(stems_of)
    df: Counter = Counter()
    pair: Counter = Counter()
    for stems in stems_of.values():
        df.update(stems)
        pair.update(combinations(sorted(stems), 2))
    edges = []
    for (a, b), c in pair.items():
        p_ab = c / n
        npmi = 1.0 if p_ab >= 1.0 else math.log(p_ab / ((df[a] / n) * (df[b] / n))) / -math.log(p_ab)
        if npmi >= STORE_NPMI:
            edges.append({"a": a, "b": b, "n": c, "npmi": round(npmi, 4)})
    return edges


def build_graph(driver, prep: Prepared, targets: str = "all", learn_from: str = "all", embedder=None, meta: Optional[dict] = None) -> "StoryGraph":
    """(Re)build the graph for one scenario; see the module docstring for `targets` / `learn_from`."""
    if targets not in ("original", "all") or learn_from not in ("original", "all"):
        raise ValueError("targets and learn_from must be 'original' or 'all'")
    if targets == "all" and learn_from == "original":
        raise ValueError("the edges cannot be learned from fewer versions than are retrievable")
    pick = lambda scope: [d for d in prep.docs if scope == "all" or d["kind"] == "original"]  # noqa: E731
    target_docs, learn_docs = pick(targets), pick(learn_from)

    wipe(driver)
    for name in ("story_vec", "term_vec"):
        _run(driver, f"DROP INDEX {name} IF EXISTS")
    for cypher in (
        "CREATE CONSTRAINT story_id IF NOT EXISTS FOR (s:Story) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT story_version_id IF NOT EXISTS FOR (v:StoryVersion) REQUIRE v.id IS UNIQUE",
        "CREATE CONSTRAINT story_term_stem IF NOT EXISTS FOR (t:StoryTerm) REQUIRE t.stem IS UNIQUE",
        "CREATE FULLTEXT INDEX story_ft IF NOT EXISTS FOR (v:StoryVersion) ON EACH [v.text] "
        "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}",
    ):
        _run(driver, cypher)
    if prep.has_vectors:
        for name, label, prop, dim in (("story_vec", "StoryVersion", "embedding", len(target_docs[0]["vec"])),
                                       ("term_vec", "StoryTerm", "embedding", len(next(iter(prep.vocab.values()))["vec"]))):
            _run(driver, f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:{label}) ON n.{prop} "
                         f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}}")

    # terms: everything the edges are learned from; idf over the retrievable versions only
    n_docs = len(target_docs)
    df = Counter(stem for d in target_docs for stem in d["tf"])
    stems = sorted({stem for d in learn_docs for stem in d["tf"]})
    _run(driver, "UNWIND $rows AS r CREATE (:StoryTerm {stem: r.stem, surface: r.surface, idf: r.idf, df: r.df, embedding: r.vec})",
         {"rows": [{"stem": s, "surface": prep.vocab[s]["surface"], "df": df.get(s, 0),
                    "idf": math.log(1 + (n_docs - df.get(s, 0) + 0.5) / (df.get(s, 0) + 0.5)),
                    "vec": prep.vocab[s]["vec"]} for s in stems]})
    _run(driver, "UNWIND $rows AS r MERGE (:Story {id: r.topic})", {"rows": [{"topic": t} for t in sorted({d["topic"] for d in target_docs})]})
    _run(driver, "UNWIND $rows AS r CREATE (v:StoryVersion {id: r.id, topic: r.topic, version: r.version, kind: r.kind, "
                 "text: r.text, length: r.length, embedding: r.vec}) WITH v, r MATCH (s:Story {id: r.topic}) CREATE (v)-[:OF]->(s)",
         {"rows": [{k: d[k] for k in ("id", "topic", "version", "kind", "text", "length")} | {"vec": d["vec"]} for d in target_docs]})
    _run(driver, "UNWIND $rows AS r MATCH (v:StoryVersion {id: r.id}), (t:StoryTerm {stem: r.stem}) CREATE (v)-[:HAS {tf: r.tf}]->(t)",
         {"rows": [{"id": d["id"], "stem": s, "tf": tf} for d in target_docs for s, tf in d["tf"].items()]})
    edges = _co_edges(learn_docs)
    for i in range(0, len(edges), 2000):
        _run(driver, "UNWIND $rows AS r MATCH (a:StoryTerm {stem: r.a}), (b:StoryTerm {stem: r.b}) CREATE (a)-[:CO {n: r.n, npmi: r.npmi}]->(b)",
             {"rows": edges[i:i + 2000]})
    _run(driver, "CALL db.awaitIndexes(60)")
    avg_len = sum(d["length"] for d in target_docs) / n_docs
    counts = {"versions": n_docs, "terms": len(stems), "co_edges": len(edges)}
    # one node that says what was loaded: the corpus fingerprint, the scenario, and what a later session needs to search it
    _run(driver, "CREATE (:StoryCorpus {corpus_hash: $hash, targets: $targets, learn_from: $learn_from, avg_len: $avg_len, "
                 "versions: $versions, stories: $stories, terms: $terms, co_edges: $co_edges, loaded_at: $loaded_at})",
         {"hash": (meta or {}).get("corpus_hash", ""), "targets": targets, "learn_from": learn_from, "avg_len": avg_len,
          "stories": len({d["topic"] for d in target_docs}), "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **counts})
    return StoryGraph(driver, prep, targets, learn_from, avg_len=avg_len, counts=counts, embedder=embedder)


def read_meta(driver) -> Optional[dict]:
    """What the loaded graph says about itself (see build_graph), or None when no story graph is loaded."""
    rows = _run(driver, "MATCH (m:StoryCorpus) RETURN properties(m) AS meta")
    return rows[0]["meta"] if rows else None


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------
_COLUMNS = "node.id AS id, node.topic AS topic, node.version AS version, node.kind AS kind, score"

_BM25 = """
UNWIND $weights AS w
MATCH (v:StoryVersion)-[h:HAS]->(t:StoryTerm {stem: w.stem})
WITH v, sum(w.w * t.idf * (h.tf * ($k1 + 1)) / (h.tf + $k1 * (1 - $b + $b * v.length / $avglen))) AS score
RETURN v.id AS id, v.topic AS topic, v.version AS version, v.kind AS kind, score
ORDER BY score DESC, id LIMIT $k
"""

_CO_NEIGHBOURS = """
UNWIND $stems AS s
MATCH (q:StoryTerm {stem: s})-[c:CO]-(n:StoryTerm)
WHERE c.npmi >= $min_npmi AND NOT n.stem IN $stems
RETURN n.stem AS stem, n.surface AS word, max(c.npmi) AS strength, collect(DISTINCT q.surface) AS via
"""


@dataclass
class StoryGraph:
    driver: object
    prep: Prepared
    targets: str
    learn_from: str
    avg_len: float
    counts: Dict[str, int]
    embedder: object = None   # the sentence-transformers model, for the query side of n_vec / n_term

    @classmethod
    def attach(cls, driver, embedder=None) -> "StoryGraph":
        """A searchable handle on the graph that is already loaded (no rebuild); LookupError when nothing is loaded."""
        meta = read_meta(driver)
        if meta is None:
            raise LookupError("no story graph is loaded in Neo4j (python data_load.py)")
        counts = {k: meta[k] for k in ("versions", "terms", "co_edges")}
        return cls(driver, Prepared([], {}, embedder is not None), meta["targets"], meta["learn_from"], meta["avg_len"], counts, embedder)

    # --- building blocks ---------------------------------------------------
    def _bm25(self, weights: Dict[str, float], size: int) -> List[Dict]:
        rows = [{"stem": s, "w": w} for s, w in weights.items() if w > 0]
        if not rows:
            return []
        return _run(self.driver, _BM25, {"weights": rows, "k": size, "k1": K1, "b": B, "avglen": self.avg_len})

    def _known(self, stems: Sequence[str]) -> set:
        """The given stems that exist in the graph (a word the corpus has never seen cannot be expanded by CO edges)."""
        if not stems:
            return set()
        return {r["stem"] for r in _run(self.driver, "MATCH (t:StoryTerm) WHERE t.stem IN $stems RETURN t.stem AS stem", {"stems": list(stems)})}

    def query_stems(self, text: str) -> List[str]:
        stems = sorted({stem for stem, _ in tokens(text)})
        known = self._known(stems)
        return [s for s in stems if s in known]

    def _weights(self, query: Searchable) -> Tuple[Dict[str, float], List[str]]:
        """({stem: weight} for the query's words the graph knows, the stems of the words that name the subject). A plain string weighs every
        word 1.0 and all of them name the subject. With tiers a word weighs what its most important tier says (TIER_WEIGHT) and only the
        core words name the subject, so only they are expanded."""
        if not isinstance(query, Query):
            stems = self.query_stems(query)
            return {s: 1.0 for s in stems}, stems
        per_tier = [(tier, sorted({stem for stem, _ in tokens(" ".join(words))})) for tier, words in query.tiers()]
        known = self._known(sorted({s for _, stems in per_tier for s in stems}))
        weights: Dict[str, float] = {}
        for tier, stems in per_tier:
            for s in stems:
                if s in known:
                    weights[s] = max(weights.get(s, 0.0), TIER_WEIGHT[tier])
        return weights, [s for tier, stems in per_tier if tier == "core" for s in stems if s in known]

    def _expanded(self, base: Dict[str, float], extra: Dict[str, float]) -> Dict[str, float]:
        """The extras share ALPHA x the total weight of `base` (len(base) when every word weighs 1.0), in proportion to their strength."""
        extra = {s: v for s, v in extra.items() if s not in base}
        total = sum(extra.values())
        return {**base, **{s: ALPHA * sum(base.values()) * v / total for s, v in extra.items()}}

    # --- expansions (also used to explain a result) ---------------------------
    def co_expansion(self, text: str, stems: Optional[List[str]] = None) -> List[dict]:
        stems = self.query_stems(text) if stems is None else stems
        if not stems:
            return []
        rows = _run(self.driver, _CO_NEIGHBOURS, {"stems": stems, "min_npmi": MIN_NPMI})
        return sorted(rows, key=lambda r: (-r["strength"], r["word"]))

    def term_expansion(self, text: str) -> List[dict]:
        """Nearest vocabulary terms to each query word by word embedding (works for words the corpus never contained)."""
        if not self.prep.has_vectors:
            return []
        words = sorted({w for _, w in tokens(text)})
        if not words:
            return []
        out: Dict[str, dict] = {}
        for word, vec in zip(words, embed(self.embedder, words)):
            rows = _run(self.driver, "MATCH (t:StoryTerm) SEARCH t IN (VECTOR INDEX term_vec FOR $vec LIMIT $k) SCORE AS score "
                                     "RETURN t.stem AS stem, t.surface AS word, score AS strength", {"k": TERM_K, "vec": vec})
            for r in rows:
                if r["strength"] >= TERM_MIN_COS and (r["stem"] not in out or r["strength"] > out[r["stem"]]["strength"]):
                    out[r["stem"]] = {**r, "via": [word]}
        return sorted(out.values(), key=lambda r: (-r["strength"], r["word"]))

    # --- arms ------------------------------------------------------------------
    def ft(self, query: Searchable, size: int = 10) -> List[Dict]:
        if isinstance(query, Query):   # Lucene's own boost syntax: word^0.33
            query = " ".join(f"{word}^{TIER_WEIGHT[tier]}" for tier, words in query.tiers() for _, word in tokens(" ".join(words)))
        else:
            query = " ".join(word for _, word in tokens(query))   # plain words only: nothing that Lucene's query syntax would read as an operator
        if not query:
            return []
        return _run(self.driver, f"MATCH (node:StoryVersion) SEARCH node IN (FULLTEXT INDEX story_ft FOR $q LIMIT $k) SCORE AS score RETURN {_COLUMNS}",
                    {"q": query, "k": size})

    def vec(self, query: Searchable, size: int = 10) -> List[Dict]:
        text = query.vector_text() if isinstance(query, Query) else query
        return _run(self.driver, f"MATCH (node:StoryVersion) SEARCH node IN (VECTOR INDEX story_vec FOR $vec LIMIT $k) SCORE AS score RETURN {_COLUMNS}",
                    {"k": size, "vec": embed(self.embedder, [text])[0]})

    def co(self, query: Searchable, size: int = 10) -> List[Dict]:
        base, subject = self._weights(query)
        extra = {r["stem"]: r["strength"] for r in self.co_expansion("", subject)}
        return self._bm25(self._expanded(base, extra) if extra else base, size)

    def term(self, query: Searchable, size: int = 10) -> List[Dict]:
        base, _ = self._weights(query)
        extra = {r["stem"]: r["strength"] for r in self.term_expansion(query.words(("core",)) if isinstance(query, Query) else query)}
        return self._bm25(self._expanded(base, extra) if extra and base else (extra or base), size)

    def arms(self, text: Searchable, size: int = 10, only: Optional[Sequence[str]] = None) -> Dict[str, List[Dict]]:
        """What each search arm returns for `text` (every arm this graph allows, or just those named in `only`; see retrieval.run_selected)."""
        builders: Dict[str, Callable] = {"n_ft": lambda get: self.ft(text, size), "n_co": lambda get: self.co(text, size)}
        if self.prep.has_vectors:
            builders["n_vec"] = lambda get: self.vec(text, size)
            builders["n_term"] = lambda get: self.term(text, size)
            builders["n_hyb"] = lambda get: rrf([get("n_co"), get("n_term"), get("n_vec")])
        else:
            builders["n_hyb"] = lambda get: rrf([get("n_co")])
        if self.targets == "all":
            builders["n_story"] = lambda get: story_level(get("n_hyb"))
        return run_selected(builders, only)
