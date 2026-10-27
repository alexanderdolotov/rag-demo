"""
Graph database (Neo4j) against search engine (OpenSearch) on the dummy story corpus - the findings of
compare_stories.py as tests. They need both services and skip themselves when one is not running; the vector
tests also skip without `sentence-transformers`. They use their own `stories_cmp_test` index in OpenSearch and the
Story / StoryVersion / StoryTerm nodes in Neo4j (rewritten from scratch by every scenario).

    docker start opensearch-poc neo4j-poc
    pytest test/test_neo4j_stories.py -v
"""
import pytest

pytest.importorskip("neo4j")
pytest.importorskip("nltk")

import compare_stories as cs  # noqa: E402
import neo4j_stories as ns  # noqa: E402
import opensearch_stories as os_s  # noqa: E402

INDEX = "stories_cmp_test"
SET1 = [q for q, _, _ in os_s.QUERIES]
SET2 = [q for q, _, _ in os_s.HELDOUT]
ALL = SET1 + SET2
TEXT = {q: t for q, t, _ in cs.all_queries()}
ORIGINALS, THESAURUS, EVERYTHING = (name for name, _, _ in cs.SCENARIOS)


# --------------------------------------------------------------------------- pure logic, no services
def test_tokens_drop_stop_words_and_stem():
    assert [s for s, _ in ns.tokens("the toad hopped by a tree trunk")] == ["toad", "hop", "tree", "trunk"]


def test_words_that_only_meet_through_rewrites_get_a_co_occurrence_edge():
    prep = ns.prepare(None)
    edge = lambda docs: {(e["a"], e["b"]): e["npmi"] for e in ns._co_edges(docs)}  # noqa: E731
    learned = edge(prep.docs)
    assert learned[("frog", "toad")] == 1.0                         # in the same three stories, never in the same sentence
    assert ("frog", "toad") not in edge([d for d in prep.docs if d["kind"] == "original"])   # 'toad' is not even a term there


def test_the_expansion_never_outweighs_the_original_query():
    weights = ns.StoryGraph._expanded(None, {"frog": 1.0, "jump": 1.0}, {"toad": 1.0, "hop": 0.5, "frog": 0.9})
    assert weights["frog"] == weights["jump"] == 1.0
    assert weights["toad"] + weights["hop"] == pytest.approx(ns.ALPHA * 2)
    assert weights["toad"] == pytest.approx(2 * weights["hop"])


# --------------------------------------------------------------------------- services
@pytest.fixture(scope="module")
def embedder():
    return os_s.load_embedder()


@pytest.fixture(scope="module")
def vectors(embedder):
    if embedder is None:
        pytest.skip("sentence-transformers not installed")
    return embedder


@pytest.fixture(scope="module")
def results(embedder):
    """{scenario: {arm: {query id: rank}}} plus the terms the graph added to each query in the thesaurus scenario."""
    if not os_s.reachable():
        pytest.skip(f"OpenSearch not reachable at {os_s.URL}")
    if not ns.reachable():
        pytest.skip(f"Neo4j not reachable at {ns.URI}")
    os_s.build_index(INDEX, embedder)
    prep = ns.prepare(embedder)
    driver = ns.connect()
    out = {}
    try:
        for name, targets, learn_from in cs.SCENARIOS:
            ranks, graph = cs.run_scenario(targets, learn_from, prep, driver, INDEX, embedder)
            out[name] = ranks
            if name == THESAURUS:
                out["co_words"] = {q: [r["word"] for r in graph.co_expansion(t)] for q, t in TEXT.items()}
                out["term_words"] = {q: [(r["word"], r["via"][0]) for r in graph.term_expansion(t)] for q, t in TEXT.items()}
    finally:
        driver.close()
        os_s.drop_index(INDEX)
    return out


def mrr(results, scenario, arm, ids=ALL):
    return cs.summarize(results[scenario][arm], ids)[1]


def hits(results, scenario, arm, ids=ALL):
    return cs.summarize(results[scenario][arm], ids)[0]


def test_the_graph_has_the_expected_shape(results, embedder):
    prep = ns.prepare(embedder)
    driver = ns.connect()
    try:
        graph = ns.build_graph(driver, prep, "original", "all", embedder)
        one = lambda cypher: ns._run(driver, cypher)[0]["n"]  # noqa: E731
        assert one("MATCH (:StoryVersion)-[:OF]->(:Story) RETURN count(*) AS n") == 11          # retrievable: the originals
        assert one("MATCH (v:StoryVersion) WHERE v.kind <> 'original' RETURN count(v) AS n") == 0
        assert one("MATCH (:StoryTerm)-[c:CO]->(:StoryTerm) RETURN count(c) AS n") == graph.counts["co_edges"] > 1000
        assert one("MATCH (t:StoryTerm {stem: 'toad'}) RETURN count(t) AS n") == 1           # a term only the rewrites contain
        assert one("MATCH (:StoryVersion)-[:HAS]->(:StoryTerm {stem: 'toad'}) RETURN count(*) AS n") == 0   # ... no original has it
    finally:
        driver.close()


# ---- the database does not change what a plain index finds
def test_neo4j_vector_index_ranks_exactly_like_opensearch_knn(results, vectors):
    for scenario in (ORIGINALS, EVERYTHING):
        assert results[scenario]["n_vec"] == results[scenario]["dense"], scenario


def test_neo4j_fulltext_index_is_in_the_same_league_as_opensearch_stemmed_bm25(results):
    for scenario in (ORIGINALS, EVERYTHING):
        assert abs(mrr(results, scenario, "n_ft") - mrr(results, scenario, "stem")) < 0.1, scenario


# ---- what the co-occurrence graph can and cannot do
def test_without_rewrites_the_graph_cannot_bridge_vocabulary(results):
    """Learned from the originals alone it is as blind as stemmed keyword search."""
    for qid in ("q3", "h9"):
        assert results[ORIGINALS]["n_co"][qid] is None and results[ORIGINALS]["stem"][qid] is None, qid


def test_rewrites_used_as_a_thesaurus_close_the_gap_for_words_the_corpus_has_seen(results):
    assert results[THESAURUS]["n_co"]["q3"] == 1                     # 'toad hopped by a tree trunk' now finds the frog original
    assert hits(results, THESAURUS, "n_co", SET1) == len(SET1) > hits(results, ORIGINALS, "n_ft", SET1)


def test_the_graph_explains_itself(results):
    assert "toad" in results["co_words"]["q1"]                       # 'frog jump' was widened with 'toad'
    assert "frog" in results["co_words"]["h11"]                      # 'toad resting ... stream' with 'frog'


def test_cooccurrence_cannot_help_with_words_the_corpus_never_contained(results):
    """h9 shares no word with any version of the mountain story, so there is no edge to follow."""
    assert results["co_words"]["h9"] == []
    assert results[THESAURUS]["n_co"]["h9"] is None


def test_word_embeddings_reach_words_the_corpus_never_contained(results, vectors):
    assert ("hiked", "walking") in results["term_words"]["h9"]
    assert 1 <= (results[ORIGINALS]["n_term"]["h9"] or 99) <= 3


# ---- the comparison
def test_the_learned_thesaurus_is_worse_than_sentence_vectors(results, vectors):
    assert mrr(results, THESAURUS, "n_co") < mrr(results, THESAURUS, "n_vec") - 0.1
    assert hits(results, THESAURUS, "n_vec") == len(ALL)


def test_fusing_the_graph_arms_does_not_beat_vectors_alone(results, vectors):
    for scenario in (ORIGINALS, THESAURUS, EVERYTHING):
        assert mrr(results, scenario, "n_hyb") <= mrr(results, scenario, "n_vec"), scenario


def test_the_story_view_exists_only_when_rewrites_are_retrievable(results):
    assert "n_story" in results[EVERYTHING] and "n_story" not in results[ORIGINALS]
