"""
Graph database (Neo4j) against search engine (OpenSearch) on the story corpus in stories/ - the findings of
test/compare_stories.py as tests. They need both services and skip themselves when one is not running; the vector
tests also skip without `sentence-transformers`. They use their own `stories_cmp_test` index in OpenSearch and a
`Test`-prefixed graph in Neo4j (conftest.py sets the prefix; every scenario rebuilds it from scratch), so the corpus
loaded by data_load.py is never touched.

    docker start opensearch-poc neo4j-poc
    pytest test/test_neo4j_stories.py -v
"""
import pytest

pytest.importorskip("neo4j")
pytest.importorskip("nltk")

import compare_stories as cs  # noqa: E402
import corpus  # noqa: E402
import neo4j_stories as ns  # noqa: E402
import opensearch_stories as os_s  # noqa: E402

INDEX = "stories_cmp_test"
DOCS = corpus.load_corpus()
SETS = corpus.load_queries(docs=DOCS)
QUERIES = cs.all_queries(SETS[:2])             # sets 1 and 2: the findings below are stated over these
SET1 = [q for q, _, _ in SETS[0].queries]
SET2 = [q for q, _, _ in SETS[1].queries]
ALL = SET1 + SET2
TEXT = {q: t for q, t, _ in QUERIES}
UNSEEN = "zxqv blorf"                           # words that no version of any story contains
ORIGINALS, THESAURUS, EVERYTHING = (name for name, _, _ in cs.SCENARIOS)


# --------------------------------------------------------------------------- pure logic, no services
def test_tokens_drop_stop_words_and_stem():
    assert [s for s, _ in ns.tokens("the toad hopped by a tree trunk")] == ["toad", "hop", "tree", "trunk"]


def test_tokens_treat_a_possessive_as_the_plain_word_and_keep_accented_words_whole():
    """"watchman's" used to give "watchman" plus a stray term "s"; "naïve" used to be cut into "na" and "ve"."""
    assert [w for _, w in ns.tokens("the night watchman's lantern")] == ["night", "watchman", "lantern"]
    assert [w for _, w in ns.tokens("Naïve café\u2019s résumé, don't")] == ["naïve", "café", "résumé", "don't"]
    assert ns.tokens("it's") == []                                  # "it" is a stop word


def test_words_that_only_meet_through_rewrites_get_a_co_occurrence_edge():
    prep = ns.prepare(None, DOCS)
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
    os_s.build_index(INDEX, embedder, DOCS)
    prep = ns.prepare(embedder, DOCS)
    driver = ns.connect()
    out = {}
    try:
        for name, targets, learn_from in cs.SCENARIOS:
            ranks, graph = cs.run_scenario(targets, learn_from, prep, driver, INDEX, embedder, QUERIES)
            out[name] = ranks
            if name == THESAURUS:
                out["co_words"] = {q: [r["word"] for r in graph.co_expansion(t)] for q, t in {**TEXT, "unseen": UNSEEN}.items()}
                out["term_words"] = {q: [(r["word"], r["via"][0]) for r in graph.term_expansion(t)] for q, t in TEXT.items()}
    finally:
        driver.close()
        os_s.drop_index(INDEX)
    return out


TINY = [{"id": f"{topic}:{version}", "topic": topic, "version": version, "kind": "original" if version == "orig" else "rewrite", "text": text}
        for topic, version, text in [
            ("a", "orig", "The night watchman's lantern lit a naïve frog."), ("a", "rw1", "A guard carried a light past an amphibian."),
            ("b", "orig", "She baked bread."), ("b", "rw1", "Loaves came out of the oven.")]]


@pytest.fixture
def tiny_graph():
    """Four texts, no vectors, built just before a test (any other graph in the Test namespace is replaced)."""
    if not ns.reachable():
        pytest.skip(f"Neo4j not reachable at {ns.URI}")
    driver = ns.connect()
    try:
        yield ns.build_graph(driver, ns.prepare(None, TINY), "all", "all")
    finally:
        driver.close()


def test_the_full_text_arm_finds_a_possessive_and_an_accented_word_by_their_plain_forms(tiny_graph):
    topics = lambda query: [r["topic"] for r in tiny_graph.ft(query, 5)]  # noqa: E731
    assert topics("watchman") == topics("watchman's") == ["a"]
    assert topics("naïve") == ["a"] and topics("loaves") == ["b"]
    assert tiny_graph.ft("the of and", 5) == []                      # only stop words: nothing is sent to Neo4j


def test_asking_for_one_graph_arm_returns_what_running_all_of_them_returns(tiny_graph):
    everything = tiny_graph.arms("frog lantern", 5)
    assert list(everything) == ["n_ft", "n_co", "n_hyb", "n_story"]   # no vectors, so no n_vec / n_term
    for arm in everything:
        assert tiny_graph.arms("frog lantern", 5, only=[arm]) == {arm: everything[arm]}, arm
    with pytest.raises(ValueError, match="no such search arm: n_vec"):
        tiny_graph.arms("frog", only=["n_vec"])


def mrr(results, scenario, arm, ids=ALL):
    return cs.summarize(results[scenario][arm], ids)[1]


def hits(results, scenario, arm, ids=ALL):
    return cs.summarize(results[scenario][arm], ids)[0]


def found(rank, top=3):
    return rank is not None and rank <= top


def test_the_graph_has_the_expected_shape(results, embedder):
    prep = ns.prepare(embedder, DOCS)
    driver = ns.connect()
    try:
        graph = ns.build_graph(driver, prep, "original", "all", embedder)
        one = lambda cypher: ns._run(driver, cypher)[0]["n"]  # noqa: E731
        assert one("MATCH (:StoryVersion)-[:OF]->(:Story) RETURN count(*) AS n") == len({d["topic"] for d in DOCS})   # retrievable: the originals
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
    """Learned from the originals alone it is as blind as stemmed keyword search: neither gets these into the top 3."""
    for qid in ("q3", "h9"):
        assert not found(results[ORIGINALS]["n_co"][qid]) and not found(results[ORIGINALS]["stem"][qid]), qid


def test_rewrites_used_as_a_thesaurus_help_the_graph_on_the_harder_queries(results):
    assert hits(results, THESAURUS, "n_co", SET2) > hits(results, ORIGINALS, "n_co", SET2)
    assert mrr(results, THESAURUS, "n_co", SET2) > mrr(results, ORIGINALS, "n_co", SET2)


def test_the_graph_explains_itself(results):
    assert "toad" in results["co_words"]["q1"]                       # 'frog jump' was widened with 'toad'
    assert "frog" in results["co_words"]["h11"]                      # 'toad resting ... stream' with 'frog'


def test_cooccurrence_cannot_help_with_words_the_corpus_never_contained(results):
    """A query of words that no version contains has no edge to follow, so nothing is added to it."""
    assert results["co_words"]["unseen"] == []


def test_word_embeddings_reach_words_the_corpus_never_contained(results, vectors):
    assert ("hiked", "walking") in results["term_words"]["h9"]
    assert results[ORIGINALS]["n_ft"]["h9"] is None and results[ORIGINALS]["n_term"]["h9"] is not None   # keywords find nothing, the word vectors do


# ---- the comparison
def test_the_learned_thesaurus_is_worse_than_sentence_vectors(results, vectors):
    assert mrr(results, THESAURUS, "n_co") < mrr(results, THESAURUS, "n_vec") - 0.1
    assert hits(results, THESAURUS, "n_vec") == len(ALL)


def test_fusing_the_graph_arms_does_not_beat_vectors_alone(results, vectors):
    """When only originals can be returned. With the rewrites retrievable too the fused arm is ahead on this corpus, so no claim is made there."""
    for scenario in (ORIGINALS, THESAURUS):
        assert mrr(results, scenario, "n_hyb") <= mrr(results, scenario, "n_vec"), scenario


def test_the_story_view_exists_only_when_rewrites_are_retrievable(results):
    assert "n_story" in results[EVERYTHING] and "n_story" not in results[ORIGINALS]
