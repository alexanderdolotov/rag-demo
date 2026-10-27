"""
Retrieval experiments on the dummy story corpus (see opensearch_stories.py). They need OpenSearch on
OPENSEARCH_URL (default http://localhost:9200) and skip themselves when it is not running; the vector tests
also skip without `sentence-transformers`. They use their own `stories_test` index and delete it afterwards.

    docker start opensearch-poc
    pytest test/test_opensearch_stories.py -v
"""
import pytest

import opensearch_stories as os_s

Q = {qid: (text, expect) for qid, text, expect in os_s.QUERIES}


@pytest.fixture(scope="module")
def embedder():
    return os_s.load_embedder()


@pytest.fixture(scope="module")
def index(embedder):
    if not os_s.reachable():
        pytest.skip(f"OpenSearch not reachable at {os_s.URL}")
    name = "stories_test"
    os_s.build_index(name, embedder)
    yield name
    os_s.drop_index(name)


def rank(index, arm, qid, kind=None, embedder=None):
    text, expect = Q[qid]
    return os_s.first_relevant_rank(os_s.run_arms(index, text, embedder, kind)[arm], expect)


# --------------------------------------------------------------------------- pure logic, no OpenSearch
def test_rrf_rewards_documents_that_several_searches_agree_on():
    a = [{"id": "x", "topic": "x"}, {"id": "y", "topic": "y"}]
    b = [{"id": "z", "topic": "z"}, {"id": "y", "topic": "y"}]
    fused = os_s.rrf([a, b])
    assert fused[0]["id"] == "y"                                      # in both lists
    assert fused[0]["score"] == pytest.approx(1 / 62 + 1 / 62)


def test_story_level_keeps_the_best_version_and_weights_rewrites_below_originals():
    fused = [{"id": "a:rw1", "topic": "a", "version": "rw1", "kind": "rewrite", "score": 0.030},
             {"id": "b:orig", "topic": "b", "version": "orig", "kind": "original", "score": 0.025},
             {"id": "a:orig", "topic": "a", "version": "orig", "kind": "original", "score": 0.010}]
    stories = os_s.story_level(fused)
    assert [s["topic"] for s in stories] == ["b", "a"]               # 0.025 beats 0.030 * 0.6
    assert stories[1]["via"] == "rw1" and len({s["topic"] for s in stories}) == len(stories)


def test_first_relevant_rank():
    hits = [{"topic": "a"}, {"topic": "b"}, {"topic": "c"}]
    assert os_s.first_relevant_rank(hits, ("c",)) == 3
    assert os_s.first_relevant_rank(hits, ("zzz",)) is None


# --------------------------------------------------------------------------- the index
def test_the_corpus_is_loaded(index):
    assert os_s.call("GET", f"/{index}/_count")["count"] == len(os_s.STORIES)


def test_the_analyzer_drops_stop_words_and_stems(index):
    assert os_s.analyze(index, "story_stem", "the toad hopped by a tree trunk") == ["toad", "hop", "tree", "trunk"]


def test_the_query_analyzer_adds_synonyms(index):
    duties = os_s.analyze(index, "story_syn", "import duties")
    assert {"tariff", "custom", "duti"} <= set(duties)
    assert {"toad", "amphibian", "hop", "leap"} <= set(os_s.analyze(index, "story_syn", "frog jumping"))


# --------------------------------------------------------------------------- keyword search and the vocabulary gap
def test_without_synonyms_a_query_in_other_words_finds_nothing_in_the_originals(index):
    """'toad hopped ... tree trunk' shares no stem with the original frog story."""
    assert rank(index, "stem", "q3", kind="original") is None


def test_synonyms_close_the_gap_on_the_originals(index):
    for qid in ("q2", "q3", "q8"):
        assert rank(index, "syn", qid, kind="original") == 1, qid


def test_rewrites_make_plain_keyword_search_work_too(index):
    """The user's rewrite idea: with rewrites in the index even plain BM25 finds the story."""
    originals = os_s.scoreboard(index, kind="original")["plain"]["mrr"]
    everything = os_s.scoreboard(index)["plain"]["mrr"]
    assert everything > originals + 0.2
    assert rank(index, "plain", "q3") == 1                            # via the toad rewrite


def test_polysemy_does_not_send_finance_queries_to_the_frogs(index):
    assert rank(index, "syn", "q5") == 1                              # 'jumped' also expands to hop/leap
    assert rank(index, "syn", "q6") == 1                              # river 'bank' ranks below the central bank


def test_a_story_level_view_returns_each_story_once(index):
    text, _ = Q["q1"]
    stories = os_s.run_arms(index, text)["story"]
    assert len({s["topic"] for s in stories}) == len(stories)
    assert stories[0]["topic"] == "frog_log"


# --------------------------------------------------------------------------- vectors (need sentence-transformers)
@pytest.fixture
def vectors(embedder):
    if embedder is None:
        pytest.skip("sentence-transformers not installed")
    return embedder


def test_vectors_find_the_story_with_no_synonym_list_and_no_rewrites(index, vectors):
    for qid in Q:
        assert rank(index, "dense", qid, kind="original", embedder=vectors) <= 3, qid
    for qid in ("q2", "q3", "q8"):                                   # the queries that share no stem with the original
        assert rank(index, "dense", qid, kind="original", embedder=vectors) == 1, qid


def test_vectors_tell_refused_to_jump_from_jumped_better_than_keywords(index, vectors):
    dense = rank(index, "dense", "q4", kind="original", embedder=vectors)
    keywords = rank(index, "syn", "q4", kind="original", embedder=vectors)
    assert dense == 1 and dense < keywords


def test_vectors_alone_are_at_least_as_good_as_the_hand_written_synonym_list(index, vectors):
    board = os_s.scoreboard(index, vectors, kind="original")
    assert board["dense"]["hit3"] == board["dense"]["n"]
    assert board["dense"]["mrr"] >= board["syn"]["mrr"]


def test_the_hybrid_and_story_views_still_answer_every_query(index, vectors):
    for arm in ("hybrid", "story"):
        board = os_s.scoreboard(index, vectors)
        assert board[arm]["hit3"] == board[arm]["n"], arm
