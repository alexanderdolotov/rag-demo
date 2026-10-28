"""
Retrieval experiments on the story corpus in stories/ and its first query set (see opensearch_stories.py). They need OpenSearch on
OPENSEARCH_URL (default http://localhost:9200) and skip themselves when it is not running; the vector tests
also skip without `sentence-transformers`. They use their own `stories_test` index and delete it afterwards.

    docker start opensearch-poc
    pytest test/test_opensearch_stories.py -v
"""
import pytest

import corpus
import opensearch_stories as os_s
import retrieval

DOCS = corpus.load_corpus()
SETS = corpus.load_queries(docs=DOCS)
QUERIES, HARDER = SETS[0].queries, SETS[1].queries      # set 1 and the harder set 2
Q = {qid: (text, expect) for qid, text, expect in QUERIES}


@pytest.fixture(scope="module")
def opensearch():
    if not os_s.reachable():
        pytest.skip(f"OpenSearch not reachable at {os_s.URL}")


@pytest.fixture(scope="module")
def index(embedder, opensearch):
    name = "stories_test"
    os_s.build_index(name, embedder, DOCS)
    yield name
    os_s.drop_index(name)


def rank(index, arm, qid, kind=None, embedder=None):
    text, expect = Q[qid]
    return retrieval.first_relevant_rank(os_s.run_arms(index, text, embedder, kind)[arm], expect)


# --------------------------------------------------------------------------- index names (no OpenSearch needed)
@pytest.mark.parametrize("name", ["stories", "stories_smoke", "stories-2", "a1"])
def test_ordinary_index_names_are_accepted(name):
    assert os_s.index_name(name) == name


@pytest.mark.parametrize("name", ["*", "stories*", "_all", "a,b", ".plugins-ml-config", "_hidden", "-x", "Stories", "a b", "../x", "a/b", ""])
def test_a_name_that_could_reach_more_than_one_index_is_refused_before_any_request(name, monkeypatch):
    """--index goes straight into DELETE, and this OpenSearch deletes by wildcard unless told otherwise."""
    def no_request(*args, **kwargs):
        raise AssertionError("a request was sent")
    monkeypatch.setattr(os_s.requests, "delete", no_request)
    monkeypatch.setattr(os_s.requests, "get", no_request)
    monkeypatch.setattr(os_s.requests, "request", no_request)
    with pytest.raises(ValueError, match="plain index name"):
        os_s.index_name(name)
    for action in (lambda: os_s.drop_index(name), lambda: os_s.index_state(name), lambda: os_s.build_index(name, None, DOCS)):
        with pytest.raises(ValueError):
            action()


def test_run_arms_refuses_an_arm_it_does_not_have():
    with pytest.raises(ValueError, match="no such search arm: dense"):
        os_s.run_arms("stories_test", "frog", None, only=["dense"])        # no embedder, so no vector arm (and no request is sent)


# --------------------------------------------------------------------------- the index
def test_the_corpus_is_loaded(index):
    assert os_s.call("GET", f"/{index}/_count")["count"] == len(DOCS)


def test_the_fingerprint_is_stored_with_the_finished_index(opensearch):
    try:
        os_s.build_index("stories_meta_test", None, DOCS)
        assert os_s.index_state("stories_meta_test") == {"docs": len(DOCS), "meta": {}}
        os_s.build_index("stories_meta_test", None, DOCS, {"corpus_hash": "abc"})
        assert os_s.index_state("stories_meta_test") == {"docs": len(DOCS), "meta": {"corpus_hash": "abc"}}
    finally:
        os_s.drop_index("stories_meta_test")


def test_a_load_that_stops_half_way_leaves_an_index_without_a_fingerprint(opensearch, monkeypatch):
    real_call = os_s.call

    def dies_on_bulk(method, path, body=None, ndjson=None):
        if path.startswith("/_bulk"):
            raise RuntimeError("the load stopped")
        return real_call(method, path, body, ndjson)
    monkeypatch.setattr(os_s, "call", dies_on_bulk)
    try:
        with pytest.raises(RuntimeError, match="stopped"):
            os_s.build_index("stories_halfway_test", None, DOCS, {"corpus_hash": "abc"})
        assert os_s.index_state("stories_halfway_test") == {"docs": 0, "meta": {}}
    finally:
        os_s.drop_index("stories_halfway_test")


def test_the_analyzer_drops_stop_words_and_stems(index):
    assert os_s.analyze(index, "story_stem", "the toad hopped by a tree trunk") == ["toad", "hop", "tree", "trunk"]


def test_a_possessive_is_the_same_word_as_the_plain_noun(index):
    """Lucene's English analyzer (Neo4j's full-text index) drops 's; without that "watchman's" is a different term from "watchman"."""
    for analyzer in ("story_stem", "story_syn"):
        assert os_s.analyze(index, analyzer, "the night watchman's lantern") == os_s.analyze(index, analyzer, "the night watchman lantern")
    assert {h["id"] for h in os_s.bm25(index, "text.stem", "watchman")} == {h["id"] for h in os_s.bm25(index, "text.stem", "watchman's")}


def test_asking_for_one_arm_returns_what_running_all_of_them_returns(index, embedder):
    text, _ = Q["q7"]
    everything = os_s.run_arms(index, text, embedder)
    for arm in everything:
        assert os_s.run_arms(index, text, embedder, only=[arm]) == {arm: everything[arm]}, arm


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
    originals = os_s.scoreboard(index, QUERIES, kind="original")["plain"]["mrr"]
    everything = os_s.scoreboard(index, QUERIES)["plain"]["mrr"]
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
    for qid in Q:                                                    # including q2, q3, q8, which share no stem with the original
        assert rank(index, "dense", qid, kind="original", embedder=vectors) <= 3, qid


def test_vectors_tell_refused_to_jump_from_jumped_better_than_keywords(index, vectors):
    dense = rank(index, "dense", "q4", kind="original", embedder=vectors)
    keywords = rank(index, "syn", "q4", kind="original", embedder=vectors)
    assert dense == 1 and dense < keywords


def test_vectors_alone_are_at_least_as_good_as_the_hand_written_synonym_list(index, vectors):
    """On set 1 the synonym list ranks higher (it was written knowing those queries), so the comparison is made on the harder set 2."""
    assert os_s.scoreboard(index, QUERIES, vectors, kind="original")["dense"]["hit3"] == len(QUERIES)
    board = os_s.scoreboard(index, HARDER, vectors, kind="original")
    assert board["dense"]["hit3"] == board["dense"]["n"]
    assert board["dense"]["mrr"] >= board["syn"]["mrr"]


def test_the_hybrid_and_story_views_still_answer_every_query(index, vectors):
    for arm in ("hybrid", "story"):
        board = os_s.scoreboard(index, QUERIES, vectors)
        assert board[arm]["hit3"] == board[arm]["n"], arm
