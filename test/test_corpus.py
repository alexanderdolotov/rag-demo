"""
The story files on disk: the loader's rules (tiny corpora built in a temp folder) and checks on the real stories corpus, so a
careless edit to a story file is caught before it silently changes what both databases hold. No services needed.
"""
import json
import re

import pytest

import corpus

STORIES = corpus.STORIES_DIR


def make(root, **stories):
    """make(tmp, frog={"original.txt": "...", "rewrite-1.txt": "..."})"""
    for name, files in stories.items():
        (root / name).mkdir()
        for filename, text in files.items():
            (root / name / filename).write_text(text)
    return root


def words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def queries_file(root, sets):
    (root / corpus.QUERIES_FILE).write_text(json.dumps({"sets": sets}))
    return root


def a_set(name="set 1", queries=None):
    return {"name": name, "description": "d", "queries": queries or [{"id": "q1", "text": "frog", "expect": ["frog"]}]}


# --------------------------------------------------------------------------- the rules
def test_a_story_folder_gives_the_original_first_then_its_numbered_rewrites(tmp_path):
    make(tmp_path, frog={"rewrite-2.txt": "toad two", "original.txt": "frog\n\nsecond paragraph\n", "rewrite-1.txt": "toad one"})
    docs = corpus.load_corpus(tmp_path)
    assert [(d["id"], d["version"], d["kind"]) for d in docs] == [
        ("frog:orig", "orig", "original"), ("frog:rw1", "rw1", "rewrite"), ("frog:rw2", "rw2", "rewrite")]
    assert docs[0]["text"] == "frog\n\nsecond paragraph" and docs[0]["topic"] == "frog"      # paragraphs kept, outer whitespace trimmed


def test_stories_come_back_in_a_stable_order(tmp_path):
    make(tmp_path, zebra={"original.txt": "z", "rewrite-1.txt": "z1"}, apple={"original.txt": "a", "rewrite-1.txt": "a1"})
    assert [d["topic"] for d in corpus.load_corpus(tmp_path)] == ["apple", "apple", "zebra", "zebra"]


@pytest.mark.parametrize("files, message", [
    ({"rewrite-1.txt": "x"}, "no original.txt"),
    ({"original.txt": "x"}, "no rewrites"),
    ({"original.txt": "x", "rewrite1.txt": "y"}, "expected original.txt or rewrite-<number>.txt"),
    ({"original.txt": "x", "notes.md": "y"}, "expected original.txt or rewrite-<number>.txt"),
    ({"original.txt": "x", "rewrite-1.txt": "  \n"}, "is empty"),
])
def test_a_story_folder_that_breaks_the_layout_is_refused_with_a_reason(tmp_path, files, message):
    make(tmp_path, frog=files)
    with pytest.raises(corpus.CorpusError, match=message):
        corpus.load_corpus(tmp_path)


def test_two_files_for_the_same_rewrite_number_are_refused(tmp_path):
    """rewrite-1.txt and rewrite-01.txt would both become version rw1: one document silently overwriting the other, or a crash half way through a load."""
    make(tmp_path, frog={"original.txt": "x", "rewrite-1.txt": "y", "rewrite-01.txt": "z"})
    with pytest.raises(corpus.CorpusError, match="rewrite-01.txt and rewrite-1.txt are both rewrite 1"):
        corpus.load_corpus(tmp_path)


def test_bad_folder_names_and_empty_roots_are_refused(tmp_path):
    with pytest.raises(corpus.CorpusError, match="no story folders"):
        corpus.load_corpus(tmp_path)
    make(tmp_path, **{"Bad-Name": {"original.txt": "x", "rewrite-1.txt": "y"}})
    with pytest.raises(corpus.CorpusError, match="lowercase letters"):
        corpus.load_corpus(tmp_path)
    with pytest.raises(corpus.CorpusError, match="is not a folder"):
        corpus.load_corpus(tmp_path / "missing")


def test_the_fingerprint_changes_with_any_text_and_ignores_order(tmp_path):
    make(tmp_path, a={"original.txt": "one", "rewrite-1.txt": "two"})
    docs = corpus.load_corpus(tmp_path)
    assert corpus.corpus_hash(docs) == corpus.corpus_hash(list(reversed(docs)))
    (tmp_path / "a" / "rewrite-1.txt").write_text("two!")
    assert corpus.corpus_hash(corpus.load_corpus(tmp_path)) != corpus.corpus_hash(docs)


def test_originals_and_stats(tmp_path):
    make(tmp_path, a={"original.txt": "one two three", "rewrite-1.txt": "four five"}, b={"original.txt": "six", "rewrite-1.txt": "seven eight nine ten"})
    docs = corpus.load_corpus(tmp_path)
    assert corpus.originals(docs) == {"a": "one two three", "b": "six"}
    stats = corpus.stats(docs)
    assert (stats["stories"], stats["versions"], stats["min_words"], stats["max_words"], stats["total_words"]) == (2, 4, 1, 4, 10)


def test_versions_longer_than_the_embedding_window_are_reported():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": text.split()}

    class Model:
        max_seq_length, tokenizer = 3, Tokenizer()

    docs = [{"id": "a:orig", "text": "one two three"}, {"id": "a:rw1", "text": "one two three four"}]
    assert corpus.over_window(docs, Model()) == ["a:rw1"] and corpus.over_window(docs, None) is None


# --------------------------------------------------------------------------- the test queries
def test_query_sets_come_back_in_file_order_as_id_text_expected_stories(tmp_path):
    queries_file(tmp_path, [a_set("b", [{"id": "q2", "text": "toad", "expect": ["frog", "pond"]}]), a_set("a")])
    sets = corpus.load_queries(tmp_path)
    assert [s.name for s in sets] == ["b", "a"] and sets[0].description == "d"
    assert sets[0].queries == [("q2", "toad", ("frog", "pond"))] and sets[1].queries == [("q1", "frog", ("frog",))]


@pytest.mark.parametrize("content, message", [
    ("not json", "not valid JSON"),
    ('{"sets": []}', "at least one set"),
    ("[]", "at least one set"),
    ('{"sets": [{"description": "d", "queries": [{"id": "q", "text": "t", "expect": ["a"]}]}]}', 'needs a "name"'),
    ('{"sets": [{"name": "s", "queries": [{"id": "q", "text": "t", "expect": ["a"]}]}]}', 'needs a "description"'),
    ('{"sets": [{"name": "s", "description": "d", "queries": []}]}', "non-empty"),
    ('{"sets": [{"name": "s", "description": "d", "queries": [{"id": "q", "text": "t", "expect": []}]}]}', "non-empty .expect. list"),
    ('{"sets": [{"name": "s", "description": "d", "queries": [{"id": "q", "text": " ", "expect": ["a"]}]}]}', 'string "text"'),
    ('{"sets": [{"name": "s", "description": "d", "queries": [{"id": "q", "text": "t", "expect": "a"}]}]}', "non-empty .expect. list"),
])
def test_a_queries_file_that_breaks_the_layout_is_refused_with_a_reason(tmp_path, content, message):
    (tmp_path / corpus.QUERIES_FILE).write_text(content)
    with pytest.raises(corpus.CorpusError, match=message):
        corpus.load_queries(tmp_path)


def test_a_missing_file_and_a_reused_query_id_are_refused(tmp_path):
    with pytest.raises(corpus.CorpusError, match="does not exist"):
        corpus.load_queries(tmp_path)
    queries_file(tmp_path, [a_set("one"), a_set("two")])              # both sets use the id q1
    with pytest.raises(corpus.CorpusError, match="'q1' is used twice"):
        corpus.load_queries(tmp_path)


def test_with_the_corpus_given_a_query_for_a_story_that_does_not_exist_is_refused(tmp_path):
    make(tmp_path, frog={"original.txt": "x", "rewrite-1.txt": "y"})
    queries_file(tmp_path, [a_set(queries=[{"id": "q1", "text": "frog", "expect": ["frog", "toad_log"]}])])
    docs = corpus.load_corpus(tmp_path)
    assert corpus.load_queries(tmp_path)[0].queries[0][2] == ("frog", "toad_log")      # without docs the ids are not checked
    with pytest.raises(corpus.CorpusError, match=r"toad_log.*not stories"):
        corpus.load_queries(tmp_path, docs)


def test_a_queries_file_next_to_the_stories_is_not_mistaken_for_a_story(tmp_path):
    make(tmp_path, frog={"original.txt": "x", "rewrite-1.txt": "y"})
    queries_file(tmp_path, [a_set()])
    assert [d["topic"] for d in corpus.load_corpus(tmp_path)] == ["frog", "frog"]


# --------------------------------------------------------------------------- the real stories
@pytest.fixture(scope="module")
def docs():
    return corpus.load_corpus(STORIES)


def test_there_are_20_stories_each_with_an_original_and_five_rewrites(docs):
    by_story = {}
    for d in docs:
        by_story.setdefault(d["topic"], []).append(d["version"])
    assert len(by_story) == 20 and len(docs) == 120
    assert all(versions == ["orig", "rw1", "rw2", "rw3", "rw4", "rw5"] for versions in by_story.values()), by_story


def test_the_versions_are_long_enough_to_matter_and_short_enough_for_the_embedding_window(docs):
    for d in docs:
        n = len(d["text"].split())
        assert 40 <= n <= 330, (d["id"], n)                    # 330 words is about 430 tokens, under the 512 the model is set to read
        if d["kind"] == "original":
            assert n >= 150, (d["id"], n)
    summaries = [len(d["text"].split()) for d in docs if d["version"] == "rw4"]
    assert max(summaries) < 120                                # rewrite-4 is the short summary of each story


def test_every_rewrite_really_is_in_different_words(docs):
    original = {d["topic"]: words(d["text"]) for d in docs if d["kind"] == "original"}
    assert len({d["text"] for d in docs}) == len(docs)         # no two versions are the same text
    for d in docs:
        if d["kind"] == "rewrite":
            a, b = original[d["topic"]], words(d["text"])
            assert len(a & b) / len(a | b) < 0.8, d["id"]


def test_the_real_queries_load_and_every_story_they_expect_exists(docs):
    sets = corpus.load_queries(STORIES, docs)                  # raises if a query expects a story that is not there
    assert sets and all(s.queries for s in sets)


GAP_WORDS = {   # words the rewrites use and the original deliberately does not: they are what makes the experiment a vocabulary-gap test
    "frog_log": ["toad", "hopped", "trunk", "amphibian", "timber"], "tariff_margin": ["duties", "levies", "customs", "recovered"],
    "tariff_hurt": ["duties", "levies", "expenses"], "bank_rates": ["monetary", "benchmark", "lender", "borrowing"],
    "stock_jump": ["surged"], "mountain_hike": ["walking", "cabin", "hill"], "lost_dog": ["missing", "puppy"],
    "cyber_incident": ["hackers", "breach", "cyberattack", "attack"], "ceo_departure": ["step", "replace"],
}


def test_the_originals_keep_the_vocabulary_gap_the_experiments_rely_on(docs):
    original = {d["topic"]: words(d["text"]) for d in docs if d["kind"] == "original"}
    for story, gap in GAP_WORDS.items():
        assert not [w for w in gap if w in original[story]], story


def test_the_look_alike_stories_are_really_look_alikes(docs):
    """The traps: 'duties' (a night watchman's job) and 'customs' (village traditions) must not be about trade."""
    text = {d["topic"]: d["text"].lower() for d in docs if d["kind"] == "original"}
    assert "duties" in text["job_duties"] and "tariff" not in text["job_duties"]
    assert "customs" in text["village_customs"] and "tariff" not in text["village_customs"]
    assert "freight" in text["supply_chain"] and "tariff" not in text["supply_chain"] and "duties" not in text["supply_chain"]
