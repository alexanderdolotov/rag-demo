"""
data_load.py against real OpenSearch + Neo4j, with a tiny three-story corpus in a temp folder (skipped when a service or
sentence-transformers is missing). The test index is `stories_load_test` and Neo4j uses its own "Test" namespace (see conftest.py),
so the corpus you loaded for real is never touched.

    docker start opensearch-poc neo4j-poc
    pytest test/test_data_load.py -v
"""
import pytest

pytest.importorskip("neo4j")
pytest.importorskip("nltk")

import corpus  # noqa: E402
import data_load  # noqa: E402
import neo4j_stories as ns  # noqa: E402
import opensearch_stories as os_s  # noqa: E402

INDEX = "stories_load_test"
FOREIGN_INDEX = "stories_foreign_test"

STORIES = {
    "frogs": ("A frog sat on a log by the pond and croaked at the sun.", ["A toad rested on a fallen trunk by the marsh.", "An amphibian called from driftwood."]),
    "bread": ("She kneaded the dough and baked two golden loaves.", ["After proofing the mixture she baked a pair of brown loaves.", "Flour and yeast became crusty bread."]),
    "rocket": ("The rocket reached orbit nine minutes after lifting off.", ["The launch vehicle arrived in space in under ten minutes.", "A booster delivered a satellite quickly."]),
}


@pytest.fixture(scope="module")
def embedder(embedder):          # the shared model (conftest.py), and this module needs both services as well
    if not os_s.reachable():
        pytest.skip(f"OpenSearch not reachable at {os_s.URL}")
    if not ns.reachable():
        pytest.skip(f"Neo4j not reachable at {ns.URI}")
    if embedder is None:
        pytest.skip("sentence-transformers not installed")
    return embedder


@pytest.fixture
def driver(embedder):
    driver = ns.connect()
    data_load.truncate(INDEX, driver)                 # start every test from nothing
    try:
        yield driver
    finally:
        data_load.truncate(INDEX, driver)
        os_s.drop_index(FOREIGN_INDEX)
        driver.close()


@pytest.fixture
def story_dir(tmp_path):
    for name, (original, rewrites) in STORIES.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "original.txt").write_text(original)
        for i, text in enumerate(rewrites, start=1):
            (tmp_path / name / f"rewrite-{i}.txt").write_text(text)
    return tmp_path


def docs_of(story_dir):
    return corpus.load_corpus(story_dir)


def test_a_first_load_puts_the_same_files_in_both_databases(driver, embedder, story_dir):
    docs = docs_of(story_dir)
    assert data_load.problems(docs, INDEX, driver)                                # nothing loaded yet: both reported
    result = data_load.run(docs, INDEX, driver, embedder)
    assert result["loaded"] == ["opensearch", "neo4j"] and result["removed"] is None
    assert data_load.problems(docs, INDEX, driver) == []
    assert os_s.index_state(INDEX)["docs"] == 9 and ns.read_meta(driver)["versions"] == 9
    assert os_s.index_state(INDEX)["meta"]["corpus_hash"] == ns.read_meta(driver)["corpus_hash"] == corpus.corpus_hash(docs)


def test_both_databases_return_every_version_with_the_original_marked(driver, embedder, story_dir):
    data_load.run(docs_of(story_dir), INDEX, driver, embedder)
    hits = os_s.call("POST", f"/{INDEX}/_search", {"size": 20, "query": {"match_all": {}}, "_source": ["topic", "version", "kind"]})["hits"]["hits"]
    in_os = sorted((h["_source"]["topic"], h["_source"]["version"], h["_source"]["kind"]) for h in hits)
    in_neo = sorted((r["topic"], r["version"], r["kind"]) for r in ns._run(driver, "MATCH (v:StoryVersion) RETURN v.topic AS topic, v.version AS version, v.kind AS kind"))
    assert in_os == in_neo and len(in_os) == 9
    assert sorted(k for _, v, k in in_os if v == "orig") == ["original"] * 3 and all(k == "rewrite" for _, v, k in in_os if v != "orig")


def test_running_it_again_does_nothing(driver, embedder, story_dir):
    docs = docs_of(story_dir)
    data_load.run(docs, INDEX, driver, embedder)
    loaded_at = ns.read_meta(driver)["loaded_at"]
    again = data_load.run(docs, INDEX, driver, embedder)
    assert again["loaded"] == [] and ns.read_meta(driver)["loaded_at"] == loaded_at


def test_a_changed_file_is_refused_until_reload_is_asked_for(driver, embedder, story_dir):
    data_load.run(docs_of(story_dir), INDEX, driver, embedder)
    (story_dir / "bread" / "rewrite-1.txt").write_text("Different words about baking a loaf.")
    changed = docs_of(story_dir)
    assert len(data_load.problems(changed, INDEX, driver)) == 2                   # both databases are now out of date
    with pytest.raises(RuntimeError, match="different corpus"):
        data_load.run(changed, INDEX, driver, embedder)
    assert os_s.index_state(INDEX)["docs"] == 9                                   # the refusal changed nothing
    nodes = ns._run(driver, "MATCH (n) WHERE n:Story OR n:StoryVersion OR n:StoryTerm OR n:StoryCorpus RETURN count(n) AS n")[0]["n"]
    result = data_load.run(changed, INDEX, driver, embedder, reload=True)
    assert result["removed"] == {"opensearch_docs": 9, "neo4j_nodes": nodes} and nodes > 9
    assert data_load.problems(changed, INDEX, driver) == []


def test_an_index_whose_load_stopped_half_way_is_reloaded_and_never_trusted(driver, embedder, story_dir, monkeypatch):
    docs = docs_of(story_dir)
    real_call = os_s.call

    def dies_on_bulk(method, path, body=None, ndjson=None):
        if path.startswith("/_bulk"):
            raise RuntimeError("the load stopped")
        return real_call(method, path, body, ndjson)
    monkeypatch.setattr(os_s, "call", dies_on_bulk)
    with pytest.raises(RuntimeError, match="stopped"):
        data_load.run(docs, INDEX, driver, embedder)
    monkeypatch.undo()
    assert data_load.opensearch_state(INDEX, corpus.corpus_hash(docs), len(docs))["status"] == "different"     # was "same" with 0 documents
    assert any("without a fingerprint" in p for p in data_load.problems(docs, INDEX, driver))
    with pytest.raises(RuntimeError, match="opensearch"):
        data_load.run(docs, INDEX, driver, embedder)                              # refuses instead of reporting "nothing to do"
    data_load.run(docs, INDEX, driver, embedder, reload=True)
    assert data_load.problems(docs, INDEX, driver) == []


def test_documents_missing_from_a_fingerprinted_index_are_reported(driver, embedder, story_dir):
    docs = docs_of(story_dir)
    data_load.run(docs, INDEX, driver, embedder)
    os_s.call("POST", f"/{INDEX}/_delete_by_query?refresh=true", {"query": {"ids": {"values": ["frogs:orig"]}}})
    assert data_load.opensearch_state(INDEX, corpus.corpus_hash(docs), len(docs))["status"] == "incomplete"
    assert any("holds 8 documents but the files have 9 versions" in p for p in data_load.problems(docs, INDEX, driver))
    with pytest.raises(RuntimeError, match="incomplete"):
        data_load.run(docs, INDEX, driver, embedder)
    data_load.run(docs, INDEX, driver, embedder, reload=True)
    assert data_load.problems(docs, INDEX, driver) == []


def test_an_index_name_that_could_match_several_indices_stops_the_command_line(story_dir, capsys):
    with pytest.raises(SystemExit):
        data_load.main(["--stories-dir", str(story_dir), "--index", "*", "--reload"])
    assert "index_name" in capsys.readouterr().err


def test_only_the_missing_database_is_loaded(driver, embedder, story_dir):
    docs = docs_of(story_dir)
    data_load.run(docs, INDEX, driver, embedder)
    loaded_at = ns.read_meta(driver)["loaded_at"]
    os_s.drop_index(INDEX)
    result = data_load.run(docs, INDEX, driver, embedder)
    assert result["loaded"] == ["opensearch"] and ns.read_meta(driver)["loaded_at"] == loaded_at     # Neo4j was left alone
    assert data_load.problems(docs, INDEX, driver) == []


def test_reload_removes_only_this_projects_data(driver, embedder, story_dir):
    ns._run(driver, "CREATE (:Unrelated {note: 'someone else'})")
    os_s.call("PUT", f"/{FOREIGN_INDEX}", {})
    os_s.call("POST", f"/{FOREIGN_INDEX}/_doc?refresh=true", {"note": "someone else"})
    try:
        docs = docs_of(story_dir)
        data_load.run(docs, INDEX, driver, embedder)
        data_load.run(docs, INDEX, driver, embedder, reload=True)
        assert ns._run(driver, "MATCH (u:Unrelated) RETURN count(u) AS n")[0]["n"] == 1
        assert os_s.call("GET", f"/{FOREIGN_INDEX}/_count")["count"] == 1
        assert data_load.problems(docs, INDEX, driver) == []
    finally:
        ns._run(driver, "MATCH (u:Unrelated) DETACH DELETE u")


def test_a_graph_left_over_from_before_the_fingerprint_counts_as_a_different_corpus(driver, embedder, story_dir):
    docs = docs_of(story_dir)
    data_load.run(docs, INDEX, driver, embedder)
    ns._run(driver, "MATCH (m:StoryCorpus) DELETE m")                            # what an old, unfingerprinted load looks like
    assert data_load.neo4j_state(driver, corpus.corpus_hash(docs))["status"] == "different"
    with pytest.raises(RuntimeError, match="neo4j"):
        data_load.run(docs, INDEX, driver, embedder)


def test_a_graph_in_another_scenario_is_not_the_loaded_corpus(driver, embedder, story_dir):
    docs = docs_of(story_dir)
    meta = {"corpus_hash": corpus.corpus_hash(docs)}
    os_s.build_index(INDEX, embedder, docs, meta)
    ns.build_graph(driver, ns.prepare(embedder, docs), "original", "all", embedder, meta)
    assert any("scenario" in p for p in data_load.problems(docs, INDEX, driver))


def test_attach_searches_the_loaded_graph_without_rebuilding_it(driver, embedder, story_dir):
    docs = docs_of(story_dir)
    data_load.run(docs, INDEX, driver, embedder)
    loaded_at = ns.read_meta(driver)["loaded_at"]
    graph = ns.StoryGraph.attach(driver, embedder)
    assert graph.targets == graph.learn_from == "all" and graph.counts["versions"] == 9
    assert graph.vec("a frog by the water", 3)[0]["topic"] == "frogs" and graph.ft("loaves", 3)[0]["topic"] == "bread"
    assert ns.read_meta(driver)["loaded_at"] == loaded_at
    data_load.truncate(INDEX, driver)
    with pytest.raises(LookupError):
        ns.StoryGraph.attach(driver, embedder)


def test_the_command_line_reports_and_returns_exit_codes(driver, embedder, story_dir, capsys):
    base = ["--stories-dir", str(story_dir), "--index", INDEX]
    assert data_load.main(base + ["--status"]) == 0 and "empty" in capsys.readouterr().out
    assert data_load.main(base) == 0 and "loaded into: opensearch, neo4j" in capsys.readouterr().out
    assert data_load.main(base) == 0 and "nothing to do" in capsys.readouterr().out
    (story_dir / "frogs" / "original.txt").write_text("A frog sat quietly.")
    assert data_load.main(base) == 1 and "--reload" in capsys.readouterr().out
    assert data_load.main(base + ["--reload"]) == 0 and "removed: 9 OpenSearch documents" in capsys.readouterr().out
    (story_dir / "bread" / "original.txt").unlink()
    assert data_load.main(base) == 2 and "stories folder problem" in capsys.readouterr().out
