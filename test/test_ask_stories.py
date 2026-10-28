"""
ask_stories.py: the pure helpers and the LLM client (no services), then the whole question flow against OpenSearch + Neo4j with a fake
LLM (skipped when a service or sentence-transformers is missing). No real LLM is ever called here.

    docker start opensearch-poc neo4j-poc
    pytest test/test_ask_stories.py -v
"""
import json
import re

import pytest

pytest.importorskip("neo4j")
pytest.importorskip("nltk")

import ask_stories as ask  # noqa: E402
import corpus  # noqa: E402
import neo4j_stories as ns  # noqa: E402
import opensearch_stories as os_s  # noqa: E402


DOCS = corpus.load_corpus()
ORIGINALS = corpus.originals(DOCS)


def hit(topic, version, score):
    return {"id": f"{topic}:{version}", "topic": topic, "version": version, "kind": "original" if version == "orig" else "rewrite", "score": score}


# --------------------------------------------------------------------------- pure helpers
def test_single_method_hits_collapse_to_stories_and_show_the_original_text():
    rows = ask.collapse([hit("frog_log", "rw1", 2.0), hit("bread", "orig", 1.0), hit("frog_log", "orig", 0.5)], ORIGINALS)
    assert [r["topic"] for r in rows] == ["frog_log", "bread"]               # its best version (the rewrite, 2.0) beats bread (1.0)
    assert rows[0]["via"] == "rw1" and rows[0]["native"] == 1 and rows[1]["native"] == 2
    assert rows[0]["text"] == ORIGINALS["frog_log"]                       # never the rewrite's wording


def test_the_answer_prompt_holds_exactly_the_given_passages_numbered():
    rows = ask.collapse([hit("bread", "orig", 2.0), hit("lost_dog", "orig", 1.0)], ORIGINALS)
    prompt = ask.answer_prompt("what was baked?", rows)
    assert f"[1] {ORIGINALS['bread']}" in prompt and f"[2] {ORIGINALS['lost_dog']}" in prompt
    assert ORIGINALS["rocket"] not in prompt and prompt.endswith("Question: what was baked?")


@pytest.mark.parametrize("reply, n, expected", [
    ("```json\n[3, 1, 2]\n```", 3, [3, 1, 2]),
    ("The best order is [2,1] - hope that helps", 3, [2, 1, 3]),         # numbers left out go last
    ("[2, 2, 9, 1]", 3, [2, 1, 3]),                                       # duplicates and out-of-range numbers dropped
    ("I cannot rank these", 3, None),
    ("[]", 3, None),
    ("[true, 1.5, \"2\"]", 3, None),                                      # only real integers count
    ("[not json]", 3, None),
])
def test_parse_ranking(reply, n, expected):
    assert ask.parse_ranking(reply, n) == expected


def test_rerank_reorders_rows_and_keeps_their_native_rank():
    rows = ask.collapse([hit("bread", "orig", 3.0), hit("lost_dog", "orig", 2.0), hit("rocket", "orig", 1.0)], ORIGINALS)
    reordered, ok = ask.rerank("q", rows, lambda system, user: "[3,1,2]")
    assert ok and [r["topic"] for r in reordered] == ["rocket", "bread", "lost_dog"]
    assert [r["native"] for r in reordered] == [3, 1, 2]


def test_rerank_keeps_the_native_order_when_the_reply_is_unusable():
    rows = ask.collapse([hit("bread", "orig", 3.0), hit("lost_dog", "orig", 2.0)], ORIGINALS)
    for reply in ("sorry, no", "[LLM error: HTTP 500]"):
        reordered, ok = ask.rerank("q", rows, lambda system, user, r=reply: r)
        assert not ok and reordered == rows


def test_overlap():
    a = [{"topic": t} for t in ("x", "y", "z")]
    b = [{"topic": t} for t in ("x", "z", "w")]
    assert ask.overlap(a, b) == {"same_first": True, "both": ["x", "z"], "only_first": ["y"], "only_second": ["w"]}
    assert ask.overlap(a, [])["same_first"] is False


def test_rewrite_is_cleaned_to_one_line_of_keywords():
    assert ask.clean_rewrite('"import duties\n tariffs   customs"') == "import duties tariffs customs"


def test_the_judge_prompt_labels_both_answers_with_their_own_passages():
    one, two = ask.collapse([hit("bread", "orig", 1.0)], ORIGINALS), ask.collapse([hit("rocket", "orig", 1.0)], ORIGINALS)
    prompt = ask.judge_prompt("q", [(one, "bread answer"), (two, "rocket answer")])
    assert prompt.index("Answer A: bread answer") < prompt.index("Passages for answer B") < prompt.index("Answer B: rocket answer")


def test_the_optional_llm_steps_are_off_by_default():
    opts = ask.Options()
    assert (opts.rewrite, opts.rerank, opts.judge) == (False, False, False) and opts.top == 5


def test_env_file_fills_missing_variables_but_never_overrides_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nOPENROUTER_API_KEY=\"sk-from-file\"\nexport OTHER_VAR = 'x y'\nnot a pair\nKEPT=file\n")
    for name in ("OPENROUTER_API_KEY", "OTHER_VAR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KEPT", "environment")
    ask.load_env(env)
    import os
    assert os.environ["OPENROUTER_API_KEY"] == "sk-from-file" and os.environ["OTHER_VAR"] == "x y"
    assert os.environ["KEPT"] == "environment"
    monkeypatch.delenv("OPENROUTER_API_KEY"), monkeypatch.delenv("OTHER_VAR")   # cleanup of what load_env set


def test_a_missing_env_file_is_fine(tmp_path):
    ask.load_env(tmp_path / "nope.env")


# --------------------------------------------------------------------------- the LLM client, with a fake HTTP post
class FakeResponse:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        return self._body


class FakePost:
    def __init__(self, response):
        self.response, self.calls = response, []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_openrouter_request_shape_and_think_tags_removed(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    post = FakePost(FakeResponse(body={"choices": [{"message": {"content": "<think>hmm</think>\nIt is [1]."}}]}))
    assert ask.chat("openrouter", "some/model", "SYS", "USER", key="sk-test", post=post) == "It is [1]."
    url, kw = post.calls[0]
    assert url == ask.OPENROUTER_URL and kw["headers"]["Authorization"] == "Bearer sk-test"
    assert kw["json"] == {"model": "some/model", "temperature": 0,
                          "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]}


def test_openrouter_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-from-env")
    post = FakePost(FakeResponse(body={"choices": [{"message": {"content": "ok"}}]}))
    ask.chat("openrouter", "m", "s", "u", post=post)
    assert post.calls[0][1]["headers"]["Authorization"] == "Bearer sk-from-env"


def test_no_key_means_no_request(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    post = FakePost(FakeResponse())
    reply = ask.chat("openrouter", "m", "s", "u", post=post)
    assert ask.failed(reply) and "OPENROUTER_API_KEY" in reply and post.calls == []


def test_ollama_request_shape():
    post = FakePost(FakeResponse(body={"message": {"content": "local answer"}}))
    assert ask.chat("ollama", "qwen3:8b", "SYS", "USER", url="http://box:11434", post=post) == "local answer"
    url, kw = post.calls[0]
    assert url == "http://box:11434/api/chat" and "headers" not in kw
    assert kw["json"]["model"] == "qwen3:8b" and kw["json"]["stream"] is False and kw["json"]["options"]["temperature"] == 0


def test_failures_become_error_text_and_never_leak_the_key():
    for response in (FakeResponse(status=401, text='{"error": "invalid key"}'), RuntimeError("connection reset")):
        reply = ask.chat("openrouter", "m", "s", "u", key="sk-secret", post=FakePost(response))
        assert ask.failed(reply) and "sk-secret" not in reply
    assert "HTTP 401" in ask.chat("openrouter", "m", "s", "u", key="k", post=FakePost(FakeResponse(status=401, text="no")))
    assert ask.failed(ask.chat("ollama", "m", "s", "u", post=FakePost(FakeResponse(body={"message": {"content": "<think>x</think>"}}))))


# --------------------------------------------------------------------------- the whole flow, fake LLM, real databases
class FakeLLM:
    """Answers by role (the system prompt) and records every call."""

    def __init__(self, rewrite="tariff duties customs", fail=False):
        self.calls, self.rewrite, self.fail = [], rewrite, fail

    def __call__(self, system, user):
        self.calls.append((system, user))
        if self.fail:
            return "[LLM error: boom]"
        if system == ask.REWRITE_SYSTEM:
            return self.rewrite
        if system == ask.RERANK_SYSTEM:
            n = max(int(m) for m in re.findall(r"^\[(\d+)\]", user, flags=re.MULTILINE))
            return json.dumps(list(range(n, 0, -1)))                     # exactly reversed
        if system == ask.JUDGE_SYSTEM:
            return "A is better supported."
        return "answer based on [1]"

    def roles(self):
        return [s for s, _ in self.calls]


class Coin:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


@pytest.fixture(scope="module")
def stores(embedder):
    if not os_s.reachable():
        pytest.skip(f"OpenSearch not reachable at {os_s.URL}")
    if not ns.reachable():
        pytest.skip(f"Neo4j not reachable at {ns.URI}")
    if embedder is None:
        pytest.skip("sentence-transformers not installed")
    index = "stories_ask_test"
    os_s.build_index(index, embedder, DOCS)
    driver = ns.connect()
    try:
        yield ask.Stores(index, ns.build_graph(driver, ns.prepare(embedder, DOCS), "all", "all", embedder), embedder, ORIGINALS)
    finally:
        driver.close()
        os_s.drop_index(index)


Q = "Is the company being hurt by import duties?"


def test_retrieval_only_returns_five_stories_per_database(stores):
    res = ask.ask(Q, stores, ask.Options(), None)
    assert set(res.rows) == {"OpenSearch", "Neo4j"} and res.answers == {} and res.query == Q
    for name, rows in res.rows.items():
        assert len(rows) == 5 and len({r["topic"] for r in rows}) == 5, name
        assert rows[0]["topic"] in ("tariff_hurt", "tariff_margin"), name
        assert all(rows[i]["score"] >= rows[i + 1]["score"] for i in range(4)), name     # ranked best first
    assert all(ms > 0 for ms in res.ms.values())


def test_the_plain_run_makes_one_answer_call_per_database_from_that_databases_own_rows(stores):
    llm = FakeLLM()
    res = ask.ask(Q, stores, ask.Options(), llm)
    assert llm.roles() == [ask.ANSWER_SYSTEM, ask.ANSWER_SYSTEM]
    assert sorted(u for _, u in llm.calls) == sorted(ask.answer_prompt(Q, res.rows[n]) for n in res.rows)
    assert set(res.answers) == {"OpenSearch", "Neo4j"} and res.judge == ""


def test_a_database_that_returns_nothing_is_not_asked_to_rerank_or_answer(stores):
    llm = FakeLLM()
    opts = ask.Options(os_arm="stem", neo_arm="n_ft", rerank=True, judge=True)
    res = ask.ask("the of and", stores, opts, llm)                       # nothing but stop words: no search term is left
    assert res.rows == {"OpenSearch": [], "Neo4j": []}
    assert llm.calls == [] and res.answers == {} and res.reranked == {} and res.judge == ""
    assert ask.render(res, opts).count("(nothing matched)") == 2


def test_the_judge_needs_an_answer_from_both_databases(stores, monkeypatch):
    monkeypatch.setattr(ask, "search_neo4j", lambda *args, **kwargs: [])
    llm = FakeLLM()
    res = ask.ask(Q, stores, ask.Options(judge=True), llm)
    assert list(res.answers) == ["OpenSearch"] and res.judge == "" and llm.roles() == [ask.ANSWER_SYSTEM]


def test_rewrite_changes_what_the_databases_are_searched_with_but_not_the_question_the_llm_answers(stores):
    llm = FakeLLM(rewrite="levy refund customs")
    res = ask.ask("how did it go for them?", stores, ask.Options(rewrite=True), llm)
    assert res.query == "levy refund customs" and res.note == ""
    assert all(rows[0]["topic"].startswith("tariff") for rows in res.rows.values())
    assert all("Question: how did it go for them?" in u for s, u in llm.calls if s == ask.ANSWER_SYSTEM)
    assert llm.roles().count(ask.REWRITE_SYSTEM) == 1


def test_rerank_puts_the_llm_order_on_screen_and_remembers_the_native_rank(stores):
    plain = ask.ask(Q, stores, ask.Options(), None)
    llm = FakeLLM()
    res = ask.ask(Q, stores, ask.Options(rerank=True), llm)
    assert res.reranked == {"OpenSearch": True, "Neo4j": True}
    for name in res.rows:
        assert res.rows[name][0]["native"] > 5                      # the previously last candidate is now first
        assert res.rows[name][0]["topic"] not in [r["topic"] for r in plain.rows[name]]
    assert llm.roles().count(ask.RERANK_SYSTEM) == 2
    assert "(was #" in ask.render(res, ask.Options(rerank=True))


def test_all_three_steps_make_six_llm_calls_and_the_judge_order_is_random(stores):
    opts = ask.Options(rewrite=True, rerank=True, judge=True)
    llm = FakeLLM()
    res = ask.ask(Q, stores, opts, llm, rng=Coin(0.9))
    assert len(llm.calls) == 6 and llm.roles()[0] == ask.REWRITE_SYSTEM and llm.roles()[-1] == ask.JUDGE_SYSTEM
    assert res.judge_order == ("Neo4j", "OpenSearch") and res.judge == "A is better supported."
    judge_prompt = llm.calls[-1][1]
    assert judge_prompt.index(res.answers["Neo4j"]) < judge_prompt.index("Passages for answer B")
    assert ask.ask(Q, stores, opts, FakeLLM(), rng=Coin(0.1)).judge_order == ("OpenSearch", "Neo4j")


def test_llm_failures_degrade_to_the_plain_native_results(stores):
    plain = ask.ask(Q, stores, ask.Options(), None)
    res = ask.ask(Q, stores, ask.Options(rewrite=True, rerank=True, judge=True), FakeLLM(fail=True))
    assert res.query == Q and "rewrite failed" in res.note
    assert res.reranked == {"OpenSearch": False, "Neo4j": False}
    assert {n: [r["topic"] for r in rows] for n, rows in res.rows.items()} == {n: [r["topic"] for r in rows] for n, rows in plain.rows.items()}
    assert res.judge == "" and all(ask.failed(a) for a in res.answers.values())
    text = ask.render(res, ask.Options())
    assert "OpenSearch" in text and "Neo4j" in text and "agreement:" in text and "native order" in text


# --------------------------------------------------------------------------- --debug: the exact requests sent to each database
def test_shorten_abbreviates_embeddings_and_long_lists_but_keeps_short_ones():
    out = ask.shorten({"vector": [0.1234567] * 384, "stems": ["a", "b"], "weights": list(range(100)), "k": 30})
    assert out["vector"].startswith("<384 floats: 0.1235, 0.1235") and out["stems"] == ["a", "b"] and out["k"] == 30
    assert len(out["weights"]) == 31 and out["weights"][-1] == "... 70 more"


def test_format_trace_prints_http_bodies_and_cypher_with_parameters():
    lines = ask.format_trace([("http", "POST /stories/_search", {"size": 30, "query": {"match": {"text.syn": "frog"}}}),
                              ("cypher", "\n    MATCH (t:StoryTerm)\n    WHERE t.stem IN $stems\n    RETURN t.stem\n", {"stems": ["frog"]})])
    text = "\n".join(lines)
    assert '1. POST /stories/_search' in text and '"text.syn": "frog"' in text
    assert "2. Cypher" in text and "| MATCH (t:StoryTerm)" in text and "| WHERE t.stem IN $stems" in text
    assert 'parameters: {"stems": ["frog"]}' in text


def test_capture_records_both_libraries_and_always_switches_the_hooks_off(monkeypatch):
    class Resp:
        status_code, text = 200, '{"ok": true}'
        def json(self): return {"ok": True}
    monkeypatch.setattr(os_s.requests, "request", lambda *a, **k: Resp())
    sink = []
    with pytest.raises(RuntimeError):
        with ask.capture(sink):
            os_s.call("POST", "/x/_search", {"q": 1})
            os_s.call("POST", "/_bulk", ndjson="{}\n")               # bulk loads are not queries
            ns.TRACE("RETURN 1", {})
            raise RuntimeError("search failed")
    assert sink == [("http", "POST /x/_search", {"q": 1}), ("cypher", "RETURN 1", {})]
    assert os_s.TRACE is None and ns.TRACE is None


def test_debug_shows_only_the_requests_the_chosen_methods_send(stores):
    res = ask.ask(Q, stores, ask.Options(debug=True), None)                        # hybrid + n_hyb
    os_bodies = [p for kind, _, p in res.trace["OpenSearch"] if kind == "http"]
    assert len(res.trace["OpenSearch"]) == 2 and all(label == "POST /stories_ask_test/_search" for _, label, _ in res.trace["OpenSearch"])
    assert os_bodies[0]["query"]["bool"]["must"] == [{"match": {"text.syn": Q}}]         # keywords, synonyms applied by the index
    knn = os_bodies[1]["query"]["knn"]["embedding"]
    assert len(knn["vector"]) == 384 and knn["k"] == ask.SEARCH_SIZE                  # the real, unabbreviated vector
    cypher = [(text, params) for kind, text, params in res.trace["Neo4j"]]
    assert any("HAS" in t and "StoryTerm" in t for t, _ in cypher) and any("VECTOR INDEX story_vec" in t for t, _ in cypher)
    assert any("duti" in params.get("stems", []) for _, params in cypher)                # 'duties' as the graph sees it
    assert res.notes["OpenSearch"] and "story_syn" in res.notes["OpenSearch"][0]
    assert os_s.TRACE is None and ns.TRACE is None


def test_a_single_method_sends_a_single_kind_of_request(stores):
    res = ask.ask(Q, stores, ask.Options(os_arm="dense", neo_arm="n_vec", debug=True), None)
    assert [("knn" in p["query"]) for kind, _, p in res.trace["OpenSearch"]] == [True]
    assert len(res.trace["Neo4j"]) == 1 and "VECTOR INDEX story_vec" in res.trace["Neo4j"][0][1]
    assert res.notes["OpenSearch"] == []                                                # vectors have no keyword analysis


def test_without_debug_nothing_is_recorded_and_the_output_has_no_debug_block(stores):
    res = ask.ask(Q, stores, ask.Options(), None)
    assert res.trace == {} and res.notes == {}
    assert "debug:" not in ask.render(res, ask.Options())


def test_the_debug_output_names_the_requests_and_stays_readable(stores):
    opts = ask.Options(debug=True)
    text = ask.render(ask.ask(Q, stores, opts, None), opts)
    assert "debug: 2 request(s) sent to OpenSearch" in text and "POST /stories_ask_test/_search" in text
    assert "<384 floats:" in text and "VECTOR INDEX story_vec" in text and "Cypher" in text
    assert max(len(line) for line in text.splitlines()) < 150


def test_each_method_returns_what_the_library_returned_when_all_methods_ran(stores):
    """Asking for one method (so that only its requests are sent) must rank exactly like running every method."""
    key = lambda hits: [(h["id"], round(h["score"], 6)) for h in hits]  # noqa: E731
    everything = os_s.run_arms(stores.index, Q, stores.embedder, size=ask.SEARCH_SIZE)
    for arm in ask.OS_ARMS:
        assert key(ask.search_opensearch(stores, Q, arm)) == key(everything[arm]), arm
    everything = stores.graph.arms(Q, ask.SEARCH_SIZE)
    for arm in ask.NEO_ARMS:
        assert key(ask.search_neo4j(stores, Q, arm)) == key(everything[arm]), arm
