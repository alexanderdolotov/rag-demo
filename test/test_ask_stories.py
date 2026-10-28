"""
ask_stories.py: the pure helpers and the LLM client (no services), then the whole question flow against OpenSearch + Neo4j with a fake
LLM (skipped when a service or sentence-transformers is missing). No real LLM is ever called here.

    docker start opensearch-poc neo4j-poc
    pytest test/test_ask_stories.py -v
"""
import json
import re
import textwrap

import pytest

pytest.importorskip("neo4j")
pytest.importorskip("nltk")

import ask_stories as ask  # noqa: E402
import corpus  # noqa: E402
import intent as it  # noqa: E402
import neo4j_stories as ns  # noqa: E402
import opensearch_stories as os_s  # noqa: E402
import retrieval  # noqa: E402


DOCS = corpus.load_corpus()
ORIGINALS = corpus.originals(DOCS)
TEXTS = {d["id"]: d["text"] for d in DOCS}


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


def test_overlap():
    a = [{"topic": t} for t in ("x", "y", "z")]
    b = [{"topic": t} for t in ("x", "z", "w")]
    assert ask.overlap(a, b) == {"same_first": True, "both": ["x", "z"], "only_first": ["y"], "only_second": ["w"]}
    assert ask.overlap(a, [])["same_first"] is False


def test_the_judge_prompt_labels_both_answers_with_their_own_passages():
    one, two = ask.collapse([hit("bread", "orig", 1.0)], ORIGINALS), ask.collapse([hit("rocket", "orig", 1.0)], ORIGINALS)
    prompt = ask.judge_prompt("q", [(one, "bread answer"), (two, "rocket answer")])
    assert prompt.index("Answer A: bread answer") < prompt.index("Passages for answer B") < prompt.index("Answer B: rocket answer")


# --------------------------------------------------------------------------- the sentence shown beside a result
def stems(query):
    return {stem for stem, _ in ns.tokens(query)}


SENTENCE = "The board said on Thursday that after a lengthy review of the many options in front of it the company would pay a special dividend of two dollars a share."


def test_a_sentence_that_fits_is_shown_whole():
    assert ask.excerpt("A short one.", (2, 7)) == "A short one."


def test_a_long_sentence_is_cut_around_the_hit_with_dots_where_text_was_left_out():
    start = SENTENCE.index("dividend")
    out = ask.excerpt(SENTENCE, (start, start + len("dividend")), 60)
    assert out.startswith("...") and out.endswith("...") and len(out) <= 60
    assert abs((out.index("dividend") + 4) - len(out) / 2) < 10          # the word sits in the middle
    assert all(word in SENTENCE.split() for word in out.replace("...", " ").split())     # nothing is cut through a word


def test_near_the_start_or_end_of_the_sentence_there_are_no_dots_on_that_side():
    assert ask.excerpt(SENTENCE, (4, 9), 60).startswith("The board") and ask.excerpt(SENTENCE, (4, 9), 60).endswith("...")
    end = len(SENTENCE) - 1
    out = ask.excerpt(SENTENCE, (end - 5, end), 60)
    assert out.startswith("...") and out.endswith("share.")


def test_the_cut_keeps_the_whole_stretch_that_holds_several_query_words():
    text = "Everything about the bonfire at the old festival was loud, and then each person throws a handful of grain into the flames before leaving."
    out = ask.evidence(text, stems("bonfire grain"), 90)
    assert "bonfire" in out and "grain" in out and len(out) <= 90


def test_the_sentence_with_the_most_different_query_words_wins_and_the_first_one_on_a_tie():
    text = "Duty came first. The company paid duty and tariff costs. Later, duty again."
    assert ask.evidence(text, stems("duty tariff")) == "The company paid duty and tariff costs."
    assert ask.evidence("Duty one. Duty two.", stems("duty")) == "Duty one."


def test_query_words_are_matched_by_stem_and_a_word_nobody_wrote_gives_nothing():
    assert ask.evidence("They owed a duty to the town.", stems("duties")) == "They owed a duty to the town."
    assert ask.evidence("The frog sat still.", stems("ceo")) is None
    assert ask.evidence("The frog sat still.", stems("the of and")) is None          # only stop words: nothing to look for


def test_the_snippet_comes_from_the_version_that_matched_else_the_original_else_there_is_none():
    stores = ask.Stores("i", None, None, {"a": "The original talks about frogs."}, {"a:orig": "The original talks about frogs.", "a:rw1": "A toad sat on a log."})
    row = lambda: {"topic": "a", "via": "rw1"}  # noqa: E731
    for query, expected in (("toad", ("A toad sat on a log.", "rw1")), ("frogs", ("The original talks about frogs.", "orig"))):
        rows = [row()]
        ask.add_snippets(rows, query, stores)
        assert (rows[0]["snippet"], rows[0]["snippet_from"]) == expected, query
    rows = [row()]
    ask.add_snippets(rows, "hello", stores)
    assert "snippet" not in rows[0]


def test_question_filler_is_not_evidence_for_a_match():
    text = "My daughter asked what it was carrying. The night watchman locked up."
    stores = ask.Stores("i", None, None, {"a": text}, {"a:orig": text})
    rows = [{"topic": "a", "via": "orig"}]
    ask.add_snippets(rows, "what were the night watchman's responsibilities?", stores)
    assert rows[0]["snippet"] == "The night watchman locked up."                # not the sentence that merely contains "what"
    rows = [{"topic": "a", "via": "orig"}]
    ask.add_snippets(rows, "what were they", stores)                           # nothing but filler: nothing to point at
    assert "snippet" not in rows[0]


def test_what_is_shown_beside_a_result():
    base = {"topic": "a", "via": "rw1", "text": "The start of the story. " + "More words follow here. " * 10}
    assert ask.shown(dict(base, snippet="A toad sat.", snippet_from="rw1")) == "A toad sat."
    assert ask.shown(dict(base, snippet="A frog sat.", snippet_from="orig")) == "[orig] A frog sat."        # not the version that matched: say so
    assert ask.shown(base).startswith("The start of the story.") and ask.shown(base).endswith("...") and len(ask.shown(base)) <= ask.SNIPPET_WIDTH


def test_the_answer_prompt_says_which_conditions_the_passages_may_not_cover():
    rows = [{"text": "A loaf."}]
    assert ask.answer_prompt("q?", rows, ["near the user"]) == (
        "Passages:\n[1] A loaf.\n\nThe question also asks for: near the user. If the passages do not show whether that holds, say so.\n\nQuestion: q?")


def test_format_trace_shows_a_step_as_a_header_and_numbers_only_the_requests():
    text = "\n".join(ask.format_trace([("step", "core: bread", None), ("http", "POST /a", {"q": 1}), ("step", "core: bread | related: loaf", None),
                                       ("http", "POST /b", {"q": 2})]))
    assert text.index("-- core: bread --") < text.index("1. POST /a") < text.index("-- core: bread | related: loaf --") < text.index("2. POST /b")


def test_the_defaults_use_both_llm_steps_and_leave_the_judge_off():
    opts = ask.Options()
    assert (opts.rewrite, opts.rerank, opts.judge) == (True, True, False) and opts.top == 5


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
ANALYSIS = {"concepts": [{"name": "tariffs", "core": ["tariff", "duties"], "related": ["customs", "levies"]}], "must": [], "context": [], "unverifiable": []}


class FakeLLM:
    """Answers by role (the system prompt) and records every call. `scores` maps a story to the score it gets (0 for any other story), or,
    keyed by concept name, one such table per concept."""

    def __init__(self, analysis=ANALYSIS, scores=None, bad_scores=False, fail=False):
        self.calls, self.analysis, self.scores, self.bad_scores, self.fail = [], analysis, scores or {}, bad_scores, fail

    def __call__(self, system, user):
        self.calls.append((system, user))
        if self.fail:
            return "[LLM error: boom]"
        if system == it.ANALYZE_SYSTEM:
            return self.analysis if isinstance(self.analysis, str) else json.dumps(self.analysis)
        if system == it.SCORE_SYSTEM:
            if self.bad_scores:
                return "I would rather not"
            subject = re.search(r"^Subject to judge the passages against: (.*?) \(core words", user, flags=re.MULTILINE).group(1)
            table = self.scores.get(subject, self.scores)
            return json.dumps({pid: table.get(pid, 0) for pid in re.findall(r"^\[(\w+)\] ", user, flags=re.MULTILINE)})
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
        yield ask.Stores(index, ns.build_graph(driver, ns.prepare(embedder, DOCS), "all", "all", embedder), embedder, ORIGINALS, TEXTS)
    finally:
        driver.close()
        os_s.drop_index(index)


Q = "Is the company being hurt by import duties?"


def test_retrieval_only_returns_five_stories_per_database(stores):
    res = ask.ask(Q, stores, ask.Options(), None)
    assert set(res.rows) == {"OpenSearch", "Neo4j"} and res.answers == {} and res.question == Q
    for name, rows in res.rows.items():
        assert len(rows) == 5 and len({r["topic"] for r in rows}) == 5, name
        assert rows[0]["topic"] in ("tariff_hurt", "tariff_margin"), name
        assert all(rows[i]["score"] >= rows[i + 1]["score"] for i in range(4)), name     # ranked best first
    assert all(ms > 0 for ms in res.ms.values())


def test_each_result_shows_the_sentence_where_the_query_words_occur_not_the_start_of_the_story(stores):
    res = ask.ask("bonfire grain", stores, ask.Options(), None)
    for name, rows in res.rows.items():
        row = next((r for r in rows if r["topic"] == "village_customs"), None)
        assert row is not None, f"{name} did not return village_customs"
        words = row["snippet"].lower()
        assert ("bonfire" in words or "grain" in words) and not words.startswith("every autumn") and len(row["snippet"]) <= ask.SNIPPET_WIDTH, name
        assert row["text"] == ORIGINALS["village_customs"]                    # what the LLM is given is still the original
    assert "bonfire" in ask.render(res, ask.Options())


def test_a_result_with_none_of_the_query_words_in_it_shows_the_start_of_the_story(stores):
    res = ask.ask("zxqv blorf", stores, ask.Options(), None)                # only the vectors can match this
    rows = [r for rows in res.rows.values() for r in rows]
    assert rows and not any("snippet" in r for r in rows)
    assert all(textwrap.shorten(r["text"], ask.SNIPPET_WIDTH, placeholder="...") in ask.render(res, ask.Options()) for r in rows)


TARIFFS = {"tariff_hurt": 3, "tariff_margin": 2}


def test_the_default_run_analyses_the_question_scores_the_candidates_and_answers_from_what_scored_well(stores):
    llm = FakeLLM(scores=TARIFFS)
    res = ask.ask(Q, stores, ask.Options(), llm)
    assert res.intent.concepts[0].core == ("tariff", "duties")
    assert llm.roles() == [it.ANALYZE_SYSTEM, it.SCORE_SYSTEM, ask.ANSWER_SYSTEM, ask.ANSWER_SYSTEM]       # one analysis, one scoring call for both databases
    for name, rows in res.rows.items():
        assert [r["topic"] for r in rows] == ["tariff_hurt", "tariff_margin"], name                       # only what scored well, best first
        assert [r["rel"] for r in rows] == [3, 2], name
    assert res.judged == {"OpenSearch": True, "Neo4j": True}
    assert all("[2]" in u and "[3]" not in u for s, u in llm.calls if s == ask.ANSWER_SYSTEM)              # the LLM answers from those two passages only
    text = ask.render(res, ask.Options())
    assert "understood: tariffs" in text and "[3/3]" in text and "judged by the LLM" in text


def test_without_the_rewrite_the_question_is_searched_as_typed_and_the_scoring_judges_against_the_question(stores):
    llm = FakeLLM(scores=TARIFFS)
    res = ask.ask(Q, stores, ask.Options(rewrite=False), llm)
    assert llm.roles() == [it.SCORE_SYSTEM, ask.ANSWER_SYSTEM, ask.ANSWER_SYSTEM] and res.intent is None
    assert f"Subject to judge the passages against: {Q} (core words: {Q})" in llm.calls[0][1]
    assert all([r["topic"] for r in rows] == ["tariff_hurt", "tariff_margin"] for rows in res.rows.values())


def test_without_the_rerank_the_databases_own_order_is_kept_after_the_analysis(stores):
    llm = FakeLLM()
    res = ask.ask(Q, stores, ask.Options(rerank=False), llm)
    assert llm.roles() == [it.ANALYZE_SYSTEM, ask.ANSWER_SYSTEM, ask.ANSWER_SYSTEM] and res.judged == {}
    assert all(len(rows) == 5 and "rel" not in rows[0] for rows in res.rows.values())


def test_the_conditions_reach_the_scoring_and_the_unverifiable_ones_reach_the_answer_step(stores):
    analysis = dict(ANALYSIS, must=["benefited, not hurt"], unverifiable=["near the user"])
    llm = FakeLLM(analysis=analysis, scores=TARIFFS)
    res = ask.ask(Q, stores, ask.Options(), llm)
    scoring = next(u for s, u in llm.calls if s == it.SCORE_SYSTEM)
    assert "Conditions a passage must meet: benefited, not hurt" in scoring and "Ignore these, a story cannot show them: near the user" in scoring
    answers = [u for s, u in llm.calls if s == ask.ANSWER_SYSTEM]
    assert answers and all("The question also asks for: near the user." in u for u in answers)
    text = ask.render(res, ask.Options())
    assert "a passage must meet: benefited, not hurt" in text and "no story can show: near the user" in text


def test_each_concept_is_searched_and_scored_on_its_own_and_the_results_take_turns_main_concept_first(stores):
    two = {"concepts": [{"name": "tariffs", "core": ["tariff"]}, {"name": "baking", "core": ["bread", "bake"]}], "must": [], "context": [], "unverifiable": []}
    llm = FakeLLM(analysis=two, scores={"tariffs": {"tariff_hurt": 3, "tariff_margin": 3}, "baking": {"bread": 3, "village_customs": 2}})
    res = ask.ask("tariffs, and who bakes?", stores, ask.Options(top=4), llm)
    assert llm.roles().count(it.SCORE_SYSTEM) == 2
    for name, rows in res.rows.items():
        assert [r["concept"] for r in rows] == ["tariffs", "baking", "tariffs", "baking"], name
        assert {r["topic"] for r in rows if r["concept"] == "baking"} == {"bread", "village_customs"}, name
    text = ask.render(res, ask.Options())
    assert "main concept: tariffs" in text and "concept 2: baking" in text and "[baking]" in text


def test_an_analysis_that_cannot_be_read_means_the_question_is_searched_as_typed_and_still_scored(stores):
    llm = FakeLLM(analysis="I would rather not", scores=TARIFFS)
    res = ask.ask(Q, stores, ask.Options(), llm)
    assert res.intent is None and "could not be read" in res.note
    assert llm.roles() == [it.ANALYZE_SYSTEM, it.SCORE_SYSTEM, ask.ANSWER_SYSTEM, ask.ANSWER_SYSTEM] and res.rows["OpenSearch"]


def test_scores_that_cannot_be_read_leave_each_databases_own_order(stores):
    res = ask.ask(Q, stores, ask.Options(), FakeLLM(bad_scores=True))
    assert res.judged == {"OpenSearch": False, "Neo4j": False} and "scores could not be read" in res.note
    assert all(len(rows) == 5 and "rel" not in rows[0] for rows in res.rows.values())
    assert "native order" in ask.render(res, ask.Options())


def test_nothing_that_scores_above_zero_means_nothing_is_shown_and_nothing_is_answered(stores):
    llm = FakeLLM(analysis={"concepts": [{"name": "zxqv", "core": ["zxqv"]}]})        # every candidate scores 0
    opts = ask.Options(judge=True)
    res = ask.ask("zxqv", stores, opts, llm)
    assert res.rows == {"OpenSearch": [], "Neo4j": []} and res.answers == {} and res.judge == ""
    assert llm.roles() == [it.ANALYZE_SYSTEM, it.SCORE_SYSTEM] and ask.render(res, opts).count("(nothing relevant found)") == 2


def test_a_database_that_returns_nothing_is_not_asked_to_score_or_answer(stores):
    llm = FakeLLM()
    opts = ask.Options(os_arm="stem", neo_arm="n_ft", rewrite=False, judge=True)
    res = ask.ask("the of and", stores, opts, llm)                       # nothing but stop words: no search term is left
    assert res.rows == {"OpenSearch": [], "Neo4j": []}
    assert llm.calls == [] and res.answers == {} and res.judge == ""
    assert ask.render(res, opts).count("(nothing relevant found)") == 2


def test_without_an_llm_the_question_is_searched_once_as_typed(stores):
    res = ask.ask(Q, stores, ask.Options(), None)
    assert res.intent is None and res.judged == {} and res.note == "" and all(len(rows) == 5 for rows in res.rows.values())


def test_the_judge_needs_an_answer_from_both_databases(stores, monkeypatch):
    monkeypatch.setattr(ask, "search_neo4j", lambda *args, **kwargs: [])
    llm = FakeLLM()
    res = ask.ask(Q, stores, ask.Options(rewrite=False, rerank=False, judge=True), llm)
    assert list(res.answers) == ["OpenSearch"] and res.judge == "" and llm.roles() == [ask.ANSWER_SYSTEM]


def test_all_the_llm_steps_together_make_five_calls_and_the_judge_order_is_random(stores):
    opts = ask.Options(judge=True)
    llm = FakeLLM(scores=TARIFFS)
    res = ask.ask(Q, stores, opts, llm, rng=Coin(0.9))
    assert llm.roles() == [it.ANALYZE_SYSTEM, it.SCORE_SYSTEM, ask.ANSWER_SYSTEM, ask.ANSWER_SYSTEM, ask.JUDGE_SYSTEM]
    assert res.judge_order == ("Neo4j", "OpenSearch") and res.judge == "A is better supported."
    judge_prompt = llm.calls[-1][1]
    assert judge_prompt.index(res.answers["Neo4j"]) < judge_prompt.index("Passages for answer B")
    assert ask.ask(Q, stores, opts, FakeLLM(scores=TARIFFS), rng=Coin(0.1)).judge_order == ("OpenSearch", "Neo4j")


def test_each_concept_query_has_boosted_tiers_and_shows_up_in_the_debug_trace(stores):
    res = ask.ask(Q, stores, ask.Options(debug=True), FakeLLM(scores=TARIFFS))
    assert [t for kind, t, _ in res.trace["OpenSearch"] if kind == "step"] == ["core: tariff, duties | related: customs, levies"]
    bodies = [p for kind, _, p in res.trace["OpenSearch"] if kind == "http"]
    assert len(bodies) == 2 and [c["match"]["text.syn"]["boost"] for c in bodies[0]["query"]["bool"]["should"]] == [1.0, retrieval.TIER_WEIGHT["related"]]
    assert "-- core: tariff, duties | related: customs, levies --" in ask.render(res, ask.Options(debug=True))


def test_llm_failures_degrade_to_a_plain_search_with_the_databases_own_order(stores):
    plain = ask.ask(Q, stores, ask.Options(), None)
    res = ask.ask(Q, stores, ask.Options(judge=True), FakeLLM(fail=True))
    assert res.intent is None and "rewrite failed" in res.note and res.judged == {"OpenSearch": False, "Neo4j": False}
    assert {n: [r["topic"] for r in rows] for n, rows in res.rows.items()} == {n: [r["topic"] for r in rows] for n, rows in plain.rows.items()}
    assert res.judge == "" and all(ask.failed(a) for a in res.answers.values())
    assert "agreement:" in ask.render(res, ask.Options())


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
