"""
Hard questions (test/hard_questions.json): what a plain search cannot get right and the LLM rewriter + reranker should. Judged by hand against
stories/, scored as precision / recall / F1 / MRR of the stories shown, in every mode, on both databases.

The LLM replies come from test/cassettes/hard_questions.json, recorded once from the real model (anthropic/claude-haiku-5.5, temperature 0) by
test/record_hard_questions.py, so these tests are deterministic, need no key and make no API call. They do need OpenSearch and Neo4j.
If a prompt, the corpus or the candidates the databases return change, a recorded reply is no longer found and the test says to record again.

Read the numbers with these limits in mind: 13 questions, 20 stories and one judge model. Questions h13-h15 were held out when the design was
written; h13 and h14 then exposed one flaw (conditions applied to the wrong subject, fixed in intent.score_prompt), so only h15 is untouched.
"""
import json
from types import SimpleNamespace

import pytest

import ask_stories as ask
import corpus
import hard_eval as he
import hard_report
import intent as it
import neo4j_stories as ns
import opensearch_stories as os_s
from llm_cassette import BudgetExceeded, Cassette, CassetteMiss

DOCS = corpus.load_corpus()
STORIES = he.stories_of(DOCS)
QUESTIONS = he.load_questions(DOCS)
IDS = [q["id"] for q in QUESTIONS]
BY_ID = {q["id"]: q for q in QUESTIONS}
PLAIN, REWRITE, FULL = "no LLM, one search", "rewrite only", "rewrite + rerank (default)"
TWO_PARTS = ("h02", "h09", "h11", "h12", "h13", "h14")        # the question names its main subject first, then a second one


# --------------------------------------------------------------------------- the benchmark itself, no services
def test_the_questions_file_is_consistent():
    assert len(IDS) == len(set(IDS)) and len(QUESTIONS) >= 12
    assert all(q["question"] and q["relevant"] and q["note"] for q in QUESTIONS)


@pytest.mark.parametrize("shown, relevant, acceptable, expected", [
    (["a", "b"], ["a", "b"], [], dict(precision=1.0, recall=1.0, f1=1.0, mrr=1.0)),
    (["x", "a", "y", "z", "w"], ["a"], [], dict(precision=0.2, recall=1.0, f1=pytest.approx(1 / 3), mrr=0.5)),
    (["a", "b", "c"], ["a", "z"], ["b"], dict(precision=0.5, recall=0.5, f1=0.5, mrr=1.0)),     # an acceptable story is not counted at all
    ([], ["a"], [], dict(precision=0.0, recall=0.0, f1=0.0, mrr=0.0)),                          # showing nothing scores 0
    (["x", "y"], ["a"], [], dict(precision=0.0, recall=0.0, f1=0.0, mrr=0.0)),
])
def test_how_a_list_of_results_is_scored(shown, relevant, acceptable, expected):
    assert he.judge_rows(shown, relevant, acceptable) == expected


# --------------------------------------------------------------------------- the modes are command lines
def options_of(mode):
    args = ask.make_parser().parse_args(he.MODES[mode])
    llm = None if args.no_llm else (lambda system, user: "")
    return args, ask.options_from(args, llm)


def test_each_mode_is_the_app_run_with_its_arguments():
    args, plain = options_of(PLAIN)
    assert args.no_llm                                                                       # no LLM at all: ask() ignores both LLM steps
    args, rewrite = options_of(REWRITE)
    assert not args.no_llm and (rewrite.rewrite, rewrite.rerank) == (True, False)
    args, full = options_of(FULL)
    assert not args.no_llm and (full.rewrite, full.rerank) == (True, True)
    for opts in (plain, rewrite, full):                                                      # and nothing else differs from the app's defaults
        assert (opts.os_arm, opts.neo_arm, opts.top, opts.judge, opts.debug) == ("hybrid", "n_hyb", 5, False, False)
    assert he.MODES[FULL] == [] and he.command(he.MODES[FULL]) == "python ask_stories.py"


def test_the_commands_shown_for_the_modes_are_accepted_by_the_apps_own_parser():
    for mode, argv in he.MODES.items():
        ask.make_parser().parse_args(argv)                                                   # an unknown flag would exit here
        assert he.command(argv).startswith("python ask_stories.py") and mode


# --------------------------------------------------------------------------- the confusion matrix, on made-up results
def made_up(shown_by_db):
    return SimpleNamespace(rows={db: [{"topic": t} for t in shown] for db, shown in shown_by_db.items()})


def test_the_confusion_matrix_counts_right_wrong_and_missing_stories_and_leaves_acceptable_ones_out():
    questions = [{"id": "q", "relevant": ["a", "b"], "acceptable": ["c"]}]
    results = {"m": {"q": made_up({"OpenSearch": ["a", "c", "x", "y"], "Neo4j": []})}}
    got = he.confusion(results, questions, ["a", "b", "c", "x", "y", "z"])["m"]
    assert got["OpenSearch"] == {"tp": 1, "fn": 1, "fp": 2, "tn": 1}                         # a is right, b is missing, x and y are wrong, z is rightly absent; c is not counted
    assert got["Neo4j"] == {"tp": 0, "fn": 2, "fp": 0, "tn": 3}
    for cell in got.values():
        assert sum(cell.values()) == 5                                                       # every story except the acceptable one, exactly once


# --------------------------------------------------------------------------- the cassette
def fake_fetch(calls):
    def fetch(system, user):
        calls.append(user)
        return f"reply to {user}"
    return fetch


def test_a_recorded_reply_is_replayed_and_a_missing_one_is_refused_when_not_recording(tmp_path):
    path, calls = tmp_path / "c.json", []
    Cassette(path, live=True, fetch=fake_fetch(calls))(it.ANALYZE_SYSTEM, "q1")
    replay = Cassette(path)
    assert replay(it.ANALYZE_SYSTEM, "q1") == "reply to q1" and calls == ["q1"]
    with pytest.raises(CassetteMiss, match="record it with"):
        replay(it.ANALYZE_SYSTEM, "q2")
    with pytest.raises(CassetteMiss):
        replay(it.SCORE_SYSTEM, "q1")                    # the same user text under another prompt is another question


def test_answers_are_never_fetched_or_recorded(tmp_path):
    import ask_stories as ask
    calls = []
    cassette = Cassette(tmp_path / "c.json", live=True, fetch=fake_fetch(calls))
    assert cassette(ask.ANSWER_SYSTEM, "anything").startswith("(answer not recorded") and calls == [] and cassette.entries == {}


def test_the_budget_counts_every_call_ever_made_not_the_replies_on_file_and_stops_before_the_call(tmp_path):
    path, calls = tmp_path / "c.json", []
    path.write_text(json.dumps({"meta": {"calls_made": 3}, "replies": {}}))              # three calls were paid for, their replies since discarded
    cassette = Cassette(path, live=True, max_calls=4, fetch=fake_fetch(calls))
    cassette(it.ANALYZE_SYSTEM, "one")                                                   # the fourth call is allowed ...
    with pytest.raises(BudgetExceeded, match="4 API calls"):
        cassette(it.ANALYZE_SYSTEM, "two")                                               # ... a fifth is refused before it is made
    assert calls == ["one"] and cassette.spent == 4 and Cassette(path).spent == 4


def test_a_failed_call_is_reported_and_not_recorded(tmp_path):
    cassette = Cassette(tmp_path / "c.json", live=True, fetch=lambda system, user: "[LLM error: HTTP 500]")
    with pytest.raises(RuntimeError, match="nothing was recorded"):
        cassette(it.ANALYZE_SYSTEM, "q")
    assert cassette.entries == {}


# --------------------------------------------------------------------------- the comparison, on the real databases and the recorded replies
@pytest.fixture(scope="module")
def replayed(embedder):
    if not os_s.reachable():
        pytest.skip(f"OpenSearch not reachable at {os_s.URL}")
    if not ns.reachable():
        pytest.skip(f"Neo4j not reachable at {ns.URI}")
    if embedder is None:
        pytest.skip("sentence-transformers not installed")
    cassette = Cassette(he.CASSETTE_FILE)                                  # replay only: a missing reply fails the test, it never calls the API
    with he.built_stores(embedder) as stores:
        results = he.evaluate(stores, QUESTIONS, cassette)
    return SimpleNamespace(results=results, table=he.scores(results, QUESTIONS), cassette=cassette)


def shown(replayed, mode, qid, db):
    return [r["topic"] for r in replayed.results[mode][qid].rows[db]]


def test_the_full_pipeline_beats_every_other_mode_by_a_wide_margin(replayed):
    t = replayed.table
    summary = he.format_table(t)
    others = [REWRITE, PLAIN]
    assert he.mean(t, FULL, "f1") >= max(he.mean(t, m, "f1") for m in others) + 0.3, summary
    assert he.mean(t, FULL, "precision") >= 0.9 and he.mean(t, FULL, "precision") >= max(he.mean(t, m, "precision") for m in others) + 0.4, summary
    assert he.mean(t, FULL, "mrr") >= max(he.mean(t, m, "mrr") for m in others), summary


def test_finding_more_is_the_rewriters_job_and_cutting_the_noise_is_the_rerankers(replayed):
    t = replayed.table
    assert he.mean(t, REWRITE, "recall") >= he.mean(t, PLAIN, "recall")                      # concept queries find at least what the plain search does
    assert he.mean(t, REWRITE, "precision") < 0.5                                             # but, without scores, the list is still padded with noise
    assert he.mean(t, FULL, "recall") >= he.mean(t, REWRITE, "recall")                        # scoring and cutting lose none of what was found


def test_a_plain_search_pads_its_list_with_noise_on_nearly_every_question(replayed):
    padded = [qid for qid in IDS for db in he.DBS
              if len(set(shown(replayed, PLAIN, qid, db)) - set(BY_ID[qid]["relevant"]) - set(BY_ID[qid]["acceptable"])) >= 2]
    assert len(padded) >= 0.8 * len(IDS) * len(he.DBS)


@pytest.mark.parametrize("qid", IDS)
def test_the_full_pipeline_shows_every_correct_story_and_nothing_else_on_both_databases(replayed, qid):
    q = BY_ID[qid]
    for db in he.DBS:
        got = shown(replayed, FULL, qid, db)
        assert set(q["relevant"]) <= set(got), (db, got)
        assert set(got) <= set(q["relevant"]) | set(q["acceptable"]), (db, got)


@pytest.mark.parametrize("qid", TWO_PARTS)
def test_the_main_subject_of_a_two_part_question_comes_first(replayed, qid):
    for db in he.DBS:
        rows = replayed.results[FULL][qid].rows[db]
        assert rows[0]["topic"] == BY_ID[qid]["relevant"][0] and [r["concept"] for r in rows[:2]] != [rows[0]["concept"]] * 2


def test_the_wrong_polarity_story_is_first_in_a_plain_search_and_absent_from_the_full_pipeline(replayed):
    for db in he.DBS:
        assert shown(replayed, PLAIN, "h03", db)[0] == "tariff_hurt"                          # "benefited from tariffs" -> the story about being hurt
        assert shown(replayed, FULL, "h03", db) == ["tariff_margin"]
        assert "tariff_margin" not in shown(replayed, FULL, "h04", db)                        # and the reverse question


def test_an_ambiguous_word_brings_a_plain_search_a_finance_story_and_the_full_pipeline_does_not(replayed):
    for db in he.DBS:
        assert set(shown(replayed, PLAIN, "h06", db)) & {"regional_bank", "bank_rates"}
        assert shown(replayed, FULL, "h06", db) == ["river_bank"]


def test_a_misspelt_question_with_a_condition_nothing_can_meet_still_finds_the_baking_stories(replayed):
    for db in he.DBS:
        assert set(shown(replayed, FULL, "h01", db)) == {"bread", "village_customs"}


def test_every_story_shown_by_the_full_pipeline_was_scored_strong_and_says_so(replayed):
    for qid in IDS:
        for db in he.DBS:
            rows = replayed.results[FULL][qid].rows[db]
            assert rows and all(r["rel"] >= 2 for r in rows), (qid, db)
            assert replayed.results[FULL][qid].judged[db] is True


def test_the_held_out_questions_do_as_well_as_the_rest(replayed):
    held = [q for q in QUESTIONS if q["id"] in ("h13", "h14", "h15")]
    table = he.scores(replayed.results, held)
    assert he.mean(table, FULL, "f1") >= 0.9 and he.mean(table, FULL, "f1") >= he.mean(table, PLAIN, "f1") + 0.3, he.format_table(table)


def test_the_confusion_matrix_of_the_recorded_runs(replayed):
    conf = he.confusion(replayed.results, QUESTIONS, STORIES)
    wanted = sum(len(q["relevant"]) for q in QUESTIONS)
    cells = sum(len(STORIES) - len(q["acceptable"]) for q in QUESTIONS)
    for mode in conf:
        for db in he.DBS:
            c = conf[mode][db]
            assert c["tp"] + c["fn"] == wanted and sum(c.values()) == cells, (mode, db, c)     # every pair counted once
    for db in he.DBS:
        assert conf[PLAIN][db]["fp"] > conf[PLAIN][db]["tp"]                                  # a plain search shows more wrong stories than right ones
        assert conf[FULL][db] == {"tp": wanted, "fn": 0, "fp": 0, "tn": cells - wanted}       # the default shows every right story and no wrong one
        assert conf[REWRITE][db]["fn"] <= conf[PLAIN][db]["fn"]                               # concept queries miss no more than a plain search


def test_the_readme_tables_are_the_recorded_results(replayed):
    """The tables in README.md are generated; if this fails run `python test/hard_report.py --write`."""
    readme = (he.HERE.parent / "README.md").read_text(encoding="utf-8")
    assert hard_report.block(readme) == he.markdown_report(replayed.results, QUESTIONS, STORIES).strip("\n")


def test_the_whole_run_needed_no_api_call(replayed):
    assert replayed.cassette.fetched == 0
