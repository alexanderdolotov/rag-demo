"""
The parts the OpenSearch and the Neo4j side share (retrieval.py): fusing rankings, collapsing versions to stories, running a chosen
subset of arms, and scoring. Pure logic, no services.
"""
import pytest

import retrieval


def row(topic, version, kind, score):
    return {"id": f"{topic}:{version}", "topic": topic, "version": version, "kind": kind, "score": score}


def test_rrf_rewards_documents_that_several_searches_agree_on():
    a = [{"id": "x", "topic": "x"}, {"id": "y", "topic": "y"}]
    b = [{"id": "z", "topic": "z"}, {"id": "y", "topic": "y"}]
    fused = retrieval.rrf([a, b])
    assert fused[0]["id"] == "y"                                      # in both lists
    assert fused[0]["score"] == pytest.approx(1 / 62 + 1 / 62)


def test_story_level_keeps_each_stories_best_version_and_its_score_as_it_is():
    fused = [row("a", "rw1", "rewrite", 0.030), row("b", "orig", "original", 0.025), row("a", "orig", "original", 0.010)]
    stories = retrieval.story_level(fused)
    assert [(s["topic"], s["score"], s["via"]) for s in stories] == [("a", 0.030, "rw1"), ("b", 0.025, "orig")]
    assert all(s["kind"] == "story" and s["id"] == s["topic"] for s in stories)


def test_a_rewrite_that_ranks_first_in_a_fused_list_is_not_pushed_below_originals_far_down_it():
    """The bug this guards: a multiplier on RRF scores (rewrites x0.6) turned a rank-1 rewrite into the equivalent of a rank-40 original."""
    fused = retrieval.rrf([[row("right", "rw1", "rewrite", 9.0)] + [row(f"other{i}", "orig", "original", 1.0) for i in range(9)]])
    assert [s["topic"] for s in retrieval.story_level(fused)][0] == "right"


def test_story_level_prefers_the_original_on_a_tie_and_keeps_the_ranking_order_otherwise():
    tie = retrieval.story_level([row("a", "rw1", "rewrite", 0.02), row("a", "orig", "original", 0.02)])
    assert tie[0]["via"] == "orig"
    assert [s["topic"] for s in retrieval.story_level([row("x", "orig", "original", 0.5), row("y", "orig", "original", 0.5)])] == ["x", "y"]


def test_first_relevant_rank():
    hits = [{"topic": "a"}, {"topic": "b"}, {"topic": "c"}]
    assert retrieval.first_relevant_rank(hits, ("c",)) == 3
    assert retrieval.first_relevant_rank(hits, ("b", "c")) == 2
    assert retrieval.first_relevant_rank(hits, ("zzz",)) is None


def test_summarize_counts_the_top_three_and_averages_reciprocal_ranks():
    assert retrieval.summarize([1, 3, 4, None]) == (2, pytest.approx((1 + 1 / 3 + 1 / 4) / 4))
    assert retrieval.summarize([2, 5], top=5)[0] == 2


def test_run_selected_runs_only_the_arms_asked_for_and_what_they_are_built_from():
    ran = []

    def arm(name, build=None):
        def run(get):
            ran.append(name)
            return build(get) if build else [name]
        return run

    builders = {"a": arm("a"), "b": arm("b"), "fused": arm("fused", lambda get: get("a") + get("b")), "unused": arm("unused")}
    assert retrieval.run_selected(builders, ["fused"]) == {"fused": ["a", "b"]} and sorted(ran) == ["a", "b", "fused"]   # not "unused"
    ran.clear()
    everything = retrieval.run_selected(builders)
    assert list(everything) == ["a", "b", "fused", "unused"] and ran.count("a") == 1      # an arm other arms need is run once


def test_run_selected_refuses_an_arm_that_does_not_exist_before_running_anything():
    ran = []
    with pytest.raises(ValueError, match="no such search arm: nope"):
        retrieval.run_selected({"a": lambda get: ran.append("a") or []}, ["a", "nope"])
    assert ran == []


def test_a_query_lists_its_tiers_most_important_first_and_skips_empty_ones():
    q = retrieval.Query(core=("bread", "bake"), context=("nearby",))
    assert q.tiers() == [("core", ("bread", "bake")), ("context", ("nearby",))]
    assert q.words() == "bread bake nearby" and q.words(("core", "related")) == "bread bake"


def test_the_vectors_embed_the_core_words_unless_told_otherwise():
    assert retrieval.Query(core=("bread", "bake"), related=("loaf",)).vector_text() == "bread bake"
    assert retrieval.Query(core=("bread",), embed="who bakes bread?").vector_text() == "who bakes bread?"


def test_the_tiers_count_less_the_less_they_matter():
    w = retrieval.TIER_WEIGHT
    assert w["core"] > w["related"] > w["context"] > 0
