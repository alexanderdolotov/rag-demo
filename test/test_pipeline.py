"""
The one-shot search with concept priority (pipeline.py): what is searched, how candidates are scored, what is shown and in which order.
No services: the databases and the LLM are replaced by scripts, so every rule can be tried with exactly the results it needs.
"""
import json
import re

import pytest

import intent as it
import pipeline as pl
from retrieval import Query

DBS = ("A", "B")


def row(topic, **extra):
    return {"topic": topic, "text": f"text of {topic}", "via": "orig", "score": 1.0, "native": 1, **extra}


class Search:
    """Returns the same stories for any query of a concept, found by its core word; remembers every query."""

    def __init__(self, results):
        self.results, self.calls = results, []

    def __call__(self, db, query):
        self.calls.append((db, query))
        key = query.core[0] if isinstance(query, Query) else query
        return [row(t) for t in self.results.get((db, key), self.results.get(key, []))]


class Scorer:
    """An LLM that scores stories by name, per concept; remembers the ids it was asked about, call by call."""

    def __init__(self, tables, reply=None):
        self.tables, self.reply, self.asked, self.prompts = tables, reply, [], []

    def __call__(self, system, user):
        assert system == it.SCORE_SYSTEM
        ids = re.findall(r"^\[(\w+)\] text of", user, re.M)
        self.asked.append(ids)
        self.prompts.append(user)
        if self.reply is not None:
            return self.reply
        subject = re.search(r"^Subject to judge the passages against: (.*?) \(core words", user, re.M).group(1)
        table = self.tables[subject]
        return json.dumps({i: table.get(i, 0) for i in ids})


def concept(name="bread", core=("bread",), related=("loaf",)):
    return it.Concept(name, core, related)


def intent_of(*concepts, must=(), context=("nearby",), unverifiable=()):
    return it.Intent(tuple(concepts) or (concept(),), must, context, unverifiable)


def shown(rows):
    return [r["topic"] for r in rows]


# --------------------------------------------------------------------------- what is searched
def test_every_concept_is_searched_in_every_database_with_its_own_tiered_query():
    search = Search({})
    two = intent_of(concept("bread"), concept("duties", ("duties",), ("levies",)), context=("nearby",))
    pl.search_all("q", two, DBS, search)
    assert [(db, q) for db, q in search.calls] == [
        ("A", Query(("bread",), ("loaf",), ("nearby",))), ("A", Query(("duties",), ("levies",), ("nearby",))),
        ("B", Query(("bread",), ("loaf",), ("nearby",))), ("B", Query(("duties",), ("levies",), ("nearby",)))]


def test_an_analysis_that_was_not_understood_means_one_search_with_the_question_as_typed():
    search = Search({})
    found = pl.search_all("who bakes?", it.flat_intent("who bakes?"), DBS, search)
    assert search.calls == [("A", "who bakes?"), ("B", "who bakes?")] and set(found) == set(DBS)


# --------------------------------------------------------------------------- scoring
def test_the_judge_sees_the_candidates_by_name_not_in_the_order_the_database_ranked_them():
    llm = Scorer({"bread": {"bread": 3}})
    assert pl.score(llm, "q", intent_of(), concept(), [row("zebra"), row("bread"), row("apple")]) == {"apple": 0, "bread": 3, "zebra": 0}
    assert llm.asked == [["apple", "bread", "zebra"]]


def test_one_scoring_call_per_concept_over_every_story_either_database_found_for_any_concept():
    found = {"A": [[row("bread"), row("dog")], [row("tariff")]], "B": [[row("dog"), row("customs")], [row("tariff"), row("bread")]]}
    llm = Scorer({"bread": {"bread": 3, "customs": 2}, "duties": {"tariff": 3}})
    got = pl.score_pool(llm, "q", intent_of(concept("bread"), concept("duties", ("duties",), ())), found)
    assert len(llm.asked) == 2 and all(sorted(ids) == ["bread", "customs", "dog", "tariff"] for ids in llm.asked)        # each story once per concept
    assert got == [{"bread": 3, "customs": 2, "dog": 0, "tariff": 0}, {"bread": 0, "customs": 0, "dog": 0, "tariff": 3}]


def test_scores_that_cannot_be_read_mean_none_and_nothing_found_means_no_call():
    found = {"A": [[row("bread")]], "B": [[row("bread")]]}
    assert pl.score_pool(Scorer({}, reply="sorry, no"), "q", intent_of(), found) is None
    llm = Scorer({})
    assert pl.score_pool(llm, "q", intent_of(), {"A": [[]], "B": [[]]}) == [{}] and llm.asked == []


def test_the_conditions_reach_the_scoring_prompt():
    llm = Scorer({"bread": {}})
    pl.score(llm, "q", intent_of(must=("benefited, not hurt",), unverifiable=("near the user",)), concept(), [row("bread")])
    assert "Conditions a passage must meet: benefited, not hurt" in llm.prompts[0] and "Ignore these, a story cannot show them: near the user" in llm.prompts[0]


# --------------------------------------------------------------------------- what is shown
def test_only_strong_candidates_are_shown_best_score_first_and_the_rest_are_dropped():
    found = [[row("weak"), row("two"), row("three"), row("none"), row("two_b")]]
    out = pl.rank([concept()], found, [{"weak": 1, "two": 2, "three": 3, "none": 0, "two_b": 2}], top=5)
    assert shown(out) == ["three", "two", "two_b"]                                  # equal scores keep the order the database gave
    assert [r["rel"] for r in out] == [3, 2, 2] and all("concept" not in r for r in out)


def test_when_nothing_is_strong_the_weak_matches_are_shown_rather_than_nothing_and_unrelated_ones_never():
    found = [[row("side"), row("unrelated")]]
    assert shown(pl.rank([concept()], found, [{"side": 1, "unrelated": 0}], top=5)) == ["side"]
    assert pl.rank([concept()], found, [{"side": 0, "unrelated": 0}], top=5) == []


def test_at_most_top_rows_are_returned():
    found = [[row(f"s{i}") for i in range(8)]]
    assert shown(pl.rank([concept()], found, [{f"s{i}": 3 for i in range(8)}], top=3)) == ["s0", "s1", "s2"]


def test_the_concepts_take_turns_the_most_important_first_and_each_row_says_which_concept_it_answers():
    two = [concept("bread"), concept("duties", ("duties",), ())]
    found = [[row("bread"), row("customs")], [row("tariff"), row("bread"), row("margin")]]
    scores = [{"bread": 3, "customs": 3, "tariff": 0, "margin": 0}, {"bread": 0, "customs": 0, "tariff": 3, "margin": 2}]
    out = pl.rank(two, found, scores, top=4)
    assert shown(out) == ["bread", "tariff", "customs", "margin"]
    assert [r["concept"] for r in out] == ["bread", "duties", "bread", "duties"]


def test_a_second_concept_is_not_crowded_out_by_a_first_with_many_answers():
    two = [concept("many"), concept("few", ("few",), ())]
    found = [[row(f"m{i}") for i in range(6)], [row("only")]]
    scores = [{**{f"m{i}": 3 for i in range(6)}, "only": 0}, {**{f"m{i}": 0 for i in range(6)}, "only": 3}]
    assert shown(pl.rank(two, found, scores, top=3)) == ["m0", "only", "m1"]


def test_a_story_that_is_strong_for_two_concepts_is_shown_once():
    two = [concept("a"), concept("b", ("b",), ())]
    out = pl.rank(two, [[row("both")], [row("both"), row("b_only")]], [{"both": 3, "b_only": 0}, {"both": 3, "b_only": 3}], top=5)
    assert shown(out) == ["both", "b_only"] and out[0]["concept"] == "a"


def test_a_story_scored_strong_for_a_concept_whose_own_search_did_not_find_it_still_counts():
    two = [concept("a"), concept("b", ("b",), ())]
    out = pl.rank(two, [[row("x")], [row("y")]], [{"x": 0, "y": 3}, {"x": 0, "y": 0}], top=5)       # y was found by concept b's search, but fits concept a
    assert shown(out) == ["y"] and out[0]["concept"] == "a"


def test_without_scores_the_databases_own_order_is_kept_with_the_concepts_still_taking_turns():
    two = [concept("a"), concept("b", ("b",), ())]
    out = pl.rank(two, [[row("a1"), row("a2")], [row("b1")]], None, top=5)
    assert shown(out) == ["a1", "b1", "a2"] and [r["concept"] for r in out] == ["a", "b", "a"] and all("rel" not in r for r in out)
    assert all("concept" not in r for r in pl.rank([concept()], [[row("x")]], None, top=5))


def test_union_rows_alternates_skips_repeats_and_stops_at_top():
    a = [row("a1"), row("a2"), row("shared"), row("a4")]
    b = [row("shared"), row("b2"), row("b3")]
    assert shown(pl.union_rows([a, b], 10)) == ["a1", "shared", "a2", "b2", "a4", "b3"]
    assert shown(pl.union_rows([a, b], 3)) == ["a1", "shared", "a2"]
    assert shown(pl.union_rows([[], b], 2))[0] == "shared" and pl.union_rows([[], []], 3) == []
