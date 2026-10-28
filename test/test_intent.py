"""
The LLM's analysis of a question and its relevance scores (intent.py): what is accepted, what degrades to searching the question as typed.
Pure logic, no services and no LLM.
"""
import json

import pytest

import intent as it

QUESTION = "who is baking breal around here?"


def reply(**parts):
    body = {"concepts": [{"name": "baking bread", "core": ["bread", "bake"], "related": ["loaf", "baker", "bread"]}],
            "must": ["a person baking"], "context": ["nearby", "Loaf"], "unverifiable": ["near the user"]}
    body.update(parts)
    return json.dumps(body)


def test_a_good_analysis_is_read_into_concepts_conditions_context_and_unverifiable_things():
    got = it.parse_intent("```json\n" + reply() + "\n```", QUESTION)
    assert got.understood and got.concepts == (it.Concept("baking bread", ("bread", "bake"), ("loaf", "baker")),)
    assert got.must == ("a person baking",) and got.context == ("nearby",) and got.unverifiable == ("near the user",)


def test_a_word_is_kept_in_the_most_important_tier_it_appears_in():
    got = it.parse_intent(reply(), QUESTION)
    assert "bread" not in got.concepts[0].related and "loaf" not in got.context       # core beat related, related beat context


def test_concepts_keep_the_order_the_llm_gave_them_most_important_first():
    two = {"concepts": [{"name": "second thought", "core": ["b"]}, {"name": "first thought", "core": ["a"]}]}
    assert [c.name for c in it.parse_intent(json.dumps(two), QUESTION).concepts] == ["second thought", "first thought"]


def test_several_concepts_are_kept_up_to_a_limit_and_one_without_core_words_is_dropped():
    many = {"concepts": [{"name": str(i), "core": [f"w{i}"]} for i in range(6)] + [{"name": "empty", "core": []}]}
    got = it.parse_intent(json.dumps(many), QUESTION)
    assert [c.name for c in got.concepts] == ["0", "1", "2"] and len(got.concepts) == it.MAX_CONCEPTS
    only_empty = it.parse_intent(json.dumps({"concepts": [{"name": "x", "core": []}]}), QUESTION)
    assert not only_empty.understood


@pytest.mark.parametrize("bad", ["", "sorry, I cannot do that", "[LLM error: HTTP 500]", "{not json}", "[1, 2]", '{"concepts": "bread"}', '{"concepts": []}'])
def test_a_reply_that_cannot_be_read_means_the_question_is_searched_as_typed(bad):
    got = it.parse_intent(bad, QUESTION)
    assert not got.understood and got.concepts == (it.Concept(QUESTION, (QUESTION,)),) and got.must == got.context == got.unverifiable == ()


def test_wrongly_typed_fields_are_ignored_rather_than_trusted():
    got = it.parse_intent(json.dumps({"concepts": [{"name": 5, "core": ["bread", 7, None, "  Bake  "], "related": "loaf"}], "context": "x", "must": [3],
                                      "unverifiable": {"a": 1}}), QUESTION)
    assert got.concepts == (it.Concept("bread bake", ("bread", "bake")),) and got.must == got.context == got.unverifiable == ()


def test_the_scoring_prompt_names_the_subject_lists_the_conditions_and_labels_passages_by_id():
    got = it.parse_intent(reply(), QUESTION)
    prompt = it.score_prompt(QUESTION, got, got.concepts[0], [("bread", "First passage."), ("village_customs", "Second passage.")])
    assert "Question: who is baking breal around here?" in prompt
    assert "baking bread (core words: bread, bake; close variants: loaf, baker)" in prompt
    assert "Conditions a passage must meet: a person baking" in prompt and "Ignore these, a story cannot show them: near the user" in prompt
    assert prompt.endswith("Passages:\n[bread] First passage.\n[village_customs] Second passage.")


def test_with_several_subjects_each_is_told_about_the_others_and_that_the_conditions_apply_only_where_they_belong():
    two = it.Intent((it.Concept("stock boost", ("shares",)), it.Concept("weak demand", ("demand",))), must=("shares rose", "demand fell"))
    first = it.score_prompt("q", two, two.concepts[0], [("a", "text")])
    assert "other subjects too, judged separately and not counted here: weak demand" in first
    assert "each one applies only to the part of the question it is about" in first and "shares rose; demand fell" in first
    assert "other subjects too" in it.score_prompt("q", two, two.concepts[1], [("a", "text")]) and "stock boost" in it.score_prompt("q", two, two.concepts[1], [("a", "text")])
    one = it.score_prompt("q", it.Intent((it.Concept("x", ("x",)),), must=("m",)), it.Concept("x", ("x",)), [("a", "t")])
    assert "other subjects" not in one and "Conditions a passage must meet: m" in one            # a single subject reads as before


def test_a_condition_line_is_left_out_when_there_is_nothing_to_say():
    plain = it.score_prompt("q", it.flat_intent("q"), it.flat_intent("q").concepts[0], [("a", "text")])
    assert "Conditions" not in plain and "Ignore these" not in plain


@pytest.mark.parametrize("text, ids, expected", [
    ('{"bread": 3, "dog": 0}', ["bread", "dog"], {"bread": 3, "dog": 0}),
    ('Here you go:\n```json\n{"bread": 1}\n```', ["bread"], {"bread": 1}),
    ('{"bread": 3, "dog": 0, "extra": 2}', ["bread", "dog"], {"bread": 3, "dog": 0}),     # an id nobody asked about is ignored
    ('{"bread": 3, "dog": 0}', ["bread", "dog", "frog"], None),                           # a story left out
    ('{"bread": 4}', ["bread"], None),                                                    # out of range
    ('{"bread": 2.5}', ["bread"], None),                                                  # not an integer
    ('{"bread": true}', ["bread"], None),
    ('{"bread": "3"}', ["bread"], None),
    ("[3, 0]", ["bread", "dog"], None),                                                   # a list is not what was asked for
    ("[LLM error: boom]", ["bread"], None),
    ("no object here", ["bread"], None),
])
def test_scores_are_accepted_only_when_every_story_has_an_integer_from_0_to_3_by_its_id(text, ids, expected):
    assert it.parse_scores(text, ids) == expected
