"""
Tests for the pure transformation logic in mtbench.py.

Nothing here touches the network — `_fetch_split`/`build` are excluded on
purpose. These fixtures are shaped exactly like a row from the
datasets-server API (lmsys/mt_bench_human_judgments), just small.
"""

import json

from eval_pipeline.ingestion import load_comparisons
from eval_pipeline.mtbench import (
    HELD_OUT_PAIR,
    _comparison_record,
    _is_held_out_pair,
    _item_id,
    _policy_output_records,
    _preference_records,
    _prompt_and_context,
    _verdict_records,
)


def turn1_row(model_a="alpaca-13b", model_b="gpt-3.5-turbo", winner="model_b", judge="author_2"):
    return {
        "question_id": 81,
        "model_a": model_a,
        "model_b": model_b,
        "winner": winner,
        "judge": judge,
        "turn": 1,
        "conversation_a": [
            {"role": "user", "content": "write a haiku"},
            {"role": "assistant", "content": f"{model_a}'s haiku"},
        ],
        "conversation_b": [
            {"role": "user", "content": "write a haiku"},
            {"role": "assistant", "content": f"{model_b}'s haiku"},
        ],
    }


def turn2_row(model_a="alpaca-13b", model_b="gpt-3.5-turbo", winner="model_a", judge="author_3"):
    return {
        "question_id": 81,
        "model_a": model_a,
        "model_b": model_b,
        "winner": winner,
        "judge": judge,
        "turn": 2,
        "conversation_a": [
            {"role": "user", "content": "write a haiku"},
            {"role": "assistant", "content": f"{model_a}'s haiku"},
            {"role": "user", "content": "now about autumn"},
            {"role": "assistant", "content": f"{model_a}'s autumn haiku"},
        ],
        "conversation_b": [
            {"role": "user", "content": "write a haiku"},
            {"role": "assistant", "content": f"{model_b}'s haiku"},
            {"role": "user", "content": "now about autumn"},
            {"role": "assistant", "content": f"{model_b}'s autumn haiku"},
        ],
    }


def test_prompt_and_context_splits_the_current_turn_from_history():
    prompt, response, context = _prompt_and_context(turn2_row()["conversation_a"])
    assert prompt == "now about autumn"
    assert response == "alpaca-13b's autumn haiku"
    assert context == (("user", "write a haiku"), ("assistant", "alpaca-13b's haiku"))


def test_prompt_and_context_on_turn_one_has_no_prior_context():
    prompt, response, context = _prompt_and_context(turn1_row()["conversation_a"])
    assert prompt == "write a haiku"
    assert response == "alpaca-13b's haiku"
    assert context == ()


def test_item_id_format():
    assert _item_id(turn1_row()) == "q81-t1-alpaca-13b-vs-gpt-3.5-turbo"


def test_is_held_out_pair_matches_either_slot_order():
    a, b = sorted(HELD_OUT_PAIR)
    assert _is_held_out_pair(turn1_row(model_a=a, model_b=b))
    assert _is_held_out_pair(turn1_row(model_a=b, model_b=a))
    assert not _is_held_out_pair(turn1_row(model_a="alpaca-13b", model_b="claude-v1"))


def test_preference_records_excludes_the_held_out_pair():
    a, b = sorted(HELD_OUT_PAIR)
    rows = [turn1_row(), turn1_row(model_a=a, model_b=b)]
    records = list(_preference_records(rows))
    assert len(records) == 1
    assert records[0]["system_a"] == "alpaca-13b"


def test_preference_records_map_winner_and_carry_the_judge_as_annotator():
    record = next(_preference_records([turn1_row(winner="model_b", judge="author_2")]))
    assert record["preferred"] == "B"
    assert record["annotator_id"] == "author_2"


def test_verdict_records_flag_inconsistent_as_a_position_unstable_tie():
    record = next(_verdict_records([turn1_row(winner="tie (inconsistent)")]))
    assert record["winner"] == "tie"
    assert record["position_unstable"] is True


def test_verdict_records_mark_decisive_verdicts_as_stable():
    record = next(_verdict_records([turn1_row(winner="model_a")]))
    assert record["winner"] == "A"
    assert record["position_unstable"] is False


def test_verdict_records_include_the_held_out_pair():
    a, b = sorted(HELD_OUT_PAIR)
    records = list(_verdict_records([turn1_row(model_a=a, model_b=b)]))
    assert len(records) == 1


def test_comparison_record_normalises_slot_order_for_any_pair():
    # Not the held-out pair — the fix has to be general, since any pair can
    # appear in either slot order in the raw data.
    forward = _comparison_record(turn1_row(model_a="alpaca-13b", model_b="claude-v1"))
    reversed_ = _comparison_record(turn1_row(model_a="claude-v1", model_b="alpaca-13b"))

    assert forward["system_a"] == "alpaca-13b" and forward["system_b"] == "claude-v1"
    assert reversed_["system_a"] == "alpaca-13b" and reversed_["system_b"] == "claude-v1"
    assert forward["id"] == reversed_["id"]
    assert forward["response_a"] == reversed_["response_a"]
    assert forward["response_b"] == reversed_["response_b"]


def test_preference_records_flip_the_label_when_canonicalising_slot_order():
    # "claude-v1" is model_a in the raw row and wins as "model_a", but it is
    # not alphabetically first — after canonicalising to
    # alpaca-13b/claude-v1, that same win must read as "B", not "A".
    record = next(_preference_records([
        turn1_row(model_a="claude-v1", model_b="alpaca-13b", winner="model_a"),
    ]))
    assert record["system_a"] == "alpaca-13b"
    assert record["preferred"] == "B"


def test_verdict_records_flip_the_label_when_canonicalising_slot_order():
    record = next(_verdict_records([
        turn1_row(model_a="claude-v1", model_b="alpaca-13b", winner="model_b"),
    ]))
    assert record["system_a"] == "alpaca-13b"
    assert record["winner"] == "A"


def test_reversed_slot_order_annotations_still_merge_through_ingestion(tmp_path):
    """
    Two annotators labelled the same item, but MT-Bench recorded one of them
    with the slots reversed. Before canonicalising at the row level, this
    silently split into two single-annotator comparisons instead of merging
    into one two-annotator comparison — undercounting repeat-labelled items.
    """
    rows = [
        turn1_row(model_a="alpaca-13b", model_b="claude-v1", winner="model_a", judge="author_2"),
        turn1_row(model_a="claude-v1", model_b="alpaca-13b", winner="model_b", judge="author_3"),
    ]
    path = tmp_path / "preferences.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in _preference_records(rows)), encoding="utf-8"
    )

    comparisons, report = load_comparisons(path)
    assert len(comparisons) == 1
    assert report.n_annotations == 2
    # Both annotators actually preferred alpaca-13b — the second row's
    # "model_a wins" is alpaca-13b too, once its reversed slots are undone.
    assert {a.preferred for a in comparisons[0].annotations} == {"A"}


def test_policy_output_records_dedupe_across_annotators_and_slot_order():
    a, b = sorted(HELD_OUT_PAIR)
    rows = [
        turn1_row(model_a=a, model_b=b, judge="author_2"),
        turn1_row(model_a=b, model_b=a, judge="author_3"),  # same item, reversed slots
        turn2_row(model_a=a, model_b=b, judge="author_2"),
    ]
    records = list(_policy_output_records(rows))
    assert len(records) == 2  # one per (question, turn), not one per row
    assert all("preferred" not in r and "annotations" not in r for r in records)


def test_multiple_human_annotations_of_one_item_merge_through_ingestion(tmp_path):
    """The adapter's output must actually round-trip through load_comparisons."""
    rows = [turn1_row(judge="author_2", winner="model_b"), turn1_row(judge="author_3", winner="tie")]
    path = tmp_path / "preferences.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in _preference_records(rows)), encoding="utf-8"
    )

    comparisons, report = load_comparisons(path)
    assert len(comparisons) == 1
    assert report.n_annotations == 2
    assert {a.preferred for a in comparisons[0].annotations} == {"B", "tie"}
