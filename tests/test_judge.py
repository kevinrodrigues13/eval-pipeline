import threading
import time

import pytest

from eval_pipeline.judge import (
    JudgeParseError,
    JudgedComparison,
    LLMJudge,
    PrecomputedJudge,
    Verdict,
    judge_many,
    judge_with_position_check,
    parse_verdict,
)
from eval_pipeline.schema import Comparison


def comparison(row_id="c0", response_a="answer a", response_b="answer b"):
    return Comparison(row_id=row_id, prompt="q", response_a=response_a, response_b=response_b)


# --- parse_verdict -----------------------------------------------------

def test_parse_verdict_reads_the_bracketed_letter():
    assert parse_verdict("some reasoning... [[A]]") == "A"
    assert parse_verdict("[[B]] because...") == "B"
    assert parse_verdict("they are equal [[C]]") == "tie"


def test_parse_verdict_last_match_wins():
    text = "leaning towards [[A]] but reconsidering... final answer: [[B]]"
    assert parse_verdict(text) == "B"


def test_parse_verdict_raises_when_no_tag_present():
    with pytest.raises(JudgeParseError):
        parse_verdict("I couldn't decide.")


# --- LLMJudge + position-swap defense -----------------------------------

def test_llm_judge_calls_the_backend_and_parses_the_reply():
    def backend(system_prompt, user_prompt):
        assert "impartial judge" in system_prompt
        assert "answer a" in user_prompt
        return "reasoning... [[A]]"

    judge = LLMJudge(backend)
    verdict = judge.verdict(comparison())
    assert verdict.winner == "A"
    assert verdict.position_unstable is None  # LLMJudge alone doesn't know this


def test_position_check_is_consistent_when_the_swapped_call_agrees():
    # A judge that genuinely tracks content, not slot: whichever slot holds
    # "answer a" wins, both before and after the swap. Consistent by
    # construction, so this exercises the "no instability" path honestly.
    def backend(system_prompt, user_prompt):
        a_section = user_prompt.split("Assistant A's Answer]\n")[1].split(
            "\n[The End of Assistant A's Answer]"
        )[0]
        return "[[A]]" if "answer a" in a_section else "[[B]]"

    judge = LLMJudge(backend)
    result = judge_with_position_check(judge, comparison())
    assert isinstance(result, JudgedComparison)
    assert result.winner == "A"
    assert result.position_unstable is False
    assert len(result.raw) == 2


def test_position_check_flags_instability_and_reports_a_tie():
    # A judge that always says "[[A]]" regardless of content is not
    # order-invariant: it will pick the opposite underlying response after
    # a swap, which is exactly the contradiction this defense exists to catch.
    judge = LLMJudge(lambda system_prompt, user_prompt: "[[A]]")
    result = judge_with_position_check(judge, comparison())
    assert result.winner == "tie"
    assert result.position_unstable is True


def test_order_invariant_judge_is_not_asked_twice():
    calls = []

    def backend(system_prompt, user_prompt):
        calls.append(user_prompt)
        return "[[A]]"

    judge = LLMJudge(backend)
    judge.is_order_invariant = True  # simulate a hypothetical order-invariant mechanism
    judge_with_position_check(judge, comparison())
    assert len(calls) == 1


# --- PrecomputedJudge ----------------------------------------------------

def test_precomputed_judge_replays_a_recorded_verdict():
    judge = PrecomputedJudge.from_records([
        {"id": "c0", "winner": "A", "position_unstable": False},
    ])
    result = judge_with_position_check(judge, comparison("c0"))
    assert result.winner == "A"
    assert result.position_unstable is False
    assert len(result.raw) == 1  # order-invariant: never asked a second time


def test_precomputed_judge_carries_forward_recorded_instability():
    judge = PrecomputedJudge.from_records([
        {"id": "c0", "winner": "tie", "position_unstable": True},
    ])
    result = judge_with_position_check(judge, comparison("c0"))
    assert result.winner == "tie"
    assert result.position_unstable is True


def test_precomputed_judge_raises_on_unrecorded_comparison():
    judge = PrecomputedJudge.from_records([{"id": "c0", "winner": "A"}])
    with pytest.raises(KeyError):
        judge.verdict(comparison("not-recorded"))


def test_precomputed_judge_strict_mode_raises_on_conflicting_records():
    with pytest.raises(ValueError):
        PrecomputedJudge.from_records([
            {"id": "c0", "winner": "A"},
            {"id": "c0", "winner": "B"},
        ], strict=True)


def test_precomputed_judge_lenient_mode_flags_conflicts_as_a_tie():
    judge = PrecomputedJudge.from_records([
        {"id": "c0", "winner": "A"},
        {"id": "c0", "winner": "B"},
    ], strict=False)
    assert judge.conflicts == 1
    result = judge_with_position_check(judge, comparison("c0"))
    assert result.winner == "tie"
    assert result.position_unstable is True


# --- judge_many: concurrency ------------------------------------------------

def test_judge_many_default_is_sequential_and_matches_direct_calls():
    rows = [comparison(f"c{i}") for i in range(5)]
    judge = PrecomputedJudge.from_records([{"id": f"c{i}", "winner": "A"} for i in range(5)])

    via_judge_many = judge_many(judge, rows)
    direct = [judge_with_position_check(judge, r) for r in rows]
    assert via_judge_many == direct


def test_judge_many_preserves_input_order_despite_out_of_order_completion():
    # Row 0 sleeps longest, so if judge_many returned results in completion
    # order rather than input order, row 0 would land last instead of first.
    delays = {"c0": 0.05, "c1": 0.01, "c2": 0.03}

    def backend(system_prompt, user_prompt):
        row_id = user_prompt.split("[User Question]\n")[1].split("\n")[0]
        time.sleep(delays.get(row_id, 0))
        return "[[A]]"

    # Encode the row id in the prompt itself so the fake backend can key off it.
    rows = [Comparison(row_id=rid, prompt=rid, response_a="x", response_b="y") for rid in ["c0", "c1", "c2"]]

    judge = LLMJudge(backend)
    results = judge_many(judge, rows, max_workers=3)
    assert [r.row_id for r in results] == ["c0", "c1", "c2"]


def test_judge_many_gives_identical_results_at_any_worker_count():
    rows = [comparison(f"c{i}", response_a="the real answer", response_b="a distractor") for i in range(8)]

    def backend(system_prompt, user_prompt):
        a_section = user_prompt.split("Assistant A's Answer]\n")[1].split(
            "\n[The End of Assistant A's Answer]"
        )[0]
        return "[[A]]" if "real answer" in a_section else "[[B]]"

    sequential = judge_many(LLMJudge(backend), rows, max_workers=1)
    concurrent_ = judge_many(LLMJudge(backend), rows, max_workers=4)
    assert sequential == concurrent_


def test_judge_many_actually_overlaps_calls_when_concurrent():
    lock = threading.Lock()
    state = {"current": 0, "max_seen": 0}

    def backend(system_prompt, user_prompt):
        with lock:
            state["current"] += 1
            state["max_seen"] = max(state["max_seen"], state["current"])
        time.sleep(0.05)
        with lock:
            state["current"] -= 1
        return "[[A]]"

    rows = [comparison(f"c{i}") for i in range(6)]
    judge_many(LLMJudge(backend), rows, max_workers=4)
    # With 6 rows x 2 calls each (position swap) across 4 workers, at least
    # some calls must have genuinely overlapped for this to be true.
    assert state["max_seen"] > 1


# --- skip_parse_errors -----------------------------------------------------

def test_without_skip_parse_errors_a_malformed_reply_raises():
    judge = LLMJudge(lambda system_prompt, user_prompt: "no tag here")
    with pytest.raises(JudgeParseError):
        judge_with_position_check(judge, comparison())


def test_skip_parse_errors_excludes_instead_of_raising_on_first_call():
    judge = LLMJudge(lambda system_prompt, user_prompt: "no tag here")
    result = judge_with_position_check(judge, comparison(), skip_parse_errors=True)
    assert result.parse_failed is True
    assert result.position_unstable is False


def test_skip_parse_errors_excludes_when_only_the_swapped_call_fails():
    calls = {"n": 0}

    def backend(system_prompt, user_prompt):
        calls["n"] += 1
        return "[[A]]" if calls["n"] == 1 else "no tag on the second call"

    judge = LLMJudge(backend)
    result = judge_with_position_check(judge, comparison(), skip_parse_errors=True)
    assert result.parse_failed is True


def test_skip_parse_errors_does_not_affect_a_clean_run():
    # Order-invariant by construction (PrecomputedJudge), so no position-swap
    # complexity to get right here — this test is only about the
    # skip_parse_errors flag being a no-op on a run with nothing to skip.
    judge = PrecomputedJudge.from_records([{"id": "c0", "winner": "A"}])
    result = judge_with_position_check(judge, comparison("c0"), skip_parse_errors=True)
    assert result.parse_failed is False
    assert result.winner == "A"


def test_judge_many_with_skip_parse_errors_isolates_the_bad_row():
    def backend(system_prompt, user_prompt):
        return "no tag" if "c1" in user_prompt else "[[A]]"

    rows = [Comparison(row_id=rid, prompt=rid, response_a="x", response_b="y") for rid in ["c0", "c1", "c2"]]
    results = judge_many(LLMJudge(backend), rows, skip_parse_errors=True)
    assert [r.parse_failed for r in results] == [False, True, False]


def test_judge_many_without_skip_parse_errors_raises_on_the_first_bad_row():
    def backend(system_prompt, user_prompt):
        return "no tag" if "c1" in user_prompt else "[[A]]"

    rows = [Comparison(row_id=rid, prompt=rid, response_a="x", response_b="y") for rid in ["c0", "c1", "c2"]]
    with pytest.raises(JudgeParseError):
        judge_many(LLMJudge(backend), rows)
