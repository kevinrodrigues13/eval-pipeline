import pytest

from eval_pipeline.judge import LLMJudge, PrecomputedJudge
from eval_pipeline.schema import Comparison
from eval_pipeline.scoring import compare_policies


def row(row_id, cluster_id=None):
    return Comparison(
        row_id=row_id, prompt="q", response_a="a", response_b="b",
        cluster_id=cluster_id or row_id,
    )


def test_wins_ties_and_losses_are_counted_and_ties_count_as_half():
    rows = [row("c0"), row("c1"), row("c2"), row("c3")]
    judge = PrecomputedJudge.from_records([
        {"id": "c0", "winner": "A"}, {"id": "c1", "winner": "A"},
        {"id": "c2", "winner": "B"}, {"id": "c3", "winner": "tie"},
    ])
    result = compare_policies(judge, rows)

    assert result.n == 4
    assert result.n_scored == 4
    assert result.n_wins_1 == 2 and result.n_wins_2 == 1 and result.n_ties == 1
    assert result.win_rate_policy1 == pytest.approx((2 + 0.5) / 4)


def test_position_unstable_comparisons_are_excluded_not_averaged_in():
    rows = [row("c0"), row("c1")]
    judge = PrecomputedJudge.from_records([
        {"id": "c0", "winner": "A", "position_unstable": False},
        {"id": "c1", "winner": "tie", "position_unstable": True},
    ])
    result = compare_policies(judge, rows)

    assert result.n == 2
    assert result.n_scored == 1  # c1 excluded, not counted as a tie
    assert result.n_position_unstable == 1
    assert result.n_wins_1 == 1
    assert result.n_ties == 0
    assert result.win_rate_policy1 == pytest.approx(1.0)
    assert result.scored_row_ids == ("c0",)


def test_all_unstable_gives_no_win_rate():
    rows = [row("c0")]
    judge = PrecomputedJudge.from_records([{"id": "c0", "winner": "A", "position_unstable": True}])
    result = compare_policies(judge, rows)

    assert result.win_rate_policy1 is None
    assert result.n_scored == 0
    assert result.is_underpowered() is True  # no data can't be anything but "can't tell"


def test_decisive_rate_excludes_ties_from_the_numerator():
    rows = [row("c0"), row("c1"), row("c2")]
    judge = PrecomputedJudge.from_records([
        {"id": "c0", "winner": "A"}, {"id": "c1", "winner": "B"}, {"id": "c2", "winner": "tie"},
    ])
    result = compare_policies(judge, rows)
    assert result.decisive_rate == pytest.approx(2 / 3)


def test_position_instability_rate_is_over_all_attempted_not_just_scored():
    rows = [row("c0"), row("c1"), row("c2"), row("c3")]
    judge = PrecomputedJudge.from_records([
        {"id": "c0", "winner": "A"}, {"id": "c1", "winner": "A"},
        {"id": "c2", "winner": "A"}, {"id": "c3", "winner": "tie", "position_unstable": True},
    ])
    result = compare_policies(judge, rows)
    assert result.position_instability_rate == pytest.approx(0.25)


def test_a_small_sample_near_fifty_fifty_is_reported_as_underpowered():
    # 6 comparisons, a bare 4-2 split: nowhere near enough to detect
    # anything at this sample size.
    rows = [row(f"c{i}") for i in range(6)]
    judge = PrecomputedJudge.from_records(
        [{"id": f"c{i}", "winner": "A"} for i in range(4)]
        + [{"id": f"c{i}", "winner": "B"} for i in range(4, 6)]
    )
    result = compare_policies(judge, rows)
    assert result.is_underpowered() is True


def test_a_large_clear_gap_is_not_underpowered():
    rows = [row(f"c{i}") for i in range(200)]
    judge = PrecomputedJudge.from_records(
        [{"id": f"c{i}", "winner": "A"} for i in range(160)]
        + [{"id": f"c{i}", "winner": "B"} for i in range(160, 200)]
    )
    result = compare_policies(judge, rows)
    assert result.win_rate_policy1 == pytest.approx(0.8)
    assert result.is_underpowered() is False


def test_confidence_interval_is_present_and_reasonable():
    rows = [row(f"c{i}") for i in range(100)]
    judge = PrecomputedJudge.from_records(
        [{"id": f"c{i}", "winner": "A"} for i in range(70)]
        + [{"id": f"c{i}", "winner": "B"} for i in range(70, 100)]
    )
    result = compare_policies(judge, rows)
    assert result.ci_lower < result.win_rate_policy1 < result.ci_upper
    assert result.ci_lower > 0.5  # clearly favours policy 1


def test_clustered_rows_are_resampled_as_a_unit():
    """
    2 big clusters (one all-A, one all-B) versus the same overall balance
    spread across 50 independent singleton clusters must produce a much
    wider interval: clustering collapses correlated evidence down to only 2
    effectively-independent units for the bootstrap to resample from,
    versus 50.
    """
    few_big_clusters = (
        [row(f"a{i}", cluster_id="c0") for i in range(25)]
        + [row(f"b{i}", cluster_id="c1") for i in range(25)]
    )
    many_small_clusters = [row(f"c{i}", cluster_id=f"c{i}") for i in range(50)]

    def judge_for(rows, winners):
        return PrecomputedJudge.from_records(
            [{"id": r.row_id, "winner": w} for r, w in zip(rows, winners)]
        )

    few_winners = ["A"] * 25 + ["B"] * 25
    many_winners = ["A" if i % 2 == 0 else "B" for i in range(50)]

    few_result = compare_policies(judge_for(few_big_clusters, few_winners), few_big_clusters)
    many_result = compare_policies(judge_for(many_small_clusters, many_winners), many_small_clusters)

    few_width = few_result.ci_upper - few_result.ci_lower
    many_width = many_result.ci_upper - many_result.ci_lower
    assert few_width > many_width


def test_compare_policies_gives_identical_results_at_any_worker_count():
    rows = [row(f"c{i}") for i in range(10)]
    judge = PrecomputedJudge.from_records(
        [{"id": f"c{i}", "winner": "A" if i % 2 == 0 else "B"} for i in range(10)]
    )

    sequential = compare_policies(judge, rows, max_workers=1)
    concurrent_ = compare_policies(judge, rows, max_workers=4)
    assert sequential.win_rate_policy1 == concurrent_.win_rate_policy1
    assert sequential.n_wins_1 == concurrent_.n_wins_1
    assert sequential.scored_row_ids == concurrent_.scored_row_ids


def _section(user_prompt, label):
    start = f"[The Start of Assistant {label}'s Answer]\n"
    end = f"\n[The End of Assistant {label}'s Answer]"
    return user_prompt.split(start)[1].split(end)[0]


def _content_tracking_backend(fail_marker):
    """A judge that genuinely tracks content (stable under a position swap),
    except it forgets to include a verdict tag whenever `fail_marker` shows
    up on either side."""
    def backend(system_prompt, user_prompt):
        a_section, b_section = _section(user_prompt, "A"), _section(user_prompt, "B")
        if fail_marker in a_section or fail_marker in b_section:
            return "no tag"
        return "[[A]]" if "answer" in a_section else "[[B]]"
    return backend


def test_compare_policies_excludes_parse_failures_with_skip_parse_errors():
    rows = [
        Comparison(row_id=f"c{i}", prompt=f"q{i}", response_a=f"answer {i}", response_b="distractor")
        for i in range(4)
    ]
    judge = LLMJudge(_content_tracking_backend("answer 1"))

    result = compare_policies(judge, rows, skip_parse_errors=True)
    assert result.n_parse_failures == 1
    assert result.n_scored == 3
    assert result.n == 4
    assert "c1" not in result.scored_row_ids


def test_compare_policies_without_skip_parse_errors_raises_through():
    rows = [
        Comparison(row_id=f"c{i}", prompt=f"q{i}", response_a=f"answer {i}", response_b="distractor")
        for i in range(4)
    ]
    judge = LLMJudge(_content_tracking_backend("answer 1"))

    with pytest.raises(Exception):  # JudgeParseError, propagated unhandled
        compare_policies(judge, rows)
