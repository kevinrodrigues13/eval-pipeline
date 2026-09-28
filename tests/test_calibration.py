import pytest

from eval_pipeline.calibration import (
    CalibrationFailed,
    CalibrationReport,
    calibrate,
    gate_on_calibration,
)
from eval_pipeline.judge import LLMJudge, PrecomputedJudge
from eval_pipeline.schema import Annotation, Comparison
from eval_pipeline.validation import flag_noisy_labels


def row(row_id, *preferred_values, response_a="a", response_b="b"):
    return Comparison(
        row_id=row_id, prompt="q", response_a=response_a, response_b=response_b,
        annotations=tuple(Annotation(p) for p in preferred_values),
    )


def test_calibrate_computes_per_annotation_agreement_and_confusion():
    rows = [row("c0", "A", "A"), row("c1", "B")]
    judge = PrecomputedJudge.from_records([
        {"id": "c0", "winner": "A"}, {"id": "c1", "winner": "A"},
    ])
    report = calibrate(judge, rows)

    assert report.n == 3  # 2 annotations on c0 + 1 on c1
    assert report.n_comparisons == 2
    assert report.agreement == pytest.approx(2 / 3)
    assert report.confusion == {("A", "A"): 2, ("B", "A"): 1}


def test_kappa_corrects_for_a_judge_that_always_answers_the_same_way():
    # Raw agreement is 2/3, but the judge never says anything but "A" — no
    # real signal, and kappa should show that even though raw agreement looks
    # decent.
    rows = [row("c0", "A", "A"), row("c1", "B")]
    judge = PrecomputedJudge.from_records([
        {"id": "c0", "winner": "A"}, {"id": "c1", "winner": "A"},
    ])
    report = calibrate(judge, rows)
    assert report.kappa == pytest.approx(0.0)


def test_order_invariant_judge_has_no_position_flip_rate():
    rows = [row("c0", "A")]
    judge = PrecomputedJudge.from_records([{"id": "c0", "winner": "A"}])
    report = calibrate(judge, rows)
    assert report.position_flip_rate is None


def test_non_order_invariant_judge_reports_a_real_position_flip_rate():
    # Stable row: content-tracking backend agrees before and after swap.
    # Unstable row: backend always says "[[A]]" regardless of content.
    stable = row("stable", "A", response_a="the real answer", response_b="a distractor")
    unstable = row("unstable", "A", response_a="x", response_b="y")

    def backend(system_prompt, user_prompt):
        if "distractor" in user_prompt or "real answer" in user_prompt:
            a_section = user_prompt.split("Assistant A's Answer]\n")[1].split(
                "\n[The End of Assistant A's Answer]"
            )[0]
            return "[[A]]" if "real answer" in a_section else "[[B]]"
        return "[[A]]"  # always A: unstable under a swap

    judge = LLMJudge(backend)
    report = calibrate(judge, [stable, unstable])
    assert report.position_flip_rate == pytest.approx(0.5)


def test_calibrate_integrates_the_human_human_ceiling_from_label_quality():
    rows = [row("c0", "A", "A"), row("c1", "B", "B"), row("c2", "A", "B")]
    judge = PrecomputedJudge.from_records([
        {"id": "c0", "winner": "A"}, {"id": "c1", "winner": "B"}, {"id": "c2", "winner": "A"},
    ])
    quality = flag_noisy_labels(rows)
    # Not enough repeat-labelled items (3 < MIN_REPEAT_LABELLED_FOR_CEILING=5)
    # to have a real ceiling, so human_agreement, human_kappa, and both
    # share_of_* properties should all come back withheld (None) together —
    # not just human_agreement, or gate_on_calibration could still trust an
    # unreliable human_kappa computed from the same too-small sample.
    report = calibrate(judge, rows, label_quality=quality)
    assert report.human_agreement is None
    assert report.human_kappa is None
    assert report.share_of_human_agreement is None
    assert report.share_of_human_kappa is None
    with pytest.raises(CalibrationFailed):
        gate_on_calibration(report)


def test_share_of_human_kappa_is_the_ratio():
    report = CalibrationReport(
        judge_name="j", n_comparisons=10, n=10, agreement=0.8,
        agreement_ci_lower=None, agreement_ci_upper=None,
        kappa=0.4, human_agreement=0.8, human_kappa=0.5,
        position_flip_rate=None, confusion={},
    )
    assert report.share_of_human_kappa == pytest.approx(0.8)
    assert report.share_of_human_agreement == pytest.approx(1.0)


# --- gate_on_calibration -------------------------------------------------

def _report(kappa, human_kappa):
    return CalibrationReport(
        judge_name="j", n_comparisons=10, n=10, agreement=0.8,
        agreement_ci_lower=None, agreement_ci_upper=None,
        kappa=kappa, human_agreement=0.8, human_kappa=human_kappa,
        position_flip_rate=None, confusion={},
    )


def test_gate_passes_when_judge_meets_the_share_bar():
    gate_on_calibration(_report(kappa=0.4, human_kappa=0.5))  # 80% of ceiling


def test_gate_fails_when_judge_is_below_the_share_bar():
    with pytest.raises(CalibrationFailed):
        gate_on_calibration(_report(kappa=0.2, human_kappa=0.5))  # 40% of ceiling


def test_gate_fails_with_no_human_ceiling_to_compare_against():
    with pytest.raises(CalibrationFailed):
        gate_on_calibration(_report(kappa=0.4, human_kappa=None))


def test_gate_fails_when_humans_dont_agree_above_chance():
    with pytest.raises(CalibrationFailed):
        gate_on_calibration(_report(kappa=0.4, human_kappa=0.0))


def test_gate_fails_when_judge_kappa_is_undefined():
    with pytest.raises(CalibrationFailed):
        gate_on_calibration(_report(kappa=None, human_kappa=0.5))


def test_calibrate_gives_identical_results_at_any_worker_count():
    rows = [row(f"c{i}", "A", "A") for i in range(10)]
    judge = PrecomputedJudge.from_records([{"id": f"c{i}", "winner": "A"} for i in range(10)])

    sequential = calibrate(judge, rows, max_workers=1)
    concurrent_ = calibrate(judge, rows, max_workers=4)
    assert sequential.agreement == concurrent_.agreement
    assert sequential.kappa == concurrent_.kappa
    assert sequential.confusion == concurrent_.confusion


def test_calibrate_excludes_parse_failures_with_skip_parse_errors():
    rows = [row(f"c{i}", "A", "A", response_a=f"answer {i}") for i in range(4)]

    def backend(system_prompt, user_prompt):
        return "no tag" if "answer 1" in user_prompt else "[[A]]"

    report = calibrate(LLMJudge(backend), rows, skip_parse_errors=True)
    assert report.n_parse_failures == 1
    assert report.n_comparisons == 3  # c1 excluded entirely
    assert report.n == 6  # 3 usable comparisons x 2 annotations each
    assert "1 parse failures excluded" in report.summary()


def test_calibrate_without_skip_parse_errors_raises_through():
    rows = [row(f"c{i}", "A", "A", response_a=f"answer {i}") for i in range(4)]

    def backend(system_prompt, user_prompt):
        return "no tag" if "answer 1" in user_prompt else "[[A]]"

    with pytest.raises(Exception):  # JudgeParseError, propagated unhandled
        calibrate(LLMJudge(backend), rows)
