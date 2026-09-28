from eval_pipeline.calibration import CalibrationReport
from eval_pipeline.report import generate_report
from eval_pipeline.scoring import ScoreResult
from eval_pipeline.validation import LabelQualityReport

NAMES = ("Policy A", "Policy B")


def calibration(
    kappa=0.5, human_kappa=0.55, human_agreement=0.7, agreement=0.68,
    position_flip_rate=0.1, confusion=None, n_parse_failures=0,
):
    return CalibrationReport(
        judge_name="test-judge", n_comparisons=200, n=250,
        agreement=agreement, agreement_ci_lower=agreement - 0.05, agreement_ci_upper=agreement + 0.05,
        kappa=kappa, human_agreement=human_agreement, human_kappa=human_kappa,
        position_flip_rate=position_flip_rate,
        confusion=confusion if confusion is not None else {("A", "A"): 100, ("B", "B"): 80, ("A", "B"): 20},
        n_parse_failures=n_parse_failures,
    )


def score(win_rate=0.7, ci_lower=0.6, ci_upper=0.8, n_scored=100, n=100, mde=0.1,
          n_unstable=0, n_parse_failures=0):
    return ScoreResult(
        n=n, n_scored=n_scored, n_wins_1=70, n_wins_2=20, n_ties=10,
        n_position_unstable=n_unstable, win_rate_policy1=win_rate,
        ci_lower=ci_lower, ci_upper=ci_upper, minimum_detectable_effect=mde,
        scored_row_ids=tuple(f"c{i}" for i in range(n_scored)),
        n_parse_failures=n_parse_failures,
    )


# --- bottom line ----------------------------------------------------------

def test_bottom_line_no_conclusion_when_gate_fails():
    text = generate_report(calibration(kappa=0.1, human_kappa=0.55), score(), policy_names=NAMES)
    assert "No conclusion" in text
    assert "Nothing below should be acted on" in text


def test_bottom_line_not_enough_data_when_underpowered():
    text = generate_report(
        calibration(), score(win_rate=0.52, ci_lower=0.42, ci_upper=0.62, mde=0.15),
        policy_names=NAMES,
    )
    assert "Not enough data to tell" in text
    assert "cannot answer the question" in text


def test_bottom_line_policy1_preferred():
    text = generate_report(calibration(), score(ci_lower=0.6, ci_upper=0.8), policy_names=NAMES)
    assert "Policy A is preferred" in text
    assert "Policy B is preferred" not in text


def test_bottom_line_policy2_preferred():
    text = generate_report(
        calibration(), score(win_rate=0.3, ci_lower=0.2, ci_upper=0.4), policy_names=NAMES,
    )
    assert "Policy B is preferred" in text


def test_bottom_line_no_reliable_difference_when_ci_straddles_half():
    text = generate_report(
        calibration(), score(win_rate=0.55, ci_lower=0.45, ci_upper=0.65, mde=0.05),
        policy_names=NAMES,
    )
    assert "No reliable difference" in text


def test_bottom_line_cites_the_human_ceiling_when_available():
    text = generate_report(calibration(agreement=0.68, human_agreement=0.7), score(), policy_names=NAMES)
    assert "68.0%" in text and "70.0%" in text
    assert "reliable as a second annotator" in text


# --- trust section ----------------------------------------------------------

def test_trust_section_shows_kappa_and_ceiling():
    text = generate_report(calibration(kappa=0.5, human_kappa=0.55), score(), policy_names=NAMES)
    assert "Cohen's kappa" in text
    assert "0.500" in text
    assert "Human-human ceiling" in text


def test_trust_section_flags_order_invariant_judge():
    text = generate_report(calibration(position_flip_rate=None), score(), policy_names=NAMES)
    assert "not applicable" in text
    assert "order-invariant" in text


def test_trust_section_warns_when_no_ceiling_available():
    text = generate_report(calibration(human_agreement=None, human_kappa=None), score(), policy_names=NAMES)
    assert "No human-human ceiling could be estimated" in text


def test_confusion_table_renders_rows_and_totals():
    text = generate_report(calibration(), score(), policy_names=NAMES)
    assert "human \\ judge" in text
    assert "| **A** |" in text


def test_label_quality_is_reported_when_available():
    quality = LabelQualityReport(
        n_items=906, n_annotations=1678, n_repeat_labelled=600,
        n_contested_items=200, n_contested_annotations=554,
        n_ties=405, n_agreeing_pairs=400, n_compared_pairs=600,
    )
    text = generate_report(calibration(), score(), label_quality=quality, policy_names=NAMES)
    assert "554 annotations" in text
    assert "200 items" in text


# --- comparison section ----------------------------------------------------

def test_comparison_section_headline_and_supporting_detail():
    text = generate_report(calibration(), score(win_rate=0.7, n_scored=100), policy_names=NAMES)
    assert "70.0%" in text
    assert "Smallest gap this sample could detect" in text


def test_position_unstable_exclusion_line_only_appears_when_nonzero():
    with_unstable = generate_report(
        calibration(), score(n=150, n_scored=100, n_unstable=50), policy_names=NAMES,
    )
    without_unstable = generate_report(
        calibration(), score(n=100, n_scored=100, n_unstable=0), policy_names=NAMES,
    )
    assert "contradicted itself" in with_unstable
    assert "contradicted itself" not in without_unstable


def test_calibration_parse_failures_line_only_appears_when_nonzero():
    with_failures = generate_report(
        calibration(n_parse_failures=3), score(), policy_names=NAMES,
    )
    without_failures = generate_report(
        calibration(n_parse_failures=0), score(), policy_names=NAMES,
    )
    assert "Excluded — no parsable verdict | 3 comparisons" in with_failures
    assert "no parsable verdict" not in without_failures


def test_score_parse_failures_line_only_appears_when_nonzero_and_shows_a_rate():
    with_failures = generate_report(
        calibration(), score(n=100, n_scored=90, n_parse_failures=10), policy_names=NAMES,
    )
    without_failures = generate_report(
        calibration(), score(n=100, n_scored=100, n_parse_failures=0), policy_names=NAMES,
    )
    assert "never returned a parsable verdict: 10 of 100 (10.0%)" in with_failures
    assert "never returned a parsable verdict" not in without_failures


# --- caveats ----------------------------------------------------------------

def test_instability_drift_caveat_fires_when_held_out_is_much_less_stable():
    text = generate_report(
        calibration(position_flip_rate=0.1),
        score(n=100, n_scored=60, n_unstable=40),  # 40% held-out vs 10% calibration
        policy_names=NAMES,
    )
    assert "less stable here than during calibration" in text


def test_caveats_flag_high_instability_even_without_a_calibration_baseline():
    # An order-invariant judge (e.g. a recorded/replay judge) never gets a
    # calibration-time instability rate, so the ratio check has nothing to
    # compare against — this must not silently pass a 31.5% exclusion rate.
    text = generate_report(
        calibration(position_flip_rate=None),
        score(n=146, n_scored=100, n_unstable=46),
        policy_names=NAMES,
    )
    assert "No calibration-time position-instability rate to compare against" in text
    assert "31.5%" in text


def test_caveats_fall_back_to_nothing_to_flag():
    text = generate_report(
        calibration(position_flip_rate=0.1), score(n=100, n_scored=95, n_unstable=5),
        policy_names=NAMES,
    )
    assert "found anything to flag here" in text


# --- overall structure -------------------------------------------------------

def test_report_sections_appear_in_order():
    text = generate_report(calibration(), score(), policy_names=NAMES)
    assert text.index("Bottom line") < text.index("1. How far the judge can be trusted")
    assert text.index("1. How far") < text.index("2. The comparison")
    assert text.index("2. The comparison") < text.index("3. What this does not account for")
