"""
Rendering results for someone who has to act on them, not just read them.

Two rules shape everything here. Claims are laid out in a ladder — no
conclusion, if the judge can't be trusted; not enough data, if the sample
can't tell; otherwise the actual comparison — so what the judge found is
never confused with what the reader can safely act on. And the gate that
decides trust (calibration.gate_on_calibration) is the single source of
truth for that decision — this module calls it rather than re-deriving its
own opinion of when a judge is good enough.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .calibration import CalibrationFailed, CalibrationReport, gate_on_calibration
from .scoring import ScoreResult
from .validation import LabelQualityReport

# Above this ratio, the judge is materially less stable on the held-out
# comparisons than it was on the data its trust was measured on — a warning
# sign the held-out prompts are harder, not just a different sample.
INSTABILITY_RATIO_WARNING = 1.5


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _bottom_line(
    calibration: CalibrationReport, score: ScoreResult, names: Tuple[str, str]
) -> str:
    first, second = names

    try:
        gate_on_calibration(calibration)
    except CalibrationFailed as reason:
        return (
            "**No conclusion.** The judge is not reliable enough on this data for "
            f"its verdicts to carry information about preference ({reason}). "
            "Nothing below should be acted on."
        )

    if score.is_underpowered():
        margin = score.minimum_detectable_effect
        return (
            f"**Not enough data to tell.** At {score.n_scored} scored comparisons "
            f"this can only detect a win rate outside "
            f"{_pct(0.5 - margin)}–{_pct(0.5 + margin)}, and the observed gap is "
            "smaller than that. This is *not* evidence the two are equally good — "
            "the sample cannot answer the question. Collect more held-out prompts."
        )

    trust = ""
    if calibration.human_agreement is not None:
        trust = (
            f" The judge agrees with humans {_pct(calibration.agreement)} of the "
            f"time against a human-human ceiling of {_pct(calibration.human_agreement)}, "
            "so its verdicts are about as reliable as a second annotator's."
        )

    if score.ci_lower is not None and score.ci_lower > 0.5:
        return (
            f"**{first} is preferred.** The judge favoured it on "
            f"{_pct(score.win_rate_policy1)} of scored comparisons, and the interval "
            f"({_pct(score.ci_lower)}–{_pct(score.ci_upper)}) sits entirely "
            f"above 50%.{trust}"
        )
    if score.ci_upper is not None and score.ci_upper < 0.5:
        return (
            f"**{second} is preferred.** The judge favoured it on "
            f"{_pct(1 - score.win_rate_policy1)} of scored comparisons, and the "
            f"interval sits entirely below 50%.{trust}"
        )

    return (
        "**No reliable difference.** The interval "
        f"({_pct(score.ci_lower)}–{_pct(score.ci_upper)}) still admits a 50/50 "
        f"split.{trust}"
    )


def _confusion_table(confusion: Dict[Tuple[str, str], int]) -> List[str]:
    """
    Where the judge's mistakes go, not just how many there are.

    The headline agreement number can't distinguish a judge that errs evenly
    from one that systematically over-calls ties or leans toward a slot —
    and those imply different fixes.
    """
    if not confusion:
        return []

    order = ["A", "B", "tie"]
    lines = ["**Where the disagreements go**", ""]
    lines.append("| human \\ judge | " + " | ".join(order) + " | total |")
    lines.append("|---|" + "---|" * (len(order) + 1))
    for human in order:
        row = [confusion.get((human, judge), 0) for judge in order]
        if not sum(row):
            continue
        lines.append(f"| **{human}** | " + " | ".join(str(n) for n in row) + f" | {sum(row)} |")
    lines.append("")
    lines.append(
        "> Rows are what the humans said, columns what the judge said. The "
        "diagonal is agreement. A heavy `tie` column means the judge declines "
        "to choose where people do; an asymmetry between the A→B and B→A "
        "cells is residual position bias that survived the swap."
    )
    lines.append("")
    return lines


def _trust_section(
    calibration: CalibrationReport, label_quality: Optional[LabelQualityReport]
) -> List[str]:
    lines = ["## 1. How far the judge can be trusted", ""]
    lines.append(
        f"Judge: **{calibration.judge_name}**, measured on "
        f"{calibration.n_comparisons} human-labelled comparisons carrying "
        f"{calibration.n} annotations. Agreement is per annotation, so it means "
        "\"matches one randomly drawn human\" — the same quantity the ceiling "
        "measures between two humans."
    )
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    ci = (
        f" [{_pct(calibration.agreement_ci_lower)}, {_pct(calibration.agreement_ci_upper)}]"
        if calibration.agreement_ci_lower is not None
        else ""
    )
    lines.append(f"| Agreement with humans | {_pct(calibration.agreement)}{ci} |")
    kappa = f"{calibration.kappa:.3f}" if calibration.kappa is not None else "n/a"
    lines.append(f"| Cohen's kappa | {kappa} |")

    if calibration.human_agreement is not None:
        human_kappa = f"{calibration.human_kappa:.3f}" if calibration.human_kappa is not None else "n/a"
        lines.append(
            f"| **Human-human ceiling** | **{_pct(calibration.human_agreement)}** "
            f"(kappa {human_kappa}) |"
        )
        lines.append(
            f"| Judge as a share of that ceiling | "
            f"{_pct(calibration.share_of_human_agreement)} raw, "
            f"{_pct(calibration.share_of_human_kappa)} chance-corrected |"
        )

    if calibration.position_flip_rate is None:
        lines.append("| Position bias | not applicable — judge is order-invariant by construction |")
    else:
        lines.append(f"| Position-bias flip rate | {_pct(calibration.position_flip_rate)} |")
    if calibration.n_parse_failures:
        lines.append(
            f"| Excluded — no parsable verdict | {calibration.n_parse_failures} comparisons |"
        )
    lines.append("")

    if calibration.human_agreement is not None:
        lines.append(
            f"> Humans agree with each other only {_pct(calibration.human_agreement)} of "
            "the time on this data, so that — not 100% — is the maximum any judge "
            "could reach. Read the judge's score against it."
        )
        lines.append("")
    else:
        lines.append(
            "> No human-human ceiling could be estimated: too few comparisons were "
            "labelled more than once. The judge's agreement is therefore being "
            "compared against an unreachable 100% and understates its quality by "
            "an unknown amount."
        )
        lines.append("")

    lines.extend(_confusion_table(calibration.confusion))

    if label_quality is not None:
        lines.append(
            f"Label quality: {_pct(label_quality.tie_rate)} of human annotations are "
            f"ties, and {label_quality.n_contested_annotations} annotations sit on "
            f"the {label_quality.n_contested_items} items where humans disagreed "
            "with each other."
        )
        lines.append("")

    return lines


def _comparison_section(score: ScoreResult, names: Tuple[str, str]) -> List[str]:
    first, second = names
    lines = ["## 2. The comparison", ""]

    lines.append("### Headline")
    lines.append("")
    if score.win_rate_policy1 is not None:
        lines.append(
            f"The judge preferred **{first}** on {_pct(score.win_rate_policy1)} of "
            f"{score.n_scored} scored comparisons "
            f"(95% CI {_pct(score.ci_lower)}–{_pct(score.ci_upper)}). "
            f"Counts: {score.n_wins_1} to {score.n_wins_2}, with {score.n_ties} ties."
        )
    else:
        lines.append("No comparison could be scored — every attempt was position-unstable.")
    lines.append("")

    lines.append("**Supporting detail**")
    lines.append("")
    lines.append(f"- Decisive only ({score.n_wins_1 + score.n_wins_2} comparisons): {_pct(score.decisive_rate)}")
    lines.append(f"- Smallest gap this sample could detect: ±{_pct(score.minimum_detectable_effect)}")
    if score.n_position_unstable:
        lines.append(
            f"- Excluded because the judge contradicted itself when the responses "
            f"were swapped: {score.n_position_unstable} of {score.n} "
            f"({_pct(score.position_instability_rate)})"
        )
    if score.n_parse_failures:
        lines.append(
            f"- Excluded because the judge never returned a parsable verdict: "
            f"{score.n_parse_failures} of {score.n} ({_pct(score.parse_failure_rate)})"
        )
    lines.append("")
    return lines


def _caveats(calibration: CalibrationReport, score: ScoreResult) -> List[str]:
    lines = ["## 3. What this does not account for", ""]

    calibration_instability = calibration.position_flip_rate
    held_out_instability = score.position_instability_rate

    if calibration_instability is not None:
        if held_out_instability > INSTABILITY_RATIO_WARNING * calibration_instability:
            lines.append(
                f"- **The judge is less stable here than during calibration.** It "
                f"contradicted itself on {_pct(held_out_instability)} of these "
                f"comparisons, against {_pct(calibration_instability)} on the data "
                "it was calibrated on. Treat the headline with proportionally "
                "more caution than its interval alone suggests."
            )
    elif score.n_position_unstable:
        # An order-invariant judge (e.g. replaying pre-resolved verdicts)
        # never gets asked twice during calibration, so there is no
        # calibration-time rate to compare the held-out rate against — but
        # a high held-out rate is still worth surfacing on its own terms
        # rather than silently passing because the ratio check has nothing
        # to divide by.
        lines.append(
            f"- **No calibration-time position-instability rate to compare "
            f"against** (this judge is order-invariant by construction, so "
            f"calibration never exercises this check) — {_pct(held_out_instability)} "
            "of held-out comparisons were excluded here for contradicting "
            "themselves under a position swap. Judge that number on its own terms."
        )

    if len(lines) == 2:  # only the heading and blank line — nothing applied
        lines.append(
            "- None of the checks this report runs (position-instability drift) "
            "found anything to flag here."
        )

    lines.append("")
    return lines


def generate_report(
    calibration: CalibrationReport,
    score: ScoreResult,
    label_quality: Optional[LabelQualityReport] = None,
    policy_names: Tuple[str, str] = ("Policy 1", "Policy 2"),
) -> str:
    lines = ["# Evaluation report", ""]
    lines.append(f"Comparing **{policy_names[0]}** against **{policy_names[1]}**.")
    lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    lines.append(_bottom_line(calibration, score, policy_names))
    lines.append("")
    lines.extend(_trust_section(calibration, label_quality))
    lines.extend(_comparison_section(score, policy_names))
    lines.extend(_caveats(calibration, score))
    return "\n".join(lines)
