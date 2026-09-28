"""
Calibration: how far can this judge be trusted?

The judge is measured against human labels the same way humans are measured
against each other (validation.flag_noisy_labels) — pick a human annotation
of an item at random and ask how often it matches. That symmetry is what
makes "judge agreement" and "human-human ceiling" comparable numbers at all,
and it is why the ceiling is the bar, not 100%: humans routinely disagree
with each other on preference data, so a judge cannot be expected to clear a
bar humans themselves do not.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from .judge import Judge, JudgedComparison, judge_many
from .schema import Comparison
from .stats import bootstrap_ci, cohens_kappa
from .validation import LabelQualityReport

# The judge must recover at least this share of the chance-corrected
# agreement a second human would, to be trusted at all. A ratio against the
# human-human ceiling rather than a fixed kappa threshold (e.g. Landis &
# Koch's bands) — those assume the reference labels are themselves
# near-perfect, which preference labels are not: humans here only agree with
# each other a bit above two-thirds of the time (validation.py). A fixed
# "kappa >= 0.6" bar would be unreachable on data this noisy regardless of
# how good the judge actually is.
MIN_SHARE_OF_HUMAN_KAPPA = 0.5


class CalibrationFailed(Exception):
    """The judge does not clear the trust bar; nothing downstream should use it."""


@dataclass
class CalibrationReport:
    judge_name: str
    n_comparisons: int
    n: int  # annotations compared, i.e. len(pairs) — agreement is per annotation
    agreement: float
    agreement_ci_lower: Optional[float]
    agreement_ci_upper: Optional[float]
    kappa: Optional[float]
    human_agreement: Optional[float]
    human_kappa: Optional[float]
    position_flip_rate: Optional[float]
    confusion: Dict[Tuple[str, str], int]
    n_parse_failures: int = 0

    @property
    def share_of_human_agreement(self) -> Optional[float]:
        if not self.human_agreement:
            return None
        return self.agreement / self.human_agreement

    @property
    def share_of_human_kappa(self) -> Optional[float]:
        if self.kappa is None or not self.human_kappa:
            return None
        return self.kappa / self.human_kappa

    def summary(self) -> str:
        ci = (
            f" [{self.agreement_ci_lower:.1%}, {self.agreement_ci_upper:.1%}]"
            if self.agreement_ci_lower is not None
            else ""
        )
        kappa = f"{self.kappa:.3f}" if self.kappa is not None else "n/a"
        against = (
            f" vs {self.human_agreement:.1%} human-human ceiling "
            f"(kappa {self.human_kappa:.3f})"
            if self.human_agreement is not None and self.human_kappa is not None
            else " (no human-human ceiling to compare against)"
        )
        flip = (
            f", {self.position_flip_rate:.1%} position-flip rate"
            if self.position_flip_rate is not None
            else ""
        )
        failures = f", {self.n_parse_failures} parse failures excluded" if self.n_parse_failures else ""
        return (
            f"{self.judge_name}: {self.agreement:.1%}{ci} agreement (kappa {kappa}) "
            f"on {self.n} annotations over {self.n_comparisons} comparisons{against}{flip}{failures}"
        )


def _human_human_pairs(rows: Sequence[Comparison]) -> List[Tuple[str, str]]:
    return [
        (first.preferred, second.preferred)
        for row in rows
        if len(row.annotations) >= 2
        for first, second in combinations(row.annotations, 2)
    ]


def calibrate(
    judge: Judge,
    rows: Sequence[Comparison],
    label_quality: Optional[LabelQualityReport] = None,
    max_workers: int = 1,
    skip_parse_errors: bool = False,
) -> CalibrationReport:
    """
    Judge every comparison once, and pair that single verdict against each
    of its human annotations — an item with three annotators contributes
    three (human, judge) pairs, all sharing the same judge verdict.

    `max_workers` only speeds up wall-clock time for a live judge; results
    are identical at any worker count (see judge.judge_many). A comparison
    the judge never returned a parsable verdict for is excluded from every
    number below (not counted as agreement or disagreement) when
    `skip_parse_errors` is set — see `judge.judge_with_position_check`.
    """
    judged: List[JudgedComparison] = judge_many(
        judge, rows, max_workers=max_workers, skip_parse_errors=skip_parse_errors
    )
    judged_by_id = {result.row_id: result for result in judged}
    usable_rows = [row for row in rows if not judged_by_id[row.row_id].parse_failed]
    n_parse_failures = len(rows) - len(usable_rows)

    pairs: List[Tuple[str, str]] = []
    confusion: Dict[Tuple[str, str], int] = {}
    for row in usable_rows:
        judge_winner = judged_by_id[row.row_id].winner
        for annotation in row.annotations:
            pairs.append((annotation.preferred, judge_winner))
            key = (annotation.preferred, judge_winner)
            confusion[key] = confusion.get(key, 0) + 1

    agreement = sum(1 for h, j in pairs if h == j) / len(pairs) if pairs else 0.0
    kappa = cohens_kappa(pairs)

    ci_lower, ci_upper = bootstrap_ci(
        usable_rows,
        cluster_id=lambda r: r.cluster_id,
        statistic=lambda batch: (
            sum(1 for r in batch for a in r.annotations if a.preferred == judged_by_id[r.row_id].winner)
            / sum(len(r.annotations) for r in batch)
            if sum(len(r.annotations) for r in batch) else None
        ),
    )

    stable_judged = [r for r in judged if not r.parse_failed]
    if judge.is_order_invariant:
        position_flip_rate = None
    else:
        position_flip_rate = (
            sum(1 for r in stable_judged if r.position_unstable) / len(stable_judged)
            if stable_judged else None
        )

    human_agreement = label_quality.human_agreement if label_quality is not None else None
    # Gated on human_agreement, not just `label_quality is not None`: below
    # MIN_REPEAT_LABELLED_FOR_CEILING, human_agreement is already withheld
    # (None) as too unreliable to report — human_kappa must be withheld on
    # the same condition, or gate_on_calibration would trust a kappa
    # computed from too few repeat-labelled items to be worth trusting.
    human_kappa = cohens_kappa(_human_human_pairs(rows)) if human_agreement is not None else None

    return CalibrationReport(
        judge_name=judge.name,
        n_comparisons=len(usable_rows),
        n=len(pairs),
        agreement=agreement,
        agreement_ci_lower=ci_lower,
        agreement_ci_upper=ci_upper,
        kappa=kappa,
        human_agreement=human_agreement,
        human_kappa=human_kappa,
        position_flip_rate=position_flip_rate,
        confusion=confusion,
        n_parse_failures=n_parse_failures,
    )


def gate_on_calibration(report: CalibrationReport) -> None:
    """
    Refuse to let a poorly-calibrated judge's verdicts feed anything
    downstream.

    Gated on kappa, not raw agreement — a judge that always says "tie"
    could post high raw agreement on tie-heavy data while adding no real
    information. Gated as a *ratio* against the human ceiling, not a fixed
    threshold, for the reason MIN_SHARE_OF_HUMAN_KAPPA documents above.
    """
    if report.kappa is None:
        raise CalibrationFailed("judge's kappa is undefined — no variability to measure agreement on")
    if report.human_kappa is None:
        raise CalibrationFailed("no human-human ceiling could be estimated — nothing to gate against")
    if report.human_kappa <= 0:
        raise CalibrationFailed(
            f"human-human kappa is {report.human_kappa:.3f} — humans themselves do not "
            "agree above chance on this data, so no judge could pass a ratio gate against it"
        )
    share = report.kappa / report.human_kappa
    if share < MIN_SHARE_OF_HUMAN_KAPPA:
        raise CalibrationFailed(
            f"judge captures only {share:.0%} of human-human chance-corrected agreement "
            f"(kappa {report.kappa:.3f} vs human {report.human_kappa:.3f}), below the "
            f"{MIN_SHARE_OF_HUMAN_KAPPA:.0%} bar"
        )
