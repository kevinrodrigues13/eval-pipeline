"""
Scoring: which of two policies does the judge prefer, and how sure is that?

Position instability is its own exclusion category, not a silent tie. A
genuine tie ("both about equally good") counts as half a win for each side —
the standard convention for scoring pairwise preferences. A judge that
contradicted itself between orderings is a different thing entirely: it
means the verdict cannot be trusted at all, so those comparisons are dropped
from the count rather than blended in as if they meant something.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .judge import Judge, JudgedComparison, judge_many
from .schema import Comparison
from .stats import bootstrap_ci, minimum_detectable_effect


@dataclass
class ScoreResult:
    n: int  # comparisons attempted
    n_scored: int  # stable, parsable comparisons actually counted
    n_wins_1: int
    n_wins_2: int
    n_ties: int
    n_position_unstable: int
    win_rate_policy1: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    minimum_detectable_effect: float
    scored_row_ids: Tuple[str, ...]
    n_parse_failures: int = 0

    @property
    def decisive_rate(self) -> float:
        decisive = self.n_wins_1 + self.n_wins_2
        return decisive / self.n_scored if self.n_scored else 0.0

    @property
    def position_instability_rate(self) -> float:
        return self.n_position_unstable / self.n if self.n else 0.0

    @property
    def parse_failure_rate(self) -> float:
        return self.n_parse_failures / self.n if self.n else 0.0

    def is_underpowered(self) -> bool:
        """
        True when the observed gap from 50% is smaller than what this
        sample size could reliably detect at all.

        Checked before the confidence interval is even looked at: a small
        sample can land a CI that happens to exclude 50% by noise alone, and
        this catches that case rather than reporting it as a real finding.
        """
        if self.win_rate_policy1 is None:
            return True
        gap = abs(self.win_rate_policy1 - 0.5)
        return gap < self.minimum_detectable_effect


def _is_usable(result: JudgedComparison) -> bool:
    return not result.position_unstable and not result.parse_failed


def _win_rate(result_by_id, rows: Sequence[Comparison]) -> Optional[float]:
    stable = [r for r in rows if _is_usable(result_by_id[r.row_id])]
    if not stable:
        return None
    wins_1 = sum(1 for r in stable if result_by_id[r.row_id].winner == "A")
    ties = sum(1 for r in stable if result_by_id[r.row_id].winner == "tie")
    return (wins_1 + 0.5 * ties) / len(stable)


def compare_policies(
    judge: Judge,
    rows: Sequence[Comparison],
    max_workers: int = 1,
    skip_parse_errors: bool = False,
) -> ScoreResult:
    """
    Judge every held-out comparison once, and score policy 1 (whichever
    system sits in slot A across this held-out set — expected to be
    consistent across all of `rows`) against policy 2.

    `max_workers` only speeds up wall-clock time for a live judge; results
    are identical at any worker count (see judge.judge_many). A comparison
    the judge never returned a parsable verdict for is excluded from
    scoring (not counted as a tie) when `skip_parse_errors` is set — see
    `judge.judge_with_position_check`.
    """
    judged: Sequence[JudgedComparison] = judge_many(
        judge, rows, max_workers=max_workers, skip_parse_errors=skip_parse_errors
    )
    result_by_id = {result.row_id: result for result in judged}

    stable = [r for r in rows if _is_usable(result_by_id[r.row_id])]
    n_wins_1 = sum(1 for r in stable if result_by_id[r.row_id].winner == "A")
    n_wins_2 = sum(1 for r in stable if result_by_id[r.row_id].winner == "B")
    n_ties = sum(1 for r in stable if result_by_id[r.row_id].winner == "tie")
    n_scored = len(stable)
    win_rate = (n_wins_1 + 0.5 * n_ties) / n_scored if n_scored else None

    def statistic(batch: Sequence[Comparison]) -> Optional[float]:
        return _win_rate(result_by_id, batch)

    ci_lower, ci_upper = bootstrap_ci(rows, cluster_id=lambda r: r.cluster_id, statistic=statistic)

    n_parse_failures = sum(1 for r in judged if r.parse_failed)
    n_position_unstable = sum(1 for r in judged if r.position_unstable and not r.parse_failed)

    return ScoreResult(
        n=len(rows),
        n_scored=n_scored,
        n_wins_1=n_wins_1,
        n_wins_2=n_wins_2,
        n_ties=n_ties,
        n_position_unstable=n_position_unstable,
        win_rate_policy1=win_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        minimum_detectable_effect=minimum_detectable_effect(n_scored) if n_scored else 0.5,
        scored_row_ids=tuple(r.row_id for r in stable),
        n_parse_failures=n_parse_failures,
    )
