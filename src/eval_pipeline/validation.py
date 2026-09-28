"""
Label-quality validation: flagging noisy or contradictory human labels
explicitly, and measuring how much humans agree with each other.

That second number matters more than it looks: it is the ceiling a judge is
compared against later, not 100%. A judge that "only" agrees with humans 80%
of the time is doing about as well as a second human would, if humans only
agree with each other 80% of the time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence

from .schema import Comparison

# Below this many repeat-labelled items, a pairwise agreement rate is too
# noisy to report as a ceiling — a handful of items could swing it by double
# digits. Withholding it (None) beats printing a number nobody should trust.
MIN_REPEAT_LABELLED_FOR_CEILING = 5


@dataclass
class LabelQualityReport:
    n_items: int
    n_annotations: int
    n_repeat_labelled: int
    n_contested_items: int
    n_contested_annotations: int
    n_ties: int
    n_agreeing_pairs: int
    n_compared_pairs: int

    @property
    def tie_rate(self) -> float:
        return self.n_ties / self.n_annotations if self.n_annotations else 0.0

    @property
    def human_agreement(self) -> Optional[float]:
        if self.n_repeat_labelled < MIN_REPEAT_LABELLED_FOR_CEILING or not self.n_compared_pairs:
            return None
        return self.n_agreeing_pairs / self.n_compared_pairs

    def summary(self) -> str:
        if self.human_agreement is not None:
            ceiling = (
                f"{self.human_agreement:.1%} human-human ceiling "
                f"({self.n_repeat_labelled} repeat-labelled items)"
            )
        else:
            ceiling = (
                f"no human-human ceiling ({self.n_repeat_labelled} repeat-labelled "
                f"items, need {MIN_REPEAT_LABELLED_FOR_CEILING})"
            )
        return (
            f"{self.n_items} items, {self.n_annotations} annotations, "
            f"{self.tie_rate:.1%} ties, {self.n_contested_items} contested items "
            f"({self.n_contested_annotations} annotations on them), {ceiling}"
        )


def is_contested(comparison: Comparison) -> bool:
    """
    An item is contested if its annotators did not unanimously agree —
    including a clear 2-1 majority, not only an outright split.

    A judge that gets a 2-1 item "right" by matching the majority still
    resolved something humans themselves did not agree on, so every
    annotation on that item is flagged, winning side included — the
    disagreement is a property of the item, not of whichever vote lost.
    """
    labels = {annotation.preferred for annotation in comparison.annotations}
    return len(labels) > 1


def flag_noisy_labels(rows: Sequence[Comparison]) -> LabelQualityReport:
    """
    Walk every comparison once, counting what's noisy rather than dropping it.

    The human-human ceiling is measured exactly the way judge-vs-human
    agreement will be measured later (§5): pick two annotations of the same
    item at random, and ask how often they agree. That symmetry is what
    makes the two numbers comparable at all.
    """
    n_annotations = 0
    n_ties = 0
    n_repeat_labelled = 0
    n_contested_items = 0
    n_contested_annotations = 0
    n_agreeing_pairs = 0
    n_compared_pairs = 0

    for row in rows:
        n_annotations += len(row.annotations)
        n_ties += sum(1 for annotation in row.annotations if annotation.preferred == "tie")

        if len(row.annotations) >= 2:
            n_repeat_labelled += 1
            if is_contested(row):
                n_contested_items += 1
                n_contested_annotations += len(row.annotations)
            for first, second in combinations(row.annotations, 2):
                n_compared_pairs += 1
                if first.preferred == second.preferred:
                    n_agreeing_pairs += 1

    return LabelQualityReport(
        n_items=len(rows),
        n_annotations=n_annotations,
        n_repeat_labelled=n_repeat_labelled,
        n_contested_items=n_contested_items,
        n_contested_annotations=n_contested_annotations,
        n_ties=n_ties,
        n_agreeing_pairs=n_agreeing_pairs,
        n_compared_pairs=n_compared_pairs,
    )


def sample_comparisons(rows: Sequence[Comparison], n: int, seed: int = 7) -> List[Comparison]:
    """
    A reproducible random subset of whole comparisons.

    Sampling comparisons rather than annotations is what makes the ceiling
    survive sampling automatically: a sampled item keeps every annotation it
    had, so it is exactly as repeat-labelled as it was before. `n <= 0` or
    `n >= len(rows)` returns every row, unsampled.
    """
    if n <= 0 or n >= len(rows):
        return list(rows)
    return random.Random(seed).sample(list(rows), n)
