"""
Core data schema: one row per pairwise comparison.

A comparison — not an annotation — is the unit everything downstream
resamples and clusters over. An item labelled by five annotators is one
comparison carrying five annotations, not five rows; treating it as five
independent rows would understate how correlated those labels are and
inflate confidence in whatever statistic gets computed over them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Annotation:
    """One human's verdict on a comparison: "A", "B", or "tie"."""

    preferred: str
    annotator_id: Optional[str] = None


@dataclass(frozen=True)
class Comparison:
    """
    One (prompt, response_A, response_B) item, carrying every human
    annotation collected for it.

    `cluster_id` groups comparisons that share a source question — e.g. two
    turns of the same dialogue — so a bootstrap can resample by cluster
    rather than treating correlated rows as independent. It defaults to the
    row's own id, which makes every comparison its own cluster unless the
    data says otherwise.

    `context_a`/`context_b` carry prior turns of a multi-turn dialogue, as
    (role, text) pairs, so a judge sees the same conversation history a human
    annotator did rather than an isolated final exchange.
    """

    row_id: str
    prompt: str
    response_a: str
    response_b: str
    annotations: Tuple[Annotation, ...] = ()
    cluster_id: Optional[str] = None
    system_a: Optional[str] = None
    system_b: Optional[str] = None
    context_a: Tuple[Tuple[str, str], ...] = ()
    context_b: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.cluster_id is None:
            object.__setattr__(self, "cluster_id", self.row_id)

    @property
    def majority(self) -> Optional[str]:
        """
        The human majority verdict: "A", "B", "tie", or "split".

        "tie" means the annotators agreed the responses were equally good.
        "split" means they disagreed with no majority — a different thing,
        because it carries no evidence about response quality, only about
        the item being contested. `None` means no annotations at all.
        """
        if not self.annotations:
            return None
        counts: dict = {}
        for annotation in self.annotations:
            counts[annotation.preferred] = counts.get(annotation.preferred, 0) + 1
        top = max(counts.values())
        winners = [choice for choice, n in counts.items() if n == top]
        return winners[0] if len(winners) == 1 else "split"
