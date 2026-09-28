"""
Statistical primitives: chance-corrected agreement, a cluster bootstrap
confidence interval, and minimum detectable effect. Kept minimal — every
function here has a real caller.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def cohens_kappa(pairs: Sequence[Tuple[str, str]]) -> Optional[float]:
    """
    Chance-corrected agreement between two raters over paired categorical
    labels.

    Raw agreement alone lies: two raters who both pick "A" 90% of the time
    will agree roughly 82% of the time by chance alone, before either one
    exercises any real judgment. Kappa subtracts that out — 0 is chance-level
    agreement, 1 is perfect, negative means the raters disagree *more* than
    chance would predict.
    """
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n

    marginal_a: Counter = Counter(a for a, _ in pairs)
    marginal_b: Counter = Counter(b for _, b in pairs)
    categories = set(marginal_a) | set(marginal_b)
    expected = sum((marginal_a[c] / n) * (marginal_b[c] / n) for c in categories)

    if expected >= 1.0:
        # Every pair used the same single category for both raters: there is
        # no variability left to chance-correct against, so kappa is
        # undefined rather than trivially 1.
        return None
    return (observed - expected) / (1 - expected)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; q in [0, 100]."""
    if not values:
        raise ValueError("no values to take a percentile of")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def group_by_cluster(items: Sequence[T], cluster_id: Callable[[T], Hashable]) -> Dict[Hashable, List[T]]:
    groups: Dict[Hashable, List[T]] = {}
    for item in items:
        groups.setdefault(cluster_id(item), []).append(item)
    return groups


def resample_clusters(cluster_keys: Sequence[Hashable], rng: random.Random) -> List[Hashable]:
    """One bootstrap draw: cluster keys sampled with replacement."""
    return [rng.choice(cluster_keys) for _ in range(len(cluster_keys))]


def bootstrap_ci(
    items: Sequence[T],
    cluster_id: Callable[[T], Hashable],
    statistic: Callable[[Sequence[T]], Optional[float]],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 7,
) -> Tuple[Optional[float], Optional[float]]:
    """
    A confidence interval that resamples whole clusters, not individual
    items.

    Items sharing a cluster — two turns of the same conversation, say — are
    correlated, not independent draws. Resampling them individually would
    treat correlated evidence as if it were independent and understate the
    interval's width. Resampling clusters instead (every item in a drawn
    cluster moves together, or not at all) is what keeps the interval
    honest about how much independent evidence there actually is.
    """
    groups = group_by_cluster(items, cluster_id)
    keys = list(groups.keys())
    if not keys:
        return None, None

    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(n_resamples):
        resampled_keys = resample_clusters(keys, rng)
        resampled_items = [item for key in resampled_keys for item in groups[key]]
        value = statistic(resampled_items)
        if value is not None:
            estimates.append(value)

    if not estimates:
        return None, None
    tail = (1 - confidence) / 2 * 100
    return percentile(estimates, tail), percentile(estimates, 100 - tail)


# z-scores for (two-sided significance, power) combinations this pipeline
# actually uses. A closed-form inverse-normal-CDF would generalise this to
# any combination, but nothing here calls for anything but the conventional
# 95%-confidence, 80%-power pairing — a small table beats an unused general
# solver.
_Z_SCORES = {
    (0.05, 0.80): (1.959963985, 0.841621234),
    (0.05, 0.90): (1.959963985, 1.281551566),
    (0.01, 0.80): (2.575829304, 0.841621234),
    (0.10, 0.80): (1.644853627, 0.841621234),
}


def minimum_detectable_effect(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """
    The smallest deviation from a 50/50 split that a sample of size n could
    reliably detect, at the given significance level and power.

    This is what separates "no difference" from "can't tell": an observed
    gap from 50% smaller than this number is not evidence the two policies
    are equally good — the sample is simply too small to answer the question
    either way. Uses the standard normal approximation for a proportion,
    with variance taken at its maximum (p=0.5), which is the conservative
    (largest) estimate of the effect size needed.
    """
    if n <= 0:
        return 0.5
    key = (alpha, power)
    if key not in _Z_SCORES:
        raise ValueError(f"unsupported (alpha, power) combination: {key}; supported: {sorted(_Z_SCORES)}")
    z_alpha, z_beta = _Z_SCORES[key]
    return (z_alpha + z_beta) * 0.5 / (n ** 0.5)
