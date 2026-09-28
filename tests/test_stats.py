import random

import pytest

from eval_pipeline.stats import (
    bootstrap_ci,
    cohens_kappa,
    group_by_cluster,
    minimum_detectable_effect,
    percentile,
    resample_clusters,
)


def test_kappa_is_one_for_perfect_agreement():
    pairs = [("A", "A"), ("A", "A"), ("B", "B"), ("B", "B")]
    assert cohens_kappa(pairs) == pytest.approx(1.0)


def test_kappa_is_zero_at_exactly_chance_level():
    # Both raters split 50/50 between A and B, paired so observed agreement
    # (0.5) exactly equals the chance-expected agreement (0.5).
    pairs = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
    assert cohens_kappa(pairs) == pytest.approx(0.0)


def test_kappa_is_negative_when_raters_disagree_more_than_chance():
    pairs = [("A", "B"), ("A", "B"), ("B", "A"), ("B", "A")]
    assert cohens_kappa(pairs) == pytest.approx(-1.0)


def test_kappa_raw_agreement_alone_would_overstate_this_case():
    # Both raters pick "A" 90% of the time; naive agreement looks high, but
    # most of it is two raters independently favouring the common label.
    pairs = [("A", "A")] * 81 + [("A", "B")] * 9 + [("B", "A")] * 9 + [("B", "B")] * 1
    raw_agreement = sum(1 for a, b in pairs if a == b) / len(pairs)
    kappa = cohens_kappa(pairs)
    assert raw_agreement == pytest.approx(0.82)
    assert kappa < 0.1  # nowhere near as good as raw agreement suggests


def test_kappa_is_none_with_no_pairs_or_no_variability():
    assert cohens_kappa([]) is None
    assert cohens_kappa([("tie", "tie"), ("tie", "tie")]) is None


def test_percentile_matches_the_median_for_odd_length_lists():
    assert percentile([1, 2, 3, 4, 5], 50) == 3


def test_percentile_interpolates_between_points():
    assert percentile([0, 10], 50) == 5


def test_percentile_single_value_ignores_q():
    assert percentile([7], 0) == 7
    assert percentile([7], 100) == 7


def test_group_by_cluster_groups_correctly():
    groups = group_by_cluster([1, 2, 3, 4, 5], lambda n: n % 2)
    assert groups == {1: [1, 3, 5], 0: [2, 4]}


def test_resample_clusters_same_size_drawn_from_input():
    rng = random.Random(0)
    keys = ["a", "b", "c"]
    resampled = resample_clusters(keys, rng)
    assert len(resampled) == len(keys)
    assert set(resampled) <= set(keys)


def test_bootstrap_ci_is_reproducible_with_the_same_seed():
    items = [("c0", 1), ("c0", 1), ("c1", 0), ("c2", 1), ("c3", 0)]
    stat = lambda batch: sum(v for _, v in batch) / len(batch) if batch else None
    ci_1 = bootstrap_ci(items, lambda x: x[0], stat, n_resamples=200, seed=7)
    ci_2 = bootstrap_ci(items, lambda x: x[0], stat, n_resamples=200, seed=7)
    assert ci_1 == ci_2


def test_bootstrap_ci_contains_the_point_estimate_on_stable_data():
    # All clusters have the same value, so every resample gives exactly 1.0.
    items = [(f"c{i}", 1) for i in range(20)]
    stat = lambda batch: sum(v for _, v in batch) / len(batch) if batch else None
    lower, upper = bootstrap_ci(items, lambda x: x[0], stat, n_resamples=200, seed=7)
    assert lower == upper == 1.0


def test_bootstrap_ci_widens_with_fewer_underlying_clusters():
    """
    100 items in 2 big clusters (one all-1s, one all-0s) versus the same
    balance of values spread across 100 independent single-item clusters:
    the number of *clusters* is the resampling unit, so 2 clusters gives the
    bootstrap only two effectively-independent draws to work with — far
    less evidence than 100 independent ones, even with the same item count
    and the same overall 50/50 split.
    """
    stat = lambda batch: sum(v for _, v in batch) / len(batch) if batch else None
    few_big_clusters = [("c0", 1)] * 50 + [("c1", 0)] * 50
    many_small_clusters = [(f"c{i}", i % 2) for i in range(100)]

    lo_few, hi_few = bootstrap_ci(few_big_clusters, lambda x: x[0], stat, n_resamples=2000, seed=7)
    lo_many, hi_many = bootstrap_ci(many_small_clusters, lambda x: x[0], stat, n_resamples=2000, seed=7)
    assert (hi_few - lo_few) > (hi_many - lo_many)


def test_bootstrap_ci_on_empty_input_returns_none():
    assert bootstrap_ci([], lambda x: x, lambda batch: 0.0) == (None, None)


def test_minimum_detectable_effect_matches_the_normal_approximation():
    # (1.959963985 + 0.841621234) * 0.5 / sqrt(100) ~= 0.1400792609
    assert minimum_detectable_effect(100) == pytest.approx(0.1400792609, abs=1e-6)


def test_minimum_detectable_effect_shrinks_as_n_grows():
    assert minimum_detectable_effect(100) > minimum_detectable_effect(1000)


def test_minimum_detectable_effect_is_maximal_at_zero_or_negative_n():
    assert minimum_detectable_effect(0) == 0.5
    assert minimum_detectable_effect(-5) == 0.5


def test_minimum_detectable_effect_rejects_unsupported_alpha_power():
    with pytest.raises(ValueError):
        minimum_detectable_effect(100, alpha=0.03, power=0.85)
