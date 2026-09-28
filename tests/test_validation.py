from eval_pipeline.schema import Annotation, Comparison
from eval_pipeline.validation import (
    MIN_REPEAT_LABELLED_FOR_CEILING,
    flag_noisy_labels,
    is_contested,
    sample_comparisons,
)


def item(row_id, *preferred_values):
    return Comparison(
        row_id=row_id, prompt="q", response_a="a", response_b="b",
        annotations=tuple(Annotation(p) for p in preferred_values),
    )


def test_unanimous_items_are_not_contested():
    assert not is_contested(item("c0", "tie", "tie"))
    assert not is_contested(item("c1", "A", "A", "A"))


def test_a_clear_majority_is_still_contested_if_not_unanimous():
    # 2-1 for A: a real majority, but the annotators did not agree.
    assert is_contested(item("c0", "A", "A", "B"))


def test_a_split_with_no_majority_is_contested():
    assert is_contested(item("c0", "A", "B"))


def test_single_annotation_item_is_not_contested():
    assert not is_contested(item("c0", "A"))


def test_tie_rate_counts_at_the_annotation_level():
    rows = [item("c0", "tie", "A"), item("c1", "B")]
    report = flag_noisy_labels(rows)
    assert report.n_annotations == 3
    assert report.n_ties == 1
    assert report.tie_rate == 1 / 3


def test_contested_items_and_annotations_are_counted():
    rows = [item("c0", "A", "A", "B"), item("c1", "tie", "tie"), item("c2", "A")]
    report = flag_noisy_labels(rows)
    assert report.n_repeat_labelled == 2  # c0 and c1; c2 has only one annotation
    assert report.n_contested_items == 1  # only c0
    assert report.n_contested_annotations == 3  # all three of c0's annotations


def test_human_agreement_is_pairwise_and_withheld_below_the_floor():
    # One repeat-labelled item, agreeing: below the floor, so withheld.
    rows = [item("c0", "A", "A")]
    report = flag_noisy_labels(rows)
    assert report.n_repeat_labelled < MIN_REPEAT_LABELLED_FOR_CEILING
    assert report.human_agreement is None
    assert "no human-human ceiling" in report.summary()


def test_human_agreement_computed_once_enough_repeat_labelled_items_exist():
    # 5 items, each with 2 annotations: 4 agree, 1 disagrees -> 4/5 pairwise agreement.
    rows = [item(f"c{i}", "A", "A") for i in range(4)] + [item("c4", "A", "B")]
    report = flag_noisy_labels(rows)
    assert report.n_repeat_labelled == MIN_REPEAT_LABELLED_FOR_CEILING
    assert report.human_agreement == 4 / 5
    assert "80.0% human-human ceiling" in report.summary()


def test_three_way_item_compares_every_pair():
    # [A, A, B]: pairs are (A,A) agree, (A,B) disagree, (A,B) disagree -> 1/3.
    rows = [item(f"c{i}", "A", "A") for i in range(4)] + [item("c4", "A", "A", "B")]
    report = flag_noisy_labels(rows)
    assert report.n_compared_pairs == 4 + 3
    assert report.n_agreeing_pairs == 4 + 1


def test_sample_comparisons_is_reproducible_and_keeps_whole_items():
    rows = [item(f"c{i}", "A", "B") for i in range(20)]
    sample_1 = sample_comparisons(rows, 5, seed=7)
    sample_2 = sample_comparisons(rows, 5, seed=7)
    assert [r.row_id for r in sample_1] == [r.row_id for r in sample_2]
    assert len(sample_1) == 5
    assert all(len(r.annotations) == 2 for r in sample_1)  # nothing stripped


def test_sample_comparisons_returns_everything_when_n_is_not_smaller():
    rows = [item(f"c{i}") for i in range(3)]
    assert sample_comparisons(rows, 0) == rows
    assert sample_comparisons(rows, 100) == rows
