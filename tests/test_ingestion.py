import json

from eval_pipeline.ingestion import comparisons_from_records, load_comparisons, load_verdicts
from eval_pipeline.schema import Annotation, Comparison


def write(tmp_path, records):
    path = tmp_path / "comparisons.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def test_basic_single_preferred_tuple_round_trips(tmp_path):
    path = write(tmp_path, [
        {"id": "c0", "prompt": "q", "response_a": "a", "response_b": "b", "preferred": "A"},
    ])
    rows, report = load_comparisons(path)

    assert len(rows) == 1
    assert rows[0] == Comparison(
        row_id="c0", prompt="q", response_a="a", response_b="b",
        annotations=(Annotation(preferred="A", annotator_id=None),),
        cluster_id="c0",
    )
    assert report.n_rows == 1
    assert report.n_annotations == 1
    assert report.n_repeat_labelled_items == 0


def test_multi_annotator_items_are_counted_as_repeat_labelled(tmp_path):
    path = write(tmp_path, [
        {
            "id": "c0", "prompt": "q", "response_a": "a", "response_b": "b",
            "annotations": [
                {"preferred": "A", "annotator_id": "e1"},
                {"preferred": "B", "annotator_id": "e2"},
            ],
        },
        {"id": "c1", "prompt": "q2", "response_a": "a", "response_b": "b", "preferred": "tie"},
    ])
    rows, report = load_comparisons(path)

    assert report.n_rows == 2
    assert report.n_annotations == 3
    assert report.n_repeat_labelled_items == 1
    assert "1 (50%)" in report.summary()


def test_majority_distinguishes_tie_from_split():
    tie = Comparison(
        row_id="c0", prompt="q", response_a="a", response_b="b",
        annotations=(Annotation("tie"), Annotation("tie")),
    )
    split = Comparison(
        row_id="c1", prompt="q", response_a="a", response_b="b",
        annotations=(Annotation("A"), Annotation("B")),
    )
    decisive = Comparison(
        row_id="c2", prompt="q", response_a="a", response_b="b",
        annotations=(Annotation("A"), Annotation("A"), Annotation("B")),
    )
    unlabelled = Comparison(row_id="c3", prompt="q", response_a="a", response_b="b")

    assert tie.majority == "tie"
    assert split.majority == "split"
    assert decisive.majority == "A"
    assert unlabelled.majority is None


def test_cluster_id_defaults_to_row_id_when_absent(tmp_path):
    path = write(tmp_path, [
        {"id": "c0", "prompt": "q", "response_a": "a", "response_b": "b", "preferred": "A"},
        {"id": "c1", "prompt": "q2", "response_a": "a", "response_b": "b", "preferred": "B",
         "cluster_id": "shared"},
    ])
    rows, _ = load_comparisons(path)

    assert rows[0].cluster_id == "c0"
    assert rows[1].cluster_id == "shared"


def test_context_pairs_round_trip_as_tuples(tmp_path):
    path = write(tmp_path, [
        {
            "id": "c0", "prompt": "turn 2", "response_a": "a", "response_b": "b",
            "preferred": "A",
            "context_a": [["user", "turn 1 question"], ["assistant", "turn 1 answer a"]],
            "context_b": [["user", "turn 1 question"], ["assistant", "turn 1 answer b"]],
        },
    ])
    rows, _ = load_comparisons(path)

    assert rows[0].context_a == (("user", "turn 1 question"), ("assistant", "turn 1 answer a"))
    assert rows[0].context_b == (("user", "turn 1 question"), ("assistant", "turn 1 answer b"))


def test_repeated_rows_sharing_an_id_merge_into_one_comparison(tmp_path):
    """One line per annotator, not one line per item — the common raw shape."""
    path = write(tmp_path, [
        {"id": "c0", "prompt": "q", "response_a": "a", "response_b": "b",
         "preferred": "A", "annotator_id": "e1"},
        {"id": "c0", "prompt": "q", "response_a": "a", "response_b": "b",
         "preferred": "B", "annotator_id": "e2"},
        {"id": "c0", "prompt": "q", "response_a": "a", "response_b": "b",
         "preferred": "A", "annotator_id": "e3"},
    ])
    rows, report = load_comparisons(path)

    assert len(rows) == 1
    assert rows[0].row_id == "c0"
    assert rows[0].annotations == (
        Annotation("A", "e1"), Annotation("B", "e2"), Annotation("A", "e3"),
    )
    assert rows[0].majority == "A"
    assert report.n_rows == 1
    assert report.n_annotations == 3
    assert report.n_repeat_labelled_items == 1


def test_repeated_rows_with_no_id_merge_by_content(tmp_path):
    path = write(tmp_path, [
        {"prompt": "q", "response_a": "a", "response_b": "b", "preferred": "tie", "annotator_id": "e1"},
        {"prompt": "q", "response_a": "a", "response_b": "b", "preferred": "tie", "annotator_id": "e2"},
        {"prompt": "different question", "response_a": "a", "response_b": "b", "preferred": "A"},
    ])
    rows, report = load_comparisons(path)

    assert len(rows) == 2
    assert len(rows[0].annotations) == 2
    assert rows[0].majority == "tie"
    assert report.n_rows == 2


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "comparisons.jsonl"
    path.write_text(
        json.dumps({"id": "c0", "prompt": "q", "response_a": "a", "response_b": "b", "preferred": "A"})
        + "\n\n\n",
        encoding="utf-8",
    )
    rows, report = load_comparisons(path)
    assert len(rows) == 1
    assert report.n_rows == 1


def test_comparisons_from_records_works_without_a_file():
    # What service.py builds on: a request body's already-parsed records,
    # no file in between.
    records = [
        {"id": "c0", "prompt": "q", "response_a": "a", "response_b": "b", "preferred": "A"},
        {"id": "c0", "prompt": "q", "response_a": "a", "response_b": "b", "preferred": "B"},
    ]
    rows, report = comparisons_from_records(records)
    assert len(rows) == 1
    assert len(rows[0].annotations) == 2
    assert report.n_repeat_labelled_items == 1


def test_explicit_id_never_collides_with_a_content_grouped_fallback_id():
    # Regression test: the old fallback row_id was a bare positional index
    # ("0", "1", "2", ...) — the same namespace real explicit ids often use.
    # An explicit id "1" used to collide with whichever content-grouped
    # (no-id) item happened to land at position 1.
    records = [
        {"id": "1", "prompt": "explicit", "response_a": "a", "response_b": "b", "preferred": "A"},
        {"prompt": "no-id-item-A", "response_a": "x", "response_b": "y", "preferred": "A"},
    ]
    rows, _ = comparisons_from_records(records)
    assert len(rows) == 2
    row_ids = {r.row_id for r in rows}
    assert len(row_ids) == 2  # no collision
    assert "1" in row_ids


def test_explicit_null_id_falls_back_to_content_grouping_not_a_shared_none_id():
    # dict.get(key, default) only returns `default` when the key is ABSENT —
    # an explicit `"id": null` is present with value None, so the old
    # `str(record.get("id", index))` fallback returned the literal string
    # "None" for every null-id record, colliding them all together.
    records = [
        {"id": None, "prompt": "q1", "response_a": "a", "response_b": "b", "preferred": "A"},
        {"id": None, "prompt": "q2", "response_a": "c", "response_b": "d", "preferred": "B"},
    ]
    rows, _ = comparisons_from_records(records)
    assert len(rows) == 2
    assert rows[0].row_id != rows[1].row_id
    assert "None" not in (rows[0].row_id, rows[1].row_id)


def test_load_verdicts_reads_one_record_per_line(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in [
            {"id": "c0", "winner": "A", "position_unstable": False},
            {"id": "c1", "winner": "tie", "position_unstable": True},
        ]),
        encoding="utf-8",
    )
    records = load_verdicts(path)
    assert records == [
        {"id": "c0", "winner": "A", "position_unstable": False},
        {"id": "c1", "winner": "tie", "position_unstable": True},
    ]


def test_load_verdicts_skips_blank_lines(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    path.write_text(
        json.dumps({"id": "c0", "winner": "A"}) + "\n\n\n",
        encoding="utf-8",
    )
    assert len(load_verdicts(path)) == 1
