"""Loading pairwise-comparison data from JSONL into the Comparison schema."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Optional, Tuple, TypedDict, Union

from .schema import Annotation, Comparison


class AnnotationRecord(TypedDict, total=False):
    """One human verdict, as it appears in an `annotations` list or inline."""

    preferred: str
    annotator_id: Optional[str]


class ComparisonRecord(TypedDict, total=False):
    """
    One JSONL line as `load_comparisons` accepts it.

    `id` and `cluster_id` are strings if present, but any JSON-hashable
    value round-trips through `str()` in practice. Every key is optional at
    the type level because the two label shapes are mutually exclusive
    (`preferred` alone, or an `annotations` list) and everything past
    `response_b` is genuinely optional metadata.
    """

    id: str
    prompt: str
    response_a: str
    response_b: str
    preferred: str
    annotator_id: str
    annotations: List[AnnotationRecord]
    cluster_id: str
    system_a: str
    system_b: str
    context_a: List[List[str]]
    context_b: List[List[str]]


@dataclass
class LoadReport:
    """What came in, so a bad file fails loudly instead of silently scoring wrong."""

    n_rows: int
    n_annotations: int
    n_repeat_labelled_items: int

    def summary(self) -> str:
        share = (self.n_repeat_labelled_items / self.n_rows) if self.n_rows else 0.0
        return (
            f"{self.n_rows} comparisons, {self.n_annotations} annotations, "
            f"{self.n_repeat_labelled_items} ({share:.0%}) labelled more than once"
        )


def _annotations_from_record(record: ComparisonRecord) -> Tuple[Annotation, ...]:
    """
    Accept either shape a comparison's labels can arrive in.

    `annotations`: a list of {preferred, annotator_id} — multiple humans
    labelled this item. `preferred`: a single string — the basic
    (prompt, response_A, response_B, preferred) tuple the brief asks for.
    Both normalise to the same `Comparison.annotations` tuple, so nothing
    downstream needs to know which shape the file used.
    """
    if record.get("annotations") is not None:
        return tuple(
            Annotation(preferred=a["preferred"], annotator_id=a.get("annotator_id"))
            for a in record["annotations"]
        )
    if record.get("preferred") is not None:
        return (Annotation(preferred=record["preferred"], annotator_id=record.get("annotator_id")),)
    return ()


def _context_from_record(record: ComparisonRecord, key: str) -> Tuple[Tuple[str, str], ...]:
    return tuple(tuple(pair) for pair in record.get(key) or [])


def _group_key(record: ComparisonRecord) -> Hashable:
    """
    Identify which rows describe the same item, so their annotations merge.

    Raw preference data is commonly one line per *annotation*, not one line
    per item — the same (prompt, response_A, response_B) triple repeated
    once per human who labelled it. An explicit `id` is the reliable key
    when present; lacking one, the content triple itself is the only honest
    key, since two lines with the same prompt and responses but no shared id
    are, for every purpose downstream, the same comparison.
    """
    if record.get("id") is not None:
        return ("id", record["id"])
    return ("content", record["prompt"], record["response_a"], record["response_b"])


def _row_id_for_key(key: Hashable) -> str:
    """
    A row_id derived from the same key `_group_key` grouped on, so it can
    never collide between an explicit-id group and a content-grouped one.

    The earlier scheme (`str(record.get("id", index))`) used a bare
    positional index as the fallback for content-grouped items — the same
    small-integer string namespace ("0", "1", "2", ...) real explicit ids
    often also use, so an explicit `"id": "1"` could collide with whichever
    content-grouped item happened to land at position 1. It also mishandled
    an explicit `"id": null`: that key is *present* with value `None`, so
    `dict.get`'s default never applies to it — every null-id group collided
    on the literal row_id `"None"`. Keying off `_group_key`'s own tagged
    tuple instead ties row_id uniqueness to exactly the same distinction
    that decided grouping, with no separate namespace to collide in.
    """
    kind, *rest = key
    if kind == "id":
        return str(rest[0])
    return "__" + "\x1f".join(str(part) for part in rest)


def _comparison_from_group(records: List[ComparisonRecord], row_id: str) -> Comparison:
    base = records[0]
    annotations: Tuple[Annotation, ...] = tuple(
        annotation for record in records for annotation in _annotations_from_record(record)
    )
    return Comparison(
        row_id=row_id,
        prompt=base["prompt"],
        response_a=base["response_a"],
        response_b=base["response_b"],
        annotations=annotations,
        cluster_id=str(base["cluster_id"]) if base.get("cluster_id") else row_id,
        system_a=base.get("system_a"),
        system_b=base.get("system_b"),
        context_a=_context_from_record(base, "context_a"),
        context_b=_context_from_record(base, "context_b"),
    )


class VerdictRecord(TypedDict, total=False):
    """One already-computed judge verdict, as `PrecomputedJudge` accepts it."""

    id: str
    prompt: str
    response_a: str
    response_b: str
    winner: str
    position_unstable: bool
    cluster_id: str
    system_a: str
    system_b: str


def load_verdicts(path: Union[str, Path]) -> List[VerdictRecord]:
    """
    Read one JSONL file of already-computed judge verdicts.

    Unlike `load_comparisons`, no grouping is needed: a verdicts file is
    already one row per item — it is the *output* of judging, not raw
    labels multiple annotators each contributed a line for.
    """
    records: List[VerdictRecord] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def comparisons_from_records(records: Iterable[ComparisonRecord]) -> Tuple[List[Comparison], LoadReport]:
    """
    The record-merging core of `load_comparisons`, usable directly on
    already-parsed records — e.g. a service.py request body — without a
    file in between.

    Records sharing an item — same `id`, or lacking one, the same
    (prompt, response_A, response_B) triple — are merged into a single
    `Comparison` carrying every annotation collected across them, rather
    than becoming duplicate rows for the same item.
    """
    groups: Dict[Hashable, List[ComparisonRecord]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)

    # A plain dict already preserves first-seen key order (guaranteed since
    # Python 3.7, and this project requires >=3.9) — no separate list needed
    # to remember it.
    rows = [_comparison_from_group(items, _row_id_for_key(key)) for key, items in groups.items()]

    report = LoadReport(
        n_rows=len(rows),
        n_annotations=sum(len(row.annotations) for row in rows),
        n_repeat_labelled_items=sum(1 for row in rows if len(row.annotations) >= 2),
    )
    return rows, report


def load_comparisons(path: Union[str, Path]) -> Tuple[List[Comparison], LoadReport]:
    """Read one JSONL file of comparisons — see `comparisons_from_records`."""
    records: List[ComparisonRecord] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return comparisons_from_records(records)
