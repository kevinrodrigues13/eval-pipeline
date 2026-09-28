"""
Fetches MT-Bench's published human judgments and GPT-4 verdicts, and maps
them onto this project's schema.

Running this once (`python -m eval_pipeline.mtbench`) is what makes every
later step runnable for free and reproducibly, before any paid judge call
happens: the human split gives calibration something real to measure
agreement against, and the gpt4_pair split gives a recorded judge that costs
nothing and never drifts between runs.

Source: lmsys/mt_bench_human_judgments on the HuggingFace Hub — 3355 human
judgments (mostly 3 judges per item) and 2400 GPT-4 judgments, ~1.4MB as
published (the files written to data/ are larger — see README), fetched via
the datasets-server REST API rather than the `datasets` package so this
module has no dependency beyond `requests`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, TypedDict

import requests

DATASET = "lmsys/mt_bench_human_judgments"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

DATA_DIR = Path("data")
PREFERENCES_PATH = DATA_DIR / "mtbench_preferences.jsonl"
VERDICTS_PATH = DATA_DIR / "mtbench_gpt4_verdicts.jsonl"
POLICY_OUTPUTS_PATH = DATA_DIR / "policy_outputs.jsonl"

# Held out entirely from calibration: never shown to the judge as a
# human-labelled example, only scored on. gpt-3.5-turbo vs vicuna-13b-v1.2 is
# a real pair in the corpus with both turns present for most questions.
HELD_OUT_PAIR = frozenset({"gpt-3.5-turbo", "vicuna-13b-v1.2"})

# The dataset's winner values map onto ours; "tie (bothbad)" is MT-Bench's own
# label for "neither response is good" — still a tie for our purposes, since
# neither response was preferred over the other.
WINNER_MAP = {"model_a": "A", "model_b": "B", "tie": "tie", "tie (bothbad)": "tie"}

# GPT-4 was run through MT-Bench's own position-swap protocol: judged once
# with each ordering, and "tie (inconsistent)" is MT-Bench's label for when
# the two orderings disagreed — not a genuine tie, a position-instability
# signal. Silently mapping it into WINNER_MAP would misreport 16% of the
# gpt4_pair split as ties; recording it separately is what calibration's
# position-flip-rate diagnostic needs. It is also its own kind of winner
# value, symmetric under a slot swap, so it needs no entry in _FLIP_WINNER.
GPT4_INCONSISTENT = "tie (inconsistent)"

_FLIP_WINNER = {"model_a": "model_b", "model_b": "model_a"}


class Turn(TypedDict):
    role: str
    content: str


def _get_with_retry(params: dict, max_attempts: int = 8) -> requests.Response:
    """The public datasets-server rate-limits bursts of requests with a 429."""
    for attempt in range(max_attempts):
        response = requests.get(ROWS_ENDPOINT, params=params, timeout=30)
        if response.status_code != 429:
            response.raise_for_status()
            return response
        wait = min(10 * 2 ** attempt, 90)
        print(f"mtbench: rate-limited, waiting {wait}s...")
        time.sleep(wait)
    response.raise_for_status()
    return response


def _fetch_split(split: str) -> List[dict]:
    rows: List[dict] = []
    offset = 0
    while True:
        response = _get_with_retry({
            "dataset": DATASET, "config": "default", "split": split,
            "offset": offset, "length": PAGE_SIZE,
        })
        batch = [entry["row"] for entry in response.json()["rows"]]
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE
        time.sleep(1.0)


def _prompt_and_context(
    conversation: List[Turn], turn: int
) -> Tuple[str, str, Tuple[Tuple[str, str], ...]]:
    """
    Split a rendered conversation into (prompt, response, prior context) for
    the specific turn this row is a judgment of.

    `conversation_a`/`conversation_b`, as published, always carry the FULL
    transcript (every turn, not just the one being judged) regardless of
    `turn` — verified directly against the raw dataset: a turn=1 row for a
    2-turn item has a 4-message `conversation_a`, identical to its turn=2
    counterpart, not the 2-message single-exchange one might expect. Slicing
    with a fixed `conversation[-2:]` is only correct for the *last* turn;
    for turn 1 of a multi-turn item it would silently return turn 2's
    question and answer instead. The 1-indexed `turn` says which (question,
    answer) pair — at zero-indexed messages `2*(turn-1)` and `2*(turn-1)+1`
    — is the one this row actually judges; everything strictly before it is
    prior-turn context, carried as (role, content) pairs so a judge sees the
    same history a human annotator did.
    """
    start = 2 * (turn - 1)
    prompt = conversation[start]["content"]
    response = conversation[start + 1]["content"]
    context = tuple((t["role"], t["content"]) for t in conversation[:start])
    return prompt, response, context


def _item_id(row: dict) -> str:
    return f"q{row['question_id']}-t{row['turn']}-{row['model_a']}-vs-{row['model_b']}"


def _is_held_out_pair(row: dict) -> bool:
    return {row["model_a"], row["model_b"]} == HELD_OUT_PAIR


def _canonical_row(row: dict) -> dict:
    """
    Normalise which system is `model_a`, for any pair.

    MT-Bench recorded each comparison from whichever slot assignment its own
    pipeline happened to use for that row — including, for some pairs,
    judging the identical (question, pair, turn) in both orders as its own
    position-bias control. Left alone, the same underlying comparison then
    appears under two different ids depending on which system landed in
    slot A: `load_comparisons` would treat two annotators' rows of one item
    as two separate single-annotator items instead of merging them, and a
    held-out pair's item would get scored twice instead of once. Fixing
    slot order to a canonical (alphabetical) choice before anything else
    touches the row collapses both problems at the source.
    """
    canonical_a, canonical_b = sorted((row["model_a"], row["model_b"]))
    if row["model_a"] == canonical_a:
        return row
    swapped = dict(row)
    swapped["model_a"], swapped["model_b"] = canonical_a, canonical_b
    swapped["conversation_a"], swapped["conversation_b"] = (
        row["conversation_b"], row["conversation_a"],
    )
    swapped["winner"] = _FLIP_WINNER.get(row["winner"], row["winner"])
    return swapped


def _comparison_record(row: dict) -> dict:
    row = _canonical_row(row)
    prompt, response_a, context_a = _prompt_and_context(row["conversation_a"], row["turn"])
    _, response_b, context_b = _prompt_and_context(row["conversation_b"], row["turn"])
    return {
        "id": _item_id(row),
        "cluster_id": f"q{row['question_id']}",
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "system_a": row["model_a"],
        "system_b": row["model_b"],
        "context_a": [list(pair) for pair in context_a],
        "context_b": [list(pair) for pair in context_b],
    }


def _preference_records(human_rows: List[dict]) -> Iterator[dict]:
    """One line per human annotation, for `load_comparisons` to merge by id."""
    for row in human_rows:
        row = _canonical_row(row)
        if _is_held_out_pair(row) or row["winner"] not in WINNER_MAP:
            continue
        record = _comparison_record(row)
        record["preferred"] = WINNER_MAP[row["winner"]]
        record["annotator_id"] = row["judge"]
        yield record


def _verdict_records(gpt4_rows: List[dict]) -> Iterator[dict]:
    """One line per item GPT-4 judged — every pair, held-out one included."""
    for row in gpt4_rows:
        row = _canonical_row(row)
        if row["winner"] == GPT4_INCONSISTENT:
            record = _comparison_record(row)
            record["winner"] = "tie"
            record["position_unstable"] = True
            yield record
            continue
        if row["winner"] not in WINNER_MAP:
            continue
        record = _comparison_record(row)
        record["winner"] = WINNER_MAP[row["winner"]]
        record["position_unstable"] = False
        yield record


def _policy_output_records(human_rows: List[dict]) -> Iterator[dict]:
    """
    The held-out pair's items, blind: no `preferred`, no `annotations`.

    Deduplicated across judges and across slot order — the human split
    repeats each held-out item once per annotator and sometimes once per
    slot ordering, but `_comparison_record` already canonicalises slot
    order, so two rows describing the same item always land on the same
    id here regardless of which order either was recorded in.
    """
    seen: Dict[str, dict] = {}
    for row in human_rows:
        if not _is_held_out_pair(row):
            continue
        record = _comparison_record(row)
        seen.setdefault(record["id"], record)
    return iter(seen.values())


def _write_jsonl(path: Path, records: Iterator[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
            count += 1
    return count


def build(force: bool = False) -> None:
    """Fetch MT-Bench and write the three derived files, unless already cached."""
    if not force and all(p.exists() for p in (PREFERENCES_PATH, VERDICTS_PATH, POLICY_OUTPUTS_PATH)):
        print("mtbench: already cached in data/ — pass force=True to refetch")
        return

    print(f"mtbench: fetching '{DATASET}' (human split)...")
    human_rows = _fetch_split("human")
    print(f"mtbench: fetching '{DATASET}' (gpt4_pair split)...")
    gpt4_rows = _fetch_split("gpt4_pair")

    n_preferences = _write_jsonl(PREFERENCES_PATH, _preference_records(human_rows))
    n_verdicts = _write_jsonl(VERDICTS_PATH, _verdict_records(gpt4_rows))
    n_policy = _write_jsonl(POLICY_OUTPUTS_PATH, _policy_output_records(human_rows))

    print(
        f"mtbench: wrote {n_preferences} preference annotations -> {PREFERENCES_PATH}\n"
        f"         wrote {n_verdicts} GPT-4 verdicts -> {VERDICTS_PATH}\n"
        f"         wrote {n_policy} held-out policy comparisons -> {POLICY_OUTPUTS_PATH}"
    )


if __name__ == "__main__":
    build()
