"""
The judge mechanism: an interface, and two implementations.

Position bias is defended twice, independently. The prompt itself
(prompts.SYSTEM_PROMPT) instructs the judge to ignore slot order. That is
an instruction, not a guarantee, so every comparison is also asked a second
time with the responses swapped — a judge that is not truly order-invariant
will sometimes contradict itself, and that contradiction is measured
(`position_unstable`) rather than averaged away.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .prompts import render_prompt, render_system_prompt
from .schema import Comparison

_VERDICT_PATTERN = re.compile(r"\[\[\s*([ABC])\s*\]\]")
_LETTER_TO_WINNER = {"A": "A", "B": "B", "C": "tie"}
_FLIP = {"A": "B", "B": "A", "tie": "tie"}


class JudgeParseError(ValueError):
    """A judge's raw reply carried no parsable [[A]]/[[B]]/[[C]] verdict."""


def parse_verdict(text: str) -> str:
    """
    Extract the judge's final verdict, letting the last match win.

    A judge sometimes reasons its way to a tentative answer, then restates
    a different one at the end of the same reply — whatever it wrote last
    is what it meant to submit, so the first match is not good enough.
    """
    matches = _VERDICT_PATTERN.findall(text)
    if not matches:
        raise JudgeParseError(f"no [[A]]/[[B]]/[[C]] verdict found in: {text!r}")
    return _LETTER_TO_WINNER[matches[-1]]


@dataclass(frozen=True)
class Verdict:
    """
    One judge call's result.

    `position_unstable` is `None` for a judge whose single call doesn't
    know the answer (an LLM judge — the wrapper below has to ask twice to
    find out) and a real bool for one that already does (a precomputed
    judge replaying a source that ran its own position-swap protocol).
    """

    winner: str
    position_unstable: Optional[bool] = None
    raw: Optional[str] = None


@dataclass(frozen=True)
class JudgedComparison:
    row_id: str
    winner: str
    position_unstable: bool
    raw: Tuple[Verdict, ...] = ()
    #: The judge never returned a parsable verdict for this comparison (in
    #: either the first call or, for a non-order-invariant judge, the
    #: swapped one) and `skip_parse_errors` was set, so it was counted and
    #: excluded rather than aborting the whole run. Distinct from
    #: `position_unstable`: that means the judge answered consistently or
    #: not; this means it never answered at all.
    parse_failed: bool = False


class Judge(ABC):
    name: str

    #: Whether one `verdict()` call already accounts for position effects,
    #: so `judge_with_position_check` should trust it rather than asking a
    #: second time with the slots swapped.
    is_order_invariant: bool = False

    @abstractmethod
    def verdict(self, comparison: Comparison) -> Verdict:
        ...


def _swap_slots(comparison: Comparison) -> Comparison:
    return replace(
        comparison,
        response_a=comparison.response_b,
        response_b=comparison.response_a,
        system_a=comparison.system_b,
        system_b=comparison.system_a,
        context_a=comparison.context_b,
        context_b=comparison.context_a,
    )


def _parse_failed(comparison: Comparison) -> JudgedComparison:
    return JudgedComparison(
        row_id=comparison.row_id, winner="tie", position_unstable=False, parse_failed=True,
    )


def judge_with_position_check(
    judge: Judge, comparison: Comparison, skip_parse_errors: bool = False
) -> JudgedComparison:
    """
    Get one judged outcome for a comparison, defended against position bias.

    An order-invariant judge is trusted after one call. Anything else is
    asked twice — once as given, once with the slots swapped — and the two
    answers are reconciled in the *original* slot's terms: if the swapped
    call's "A" is really the original response B, a genuinely consistent
    judge should have said "B" both times. Disagreement becomes a tie
    rather than an arbitrary pick of one call over the other, matching how
    MT-Bench's own GPT-4 verdicts record this same situation
    ("tie (inconsistent)") — see mtbench.GPT4_INCONSISTENT.

    `skip_parse_errors` mirrors the same-named CLI flag: off by default, a
    malformed reply is loud on purpose (a judge that never produced a usable
    verdict for whatever it was just asked is worth knowing about, not
    quietly papering over). With it on, a `JudgeParseError` — the live model
    wrote a full reply but never emitted [[A]]/[[B]]/[[C]] — is caught and
    the comparison is excluded (`parse_failed=True`) instead of aborting a
    run that may already represent real, paid-for API calls on every row
    before it.
    """
    try:
        first = judge.verdict(comparison)
    except JudgeParseError:
        if not skip_parse_errors:
            raise
        return _parse_failed(comparison)

    if judge.is_order_invariant:
        return JudgedComparison(
            row_id=comparison.row_id,
            winner=first.winner,
            position_unstable=bool(first.position_unstable),
            raw=(first,),
        )

    try:
        second = judge.verdict(_swap_slots(comparison))
    except JudgeParseError:
        if not skip_parse_errors:
            raise
        return _parse_failed(comparison)

    second_in_original_terms = _FLIP[second.winner]
    consistent = first.winner == second_in_original_terms
    winner = first.winner if consistent else "tie"
    return JudgedComparison(
        row_id=comparison.row_id,
        winner=winner,
        position_unstable=not consistent,
        raw=(first, second),
    )


def judge_many(
    judge: Judge,
    rows: Sequence[Comparison],
    max_workers: int = 1,
    skip_parse_errors: bool = False,
) -> List[JudgedComparison]:
    """
    Judge every row, optionally concurrently.

    `max_workers=1` (default) runs strictly sequentially — identical timing
    and behaviour to code that never knew this parameter existed. Judge
    calls are the pipeline's actual bottleneck (network round-trips to a
    live model), and each one is independent of every other, so there is
    no reproducibility hazard in running them concurrently the way there
    would be for anything involving shared random state: `judge.verdict`
    is a pure function of one comparison, and `ThreadPoolExecutor.map`
    returns results in call order regardless of completion order, so
    everything downstream that matches judged results back to rows
    positionally is unaffected by worker count.

    One real tradeoff at `max_workers > 1`: `ThreadPoolExecutor.map` submits
    every row's call *before* consuming any result, so a `JudgeParseError`
    on row N no longer stops the run after N paid-for calls the way the
    sequential path does — every call already dispatched by the time the
    failing one is reached still executes and gets billed before the
    exception propagates. `skip_parse_errors=True` is the fix either way (a
    bad row is counted and excluded instead of raising at all), and is
    worth turning on by default whenever concurrency is, on a live judge.
    """
    if max_workers <= 1:
        return [
            judge_with_position_check(judge, row, skip_parse_errors=skip_parse_errors)
            for row in rows
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(
            lambda row: judge_with_position_check(judge, row, skip_parse_errors=skip_parse_errors),
            rows,
        ))


class LLMJudge(Judge):
    """
    A judge backed by a live model call.

    `backend` is injected rather than imported directly (the Anthropic SDK, an HTTP
    client, whatever) — this class only needs something that turns a
    (system_prompt, user_prompt) pair into raw text, which makes it testable
    with a plain function and keeps provider/cost/caching concerns entirely
    out of the judging logic itself.
    """

    is_order_invariant = False

    def __init__(self, backend: Callable[[str, str], str], name: str = "llm-judge"):
        self.name = name
        self._backend = backend

    def verdict(self, comparison: Comparison) -> Verdict:
        raw = self._backend(render_system_prompt(comparison), render_prompt(comparison))
        return Verdict(winner=parse_verdict(raw), raw=raw)


class PrecomputedJudge(Judge):
    """
    Replays already-computed verdicts instead of calling a live model: free,
    deterministic, and identical across runs. Used both to validate the
    pipeline for nothing (MT-Bench's own GPT-4 verdicts) and to replay a
    live run's results without re-spending on it.

    The recorded winner is already position-resolved by whatever produced
    it, so this judge is order-invariant by construction — asking it twice
    with the slots swapped would just look up the same row again.
    """

    is_order_invariant = True

    def __init__(
        self,
        verdicts: Dict[str, Tuple[str, bool]],
        conflicts: int = 0,
        name: str = "recorded",
    ):
        self.name = name
        self._verdicts = verdicts
        self.conflicts = conflicts

    @classmethod
    def from_records(
        cls, records: Sequence[dict], strict: bool = True, name: str = "recorded"
    ) -> "PrecomputedJudge":
        """
        `strict=True` raises on the same id carrying two different recorded
        verdicts — a real data-quality problem, not something to paper over.
        `strict=False` instead resolves the conflict to a flagged tie (can't
        trust either recorded answer) and counts it, for a caller that would
        rather report the conflict than abort the run.
        """
        table: Dict[str, Tuple[str, bool]] = {}
        conflicting_ids = set()
        for record in records:
            key = str(record["id"])
            entry = (record["winner"], bool(record.get("position_unstable", False)))
            if key in table and table[key] != entry:
                conflicting_ids.add(key)
                if strict:
                    raise ValueError(f"conflicting recorded verdicts for id={key!r}")
                table[key] = ("tie", True)
                continue
            table[key] = entry
        return cls(table, conflicts=len(conflicting_ids), name=name)

    def verdict(self, comparison: Comparison) -> Verdict:
        winner, position_unstable = self._verdicts[comparison.row_id]
        return Verdict(winner=winner, position_unstable=position_unstable)
