"""
A callable HTTP service wrapping the pipeline, for deployment on
DigitalOcean App Platform.

Recorded-judge mode (the default) spends nothing and needs no
authentication. Live-judge mode is also reachable over HTTP, but only
because it closes the specific gap that kept it out before: `cli.py`'s live
path is safe to expose interactively because a human sees a printed cost
estimate and has to type `y` before anything is spent; a stateless POST
request has no equivalent moment, and a publicly reachable endpoint that
can spend the operator's Anthropic budget on an unauthenticated request is
a real liability. Live mode substitutes for that confirmation with three
things a bare POST doesn't have: a shared secret only the operator knows
(`LIVE_JUDGE_API_KEY`, checked against the `X-Live-Judge-Key` header — and
if the operator never set it, live mode is unreachable regardless of what a
caller sends), payload caps an order of magnitude tighter than recorded
mode's, and a two-step flow — the same request with `confirm_spend: false`
returns the exact call count and cost estimate without spending a cent,
and only a second request with `confirm_spend: true` actually calls the
model. That's the same estimate-then-confirm shape as the CLI's `y/N`
prompt, just spread across two stateless requests instead of one
interactive one.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .calibration import CalibrationFailed, calibrate, gate_on_calibration
from .ingestion import comparisons_from_records
from .judge import LLMJudge, PrecomputedJudge
from .providers import DEFAULT_MODEL
from .report import generate_report
from .scoring import compare_policies
from .validation import flag_noisy_labels, sample_comparisons

# Bounds on one request's payload. Recorded mode spends no API budget, but
# an unbounded payload is still a CPU/memory liability on a publicly
# reachable endpoint — the cluster bootstrap alone is O(n_resamples) work
# per request.
MAX_PREFERENCE_RECORDS = 4000
MAX_POLICY_RECORDS = 1000
MAX_VERDICT_RECORDS = 5000  # preferences + policy_outputs, generously

# Live mode's caps are far tighter than recorded mode's: every row over
# them is real money, on an endpoint an authenticated caller could still
# hit by mistake with a much bigger payload than intended. Defense in
# depth on top of the LIVE_JUDGE_API_KEY gate, not a substitute for it.
MAX_LIVE_CALIBRATION_ITEMS = 50
MAX_LIVE_POLICY_RECORDS = 50

# How many live calls run in flight at once — bounds request wall-clock
# time on a platform with an HTTP request timeout, same mechanism as
# cli.py's --concurrency. A malformed reply is counted and excluded rather
# than aborting the whole request, since there is nobody watching this
# request to retry it by hand the way an operator running the CLI can.
LIVE_MODE_CONCURRENCY = 10

LIVE_JUDGE_KEY_ENV = "LIVE_JUDGE_API_KEY"
LIVE_JUDGE_KEY_HEADER = "X-Live-Judge-Key"


class EvaluateRequest(BaseModel):
    preferences: List[dict] = Field(..., description="human-labelled comparisons")
    policy_outputs: List[dict] = Field(..., description="held-out comparisons to score")
    verdicts: List[dict] = Field(
        default_factory=list,
        description="recorded judge verdicts covering both sets (required when judge_mode='recorded')",
    )
    calibration_items: int = Field(200, ge=0, description="0 uses every preference record")
    policy_1_name: str = "Policy 1"
    policy_2_name: str = "Policy 2"
    judge_name: str = "recorded"
    seed: int = 7
    judge_mode: Literal["recorded", "live"] = "recorded"
    model: str = Field(DEFAULT_MODEL, description="live mode only")
    confirm_spend: bool = Field(
        False,
        description="live mode only — false returns a cost estimate and spends nothing; "
        "true actually calls the model",
    )


class EvaluateResponse(BaseModel):
    judge_mode: str
    spend_confirmed: bool = True
    estimated_cost: Optional[float] = None
    n_live_calls: Optional[int] = None
    report_markdown: Optional[str] = None
    gate_passed: Optional[bool] = None
    calibration_summary: Optional[str] = None
    score_summary: Optional[Dict[str, Optional[float]]] = None
    conflicting_verdicts: Optional[int] = None


def _live_cost_estimate(rows, policy_rows, model: str) -> Dict[str, float]:
    from .providers import estimate_cost

    n_calls = (len(rows) + len(policy_rows)) * 2
    characters = sum(
        len(r.prompt) + len(r.response_a) + len(r.response_b) for r in rows
    ) / max(len(rows), 1)
    avg_input_tokens = int(characters / 4) + 300
    return {"n_calls": n_calls, "cost": estimate_cost(n_calls, avg_input_tokens, model)}


def _build_live_judge(model: str, judge_name: str) -> LLMJudge:
    from .providers import anthropic_judge_backend

    try:
        return LLMJudge(anthropic_judge_backend(model=model), name=judge_name)
    except Exception as exc:  # anthropic missing, malformed model/cache config, etc.
        raise HTTPException(503, f"live judge unavailable on this deployment: {exc}") from None


def _run_live_calls(judge: LLMJudge, rows, policy_rows, quality):
    """
    The anthropic SDK doesn't validate ANTHROPIC_API_KEY (or catch a network
    failure, rate limit, etc.) at client construction — it raises on the
    first actual `.messages.create()` call, which happens deep inside
    `calibrate`/`compare_policies`, past `_build_live_judge`'s try/except.
    Confirmed live: a missing key surfaced as a raw, unhandled `TypeError`
    from inside the SDK, which FastAPI would otherwise turn into a bare 500
    with a stack trace leaked to the caller. Every live-call failure —
    missing key, network error, a bad `model` the caller passed — is
    caught here and turned into a clean error instead.
    """
    try:
        calibration = calibrate(judge, rows, label_quality=quality, skip_parse_errors=True)
        score = compare_policies(
            judge, policy_rows, max_workers=LIVE_MODE_CONCURRENCY, skip_parse_errors=True
        )
    except Exception as exc:
        raise HTTPException(502, f"live judge call failed: {exc}") from None
    return calibration, score


def run_evaluation(payload: EvaluateRequest, live_judge_key: Optional[str] = None) -> EvaluateResponse:
    """The request handler's logic, kept separate from FastAPI plumbing so it's testable without an HTTP client."""
    import os

    if len(payload.preferences) > MAX_PREFERENCE_RECORDS:
        raise HTTPException(413, f"preferences exceeds {MAX_PREFERENCE_RECORDS} records")
    if len(payload.policy_outputs) > MAX_POLICY_RECORDS:
        raise HTTPException(413, f"policy_outputs exceeds {MAX_POLICY_RECORDS} records")

    if payload.judge_mode == "live":
        configured_key = os.environ.get(LIVE_JUDGE_KEY_ENV)
        if not configured_key:
            raise HTTPException(503, "live judge is not enabled on this deployment")
        if live_judge_key != configured_key:
            raise HTTPException(401, f"missing or incorrect {LIVE_JUDGE_KEY_HEADER} header")
    else:
        if len(payload.verdicts) > MAX_VERDICT_RECORDS:
            raise HTTPException(413, f"verdicts exceeds {MAX_VERDICT_RECORDS} records")
        if not payload.verdicts:
            raise HTTPException(400, "verdicts is required when judge_mode='recorded'")

    all_rows, _ = comparisons_from_records(payload.preferences)
    rows = sample_comparisons(all_rows, payload.calibration_items, seed=payload.seed)
    quality = flag_noisy_labels(rows)
    policy_rows, _ = comparisons_from_records(payload.policy_outputs)

    if payload.judge_mode == "live":
        if len(rows) > MAX_LIVE_CALIBRATION_ITEMS:
            raise HTTPException(
                413,
                f"live mode: calibration sample ({len(rows)}) exceeds "
                f"{MAX_LIVE_CALIBRATION_ITEMS} — pass a smaller calibration_items",
            )
        if len(policy_rows) > MAX_LIVE_POLICY_RECORDS:
            raise HTTPException(
                413, f"live mode: policy_outputs exceeds {MAX_LIVE_POLICY_RECORDS} records"
            )

        estimate = _live_cost_estimate(rows, policy_rows, payload.model)
        if not payload.confirm_spend:
            return EvaluateResponse(
                judge_mode="live",
                spend_confirmed=False,
                estimated_cost=estimate["cost"],
                n_live_calls=estimate["n_calls"],
            )

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(503, "live judge unavailable on this deployment: ANTHROPIC_API_KEY is not set")

        judge = _build_live_judge(payload.model, payload.judge_name or payload.model)
        calibration, score = _run_live_calls(judge, rows, policy_rows, quality)
    else:
        judge = PrecomputedJudge.from_records(payload.verdicts, strict=False, name=payload.judge_name)
        calibration = calibrate(judge, rows, label_quality=quality)
        score = compare_policies(judge, policy_rows)
        estimate = None

    gate_passed = True
    try:
        gate_on_calibration(calibration)
    except CalibrationFailed:
        gate_passed = False

    markdown = generate_report(
        calibration, score, label_quality=quality,
        policy_names=(payload.policy_1_name, payload.policy_2_name),
    )

    return EvaluateResponse(
        judge_mode=payload.judge_mode,
        spend_confirmed=True,
        estimated_cost=estimate["cost"] if estimate else None,
        n_live_calls=estimate["n_calls"] if estimate else None,
        report_markdown=markdown,
        gate_passed=gate_passed,
        calibration_summary=calibration.summary(),
        score_summary={
            "win_rate_policy1": score.win_rate_policy1,
            "ci_lower": score.ci_lower,
            "ci_upper": score.ci_upper,
            "n_scored": score.n_scored,
        },
        conflicting_verdicts=getattr(judge, "conflicts", 0),
    )


app = FastAPI(
    title="Evaluation & Judge Calibration Pipeline",
    description="Recorded- and live-judge evaluation, callable over HTTP. See README.md for the full pipeline.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(
    payload: EvaluateRequest,
    x_live_judge_key: Optional[str] = Header(None, alias=LIVE_JUDGE_KEY_HEADER),
) -> EvaluateResponse:
    return run_evaluation(payload, live_judge_key=x_live_judge_key)
