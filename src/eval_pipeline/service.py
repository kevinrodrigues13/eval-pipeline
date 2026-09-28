"""
A callable HTTP service wrapping the pipeline, for deployment on
DigitalOcean App Platform.

Deliberately recorded-judge only — no live model call is reachable over
HTTP. cli.py's live path is safe to expose interactively because a human
sees the cost estimate and types y/N before anything is spent; a stateless
POST request has no equivalent confirmation step, and a publicly reachable
endpoint that can spend the operator's API budget on an unauthenticated
request is a real liability, not a hypothetical one. A caller who wants a
live judge runs the CLI, which is what the confirmation gate exists for.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .calibration import CalibrationFailed, calibrate, gate_on_calibration
from .ingestion import comparisons_from_records
from .judge import PrecomputedJudge
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


class EvaluateRequest(BaseModel):
    preferences: List[dict] = Field(..., description="human-labelled comparisons")
    policy_outputs: List[dict] = Field(..., description="held-out comparisons to score")
    verdicts: List[dict] = Field(..., description="recorded judge verdicts covering both sets")
    calibration_items: int = Field(200, ge=0, description="0 uses every preference record")
    policy_1_name: str = "Policy 1"
    policy_2_name: str = "Policy 2"
    judge_name: str = "recorded"
    seed: int = 7


class EvaluateResponse(BaseModel):
    report_markdown: str
    gate_passed: bool
    calibration_summary: str
    score_summary: Dict[str, Optional[float]]
    conflicting_verdicts: int


def run_evaluation(payload: EvaluateRequest) -> EvaluateResponse:
    """The request handler's logic, kept separate from FastAPI plumbing so it's testable without an HTTP client."""
    if len(payload.preferences) > MAX_PREFERENCE_RECORDS:
        raise HTTPException(413, f"preferences exceeds {MAX_PREFERENCE_RECORDS} records")
    if len(payload.policy_outputs) > MAX_POLICY_RECORDS:
        raise HTTPException(413, f"policy_outputs exceeds {MAX_POLICY_RECORDS} records")
    if len(payload.verdicts) > MAX_VERDICT_RECORDS:
        raise HTTPException(413, f"verdicts exceeds {MAX_VERDICT_RECORDS} records")

    all_rows, _ = comparisons_from_records(payload.preferences)
    rows = sample_comparisons(all_rows, payload.calibration_items, seed=payload.seed)
    quality = flag_noisy_labels(rows)

    policy_rows, _ = comparisons_from_records(payload.policy_outputs)

    judge = PrecomputedJudge.from_records(payload.verdicts, strict=False, name=payload.judge_name)

    calibration = calibrate(judge, rows, label_quality=quality)

    gate_passed = True
    try:
        gate_on_calibration(calibration)
    except CalibrationFailed:
        gate_passed = False

    score = compare_policies(judge, policy_rows)

    markdown = generate_report(
        calibration, score, label_quality=quality,
        policy_names=(payload.policy_1_name, payload.policy_2_name),
    )

    return EvaluateResponse(
        report_markdown=markdown,
        gate_passed=gate_passed,
        calibration_summary=calibration.summary(),
        score_summary={
            "win_rate_policy1": score.win_rate_policy1,
            "ci_lower": score.ci_lower,
            "ci_upper": score.ci_upper,
            "n_scored": score.n_scored,
        },
        conflicting_verdicts=judge.conflicts,
    )


app = FastAPI(
    title="Evaluation & Judge Calibration Pipeline",
    description="Recorded-judge evaluation, callable over HTTP. See README.md for the full pipeline.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    return run_evaluation(payload)
