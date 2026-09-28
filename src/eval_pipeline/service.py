"""
A callable HTTP service wrapping the pipeline, for deployment on
DigitalOcean App Platform.

Recorded-judge mode (the default) spends nothing, needs no authentication,
and is fast — one request, one response. Live-judge mode is also reachable
over HTTP, but a confirmed live run never executes inline inside a request:
a batch can be dozens to hundreds of real model calls, and holding an HTTP
connection open for however long that takes is the wrong shape for it — it
risks the request timing out on most infrastructure long before the batch
finishes, and a failure partway through loses everything with nothing
returned. `POST /evaluate/live-jobs` instead accepts the batch, returns a
job id immediately (202), and runs the calls in a background thread;
`GET /evaluate/live-jobs/{id}` polls for the result. `POST /evaluate`
itself stays synchronous for both its jobs — recorded mode, and a live-mode
*estimate* (arithmetic over the payload, no model calls, genuinely fast) —
and refuses a live request that asks it to actually spend, pointing the
caller at the job endpoint instead.

Live mode substitutes for the CLI's interactive y/N confirmation with a
shared secret only the operator knows (`LIVE_JUDGE_API_KEY`, checked
against the `X-Live-Judge-Key` header — unreachable unless the deployment
opts in), payload caps well below recorded mode's, and the
estimate-then-submit split above.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
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

# Live mode's caps are well below recorded mode's: every row over them is
# real money, on an endpoint an authenticated caller could still hit by
# mistake with a much bigger payload than intended. Not a timeout-avoidance
# measure any more (the job runs off the request thread) — purely a
# budget-liability bound, sized to what a real live run in this project
# actually used (§5.5, METHODOLOGY.md: 250 calibration + 146 held-out).
MAX_LIVE_CALIBRATION_ITEMS = 300
MAX_LIVE_POLICY_RECORDS = 200

# How many live calls run in flight at once within one job — same
# mechanism as cli.py's --concurrency. A malformed reply is counted and
# excluded rather than aborting the job, since there is nobody watching a
# background job to retry it by hand the way an operator running the CLI can.
LIVE_MODE_CONCURRENCY = 10

# How many live jobs may be pending/running at once. Defense against a
# burst of confirmed submissions each spawning LIVE_MODE_CONCURRENCY calls
# simultaneously — independent of the per-job payload caps above.
MAX_CONCURRENT_LIVE_JOBS = 5

# Where job records are persisted — a JSONL cache, the same append-only
# shape as providers.ResponseCache, so a job's result survives a process
# restart instead of vanishing with an in-memory dict. Configurable so a
# real deployment can point it at a mounted volume.
JOB_STORE_PATH = os.environ.get("JOB_STORE_PATH", ".service_jobs.jsonl")

# The store is still a single local file, which only works for one
# instance (.do/app.yaml pins instance_count: 1 for exactly this reason,
# and JOB_STORE_PATH isn't shared storage by default). Scaling this service
# horizontally would need the file replaced with a real shared store
# (Redis, a database) — a job created on one instance is invisible to
# another. Stated as a real limitation, not papered over.
_MAX_STORED_JOBS = 200

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


class EvaluateResponse(BaseModel):
    judge_mode: str
    estimated_cost: Optional[float] = None
    n_live_calls: Optional[int] = None
    report_markdown: Optional[str] = None
    gate_passed: Optional[bool] = None
    calibration_summary: Optional[str] = None
    score_summary: Optional[Dict[str, Optional[float]]] = None
    conflicting_verdicts: Optional[int] = None


class LiveJobRequest(BaseModel):
    preferences: List[dict] = Field(..., description="human-labelled comparisons")
    policy_outputs: List[dict] = Field(..., description="held-out comparisons to score")
    calibration_items: int = Field(200, ge=0, description="0 uses every preference record")
    policy_1_name: str = "Policy 1"
    policy_2_name: str = "Policy 2"
    judge_name: str = "live"
    seed: int = 7
    model: str = DEFAULT_MODEL


class JobAccepted(BaseModel):
    job_id: str
    status: str


class JobResult(BaseModel):
    job_id: str
    status: str  # pending | running | succeeded | failed
    result: Optional[EvaluateResponse] = None
    error: Optional[str] = None


class JobStore:
    """
    A JSONL-backed store for live-job records — the results endpoint reads
    from this, not from a request-scoped or purely in-memory value.

    Same shape as `providers.ResponseCache`: append-only, one line per
    write, and loading the file replays every line into a dict keyed by
    job_id, so the *last* line for a given id wins (pending -> running ->
    succeeded/failed, in write order — each transition appends a new line
    rather than rewriting the file in place). That append-only design is
    what makes a write safe to call from a background thread while a
    request thread might be reading the same job concurrently, and what
    lets a job's current state be recovered after a process restart — the
    one thing a plain in-memory dict cannot do at all.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._records: Dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._records[record["job_id"]] = record

    def put(
        self, job_id: str, status: str,
        result: Optional[dict] = None, error: Optional[str] = None,
    ) -> None:
        record = {
            "job_id": job_id, "status": status, "result": result,
            "error": error, "written_at": time.time(),
        }
        with self._lock:
            self._write_locked(record)

    def reserve(self, job_id: str, max_in_flight: int) -> bool:
        """
        Atomically checks the in-flight count and writes a 'pending' record
        for job_id if under the cap, in one locked step. Checking
        count_in_flight() and then calling put() as two separate calls
        would race two concurrent submissions both past the cap in the gap
        between them — this closes that gap by holding the lock for both.
        Returns False (and writes nothing) if the cap is already reached.
        """
        with self._lock:
            if self.count_in_flight() >= max_in_flight:
                return False
            self._write_locked(
                {"job_id": job_id, "status": "pending", "result": None, "error": None, "written_at": time.time()}
            )
            return True

    def get(self, job_id: str) -> Optional[dict]:
        return self._records.get(job_id)

    def count_in_flight(self) -> int:
        return sum(1 for r in self._records.values() if r["status"] in ("pending", "running"))

    def _write_locked(self, record: dict) -> None:
        """Caller must hold self._lock."""
        self._records[record["job_id"]] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")
        self._prune_locked()

    def _prune_locked(self) -> None:
        """Caller must hold self._lock. Drops the oldest *finished* jobs
        from the in-memory index once it grows past _MAX_STORED_JOBS — an
        unbounded dict is a slow memory leak on a long-lived process; a
        completed job's result has already been (or will be) polled and
        copied out by its caller. The on-disk file itself is untouched —
        it's an append-only log, and compacting it isn't attempted here;
        a long-lived real deployment would want that (or a TTL-based store)
        eventually, stated as a known gap rather than built out."""
        if len(self._records) <= _MAX_STORED_JOBS:
            return
        finished = sorted(
            (r for r in self._records.values() if r["status"] in ("succeeded", "failed")),
            key=lambda r: r["written_at"],
        )
        for record in finished[: len(self._records) - _MAX_STORED_JOBS]:
            del self._records[record["job_id"]]


_job_store = JobStore(JOB_STORE_PATH)


def _live_cost_estimate(rows, policy_rows, model: str) -> Dict[str, float]:
    from .providers import estimate_cost

    n_calls = (len(rows) + len(policy_rows)) * 2
    characters = sum(
        len(r.prompt) + len(r.response_a) + len(r.response_b) for r in rows
    ) / max(len(rows), 1)
    avg_input_tokens = int(characters / 4) + 300
    return {"n_calls": n_calls, "cost": estimate_cost(n_calls, avg_input_tokens, model)}


def _check_live_auth(live_judge_key: Optional[str]) -> None:
    configured_key = os.environ.get(LIVE_JUDGE_KEY_ENV)
    if not configured_key:
        raise HTTPException(503, "live judge is not enabled on this deployment")
    if live_judge_key != configured_key:
        raise HTTPException(401, f"missing or incorrect {LIVE_JUDGE_KEY_HEADER} header")


def run_evaluation(payload: EvaluateRequest, live_judge_key: Optional[str] = None) -> EvaluateResponse:
    """
    The synchronous path: recorded mode (runs fully), or live mode (returns
    a cost estimate only — never calls a model). Kept separate from FastAPI
    plumbing so it's testable without an HTTP client.
    """
    if len(payload.preferences) > MAX_PREFERENCE_RECORDS:
        raise HTTPException(413, f"preferences exceeds {MAX_PREFERENCE_RECORDS} records")
    if len(payload.policy_outputs) > MAX_POLICY_RECORDS:
        raise HTTPException(413, f"policy_outputs exceeds {MAX_POLICY_RECORDS} records")

    if payload.judge_mode == "live":
        _check_live_auth(live_judge_key)
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
        # This endpoint never spends anything — it can only ever answer
        # "what would this cost." An actual run goes through
        # POST /evaluate/live-jobs, which is the only place a live model is
        # ever called from.
        estimate = _live_cost_estimate(rows, policy_rows, payload.model)
        return EvaluateResponse(
            judge_mode="live",
            estimated_cost=estimate["cost"],
            n_live_calls=estimate["n_calls"],
        )

    judge = PrecomputedJudge.from_records(payload.verdicts, strict=False, name=payload.judge_name)
    calibration = calibrate(judge, rows, label_quality=quality)
    score = compare_policies(judge, policy_rows)

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
        judge_mode="recorded",
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


def _run_live_job(job_id: str, payload: LiveJobRequest, rows, policy_rows, quality, estimate) -> None:
    """Runs off the request thread entirely — every outcome is written to
    _job_store, never raised to a caller directly (there is no caller
    connected by the time this runs)."""
    from .providers import anthropic_judge_backend

    _job_store.put(job_id, "running")
    try:
        judge = LLMJudge(
            anthropic_judge_backend(model=payload.model), name=payload.judge_name or payload.model
        )
        calibration = calibrate(judge, rows, label_quality=quality, skip_parse_errors=True)
        score = compare_policies(
            judge, policy_rows, max_workers=LIVE_MODE_CONCURRENCY, skip_parse_errors=True
        )

        gate_passed = True
        try:
            gate_on_calibration(calibration)
        except CalibrationFailed:
            gate_passed = False

        markdown = generate_report(
            calibration, score, label_quality=quality,
            policy_names=(payload.policy_1_name, payload.policy_2_name),
        )
        result = EvaluateResponse(
            judge_mode="live",
            estimated_cost=estimate["cost"],
            n_live_calls=estimate["n_calls"],
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
        _job_store.put(job_id, "succeeded", result=result.model_dump())
    except Exception as exc:  # anthropic missing, bad credentials, network failure, mid-run API error
        _job_store.put(job_id, "failed", error=str(exc))


def submit_live_job(payload: LiveJobRequest, live_judge_key: Optional[str] = None) -> JobAccepted:
    """Validates and enqueues a live batch; returns immediately. The actual model calls happen in a background thread."""
    _check_live_auth(live_judge_key)

    if len(payload.preferences) > MAX_PREFERENCE_RECORDS:
        raise HTTPException(413, f"preferences exceeds {MAX_PREFERENCE_RECORDS} records")
    if len(payload.policy_outputs) > MAX_POLICY_RECORDS:
        raise HTTPException(413, f"policy_outputs exceeds {MAX_POLICY_RECORDS} records")

    all_rows, _ = comparisons_from_records(payload.preferences)
    rows = sample_comparisons(all_rows, payload.calibration_items, seed=payload.seed)
    quality = flag_noisy_labels(rows)
    policy_rows, _ = comparisons_from_records(payload.policy_outputs)

    if len(rows) > MAX_LIVE_CALIBRATION_ITEMS:
        raise HTTPException(
            413,
            f"calibration sample ({len(rows)}) exceeds {MAX_LIVE_CALIBRATION_ITEMS} "
            "— pass a smaller calibration_items",
        )
    if len(policy_rows) > MAX_LIVE_POLICY_RECORDS:
        raise HTTPException(413, f"policy_outputs exceeds {MAX_LIVE_POLICY_RECORDS} records")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "live judge unavailable on this deployment: ANTHROPIC_API_KEY is not set")

    estimate = _live_cost_estimate(rows, policy_rows, payload.model)

    job_id = str(uuid.uuid4())
    if not _job_store.reserve(job_id, MAX_CONCURRENT_LIVE_JOBS):
        in_flight = _job_store.count_in_flight()
        raise HTTPException(
            429,
            f"{in_flight} live jobs already in flight (max {MAX_CONCURRENT_LIVE_JOBS}) "
            "— try again once one finishes",
        )

    threading.Thread(
        target=_run_live_job, args=(job_id, payload, rows, policy_rows, quality, estimate), daemon=True
    ).start()

    return JobAccepted(job_id=job_id, status="pending")


def get_live_job(job_id: str) -> JobResult:
    record = _job_store.get(job_id)
    if record is None:
        raise HTTPException(404, "job not found (unknown id, or it aged out of the job store)")
    return JobResult(
        job_id=record["job_id"], status=record["status"],
        result=record["result"], error=record["error"],
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


@app.post("/evaluate/live-jobs", response_model=JobAccepted, status_code=202)
def create_live_job(
    payload: LiveJobRequest,
    x_live_judge_key: Optional[str] = Header(None, alias=LIVE_JUDGE_KEY_HEADER),
) -> JobAccepted:
    return submit_live_job(payload, live_judge_key=x_live_judge_key)


@app.get("/evaluate/live-jobs/{job_id}", response_model=JobResult)
def read_live_job(job_id: str) -> JobResult:
    return get_live_job(job_id)
