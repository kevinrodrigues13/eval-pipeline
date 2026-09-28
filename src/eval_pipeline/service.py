"""
A callable HTTP service wrapping the pipeline's live-judge evaluation, for
deployment on DigitalOcean App Platform.

Live-judge only, and asynchronous: a batch can be dozens to hundreds of
real Claude calls, and running that inline inside one request would hold
the connection open for however long it takes — a real risk of the request
timing out on typical infrastructure, and a failure partway through would
return nothing at all. `POST /evaluate` instead validates the payload and
returns a job id immediately (`202`); the calls run in a background
thread. `GET /evaluate/{job_id}` polls for the result.

Every request here costs real money, so the service is gated behind an
operator-set secret (`LIVE_JUDGE_API_KEY`, checked against the
`X-Live-Judge-Key` header) rather than being reachable by anyone who finds
the URL — `cli.py`'s live path is safe to expose interactively because a
human sees a cost estimate and types `y` before anything is spent; nothing
here has that moment, so payload caps well below what a naive run could
reach, and a hard limit on concurrently in-flight jobs, do the same job the
confirmation gate does on the CLI.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .calibration import CalibrationFailed, calibrate, gate_on_calibration
from .ingestion import comparisons_from_records
from .judge import LLMJudge
from .providers import DEFAULT_MODEL
from .report import generate_report
from .scoring import compare_policies
from .validation import flag_noisy_labels, sample_comparisons

# Raw payload size — a CPU/memory liability on a publicly reachable
# endpoint regardless of judge mode (parsing, the cluster bootstrap) —
# checked before any sampling, so it bounds work done even on a request
# that will fail every other check.
MAX_PREFERENCE_RECORDS = 4000

# Real model calls, x2 for the position swap — sized to what a real live
# run in this project actually used (METHODOLOGY.md §5.5: 250 calibration
# + 146 held-out), not the CPU-safety caps above.
MAX_CALIBRATION_ITEMS = 300
MAX_POLICY_RECORDS = 200

# How many judge calls run in flight at once within one job — same
# mechanism as cli.py's --concurrency. A malformed reply is counted and
# excluded rather than aborting the job, since there is nobody watching a
# background job to retry it by hand the way an operator running the CLI can.
JUDGE_CONCURRENCY = 10

# How many jobs may be pending/running at once — independent of the
# per-job payload caps above, this bounds how many batches of real spend
# can be in flight simultaneously.
MAX_CONCURRENT_JOBS = 5

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
    calibration_items: int = Field(200, ge=0, description="0 uses every preference record")
    policy_1_name: str = "Policy 1"
    policy_2_name: str = "Policy 2"
    judge_name: str = "live"
    seed: int = 7
    model: str = DEFAULT_MODEL


class EvaluateResult(BaseModel):
    report_markdown: str
    gate_passed: bool
    calibration_summary: str
    score_summary: Dict[str, Optional[float]]
    conflicting_verdicts: int


class JobAccepted(BaseModel):
    job_id: str
    status: str


class JobResult(BaseModel):
    job_id: str
    status: str  # pending | running | succeeded | failed
    result: Optional[EvaluateResult] = None
    error: Optional[str] = None


class JobStore:
    """
    A JSONL-backed store for job records — the results endpoint reads from
    this, not from a request-scoped or purely in-memory value.

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


def _check_live_auth(live_judge_key: Optional[str]) -> None:
    configured_key = os.environ.get(LIVE_JUDGE_KEY_ENV)
    if not configured_key:
        raise HTTPException(503, "live judge is not enabled on this deployment")
    if live_judge_key != configured_key:
        raise HTTPException(401, f"missing or incorrect {LIVE_JUDGE_KEY_HEADER} header")


def _run_job(job_id: str, payload: EvaluateRequest, rows, policy_rows, quality) -> None:
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
            judge, policy_rows, max_workers=JUDGE_CONCURRENCY, skip_parse_errors=True
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
        result = EvaluateResult(
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


def submit_job(payload: EvaluateRequest, live_judge_key: Optional[str] = None) -> JobAccepted:
    """Validates and enqueues a batch; returns immediately. The actual model calls happen in a background thread."""
    _check_live_auth(live_judge_key)

    if len(payload.preferences) > MAX_PREFERENCE_RECORDS:
        raise HTTPException(413, f"preferences exceeds {MAX_PREFERENCE_RECORDS} records")
    if len(payload.policy_outputs) > MAX_POLICY_RECORDS:
        raise HTTPException(413, f"policy_outputs exceeds {MAX_POLICY_RECORDS} records")

    all_rows, _ = comparisons_from_records(payload.preferences)
    rows = sample_comparisons(all_rows, payload.calibration_items, seed=payload.seed)
    quality = flag_noisy_labels(rows)
    policy_rows, _ = comparisons_from_records(payload.policy_outputs)

    if len(rows) > MAX_CALIBRATION_ITEMS:
        raise HTTPException(
            413,
            f"calibration sample ({len(rows)}) exceeds {MAX_CALIBRATION_ITEMS} "
            "— pass a smaller calibration_items",
        )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "live judge unavailable on this deployment: ANTHROPIC_API_KEY is not set")

    job_id = str(uuid.uuid4())
    if not _job_store.reserve(job_id, MAX_CONCURRENT_JOBS):
        in_flight = _job_store.count_in_flight()
        raise HTTPException(
            429,
            f"{in_flight} jobs already in flight (max {MAX_CONCURRENT_JOBS}) "
            "— try again once one finishes",
        )

    threading.Thread(
        target=_run_job, args=(job_id, payload, rows, policy_rows, quality), daemon=True
    ).start()

    return JobAccepted(job_id=job_id, status="pending")


def get_job(job_id: str) -> JobResult:
    record = _job_store.get(job_id)
    if record is None:
        raise HTTPException(404, "job not found (unknown id, or it aged out of the job store)")
    return JobResult(
        job_id=record["job_id"], status=record["status"],
        result=record["result"], error=record["error"],
    )


app = FastAPI(
    title="Evaluation & Judge Calibration Pipeline",
    description="Live-judge evaluation, callable over HTTP as an async job. See README.md for the full pipeline.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/evaluate", response_model=JobAccepted, status_code=202)
def evaluate(
    payload: EvaluateRequest,
    x_live_judge_key: Optional[str] = Header(None, alias=LIVE_JUDGE_KEY_HEADER),
) -> JobAccepted:
    return submit_job(payload, live_judge_key=x_live_judge_key)


@app.get("/evaluate/{job_id}", response_model=JobResult)
def read_job(job_id: str) -> JobResult:
    return get_job(job_id)
