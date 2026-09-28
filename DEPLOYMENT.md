# Deployment: the pipeline as a callable service

`service.py` wraps the pipeline as an HTTP service, deployable on
DigitalOcean App Platform. Recorded-judge mode is the default and needs no
authentication. Live-judge mode is also reachable over HTTP, gated so a
publicly reachable endpoint can't spend the operator's Anthropic budget on
an unauthenticated request — and never executed inline inside a request.

## Why a live batch is a job, not a request/response

A live run can be dozens to hundreds of real model calls. Running that
batch inline inside `POST /evaluate` and holding the HTTP connection open
until it finishes is the wrong shape for it: it risks the request timing
out on most infrastructure long before the batch completes, and a failure
partway through returns nothing at all rather than whatever progress was
made. So the service splits live mode into two moves instead of one big
synchronous call:

- **`POST /evaluate`** (`judge_mode: "live"`) only ever answers "what would
  this cost" — pure arithmetic over the payload, no model calls, genuinely
  fast — and can't be asked to actually spend. There's no `confirm_spend`
  flag on it any more; that ambiguity is gone by construction.
- **`POST /evaluate/live-jobs`** accepts the batch, validates it, and
  returns a job id immediately (`202 Accepted`) — the actual model calls
  run afterward, in a background thread, off the request entirely.
- **`GET /evaluate/live-jobs/{id}`** polls for the result: `pending` →
  `running` → `succeeded`/`failed`.

Recorded mode is unaffected by any of this — `POST /evaluate` still runs it
fully and returns the report in one response, exactly as before, since it
never calls an external model and has no timeout risk to design around.

## How live mode stays safe without a human in the loop

`cli.py`'s live path is safe to expose interactively because a human sees a
printed cost estimate and has to type `y` before anything is spent. Neither
`POST /evaluate`'s estimate nor `POST /evaluate/live-jobs`'s submission has
that moment, so live mode substitutes for it with:

1. **A shared secret only the operator knows.** `LIVE_JUDGE_API_KEY`, set
   as an environment variable at deploy time, checked against the
   `X-Live-Judge-Key` request header on both live endpoints. If the
   operator never sets it, live mode is unreachable regardless of what a
   caller sends — off by construction, not a flag a caller can turn on.
2. **Payload caps well below recorded mode's.** `MAX_LIVE_CALIBRATION_ITEMS`
   / `MAX_LIVE_POLICY_RECORDS` (300 / 200 — sized to what a real live run in
   this project actually used, METHODOLOGY.md §5.5) bound every job's worst
   case cost. `MAX_CONCURRENT_LIVE_JOBS` (5) separately bounds how many
   jobs can be spending at once — an authenticated caller submitting a
   burst of jobs still can't multiply that exposure unboundedly.
3. **See-the-cost-before-you-submit.** A caller is expected to call
   `POST /evaluate` first to see the estimate, then `POST /evaluate/live-jobs`
   to actually run it — the same two-step shape as the CLI's `y/N` prompt,
   just as two separate requests instead of one interactive one.

The Anthropic API key itself (`ANTHROPIC_API_KEY`) is a separate
environment variable, checked before a job is even accepted — a deployment
with `LIVE_JUDGE_API_KEY` set but no `ANTHROPIC_API_KEY` fails the
submission cleanly with a 503, rather than accepting the job and failing it
silently in the background.

## Job results are durable, not just in-memory

`JobStore` (in `service.py`) persists every job's state to a JSONL file —
the same append-only shape as `providers.ResponseCache` — instead of
keeping it only in a process-local dict. Each status transition
(`pending` → `running` → `succeeded`/`failed`) appends a new line rather
than rewriting the file, and reopening the file replays every line so the
*latest* one per job id wins. That's what lets `GET /evaluate/live-jobs/{id}`
recover a job's true state after the process restarts — verified for real:
submitted a job, polled it to `"failed"`, killed the `uvicorn` process,
started a **new** one pointed at the same file, and queried the same job id
from that fresh process — it returned the identical result, not a 404.

This is still a single local file, which only works for one instance
(`.do/app.yaml` pins `instance_count: 1` for exactly this reason, and
`JOB_STORE_PATH` isn't shared storage by default). Scaling this service
horizontally would need the file replaced with a real shared store (Redis,
a database) — a job created on one instance would otherwise be invisible
to another. Stated as a real limitation, not built around.

## API

**`GET /health`** → `{"status": "ok"}` — used by App Platform's health check.

**`POST /evaluate`** — recorded mode (default, runs and returns fully):

```json
{
  "preferences": [ {"...": "comparison records, same shape as a preferences JSONL file"} ],
  "policy_outputs": [ {"...": "held-out comparisons, same shape as policy_outputs.jsonl"} ],
  "verdicts": [ {"...": "recorded judge verdicts, same shape as a verdicts JSONL file"} ],
  "calibration_items": 300,
  "policy_1_name": "gpt-3.5-turbo",
  "policy_2_name": "vicuna-13b-v1.2",
  "judge_name": "GPT-4 (MT-Bench recorded verdicts)",
  "seed": 7
}
```

**`POST /evaluate`** — live mode (estimate only, no `X-Live-Judge-Key`
needed beyond auth, nothing spent):

```json
{"preferences": ["..."], "policy_outputs": ["..."], "judge_mode": "live", "model": "claude-haiku-4-5"}
```
→
```json
{"judge_mode": "live", "estimated_cost": 0.03, "n_live_calls": 4,
 "report_markdown": null, "gate_passed": null, "calibration_summary": null,
 "score_summary": null, "conflicting_verdicts": null}
```

**`POST /evaluate/live-jobs`** (`X-Live-Judge-Key: <the operator's secret>`)
— submits the same shape of payload (no `verdicts`, no `judge_mode` field)
and returns immediately:

```json
{"job_id": "0ba8344a-930e-4ee3-ba68-c3cc5805a81d", "status": "pending"}
```
→ `202 Accepted`

**`GET /evaluate/live-jobs/{job_id}`** — poll for the result:

```json
{
  "job_id": "0ba8344a-...", "status": "succeeded", "error": null,
  "result": {
    "judge_mode": "live", "estimated_cost": 0.03, "n_live_calls": 4,
    "report_markdown": "# Evaluation report\n\n...",
    "gate_passed": true,
    "calibration_summary": "claude-haiku-4-5: 74.8% ...",
    "score_summary": {"win_rate_policy1": 0.81, "ci_lower": 0.74, "ci_upper": 0.89, "n_scored": 123},
    "conflicting_verdicts": 0
  }
}
```

`report_markdown` carries its own "no conclusion" bottom line when
`gate_passed` is `false`, same as the CLI — a failed calibration gate is
still a `"succeeded"` job with a report explaining why, not a `"failed"`
one. `"failed"` means the job itself errored (bad credentials, a network
failure, a mid-run API error) — `error` carries that message, `result` is
`null`.

Payloads are capped (`MAX_PREFERENCE_RECORDS=4000`, `MAX_POLICY_RECORDS=1000`
generally; `MAX_LIVE_CALIBRATION_ITEMS=300`, `MAX_LIVE_POLICY_RECORDS=200`
for a live job → HTTP 413 over either; `MAX_CONCURRENT_LIVE_JOBS=5` jobs in
flight at once → HTTP 429 over that). Recorded mode spends no API budget,
but an unbounded payload is still a CPU/memory liability on a publicly
reachable endpoint (the cluster bootstrap is `O(n_resamples)` work per
request); live mode's caps exist for the money, not the CPU.

## Running it

```bash
pip install -e ".[service]"          # includes anthropic now — live mode needs it importable
export ANTHROPIC_API_KEY=sk-...      # only if live mode will actually be used
export LIVE_JUDGE_API_KEY=...        # only if live mode should be reachable at all
export JOB_STORE_PATH=/data/jobs.jsonl  # optional — point at a mounted volume for real persistence
uvicorn eval_pipeline.service:app --host 0.0.0.0 --port 8080
```

```bash
docker build -t eval-pipeline .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-... -e LIVE_JUDGE_API_KEY=... eval-pipeline
```

Omitting `LIVE_JUDGE_API_KEY` deploys a recorded-judge-only service, same
as before this feature existed — live mode is additive, not a change to the
default behavior.

## Deploying to DigitalOcean

```bash
doctl apps create --spec .do/app.yaml
```

**Not deployed as part of this submission** — that would need this
project's own DigitalOcean account and would incur real hosting cost (and
a real live-mode deployment would need `ANTHROPIC_API_KEY` and
`LIVE_JUDGE_API_KEY` set as App Platform secrets, not committed to
`.do/app.yaml`, plus a mounted volume for `JOB_STORE_PATH` to actually
survive a redeploy rather than just a crash). What *was* verified locally,
for real, rather than assumed:

- `pip install ".[service]"` into a **clean** virtualenv (not the dev
  environment — no `pytest`, nothing pre-installed) succeeds and the
  service module imports and registers its routes correctly with only that
  minimal dependency set — i.e. exactly what the Dockerfile's `RUN` step
  does. `anthropic` is now part of the `[service]` extra (live mode needs
  it importable even when no key is configured).
- `uvicorn eval_pipeline.service:app` started as a **real server process**
  (not FastAPI's in-process `TestClient`), and answered real HTTP requests
  over an actual socket: `GET /health`; a recorded-mode `POST /evaluate`
  with the full real MT-Bench dataset (2994 raw preference records, 146
  held-out comparisons), returning the identical win-rate and CI the CLI
  produces on the same data; live mode with no `LIVE_JUDGE_API_KEY`
  configured (503); a wrong key (401); and the estimate-only response
  (zero calls made, confirmed against a counting fake backend).
- **The durability claim, specifically**: submitted a live job against a
  real server process (an invalid `ANTHROPIC_API_KEY`, so it genuinely
  failed rather than needing a real paid call to exercise), polled it to
  `"failed"` with the real error message from Anthropic's API, killed that
  `uvicorn` process, started a **second, independent** process pointed at
  the same `JOB_STORE_PATH`, and queried the same job id from it — it
  returned the identical result. Not simulated; two actually-separate
  processes.
- Earlier in building this feature, running it against a real server also
  caught a real bug: the `anthropic` SDK doesn't validate credentials at
  client construction, only on the first actual API call. An uncaught
  exception from deep inside the SDK surfaced as a bare 500 with a raw
  stack trace, not the clean error the code intended. Fixed by checking
  `ANTHROPIC_API_KEY` explicitly before accepting a job, and by catching
  any in-flight failure inside the background job runner so it always
  lands in `job.error` instead of crashing anything. Regression tests for
  both are in `tests/test_service.py`.
- `docker build` itself was **not** run — the Docker daemon isn't available
  in this sandboxed environment (the `docker` CLI is installed and its
  client half responds, but `docker info`'s server half reports "Docker
  Desktop is unable to start," and it doesn't come up headlessly here).
  The Dockerfile's only two real steps, `pip install ".[service]"` and
  running `uvicorn`, were each verified directly instead, which is what a
  `docker build` + `docker run` would actually exercise — but the
  container build itself is unverified, stated plainly rather than glossed
  over.
