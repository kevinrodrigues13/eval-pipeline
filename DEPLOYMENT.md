# Deployment: the pipeline as a callable service

`service.py` wraps the pipeline's live-judge evaluation as an HTTP service,
deployable on DigitalOcean App Platform. Every request costs real money
(a real Claude call), so it's gated behind an operator-set secret rather
than reachable by anyone who finds the URL — and a batch never runs inline
inside a request; it's submitted as a job and polled for its result.

## Why a batch is a job, not a request/response

A live run can be dozens to hundreds of real model calls. Running that
batch inline inside `POST /evaluate` and holding the HTTP connection open
until it finishes is the wrong shape for it: it risks the request timing
out on most infrastructure long before the batch completes, and a failure
partway through returns nothing at all rather than whatever progress was
made. So:

- **`POST /evaluate`** validates the payload and returns a job id
  immediately (`202 Accepted`) — the actual model calls run afterward, in
  a background thread, off the request entirely.
- **`GET /evaluate/{job_id}`** polls for the result: `pending` → `running`
  → `succeeded`/`failed`.

## How it stays safe without a human in the loop

`cli.py`'s live path is safe to expose interactively because a human sees
a printed cost estimate and has to type `y` before anything is spent.
Nothing in a stateless HTTP request has that moment, so this service
substitutes for it with:

1. **A shared secret only the operator knows.** `LIVE_JUDGE_API_KEY`, set
   as an environment variable at deploy time, checked against the
   `X-Live-Judge-Key` request header. If the operator never sets it, the
   service is unreachable regardless of what a caller sends — off by
   construction, not a flag a caller can turn on.
2. **Payload caps sized to real usage, not a naive maximum.**
   `MAX_CALIBRATION_ITEMS` / `MAX_POLICY_RECORDS` (300 / 200) bound every
   job's worst-case cost — sized to what a real live run in this project
   actually used (METHODOLOGY.md §5.5: 250 calibration + 146 held-out).
   `MAX_CONCURRENT_JOBS` (5) separately bounds how many jobs can be
   spending at once, so an authenticated caller submitting a burst of jobs
   still can't multiply that exposure unboundedly.

The Anthropic API key itself (`ANTHROPIC_API_KEY`) is a separate
environment variable, checked before a job is even accepted — a deployment
with `LIVE_JUDGE_API_KEY` set but no `ANTHROPIC_API_KEY` fails the
submission cleanly with a `503`, rather than accepting the job and failing
it silently in the background.

## Job results are durable, not just in-memory

`JobStore` (in `service.py`) persists every job's state to a JSONL file —
the same append-only shape as `providers.ResponseCache` — instead of
keeping it only in a process-local dict. Each status transition appends a
new line rather than rewriting the file, and reopening the file replays
every line so the *latest* one per job id wins. That's what lets
`GET /evaluate/{job_id}` recover a job's true state after the process
restarts — verified for real: submitted a job, polled it to `"failed"`,
killed the `uvicorn` process, started a **new** one pointed at the same
file, and queried the same job id from that fresh process — it returned
the identical result, not a 404.

This is still a single local file, which only works for one instance
(`.do/app.yaml` pins `instance_count: 1` for exactly this reason, and
`JOB_STORE_PATH` isn't shared storage by default). Scaling this service
horizontally would need the file replaced with a real shared store (Redis,
a database) — a job created on one instance would otherwise be invisible
to another. Stated as a real limitation, not built around.

## API

**`GET /health`** → `{"status": "ok"}` — used by App Platform's health check.

**`POST /evaluate`** (`X-Live-Judge-Key: <the operator's secret>`):

```json
{
  "preferences": [ {"...": "human-labelled comparisons, same shape as a preferences JSONL file"} ],
  "policy_outputs": [ {"...": "held-out comparisons, same shape as policy_outputs.jsonl"} ],
  "calibration_items": 200,
  "policy_1_name": "gpt-3.5-turbo",
  "policy_2_name": "vicuna-13b-v1.2",
  "model": "claude-haiku-4-5",
  "seed": 7
}
```
→ `202 Accepted`
```json
{"job_id": "0ba8344a-930e-4ee3-ba68-c3cc5805a81d", "status": "pending"}
```

**`GET /evaluate/{job_id}`** — poll for the result:

```json
{
  "job_id": "0ba8344a-...", "status": "succeeded", "error": null,
  "result": {
    "report_markdown": "# Evaluation report\n\n...",
    "gate_passed": true,
    "calibration_summary": "live: 74.8% ...",
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

Payloads are capped (`MAX_PREFERENCE_RECORDS=4000` on the raw payload;
`MAX_CALIBRATION_ITEMS=300`, `MAX_POLICY_RECORDS=200` on what actually
drives real model calls → `413` over any of them; `MAX_CONCURRENT_JOBS=5`
jobs in flight at once → `429` over that).

## Running it

```bash
pip install -e ".[service]"          # includes anthropic — the service is live-judge only
export ANTHROPIC_API_KEY=sk-...      # required for a job to actually run
export LIVE_JUDGE_API_KEY=...        # required for the service to be reachable at all
export JOB_STORE_PATH=/data/jobs.jsonl  # optional — point at a mounted volume for real persistence
uvicorn eval_pipeline.service:app --host 0.0.0.0 --port 8080
```

```bash
docker build -t eval-pipeline .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-... -e LIVE_JUDGE_API_KEY=... eval-pipeline
```

Omitting `LIVE_JUDGE_API_KEY` deploys a service that only ever answers
`503` on `/evaluate` — `/health` still works, nothing else does.

## Deploying to DigitalOcean

```bash
doctl apps create --spec .do/app.yaml
```

**Not deployed as part of this submission** — that would need this
project's own DigitalOcean account and would incur real hosting cost (and
a real deployment would need `ANTHROPIC_API_KEY` and `LIVE_JUDGE_API_KEY`
set as App Platform secrets, not committed to `.do/app.yaml`, plus a
mounted volume for `JOB_STORE_PATH` to actually survive a redeploy rather
than just a crash). What *was* verified locally, for real, rather than
assumed:

- `pip install ".[service]"` into a **clean** virtualenv (not the dev
  environment — no `pytest`, nothing pre-installed) succeeds and the
  service module imports and registers its routes correctly with only that
  minimal dependency set — i.e. exactly what the Dockerfile's `RUN` step
  does.
- `uvicorn eval_pipeline.service:app` started as a **real server process**
  (not FastAPI's in-process `TestClient`), and answered real HTTP requests
  over an actual socket: `GET /health`; `POST /evaluate` with no
  `LIVE_JUDGE_API_KEY` configured (`503`); a wrong key (`401`); a correct
  key with a valid-looking-but-fake `ANTHROPIC_API_KEY`, returning `202`
  immediately with a job id; and polling that job to a `"failed"` status
  carrying the real `401 authentication_error` Anthropic's own API
  returned for the fake key.
- **The durability claim, specifically**: submitted a job against one real
  server process, polled it to `"failed"`, killed that `uvicorn` process,
  started a **second, independent** process pointed at the same
  `JOB_STORE_PATH`, and queried the same job id from it — it returned the
  identical result. Not simulated; two actually-separate processes.
- Earlier in building this feature, running it against a real server also
  caught a real bug: the `anthropic` SDK doesn't validate credentials at
  client construction, only on the first actual API call. An uncaught
  exception from deep inside the SDK surfaced as a bare 500 with a raw
  stack trace, not a clean error. Fixed by checking `ANTHROPIC_API_KEY`
  explicitly before accepting a job, and by catching any in-flight failure
  inside the background job runner so it always lands in `job.error`
  instead of crashing anything. Regression tests for both are in
  `tests/test_service.py`.
- `docker build` itself was **not** run — the Docker daemon isn't available
  in this sandboxed environment (the `docker` CLI is installed and its
  client half responds, but `docker info`'s server half reports "Docker
  Desktop is unable to start," and it doesn't come up headlessly here).
  The Dockerfile's only two real steps, `pip install ".[service]"` and
  running `uvicorn`, were each verified directly instead, which is what a
  `docker build` + `docker run` would actually exercise — but the
  container build itself is unverified, stated plainly rather than glossed
  over.
