# Deployment: the pipeline as a callable service

`service.py` wraps the pipeline as an HTTP service, deployable on
DigitalOcean App Platform. It is **recorded-judge only** — deliberately, not
as a missing feature.

## Why no live judge over HTTP

`cli.py`'s live path is safe to expose interactively because a human sees a
printed cost estimate and has to type `y` before anything is spent
(`--yes` bypasses this only for a script the *operator* controls). A
stateless `POST /evaluate` has no equivalent moment — there is nobody on
the other end of an API call to confirm a spend to. A publicly reachable
endpoint that can spend the operator's Anthropic budget on an
unauthenticated request is a real liability, not a hypothetical one, so
the service only ever replays verdicts the caller already supplies
(`PrecomputedJudge`) — the same free, deterministic mode `--judge recorded`
uses. A caller who wants a live judge runs the CLI, where the confirmation
gate lives.

## API

**`GET /health`** → `{"status": "ok"}` — used by App Platform's health check.

**`POST /evaluate`**

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

→

```json
{
  "report_markdown": "# Evaluation report\n\n...",
  "gate_passed": true,
  "calibration_summary": "GPT-4 (MT-Bench recorded verdicts): 71.9% ...",
  "score_summary": {"win_rate_policy1": 0.74, "ci_lower": 0.655, "ci_upper": 0.823, "n_scored": 100},
  "conflicting_verdicts": 0
}
```

Same behaviour as the CLI: the report is written and returned even when
`gate_passed` is `false` — `report_markdown` carries its own "no
conclusion" bottom line in that case, exactly as `cli.py` produces
(`tests/test_service.py::test_evaluate_still_returns_a_report_when_the_gate_fails`).
Payloads are capped (`MAX_PREFERENCE_RECORDS=4000`,
`MAX_POLICY_RECORDS=1000`, → HTTP 413 over) — recorded mode spends no API
budget, but an unbounded payload is still a CPU/memory liability on a
publicly reachable endpoint (the cluster bootstrap is `O(n_resamples)`
work per request).

## Running it

```bash
pip install -e ".[service]"
uvicorn eval_pipeline.service:app --host 0.0.0.0 --port 8080
```

```bash
docker build -t eval-pipeline .
docker run -p 8080:8080 eval-pipeline
```

## Deploying to DigitalOcean

```bash
doctl apps create --spec .do/app.yaml
```

**Not deployed as part of this submission** — that would need this
project's own DigitalOcean account and would incur real hosting cost, and
wasn't asked for. What *was* verified locally, for real, rather than
assumed:

- `pip install ".[service]"` into a **clean** virtualenv (not the dev
  environment — no `anthropic`, no `pytest`) succeeds and the service
  module imports and registers its routes correctly with only that
  minimal dependency set — i.e. exactly what the Dockerfile's `RUN`
  step does.
- `uvicorn eval_pipeline.service:app` started as a **real server process**
  (not FastAPI's in-process `TestClient`) and answered real HTTP requests —
  `GET /health` and a `POST /evaluate` with the full real MT-Bench dataset
  (2994 raw preference records, 146 held-out comparisons) over an actual
  socket, returning the identical win-rate and CI the CLI produces on the
  same data (74.0%, [65.5%, 82.3%]).
- `docker build` itself was **not** run — the Docker daemon isn't available
  in this sandboxed environment (the `docker` CLI is installed and its
  client half responds, but `docker info`'s server half reports "Docker
  Desktop is unable to start," and it doesn't come up headlessly here).
  The Dockerfile's only two real steps, `pip install ".[service]"` and running
  `uvicorn`, were each verified directly instead, which is what a `docker
  build` + `docker run` would actually exercise — but the container build
  itself is unverified, stated plainly rather than glossed over.
