# Deployment: the pipeline as a callable service

`service.py` wraps the pipeline as an HTTP service, deployable on
DigitalOcean App Platform. Recorded-judge mode is the default and needs no
authentication. Live-judge mode is also reachable over HTTP, gated so a
publicly reachable endpoint can't spend the operator's Anthropic budget on
an unauthenticated request.

## How live mode stays safe without a human in the loop

`cli.py`'s live path is safe to expose interactively because a human sees a
printed cost estimate and has to type `y` before anything is spent. A
stateless `POST /evaluate` has no equivalent moment, so live mode
substitutes three things for it:

1. **A shared secret only the operator knows.** `LIVE_JUDGE_API_KEY`, set
   as an environment variable at deploy time, checked against the
   `X-Live-Judge-Key` request header. If the operator never sets it, live
   mode is unreachable regardless of what a caller sends — it isn't a
   feature flag the caller can turn on, it's off by construction unless the
   deployment explicitly opts in.
2. **Tighter payload caps.** `MAX_LIVE_CALIBRATION_ITEMS` /
   `MAX_LIVE_POLICY_RECORDS` (50 each) are an order of magnitude below
   recorded mode's caps — every row over them is real money, on an endpoint
   an authenticated caller could still hit by mistake with a much bigger
   payload than intended.
3. **A two-step estimate-then-confirm flow.** The same request with
   `"confirm_spend": false` (the default) returns the exact call count and
   dollar estimate — computed the same way `cli.py` computes it before
   printing its own prompt — and calls the model zero times. Only a second
   request with `"confirm_spend": true` actually spends anything. That's
   the same shape as the CLI's `y/N` prompt, split across two stateless
   requests instead of one interactive one.

The Anthropic API key itself (`ANTHROPIC_API_KEY`) is a separate
environment variable, checked before any call is attempted — a deployment
with `LIVE_JUDGE_API_KEY` set but no `ANTHROPIC_API_KEY` fails cleanly with
a 503, not a raw call to a client with no credentials.

## API

**`GET /health`** → `{"status": "ok"}` — used by App Platform's health check.

**`POST /evaluate`**

Recorded mode (default, `"judge_mode": "recorded"`):

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

Live mode — first call, no header needed beyond auth, nothing spent:

```json
{
  "preferences": [ "..." ], "policy_outputs": [ "..." ],
  "judge_mode": "live", "model": "claude-haiku-4-5", "confirm_spend": false
}
```
→
```json
{"judge_mode": "live", "spend_confirmed": false, "estimated_cost": 0.03, "n_live_calls": 4,
 "report_markdown": null, "gate_passed": null, "calibration_summary": null,
 "score_summary": null, "conflicting_verdicts": null}
```

Live mode — second call, `confirm_spend: true`, `X-Live-Judge-Key: <the operator's secret>`:

```json
{
  "report_markdown": "# Evaluation report\n\n...",
  "gate_passed": true,
  "calibration_summary": "claude-haiku-4-5: 74.8% ...",
  "score_summary": {"win_rate_policy1": 0.81, "ci_lower": 0.74, "ci_upper": 0.89, "n_scored": 123},
  "conflicting_verdicts": 0,
  "judge_mode": "live", "spend_confirmed": true, "estimated_cost": 0.03, "n_live_calls": 4
}
```

`verdicts` isn't needed at all in live mode — the model judges the
comparisons itself rather than replaying stored ones. Same report shape as
the CLI otherwise: `report_markdown` carries its own "no conclusion" bottom
line when `gate_passed` is `false`, rather than failing the request.

Payloads are capped (`MAX_PREFERENCE_RECORDS=4000`, `MAX_POLICY_RECORDS=1000`
generally; `MAX_LIVE_CALIBRATION_ITEMS=50`, `MAX_LIVE_POLICY_RECORDS=50` in
live mode → HTTP 413 over either). Recorded mode spends no API budget, but
an unbounded payload is still a CPU/memory liability on a publicly
reachable endpoint (the cluster bootstrap is `O(n_resamples)` work per
request); live mode's caps exist for the money, not the CPU.

## Running it

```bash
pip install -e ".[service]"          # includes anthropic now — live mode needs it importable
export ANTHROPIC_API_KEY=sk-...      # only if live mode will actually be used
export LIVE_JUDGE_API_KEY=...        # only if live mode should be reachable at all
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
`.do/app.yaml`). What *was* verified locally, for real, rather than
assumed:

- `pip install ".[service]"` into a **clean** virtualenv (not the dev
  environment — no `pytest`, nothing pre-installed) succeeds and the
  service module imports and registers its routes correctly with only that
  minimal dependency set — i.e. exactly what the Dockerfile's `RUN` step
  does. `anthropic` is now part of the `[service]` extra (live mode needs
  it importable even when no key is configured), confirmed importable in
  that same clean environment.
- `uvicorn eval_pipeline.service:app` started as a **real server process**
  (not FastAPI's in-process `TestClient`) in that clean environment, and
  answered real HTTP requests over an actual socket: `GET /health`; a
  recorded-mode `POST /evaluate` with the full real MT-Bench dataset (2994
  raw preference records, 146 held-out comparisons), returning the
  identical win-rate and CI the CLI produces on the same data; live mode
  with no `LIVE_JUDGE_API_KEY` configured (503); live mode with a wrong key
  (401); live mode's estimate-only response with a correct key
  (`confirm_spend: false`, zero calls made — confirmed against a counting
  fake backend); and a `confirm_spend: true` request against a real,
  unpatched deployment missing `ANTHROPIC_API_KEY`.
- That last check caught a real bug: the `anthropic` SDK doesn't validate
  credentials at client construction, only on the first actual API call —
  which happens inside `calibrate`, past the try/except that was supposed
  to catch exactly this. The uncaught exception surfaced as a bare 500 with
  a raw stack trace in the response, not the clean 503 the code intended.
  Fixed by checking `ANTHROPIC_API_KEY` explicitly before attempting any
  live call, and by wrapping the live calls themselves in a broad
  try/except (`_run_live_calls`) so any in-flight failure — a bad model
  name, a network error, a rate limit — returns a clean 502 instead of a
  leaked traceback. Regression tests for both are in `tests/test_service.py`.
- `docker build` itself was **not** run — the Docker daemon isn't available
  in this sandboxed environment (the `docker` CLI is installed and its
  client half responds, but `docker info`'s server half reports "Docker
  Desktop is unable to start," and it doesn't come up headlessly here).
  The Dockerfile's only two real steps, `pip install ".[service]"` and
  running `uvicorn`, were each verified directly instead, which is what a
  `docker build` + `docker run` would actually exercise — but the
  container build itself is unverified, stated plainly rather than glossed
  over.
