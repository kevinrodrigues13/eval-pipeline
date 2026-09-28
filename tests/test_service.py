import threading

import pytest
from fastapi.testclient import TestClient

from eval_pipeline import providers
from eval_pipeline import service as service_module
from eval_pipeline.service import (
    LIVE_JUDGE_KEY_ENV,
    LIVE_JUDGE_KEY_HEADER,
    MAX_CALIBRATION_ITEMS,
    MAX_CONCURRENT_JOBS,
    MAX_POLICY_RECORDS,
    MAX_PREFERENCE_RECORDS,
    JobStore,
    app,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_job_store(tmp_path, monkeypatch):
    # The module-level _job_store defaults to .service_jobs.jsonl in the
    # working directory — every test gets its own tmp-backed store instead,
    # so a test run never writes into (or reads stale state from) the repo.
    monkeypatch.setattr(service_module, "_job_store", JobStore(tmp_path / "jobs.jsonl"))


def _ground_truth(i):
    return "A" if i % 3 == 0 else "B"


def _flip(label):
    return "B" if label == "A" else "A"


def preference_record(i):
    e1 = _ground_truth(i)
    e2 = e1 if i % 5 != 0 else _flip(e1)
    return {
        "id": f"c{i}", "prompt": f"question {i}",
        "response_a": f"a considered answer to question {i}",
        "response_b": f"short {i}", "cluster_id": f"q{i % 10}",
        "system_a": "alpha", "system_b": "beta",
        "annotations": [
            {"preferred": e1, "annotator_id": "e1"},
            {"preferred": e2, "annotator_id": "e2"},
        ],
    }


def policy_record(i):
    return {
        "id": f"p{i}", "prompt": f"held-out question {i}",
        "response_a": f"a considered answer to held-out question {i}",
        "response_b": f"short {i}", "cluster_id": f"pq{i % 10}",
        "system_a": "alpha", "system_b": "beta",
    }


def base_payload(**overrides):
    payload = {
        "preferences": [preference_record(i) for i in range(10)],
        "policy_outputs": [policy_record(i) for i in range(6)],
        "calibration_items": 0,
        "policy_1_name": "alpha",
        "policy_2_name": "beta",
    }
    payload.update(overrides)
    return payload


def _fake_backend_factory(model=None, cache=None):
    def backend(system_prompt, user_prompt):
        return "[[A]]"
    return backend


def _poll_job(job_id, timeout=5.0):
    """Jobs run on a real thread even under TestClient — poll until it
    leaves pending/running rather than assuming it's done."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/evaluate/{job_id}")
        body = response.json()
        if body["status"] not in ("pending", "running"):
            return response
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_evaluate_is_disabled_without_the_env_var_configured(monkeypatch):
    monkeypatch.delenv(LIVE_JUDGE_KEY_ENV, raising=False)
    response = client.post("/evaluate", json=base_payload())
    assert response.status_code == 503


def test_evaluate_rejects_a_missing_or_wrong_key(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    response = client.post("/evaluate", json=base_payload())  # no header at all
    assert response.status_code == 401

    response = client.post(
        "/evaluate", json=base_payload(), headers={LIVE_JUDGE_KEY_HEADER: "wrong-secret"}
    )
    assert response.status_code == 401


def test_evaluate_rejects_a_malformed_payload(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    response = client.post(
        "/evaluate", json={"preferences": []}, headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"}
    )  # missing required fields
    assert response.status_code == 422


def test_evaluate_rejects_an_oversized_preferences_payload(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    payload = base_payload(preferences=[preference_record(i) for i in range(MAX_PREFERENCE_RECORDS + 1)])
    response = client.post("/evaluate", json=payload, headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"})
    assert response.status_code == 413


def test_evaluate_rejects_an_oversized_policy_outputs_payload(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    payload = base_payload(policy_outputs=[policy_record(i) for i in range(MAX_POLICY_RECORDS + 1)])
    response = client.post("/evaluate", json=payload, headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"})
    assert response.status_code == 413


def test_evaluate_rejects_an_oversized_calibration_sample(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    payload = base_payload(
        preferences=[preference_record(i) for i in range(MAX_CALIBRATION_ITEMS + 5)],
        calibration_items=MAX_CALIBRATION_ITEMS + 1,
    )
    response = client.post("/evaluate", json=payload, headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"})
    assert response.status_code == 413


def test_evaluate_returns_immediately_and_runs_in_the_background(monkeypatch):
    # The whole point: submitting returns a job id right away, not after the
    # batch finishes — this asserts the submit call itself, before polling.
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-this-test")
    monkeypatch.setattr(providers, "anthropic_judge_backend", _fake_backend_factory)

    response = client.post(
        "/evaluate", json=base_payload(), headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"}
    )
    assert response.status_code == 202
    body = response.json()
    # A race against the background thread: for an instant fake backend it
    # may already have flipped to "running" (or even finished) by the time
    # this response is inspected. What matters is the submit call itself
    # returned 202 immediately, not that it blocked until the batch was done.
    assert body["status"] in ("pending", "running", "succeeded")
    assert body["job_id"]


def test_job_completes_with_a_full_report(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-this-test")
    monkeypatch.setattr(providers, "anthropic_judge_backend", _fake_backend_factory)

    submit = client.post(
        "/evaluate", json=base_payload(), headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"}
    )
    job_id = submit.json()["job_id"]

    response = _poll_job(job_id)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["error"] is None
    assert "# Evaluation report" in body["result"]["report_markdown"]


def test_job_still_completes_when_the_gate_fails(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-this-test")

    def coinflip_factory(model=None, cache=None):
        counter = {"n": 0}

        def backend(system_prompt, user_prompt):
            counter["n"] += 1
            return "[[A]]" if counter["n"] % 2 == 0 else "[[B]]"
        return backend

    monkeypatch.setattr(providers, "anthropic_judge_backend", coinflip_factory)

    submit = client.post(
        "/evaluate", json=base_payload(), headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"}
    )
    body = _poll_job(submit.json()["job_id"]).json()
    # A gate failure is still a "succeeded" job with an explanatory report —
    # not a "failed" job. "failed" means the job itself errored.
    assert body["status"] == "succeeded"
    assert body["result"] is not None


def test_job_reports_a_clean_failure_when_anthropic_api_key_is_unset(monkeypatch):
    # Regression test for a real bug: the anthropic SDK doesn't validate
    # ANTHROPIC_API_KEY at client construction, only on the first actual
    # call — confirmed live against a real uvicorn process, a missing key
    # surfaced as a raw, unhandled TypeError deep inside the SDK.
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    response = client.post(
        "/evaluate", json=base_payload(), headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"}
    )
    # Caught synchronously at submission — this failure mode doesn't even
    # need a background job to surface it.
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_job_wraps_an_in_flight_backend_failure_as_a_failed_job(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-this-test")

    def failing_factory(model=None, cache=None):
        def backend(system_prompt, user_prompt):
            raise RuntimeError("simulated network failure")
        return backend

    monkeypatch.setattr(providers, "anthropic_judge_backend", failing_factory)

    submit = client.post(
        "/evaluate", json=base_payload(), headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"}
    )
    assert submit.status_code == 202  # accepted even though it will fail once it runs
    job_id = submit.json()["job_id"]

    response = _poll_job(job_id)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "simulated network failure" in body["error"]
    assert body["result"] is None


def test_unknown_job_id_returns_404():
    response = client.get("/evaluate/does-not-exist")
    assert response.status_code == 404


def test_job_submission_is_rejected_once_too_many_are_in_flight(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-this-test")

    started = threading.Event()
    release = threading.Event()

    def blocking_factory(model=None, cache=None):
        def backend(system_prompt, user_prompt):
            started.set()
            release.wait(timeout=5)
            return "[[A]]"
        return backend

    monkeypatch.setattr(providers, "anthropic_judge_backend", blocking_factory)

    submitted_ids = []
    try:
        for _ in range(MAX_CONCURRENT_JOBS):
            response = client.post(
                "/evaluate", json=base_payload(), headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"}
            )
            assert response.status_code == 202
            submitted_ids.append(response.json()["job_id"])
        started.wait(timeout=5)  # at least one job is now actually running

        response = client.post(
            "/evaluate", json=base_payload(), headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"}
        )
        assert response.status_code == 429
    finally:
        release.set()  # let every blocked job finish so it doesn't leak past the test
        for job_id in submitted_ids:
            try:
                _poll_job(job_id)
            except AssertionError:
                pass


# --- JobStore: the durability claim, tested directly -----------------------

def test_job_store_survives_being_reopened(tmp_path):
    # The whole reason this exists instead of a plain in-memory dict: a
    # result written by one JobStore instance must be readable by a totally
    # separate instance pointed at the same file — simulating a process
    # restart, which an in-memory dict cannot survive at all.
    path = tmp_path / "jobs.jsonl"
    first = JobStore(path)
    first.put("job-1", "succeeded", result={"report_markdown": "# hi"})

    second = JobStore(path)  # a fresh instance, as a restarted process would create
    record = second.get("job-1")
    assert record is not None
    assert record["status"] == "succeeded"
    assert record["result"] == {"report_markdown": "# hi"}


def test_job_store_keeps_the_latest_status_per_job(tmp_path):
    # Each transition appends a new line rather than rewriting in place —
    # reloading the file must resolve to the LAST line for a given id, not
    # the first, or a restarted process would see a job as forever pending.
    path = tmp_path / "jobs.jsonl"
    store = JobStore(path)
    store.put("job-1", "pending")
    store.put("job-1", "running")
    store.put("job-1", "succeeded", result={"report_markdown": "done"})

    assert len(path.read_text().strip().splitlines()) == 3  # every transition appended, none overwritten

    reopened = JobStore(path)
    assert reopened.get("job-1")["status"] == "succeeded"


def test_job_store_reserve_is_atomic_under_concurrent_submission(tmp_path):
    # reserve() must never let more than max_in_flight submissions through,
    # even when many threads call it at the exact same moment — this is
    # what stands between a burst of confirmed live-job requests and a
    # budget overrun past MAX_CONCURRENT_JOBS.
    path = tmp_path / "jobs.jsonl"
    store = JobStore(path)
    max_in_flight = 5
    accepted = []
    lock = threading.Lock()

    def attempt(i):
        if store.reserve(f"job-{i}", max_in_flight):
            with lock:
                accepted.append(i)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(accepted) == max_in_flight
    assert store.count_in_flight() == max_in_flight
