import pytest
from fastapi.testclient import TestClient

from eval_pipeline import providers
from eval_pipeline.service import (
    LIVE_JUDGE_KEY_ENV,
    LIVE_JUDGE_KEY_HEADER,
    MAX_LIVE_CALIBRATION_ITEMS,
    MAX_LIVE_POLICY_RECORDS,
    MAX_POLICY_RECORDS,
    MAX_PREFERENCE_RECORDS,
    MAX_VERDICT_RECORDS,
    app,
)

client = TestClient(app)


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


def verdict_record(record_id, i):
    winner = _ground_truth(i) if i % 7 != 0 else _flip(_ground_truth(i))
    return {"id": record_id, "winner": winner, "position_unstable": False}


def base_payload():
    return {
        "preferences": [preference_record(i) for i in range(40)],
        "policy_outputs": [policy_record(i) for i in range(20)],
        "verdicts": (
            [verdict_record(f"c{i}", i) for i in range(40)]
            + [verdict_record(f"p{i}", i) for i in range(20)]
        ),
        "calibration_items": 0,
        "policy_1_name": "alpha",
        "policy_2_name": "beta",
    }


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_evaluate_returns_a_full_report():
    response = client.post("/evaluate", json=base_payload())
    assert response.status_code == 200
    body = response.json()
    assert "# Evaluation report" in body["report_markdown"]
    assert body["gate_passed"] is True
    assert body["conflicting_verdicts"] == 0
    assert body["score_summary"]["n_scored"] > 0


def test_evaluate_still_returns_a_report_when_the_gate_fails():
    payload = base_payload()
    # Verdicts with no relationship to the human labels: fails calibration.
    payload["verdicts"] = (
        [{"id": f"c{i}", "winner": "A" if i % 2 == 0 else "B"} for i in range(40)]
        + [{"id": f"p{i}", "winner": "A" if i % 2 == 0 else "B"} for i in range(20)]
    )
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["gate_passed"] is False
    assert "No conclusion" in body["report_markdown"]


def test_evaluate_surfaces_conflicting_verdicts():
    payload = base_payload()
    payload["verdicts"].append({"id": "c0", "winner": _flip(verdict_record("c0", 0)["winner"])})
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    assert response.json()["conflicting_verdicts"] == 1


def test_evaluate_rejects_an_oversized_preferences_payload():
    payload = base_payload()
    payload["preferences"] = [preference_record(i) for i in range(MAX_PREFERENCE_RECORDS + 1)]
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 413


def test_evaluate_rejects_an_oversized_policy_outputs_payload():
    payload = base_payload()
    payload["policy_outputs"] = [policy_record(i) for i in range(MAX_POLICY_RECORDS + 1)]
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 413


def test_evaluate_rejects_an_oversized_verdicts_payload():
    payload = base_payload()
    payload["verdicts"] = [{"id": f"v{i}", "winner": "A"} for i in range(MAX_VERDICT_RECORDS + 1)]
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 413


def test_evaluate_rejects_a_malformed_payload():
    response = client.post("/evaluate", json={"preferences": []})  # missing required fields
    assert response.status_code == 422


def test_recorded_mode_requires_verdicts():
    payload = base_payload()
    payload["verdicts"] = []
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 400


# --- live mode --------------------------------------------------------------

def live_payload(**overrides):
    payload = {
        "preferences": [preference_record(i) for i in range(10)],
        "policy_outputs": [policy_record(i) for i in range(6)],
        "calibration_items": 0,
        "policy_1_name": "alpha",
        "policy_2_name": "beta",
        "judge_mode": "live",
    }
    payload.update(overrides)
    return payload


def _fake_backend_factory(model=None, cache=None):
    def backend(system_prompt, user_prompt):
        return "[[A]]"
    return backend


def test_live_mode_is_disabled_without_the_env_var_configured(monkeypatch):
    monkeypatch.delenv(LIVE_JUDGE_KEY_ENV, raising=False)
    response = client.post("/evaluate", json=live_payload())
    assert response.status_code == 503


def test_live_mode_rejects_a_missing_or_wrong_key(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    response = client.post("/evaluate", json=live_payload())  # no header at all
    assert response.status_code == 401

    response = client.post(
        "/evaluate", json=live_payload(), headers={LIVE_JUDGE_KEY_HEADER: "wrong-secret"}
    )
    assert response.status_code == 401


def test_live_mode_estimate_only_spends_nothing(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    call_count = {"n": 0}

    def counting_factory(model=None, cache=None):
        def backend(system_prompt, user_prompt):
            call_count["n"] += 1
            return "[[A]]"
        return backend

    monkeypatch.setattr(providers, "anthropic_judge_backend", counting_factory)

    response = client.post(
        "/evaluate",
        json=live_payload(confirm_spend=False),
        headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["spend_confirmed"] is False
    assert body["estimated_cost"] > 0
    assert body["n_live_calls"] > 0
    assert body["report_markdown"] is None
    assert call_count["n"] == 0  # not one dollar spent on a dry run


def test_live_mode_confirmed_spend_calls_the_backend_and_returns_a_report(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-this-test")
    monkeypatch.setattr(providers, "anthropic_judge_backend", _fake_backend_factory)

    response = client.post(
        "/evaluate",
        json=live_payload(confirm_spend=True),
        headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["spend_confirmed"] is True
    assert body["judge_mode"] == "live"
    assert "# Evaluation report" in body["report_markdown"]
    assert body["n_live_calls"] > 0


def test_live_mode_reports_a_clean_error_when_anthropic_api_key_is_unset(monkeypatch):
    # Regression test for a real bug: the anthropic SDK doesn't validate
    # ANTHROPIC_API_KEY at client construction, only on the first actual
    # call — deep inside calibrate(), past _build_live_judge's try/except.
    # Confirmed live against a real uvicorn process: a missing key surfaced
    # as a raw, unhandled TypeError and a bare 500 with no clean message.
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    response = client.post(
        "/evaluate",
        json=live_payload(confirm_spend=True),
        headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"},
    )
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_live_mode_wraps_a_backend_failure_in_flight_as_a_clean_502(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-this-test")

    def failing_factory(model=None, cache=None):
        def backend(system_prompt, user_prompt):
            raise RuntimeError("simulated network failure")
        return backend

    monkeypatch.setattr(providers, "anthropic_judge_backend", failing_factory)

    response = client.post(
        "/evaluate",
        json=live_payload(confirm_spend=True),
        headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"},
    )
    assert response.status_code == 502
    assert "live judge call failed" in response.json()["detail"]


def test_live_mode_rejects_an_oversized_calibration_sample(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    payload = live_payload(
        preferences=[preference_record(i) for i in range(MAX_LIVE_CALIBRATION_ITEMS + 5)],
        calibration_items=MAX_LIVE_CALIBRATION_ITEMS + 1,
    )
    response = client.post("/evaluate", json=payload, headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"})
    assert response.status_code == 413


def test_live_mode_rejects_an_oversized_policy_outputs_payload(monkeypatch):
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    payload = live_payload(
        policy_outputs=[policy_record(i) for i in range(MAX_LIVE_POLICY_RECORDS + 1)],
    )
    response = client.post("/evaluate", json=payload, headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"})
    assert response.status_code == 413


def test_live_mode_ignores_verdicts_entirely(monkeypatch):
    # Live mode never replays verdicts — a payload with none at all is valid,
    # unlike recorded mode (test_recorded_mode_requires_verdicts above).
    monkeypatch.setenv(LIVE_JUDGE_KEY_ENV, "correct-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-this-test")
    monkeypatch.setattr(providers, "anthropic_judge_backend", _fake_backend_factory)
    payload = live_payload(confirm_spend=True)
    assert "verdicts" not in payload

    response = client.post("/evaluate", json=payload, headers={LIVE_JUDGE_KEY_HEADER: "correct-secret"})
    assert response.status_code == 200
