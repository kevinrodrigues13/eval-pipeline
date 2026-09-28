import pytest
from fastapi.testclient import TestClient

from eval_pipeline.service import MAX_POLICY_RECORDS, MAX_PREFERENCE_RECORDS, MAX_VERDICT_RECORDS, app

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
