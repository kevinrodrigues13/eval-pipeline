"""
End-to-end tests for cli.main(). Shallow on purpose: they assert the
pipeline runs and produces a report, not what it concludes — the per-module
tests already cover correctness of the scoring and statistical logic.
"""

import json

import pytest

from eval_pipeline import cli


def _flip(label):
    return "B" if label == "A" else "A"


def _ground_truth(i):
    return "A" if i % 3 == 0 else "B"


def _preference_record(i):
    e1 = _ground_truth(i)
    e2 = e1 if i % 5 != 0 else _flip(e1)
    return {
        "id": f"c{i}",
        "prompt": f"question {i} about a topic",
        "response_a": f"a considered answer to question {i}, with some detail",
        "response_b": f"short {i}",
        "cluster_id": f"q{i % 10}",
        "system_a": "alpha",
        "system_b": "beta",
        "annotations": [
            {"preferred": e1, "annotator_id": "e1"},
            {"preferred": e2, "annotator_id": "e2"},
        ],
    }


def _policy_record(i):
    return {
        "id": f"p{i}",
        "prompt": f"held-out question {i}",
        "response_a": f"a considered answer to held-out question {i}",
        "response_b": f"short {i}",
        "cluster_id": f"pq{i % 10}",
        "system_a": "alpha",
        "system_b": "beta",
    }


def _verdict_record(record_id, i):
    # Agrees with the ground truth 6 times out of 7 — a judge clearly better
    # than either noisy annotator (who disagrees with the other 1 time in 5).
    winner = _ground_truth(i) if i % 7 != 0 else _flip(_ground_truth(i))
    return {"id": record_id, "winner": winner, "position_unstable": False}


@pytest.fixture
def corpus(tmp_path):
    preferences = [_preference_record(i) for i in range(40)]
    policy_outputs = [_policy_record(i) for i in range(20)]
    verdicts = (
        [_verdict_record(f"c{i}", i) for i in range(40)]
        + [_verdict_record(f"p{i}", i) for i in range(20)]
    )

    paths = {}
    for name, records in (
        ("preferences", preferences),
        ("policy_outputs", policy_outputs),
        ("verdicts", verdicts),
    ):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        paths[name] = str(path)
    paths["output"] = str(tmp_path / "report.md")
    return paths


def test_cli_runs_end_to_end_and_passes_the_gate(corpus):
    exit_code = cli.main([
        "--judge", "recorded",
        "--preferences", corpus["preferences"],
        "--verdicts", corpus["verdicts"],
        "--policy-outputs", corpus["policy_outputs"],
        "--output", corpus["output"],
        "--calibration-items", "0",
    ])

    assert exit_code == 0
    text = open(corpus["output"]).read()
    assert "# Evaluation report" in text
    assert "No conclusion" not in text


def test_cli_rejects_recorded_without_verdicts(corpus):
    with pytest.raises(SystemExit):
        cli.main([
            "--judge", "recorded",
            "--preferences", corpus["preferences"],
            "--policy-outputs", corpus["policy_outputs"],
        ])


def test_cli_refuses_a_live_run_it_cannot_confirm(corpus, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    exit_code = cli.main([
        "--preferences", corpus["preferences"],
        "--policy-outputs", corpus["policy_outputs"],
        "--output", corpus["output"],
    ])

    assert exit_code == 1


def test_cli_still_writes_a_report_when_the_gate_fails(corpus, tmp_path):
    # A verdicts file with no relationship to the human labels at all: the
    # judge should fail calibration, but the report must still be written,
    # with its own "no conclusion" bottom line rather than a bare crash.
    bad_verdicts = [
        {"id": f"c{i}", "winner": "A" if i % 2 == 0 else "B"} for i in range(40)
    ] + [
        {"id": f"p{i}", "winner": "A" if i % 2 == 0 else "B"} for i in range(20)
    ]
    path = tmp_path / "bad_verdicts.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in bad_verdicts), encoding="utf-8")

    exit_code = cli.main([
        "--judge", "recorded",
        "--preferences", corpus["preferences"],
        "--verdicts", str(path),
        "--policy-outputs", corpus["policy_outputs"],
        "--output", corpus["output"],
        "--calibration-items", "0",
    ])

    assert exit_code == 1
    text = open(corpus["output"]).read()
    assert "# Evaluation report" in text
    assert "No conclusion" in text


def test_cli_accepts_concurrency_and_skip_parse_errors_flags(corpus):
    # Proves these flags actually reach calibrate()/compare_policies() (not
    # silently dropped by argparse) by producing a normal successful run —
    # the underlying max_workers/skip_parse_errors mechanics themselves are
    # covered directly in test_calibration.py/test_scoring.py.
    exit_code = cli.main([
        "--judge", "recorded",
        "--preferences", corpus["preferences"],
        "--verdicts", corpus["verdicts"],
        "--policy-outputs", corpus["policy_outputs"],
        "--output", corpus["output"],
        "--calibration-items", "0",
        "--concurrency", "4",
        "--skip-parse-errors",
    ])
    assert exit_code == 0
    assert "# Evaluation report" in open(corpus["output"]).read()
