import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from eval_pipeline.providers import ResponseCache, _is_cacheable, _request_kwargs, estimate_cost


def test_request_kwargs_includes_effort_for_opus():
    kwargs = _request_kwargs("claude-opus-5", "system", "user")
    assert kwargs["output_config"] == {"effort": "low"}


def test_request_kwargs_omits_effort_for_haiku():
    # Confirmed against the real API: Haiku 4.5 returns a 400
    # invalid_request_error ("This model does not support the effort
    # parameter") if output_config.effort is sent at all.
    kwargs = _request_kwargs("claude-haiku-4-5", "system", "user")
    assert "output_config" not in kwargs


def test_request_kwargs_carries_the_actual_prompts_and_model():
    kwargs = _request_kwargs("claude-opus-5", "sys prompt", "user prompt")
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["system"] == "sys prompt"
    assert kwargs["messages"] == [{"role": "user", "content": "user prompt"}]


def test_estimate_cost_scales_with_calls_and_tokens():
    assert estimate_cost(100, 500, "claude-opus-5") == 2 * estimate_cost(50, 500, "claude-opus-5")
    assert estimate_cost(10, 1000) > estimate_cost(10, 500)


def test_estimate_cost_falls_back_for_an_unlisted_model():
    # Doesn't raise, just uses the conservative default price.
    assert estimate_cost(10, 500, "some-future-model") > 0


def test_estimate_cost_assumes_far_more_output_tokens_for_haiku():
    # Haiku 4.5 can't be steered to a terse reply (no effort control) and
    # was observed live writing 1000+ tokens without reaching its verdict
    # tag — its output-token assumption must be much larger than a model
    # that can be told to keep effort low, or the printed estimate a user
    # approves spending against understates what the call will actually do.
    haiku_output_only = estimate_cost(1, 0, "claude-haiku-4-5")
    opus_output_only = estimate_cost(1, 0, "claude-opus-5")
    # haiku: 1200 tokens * $5/M = $0.006; opus: 200 tokens * $25/M = $0.005
    assert haiku_output_only == pytest.approx(0.006)
    assert opus_output_only == pytest.approx(0.005)
    assert haiku_output_only > opus_output_only


def test_response_cache_round_trips_across_instances(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = ResponseCache(path)
    key = ResponseCache.key("m", "sys", "user")
    assert cache.get(key) is None

    cache.put(key, "the response")
    assert cache.get(key) == "the response"

    reloaded = ResponseCache(path)
    assert reloaded.get(key) == "the response"
    assert list(reloaded.entries) == [(key, "the response")]


def test_a_truncated_response_is_not_cacheable():
    # Confirmed live: caching a max_tokens-truncated reply made a token-
    # budget bug sticky, since raising the limit afterwards never re-called
    # the model — the stale incomplete text just kept getting replayed.
    assert _is_cacheable("max_tokens") is False


def test_completed_responses_are_cacheable():
    assert _is_cacheable("end_turn") is True
    assert _is_cacheable("stop_sequence") is True


def test_response_cache_key_is_sensitive_to_every_input():
    a = ResponseCache.key("model-1", "sys", "user")
    b = ResponseCache.key("model-2", "sys", "user")
    c = ResponseCache.key("model-1", "sys", "different user")
    assert len({a, b, c}) == 3


def test_response_cache_put_is_safe_under_real_concurrent_writers(tmp_path):
    # cli.py's --concurrency can call put() from many threads at once.
    # Unlocked, interleaved file.write() calls could corrupt the JSONL
    # (a line from one entry splicing into another); a lost dict update
    # under the GIL is unlikely but the file corruption risk is real.
    path = tmp_path / "cache.jsonl"
    cache = ResponseCache(path)
    n = 200

    def write_one(i):
        cache.put(f"key-{i}", f"response number {i} with some padding text")

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(write_one, range(n)))

    # Every entry landed in memory...
    assert len(list(cache.entries)) == n
    for i in range(n):
        assert cache.get(f"key-{i}") == f"response number {i} with some padding text"

    # ...and the file itself is exactly n well-formed JSON lines — the real
    # test: a corrupted interleaved write would show up here as a line
    # that fails to parse, or a line count that doesn't match n.
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n
    for line in lines:
        json.loads(line)  # raises if any line is a corrupted interleaving

    # Reloading from disk must reproduce every entry, in a fresh instance.
    reloaded = ResponseCache(path)
    for i in range(n):
        assert reloaded.get(f"key-{i}") == f"response number {i} with some padding text"
