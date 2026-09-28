"""
The live judge backend: a real Claude call via the official Anthropic SDK,
a response cache so identical calls are never paid for twice, and a cost
estimate printed before anything is spent.

`anthropic` itself is imported lazily inside `anthropic_judge_backend`, not
at module load — everything else in the pipeline (recorded-judge mode,
every test) never needs it installed at all.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

DEFAULT_MODEL = "claude-opus-5"
# 1024 truncated real replies from claude-haiku-4-5 before it reached its
# final [[A]]/[[B]]/[[C]] tag — confirmed live: Haiku doesn't take
# output_config.effort (see _EFFORT_UNSUPPORTED_MODELS below), so it can't
# be steered to a terse reply the way claude-opus-5 can, and its
# uncontrolled analysis routinely runs past 1024 tokens.
MAX_OUTPUT_TOKENS = 2048

# Published per-million-token prices, USD (input, output) — precise enough
# to warn a user before real money is spent, not an invoice. Falls back to
# a conservative estimate for an unlisted model.
_PRICE_PER_MILLION_TOKENS = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
_DEFAULT_PRICE_PER_MILLION_TOKENS = (5.0, 25.0)

# Haiku 4.5 rejects output_config.effort outright (400 invalid_request_error,
# "This model does not support the effort parameter") — confirmed against
# the real API, not assumed. Every other model this project targets
# (claude-opus-5, claude-sonnet-5) accepts it.
_EFFORT_UNSUPPORTED_MODELS = {"claude-haiku-4-5"}

# A judge's reply is a short justification plus a "[[A]]"/"[[B]]"/"[[C]]"
# tag — a few sentences, not an essay, *when the model can be steered there*
# via output_config.effort=low. A model in _EFFORT_UNSUPPORTED_MODELS can't
# be steered at all: a live Haiku run was observed writing 1000+ tokens of
# unprompted analysis and still not reaching its verdict tag, so its
# estimate uses a much higher figure — underselling this in the printed
# cost estimate would mean a user approves a spend based on a number the
# actual call has no intention of honouring.
_DEFAULT_OUTPUT_TOKENS_PER_CALL = 200
_UNCONTROLLED_OUTPUT_TOKENS_PER_CALL = 1200


def _request_kwargs(model: str, system_prompt: str, user_prompt: str) -> dict:
    """
    The keyword arguments for one `client.messages.create` call.

    Split out from `anthropic_judge_backend` so the model-specific
    parameter logic is testable without a real SDK client or network call.
    """
    kwargs = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if model not in _EFFORT_UNSUPPORTED_MODELS:
        kwargs["output_config"] = {"effort": "low"}
    return kwargs


def _is_cacheable(stop_reason: str) -> bool:
    """
    A response cut off by max_tokens is not a completed answer — it is
    missing whatever the model would have said next, including possibly
    the [[A]]/[[B]]/[[C]] tag itself. Caching it would make a truncation
    bug sticky: raising MAX_OUTPUT_TOKENS to fix it would do nothing,
    because every future run would keep replaying the same incomplete text
    from cache instead of ever calling the model again with the new limit.
    """
    return stop_reason != "max_tokens"


def estimate_cost(n_calls: int, avg_input_tokens: int, model: str = DEFAULT_MODEL) -> float:
    input_price, output_price = _PRICE_PER_MILLION_TOKENS.get(model, _DEFAULT_PRICE_PER_MILLION_TOKENS)
    output_tokens = (
        _UNCONTROLLED_OUTPUT_TOKENS_PER_CALL
        if model in _EFFORT_UNSUPPORTED_MODELS
        else _DEFAULT_OUTPUT_TOKENS_PER_CALL
    )
    input_cost = n_calls * avg_input_tokens * input_price / 1_000_000
    output_cost = n_calls * output_tokens * output_price / 1_000_000
    return input_cost + output_cost


class ResponseCache:
    """
    A JSONL-backed cache from a call's identity to its raw response text.

    Appends rather than rewrites, so a run that is interrupted partway still
    leaves every call made so far cached for the next attempt.

    `put` is guarded by a lock because concurrent judge calls (cli.py's
    `--concurrency`) can finish out of order and call it from different
    threads at once — two unlocked appends could interleave their writes
    and corrupt the file. `get` is deliberately left unlocked, and the
    cache-check -> network-call -> put sequence in the backend below is not
    atomic as a whole: two concurrent calls for the same uncached prompt can
    both miss and both hit the network. That's an accepted, narrow
    same-key race (an occasional duplicate paid call, never a corrupted
    cache) — locking the whole sequence would serialize every judge call
    and erase the entire point of concurrency.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._entries: dict = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._entries[record["key"]] = record["response"]

    @staticmethod
    def key(model: str, system_prompt: str, user_prompt: str) -> str:
        digest = hashlib.sha256(f"{model}\n{system_prompt}\n{user_prompt}".encode("utf-8"))
        return digest.hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._entries.get(key)

    def put(self, key: str, response: str) -> None:
        with self._lock:
            self._entries[key] = response
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "response": response}))
                handle.write("\n")

    @property
    def entries(self) -> Iterator[Tuple[str, str]]:
        return iter(self._entries.items())


def anthropic_judge_backend(
    model: str = DEFAULT_MODEL, cache: Optional[ResponseCache] = None
) -> Callable[[str, str], str]:
    """
    Build a judge backend that calls Claude directly through the official
    Anthropic SDK (`client.messages.create`) — no multi-provider
    abstraction layer, since this project only ever calls Claude.

    Effort is set low: a pairwise preference judgment with a short
    justification is much closer to a classification task than to
    long-horizon agentic work, which is where higher effort actually pays
    for itself. Thinking is left on (adaptive, the model's default) rather
    than disabled — the model's own failure modes with thinking disabled
    (tool-call-shaped text leaking into the reply, stray tags) are worse
    than the cost of low-effort adaptive thinking here.
    """
    import anthropic

    client = anthropic.Anthropic()

    def backend(system_prompt: str, user_prompt: str) -> str:
        cache_key = ResponseCache.key(model, system_prompt, user_prompt) if cache is not None else None
        if cache is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        response = client.messages.create(**_request_kwargs(model, system_prompt, user_prompt))
        text = next((block.text for block in response.content if block.type == "text"), "")

        if cache is not None and _is_cacheable(response.stop_reason):
            cache.put(cache_key, text)
        return text

    return backend
