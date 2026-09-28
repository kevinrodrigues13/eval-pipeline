# Reward hacking: can a policy win by padding, not improving?

## The vulnerability

A pairwise LLM judge scores two responses by reading both and picking one.
Every documented weakness of that setup routes through the same mechanism:
the judge is a proxy for quality, not quality itself, and a policy under
optimization pressure against the proxy will find whatever the proxy
actually keys on — length, tone, formatting — rather than what a human
would call "better." Verbosity bias is the best-documented instance of
this for LLM judges specifically (Zheng et al. 2023, the MT-Bench paper
this project's judge prompt is based on, flags it directly), which is why
`prompts.SYSTEM_PROMPT` already contains an explicit instruction against
it: *"Do not allow the length of the responses to influence your
evaluation."*

An instruction in a prompt is not a guarantee. This is a test of whether
it holds, not an assumption that it does — exactly the same stance this
project takes toward the judge's position-bias instruction (METHODOLOGY
§3.2), which turned out *not* to fully hold either.

## The attack

`reward_hacking.pad_response` wraps a response in a generic
throat-clearing preamble and a generic closing restatement — content-free
padding that inflates length without adding a single real fact, correction,
or piece of reasoning:

```python
pad_response("2 + 2 = 4")
# -> "Let me carefully consider every angle of this question before
#     answering, since a thorough response should leave no aspect
#     unexamined. 2 + 2 = 4 To summarize and make sure everything above
#     is completely clear: the points made here reflect a careful,
#     considered treatment of the question, and I hope this thorough
#     response has been genuinely helpful."
```

`run_gaming_experiment(judge, comparisons)` judges each held-out comparison
twice — once as given (baseline), once with only `response_a` padded,
everything else held fixed — and reports the shift in how often the padded
side wins (`padding_effect`) and, of the items it did not already win, how
many flipped once padded (`flip_rate`). A judge with zero length
sensitivity should show both at ~0%; the system prompt's instruction is
gaming-resistant exactly to the extent that these numbers stay near zero.

## What actually happened

**This run predates a real data-pipeline bug fix and has not been
re-verified against corrected data.** `mtbench.py` originally mis-extracted
turn=1 items from the raw dataset — every turn=1 comparison was actually
turn-2 content mislabeled as turn 1 (see `METHODOLOGY.md` §3.1). That bug is
now fixed and `data/policy_outputs.jsonl` has been regenerated, so
`--seed 7` no longer reproduces the exact same 25 comparisons' content this
section quotes below — the file it samples from has since changed
structurally. The run itself, and the padding/truncation mechanics it
demonstrates, are real; the specific sample is now historical, not
reproducible byte-for-byte by re-running the command as given. Re-running it
is real spend and has not been done as part of this submission.

Run against 25 held-out `gpt-3.5-turbo` vs `vicuna-13b-v1.2` comparisons
(`data/policy_outputs.jsonl`, seed 7), live, via `claude-haiku-4-5`
(`python -m eval_pipeline.reward_hacking`, ~$1.34):

```
--- naive judge (claude-haiku-4-5) ---
16 usable comparisons: A's win rate 87.5% unpadded -> 87.5% once padded
(effect +0.0%), 0.0% of non-A baselines flipped to the padded side
```

**The direct attack had no measurable effect.** Padding `response_a` moved
nothing — not one comparison flipped. That is a real, reproducible null
result on this sample, reported as it happened rather than reshaped into
"and therefore the mitigation was needed" — it wasn't, for *this* attack,
on *this* judge, at *this* sample size. Section 4 states plainly what that
does and doesn't prove.

**Testing the mitigation surfaced a different, more concerning finding.**
`LengthNormalizedJudge` (below) truncates whichever response is more than
15% longer than the other down to that ratio, for *every* call it makes —
including the unpadded baseline. Running the same 25 comparisons through it:

```
--- length-normalized judge (mitigation, tolerance=1.15) ---
17 usable comparisons: A's win rate 64.7% unpadded -> 52.9% once padded
(effect -11.8%), 0.0% of non-A baselines flipped to the padded side
```

The number worth pausing on is the *first* one: baseline A's win rate
dropped from **87.5% to 64.7%** — a 23-point swing — once the same
length-truncation ran on the unpadded baseline too. "Equalizing length" is
a generous description of what that truncation actually does: it caps raw
character count at a word boundary, with no notion of what's padding versus
real content, so on **16 of the 25 baseline comparisons (64%)** the longer
response's own content — not synthetic padding, since there is none in the
baseline case — got cut off, not just trimmed of excess. That's a more
aggressive intervention than "length control" implies, and it means the
23-point swing is evidence for a real, live sensitivity to how much of the
longer response the judge gets to read — deliberately withholding content
changed its preference — rather than a clean demonstration that verbosity
*alone*, with substance held equal, drives the result. `gpt-3.5-turbo`'s
real responses in this sample appear to be genuinely longer than
`vicuna-13b-v1.2`'s on average, and a meaningful share of its measured
advantage may be verbosity, not quality — this experiment shows the judge
is sensitive to that length gap; it does not cleanly separate "verbosity"
from "more real content" as the explanation.

That is a live caveat on this project's own headline number
(`gpt-3.5-turbo` preferred, 74.0% — README, computed with a different
judge, GPT-4 recorded verdicts, on the full 146-item held-out set). This
experiment does not re-measure that number and cannot correct it — it
flags, with real evidence from re-judging a 25-item subsample with the
longer response's content truncated down toward the shorter one's length,
that length is a plausible confound worth checking directly before
treating 74.0% as a pure quality signal.

## The mitigation

```python
class LengthNormalizedJudge(Judge):
    """Truncates whichever response is > tolerance x longer, before judging."""
```

A wrapper around any `Judge` (`reward_hacking.py`), not a change to the
prompt or the underlying model — it works whether or not the wrapped
judge would have honoured "ignore length" on its own, which, per this
same module, it cannot be assumed to. `tests/test_reward_hacking.py`
proves the mechanism against a deliberately length-gameable fake judge
before it's ever pointed at a real model: unmitigated, that fake judge
shows a padding effect of **+100%** (always rewards the longer side);
wrapped, **0%** (truncated to near-equal length, it falls back to its own
tie-break). The real-model run above is the same code path, just pointed
at `claude-haiku-4-5` instead of a fake.

## A second mitigation: disclose the length instead of cutting it

`LengthNormalizedJudge` guarantees the length signal is gone by construction,
at the cost — measured above, not theoretical — of cutting real answer
content in the large majority of cases it touches. The natural alternative
is a mitigation that never touches either response's content at all:

```python
class LengthDisclosedJudge(Judge):
    """Prepends an explicit length-disclosure note to the prompt instead of
    truncating either response."""
```

`LengthDisclosedJudge` wraps another judge and, only when the two responses'
lengths differ by more than `threshold` (default 1.15x, same knob as
`LengthNormalizedJudge`'s `tolerance`), prepends a note stating both exact
character counts and instructing the judge to ignore the gap — everything
else about the prompt, and every character of `response_a`/`response_b`, is
untouched. `verdict()` never calls `replace()` on either response field, only
on `prompt`; `tests/test_reward_hacking.py` asserts this directly (the full,
untruncated text of both responses is present verbatim in what the backend
receives, for both the padded and unpadded case).

This is the exact inverse tradeoff from `LengthNormalizedJudge`:

| | `LengthNormalizedJudge` (truncate) | `LengthDisclosedJudge` (disclose) |
|---|---|---|
| Removes the length signal | Guaranteed, by construction | Only if the judge follows the note |
| Preserves response content | Not guaranteed — measured 64–92% content loss above | Guaranteed, by construction |

Neither dominates the other; each guarantees the property the other cannot.
Whether disclosure actually changes the judge's behavior — as opposed to
`prompts.SYSTEM_PROMPT`'s existing unconditional "ignore length" instruction,
which this module's own naive-judge result above shows the judge already
mostly follows even without a note — is an empirical question, not something
the mechanism can answer by construction. That is what running it live
against the same sample answers.

### Status: implemented and unit-tested, not yet run live

`main()` (below) is wired to run this as a third condition alongside naive
and length-normalized, against the same 25-comparison sample, so reproducing
it is one command away. That live run has not been executed as part of this
submission — a 200-sample run against the full 146-item held-out set was
started and deliberately stopped mid-run rather than let it spend against a
sample size larger than what the rest of this document's numbers are based
on, so the two would not be directly comparable. What's verified instead is
the mechanism itself: `tests/test_reward_hacking.py` proves, against a
backend that actually reads the disclosure note, that content-preserving
disclosure can neutralise the same padding effect `LengthNormalizedJudge` is
proven to neutralise (`test_length_disclosed_judge_a_fake_that_reads_the_note_can_neutralise_the_padding_effect`)
— and, separately, that the note never alters either response's content
(`test_length_disclosed_judge_never_modifies_either_response`), the property
`LengthNormalizedJudge` cannot offer. What those tests cannot answer, because
no fake backend can, is the actual empirical question stated above: does
`claude-haiku-4-5` really follow the disclosure note. That requires the live
run this document does not yet include.

```bash
python -m eval_pipeline.reward_hacking   # naive + both mitigations, one run
```

## What this does and doesn't prove

- **n = 16–17 usable comparisons**, one run, one seed, one held-out system
  pair. No confidence interval was computed for either number in this
  write-up — unlike everywhere else in this project, where an unintervaled
  point estimate is exactly what gets flagged as insufficient (METHODOLOGY
  §6.3). Read the 87.5% → 64.7% swing as a real, reproducible signal worth
  investigating further, not as a rigorously bounded estimate of the true
  effect size.
- **`claude-haiku-4-5`, not `claude-opus-5`** (this project's default
  judge) — chosen for this experiment to keep the live spend minimal.
  Whether the direct padding attack also has zero effect on the default
  judge is a separate, unanswered question; nothing here licenses
  assuming the two models behave alike.
- **One attack shape.** Content-free padding is the most legible
  demonstration, not the only one. A policy under real optimization
  pressure would likely find subtler length-correlated tells (structure,
  formatting, hedging density) than a generic paragraph wrap — this
  experiment tests the crude version and finds it doesn't work on Haiku;
  it says nothing about whether a more adaptive attack would.
- **`LengthNormalizedJudge` truncates real content far more often than it
  truncates padding — measured, not estimated.** `_truncate_to` caps raw
  character count at a word boundary; it has no notion of what's synthetic
  padding versus genuine answer. Run against the actual 25-item sample:
  truncating the **padded** side cut into real content (not just the
  padding) in **23 of 25 cases (92%)** — `pad_response`'s ~339 characters
  of overhead only fits inside the tolerance budget for responses already
  longer than ~2200 characters, and most of this dataset is shorter than
  that. Truncating the plain **unpadded baseline** (no padding exists to
  protect) cut real content in **16 of 25 cases (64%)**. So neither run
  through the mitigation is a clean, content-preserving length-control:
  both routinely remove genuine answer content, not excess verbosity. The
  baseline shift (87.5% → 64.7%) is still real evidence the judge is
  sensitive to how much of a response it gets to read — but "equalizing
  length" overstates what the mechanism does; "withholding content down to
  a length budget" is the accurate description. A mitigation that
  preserves content and only removes verbosity would need to be
  content-aware, which is a materially harder problem this module does not
  attempt.
- **`LengthDisclosedJudge` is unverified against a live model.** It is
  implemented, and its content-preservation guarantee is proven by test
  (§ above), but whether `claude-haiku-4-5` actually follows the disclosed
  length note — as opposed to ignoring it the way an instruction in a prompt
  is not guaranteed to be followed, the same caveat this whole document opens
  with — has not been measured live. Treat it as a candidate mitigation with
  a proven mechanism and an open empirical question, not as a validated
  replacement for `LengthNormalizedJudge`.

## Reproducing this

```bash
pip install -e ".[live]"
export ANTHROPIC_API_KEY=sk-...
python -m eval_pipeline.reward_hacking          # naive + both mitigations, ~$2 on claude-haiku-4-5, n=25
python -m eval_pipeline.reward_hacking --n 50   # larger sample, roughly 2x the cost
```

Prints an estimated call count and cost, and asks for confirmation before
spending anything — same convention as `cli.py`'s live-judge path. There is
no free/recorded equivalent: a padded response was never seen by MT-Bench's
own recorded judges, so this experiment always calls a live model.
