# Methodology

## 1. The problem, stated precisely

Given `(prompt, response_A, response_B, preferred)` tuples labelled by
humans, build a mechanism that predicts which response is preferred, measure
how much that mechanism can be trusted against the human labels, and use it
to compare two candidate policies on held-out prompts with a confidence
measure — not a bare point estimate.

Three things have to stay distinct for that to mean anything: what the judge
said, what a human would have said, and what the evidence can actually
support. Conflating any two of them is how an evaluation report ends up
overclaiming.

---

## 2. Data and schema

### 2.1 Source

[`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
(CC-BY-4.0), fetched via the HuggingFace datasets-server REST API
(`mtbench.py`) rather than the `datasets` package, so the only dependency is
`requests`. Two splits: `human` (3355 rows, mostly 3 judges per item) and
`gpt4_pair` (2400 rows, GPT-4 as judge). ~1.4MB as fetched; ~24MB once
written to `data/`, since each row carries its own full copy of the prompt
and both responses rather than a shared, deduplicated one.

### 2.2 Schema: one row per comparison, not per annotation

`schema.Comparison` carries a `prompt`, both responses, and a tuple of every
`Annotation` it received — not one row per annotation. This is not
cosmetic. An item labelled by three humans is one unit of evidence about one
underlying question ("which response is better here"), and treating it as
three independent rows would silently triple-count it in anything that
assumes independence — including confidence intervals, which is exactly why
§6.2's cluster bootstrap exists.

### 2.3 Decision: `preferred` is permanently a statement about a slot

An `Annotation.preferred` value ("A", "B", "tie") means "the response in
slot A" — nothing about which *system* produced it. This is what makes the
position-swap defense (§3.2) possible at all: swapping `response_a` and
`response_b` and asking again only tests anything if "A" unambiguously means
"whatever is in slot A right now."

### 2.4 Held-out policy comparison, and a bug this design caught twice

`gpt-3.5-turbo` vs `vicuna-13b-v1.2` is held out entirely from
`mtbench_preferences.jsonl` (calibration) and scored only via
`policy_outputs.jsonl`, blind — no `preferred`, no `annotations`. The judge
never sees this pair's human labels before being scored on it.

Building this held-out set surfaced the same bug twice, through two
different symptoms, because the first fix was scoped narrower than the
actual problem. MT-Bench ran some pairs — including this one — in **both
slot orders** as its own position-bias control, so the same underlying
comparison appears twice in the raw data with `model_a`/`model_b` swapped.

**First symptom:** a naive dedup-by-id produced 213 held-out comparisons
where only 146 real (question, turn) items exist — a **46% inflation** —
caught not by any error but by checking the numbers directly (73 unique
questions, `system_a` split 109/104 across the "held-out" rows, which only
makes sense if roughly half were slot-reversed duplicates of the other
half). The first fix (`_canonical_pair_record`, since removed) normalised
slot order only for the held-out pair specifically, and resolved that
inflation — 213 became 146.

**Second symptom, found later:** while integration-testing the judge
(§3.5) against real data, `PrecomputedJudge` raised a `KeyError` replaying
the recorded GPT-4 verdicts against that same held-out set, because
`_verdict_records` built its ids from the raw, uncanonicalized row — the
first fix never touched that function. Tracing the `KeyError` back is what
revealed the fix needed to be general, not pair-specific: the real fix
(`_canonical_row`, `mtbench.py`) normalises slot order for *every* pair at
the row level, before anything else touches it — because the identical
silent duplication was also splitting ordinary calibration items across two
single-annotator rows instead of merging them into one multi-annotator row.
That third effect was the quietest of all: it never crashed anything, it
just undercounted repeat-labelled items by 198 (657 → 855, verified after
the fix), which would have understated the human-human ceiling's
reliability without ever raising an error.

**Lesson applied elsewhere in this codebase:** a special case fixed only
where it was first noticed is a bug still waiting in every other place the
same assumption is made silently.

---

## 3. The judge

### 3.1 Decision: use MT-Bench's own published prompt

`prompts.py`'s `SYSTEM_PROMPT` and `render_prompt` are based on MT-Bench's
published pair-wise judge prompts (Zheng et al. 2023) — a single-turn
template and a multi-turn variant that renders each side's own prior
exchange ahead of the current question (they can differ, since the two
systems may have answered turn 1 differently). Matching the published
prompt is what makes the judge's own kappa comparable to MT-Bench's
recorded GPT-4 kappa on the same items — a different prompt would mean
measuring a different judge, not judging this one.

### 3.2 Decision: position bias is defended twice, independently

The system prompt instructs the judge to ignore response order. That is an
instruction, not a guarantee, so `judge_with_position_check`
(`judge.py`) also asks every comparison a second time with the two
responses swapped, and checks whether the two verdicts agree once
translated back into the same slot's terms. A judge that is not actually
order-invariant will sometimes contradict itself — and MT-Bench's own
recorded GPT-4 verdicts do, on `"tie (inconsistent)"`, at 16% of the
`gpt4_pair` split. Silently mapping that into an ordinary tie was the first
draft; it was caught only because 16% of 2400 rows disappearing from the
"decisive" count would have been a large, unexplained gap. It is now
preserved as `position_unstable=True` rather than being folded in — see
§3.3.

On the held-out policy comparison specifically, this pair — the two systems
closest in overall quality — showed a striking **31.5%** position-instability
rate (46 of 146), several times a typical rate. That is not a data-quality
problem to paper over; it is exactly the signal a position-swap defense
exists to surface: this judge is measurably less reliable on close calls
than on clear-cut ones.

### 3.3 Decision: `[[A]]`/`[[B]]`/`[[C]]` parsing, last match wins

`judge.parse_verdict` takes the *last* bracketed tag in a reply, not the
first. A judge sometimes reasons toward a tentative answer and restates a
different one at the end — the last one is what it meant to submit. No tag
found raises `JudgeParseError` rather than defaulting to a tie: a judge
that never answered is a different situation from a judge that answered
"tie," and collapsing them would hide judge unreliability inside the tie
rate.

### 3.4 Decision: position-instability is an exclusion, not a tie

`scoring.compare_policies` drops a position-unstable comparison from the
scored count entirely (`n_position_unstable`, `position_instability_rate`)
rather than counting it as a tie. A genuine tie is real evidence ("about
equally good"); a judge contradicting itself under a swap is not evidence
of anything — averaging it in as "tie" would quietly convert judge
unreliability into apparent policy closeness. `report.py`'s caveats section
surfaces this rate explicitly, and — since an order-invariant judge (a
replay/recorded judge) never gets asked twice, so has no calibration-time
comparison point — flags a high held-out instability rate on its own terms
rather than silently passing because there was nothing to compare it
against (found while checking the caveats section against this build's own
31.5% figure and noticing it wasn't showing up).

### 3.5 Two judge implementations, one interface

`Judge` is a two-method interface (`verdict`, `is_order_invariant`).
`LLMJudge` calls a live model through an injected `backend: Callable[[str,
str], str]` — deliberately decoupled from any specific provider, so the
judging logic is testable with a plain function and never needs a network
call in a test. `providers.anthropic_judge_backend` is the concrete
backend: the official `anthropic` SDK, called directly (`client.messages.
create`), not through a multi-provider abstraction — this project only ever
calls Claude, so a routing layer would add a dependency for no capability
gained. `PrecomputedJudge` replays already-computed verdicts and is
`is_order_invariant=True` by construction: whatever produced the recorded
verdict already resolved position effects (MT-Bench ran its own swap
protocol for `gpt4_pair`), so asking it twice would just look up the same
row twice.

### 3.6 Decision: recorded-verdict conflicts are surfaced, not silently overwritten

`PrecomputedJudge.from_records(strict=False)` resolves two different
recorded verdicts for the same id to a flagged tie and counts the conflict
(`judge.conflicts`), rather than letting the second silently clobber the
first in a dict. In this build's own data, `judge.conflicts == 0` on the
real verdicts file — verified, not assumed — but the mechanism exists for
any verdicts file that isn't this project's own clean output.

### 3.7 Decision: concurrency is opt-in, and a parse failure is counted rather than fatal — and a real cost tradeoff between the two

`judge.judge_many` judges every row sequentially by default
(`--concurrency 1`) and, given a higher worker count, dispatches calls
through a `ThreadPoolExecutor` instead — identical results at any worker
count (`judge.verdict` is a pure function of one comparison, so there is no
reproducibility hazard the way there would be for anything touching shared
random state), only faster wall-clock time against a live judge. Separately,
`skip_parse_errors` catches a `JudgeParseError` (the judge never emitted a
parsable `[[A]]`/`[[B]]`/`[[C]]` tag — see §3.3) and excludes that
comparison instead of aborting the run.

Both were added in response to the same real failure, hit running this
project's own live judge: a `claude-haiku-4-5` run crashed the first two
times, on a reply that either got truncated before reaching its verdict tag
(a `MAX_OUTPUT_TOKENS` too low for a model that cannot be told to keep
effort low — `providers.py` has no `output_config.effort` support for
Haiku, confirmed against the real API, not assumed) or completed without a
tag at all. `--skip-parse-errors` turns that from "the whole run aborts,
including every already-paid-for call before the failure" into "one
comparison is excluded and counted."

That framing has one more layer to it once concurrency is added: a
`ThreadPoolExecutor.map` dispatches *every* row's call before consuming any
result, so without `skip_parse_errors`, a parse failure on row N no longer
bounds the damage to "N calls spent" the way the sequential path does —
every call already in flight when the failure surfaces still executes and
gets billed. The fix is the same one either way (turn on
`skip_parse_errors`), but the sequential path's implicit fail-fast cost
protection quietly stops applying once concurrency is on, which is worth
knowing before combining the two on a real paid run.

---

## 4. Label quality and the human ceiling

### 4.1 The measurement

`validation.flag_noisy_labels` computes two things: how many annotations
are outright ties, and — the more important number — how often humans agree
with *each other*. That ceiling is measured the same way judge-vs-human
agreement will be measured later (§5.1): pick two annotations of the same
item at random, ask how often they match. On the calibration sample used in
this build's own run: **72.6%** (146 repeat-labelled items). That, not
100%, is the maximum any judge could be expected to reach on this data.

### 4.2 Decision: contestedness is a property of the item

`validation.is_contested` flags an item whenever its annotators did not
*unanimously* agree — including a 2-1 majority, not only an outright split.
A judge that matches a 2-1 majority still resolved something humans
themselves did not agree on; every annotation on that item is flagged,
winning side included, because the disagreement belongs to the item, not to
whichever vote lost.

### 4.3 Decision: the ceiling is withheld below 5 repeat-labelled items

`MIN_REPEAT_LABELLED_FOR_CEILING = 5`. A pairwise-agreement rate computed
from a handful of items can swing by double digits on one more or fewer
agreement. Printing `None` and saying so is more honest than printing a
number nobody should trust at that sample size — `report.py` renders this
case as its own sentence rather than a number.

---

## 5. Calibration

### 5.1 Decision: measure agreement the same way the ceiling is measured

`calibration.calibrate` judges each comparison once and pairs that single
verdict against *every* human annotation the item carries — an item with
three annotators contributes three (human, judge) pairs, all sharing one
judge verdict. This is the same per-annotation, pairwise structure as the
human-human ceiling (§4.1), which is what makes "judge agreement" and
"ceiling" comparable numbers rather than two different statistics that
happen to share a unit.

### 5.2 Decision: gate on kappa, not raw agreement

Raw agreement rewards a judge that always answers the majority class. A
worked example from `stats.py`'s own tests: a judge that agrees 2 times out
of 3 by always saying "A" scores 66.7% raw agreement but **kappa 0.0** —
correctly, since it added no real information over guessing the common
label. `cohens_kappa` chance-corrects for exactly this. Verified against
this build's real data: the recorded GPT-4 judge scores 71.9% raw agreement
(barely below the 72.6% ceiling) but the more informative comparison is
kappa — 0.563 against a human ceiling of 0.572, i.e. **98.3%** of the
ceiling's chance-corrected agreement.

### 5.3 Decision: the gate is a ratio against the human ceiling, not a fixed threshold

`gate_on_calibration` fails unless the judge's kappa is at least
`MIN_SHARE_OF_HUMAN_KAPPA = 0.5` (50%) of the human-human kappa — not a
fixed bar like Landis & Koch's conventional kappa bands. Those bands assume
the reference labels are themselves close to perfect; here they are not —
humans agree with each other only a bit above two-thirds of the time on
this data (§4.1). A fixed "kappa ≥ 0.6" gate would be unreachable on data
this noisy regardless of how good the judge actually is, and would be
measuring the data's noise floor, not the judge. A ratio against a ceiling
measured the same way remains meaningful as the underlying data's own
difficulty changes.

**A real bug this gate had, found by code review and fixed:**
`human_agreement` is correctly withheld (`None`) below
`MIN_REPEAT_LABELLED_FOR_CEILING` (§4.3) — but `human_kappa` was computed
independently, gated only on `label_quality is not None`, not on that same
reliability floor. On a small calibration sample (a handful of
repeat-labelled items), `human_agreement` would correctly come back `None`
while `human_kappa` silently returned a real number computed from too
little data — and `gate_on_calibration` trusts `human_kappa` directly, so
the one function whose entire job is deciding whether to trust the judge
could pass or fail based on a kappa the codebase's own design says is too
noisy to display. Fixed by gating `human_kappa` on `human_agreement is not
None` — the same condition, not a separately-tracked one.

### 5.4 The confusion table and the position-flip rate

`calibration.confusion` (`(human, judge) -> count`) exists because the
headline agreement number can't distinguish a judge that errs evenly from
one that systematically over-calls ties or leans toward a slot — and those
imply different fixes. `position_flip_rate` is `None` for an order-invariant
judge (nothing to measure — see §3.5) rather than a misleading 0%.

### 5.5 A second judge, run live — the headline number is judge-dependent

§8 originally listed running a second judge through this harness as future
work: not anymore. `claude-opus-5` (this project's default) has run via
`PrecomputedJudge` replaying GPT-4's recorded MT-Bench verdicts throughout
this document; `claude-haiku-4-5` has since been run **live**, for real,
against the same held-out pair (`--calibration-items 250 --concurrency 10
--skip-parse-errors`, README).

Both clear the trust gate comfortably, but not by the same margin, and they
do not agree on the magnitude of the headline finding:

| | GPT-4 (recorded) | claude-haiku-4-5 (live) |
|---|---|---|
| Calibration sample | 300 sampled, 300 usable | 250 sampled, 248 usable (2 parse failures) |
| Human-human ceiling | 72.6% (kappa 0.572) | 75.4% (kappa 0.618) |
| Judge kappa | 0.563 | 0.551 |
| Share of human kappa | 98.3% | 89.1% |
| Held-out win rate (`gpt-3.5-turbo`) | **74.0%** [65.5%, 82.3%] | **86.6%** [78.1%, 94.5%] |
| Held-out position-instability | 31.5% (46/146) | 23.3% (34/146) |

The two win-rate intervals do not overlap. This is the exact finding "is
this judge good" vs "which judge is better" (§8's original framing) was
meant to surface: the 74.0% headline this project otherwise reports is
*this judge's* measured win rate, not *the* win rate, and it has now been
shown to move by more than either interval's width when the judge changes.

**What this does and doesn't explain, stated plainly rather than papered
over with a causal story:** the two calibration samples differ in size and
composition (300 vs 248 usable, independently sampled — even the
human-human ceiling itself differs between them, 72.6% vs 75.4%, which is
its own reminder that the ceiling is a property of *which items got
sampled*, not a fixed constant of the corpus). A tempting story would be
"the less stable judge is also the more decisive one" — but the data runs
the other way: Haiku was *more* stable on the held-out set (23.3% vs
31.5%) while *also* reporting the more extreme win rate, the opposite of
what that story predicts. No explanation here is backed by a controlled
comparison, so none is asserted. Settling this needs the same calibration
sample and seed run through both judges — listed in §8.

---

## 6. Scoring and statistics

### 6.1 Cohen's kappa

See §5.2. `stats.cohens_kappa` returns `None` (not 0 or 1) when every pair
used a single category for both raters — there is no variability left to
chance-correct against, and claiming a value there would imply more
information than the data contains.

### 6.2 Cluster bootstrap — because rows are not independent

`stats.bootstrap_ci` resamples whole `cluster_id` groups with replacement,
not individual rows. `mtbench.py` sets `cluster_id` to the MT-Bench question
id, so both turns of one dialogue move together in a resample. Verified with
a constructed example (`test_stats.py`): the same overall 50/50 split of
values, once concentrated into 2 big clusters and once spread across 100
independent ones, produces a dramatically wider interval in the 2-cluster
case — because the bootstrap has only 2 effectively-independent units to
resample from, not 100. Resampling individual rows here would report false
precision. This one function powers both calibration's agreement interval
and scoring's win-rate interval — same primitive, same reason it's needed
in both places.

### 6.3 Minimum detectable effect — "no difference" and "can't tell" are different findings

`stats.minimum_detectable_effect` gives the smallest deviation from 50% a
sample of a given size could reliably detect (normal approximation,
variance at its conservative maximum, p=0.5). `ScoreResult.is_underpowered`
checks the *observed* gap against this **before** looking at the confidence
interval at all — a small sample's CI can exclude 50% by noise alone, and
this check exists so that case is reported as "can't tell," not
misread as "no difference." These are different findings that call for
different actions: the first is fixed by collecting more held-out prompts;
the second is not fixable by more data, because the two policies really are
close.

### 6.4 Decision: ties count as half a win

`scoring._win_rate` scores a genuine tie as 0.5 toward each side — the
standard convention for pairwise preference scoring (equivalent to a
Bradley-Terry model with ties split evenly). Position-unstable comparisons
are excluded before this calculation runs at all (§3.4); only a judge's
*trusted* verdicts are averaged.

---

## 7. Assumptions, collected

| # | Assumption | If violated | Checked? |
|---|---|---|---|
| 1 | Slot order carries no quality information | Judge flattered by position bias, not response quality | Defended twice (§3.2); the mtbench-side canonicalization bug (§2.4) shows how easily this breaks silently |
| 2 | Comparisons sharing a cluster are correlated, not independent | Intervals too narrow, false precision | Enforced — cluster bootstrap (§6.2), demonstrated on constructed data |
| 3 | The human majority is a reasonable ground truth | The judge is being calibrated against noisy labels | **Not fully defensible** — a real share of items are contested (§4.2); reported, not resolved |
| 4 | The judge behaves comparably on this project's data and on MT-Bench's own | Calibration doesn't transfer | Same corpus here, so trivially satisfied in this build; would need re-checking on any other corpus |
| 5 | Both systems in a held-out comparison answered the same question | Comparison not controlled; a row would be meaningless | Enforced structurally (§2.4) |
| 6 | Judge agreement is comparable to the human-human ceiling | The gate's ratio means nothing | Enforced by construction — both measured pairwise, per-annotation (§5.1) |
| 7 | A position-unstable verdict carries no information | Averaging it in would misrepresent judge unreliability as policy closeness | Excluded, not tied (§3.4) |
| 8 | 5 points of win rate is a difference worth acting on | "Underpowered" mislabelled | Configurable (`minimum_detectable_effect`'s alpha/power), not independently validated against a real decision threshold |

Assumption 3 is the least defensible of these and is not currently tested —
the named fix (stratifying by how contested an item is, rather than pooling
all items into one ceiling) is listed in §8.

---

## 8. What would change with more time

**In priority order:**

1. **Test assumption 3 directly** — does calibration measured only on
   uncontested items produce a materially different kappa/ceiling than the
   pooled number this build reports? If so, the ceiling used for gating
   should probably stratify by contestedness rather than pool across it.
2. **A bias-correction stage.** This build deliberately does not include
   one (README, "Assumptions and known limits") — the judge's own
   calibrated, gated verdicts are reported directly. A Rogan–Gladen-style
   correction (inverting the judge's measured sensitivity/specificity) is
   the standard next step, but it would need its own held-out validation
   (leave-one-pair-out across several system pairs, not just this one) to
   trust before using — not a small addition, and worth building only with
   enough held-out pairs to actually test it against.
3. **Reward hacking — the length-confound finding, followed up properly.**
   `REWARD_HACKING.md`'s length-normalized re-judging (25 items, one judge,
   one seed) found A's win rate drop from 87.5% to 64.7% once response
   length was equalized — a real signal, not a rigorous estimate (no CI was
   computed, matching §6.3's own standard elsewhere in this document). The
   next step is re-running that same length-normalized comparison at the
   full 146-item scale, with a bootstrap CI, against this project's default
   judge (`claude-opus-5`, not the `claude-haiku-4-5` used to keep that
   experiment's live spend minimal) — to find out whether the confound
   survives at the sample size the headline number is actually reported at.
4. **A second judge model — done; the controlled version of it is not.**
   §5.5 ran `claude-haiku-4-5` live against the same held-out pair
   GPT-4 was measured on, and found a real, non-overlapping difference in
   the headline win rate (74.0% vs 86.6%). What's still missing is the
   *controlled* version — the same calibration sample and seed run through
   both judges, so a genuine judge effect can be separated from the two
   runs having sampled different calibration items.
5. **Deeper dialogues.** MT-Bench is two turns; a corpus with longer
   conversations would need `prompts.render_prompt`'s multi-turn template
   generalized past "one prior exchange," and a template invented past what
   MT-Bench's own recorded verdicts were judged against would stop being
   comparable to them.

---

## References

- Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — judge prompts, position-bias protocol, human-preference corpus
- Cohen (1960) — the kappa statistic
- Efron (1979) — the bootstrap
