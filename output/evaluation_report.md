# Evaluation report

Comparing **gpt-3.5-turbo** against **vicuna-13b-v1.2**.

## Bottom line

**gpt-3.5-turbo is preferred.** The judge favoured it on 74.0% of scored comparisons, and the interval (65.5%–82.3%) sits entirely above 50%. The judge agrees with humans 71.9% of the time against a human-human ceiling of 72.6%, so its verdicts are about as reliable as a second annotator's.

## 1. How far the judge can be trusted

Judge: **GPT-4 (MT-Bench recorded verdicts)**, measured on 300 human-labelled comparisons carrying 534 annotations. Agreement is per annotation, so it means "matches one randomly drawn human" — the same quantity the ceiling measures between two humans.

| | |
|---|---|
| Agreement with humans | 71.9% [66.6%, 76.9%] |
| Cohen's kappa | 0.563 |
| **Human-human ceiling** | **72.6%** (kappa 0.572) |
| Judge as a share of that ceiling | 99.1% raw, 98.3% chance-corrected |
| Position bias | not applicable — judge is order-invariant by construction |

> Humans agree with each other only 72.6% of the time on this data, so that — not 100% — is the maximum any judge could reach. Read the judge's score against it.

**Where the disagreements go**

| human \ judge | A | B | tie | total |
|---|---|---|---|---|
| **A** | 161 | 21 | 27 | 209 |
| **B** | 22 | 166 | 22 | 210 |
| **tie** | 22 | 36 | 57 | 115 |

> Rows are what the humans said, columns what the judge said. The diagonal is agreement. A heavy `tie` column means the judge declines to choose where people do; an asymmetry between the A→B and B→A cells is residual position bias that survived the swap.

Label quality: 21.5% of human annotations are ties, and 144 annotations sit on the 50 items where humans disagreed with each other.

## 2. The comparison

### Headline

The judge preferred **gpt-3.5-turbo** on 74.0% of 100 scored comparisons (95% CI 65.5%–82.3%). Counts: 66 to 18, with 16 ties.

**Supporting detail**

- Decisive only (84 comparisons): 84.0%
- Smallest gap this sample could detect: ±14.0%
- Excluded because the judge contradicted itself when the responses were swapped: 46 of 146 (31.5%)

## 3. What this does not account for

- **No calibration-time position-instability rate to compare against** (this judge is order-invariant by construction, so calibration never exercises this check) — 31.5% of held-out comparisons were excluded here for contradicting themselves under a position swap. Judge that number on its own terms.
