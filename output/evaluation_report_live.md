# Evaluation report

Comparing **gpt-3.5-turbo** against **vicuna-13b-v1.2**.

## Bottom line

**gpt-3.5-turbo is preferred.** The judge favoured it on 86.6% of scored comparisons, and the interval (78.1%–94.5%) sits entirely above 50%. The judge agrees with humans 71.8% of the time against a human-human ceiling of 75.4%, so its verdicts are about as reliable as a second annotator's.

## 1. How far the judge can be trusted

Judge: **claude-haiku-4-5**, measured on 248 human-labelled comparisons carrying 432 annotations. Agreement is per annotation, so it means "matches one randomly drawn human" — the same quantity the ceiling measures between two humans.

| | |
|---|---|
| Agreement with humans | 71.8% [65.5%, 77.6%] |
| Cohen's kappa | 0.551 |
| **Human-human ceiling** | **75.4%** (kappa 0.618) |
| Judge as a share of that ceiling | 95.1% raw, 89.1% chance-corrected |
| Position-bias flip rate | 10.1% |
| Excluded — no parsable verdict | 2 comparisons |

> Humans agree with each other only 75.4% of the time on this data, so that — not 100% — is the maximum any judge could reach. Read the judge's score against it.

**Where the disagreements go**

| human \ judge | A | B | tie | total |
|---|---|---|---|---|
| **A** | 142 | 15 | 6 | 163 |
| **B** | 13 | 144 | 16 | 173 |
| **tie** | 36 | 36 | 24 | 96 |

> Rows are what the humans said, columns what the judge said. The diagonal is agreement. A heavy `tie` column means the judge declines to choose where people do; an asymmetry between the A→B and B→A cells is residual position bias that survived the swap.

Label quality: 22.2% of human annotations are ties, and 103 annotations sit on the 36 items where humans disagreed with each other.

## 2. The comparison

### Headline

The judge preferred **gpt-3.5-turbo** on 86.6% of 112 scored comparisons (95% CI 78.1%–94.5%). Counts: 97 to 15, with 0 ties.

**Supporting detail**

- Decisive only (112 comparisons): 100.0%
- Smallest gap this sample could detect: ±13.2%
- Excluded because the judge contradicted itself when the responses were swapped: 34 of 146 (23.3%)

## 3. What this does not account for

- **The judge is less stable here than during calibration.** It contradicted itself on 23.3% of these comparisons, against 10.1% on the data it was calibrated on. Treat the headline with proportionally more caution than its interval alone suggests.
