# Evaluation report

Comparing **gpt-3.5-turbo** against **vicuna-13b-v1.2**.

## Bottom line

**gpt-3.5-turbo is preferred.** The judge favoured it on 81.3% of scored comparisons, and the interval (74.3%–88.6%) sits entirely above 50%. The judge agrees with humans 74.8% of the time against a human-human ceiling of 75.4%, so its verdicts are about as reliable as a second annotator's.

## 1. How far the judge can be trusted

Judge: **claude-haiku-4-5 (live)**, measured on 250 human-labelled comparisons carrying 436 annotations. Agreement is per annotation, so it means "matches one randomly drawn human" — the same quantity the ceiling measures between two humans.

| | |
|---|---|
| Agreement with humans | 74.8% [69.1%, 80.3%] |
| Cohen's kappa | 0.605 |
| **Human-human ceiling** | **75.4%** (kappa 0.618) |
| Judge as a share of that ceiling | 99.1% raw, 97.8% chance-corrected |
| Position-bias flip rate | 15.2% |

> Humans agree with each other only 75.4% of the time on this data, so that — not 100% — is the maximum any judge could reach. Read the judge's score against it.

**Where the disagreements go**

| human \ judge | A | B | tie | total |
|---|---|---|---|---|
| **A** | 141 | 10 | 13 | 164 |
| **B** | 12 | 145 | 18 | 175 |
| **tie** | 24 | 33 | 40 | 97 |

> Rows are what the humans said, columns what the judge said. The diagonal is agreement. A heavy `tie` column means the judge declines to choose where people do; an asymmetry between the A→B and B→A cells is residual position bias that survived the swap.

Label quality: 22.2% of human annotations are ties, and 103 annotations sit on the 36 items where humans disagreed with each other.

## 2. The comparison

### Headline

The judge preferred **gpt-3.5-turbo** on 81.3% of 123 scored comparisons (95% CI 74.3%–88.6%). Counts: 98 to 21, with 4 ties.

**Supporting detail**

- Decisive only (119 comparisons): 96.7%
- Smallest gap this sample could detect: ±12.6%
- Excluded because the judge contradicted itself when the responses were swapped: 23 of 146 (15.8%)

## 3. What this does not account for

- None of the checks this report runs (position-instability drift) found anything to flag here.
