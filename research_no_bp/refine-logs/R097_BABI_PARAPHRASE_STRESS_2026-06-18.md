# R097 bAbI Paraphrase Stress

**Date**: 2026-06-18

## Purpose

R091 solved bAbI QA1/QA2/QA3 with learned local event/query front-ends plus a no-BP role-binding state, but the remaining concern was that the front-ends may only learn the narrow original bAbI grammar. R097 rewrites bAbI surface forms while preserving answers and world-state semantics, then evaluates whether the learned no-BP front-end generalizes out of distribution or can adapt online-style when trained on the paraphrased surface forms.

This is a boundary test, not a final natural-language solution. The method remains pure local no-BP: fixed random token/position features, local perceptron-style event/query type updates, local slot prototypes, and local delta-Hebbian role/state matrices. Test answers are not used for updates. The diagnostic normalized front-end is reported only as an analysis aid, not as the main method.

## Data

- Source: `data/babi_qa_processed/en-qa2` and `data/babi_qa_processed/en-qa3`
- Splits: 900 train / 100 validation / 1000 test per task
- Stress mode: `strong` paraphrase at evaluation time
- Examples of rewrites:
  - `moved to` -> `relocated to`
  - `went to` -> `headed to`
  - `got the football` -> `acquired the football`
  - `Where is the football?` -> `Which room contains the football?`
  - `Where was the apple before the bedroom?` -> `Before the bedroom, which room held the apple?`

Implementation note: detector features use the paraphrased text, but training/evaluation labels for event and query diagnostics are derived from `_source_text` / `_source_question`, i.e. the original semantic form. This prevents the diagnostic labels from becoming invalid after the rewrite.

## Commands

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_paraphrase_stress_experiment.py \
  --out-dir output/babi_paraphrase_stress_smoke_fix \
  --configs en-qa2 en-qa3 \
  --train-strengths none strong \
  --eval-strength strong \
  --train-limit 80 --eval-limit 80 \
  --phase-dim 8 --phase-epochs 1 \
  --event-eval-limit 1500 --query-eval-limit 300
```

Medium QA2 key methods:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_paraphrase_stress_experiment.py \
  --out-dir output/babi_paraphrase_stress_qa2_medium_key \
  --configs en-qa2 \
  --train-strengths none strong \
  --eval-strength strong \
  --methods learned aware \
  --event-eval-limit 3000 --query-eval-limit 1000
```

Medium QA3 key methods:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_paraphrase_stress_experiment.py \
  --out-dir output/babi_paraphrase_stress_qa3_medium_key \
  --configs en-qa3 \
  --train-strengths none strong \
  --eval-strength strong \
  --methods learned aware \
  --event-eval-limit 3000 --query-eval-limit 1000
```

## Medium Results

All rows below are evaluated on `strong` paraphrased test inputs.

| Task | Train surface | Method | Test acc | Test CE | Main detector result |
|---|---|---|---:|---:|---|
| QA2 | original | learned no-BP front-end | 0.409 | 4.457 | event acc 0.773, query acc 1.000, subject acc 0.357 |
| QA2 | original | normalized diagnostic front-end | 0.745 | 1.940 | event acc 0.843, query/subject acc 1.000 |
| QA2 | strong paraphrase | learned no-BP front-end | 1.000 | 0.002 | event/query/slot acc 1.000 |
| QA2 | strong paraphrase | normalized diagnostic front-end | 0.915 | 0.684 | event acc 0.979, query/subject acc 1.000 |
| QA3 | original | learned no-BP front-end | 0.739 | 1.559 | event acc 0.919, query acc 1.000, subject acc 0.601, destination acc 1.000 |
| QA3 | original | normalized diagnostic front-end | 0.987 | 0.082 | event/query/slot acc 1.000 |
| QA3 | strong paraphrase | learned no-BP front-end | 1.000 | 0.003 | event/query/slot acc 1.000 |
| QA3 | strong paraphrase | normalized diagnostic front-end | 0.984 | 0.137 | event acc 0.984, query/slot acc 1.000 |

Smoke results reproduce the same pattern at 80 examples per split:

- QA2 original-train -> strong-test learned acc 0.400; strong-train -> strong-test learned acc 1.000.
- QA3 original-train -> strong-test learned acc 0.562; strong-train -> strong-test learned acc 1.000.

## Interpretation

R097 gives a useful negative/positive boundary:

- Negative: R091 is not robust enough to claim broad natural-language parsing. When trained only on original bAbI grammar and tested on strong paraphrases, QA2 falls to 0.409 and QA3 to 0.739. The failure localizes mostly to event type and query-subject slot extraction rather than the role-binding state itself.
- Positive: the same local no-BP front-end recovers to 1.000 test accuracy when exposed to the paraphrased surface forms during training. This means the local perceptron/prototype event-query front-end can adapt to new grammar without answer-label updates or raw replay.
- Diagnostic: the normalized front-end improves original-train strong-test performance, especially QA3, but it is a hand-normalization analysis path and should not be promoted to the final biomimetic method.

## Next Step

The next bAbI QA step should reduce local structural supervision. A concrete R098 direction is delayed QA-level credit: store event/query eligibility traces during a story, predict the answer, then use answer error as an apical/third-factor signal to update event/query front-end weights and slot prototypes. That would move the current system from locally labeled parser learning toward a more realistic online no-BP correction loop.
