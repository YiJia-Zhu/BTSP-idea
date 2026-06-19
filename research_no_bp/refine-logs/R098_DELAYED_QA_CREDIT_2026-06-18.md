# R098 Delayed QA-Level Credit

**Date**: 2026-06-18

## Purpose

R097 showed that R091's learned event/query front-end is grammar-fragile under strong paraphrase. R098 tests whether the front-end can be corrected by delayed answer-level credit: read the story, predict the answer, then use the answer error as an apical/third-factor signal to update event/query detector weights and slot prototypes.

This is intentionally not BP. The trainer searches local event/query alternatives, scores the final QA answer, and applies local target/wrong-winner updates only to the selected sentence or question feature. Train answers are used as the delayed credit signal; test answers are used only for evaluation. No raw story replay is stored in model state.

## Implementation

Script: `babi_delayed_credit_experiment.py`

Main variants:

- `r097_original_structural_seed`: R097-style structural local-label front-end trained on original bAbI grammar, evaluated on strong paraphrase.
- `qa_credit_cold_answer_only`: no structural seed; trains on strong paraphrase using only final QA answer credit.
- `qa_credit_seeded_answer_only`: starts from original structural seed, then adapts on strong paraphrase using only final QA answer credit.
- `r097_same_surface_structural_upper`: local structural labels on strong paraphrase; diagnostic upper bound only.

Candidate search is local: query candidates are generated from question tokens; event candidates are generated from sentence tokens, known answer-location vocabulary, and observed object-like tokens. The selected update is the candidate that most improves final answer log-probability.

## Commands

QA2 smoke, 80 examples:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_delayed_credit_experiment.py \
  --out-dir output/babi_delayed_credit_qa2_smoke \
  --configs en-qa2 \
  --train-limit 80 --eval-limit 80 \
  --credit-epochs 1 \
  --max-event-updates-per-row 1 \
  --max-credit-sentences-per-row 8 \
  --event-eval-limit 1200 --query-eval-limit 300 \
  --include-structural-upper
```

QA2 query-only ablation:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_delayed_credit_experiment.py \
  --out-dir output/babi_delayed_credit_qa2_query_smoke \
  --configs en-qa2 \
  --train-limit 80 --eval-limit 80 \
  --credit-epochs 1 \
  --max-event-updates-per-row 0 \
  --max-credit-sentences-per-row 4 \
  --event-eval-limit 1200 --query-eval-limit 300 \
  --include-structural-upper
```

QA2 larger seeded run:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_delayed_credit_experiment.py \
  --out-dir output/babi_delayed_credit_qa2_seeded_300 \
  --configs en-qa2 \
  --seed-mode seeded \
  --train-limit 300 --eval-limit 300 \
  --credit-epochs 1 \
  --max-event-updates-per-row 1 \
  --max-credit-sentences-per-row 8 \
  --event-eval-limit 2000 --query-eval-limit 600
```

QA3 query-only smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_delayed_credit_experiment.py \
  --out-dir output/babi_delayed_credit_qa3_query_smoke \
  --configs en-qa3 \
  --train-limit 80 --eval-limit 80 \
  --credit-epochs 1 \
  --max-event-updates-per-row 0 \
  --max-credit-sentences-per-row 4 \
  --event-eval-limit 1200 --query-eval-limit 300 \
  --include-structural-upper
```

QA3 event-credit tiny smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_delayed_credit_experiment.py \
  --out-dir output/babi_delayed_credit_qa3_event30_smoke \
  --configs en-qa3 \
  --seed-mode seeded \
  --train-limit 30 --eval-limit 30 \
  --credit-epochs 1 \
  --max-event-updates-per-row 1 \
  --max-credit-sentences-per-row 8 \
  --event-eval-limit 800 --query-eval-limit 200
```

## Results

All test rows use strong paraphrase.

| Task / size | Method | Test acc | Test CE | Notes |
|---|---|---:|---:|---|
| QA2 80 | majority | 0.075 | 2.007 | no-memory baseline |
| QA2 80 | R097 original structural seed | 0.400 | 4.562 | grammar-fragile seed |
| QA2 80 | structural same-surface upper | 1.000 | 0.002 | uses local structural labels on strong paraphrase |
| QA2 80 | cold answer-credit only | 0.075 | 2.007 | no bootstrapping from answer signal alone |
| QA2 80 | seeded answer-credit, query only | 0.762 | 1.902 | 15 query updates |
| QA2 80 | seeded answer-credit, query + event | 0.900 | 0.843 | 15 query updates, 20 event updates |
| QA2 300 | R097 original structural seed | 0.447 | 4.357 | larger partial eval |
| QA2 300 | seeded answer-credit, query + event | 0.863 | 1.211 | 50 query updates, 99 event updates |
| QA3 80 | R097 original structural seed | 0.562 | 2.402 | before-location setting |
| QA3 80 | seeded answer-credit, query only | 0.588 | 3.391 | acc improves slightly, CE worsens |
| QA3 30 | R097 original structural seed | 0.500 | 2.450 | tiny event-credit run |
| QA3 30 | seeded answer-credit, query + event | 0.600 | 2.612 | acc improves, CE still worsens |

QA2 detector probes on the 300-example run also move in the intended direction:

- R097 original seed test event acc 0.8085, query subject acc 0.330.
- Seeded answer-credit test event acc 0.888, query subject acc 1.000.

## Interpretation

R098 is a partial positive result, not a solved parser:

- Positive: answer-level credit can adapt a seeded no-BP front-end to strong paraphrase without local structural labels on the paraphrased training stream. QA2 improves from 0.400/0.447 to 0.900/0.863 in smoke and 300-example runs.
- Boundary: cold-start answer-credit does not bootstrap from zero; it stays at majority on QA2. Some prior structure or a stronger recurrent/dendritic parser state is still needed.
- Boundary: QA3 gains are smaller and CE worsens. Before-location reasoning needs more than query repair; event credit must become more efficient and better calibrated.
- Engineering boundary: naive event candidate search is slow on QA3 long contexts. Full medium needs prefix/suffix state caching or a learned sentence-attention gate before scaling.

## Next Step

Do not claim R098 as final. The next implementation should cache role-state prefixes/suffixes so candidate event perturbations can be scored without replaying the whole story, then rerun QA2/QA3 full 900/1000 splits. Mechanistically, the next model should replace brute candidate search with a local dendritic/recurrent parser state that emits eligibility traces for the answer-error apical signal.
