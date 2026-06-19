# R141 Pair Statement Credit Boundary

**Date**: 2026-06-19  
**Task**: bAbI `en-qa15` and `en-qa16`, full `900/100/1000` split  
**Goal**: Test whether distributing one final-answer credit signal to a pair of candidate statements improves R140, especially the residual QA16 value-slot gap.

## Setup

R141 extends `babi_attribute_delayed_credit_experiment.py` with:

- `--enable-pair-statement-credit`

For each wrong train row, the trainer now evaluates both:

- single-statement event replacements;
- two-statement event replacement pairs from the same local surface-cue candidate set.

If a pair gives the best final-answer log-probability improvement, both statement detector updates receive the same third-factor gain. This is a small eligibility-style extension over R140's best-one-statement credit.

Command pattern:

```bash
python babi_attribute_delayed_credit_experiment.py \
  --out-dir output/babi_attribute_delayed_credit_pair_seed0 \
  --configs en-qa15 en-qa16 \
  --enable-pair-statement-credit \
  --seed 0
```

Repeated for seeds `1` and `2`.

Outputs:

- `output/babi_attribute_pair_credit_r141/pair_aggregate_summary.csv`
- `output/babi_attribute_pair_credit_r141/paired_deltas.csv`
- `output/babi_attribute_pair_credit_r141/delta_summary.csv`
- `output/babi_attribute_pair_credit_r141/pair_credit_summary.csv`
- `output/babi_attribute_pair_credit_r141/selection.json`

## Raw Data Table

Paired test deltas versus R140 two-epoch statement-only answer credit:

| Task | Delta acc mean | Delta acc std | Delta CE mean | Delta CE std |
|---|---:|---:|---:|---:|
| QA15 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| QA16 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Pair-credit update activity:

| Task | Epoch | Pair update mean | Pair candidates searched mean |
|---|---:|---:|---:|
| QA15 | 0 | 0.00 | 44.00 |
| QA15 | 1 | 0.00 | 0.00 |
| QA16 | 0 | 1.67 | 940.00 |
| QA16 | 1 | 0.00 | 657.33 |

## Findings

1. Pair-statement credit does not improve R140 on test accuracy or CE. The paired delta is exactly zero across QA15 and QA16 for all three seeds.

2. The mechanism is not completely inactive. QA16 epoch 0 selects pair updates in some rows, but these updates do not change final held-out metrics.

3. The residual QA16 gap is therefore not solved by simply replacing two statements jointly. The stronger hypothesis is that value-slot prototype coverage needs broader eligibility over all relevant `classified` / `colored` statements, or a separate value-slot consolidation rule, rather than pairwise coordinate replacement.

## Decision

R141 is a negative boundary. Keep `--enable-pair-statement-credit` as a diagnostic option, but do not make it the default or claim it as an improvement.

Next step:

- test value-slot consolidation driven by answer-credit eligibility, for example updating all high-confidence same-cue value prototypes in a row when the answer-credit candidate identifies the missing cue class;
- alternatively add a train-only slot coverage objective that remains local and no-BP but does not require paraphrase-local structure labels.
