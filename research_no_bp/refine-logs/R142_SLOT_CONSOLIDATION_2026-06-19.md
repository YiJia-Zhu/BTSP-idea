# R142 Slot Consolidation Credit

**Date**: 2026-06-19  
**Task**: bAbI `en-qa15` and `en-qa16`, full `900/100/1000` split  
**Goal**: Close R140's residual QA16 value-slot gap without using paraphrase-local structure labels.

## Setup

R142 extends `babi_attribute_delayed_credit_experiment.py` with:

- `--slot-consolidation-mode {off,credit,error,all}`

The claimable R142 setting is:

```bash
python babi_attribute_delayed_credit_experiment.py \
  --out-dir output/babi_attribute_slot_consolidation_error_both_seed0 \
  --configs en-qa15 en-qa16 \
  --slot-consolidation-mode error \
  --seed 0
```

Repeated for seeds `1` and `2`.

Mechanism:

- R140 still performs final-answer-gated statement credit.
- On rows that remain answer-wrong, R142 additionally consolidates slot prototypes for local surface-cue candidates in that story.
- Consolidation is slot-only: it updates entity/value prototypes, not event weights.
- It uses paraphrased sentence surface cues (`fear`, `classified`, `colored`, color words), not `_source_text` labels or test labels.

`error` mode was selected because it repairs QA16 seed2 while staying stricter than `all`; `credit` mode repairs seed1 but not seed2.

Outputs:

- `output/babi_attribute_slot_consolidation_r142/aggregate_summary.csv`
- `output/babi_attribute_slot_consolidation_r142/detector_summary.csv`
- `output/babi_attribute_slot_consolidation_r142/credit_summary.csv`
- `output/babi_attribute_slot_consolidation_r142/paired_deltas.csv`
- `output/babi_attribute_slot_consolidation_r142/selection.json`

## Raw Data Table

Three-seed test aggregate:

| Task | Method | Mean acc | Std acc | Mean CE | Std CE |
|---|---|---:|---:|---:|---:|
| QA15 | seeded pre-credit | 0.567333 | 0.059744 | 2.346010 | 0.507952 |
| QA15 | slot-consolidated credit | 1.000000 | 0.000000 | 0.003610 | 0.000407 |
| QA15 | structural upper | 1.000000 | 0.000000 | 0.002293 | 0.001647 |
| QA16 | seeded pre-credit | 0.332000 | 0.100464 | 2.255045 | 0.727103 |
| QA16 | slot-consolidated credit | 0.995000 | 0.000000 | 0.041655 | 0.004701 |
| QA16 | structural upper | 0.995000 | 0.000000 | 0.042041 | 0.002174 |

Paired delta versus R140:

| Task | Delta acc mean | Delta CE mean |
|---|---:|---:|
| QA15 | 0.000000 | 0.000000 |
| QA16 | +0.025000 | -0.090398 |

Detector test aggregate for slot-consolidated credit:

| Task | Statement event | Statement entity | Statement value | Query type | Query subject |
|---|---:|---:|---:|---:|---:|
| QA15 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| QA16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Findings

1. Error-row slot consolidation closes the QA16 residual gap. QA16 improves from R140 `0.970 / 0.132` to `0.995 / 0.0417`, matching the structural-label upper bound.

2. QA15 remains unchanged at perfect accuracy, so consolidation does not damage the already solved relation-deduction task.

3. The improvement comes from value-slot coverage. R140 QA16 statement value accuracy was `0.985`; R142 raises it to `1.000` while keeping event/entity/query metrics at `1.000`.

4. This is more targeted than R141 pair credit. Pairwise replacement had zero held-out delta; slot consolidation gives a positive QA16 paired delta without broad query updates or raw replay.

## Decision

R142 is a positive result and should supersede R140 as the QA15/QA16 strong-paraphrase answer-credit point.

Boundary:

- the candidate generator is still surface-cue constrained;
- consolidation is local and no-BP, but it is not yet a general language parser;
- next step should test whether the same consolidation idea transfers to less templated paraphrases or newly exported bAbI tasks beyond QA15/QA16.
