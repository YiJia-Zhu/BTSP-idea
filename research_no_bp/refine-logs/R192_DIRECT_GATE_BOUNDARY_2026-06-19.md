# R192: Direct Edge-Path Gate Boundary

Date: 2026-06-19

## Purpose

R191 showed that R190's direct edge-path score channel has train-post gains but
held-out helpful and harmful exact flips cancel. R192 adds a small local
inhibitory gate to test whether protecting confident pre-direct predictions
reduces harmful direct flips.

This remains no-BP and local:

- no pretrained model or API backbone
- no raw prompt storage
- no task-specific parser
- no statistical n-gram table
- inference gate uses only local score margins

## Implementation

Added optional direct-score gates inside `AnswerSlotReadoutMemory`.

New args:

- `--answer-slot-direct-pre-margin-protect`
- `--answer-slot-direct-margin-min`

Behavior when enabled:

1. Compute the answer-slot scores before direct edge-path delta.
2. Compute direct-delta top score and margin.
3. Suppress direct delta if it is weak.
4. Suppress direct delta if pre-direct answer readout already has enough margin.

Default values are `0.0`, so old R190 behavior is preserved unless the gate is
explicitly enabled.

## Runs

Smoke:

- `output/babi_unified_qa19_r192_direct_gate_smoke`

Medium seed0:

- `output/babi_unified_qa19_r192_direct_gate_medium_s0`
- comparison: `output/babi_unified_qa19_r192_direct_gate_comparison`
- flip versus R186: `output/babi_unified_qa19_r192_gate_vs_r186_flip_diagnostic`
- flip versus R190: `output/babi_unified_qa19_r192_gate_vs_r190_flip_diagnostic`

Gate setting:

- `answer_slot_direct_pre_margin_protect = 0.20`
- `answer_slot_direct_margin_min = 0.02`
- direct score scale: `1.0`

## Results

`output/babi_unified_qa19_r192_direct_gate_comparison/comparison_summary.csv`:

| variant | split | full acc | full-token acc | full CE |
|---|---|---:|---:|---:|
| R186 soft | validation | 0.1300 | 0.3100 | 1.3200 |
| R186 soft | test | 0.1267 | 0.3300 | 1.2818 |
| R190 direct | validation | 0.1400 | 0.3150 | 1.3251 |
| R190 direct | test | 0.1233 | 0.3350 | 1.2813 |
| R192 gate | validation | 0.1300 | 0.3200 | 1.3131 |
| R192 gate | test | 0.1233 | 0.3300 | 1.2880 |

Gate stats:

- checks: 4000
- applied: 1702
- protected: 1806
- weak/direct suppressed: 492

Flip summary:

| comparison | split | helpful | harmful | net | full acc delta | token acc delta |
|---|---|---:|---:|---:|---:|---:|
| R192 - R186 | validation | 3 | 3 | 0 | +0.0000 | +0.0100 |
| R192 - R186 | test | 10 | 11 | -1 | -0.0033 | +0.0000 |
| R192 - R190 | validation | 0 | 1 | -1 | -0.0100 | +0.0050 |
| R192 - R190 | test | 4 | 4 | 0 | +0.0000 | -0.0050 |

## Interpretation

The gate is active and suppresses many direct-score applications, but it does
not improve held-out full-answer exact match.

Compared with R190:

- test full exact is unchanged (`0.1233`)
- test full-token accuracy drops (`0.3350 -> 0.3300`)
- test CE worsens (`1.2813 -> 1.2880`)

Compared with R186:

- test full exact remains worse (`0.1267 -> 0.1233`)
- exact helpful/harmful flips remain net negative (`10 - 11 = -1`)

This means scalar margin protection is too coarse. It removes some direct
changes but does not identify the harmful path cases that matter for QA19.

## Boundary

Supported claim:

> A local pre-direct margin gate can reduce direct score applications, but this
> specific scalar gate does not recover held-out QA19 full-answer exact accuracy.

Unsupported claims:

- Protecting high-margin pre-direct predictions solves the R190 harmful flips.
- Direct score failures can be fixed with a single scalar margin threshold.
- R192 should replace R186 as the QA19 default.

## Next

The next mechanism should use richer local counterfactual evidence:

1. Train a near-miss path suppressor from competing path features that share one
   query endpoint but lead to the wrong direction token.
2. Add answer-slot coupling so slot 1 is conditioned by a local trace of slot 0
   rather than an independent edge-path score.
3. Instrument component margins in prediction rows before another gate sweep.
