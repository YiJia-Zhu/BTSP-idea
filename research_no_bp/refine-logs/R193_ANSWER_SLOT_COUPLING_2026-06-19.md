# R193: Answer-Slot Coupling Boundary

Date: 2026-06-19

## Purpose

R191 showed that direct edge-path scores can improve individual answer-token
probabilities but do not reliably improve full two-token QA19 answers. R193
tests whether a local answer-slot coupling trace can coordinate the second
direction token with the first direction token.

This remains no-BP and local:

- no pretrained model or API backbone
- no raw prompt storage
- no task-specific parser
- no statistical n-gram table
- slot 1 coupling uses only the current slot feature and previous answer token

## Implementation

Added an optional coupling prototype bank inside `AnswerSlotReadoutMemory`.

New args:

- `--answer-slot-coupling-slots`
- `--answer-slot-coupling-lr`
- `--answer-slot-coupling-wrong-lr`
- `--answer-slot-coupling-score-scale`

Mechanism:

1. For answer slot 0, coupling is inactive.
2. For answer slot 1+, compute a local multiplicative feature:
   `slot_feature * token_code(previous_answer_token)`.
3. Train a slot-specific prototype bank for the target answer token, with the
   same target/wrong local update style as the main answer-slot readout.

Default score scale is `0.0`, so previous runs are unchanged unless coupling is
explicitly enabled.

## Runs

Smoke:

- `output/babi_unified_qa19_r193_slot_coupling_smoke`

Medium seed0:

- coupling scale 0.5: `output/babi_unified_qa19_r193_slot_coupling_scale05_medium_s0`
- coupling scale 1.0: `output/babi_unified_qa19_r193_slot_coupling_medium_s0`
- direct + coupling: `output/babi_unified_qa19_r193_direct_coupling_medium_s0`
- comparison: `output/babi_unified_qa19_r193_slot_coupling_comparison`

Flip diagnostics:

- `output/babi_unified_qa19_r193_coupling_vs_r186_flip_diagnostic`
- `output/babi_unified_qa19_r193_coupling_s05_vs_r186_flip_diagnostic`
- `output/babi_unified_qa19_r193_direct_coupling_vs_r186_flip_diagnostic`
- `output/babi_unified_qa19_r193_direct_coupling_vs_r190_flip_diagnostic`

## Results

`output/babi_unified_qa19_r193_slot_coupling_comparison/comparison_summary.csv`:

| variant | split | full acc | full-token acc | full CE |
|---|---|---:|---:|---:|
| R186 soft | validation | 0.1300 | 0.3100 | 1.3200 |
| R186 soft | test | 0.1267 | 0.3300 | 1.2818 |
| R190 direct | validation | 0.1400 | 0.3150 | 1.3251 |
| R190 direct | test | 0.1233 | 0.3350 | 1.2813 |
| R193 coupling 0.5 | validation | 0.1200 | 0.3000 | 1.3151 |
| R193 coupling 0.5 | test | 0.1233 | 0.3217 | 1.2753 |
| R193 coupling 1.0 | validation | 0.1300 | 0.3000 | 1.3128 |
| R193 coupling 1.0 | test | 0.1267 | 0.3250 | 1.2714 |
| R193 direct+coupling | validation | 0.1400 | 0.2950 | 1.3198 |
| R193 direct+coupling | test | 0.1200 | 0.3267 | 1.2731 |

Coupling stats for scale 1.0:

- coupling active slots: 64
- coupling updates: `[0, 300]`
- coupling wrong updates: `[0, 175]`
- state: 42.47 MiB, about +4.03 MiB over R186

Flip summary for coupling 1.0 versus R186:

| split | helpful | harmful | net | full acc delta | token acc delta |
|---|---:|---:|---:|---:|---:|
| train_post | 6 | 1 | +5 | +0.0167 | +0.0017 |
| validation | 1 | 1 | 0 | +0.0000 | -0.0100 |
| test | 1 | 1 | 0 | +0.0000 | -0.0050 |

## Interpretation

Answer-slot coupling is a clean CE-positive but exact-neutral mechanism.

The best R193 point is coupling scale 1.0:

- test full CE improves over R186: `1.2818 -> 1.2714`
- validation full CE improves over R186: `1.3200 -> 1.3128`
- test full exact is unchanged: `0.1267 -> 0.1267`
- test full-token accuracy drops: `0.3300 -> 0.3250`

This suggests the coupling feature improves calibration of the second answer
slot but does not change enough winners to improve exact sequence accuracy.
Combining coupling with R190 direct is worse on exact (`0.1200` test), so the
two mechanisms do not currently compose.

## Boundary

Supported claim:

> A local previous-answer coupling trace improves QA19 full-answer CE on medium
> seed0 without BP or raw prompt storage, but it does not improve held-out
> full-answer exact match.

Unsupported claims:

- Coupling solves QA19.
- Coupling and direct scores are complementary in the current form.
- Lower coupling scale fixes the exact-answer issue.

## Next

R194 should instrument component margins for slot 1 and identify whether CE
improvement comes from correct-answer probability increases on already-wrong
examples or from over-smoothing. The next mechanism should likely be a local
winner-selection cleanup for slot 1, not more coupling scale sweeps.
