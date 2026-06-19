# R190: QA19 Direct Edge-Path Score Boundary

Date: 2026-06-19

## Purpose

R189 showed that changing the answer-slot feature vector to a soft edge-path
mixture did not scale on full-limit QA19. R190 tests a more direct local
mechanism: use the same parser-free soft path mixture as a local prototype key
that adds a direct answer-token score delta.

This is still pure no-BP:

- no pretrained model or API backbone
- no raw prompt storage
- no statistical n-gram table
- local target/wrong prototype updates only

## Implementation

Added `edge_path_soft_direct` to `babi_unified_token_qa_experiment.py`.

Mechanism:

1. `edge_path_soft_state` computes the soft mixture over local edge-path
   candidates and caches the path-feature mixture per answer slot.
2. A slot-specific direct prototype bank learns target answer tokens from that
   cached path feature.
3. During answer-token scoring, `AnswerSlotReadoutMemory` adds the direct
   path-feature token score delta on top of the ordinary answer-slot readout.

New args:

- `--answer-slot-feature-mode edge_path_soft_direct`
- `--edge-path-direct-answer-slots`
- `--edge-path-direct-slots`
- `--edge-path-direct-lr`
- `--edge-path-direct-wrong-lr`
- `--edge-path-direct-score-scale`

## Runs

Smoke:

- `output/babi_unified_qa19_r190_edge_path_soft_direct_smoke`

Medium seed0 scale sweep:

- `output/babi_unified_qa19_r190_edge_path_soft_direct_medium_s0_scale05`
- `output/babi_unified_qa19_r190_edge_path_soft_direct_medium_s0`
- `output/babi_unified_qa19_r190_edge_path_soft_direct_medium_s0_scale20`
- comparison: `output/babi_unified_qa19_r190_edge_path_soft_direct_comparison`

Common setup:

- bAbI `en-qa19`
- train/eval: 300/300, validation limit 100 from the split file
- seed: 0
- R174-style role-transition backbone
- answer-slot readout
- full multi-token answer evaluation

## Results

`output/babi_unified_qa19_r190_edge_path_soft_direct_comparison/comparison_summary.csv`:

| variant | split | direct scale | first acc | full acc | full-token acc | full CE |
|---|---|---:|---:|---:|---:|---:|
| R184 edge | validation | - | 0.2700 | 0.1100 | 0.2900 | 1.3179 |
| R184 edge | test | - | 0.2867 | 0.1200 | 0.3300 | 1.2802 |
| R186 soft t0.20 c0.00 | validation | - | 0.2900 | 0.1300 | 0.3100 | 1.3200 |
| R186 soft t0.20 c0.00 | test | - | 0.2967 | 0.1267 | 0.3300 | 1.2818 |
| R190 direct | validation | 0.5 | 0.2700 | 0.1300 | 0.3100 | 1.3204 |
| R190 direct | test | 0.5 | 0.2867 | 0.1167 | 0.3300 | 1.2791 |
| R190 direct | validation | 1.0 | 0.2500 | 0.1400 | 0.3150 | 1.3251 |
| R190 direct | test | 1.0 | 0.3067 | 0.1233 | 0.3350 | 1.2813 |
| R190 direct | validation | 2.0 | 0.1900 | 0.0800 | 0.2600 | 1.3510 |
| R190 direct | test | 2.0 | 0.2933 | 0.0933 | 0.3233 | 1.2986 |

Direct-channel stats for scale 1.0:

- direct active slots: 128
- direct target updates: `[300, 300]`
- direct wrong updates: `[225, 181]`
- direct score checks: `[1998, 1998]`
- state: 38.43 MiB

## Interpretation

The direct score channel is active and changes behavior, but it is not a robust
QA19 improvement.

Observed effects:

- `scale=0.5` gives the best test CE among the R190 settings (`1.2791`), slightly
  better than R184 and R186, but test full-answer exact drops to `0.1167`.
- `scale=1.0` gives the best R190 exact (`0.1233`) and improves first-token and
  full-token accuracy over R186, but still loses full-answer exact to R186
  (`0.1267`) and worsens CE relative to R184.
- `scale=2.0` is clearly harmful: validation full exact drops to `0.0800` and
  test full exact to `0.0933`.

This means the soft path mixture contains usable local evidence, but directly
adding a generic token score is too coarse. It can move probability mass, yet it
does not reliably assemble both QA19 answer directions.

## Boundary

Supported claim:

> A local direct path-feature prototype bank can influence QA19 answer-token
> scores without BP or raw prompt storage, and weakly improves either CE or
> token-level accuracy depending on scale.

Unsupported claims:

- R190 solves QA19.
- Direct score injection improves full-answer exact over R186.
- Stronger direct path scores are better.

## Next

R191 should not keep scaling the same direct score. Better candidates:

1. Add counterfactual path competition: train local prototypes to distinguish
   target path features from near-miss path features that share one endpoint.
2. Add answer-slot coupling: the second direction token should condition on the
   first predicted/teacher-forced direction through a local eligibility trace,
   not through an independent slot bank.
3. Measure direct-channel helpful/harmful flips against R186 on the same
   examples before adding more capacity.
