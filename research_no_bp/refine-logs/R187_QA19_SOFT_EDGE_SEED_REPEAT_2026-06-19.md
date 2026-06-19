# R187: QA19 Soft Edge Seed Repeat

Date: 2026-06-19

## Purpose

R186 found a weak-positive seed0 signal for `edge_path_soft`, especially the
`temperature=0.20`, `consistency=0.00` setting. R187 repeats that setting on
seeds 1/2 and compares against paired R184 `edge_path` baselines.

The goal is to check whether the R186 exact-match improvement is a seed0
artifact before treating soft multi-candidate edge routing as a useful QA19
mechanism.

## Compared Methods

Both methods use the same R174-style role-transition backbone, answer-slot
readout, full multi-token QA19 evaluation, and no raw prompt storage.

- R184 baseline: `--answer-slot-feature-mode edge_path`
- R186 candidate: `--answer-slot-feature-mode edge_path_soft`
  - `--edge-path-soft-top-k 6`
  - `--edge-path-soft-temperature 0.20`
  - `--edge-path-soft-consistency-scale 0.00`
  - `--edge-path-soft-learned-scale 0.00`

## Runs

Seed0 reused existing R184/R186 medium outputs.

New R187 runs:

- `output/babi_unified_qa19_r187_r184_edge_path_aslot_medium_s1`
- `output/babi_unified_qa19_r187_r184_edge_path_aslot_medium_s2`
- `output/babi_unified_qa19_r187_r186_soft_t020_c000_aslot_medium_s1`
- `output/babi_unified_qa19_r187_r186_soft_t020_c000_aslot_medium_s2`

Comparison:

- `output/babi_unified_qa19_r187_soft_seed_repeat_comparison/paired_summary.csv`
- `output/babi_unified_qa19_r187_soft_seed_repeat_comparison/aggregate_summary.csv`
- `output/babi_unified_qa19_r187_soft_seed_repeat_comparison/slot_feature_summary.csv`

## Results

Three-seed aggregate:

| variant | split | full acc mean | full acc std | full-token acc mean | full CE mean |
|---|---|---:|---:|---:|---:|
| R184 edge | validation | 0.1133 | 0.0058 | 0.2933 | 1.3253 |
| R186 soft | validation | 0.1133 | 0.0153 | 0.2950 | 1.3258 |
| delta | validation | 0.0000 | 0.0173 | 0.0017 | 0.0006 |
| R184 edge | test | 0.1222 | 0.0069 | 0.3300 | 1.2965 |
| R186 soft | test | 0.1267 | 0.0067 | 0.3294 | 1.2949 |
| delta | test | 0.0044 | 0.0019 | -0.0006 | -0.0015 |

Paired test deltas:

| seed | full acc delta | full CE delta | full-token acc delta |
|---:|---:|---:|---:|
| 0 | +0.0067 | +0.0016 | +0.0000 |
| 1 | +0.0033 | -0.0070 | -0.0050 |
| 2 | +0.0033 | +0.0009 | +0.0033 |

## Interpretation

R187 gives a narrow, weak-positive signal for soft edge mixture on QA19 test:

- Full-answer exact improves on all three seeds.
- Mean test full-answer exact improves `0.1222 -> 0.1267`.
- Mean test full CE improves slightly `1.2965 -> 1.2949`.

However, validation does not support the same claim:

- Validation full-answer exact mean is unchanged at `0.1133`.
- Validation CE is slightly worse by `+0.0006`.

This means the mechanism is promising but not yet selection-stable. The result
supports continuing soft multi-candidate routing, but it does not justify
claiming solved QA19 or adopting this setting as a final default.

## Boundary

QA19 remains near majority-baseline territory. R181 majority baselines were
validation `0.1300` and test `0.1100`; R187 test improves above that baseline,
but validation remains below it.

Supported claim:

> In the medium QA19 setting, parser-free soft multi-candidate edge routing
> gives a small but seed-consistent test exact-match improvement over the R184
> edge-path average, without storing raw text or using BP.

Unsupported claims:

- Robust path reasoning.
- General validation-selected improvement.
- GPT/API-quality QA behavior.

## Next

1. Add validation-based selection over soft temperature/consistency, then report
   the selected test result.
2. Test the selected soft setting on larger QA19 train/eval limits.
3. Add a direct local consistency score channel to the answer-slot readout
   instead of only changing the feature vector.
