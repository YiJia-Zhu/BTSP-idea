# R186: Edge-Path Soft Mixture

Date: 2026-06-19

## Purpose

R185 showed that hard WTA over a single intermediate edge-path candidate is not
the right QA19 default. R186 tests the next hypothesis: keep several candidate
intermediate path traces active and combine them with local soft weights.

This is still a pure no-BP mechanism. It uses compact token IDs, local edge
events, role-transition traces, answer-slot prototypes, and local modulatory
updates. It does not use a parser, symbolic graph labels, raw prompt replay,
pretrained weights, or BP.

## Implementation

Changed `babi_unified_token_qa_experiment.py`.

Added `--answer-slot-feature-mode edge_path_soft`.

New soft mixture controls:

- `--edge-path-soft-top-k`
- `--edge-path-soft-temperature`
- `--edge-path-soft-consistency-scale`
- `--edge-path-soft-learned-scale`

The feature reuses R185 candidate generation, then weights the top local
candidate path states by:

```text
score = support + consistency_scale * closure + learned_scale * cleanup_score
weight = support * exp((score - max_score) / temperature)
```

The default R186 probes set `learned_scale=0.0` because R185 found learned
cleanup scores harmful in this feature space.

## Runs

Smoke:

- `output/babi_unified_qa19_r186_edge_path_soft_aslot_smoke`

Medium, same QA19 setting as R184/R185:

- `output/babi_unified_qa19_r186_edge_path_soft_aslot_medium_s2`
- `output/babi_unified_qa19_r186_edge_path_soft_support_aslot_medium_s2`
- `output/babi_unified_qa19_r186_edge_path_soft_temp05_aslot_medium_s2`
- `output/babi_unified_qa19_r186_edge_path_soft_support_temp1_aslot_medium_s2`
- comparison: `output/babi_unified_qa19_r186_edge_path_soft_comparison`

All medium runs use `train-limit=300`, `eval-limit=300`, `state_dim=64`,
`state_order=224`, R174-style role-transition, answer-slot readout, and full
multi-token QA19 evaluation.

## Results

`output/babi_unified_qa19_r186_edge_path_soft_comparison/comparison_summary.csv`:

| variant | split | first acc | full acc | full-token acc | full CE | state MB |
|---|---|---:|---:|---:|---:|---:|
| R184 edge average | validation | 0.2700 | 0.1100 | 0.2900 | 1.3179 | 35.39 |
| R184 edge average | test | 0.2867 | 0.1200 | 0.3300 | 1.2802 | 35.39 |
| R185 learned WTA | validation | 0.2600 | 0.0900 | 0.2950 | 1.3423 | 36.40 |
| R185 learned WTA | test | 0.2900 | 0.1133 | 0.3283 | 1.2876 | 36.40 |
| R185 support WTA | validation | 0.3100 | 0.1300 | 0.3000 | 1.3268 | 36.40 |
| R185 support WTA | test | 0.2967 | 0.1033 | 0.3267 | 1.2914 | 36.40 |
| R186 soft t0.20 c0.50 | validation | 0.3100 | 0.1400 | 0.3250 | 1.3233 | 36.40 |
| R186 soft t0.20 c0.50 | test | 0.2833 | 0.1267 | 0.3267 | 1.2841 | 36.40 |
| R186 soft t0.20 c0.00 | validation | 0.2900 | 0.1300 | 0.3100 | 1.3200 | 36.40 |
| R186 soft t0.20 c0.00 | test | 0.2967 | 0.1267 | 0.3300 | 1.2818 | 36.40 |
| R186 soft t0.50 c0.50 | validation | 0.2700 | 0.1200 | 0.2950 | 1.3217 | 36.40 |
| R186 soft t0.50 c0.50 | test | 0.2967 | 0.1233 | 0.3350 | 1.2821 | 36.40 |
| R186 soft t1.00 c0.00 | validation | 0.2800 | 0.1200 | 0.3000 | 1.3186 | 36.40 |
| R186 soft t1.00 c0.00 | test | 0.2833 | 0.1100 | 0.3250 | 1.2816 | 36.40 |

## Interpretation

R186 is weak-positive relative to R184 for full-answer exact accuracy, but not
for CE.

Best exact-match point:

- `t0.20 c0.50`: validation exact `0.1100 -> 0.1400`, test exact
  `0.1200 -> 0.1267`.

More stable exact/CE point:

- `t0.20 c0.00`: validation exact `0.1100 -> 0.1300`, test exact
  `0.1200 -> 0.1267`, with test CE only `1.2802 -> 1.2818`.

Higher temperature softens the mixture and moves CE closer to R184, but loses
the exact-match gain:

- `t1.00 c0.00`: test exact `0.1100`, below R184.

This supports the R185 boundary: hard WTA is too brittle, but soft multi-candidate
edge routing is more promising. The improvement is still small and only one
seed, so it is not a robust QA19 solution.

## Boundary

R186 does not prove robust path reasoning. QA19 remains mostly unsolved, and the
model is still near the majority baseline region:

- R181 majority baselines: validation `0.1300`, test `0.1100`.
- R186 best test exact `0.1267`, above majority but still low.

The supported claim is narrow: soft multi-candidate edge-path features improve
QA19 exact full-answer accuracy over R184/R185 in the medium seed0 setting,
while CE remains roughly neutral to slightly worse.

## Next

1. Repeat the best two R186 settings across seeds 1/2 before treating it as a
   stable mechanism.
2. Add validation-selected temperature/consistency selection instead of choosing
   from test-side behavior.
3. Test whether soft edge mixture transfers to larger QA19 train/eval limits.
4. Consider a local consistency channel that affects answer-slot scores directly
   instead of only changing the feature vector.
