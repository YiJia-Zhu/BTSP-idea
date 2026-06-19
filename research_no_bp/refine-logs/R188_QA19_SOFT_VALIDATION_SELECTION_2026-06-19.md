# R188: QA19 Soft Edge Validation Selection

Date: 2026-06-19

## Purpose

R187 showed that a fixed R186 soft-edge setting, `temperature=0.20` and
`consistency=0.00`, gives a small three-seed test improvement over R184. R188
tests whether the soft temperature/consistency setting can be selected from the
validation split rather than inferred from test behavior.

This is an evaluation-selection integrity check. It does not add a new learning
mechanism.

## Candidate Set

All candidates use `--answer-slot-feature-mode edge_path_soft` with no learned
cleanup score:

| candidate | temperature | consistency |
|---|---:|---:|
| `soft_t020_c000` | 0.20 | 0.00 |
| `soft_t020_c050` | 0.20 | 0.50 |
| `soft_t050_c050` | 0.50 | 0.50 |
| `soft_t100_c000` | 1.00 | 0.00 |

Seed0 reuses R186 outputs. R188 adds missing seeds 1/2 for the other candidates.

New runs:

- `output/babi_unified_qa19_r188_soft_t020_c050_aslot_medium_s1`
- `output/babi_unified_qa19_r188_soft_t020_c050_aslot_medium_s2`
- `output/babi_unified_qa19_r188_soft_t050_c050_aslot_medium_s1`
- `output/babi_unified_qa19_r188_soft_t050_c050_aslot_medium_s2`
- `output/babi_unified_qa19_r188_soft_t100_c000_aslot_medium_s1`
- `output/babi_unified_qa19_r188_soft_t100_c000_aslot_medium_s2`

Comparison outputs:

- `output/babi_unified_qa19_r188_validation_selected_soft_comparison/candidate_summary.csv`
- `output/babi_unified_qa19_r188_validation_selected_soft_comparison/baseline_summary.csv`
- `output/babi_unified_qa19_r188_validation_selected_soft_comparison/selection_summary.csv`
- `output/babi_unified_qa19_r188_validation_selected_soft_comparison/aggregate_summary.csv`

## Selection Policies

Two deterministic validation-only policies were tested:

1. `val_exact_then_ce`: choose highest validation full-answer exact; tie-break
   by lower validation full CE.
2. `val_ce_then_exact`: choose lowest validation full CE; tie-break by higher
   validation full-answer exact.

Both policies choose per seed using validation only, then report test.

## Results

Three-seed aggregate:

| group | split | selected candidates | full acc mean | full CE mean | delta full acc vs R184 | delta full CE vs R184 |
|---|---|---|---:|---:|---:|---:|
| R184 edge | validation | - | 0.1133 | 1.3253 | - | - |
| R184 edge | test | - | 0.1222 | 1.2965 | - | - |
| val exact then CE | validation | s0=t020c050; s1=t050c050; s2=t020c050 | 0.1300 | 1.3278 | +0.0167 | +0.0025 |
| val exact then CE | test | s0=t020c050; s1=t050c050; s2=t020c050 | 0.1178 | 1.2980 | -0.0044 | +0.0016 |
| val CE then exact | validation | s0=t100c000; s1=t020c000; s2=t020c000 | 0.1100 | 1.3254 | -0.0033 | +0.0001 |
| val CE then exact | test | s0=t100c000; s1=t020c000; s2=t020c000 | 0.1211 | 1.2949 | -0.0011 | -0.0016 |

Per-seed selected test deltas:

| policy | seed | selected | full acc delta | full CE delta |
|---|---:|---|---:|---:|
| val exact then CE | 0 | t020c050 | +0.0067 | +0.0039 |
| val exact then CE | 1 | t050c050 | -0.0067 | -0.0017 |
| val exact then CE | 2 | t020c050 | -0.0133 | +0.0025 |
| val CE then exact | 0 | t100c000 | -0.0100 | +0.0014 |
| val CE then exact | 1 | t020c000 | +0.0033 | -0.0070 |
| val CE then exact | 2 | t020c000 | +0.0033 | +0.0009 |

## Interpretation

R188 is a boundary/negative result for naive validation selection.

Validation exact selection overfits the 100-example validation split:

- It improves validation exact by `+0.0167`.
- But test exact falls by `-0.0044` and test CE worsens by `+0.0016`.

Validation CE selection is safer for CE:

- Test CE improves by `-0.0016`.
- But test exact falls by `-0.0011`, and it does not preserve R187's fixed-setting
  exact gain.

This means the R187 fixed-setting gain cannot yet be converted into a clean
validation-selected protocol. The current validation split is too small/noisy
for selecting among these close soft-routing settings.

## Boundary

Supported claim:

> The R186/R187 soft-edge mechanism remains promising, but simple validation
> selection over temperature/consistency is not reliable on the current
> 100-example QA19 validation split.

Unsupported claim:

- Validation-selected soft edge routing improves QA19 full-answer accuracy.
- The soft setting can be chosen robustly without a stronger calibration split
  or larger validation set.

## Next

1. Use larger validation/eval limits or cross-validation before selecting soft
   routing temperature.
2. Test fixed `t0.20 c0.00` on larger train/eval limits before further tuning.
3. Add a local consistency score channel to answer-slot scores, then validate
   whether it reduces selection sensitivity.
