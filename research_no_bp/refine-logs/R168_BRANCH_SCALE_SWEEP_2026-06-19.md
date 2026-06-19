# R168 Branch Score Scale Sweep

**Date**: 2026-06-19

## Goal

R167 introduced branch-separated readout. It protected QA18 and improved QA14,
but QA17 preferred the older concatenated role readout. R168 tests whether a
single branch score scale can recover QA17 while preserving QA14/QA18.

This is a parameter sweep, not a new mechanism. It keeps the pure no-BP
constraints: local prototype updates, no BP, no pretrained backbone, no parser,
no answer head, and no raw replay.

## Setup

Fixed settings:

- branch readout enabled;
- base branch score scale `8.0`;
- role branch score scale swept over `4`, `8`, `12`, `16`;
- direct role score scale remains `1.5`;
- role event cache size `4096`;
- seed 0.

Medium sweep uses `train_limit=300`, `eval_limit=300`.

Full validation checks use full local splits for `r=4`, compared to existing
R167 `r=8`, R165 concat, and microproto.

Outputs:

- `output/babi_unified_role_transition_r168_branch_scale_sweep/sweep_summary.csv`
- `output/babi_unified_role_transition_r168_branch_scale_full/comparison_summary.csv`

## Medium Sweep

| Task | Role scale | Val acc | Val CE | Test acc | Test CE |
|---|---:|---:|---:|---:|---:|
| QA14 | 4 | 0.290 | 1.7369 | 0.360 | 1.6771 |
| QA14 | 8 | 0.240 | 1.7416 | 0.340 | 1.6808 |
| QA14 | 12 | 0.240 | 1.7405 | 0.330 | 1.6895 |
| QA14 | 16 | 0.220 | 1.7528 | 0.297 | 1.7049 |
| QA17 | 4 | 0.458 | 0.7265 | 0.533 | 0.6994 |
| QA17 | 8 | 0.448 | 0.7407 | 0.527 | 0.7001 |
| QA17 | 12 | 0.427 | 0.7565 | 0.540 | 0.7029 |
| QA17 | 16 | 0.417 | 0.7739 | 0.523 | 0.7071 |
| QA18 | 4 | 0.821 | 0.5195 | 0.893 | 0.4943 |
| QA18 | 8 | 0.821 | 0.5191 | 0.890 | 0.4969 |
| QA18 | 12 | 0.821 | 0.5187 | 0.853 | 0.5001 |
| QA18 | 16 | 0.842 | 0.5184 | 0.847 | 0.5036 |

Medium validation/test averages favored role scale `4`, so it was expanded to
full seed0.

## Full Seed0 Check

| Task | Variant | Val acc | Val CE | Test acc | Test CE |
|---|---|---:|---:|---:|---:|
| QA14 | microproto | 0.130 | 1.7609 | 0.211 | 1.7269 |
| QA14 | role concat | 0.300 | 1.7246 | 0.351 | 1.7037 |
| QA14 | branch r8 | 0.370 | 1.6795 | 0.398 | 1.6529 |
| QA14 | branch r4 | 0.380 | 1.6760 | 0.385 | 1.6560 |
| QA17 | microproto | 0.510 | 0.6979 | 0.482 | 0.6994 |
| QA17 | role concat | 0.573 | 0.6788 | 0.578 | 0.6734 |
| QA17 | branch r8 | 0.583 | 0.7208 | 0.512 | 0.7076 |
| QA17 | branch r4 | 0.573 | 0.7014 | 0.508 | 0.7075 |
| QA18 | microproto | 0.811 | 0.5105 | 0.912 | 0.4674 |
| QA18 | role concat | 0.821 | 0.6025 | 0.834 | 0.6023 |
| QA18 | branch r8 | 0.853 | 0.5131 | 0.909 | 0.4764 |
| QA18 | branch r4 | 0.842 | 0.5150 | 0.911 | 0.4759 |

## Findings

1. Medium sweep suggested role scale `4` as the best broad setting, but full
   seed0 did not turn it into a clear win over R167 `r=8`.

2. `r=4` slightly improves QA18 versus `r=8` (`0.909/0.4764 -> 0.911/0.4759`)
   but slightly hurts QA14 (`0.398/1.6529 -> 0.385/1.6560`).

3. QA17 remains unresolved. `r=4` improves validation CE versus `r=8`, but full
   test accuracy remains near `0.51`, far below role-concat `0.578`.

4. A single static role-branch scale is not enough. The branch readout itself is
   useful, but different tasks prefer different merge behavior.

## Interpretation

R168 narrows the next step. Branch separation was the right structural fix for
QA18, but static branch weighting cannot recover QA17. The model needs
example-level or task-family-level arbitration between concatenated-style role
drive and separated-branch protection.

The next useful mechanism is a local branch arbiter trained from train/validation
signals, not another fixed scale sweep.

## Next Steps

1. Add branch agreement/margin diagnostics for full QA14/17/18.
2. Use train-split flip labels to train a local branch arbiter.
3. Validate the arbiter on QA14/17/18 before seed repeats.
