# R167 Branch-Separated Readout

**Date**: 2026-06-19

## Goal

R166 showed that final-logit role-score gates are too late: QA18 interference
remains even when direct role logits are gated or removed. R167 tests the next
mechanism: keep base-state and role-state prototype readouts separate before
their scores are merged.

This is still pure no-BP. The method uses local prototype updates only, no BP,
no pretrained backbone, no parser, no symbolic state, no answer head, and no
raw replay.

## Implementation

Added optional branch-separated readout to
`babi_unified_token_qa_experiment.py`:

- `--role-branch-readout`
- `--role-branch-base-score-scale`
- `--role-branch-role-score-scale`

Default behavior is unchanged. When enabled, the model stores:

- base recurrent-state prototypes;
- role-transition-state prototypes;
- separate branch scores that are added before the direct role score.

The old concatenated prototype bank is not used in this mode.

## Runs

Compatibility smoke:

- `output/babi_unified_role_transition_r167_compat_smoke_default`
- `output/babi_unified_role_transition_r167_compat_smoke_branch`

Medium branch scan:

- `output/babi_unified_role_transition_r167_branch_readout_medium/comparison_summary.csv`

Full seed0 branch scan:

- `output/babi_unified_role_transition_r167_branch_readout_full/comparison_summary.csv`

## Medium Results

Medium uses `train_limit=300`, `eval_limit=300`, seed 0.

| Task | Split | No-branch acc | Branch acc | Acc delta | No-branch CE | Branch CE | CE delta |
|---|---|---:|---:|---:|---:|---:|---:|
| QA14 | test | 0.330 | 0.340 | +0.010 | 1.7123 | 1.6808 | -0.0315 |
| QA17 | test | 0.480 | 0.527 | +0.047 | 0.7002 | 0.7001 | -0.0001 |
| QA18 | test | 0.790 | 0.890 | +0.100 | 0.6172 | 0.4969 | -0.1203 |

This was strong enough to justify full seed0 runs.

## Full Seed0 Results

| Task | Variant | Val acc | Val CE | Test acc | Test CE | State bytes |
|---|---|---:|---:|---:|---:|---:|
| QA14 | microproto | 0.130 | 1.7609 | 0.211 | 1.7269 | 4,384,768 |
| QA14 | role concat | 0.300 | 1.7246 | 0.351 | 1.7037 | 12,776,192 |
| QA14 | role branch | 0.370 | 1.6795 | 0.398 | 1.6529 | 12,841,728 |
| QA17 | microproto | 0.510 | 0.6979 | 0.482 | 0.6994 | 4,384,768 |
| QA17 | role concat | 0.573 | 0.6788 | 0.578 | 0.6734 | 12,776,192 |
| QA17 | role branch | 0.583 | 0.7208 | 0.512 | 0.7076 | 12,841,728 |
| QA18 | microproto | 0.811 | 0.5105 | 0.912 | 0.4674 | 4,384,768 |
| QA18 | role concat | 0.821 | 0.6025 | 0.834 | 0.6023 | 12,776,192 |
| QA18 | role branch | 0.853 | 0.5131 | 0.909 | 0.4764 | 12,841,728 |

Branch readout versus concatenated role readout:

- QA14: test acc `+0.047`, CE `-0.0508`;
- QA17: test acc `-0.066`, CE `+0.0342`;
- QA18: test acc `+0.075`, CE `-0.1259`.

Branch readout versus microproto:

- QA14: test acc `+0.187`, CE `-0.0740`;
- QA17: test acc `+0.030`, CE `+0.0081`;
- QA18: test acc `-0.003`, CE `+0.0090`.

## Findings

1. Branch separation fixes most of the QA18 damage from R165. QA18 recovers
   from role-concat `0.834/0.6023` to `0.909/0.4764`, nearly matching the
   microproto baseline `0.912/0.4674`.

2. Branch separation improves QA14 beyond R165: `0.351/1.7037` becomes
   `0.398/1.6529`.

3. QA17 is the tradeoff. It remains slightly above microproto in accuracy, but
   loses most of the R165 role-concat gain and worsens CE.

4. The state cost barely changes relative to role-concat: `12.78MB -> 12.84MB`
   in these settings, because branch prototypes replace the concatenated bank
   rather than duplicating it.

## Interpretation

R167 confirms the R166 diagnosis: the harmful interference is inside the shared
feature/prototype readout, not merely the final direct role-score vector.
Separating base and role branches is a real mechanism improvement: it preserves
QA18's strong local comparison behavior while improving QA14.

The remaining issue is dynamic arbitration between the separated branches. QA17
prefers the old concatenated role readout, while QA18 prefers branch separation.
The next step should learn or adapt a local branch arbitration signal rather
than choosing one fixed merge rule for every task.

## Next Steps

1. Add branch-level local arbitration over base-score margin, role-score margin,
   and branch agreement.
2. Rerun QA14/17/18 full seed0 with branch arbitration.
3. Only after a single operating point improves or protects all three tasks,
   repeat seeds 1/2.
