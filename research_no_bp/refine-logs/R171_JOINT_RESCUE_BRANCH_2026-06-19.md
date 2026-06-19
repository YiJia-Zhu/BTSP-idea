# R171 Joint Rescue Branch

**Date**: 2026-06-19

## Goal

R169/R170 showed that simple arbitration could not reconcile the relation-task
tradeoff. QA17's best result still came from the older role-concat readout,
suggesting that some useful shared base+role feature interactions were lost when
R167 separated the readout into base and role banks.

R171 adds a third, parallel no-BP readout path: a full base+role joint prototype
bank on top of the branch-separated base and role banks. This tests whether a
separate joint rescue circuit can recover shared-feature evidence while keeping
the branch-separated structure available.

## Implementation

Extended `babi_unified_token_qa_experiment.py` with:

- `--role-joint-rescue-readout`
- `--role-joint-rescue-score-scale`

When enabled with `--role-branch-readout`, the model stores:

- base-state prototypes;
- role-state prototypes;
- joint full-feature prototypes.

All three banks are updated by the same local answer-token target/wrong signal.
The method is still random/local initialization only, no BP, no pretrained
backbone, no task-specific QA head, no parser, and no raw replay.

## Runs

Smoke:

- `output/babi_unified_role_transition_r171_qa17_smoke_joint2`

Medium seed0, `train-limit=300 eval-limit=300`:

- `output/babi_unified_role_transition_r171_qa14_medium_joint1`
- `output/babi_unified_role_transition_r171_qa17_medium_joint1`
- `output/babi_unified_role_transition_r171_qa18_medium_joint1`
- `output/babi_unified_role_transition_r171_qa14_medium_joint2`
- `output/babi_unified_role_transition_r171_qa17_medium_joint2`
- `output/babi_unified_role_transition_r171_qa18_medium_joint2`
- `output/babi_unified_role_transition_r171_qa14_medium_joint4`

Full seed0, joint scale 2:

- `output/babi_unified_role_transition_r171_qa14_full_joint2`
- `output/babi_unified_role_transition_r171_qa17_full_joint2`
- `output/babi_unified_role_transition_r171_qa18_full_joint2`

## Medium Results

Test split:

| Task | Variant | Accuracy | CE |
|---|---|---:|---:|
| QA14 | R167 branch r8 | 0.340 | 1.6808 |
| QA14 | joint1 | 0.343 | 1.6795 |
| QA14 | joint2 | 0.353 | 1.6785 |
| QA14 | joint4 | 0.343 | 1.6769 |
| QA17 | R167 branch r8 | 0.527 | 0.7001 |
| QA17 | joint1 | 0.533 | 0.7005 |
| QA17 | joint2 | 0.540 | 0.7012 |
| QA18 | R167 branch r8 | 0.890 | 0.4969 |
| QA18 | joint1 | 0.880 | 0.4895 |
| QA18 | joint2 | 0.880 | 0.4824 |

Medium suggests that joint rescue helps QA14/QA17 top-1 but trades QA18
top-1 for better CE.

## Full Seed0 Results

Test split:

| Task | Variant | Accuracy | CE | State |
|---|---|---:|---:|---:|
| QA14 | R167 branch r8 | 0.398 | 1.6529 | 12.84 MB |
| QA14 | R171 joint2 | 0.383 | 1.6487 | 25.49 MB |
| QA17 | R167 branch r8 | 0.512 | 0.7076 | 12.84 MB |
| QA17 | R165 role concat | 0.578 | 0.6734 | 12.78 MB |
| QA17 | R171 joint2 | 0.513 | 0.7007 | 25.49 MB |
| QA18 | R165 microproto | 0.912 | 0.4674 | 4.38 MB |
| QA18 | R168 branch r4 | 0.911 | 0.4759 | 12.84 MB |
| QA18 | R171 joint2 | 0.903 | 0.4599 | 25.49 MB |

Validation split had positive signals:

- QA14 validation accuracy improved to `0.380`;
- QA17 validation accuracy improved to `0.531`;
- QA18 validation CE improved to `0.4979`.

## Findings

1. The joint rescue path improves probability calibration but not reliable
   top-1 selection on full data. QA14 and QA18 get better CE than their
   branch-separated references, but their test accuracy drops.

2. QA17 still does not recover the role-concat result. Joint2 improves QA17 CE
   over branch r8 (`0.7076 -> 0.7007`) but remains far below role-concat
   (`0.6734` CE, `0.578` accuracy).

3. The cost is substantial. State roughly doubles from `12.84 MB` to
   `25.49 MB`, because a third prototype bank is added.

## Interpretation

R171 supports the hypothesis that shared base+role evidence is useful, but
adding an ungated joint pathway mainly moves probability mass rather than
selecting the right winner. This is consistent with the broader trajectory:
local feature circuits can create evidence, but the bottleneck is now local
candidate cleanup/inhibition.

## Next Step

Do not expand joint rescue directly to seed repeats yet. The next mechanism
should add local WTA/candidate cleanup on the joint path, for example:

- keep only top-k joint candidate outputs before adding them to branch scores;
- inhibit joint candidates that conflict with a high-margin base branch;
- use direct-role evidence as a disinhibitory signal for QA14-like temporal
  transitions.

R171 is **DONE-MIXED**: useful CE evidence, not a new accuracy-leading method.
