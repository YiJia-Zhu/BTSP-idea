# R173 Joint Suppression Trace

**Date**: 2026-06-19

## Goal

R172 showed that static top-k cleanup can make the joint branch confidently
wrong. R173 replaces label-free cleanup with a local answer-token error signal:
when the joint direct path prefers a wrong candidate over the target, write an
inhibitory prototype for that wrong candidate under the current full feature.

This tests whether a target/wrong-modulated suppression trace can keep R171's
CE benefit while improving top-1 winner selection.

## Implementation

Extended `babi_unified_token_qa_experiment.py` with:

- `--role-joint-suppress-slots`
- `--role-joint-suppress-lr`
- `--role-joint-suppress-score-scale`
- `--role-joint-suppress-margin`

When `--role-joint-suppress-score-scale > 0`, the model allocates a separate
joint suppression prototype bank. At scoring time, suppression scores are
subtracted from the joint rescue delta before it is added to the branch readout.
At update time, if `base_plus_direct_joint` ranks a non-target above the target
by the margin, the current feature is written into the suppression bank for the
wrong token.

The mechanism is still no-BP, local, default-off, and stores derived numeric
features rather than raw text.

## Runs

Smoke:

- `output/babi_unified_role_transition_r173_qa18_smoke_joint2_suppress1`

Medium seed0, `train-limit=300 eval-limit=300`:

- `output/babi_unified_role_transition_r173_qa14_medium_joint2_suppress05`
- `output/babi_unified_role_transition_r173_qa17_medium_joint2_suppress05`
- `output/babi_unified_role_transition_r173_qa18_medium_joint2_suppress05`
- `output/babi_unified_role_transition_r173_qa14_medium_joint2_suppress1`
- `output/babi_unified_role_transition_r173_qa17_medium_joint2_suppress1`
- `output/babi_unified_role_transition_r173_qa18_medium_joint2_suppress1`
- `output/babi_unified_role_transition_r173_qa14_medium_joint2_suppress1_m02`
- `output/babi_unified_role_transition_r173_qa17_medium_joint2_suppress1_m02`
- `output/babi_unified_role_transition_r173_qa18_medium_joint2_suppress1_m02`

Full seed0, suppress scale 1, margin 0.2:

- `output/babi_unified_role_transition_r173_qa14_full_joint2_suppress1_m02`
- `output/babi_unified_role_transition_r173_qa17_full_joint2_suppress1_m02`
- `output/babi_unified_role_transition_r173_qa18_full_joint2_suppress1_m02`

## Medium Results

Test split:

| Task | Variant | Accuracy | CE |
|---|---|---:|---:|
| QA14 | R167 branch r8 | 0.340 | 1.6808 |
| QA14 | R171 joint2 | 0.353 | 1.6785 |
| QA14 | suppress0.5 | 0.340 | 1.6787 |
| QA14 | suppress1 | 0.337 | 1.6801 |
| QA14 | suppress1 m0.2 | 0.350 | 1.6801 |
| QA17 | R167 branch r8 | 0.527 | 0.7001 |
| QA17 | R171 joint2 | 0.540 | 0.7012 |
| QA17 | suppress0.5 | 0.530 | 0.7001 |
| QA17 | suppress1 | 0.517 | 0.7000 |
| QA17 | suppress1 m0.2 | 0.553 | 0.7011 |
| QA18 | R167 branch r8 | 0.890 | 0.4969 |
| QA18 | R171 joint2 | 0.880 | 0.4824 |
| QA18 | suppress0.5 | 0.880 | 0.4796 |
| QA18 | suppress1 | 0.883 | 0.4782 |
| QA18 | suppress1 m0.2 | 0.883 | 0.4774 |

The medium sweep selected suppress scale 1, margin 0.2 for full testing because
it gave the best QA17 accuracy while improving QA18 CE.

## Full Seed0 Results

Test split:

| Task | Variant | Accuracy | CE | State |
|---|---|---:|---:|---:|
| QA14 | R167 branch r8 | 0.398 | 1.6529 | 12.84 MB |
| QA14 | R171 joint2 | 0.383 | 1.6487 | 25.49 MB |
| QA14 | R173 suppress1 m0.2 | 0.377 | 1.6507 | 28.65 MB |
| QA17 | R167 branch r8 | 0.512 | 0.7076 | 12.84 MB |
| QA17 | R171 joint2 | 0.513 | 0.7007 | 25.49 MB |
| QA17 | R173 suppress1 m0.2 | 0.529 | 0.6986 | 28.65 MB |
| QA18 | R165 microproto | 0.912 | 0.4674 | 4.38 MB |
| QA18 | R171 joint2 | 0.903 | 0.4599 | 25.49 MB |
| QA18 | R173 suppress1 m0.2 | 0.903 | 0.4549 | 28.65 MB |

Suppression activity in full runs:

| Task | Updates | Active suppress slots |
|---|---:|---:|
| QA14 | 741 | 97 |
| QA17 | 457 | 33 |
| QA18 | 144 | 33 |

## Findings

1. Target/wrong-modulated suppression is better than static top-k cleanup for
   QA17. Full QA17 improves from R171 joint2 `0.513/0.7007` to
   `0.529/0.6986`, though it remains well below the older role-concat result
   `0.578/0.6734`.

2. QA18 CE improves again and is the best CE observed in this relation-task
   group so far (`0.4549`), but top-1 remains below the simpler microproto and
   branch baselines.

3. QA14 is hurt by suppression. It likely suppresses direct-role rescue
   candidates that are necessary for temporal-before reasoning.

4. State cost rises to `28.65 MB`, so this path is not yet an efficient default.

## Interpretation

R173 validates the direction of credit-modulated inhibitory cleanup: it can
reduce confident wrong joint evidence and improve QA17/QA18 CE. However, a
single suppression policy is still too broad. QA14 needs disinhibition for
direct-role transitions, while QA18 benefits from conservative suppression and
base protection.

## Next Step

Do not seed-repeat R173 as-is. The next mechanism should make suppression
conditional on local evidence type:

- apply suppression to joint-only conflicts;
- avoid suppressing direct-role rescue candidates;
- add a base-protection/disinhibition signal so QA18-style comparison cues can
  dominate without damaging QA14 temporal transitions.

R173 is **DONE-MIXED**: positive for QA17 and QA18 CE, not an accuracy-leading
method.
