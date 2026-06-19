# R176 Base-Protection Rescue Controller

**Date**: 2026-06-19

## Goal

R175 showed that a generic local WTA arbiter over six branch paths is too noisy.
R176 tests a more structured two-stage local controller:

1. protect `base_only` when the base branch has high margin;
2. otherwise, or when rescue evidence is strong, use the R174
   `base_plus_direct_joint` path.

This keeps the controller local and no-BP. It uses branch margins and derived
role/joint scores only, without raw replay or task-specific labels.

## Implementation

Extended `babi_unified_token_qa_experiment.py` with:

- `--role-branch-arbiter base_margin_rescue`
- `--role-branch-arbiter-rescue-role-threshold`
- `--role-branch-arbiter-rescue-joint-threshold`

The mode chooses `base_only` if:

- `base_only` top-2 margin is at least `--role-branch-arbiter-base-margin`;
- max direct-role evidence is below the role rescue threshold;
- max joint-rescue evidence is below the joint rescue threshold.

Otherwise it uses `--role-branch-arbiter-default`, which was set to
`base_plus_direct_joint` in the R176 probes.

## Runs

Adaptive base protection with joint default:

- `output/babi_unified_role_transition_r176_qa14_medium_base_protect_joint_default`
- `output/babi_unified_role_transition_r176_qa17_medium_base_protect_joint_default`
- `output/babi_unified_role_transition_r176_qa18_medium_base_protect_joint_default`

Fixed base-margin rescue with low role threshold `0.05`:

- `output/babi_unified_role_transition_r176_qa14_medium_base_margin_rescue`
- `output/babi_unified_role_transition_r176_qa17_medium_base_margin_rescue`
- `output/babi_unified_role_transition_r176_qa18_medium_base_margin_rescue`

Fixed base-margin rescue with role threshold `0.5`:

- `output/babi_unified_role_transition_r176_qa14_medium_base_margin_rescue_rr05`
- `output/babi_unified_role_transition_r176_qa17_medium_base_margin_rescue_rr05`
- `output/babi_unified_role_transition_r176_qa18_medium_base_margin_rescue_rr05`

## Medium Results

Test split:

| Task | Variant | Accuracy | CE | Base choices | Rescue choices |
|---|---|---:|---:|---:|---:|
| QA14 | R174 protect-direct | 0.370 | 1.6726 | - | - |
| QA14 | adaptive base-protect | 0.327 | 1.6953 | 456 | 544 |
| QA14 | fixed rescue rr0.05 | 0.370 | 1.6726 | 0 | 1000 |
| QA14 | fixed rescue rr0.5 | 0.343 | 1.6889 | 325 | 675 |
| QA17 | R174 protect-direct | 0.553 | 0.7011 | - | - |
| QA17 | adaptive base-protect | 0.550 | 0.6946 | 467 | 529 |
| QA17 | fixed rescue rr0.05 | 0.553 | 0.7011 | 0 | 996 |
| QA17 | fixed rescue rr0.5 | 0.550 | 0.6991 | 549 | 447 |
| QA18 | R174 protect-direct | 0.883 | 0.4774 | - | - |
| QA18 | adaptive base-protect | 0.883 | 0.4835 | 333 | 662 |
| QA18 | fixed rescue rr0.05 | 0.883 | 0.4774 | 0 | 995 |
| QA18 | fixed rescue rr0.5 | 0.883 | 0.4889 | 799 | 196 |

## Interpretation

The low rescue threshold is too permissive: all tasks almost always use the
R174 direct-joint path, so the controller reduces to R174. Raising the role
threshold forces base protection, but QA14 loses accuracy and QA18 does not gain
accuracy. QA17 shows small CE gains under base protection, but top-1 does not
improve enough to justify full runs.

The diagnostic is useful: simple scalar role evidence does not cleanly separate
QA14's needed direct-role rescue from QA18's base-protection cases. The
controller needs a more specific rescue signal, likely tied to agreement between
the base candidate, direct-role candidate, and joint candidate rather than only
max evidence magnitude.

## Boundary

R176 rejects scalar base-margin plus max-role-evidence gating as the next main
branch controller. It is safer than R175's six-way WTA in the sense that it can
fall back to R174, but it does not improve the accuracy frontier.

Next step: build an agreement-sensitive local controller, for example protect
base only when base and rescue paths disagree and the rescue path lacks a
target-consistent evidence trace.
