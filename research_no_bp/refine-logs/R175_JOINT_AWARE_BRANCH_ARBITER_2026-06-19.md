# R175 Joint-Aware Branch Arbiter

**Date**: 2026-06-19

## Goal

R174 made joint suppression safer by protecting candidates with direct role
evidence, but the final readout still uses one fixed base/role/joint mixture.
R175 tests whether the existing local prototype branch arbiter can be extended
to choose joint rescue paths per prompt.

## Implementation

Extended `babi_unified_token_qa_experiment.py` with a default-off switch:

- `--role-branch-arbiter-joint-variants`

When enabled, the local branch arbiter candidate set expands from:

- `base_only`
- `role_only`
- `base_plus_role`
- `base_plus_direct`

to also include:

- `base_plus_role_joint`
- `base_plus_direct_joint`

The default behavior is unchanged. Joint arbiter variants require
`--role-joint-rescue-readout`, and stats now report the active arbiter variant
list plus chosen/target counts for all variants.

## Runs

Smoke:

- `output/babi_unified_role_transition_r175_qa14_smoke_joint_arbiter`

Medium seed0, exact R174 protect-direct config plus local prototype arbiter with
joint variants:

- `output/babi_unified_role_transition_r175_qa14_medium_joint_arbiter`
- `output/babi_unified_role_transition_r175_qa17_medium_joint_arbiter`
- `output/babi_unified_role_transition_r175_qa18_medium_joint_arbiter`

## Medium Results

Test split:

| Task | Variant | Accuracy | CE |
|---|---|---:|---:|
| QA14 | R174 protect-direct | 0.370 | 1.6726 |
| QA14 | R175 joint-aware local arbiter | 0.310 | 1.7257 |
| QA17 | R174 protect-direct | 0.553 | 0.7011 |
| QA17 | R175 joint-aware local arbiter | 0.527 | 0.6974 |
| QA18 | R174 protect-direct | 0.883 | 0.4774 |
| QA18 | R175 joint-aware local arbiter | 0.803 | 0.5043 |

Arbiter chosen counts:

| Task | base | role | base+role | direct | joint | direct+joint |
|---|---:|---:|---:|---:|---:|---:|
| QA14 | 293 | 165 | 102 | 164 | 136 | 140 |
| QA17 | 230 | 251 | 273 | 0 | 241 | 1 |
| QA18 | 219 | 167 | 196 | 0 | 412 | 1 |

Target-update counts:

| Task | base | role | base+role | direct | joint | direct+joint |
|---|---:|---:|---:|---:|---:|---:|
| QA14 | 77 | 53 | 9 | 31 | 35 | 95 |
| QA17 | 76 | 72 | 18 | 0 | 134 | 0 |
| QA18 | 72 | 25 | 14 | 0 | 189 | 0 |

## Interpretation

The new code path works: joint variants enter the arbiter feature, prototype
bank, selection counts, and target updates. Behaviorally, however, the local
prototype arbiter is negative. It over-switches among path families and loses
top-1 accuracy on all three medium tasks.

The QA17 CE improvement (`0.7011 -> 0.6974`) suggests joint-aware path selection
can move probability mass in a useful direction, but the winner selection is
too noisy. QA18 is the clearest failure: the arbiter selects `base_plus_role_joint`
412 times even though earlier diagnostics showed QA18 often needs base-protected
behavior.

## Boundary

R175 rejects a generic local WTA classifier over branch score paths as the next
main mechanism. The branch selector needs stronger structural evidence, likely
separate base-protection and direct-rescue gates, rather than a single prototype
bank trained to imitate whichever branch has the best target margin.

Next step: use the R169/R175 diagnostics to build a two-stage local controller:
first protect base when base margin is high, then allow direct/joint rescue only
when role or joint evidence clears a task-local confidence test.
