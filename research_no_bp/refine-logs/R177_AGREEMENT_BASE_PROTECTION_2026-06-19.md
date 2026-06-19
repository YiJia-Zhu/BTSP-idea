# R177 Agreement-Sensitive Base Protection

**Date**: 2026-06-19

## Goal

R176 showed that scalar max-role evidence is not enough to decide when to
protect the base branch. R177 tests a more specific hypothesis: only consider
base protection when the base path and rescue path predict different tokens.

## Implementation

Extended `babi_branch_arbitration_diagnostic.py` so it now uses
`branch_component_scores()` directly. This means diagnostics include the same
joint rescue and suppression paths used by the behavioral model. It also writes
`branch_pair_agreement_summary.csv`, with pairwise agreement, left/right wins on
disagreement, and both-wrong counts.

Extended `babi_unified_token_qa_experiment.py` with:

- `--role-branch-arbiter agreement_base_protect`

This default-off controller chooses `base_only` only when:

- `base_only` and the default rescue path predict different tokens;
- `base_only` top-2 margin is at least `--role-branch-arbiter-base-margin`.

Otherwise it uses the default path, set to `base_plus_direct_joint` in R177.

## Diagnostic Runs

Single-task diagnostics, exact R174 protect-direct config:

- `output/babi_branch_arbitration_r177_qa14_joint_agreement`
- `output/babi_branch_arbitration_r177_qa17_joint_agreement`
- `output/babi_branch_arbitration_r177_qa18_joint_agreement`

The earlier combined-config diagnostic
`output/babi_branch_arbitration_r177_joint_agreement` is not used for the main
claim because it trains all tasks into one memory and is not directly comparable
to the single-task R174 runs.

## Diagnostic Results

Test split, single-task memory:

| Task | Variant | Accuracy | CE |
|---|---|---:|---:|
| QA14 | base_only | 0.218 | 1.7289 |
| QA14 | base_plus_direct | 0.391 | 1.6532 |
| QA14 | base_plus_direct_joint/full | 0.394 | 1.6482 |
| QA17 | base_only | 0.487 | 0.7087 |
| QA17 | base_plus_direct | 0.516 | 0.7056 |
| QA17 | base_plus_direct_joint/full | 0.529 | 0.6986 |
| QA18 | base_only | 0.917 | 0.4750 |
| QA18 | base_plus_direct | 0.909 | 0.4761 |
| QA18 | base_plus_direct_joint/full | 0.903 | 0.4549 |

Pairwise base-vs-joint disagreement:

| Task | Agree rate | Disagreements | Base wins | Joint wins | Both wrong |
|---|---:|---:|---:|---:|---:|
| QA14 | 0.332 | 668 | 90 | 266 | 312 |
| QA17 | 0.772 | 228 | 93 | 135 | 0 |
| QA18 | 0.966 | 34 | 24 | 10 | 0 |

This confirms that disagreement is informative. QA18 is the only task where
base wins more often than joint on disagreements, but these disagreements are
rare.

## Controller Runs

Medium seed0:

- `output/babi_unified_role_transition_r177_qa14_medium_agree_base_protect`
- `output/babi_unified_role_transition_r177_qa17_medium_agree_base_protect`
- `output/babi_unified_role_transition_r177_qa18_medium_agree_base_protect`
- `output/babi_unified_role_transition_r177_qa14_medium_agree_base_protect_m02`
- `output/babi_unified_role_transition_r177_qa17_medium_agree_base_protect_m02`
- `output/babi_unified_role_transition_r177_qa18_medium_agree_base_protect_m02`

Full QA18 sanity check:

- `output/babi_unified_role_transition_r177_qa18_full_agree_base_protect`

## Controller Results

Medium test split:

| Task | Variant | Accuracy | CE | Base choices |
|---|---|---:|---:|---:|
| QA14 | R174 protect-direct | 0.370 | 1.6726 | - |
| QA14 | agreement m0.5 | 0.370 | 1.6726 | 9 |
| QA14 | agreement m0.2 | 0.340 | 1.6878 | 77 |
| QA17 | R174 protect-direct | 0.553 | 0.7011 | - |
| QA17 | agreement m0.5 | 0.553 | 0.7011 | 1 |
| QA17 | agreement m0.2 | 0.550 | 0.7018 | 28 |
| QA18 | R174 protect-direct | 0.883 | 0.4774 | - |
| QA18 | agreement m0.5 | 0.883 | 0.4774 | 1 |
| QA18 | agreement m0.2 | 0.883 | 0.4774 | 7 |

Full QA18:

| Variant | Accuracy | CE | Base choices |
|---|---:|---:|---:|
| R174 protect-direct | 0.903 | 0.4549 | - |
| agreement m0.5 | 0.903 | 0.4549 | 1 |

## Interpretation

Agreement is the right diagnostic axis but not yet a sufficient controller.
The oracle-style pair table shows QA18 would benefit from choosing base on some
base-vs-joint conflicts. However, the simple base-margin threshold misses almost
all useful QA18 conflicts at `0.5`, while lower threshold `0.2` starts selecting
bad base fallbacks on QA14 and QA17.

The next controller should learn a conflict-specific local trace: not a six-way
path classifier, and not a scalar margin threshold, but a binary base-vs-rescue
conflict predictor updated only on disagreements with answer-token credit.

## Boundary

R177 rejects hand-set agreement plus base-margin thresholds as the final branch
controller. It does provide the next target: a local disagreement-only
eligibility memory for base-vs-rescue arbitration.
