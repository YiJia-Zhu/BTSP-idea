# R166 Role-Score Gate Boundary

**Date**: 2026-06-19

## Goal

R165 showed mixed transfer: role-transition helps QA14/QA17 but hurts QA18.
R166 tests a simple local arbiter: only add direct role scores when the base
readout margin is low. The intended mechanism is biologically plausible
competition/arbitration: if the existing readout is confident, suppress the
extra transition drive.

This is still pure no-BP. The gate uses only local scores and fixed margins, no
test labels, no parser, no answer head, and no BP.

## Implementation

Added optional role-score gate arguments to `babi_unified_token_qa_experiment.py`:

- `--role-score-gate-mode`
  - `none` keeps prior behavior;
  - `base_low_margin`;
  - `role_high_margin`;
  - `base_low_and_role_high`.
- `--role-score-gate-base-margin`
- `--role-score-gate-role-margin`

Defaults preserve R160-R165 behavior. The gate only controls whether the direct
role-score vector is added to the final full-vocabulary readout. It does not
change role feature construction, role-gate learning, or prototype updates.

## Runs

Medium scan with `train_limit=300`, `eval_limit=300`, seed 0:

- QA14/QA17/QA18 no-gate;
- QA14/QA17/QA18 `base_low_margin` with margins `0.25`, `0.5`, `1.0`;
- QA14 and QA18 `role_score_scale=0.0` probes.

Aggregate outputs:

- `output/babi_unified_role_transition_r166_role_score_gate_medium/sweep_summary.csv`
- `output/babi_unified_role_transition_r166_role_score_gate_medium/variant_summary.csv`

## Results

### Base-Low-Margin Gate

| Task | Variant | Val acc | Val CE | Test acc | Test CE | Gate open rate |
|---|---|---:|---:|---:|---:|---:|
| QA14 | none | 0.240 | 1.7518 | 0.330 | 1.7123 | 1.000 |
| QA14 | margin 0.25 | 0.240 | 1.7570 | 0.330 | 1.7179 | 0.610 |
| QA14 | margin 0.5 | 0.240 | 1.7518 | 0.330 | 1.7134 | 0.811 |
| QA14 | margin 1.0 | 0.240 | 1.7518 | 0.330 | 1.7123 | 0.979 |
| QA17 | none | 0.479 | 0.7095 | 0.480 | 0.7002 | 1.000 |
| QA17 | margin 0.25 | 0.479 | 0.7095 | 0.480 | 0.7002 | 0.464 |
| QA17 | margin 0.5 | 0.479 | 0.7095 | 0.480 | 0.7002 | 0.697 |
| QA17 | margin 1.0 | 0.479 | 0.7095 | 0.480 | 0.7002 | 0.921 |
| QA18 | none | 0.832 | 0.6077 | 0.790 | 0.6172 | 1.000 |
| QA18 | margin 0.25 | 0.832 | 0.6077 | 0.790 | 0.6172 | 0.416 |
| QA18 | margin 0.5 | 0.832 | 0.6077 | 0.790 | 0.6172 | 0.742 |
| QA18 | margin 1.0 | 0.832 | 0.6077 | 0.790 | 0.6172 | 0.969 |

The gate changes open rate substantially, but it does not materially change
accuracy. CE shifts are tiny and not in a useful direction.

### Role-Score-Zero Probe

| Task | Variant | Val acc | Val CE | Test acc | Test CE |
|---|---|---:|---:|---:|---:|
| QA14 | no-gate | 0.240 | 1.7518 | 0.330 | 1.7123 |
| QA14 | role score 0 | 0.170 | 1.8081 | 0.260 | 1.7659 |
| QA18 | no-gate | 0.832 | 0.6077 | 0.790 | 0.6172 |
| QA18 | role score 0 | 0.811 | 0.6114 | 0.770 | 0.6186 |

Removing the direct role logits does not protect QA18 and hurts QA14. This
suggests the interference is not only a final-logit problem; the role feature
and prototype path also matter.

## Findings

1. Simple base-margin arbitration is ineffective. It changes gate open rates
   from about `0.42` to `0.98`, but leaves QA14/17/18 top-1 behavior almost
   unchanged.

2. Direct role-score removal is also not the fix. QA18 remains below the R165
   microproto baseline, and QA14 loses much of the role-transition benefit.

3. The QA18 failure likely happens inside the shared role-augmented feature
   readout, not just in the direct role-score vector. A scalar gate over final
   logits is too late and too coarse.

## Interpretation

R166 rules out the easiest arbiter. The next useful mechanism should separate
feature channels before they merge into shared prototypes, or learn a local
feature-level arbiter from train/validation signals. This is consistent with
the broader no-BP direction: the model needs inhibitory routing between
competing circuits, not a hand-set confidence threshold on the final logits.

## Next Steps

1. Add a branch-separated readout where base-state and role-state prototypes
   keep separate scores, then arbitrate scores locally.
2. Use train/validation-only flip diagnostics to learn whether role branch
   helps or hurts per prompt family.
3. Only after that, repeat QA14/17/18 full seeds 1/2.
