# R169 Branch Arbitration Diagnostic

**Date**: 2026-06-19

## Goal

R168 showed that fixed branch scaling cannot resolve the QA14/QA17/QA18
tradeoff. R169 decomposes the branch-separated readout into component scores so
the next arbiter can be designed from evidence rather than another scalar sweep.

This is diagnostic only. No component choice or arbiter is used for training.

## Implementation

Added `babi_branch_arbitration_diagnostic.py`.

The script trains the same branch-separated role-transition memory, then scores
each prompt with:

- `base_only`: output bias + base-state prototype scores;
- `role_only`: output bias + role-state prototype scores;
- `base_plus_role`: base and role branch prototype scores;
- `base_plus_direct`: branch scores plus direct role score;
- `full`: the model's actual score path.

It writes numeric-only diagnostics and does not store raw prompt text:

- `branch_rows.csv`;
- `branch_component_summary.csv`;
- `branch_flip_summary.csv`;
- `config.json`.

## Runs

Full seed0 branch diagnostics, trained separately per config:

- `output/babi_branch_arbitration_r169_qa14_full`
- `output/babi_branch_arbitration_r169_qa17_full`
- `output/babi_branch_arbitration_r169_qa18_full`

Aggregate output:

- `output/babi_branch_arbitration_r169_summary/component_summary.csv`
- `output/babi_branch_arbitration_r169_summary/flip_summary.csv`

## Component Results

Test split:

| Task | Component | Accuracy | Loss | Mean margin |
|---|---|---:|---:|---:|
| QA14 | base_only | 0.219 | 1.7287 | 0.064 |
| QA14 | role_only | 0.197 | 1.7825 | 0.087 |
| QA14 | base_plus_role | 0.220 | 1.7180 | 0.108 |
| QA14 | full | 0.398 | 1.6529 | 0.161 |
| QA17 | base_only | 0.489 | 0.7087 | 0.215 |
| QA17 | role_only | 0.514 | 0.6926 | 0.142 |
| QA17 | base_plus_role | 0.512 | 0.7076 | 0.265 |
| QA17 | full | 0.512 | 0.7076 | 0.265 |
| QA18 | base_only | 0.917 | 0.4752 | 0.568 |
| QA18 | role_only | 0.486 | 0.6960 | 0.073 |
| QA18 | base_plus_role | 0.909 | 0.4764 | 0.571 |
| QA18 | full | 0.909 | 0.4764 | 0.571 |

Flip summary versus `base_only`:

| Task | Helpful flips | Harmful flips | Same prediction | Base-role agree | Base-full agree | Role-full agree |
|---|---:|---:|---:|---:|---:|---:|
| QA14 | 261 | 82 | 362 | 0.268 | 0.362 | 0.412 |
| QA17 | 112 | 89 | 799 | 0.499 | 0.799 | 0.692 |
| QA18 | 7 | 15 | 978 | 0.477 | 0.978 | 0.499 |

## Findings

1. QA14 needs the direct role score. Base and role branch prototypes alone do
   not explain the full gain: `base_plus_role` is only `0.220`, while `full`
   reaches `0.398`.

2. QA18 should mostly trust the base branch. `base_only` is actually slightly
   better than full (`0.917/0.4752` vs `0.909/0.4764`), and full changes only
   22/1000 predictions relative to base.

3. QA17 is qualitatively different. `role_only` is better than `base_only`
   (`0.514/0.6926` vs `0.489/0.7087`), but simple branch addition does not turn
   that into the concat result from R165 (`0.578/0.6734`).

4. Agreement is informative but not sufficient alone. QA14 has low base-role
   agreement and many helpful flips; QA18 also has low base-role agreement but
   should mostly reject role changes. The arbiter needs margins and task-local
   evidence, not just agreement.

## Interpretation

R169 points to a component-selective arbitration mechanism:

- QA14: allow direct role drive when it can cause a confident transition flip;
- QA18: protect base branch unless role evidence is unusually strong;
- QA17: use role branch as a candidate source, but not by naive score addition.

This is close to the biologically inspired direction: separate circuits provide
competing hypotheses, and an inhibitory/arbiter circuit must learn when each
one should control behavior.

## Next Steps

1. Add train-split branch flip diagnostics with margin features.
2. Train a local no-BP branch arbiter that predicts helpful versus harmful
   branch intervention from local margins/agreement.
3. Validate the arbiter on QA14/17/18 full seed0 before seed repeats.
