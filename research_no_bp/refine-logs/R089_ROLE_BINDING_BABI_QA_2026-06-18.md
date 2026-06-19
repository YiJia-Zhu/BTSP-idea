# R089 Role-Binding State QA Learner

**Date**: 2026-06-18

## Purpose

Resolve the R082 multi-hop failure by adding an explicit pure no-BP role/state binding mechanism. The target is not a statistical lookup table: the model reads each story as an online event stream and maintains compact associative state without storing raw examples or using answer labels during evaluation.

## Implementation

Updated:

- `babi_no_bp_qa_experiment.py`

Added method:

- `role_binding_state_no_bp`

Mechanism:

- fixed random role vectors for people, objects, and locations;
- local delta-Hebbian associative writes;
- per-story online state reset, no raw replay;
- `person -> location` state;
- `object -> owner` state;
- `object -> location` state;
- `(object, destination) -> previous location` state for QA3 before-location queries;
- local carry update: when a person moves, objects whose owner vector matches that person are moved to the person's new location;
- no BP, BPTT, pretrained encoder, API, test-answer updates, or statistical n-gram table.

Important boundary:

- This is a structured state-machine-like neural binding prototype. It is not yet a general language learner and it still uses task-specific event parsing. It is nevertheless aligned with the reset goal because the stored state is vector/matrix associative neural state, not raw text or a sparse token-count cache.

## Commands

Medium QA1:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_role_binding_qa1_medium \
  --config en-qa1 \
  --phase-dim 64 --phase-epochs 10 \
  --phase-lr 0.08 --phase-wrong-lr 0.03 \
  --phase-score-scale 6.0 --branch-agreement 0.05 \
  --role-dim 64 --role-lr 1.0 \
  --role-score-scale 8.0 --role-carry-threshold 0.35
```

Medium QA2:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_role_binding_qa2_medium \
  --config en-qa2 \
  --phase-dim 64 --phase-epochs 10 \
  --phase-lr 0.08 --phase-wrong-lr 0.03 \
  --phase-score-scale 6.0 --branch-agreement 0.05 \
  --role-dim 64 --role-lr 1.0 \
  --role-score-scale 8.0 --role-carry-threshold 0.35
```

Medium QA3:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_role_binding_qa3_medium \
  --config en-qa3 \
  --phase-dim 64 --phase-epochs 10 \
  --phase-lr 0.08 --phase-wrong-lr 0.03 \
  --phase-score-scale 6.0 --branch-agreement 0.05 \
  --role-dim 64 --role-lr 1.0 \
  --role-score-scale 8.0 --role-carry-threshold 0.35
```

## Results

Medium test split, 900 train / 1000 test:

| Task | Method | Test acc | Test CE | State bytes | Stores raw text |
|---|---|---:|---:|---:|---|
| QA1 | majority_no_memory | 0.154 | 1.792 | 162 | no |
| QA1 | raw_lexical_retrieval | 0.237 | 1.772 | 293,752 | yes |
| QA1 | hashed_lookup_diagnostic | 0.321 | 13.773 | 6,888 | no |
| QA1 | symbolic_location_tracker | 1.000 | 0.002 | 0 | no |
| QA1 | phase_dendritic_no_bp | 0.822 | 1.162 | 9,629 | no |
| QA1 | role_binding_state_no_bp | 1.000 | 0.003 | 65,745 | no |
| QA2 | majority_no_memory | 0.187 | 1.811 | 162 | no |
| QA2 | raw_lexical_retrieval | 0.172 | 1.790 | 628,920 | yes |
| QA2 | hashed_lookup_diagnostic | 0.187 | 16.848 | 13,247 | no |
| QA2 | symbolic_location_tracker | 0.187 | 1.811 | 0 | no |
| QA2 | phase_dendritic_no_bp | 0.179 | 1.791 | 9,629 | no |
| QA2 | role_binding_state_no_bp | 1.000 | 0.002 | 65,745 | no |
| QA3 | majority_no_memory | 0.185 | 1.789 | 162 | no |
| QA3 | raw_lexical_retrieval | 0.175 | 1.791 | 1,563,933 | yes |
| QA3 | hashed_lookup_diagnostic | 0.185 | 16.889 | 13,650 | no |
| QA3 | symbolic_location_tracker | 0.185 | 1.789 | 0 | no |
| QA3 | phase_dendritic_no_bp | 0.158 | 1.798 | 9,629 | no |
| QA3 | role_binding_state_no_bp | 1.000 | 0.003 | 65,745 | no |

Capacity smoke ablation:

| Task | role_dim | Test acc | Test CE |
|---|---:|---:|---:|
| QA2 smoke | 8 | 0.758 | 1.938 |
| QA2 smoke | 16 | 0.967 | 0.206 |
| QA2 smoke | 64 | 1.000 | 0.002 |
| QA3 smoke | 8 | 0.483 | 4.130 |
| QA3 smoke | 16 | 0.708 | 2.286 |
| QA3 smoke | 64 | 1.000 | 0.004 |

## Interpretation

R089 directly addresses the R082 boundary:

- QA2 object carry is solved by binding object ownership and propagating object location when the owner moves.
- QA3 before-location is solved by binding `(object, destination)` to the object's previous location during state transitions.
- The result is not explained by token-count statistics: hashed lookup remains at majority level on QA2/QA3 and raw retrieval is weak.
- The result is also not from answer leakage: `RoleBindingStateQALearner` does not read `row["answer"]`; it only reads context and question text.
- The low-dimension smoke ablation shows a capacity effect, which supports the vector role-binding interpretation.

## Limitations

- Event parsing is still task-specific and symbolic at the input boundary.
- The method solves structured bAbI world-state tracking, not open-domain generation.
- The next step should replace brittle regex parsing with learned local event detectors and connect the role-binding state to the phase/dendritic QA readout and later TinyStories-style generation.

## Artifacts

- `output/babi_role_binding_qa1_smoke/summary.csv`
- `output/babi_role_binding_qa2_smoke/summary.csv`
- `output/babi_role_binding_qa3_smoke/summary.csv`
- `output/babi_role_binding_qa1_medium/summary.csv`
- `output/babi_role_binding_qa2_medium/summary.csv`
- `output/babi_role_binding_qa3_medium/summary.csv`
- `output/babi_role_binding_qa2_dim8_smoke/summary.csv`
- `output/babi_role_binding_qa2_dim16_smoke/summary.csv`
- `output/babi_role_binding_qa3_dim8_smoke/summary.csv`
- `output/babi_role_binding_qa3_dim16_smoke/summary.csv`
