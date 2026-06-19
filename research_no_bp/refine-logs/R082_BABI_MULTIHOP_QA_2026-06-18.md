# R082 bAbI Multi-Hop QA Boundary

**Date**: 2026-06-18

## Purpose

Test whether the QA1 pure no-BP answer selector transfers beyond single-fact location recall. This directly probes the current model's missing state transition / multi-hop binding capacity.

## Commands

QA2:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_no_bp_qa2_medium \
  --config en-qa2 \
  --phase-dim 64 --phase-epochs 10 \
  --phase-lr 0.08 --phase-wrong-lr 0.03 \
  --phase-score-scale 6.0 --branch-agreement 0.05
```

QA3:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_no_bp_qa3_medium \
  --config en-qa3 \
  --phase-dim 64 --phase-epochs 10 \
  --phase-lr 0.08 --phase-wrong-lr 0.03 \
  --phase-score-scale 6.0 --branch-agreement 0.05
```

Both runs use 900 train examples, 100 validation examples, and 1000 test examples. Test answers are not used for updates.

## Results

QA2 test:

| Method | Test acc | Test CE | State bytes | Role |
|---|---:|---:|---:|---|
| majority_no_memory | 0.187 | 1.811 | 162 | baseline |
| raw_lexical_retrieval | 0.172 | 1.790 | 628,920 | raw diagnostic |
| hashed_lookup_diagnostic | 0.187 | 16.848 | 13,247 | statistical diagnostic |
| symbolic_location_tracker | 0.187 | 1.811 | 0 | QA1-style symbolic bound, not applicable to object carry |
| phase_dendritic_no_bp | 0.179 | 1.791 | 9,629 | pure no-BP neural |

QA3 test:

| Method | Test acc | Test CE | State bytes | Role |
|---|---:|---:|---:|---|
| majority_no_memory | 0.185 | 1.789 | 162 | baseline |
| raw_lexical_retrieval | 0.175 | 1.791 | 1,563,933 | raw diagnostic |
| hashed_lookup_diagnostic | 0.185 | 16.889 | 13,650 | statistical diagnostic |
| symbolic_location_tracker | 0.185 | 1.789 | 0 | QA1-style symbolic bound, not applicable to before-location reasoning |
| phase_dendritic_no_bp | 0.158 | 1.798 | 9,629 | pure no-BP neural |

## Interpretation

R082 is a clear mechanism boundary:

- QA1 success does not transfer to QA2/QA3.
- The current phase/dendritic answer selector is good at single-fact subject-location association but does not learn object carry, temporal before-location, or multi-hop state transitions.
- Raw retrieval and hashed lookup are also weak, so the failure is not merely because a neural method lost to a statistical table.
- The QA1 symbolic tracker intentionally does not solve QA2/QA3 because it only tracks person movement; it is retained as a diagnostic warning that task-specific symbolic state can solve narrow cases but is not the target method.

## Consequence

The next model step should not be another minor readout tweak. The architecture needs an explicit no-BP state mechanism for subject-object-location binding, such as:

- recurrent/SSM state with local eligibility-gated transition writes;
- dendritic branches for subject, object, relation, and recency roles;
- local inhibitory conflict resolution between object-owner and object-location traces;
- apical/feedback-alignment-style credit signals for multi-hop state errors.

## Artifacts

- `output/babi_no_bp_qa2_medium/summary.csv`
- `output/babi_no_bp_qa2_medium/predictions_sample.csv`
- `output/babi_no_bp_qa2_medium/config.json`
- `output/babi_no_bp_qa3_medium/summary.csv`
- `output/babi_no_bp_qa3_medium/predictions_sample.csv`
- `output/babi_no_bp_qa3_medium/config.json`
