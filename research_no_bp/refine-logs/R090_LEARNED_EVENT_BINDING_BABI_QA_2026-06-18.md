# R090 Learned Event Front-End For Role Binding

**Date**: 2026-06-18

## Purpose

Reduce the main limitation of R089: the role-binding state learner solved bAbI QA2/QA3, but it used regex/event parsing at the input boundary. R090 replaces that input boundary with a local no-BP learned event detector while keeping the same no-raw-data role-binding state.

## Implementation

Updated:

- `babi_no_bp_qa_experiment.py`

Added:

- `LearnedEventDetector`
- `--role-event-mode regex|learned|hybrid`
- event detector hyperparameters and `event_metrics.csv`

Mechanism:

- fixed random token/position features;
- local perceptron-style target/wrong-winner updates for event type: `none`, `move`, `pickup`, `drop`;
- local prototype averaging for person/object/location slots;
- labels are derived from sentence-local event structure only, not from QA answers;
- role-binding state consumes the learned event outputs and updates the same associative matrices as R089.

The learned detector block does not read `row["answer"]` or `answer_to_idx`.

## Commands

Smoke QA2:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_learned_event_qa2_smoke \
  --config en-qa2 --train-limit 120 --eval-limit 120 \
  --phase-dim 16 --phase-epochs 1 \
  --role-event-mode learned \
  --role-dim 64 --role-lr 1.0 \
  --role-score-scale 8.0 --role-carry-threshold 0.35 \
  --event-dim 64 --event-lr 0.08 --event-epochs 3 \
  --event-score-scale 6.0 --event-confidence-threshold 0.0
```

Smoke QA3:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_learned_event_qa3_smoke \
  --config en-qa3 --train-limit 120 --eval-limit 120 \
  --phase-dim 16 --phase-epochs 1 \
  --role-event-mode learned \
  --role-dim 64 --role-lr 1.0 \
  --role-score-scale 8.0 --role-carry-threshold 0.35 \
  --event-dim 64 --event-lr 0.08 --event-epochs 3 \
  --event-score-scale 6.0 --event-confidence-threshold 0.0
```

Medium QA2:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_learned_event_qa2_medium \
  --config en-qa2 \
  --phase-dim 64 --phase-epochs 10 \
  --phase-lr 0.08 --phase-wrong-lr 0.03 \
  --phase-score-scale 6.0 --branch-agreement 0.05 \
  --role-event-mode learned \
  --role-dim 64 --role-lr 1.0 \
  --role-score-scale 8.0 --role-carry-threshold 0.35 \
  --event-dim 64 --event-lr 0.08 --event-epochs 3 \
  --event-score-scale 6.0 --event-confidence-threshold 0.0
```

Medium QA3:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_learned_event_qa3_medium \
  --config en-qa3 \
  --phase-dim 64 --phase-epochs 10 \
  --phase-lr 0.08 --phase-wrong-lr 0.03 \
  --phase-score-scale 6.0 --branch-agreement 0.05 \
  --role-event-mode learned \
  --role-dim 64 --role-lr 1.0 \
  --role-score-scale 8.0 --role-carry-threshold 0.35 \
  --event-dim 64 --event-lr 0.08 --event-epochs 3 \
  --event-score-scale 6.0 --event-confidence-threshold 0.0
```

## Results

Medium QA:

| Task | Method | Test acc | Test CE | State bytes | Event detector bytes |
|---|---|---:|---:|---:|---:|
| QA2 | role_binding_state_no_bp + learned event | 1.000 | 0.002 | 65,794 | 5,104 |
| QA3 | role_binding_state_no_bp + learned event | 1.000 | 0.003 | 65,794 | 5,104 |

Event detector test metrics:

| Task | Test sentences | Event acc | Person acc | Object acc | Location acc |
|---|---:|---:|---:|---:|---:|
| QA2 | 15,426 | 1.000 | 1.000 | 1.000 | 1.000 |
| QA3 | 50,968 | 1.000 | 1.000 | 1.000 | 1.000 |

Smoke:

| Task | Test acc | Test CE | Event test acc |
|---|---:|---:|---:|
| QA2 smoke | 0.983 | 0.141 | 0.989 |
| QA3 smoke | 1.000 | 0.004 | 0.999 |

## Interpretation

R090 moves the pipeline one step closer to the desired no-BP neural framework:

- R089's task-specific regex parser is no longer required for QA2/QA3 medium.
- The event front-end uses local perceptron/prototype updates, fixed random features, and no BP.
- QA performance remains at 1.000 after replacing regex events with learned events.
- The learned front-end stores about 5KB of detector state and the role-binding state remains about 66KB.

Important boundary:

- The detector still uses sentence-local labels generated from bAbI sentence structure during training. It is not yet learning open-domain event semantics from only QA reward.
- Slot labels are easy in this synthetic grammar. Next work should introduce paraphrases/noise or reduce supervision toward delayed QA-level credit.
- The state/query layer still uses explicit question parsing; this must also become learned before calling the model general.

## Artifacts

- `output/babi_learned_event_qa2_smoke/summary.csv`
- `output/babi_learned_event_qa2_smoke/event_metrics.csv`
- `output/babi_learned_event_qa3_smoke/summary.csv`
- `output/babi_learned_event_qa3_smoke/event_metrics.csv`
- `output/babi_learned_event_qa2_medium/summary.csv`
- `output/babi_learned_event_qa2_medium/event_metrics.csv`
- `output/babi_learned_event_qa3_medium/summary.csv`
- `output/babi_learned_event_qa3_medium/event_metrics.csv`
