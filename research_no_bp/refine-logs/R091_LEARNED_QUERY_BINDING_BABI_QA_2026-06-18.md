# R091 Learned Query Front-End For Role Binding

**Date**: 2026-06-18

## Purpose

R090 replaced regex event parsing with a local learned event detector, but the QA query side still used explicit question parsing. R091 adds a local no-BP learned query detector and connects it to the role-binding state learner, so both context events and questions are parsed by learned local front-ends.

## Implementation

Updated:

- `babi_no_bp_qa_experiment.py`

Added:

- `LearnedQueryDetector`
- `--role-query-mode regex|learned|hybrid`
- query detector hyperparameters
- `query_metrics.csv`

Mechanism:

- fixed random token/position features;
- local perceptron-style target/wrong-winner updates for query type: `where_is`, `where_before`;
- local prototype averaging for subject and destination slots;
- query labels are derived only from question-local structure, not from QA answers;
- role-binding state consumes learned query output instead of `parse_question_subject()` / `parse_before_question()` when `--role-query-mode learned`.

Static audit:

- `LearnedEventDetector`, `LearnedQueryDetector`, and `RoleBindingStateQALearner` do not read `row["answer"]` or `answer_to_idx[row...]`.

## Commands

Medium QA1:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_learned_event_query_qa1_medium \
  --config en-qa1 \
  --phase-dim 64 --phase-epochs 10 \
  --phase-lr 0.08 --phase-wrong-lr 0.03 \
  --phase-score-scale 6.0 --branch-agreement 0.05 \
  --role-event-mode learned --role-query-mode learned \
  --role-dim 64 --role-lr 1.0 \
  --role-score-scale 8.0 --role-carry-threshold 0.35 \
  --event-dim 64 --event-lr 0.08 --event-epochs 3 \
  --event-score-scale 6.0 --event-confidence-threshold 0.0 \
  --query-dim 64 --query-lr 0.08 --query-epochs 3 \
  --query-score-scale 6.0 --query-confidence-threshold 0.0
```

Medium QA2:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_learned_event_query_qa2_medium \
  --config en-qa2 \
  --phase-dim 64 --phase-epochs 10 \
  --phase-lr 0.08 --phase-wrong-lr 0.03 \
  --phase-score-scale 6.0 --branch-agreement 0.05 \
  --role-event-mode learned --role-query-mode learned \
  --role-dim 64 --role-lr 1.0 \
  --role-score-scale 8.0 --role-carry-threshold 0.35 \
  --event-dim 64 --event-lr 0.08 --event-epochs 3 \
  --event-score-scale 6.0 --event-confidence-threshold 0.0 \
  --query-dim 64 --query-lr 0.08 --query-epochs 3 \
  --query-score-scale 6.0 --query-confidence-threshold 0.0
```

Medium QA3:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_no_bp_qa_experiment.py \
  --out-dir output/babi_learned_event_query_qa3_medium \
  --config en-qa3 \
  --phase-dim 64 --phase-epochs 10 \
  --phase-lr 0.08 --phase-wrong-lr 0.03 \
  --phase-score-scale 6.0 --branch-agreement 0.05 \
  --role-event-mode learned --role-query-mode learned \
  --role-dim 64 --role-lr 1.0 \
  --role-score-scale 8.0 --role-carry-threshold 0.35 \
  --event-dim 64 --event-lr 0.08 --event-epochs 3 \
  --event-score-scale 6.0 --event-confidence-threshold 0.0 \
  --query-dim 64 --query-lr 0.08 --query-epochs 3 \
  --query-score-scale 6.0 --query-confidence-threshold 0.0
```

## Results

Medium QA:

| Task | Method | Test acc | Test CE | State bytes |
|---|---|---:|---:|---:|
| QA1 | learned event + learned query + role binding | 1.000 | 0.003 | 65,843 |
| QA2 | learned event + learned query + role binding | 1.000 | 0.002 | 65,843 |
| QA3 | learned event + learned query + role binding | 1.000 | 0.003 | 65,843 |

Detector metrics on test:

| Task | Event acc | Query acc | Subject acc | Object acc | Location acc | Destination acc |
|---|---:|---:|---:|---:|---:|---:|
| QA1 | 1.000 | 1.000 | 1.000 | n/a | 1.000 | n/a |
| QA2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| QA3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Detector state sizes:

| Task | Event detector bytes | Query detector bytes |
|---|---:|---:|
| QA1 | 4,233 | 1,956 |
| QA2 | 5,104 | 1,669 |
| QA3 | 5,104 | 3,416 |

## Interpretation

R091 removes another explicit symbolic component from the bAbI pipeline:

- Context event parsing is learned locally.
- Query parsing is learned locally.
- Role/state updates remain local associative matrix writes.
- QA1/QA2/QA3 remain solved on 900 train / 1000 test without raw text replay, BP, pretrained encoders, API, or statistical lookup.

The main remaining boundaries are now clearer:

- Event/query labels still come from local sentence/question structure during training.
- bAbI grammar is synthetic and narrow.
- The next step should reduce supervision toward QA-level delayed credit, add paraphrase/noise stress tests, or connect this structured state to a generative token learner.

## Artifacts

- `output/babi_learned_event_query_qa1_smoke/summary.csv`
- `output/babi_learned_event_query_qa1_smoke/event_metrics.csv`
- `output/babi_learned_event_query_qa1_smoke/query_metrics.csv`
- `output/babi_learned_event_query_qa2_smoke/summary.csv`
- `output/babi_learned_event_query_qa2_smoke/event_metrics.csv`
- `output/babi_learned_event_query_qa2_smoke/query_metrics.csv`
- `output/babi_learned_event_query_qa3_smoke/summary.csv`
- `output/babi_learned_event_query_qa3_smoke/event_metrics.csv`
- `output/babi_learned_event_query_qa3_smoke/query_metrics.csv`
- `output/babi_learned_event_query_qa1_medium/summary.csv`
- `output/babi_learned_event_query_qa1_medium/event_metrics.csv`
- `output/babi_learned_event_query_qa1_medium/query_metrics.csv`
- `output/babi_learned_event_query_qa2_medium/summary.csv`
- `output/babi_learned_event_query_qa2_medium/event_metrics.csv`
- `output/babi_learned_event_query_qa2_medium/query_metrics.csv`
- `output/babi_learned_event_query_qa3_medium/summary.csv`
- `output/babi_learned_event_query_qa3_medium/event_metrics.csv`
- `output/babi_learned_event_query_qa3_medium/query_metrics.csv`
