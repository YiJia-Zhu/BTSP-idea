# R180: Full-Answer QA19 Evaluation

Date: 2026-06-19

## Purpose

The current local data does not include CLUTRR. The available QA stress set is
`data/babi_qa_processed`, including `en-qa19`, whose answers are two-token path
directions such as `south east`.

Before using QA19 as a main benchmark, the unified bAbI token evaluator needed a
full-answer metric. The old evaluation path skipped multi-token answers by
default and, when multi-token answers were allowed, only scored the first answer
token. That is not a valid final metric for QA19.

## Implementation

Changed `babi_unified_token_qa_experiment.py` only.

- Preserve existing `answer_accuracy`, `answer_loss`, and `answer_ppl` as the
  first answer-token metric for compatibility with QA1/2/3/14/15/16/17/18.
- Add teacher-forced full-answer token metrics:
  - `full_answer_token_accuracy`
  - `full_answer_loss`
  - `full_answer_ppl`
  - `full_answer_token_targets`
- Add greedy exact-match full-answer sequence metric:
  - `full_answer_accuracy`
  - `full_answer_sequences`
- Add prediction sample fields:
  - `target_answer_decoded`
  - `prediction_answer_decoded`
  - `full_correct`
  - `full_answer_token_correct`
  - `full_answer_token_total`
- Force all answer tokens into the compact vocabulary, not only the first answer
  token, so QA19 second-direction tokens cannot silently disappear.

This is an evaluation-integrity change, not a model-quality improvement.

## Smoke Command

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --configs en-qa19 \
  --out-dir output/babi_unified_token_qa_r180_full_answer_smoke \
  --train-limit 20 --eval-limit 20 --train-epochs 1 \
  --allow-multi-token-answer \
  --method state_microproto_online \
  --state-dim 32 --state-order 64 --state-decay 0.90 \
  --micro-slots 8 --micro-lr 0.35 --micro-wrong-lr 0.02 \
  --micro-score-scale 8.0 --max-vocab 256
```

## Smoke Results

`output/babi_unified_token_qa_r180_full_answer_smoke/summary.csv`:

| split | evaluated | skipped | first-token acc | full-answer acc | full-token acc | full-token targets |
|---|---:|---:|---:|---:|---:|---:|
| validation | 20 | 0 | 0.15 | 0.00 | 0.20 | 40 |
| test | 20 | 0 | 0.25 | 0.05 | 0.30 | 40 |

The smoke confirms that QA19 now evaluates all two-token answers and exposes the
gap between first-token accuracy and actual full-answer correctness.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  babi_unified_token_qa_experiment.py \
  babi_branch_arbitration_diagnostic.py \
  babi_role_gate_alignment_diagnostic.py \
  synthetic_object_carry_token_experiment.py
```

Passed.

The generated pyc files for these scripts were removed after compilation.

## Next

Run a medium/full QA19 evaluation with `--allow-multi-token-answer` and report
both first-token and full-answer metrics. Do not use the old first-token-only
number as a QA19 claim.
