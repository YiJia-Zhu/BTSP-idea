# R146 Unified Token bAbI QA Prompt Evaluation

**Date**: 2026-06-19
**Status**: DONE-BOUNDARY
**Task**: bAbI `en-qa1` as prompt-formatted next-token QA

## Purpose

The 2026-06-19 goal reset requires QA to be evaluated through the same unified
token model used for generation: raw token sequence in, full compact-vocabulary
next-token probabilities out.  This run adds that path and deliberately avoids
the archived bAbI answer heads, symbolic parsers, role states, and task-specific
front-ends.

Each row is serialized as:

```text
Context:
...
Question: ...
Answer:
```

The metric is exact match on the first answer token after `Answer:`.  For
`en-qa1`, all six location answers are single tokenizer tokens, so no examples
are skipped.

## Implementation

Added:

- `babi_unified_token_qa_experiment.py`

Properties:

- reuses the same no-BP token memories from `phase_binding_online_stream_experiment.py`;
- supports trace, DLL, NoProp, and Hebbian KV token memories;
- supports optional answer-position-only local tuning via `--answer-only-train`;
- writes parseable `summary.csv`, `predictions_sample.csv`, and `config.json`;
- stores no raw text in the model state.  Prediction artifacts contain decoded
  text only for inspection.

## Commands

Smoke trace:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --out-dir output/babi_unified_token_qa_smoke \
  --configs en-qa1 --train-limit 80 --eval-limit 80 --max-vocab 256 \
  --method phase_trace_competitive_online \
  --phase-dim 32 --trace-dim 32 --trace-order 64 \
  --branch-orders 1 2 --branch-weights 0.5 0.5 \
  --competitive-lr 0.12 --competitive-neg-k 4 \
  --competitive-score-scale 8.0 --train-epochs 1 --seed 0
```

Full NoProp, all prompt-token updates:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --out-dir output/babi_unified_token_qa_enqa1_full_seed0 \
  --configs en-qa1 --max-vocab 512 \
  --method phase_trace_noprop_local_competitive_online \
  --phase-dim 64 --trace-dim 64 --trace-order 96 \
  --branch-orders 1 2 --branch-weights 0.5 0.5 \
  --competitive-lr 0.14 --competitive-neg-k 8 \
  --competitive-score-scale 9.0 \
  --noprop-hidden-dims 256 --noprop-label-dim 128 \
  --noprop-alpha-start 0.95 --noprop-alpha-end 0.50 \
  --adaptive-inhibition --feature-calibration \
  --feature-calibration-derived-codes --readout-gain 1.15 \
  --train-epochs 1 --seed 0
```

Full KV, answer-position updates:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --out-dir output/babi_unified_token_qa_enqa1_kv_answer_only_seed0_postdiag \
  --configs en-qa1 --max-vocab 512 \
  --method phase_trace_kv_competitive_online \
  --phase-dim 64 --trace-dim 64 --trace-order 128 \
  --branch-orders 1 2 --branch-weights 0.5 0.5 \
  --competitive-lr 0.16 --competitive-neg-k 8 \
  --competitive-score-scale 9.0 \
  --kv-order 128 --kv-dim 128 --kv-score-weight 1.0 \
  --kv-weight 0.5 --kv-lr 0.08 --kv-decay 0.0 \
  --adaptive-inhibition --readout-gain 1.15 \
  --answer-only-train --train-epochs 1 --seed 0
```

## Raw Data Table

| Run | Train mode | Train post acc | Val acc | Test acc | Test CE | State bytes |
|---|---|---:|---:|---:|---:|---:|
| trace smoke 80/80 | all prompt tokens | n/a | 0.1625 | 0.2125 | 2.1571 | 663,552 |
| NoProp smoke 80/80 | all prompt tokens | n/a | 0.1875 | 0.2375 | 1.8479 | 5,613,828 |
| NoProp full | all prompt tokens | n/a | 0.2000 | 0.1830 | 1.9058 | 18,108,676 |
| NoProp full | answer-only, 1 epoch | n/a | 0.1800 | 0.1960 | 1.8002 | 18,108,676 |
| NoProp full | answer-only, 5 epochs | n/a | 0.2100 | 0.2080 | 1.8584 | 22,302,980 |
| KV full | answer-only, 1 epoch | 0.1922 | 0.2100 | 0.1900 | 1.7860 | 4,540,416 |

Reference from archived R081:

- majority baseline on full QA1 test: `0.154`;
- task-specific `phase_dendritic_no_bp` answer selector: `0.822`;
- later role-binding state path: `1.000`.

These older QA scores are not unified-model scores because they use separate
answer-selection/state modules.

## Findings

1. The unified token QA evaluator is now operational on the local bAbI data.
   It uses dataset ground-truth answer tokens and reports exact-match accuracy
   without a separate QA head.

2. Current unified token memories do not solve even QA1 under prompt formatting.
   The best tested full result is only about `0.208` test accuracy, barely above
   majority and far below the archived task-specific answer selectors.

3. The failure is not only held-out generalization.  KV answer-only training
   reaches only `0.1922` on `train_post`, so the current trace/KV readout is not
   even memorizing prompt-to-answer mappings reliably.

4. The model tends to collapse onto high-prior answer tokens such as `garden`
   or `bedroom`.  This means the present token learner lacks a state transition
   that can bind "person -> current location" inside the raw token stream.

## Decision

R146 is a necessary boundary result, not a main positive claim.  It prevents
the archived bAbI parser/state results from being mistaken for unified GPT-like
QA ability.

Next mechanism work should add a unified recurrent/state-binding branch that is
still token-level and local no-BP:

- update an internal state while reading the prompt;
- expose that state to the same next-token WTA readout;
- use local target/error/eligibility credit at answer positions;
- avoid hand-written bAbI parsers and avoid a separate QA classifier.

Until that exists, bAbI QA should be reported as an explicit weakness of the
current unified token architecture.
