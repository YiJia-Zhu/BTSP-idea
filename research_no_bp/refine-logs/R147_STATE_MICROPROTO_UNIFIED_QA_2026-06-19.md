# R147 State Micro-Prototype Unified QA Branch

**Date**: 2026-06-19
**Status**: DONE-PARTIAL
**Task**: unified token bAbI QA prompt evaluation on `en-qa1` and `en-qa2`

## Purpose

R146 showed that the strict unified next-token QA evaluator works, but the
existing trace/NoProp/KV token memories collapse to answer priors.  The key
failure was not only held-out generalization: KV answer-only training reached
only `0.192` on `train_post`.

R147 tests whether the unified model needs a higher-capacity local state
readout before adding more structured recurrent binding.  The new branch is
still token-level and full-vocabulary:

- no task-specific QA head;
- no bAbI parser, entity table, role state, or symbolic tracker;
- no raw text storage or replay;
- local target/wrong-winner updates only.

## Implementation

Extended `babi_unified_token_qa_experiment.py` with:

- `OnlineStateMicroPrototypeMemory`
- `--method state_microproto_online`
- state/readout hyperparameters:
  - `--state-dim`
  - `--state-order`
  - `--state-decay`
  - `--micro-slots`
  - `--micro-lr`
  - `--micro-wrong-lr`
  - `--micro-score-scale`
  - `--micro-margin`

Mechanism:

1. Read the prompt as a recurrent random token/position state.
2. Score every compact-vocabulary token with a bounded set of local prototype
   slots per output token.
3. On the answer token, update only the target token's nearest/empty prototype
   and optionally depress the current wrong winner's nearest prototype.

This is closer to a neural micro-column/prototype memory than an n-gram table:
it stores learned state vectors, not raw strings or token-count contexts.

## Commands

Best QA1 slots64 seed pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --out-dir output/babi_unified_token_qa_enqa1_state_microproto_slots64_seed0 \
  --configs en-qa1 --max-vocab 512 \
  --method state_microproto_online \
  --state-dim 128 --state-order 128 --state-decay 0.90 \
  --micro-slots 64 --micro-lr 0.35 --micro-wrong-lr 0.02 \
  --micro-score-scale 9.0 --micro-margin 0.0 \
  --phase-bias-weight 1.0 \
  --answer-only-train --train-epochs 1 --seed 0
```

Repeated for seeds `1` and `2`, and also on `en-qa2`.

## Raw Data Table

QA1 slot sweep, seed 0:

| Slots | Train post acc | Val acc | Test acc | Test CE | State bytes |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.5822 | 0.4800 | 0.4350 | 1.7228 | 4,558,848 |
| 32 | 0.6956 | 0.5300 | 0.4740 | 1.7168 | 8,785,920 |
| 64 | 0.8467 | 0.5800 | 0.5080 | 1.7084 | 17,240,064 |
| 128 | 0.9656 | 0.5600 | 0.4880 | 1.7074 | 34,148,352 |

Three-seed slots64:

| Config | Split | Acc mean | Acc std | CE mean | CE std | State bytes |
|---|---|---:|---:|---:|---:|---:|
| QA1 | train_post | 0.8448 | 0.0087 | 1.5879 | 0.0197 | 17,240,064 |
| QA1 | validation | 0.5533 | 0.0205 | 1.7008 | 0.0098 | 17,240,064 |
| QA1 | test | 0.4997 | 0.0132 | 1.7115 | 0.0067 | 17,240,064 |
| QA2 | train_post | 0.4196 | 0.0069 | 1.6059 | 0.0066 | 17,240,064 |
| QA2 | validation | 0.1767 | 0.0236 | 1.8578 | 0.0035 | 17,240,064 |
| QA2 | test | 0.2020 | 0.0029 | 1.8032 | 0.0008 | 17,240,064 |

R146 comparison:

| Method | Config | Best full test acc |
|---|---|---:|
| NoProp answer-only 5 epochs | QA1 | 0.2080 |
| KV answer-only | QA1 | 0.1900 |
| state micro-prototype slots64 | QA1 | 0.4997 +/- 0.0132 |
| state micro-prototype slots64 | QA2 | 0.2020 +/- 0.0029 |

## Findings

1. A higher-capacity local state readout fixes the R146 train-memory failure on
   QA1.  `train_post` rises from KV's `0.192` to `0.845 +/- 0.009` at slots64.

2. QA1 held-out accuracy improves substantially but remains far below the old
   task-specific bAbI systems.  Test accuracy `0.500 +/- 0.013` is a real
   unified-token gain over R146, but not a solved QA result.

3. More slots are not monotonically better.  Slots128 memorizes train better
   (`0.966`) but has lower QA1 test accuracy than slots64 (`0.488` vs `0.508`),
   so capacity is already trading off with generalization.

4. QA2 remains near the weak baseline.  The model does not learn the object
   state transition "person carries object to new location" from raw prompt
   tokens.  This confirms the next bottleneck is structured state transition,
   not merely readout capacity.

## Decision

R147 is a partial positive:

- Keep `state_microproto_online` as the first viable unified-token QA branch.
- Do not claim bAbI reasoning.  It improves QA1 prompt-state recall but fails
  QA2 state transition.
- The next mechanism should add a token-level local binding/write-read circuit:
  a recurrent state that can bind entity-like token spans to location-like token
  spans and update those bindings through local eligibility, while still
  exposing only full-vocabulary next-token scores.

Suggested R148:

- Add a learned local binding branch with fixed random token keys/values.
- Update the branch continuously while reading prompt tokens, not only at
  answer positions.
- Use answer-token third-factor credit to calibrate the readout.
- Evaluate QA1/QA2 together and include train_post to separate memorization from
  held-out state transition.
