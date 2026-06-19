# R153 Event-Cell Branch for Unified QA

**Date**: 2026-06-19

## Goal

R152 showed that fixed query-seeded spreading does not learn QA2 object-location
state transitions.  R153 adds a parser-free learned event-cell branch:

- local token-window event features;
- fixed random event-cell keys;
- local WTA over event cells;
- eligibility-style event-cell value writes;
- optional target/wrong answer-token credit when the answer token appears inside
  the local event window;
- the same full-vocabulary next-token readout.

No bAbI parser, symbolic entity table, QA head, BP, raw replay, or test-answer
updates are added.

## Implementation

Updated `babi_unified_token_qa_experiment.py` with:

- `--event-cell-branch`
- `--event-cell-count`
- `--event-cell-window`
- `--event-cell-top-k`
- `--event-cell-lr`
- `--event-cell-credit-lr`
- `--event-cell-neg-lr`
- `--event-cell-query-weight`
- `--event-cell-recency-decay`

Each event feature is a local multiplicative code over token and relative
position codes.  Candidate cells are selected by a combination of fixed random
cell-key score, query-state similarity, and recency.  The selected cell values
are concatenated as an additional feature branch for the micro-prototype
readout.  During training, selected cells update their value vectors with local
event features and answer-token target/wrong credit.

## Main Command

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --out-dir output/babi_unified_token_qa_enqa2_state_microproto_event_cell_seed0 \
  --configs en-qa2 --max-vocab 512 \
  --method state_microproto_online \
  --state-dim 128 --state-order 128 --state-decay 0.90 \
  --micro-slots 64 --micro-lr 0.35 --micro-wrong-lr 0.02 \
  --micro-score-scale 9.0 --micro-margin 0.0 \
  --binding-query-order 8 --binding-query-mode prefix_overlap \
  --binding-focus-k 2 \
  --event-cell-branch --event-cell-count 64 \
  --event-cell-window 4 --event-cell-top-k 12 \
  --event-cell-lr 0.08 --event-cell-credit-lr 0.05 \
  --event-cell-neg-lr 0.03 --event-cell-query-weight 1.0 \
  --event-cell-recency-decay 0.98 \
  --phase-bias-weight 1.0 \
  --answer-only-train --train-epochs 1 --seed 0
```

## Raw Results

| Run | Train-post acc | Val acc | Test acc | Test CE | State bytes | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| R147 no binding | 0.4100 | 0.2100 | 0.1980 | 1.8026 | 17,240,064 | 4.3s |
| R150 sparse span | 0.6622 | 0.1700 | 0.1980 | 1.7988 | 50,794,496 | 9.8s |
| R152 prefix transition | 0.5467 | 0.1900 | 0.1900 | 1.8054 | 34,017,280 | 15.3s |
| R153 event cell | 0.4756 | 0.1300 | 0.1890 | 1.8012 | 34,087,680 | 64.9s |

## Key Findings

1. The event-cell branch does not improve QA2 held-out accuracy.
   Test accuracy drops to `0.1890`, below R147/R150's `0.1980`.

2. Event cells also reduce training recall relative to R150.
   Train-post accuracy is `0.4756`, far below R150's `0.6622`.

3. Runtime is too high for the benefit.
   Full seed0 takes `64.9s`, mainly from scanning token windows and scoring
   cell keys for every prompt.

## Interpretation

This event-cell design is still too weak.  It adds local WTA and eligibility
writes, but the cell objective is only window salience plus target-token
presence.  It does not discover stable roles such as carrier/person, object,
drop/pick event, current location, and previous location.  As a result, the
branch behaves like a slower local feature memory, not a reusable state
transition circuit.

## Next Step

Stop adding branches directly to full bAbI QA2 until the transition mechanism is
validated in a smaller controlled task.  The next aligned experiment should be a
parser-free synthetic object-carry task with raw token prompts and full-vocab
answer-token evaluation, where the only required skill is:

```text
object picked up by carrier -> carrier moves -> answer object's final location
```

This isolates whether a local event-cell/eligibility circuit can learn the
state transition before spending more runs on full bAbI QA2.
