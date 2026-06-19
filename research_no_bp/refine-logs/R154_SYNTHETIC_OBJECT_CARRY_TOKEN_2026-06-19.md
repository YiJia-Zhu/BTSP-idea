# R154 Synthetic Object-Carry Token QA

**Date**: 2026-06-19

## Goal

R153 showed that adding event-cell branches directly to full bAbI QA2 is too
noisy and slow.  R154 creates a smaller parser-free synthetic token task that
isolates the missing transition:

```text
object picked up by carrier -> carrier moves -> answer object's final location
```

The task is evaluated exactly like unified bAbI QA: raw prompt tokens in,
full-vocabulary next-token answer out.  There is no parser, symbolic entity
state, QA head, BP, raw replay, or test-answer update.

## Implementation

Added `synthetic_object_carry_token_experiment.py`.

Each example is a token prompt such as:

```text
Context: p5 at l2 . p3 at l0 . o3 at l0 . o1 at l0 .
p5 pick o3 . p5 move l1 . p3 move l6 .
p5 move l5 . p3 move l7 . o1 at l7 . o1 at l5 .
Question: where is o3 ? Answer:
```

The answer is the final location token of the carrier after the queried object
was picked up.  Distractor person moves and distractor object location statements
are placed after the relevant events, so simply copying the last location is not
enough.

The script reuses `OnlineStateMicroPrototypeMemory` from the unified bAbI token
evaluator and compares:

- `baseline`: recurrent state + micro-prototype readout;
- `span`: query-focused sparse span binding;
- `event_cell`: R153 event-cell branch;
- `span_event_cell`: span plus event cells.

## Commands

Main full run:

```bash
PYTHONDONTWRITEBYTECODE=1 python synthetic_object_carry_token_experiment.py \
  --out-dir output/synthetic_object_carry_token_r154_full \
  --train-examples 5000 --valid-examples 1000 --test-examples 1000 \
  --methods baseline span event_cell span_event_cell \
  --state-dim 96 --state-order 96 --micro-slots 64 \
  --micro-score-scale 9.0 \
  --event-cell-count 64 --event-cell-window 3 --event-cell-top-k 8 \
  --seed 0
```

Seed-repeat runs used the same settings with `--methods baseline span` and
`--seed 1` / `--seed 2`.

## Raw Results

Full seed0:

| Method | Train-post acc | Val acc | Test acc | Test CE | State bytes | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.4122 | 0.1510 | 0.1770 | 2.0556 | 793,584 | 8.6s |
| span | 0.4628 | 0.1970 | 0.2170 | 2.0224 | 2,268,144 | 22.2s |
| event_cell | 0.4066 | 0.1680 | 0.1520 | 2.0595 | 1,582,960 | 74.6s |
| span_event_cell | 0.4674 | 0.1880 | 0.2040 | 2.0306 | 3,057,520 | 131.2s |

Baseline vs span seed repeat:

| Seed | Baseline test acc | Span test acc | Delta | Baseline CE | Span CE |
|---|---:|---:|---:|---:|---:|
| 0 | 0.1770 | 0.2170 | +0.0400 | 2.0556 | 2.0224 |
| 1 | 0.1680 | 0.2180 | +0.0500 | 2.0599 | 2.0308 |
| 2 | 0.1850 | 0.2170 | +0.0320 | 2.0605 | 2.0265 |
| mean | 0.1767 | 0.2173 | +0.0407 | 2.0587 | 2.0266 |

## Key Findings

1. The synthetic task is not trivially solved by the baseline.
   Baseline test accuracy is `0.1767 +/- 0.0069`, above chance for 8 locations
   but far from solved.

2. Query-focused span binding gives a stable positive signal.
   Span improves test accuracy by `+0.0407` mean over three seeds and improves
   CE from `2.0587` to `2.0266`.  All three paired deltas are positive.

3. The current event-cell branch is still negative.
   On seed0, `event_cell` test accuracy is `0.1520`, and `span_event_cell`
   underperforms span alone (`0.2040` vs `0.2170`) while costing far more time.

## Interpretation

R154 is a controlled positive result.  It shows that the unified no-BP
span-binding branch can exploit object-carry structure in a parser-free raw
token task, even though it failed to generalize on full bAbI QA2.  The gap
suggests that bAbI QA2 adds enough surface variation, distractor structure, and
longer contexts that fixed span binding is insufficient.

The event-cell branch remains the wrong shape: local WTA windows and simple
answer-token credit do not yet create a useful carrier/object/location state.

## Next Step

Use this synthetic task as the development bench before returning to bAbI QA2.
The next useful experiment is an ablation over span mechanism details on R154:

- number of carrier moves and distractors;
- query focus mode;
- span window/top-k/hops;
- state order;
- held-out object/carrier/location combinations.

Only after the span or event-cell mechanism solves this controlled task robustly
should it be ported back to bAbI QA2.
