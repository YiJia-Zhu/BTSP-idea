# R151 Learned Span Event Gate for Unified QA

**Date**: 2026-06-19

## Goal

R150 showed that fixed sparse span binding is fast but only improves training
recall, not held-out QA2.  R151 tests whether a local answer-credit gate over
span events can make sparse binding selective enough to improve QA2 without a
task-specific parser, QA head, symbolic state, raw replay, or BP.

## Implementation

Updated `babi_unified_token_qa_experiment.py` with optional learned span gates:

- `--binding-span-learned-gate`
- `--binding-span-gate-lr`
- `--binding-span-gate-neg-lr`
- `--binding-span-gate-strength`
- `--binding-span-gate-clip`

For each sparse span event, the gate feature is a local multiplicative binding
of:

```text
seed token code * neighbor token code * distance code
```

During answer-token training, if the target answer token appears as a span
neighbor, the event gate is reinforced.  If the current wrong winner appears as
a span neighbor, the event gate is suppressed.  This is a local target/wrong
third-factor update over event features.  Test rows are never used for updates.

The default remains unchanged: learned gates are disabled unless explicitly
requested.

## Main Command

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --out-dir output/babi_unified_token_qa_enqa2_state_microproto_span_gate_seed0 \
  --configs en-qa2 --max-vocab 512 \
  --method state_microproto_online \
  --state-dim 128 --state-order 128 --state-decay 0.90 \
  --micro-slots 64 --micro-lr 0.35 --micro-wrong-lr 0.02 \
  --micro-score-scale 9.0 --micro-margin 0.0 \
  --binding-hops 2 --binding-query-order 8 \
  --binding-query-mode prefix_overlap --binding-focus-k 2 \
  --binding-mode span_sparse --binding-span-window 6 \
  --binding-span-top-k 4 --binding-span-decay 0.95 \
  --binding-span-learned-gate \
  --binding-span-gate-lr 0.05 --binding-span-gate-neg-lr 0.05 \
  --binding-span-gate-strength 0.75 --binding-span-gate-clip 2.0 \
  --phase-bias-weight 1.0 \
  --answer-only-train --train-epochs 1 --seed 0
```

## Raw Results

| Run | Train-post acc | Val acc | Test acc | Test CE | State bytes | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| R149 low-rank focused bind2 | 0.5167 | 0.1900 | 0.1990 | 1.8013 | 50,794,496 | 107.9s |
| R150 sparse span bind2 | 0.6622 | 0.1700 | 0.1980 | 1.7988 | 50,794,496 | 9.8s |
| R151 learned gate default | 0.6556 | 0.1400 | 0.1960 | 1.7977 | 50,798,080 | 40.3s |
| R151 weak gate | 0.6556 | 0.1700 | 0.1910 | 1.7968 | 50,798,080 | 110.4s |
| R151 positive-only gate | 0.6678 | 0.1600 | 0.1950 | 1.7962 | 50,798,080 | 107.5s |

## Key Findings

1. Learned event gates slightly improve CE but hurt top-1 accuracy.
   The best CE probe reaches `1.7962`, but test accuracy drops to `0.1950`.

2. Target-only reinforcement does not fix the failure.
   Removing wrong-token suppression gives higher train recall (`0.6678`) but
   still lower held-out accuracy than R150.

3. The gate adds runtime without useful held-out gain.
   The extra event feature/gate computation raises wall time from `9.8s` to
   `40-110s`, depending on probe settings.

## Interpretation

R151 rules out a naive answer-token span gate: simply reinforcing span events
that contain the final answer token does not produce a reusable object-location
state transition.  The model can sharpen local prompt features and CE slightly,
but the top-1 answer remains governed by surface priors and train prompt
memorization.

The missing mechanism is more structured than a scalar event salience gate.  It
likely needs a latent state update with separate roles for carrier/person,
object, source location, destination location, and recency, but this must be
learned as a neural/local circuit rather than reintroducing a bAbI parser.

## Next Step

Stop tuning scalar span gates.  The next aligned experiment should add a
parser-free latent transition branch: local candidate event cells compete over
token spans, write a compact object/carrier/location state through eligibility
traces, and expose only the resulting state vector to the same full-vocabulary
next-token readout.
