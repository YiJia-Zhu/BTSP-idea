# R161 Channel-Final Role Transition

**Date**: 2026-06-19

## Goal

R160 showed that parser-free local role traversal improves unified bAbI QA2.
R161 tests whether the transition path becomes more state-like when:

- role gates are separated by hop/channel;
- only the final hop contributes direct full-vocabulary role scores;
- optional top-k direct-score inhibition is tested as a candidate competition
  mechanism.

The method keeps the R160 constraints: no BP, no pretrained backbone, no bAbI
parser, no symbolic object state, no answer head, no structure labels, and no
raw replay.

## Implementation

Extended `OnlineLocalRoleTransitionMemory` in
`babi_unified_token_qa_experiment.py` with optional R161 parameters:

- `--role-channel-gates`: use one local gate vector per hop instead of one
  shared gate;
- `--role-final-score-only`: add direct role scores only from the last hop;
- `--role-score-top-k`: keep only top-k direct role-score candidates;
- `--role-score-inhibit`: apply negative direct-score inhibition to non-top-k
  traversed candidates.

Defaults preserve R160 behavior.

## QA2 Smoke

Settings: `en-qa2`, `60/60/60`, `max_vocab=256`, `state_dim=48`,
`state_order=128`, `micro_slots=8`, seed 0.

| Variant | Val acc | Test acc | Test CE | State bytes |
|---|---:|---:|---:|---:|
| R160 default rerun | 0.1167 | 0.2667 | 1.8014 | 1,265,536 |
| channel gates | 0.1333 | 0.2667 | 1.7925 | 1,265,728 |
| channel gates + final score only | 0.1167 | 0.2667 | 1.7889 | 1,265,728 |
| channel gates + final + top6 inhibit0.25 | 0.1333 | 0.2333 | 1.8145 | 1,265,728 |

Top-k inhibition is negative in smoke and was not expanded.

## QA2 Full Results

Settings: full `900/100/1000`, `max_vocab=256`, `state_dim=64`,
`state_order=224`, `micro_slots=64`, `role_hops=2`, `role_window=4`,
`role_top_k=6`, `role_score_scale=1.5`.

| Seed | R160 acc | R160 CE | R161 channel+final acc | R161 channel+final CE |
|---:|---:|---:|---:|---:|
| 0 | 0.2380 | 1.7612 | 0.2400 | 1.7566 |
| 1 | 0.2290 | 1.7574 | 0.2350 | 1.7494 |
| 2 | 0.2340 | 1.7572 | 0.2530 | 1.7450 |

Aggregate:

| Method | Test acc mean | Test acc std | Test CE mean | Test CE std |
|---|---:|---:|---:|---:|
| microproto baseline | 0.2003 | 0.0042 | 1.8045 | 0.0013 |
| R160 role-transition | 0.2337 | 0.0037 | 1.7586 | 0.0019 |
| R161 channel+final | 0.2427 | 0.0076 | 1.7504 | 0.0048 |

R161 paired deltas versus R160:

- accuracy: `+0.002`, `+0.006`, `+0.019`;
- CE: `-0.0046`, `-0.0080`, `-0.0121`.

## QA2 Seed0 Ablation

| Variant | Test acc | Test CE | Note |
|---|---:|---:|---|
| microproto | 0.1960 | 1.8048 | same-setting baseline |
| R160 default | 0.2380 | 1.7612 | previous best |
| channel gates only | 0.2300 | 1.7606 | lower top-1, slight CE gain |
| channel gates + final score only | 0.2400 | 1.7566 | best seed0 |

## QA3 Seed0 Pressure Test

Same settings as QA2, but `en-qa3`.

| Method | Train-post acc | Val acc | Test acc | Test CE | State bytes | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| microproto | 0.3856 | 0.1300 | 0.2190 | 1.7202 | 4,384,768 | 20.3 |
| R161 channel+final | 0.7978 | 0.3100 | 0.3440 | 1.7041 | 12,776,192 | 106.0 |

This is only a seed0 pressure test, but it shows that the R161 transition path
is not QA2-only.  Runtime grows about 5x on QA3.

## Findings

1. Hop/channel-specific gates plus final-hop direct scoring improve R160 across
   QA2 seeds 0/1/2.  The gain is modest but consistent: mean test accuracy
   `0.2337 -> 0.2427`, CE `1.7586 -> 1.7504`.

2. Direct top-k inhibition is currently harmful.  It likely suppresses useful
   weak candidates before the prompt-local traversal has a stable state.

3. QA3 seed0 also improves strongly over microproto: `0.2190 -> 0.3440` test
   accuracy.  This supports the transition-circuit direction, but does not yet
   prove robust multi-hop/before-location reasoning.

4. Cost is now a real bottleneck.  QA3 seed0 role-transition takes `106s`
   versus `20s` for microproto, with about `12.8MB` state.

## Interpretation

R161 strengthens the R160 conclusion.  Separating transition channels and
scoring only the final hop moves the unified no-BP QA model closer to a real
local state-transition circuit.  It is still shallow and prompt-local: it does
not yet maintain a reusable object/carrier/location state, and it remains far
from solving QA2/QA3.

The next useful run should be R162:

- add a center-difference diagnostic for the role-gate update direction;
- add runtime-oriented sparse event caching or candidate pruning that does not
  reproduce the harmful top-k inhibition;
- run QA3 seeds 1/2 only after the runtime issue is reduced;
- then test whether the same transition branch can help TinyStories-style token
  prediction or a small synthetic human-interference benchmark.
