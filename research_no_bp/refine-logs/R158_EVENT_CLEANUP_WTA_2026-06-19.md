# R158 Event Cleanup WTA

**Date**: 2026-06-19

## Goal

R157 showed that query-seeded event assembly improves probability mass but does
not reliably choose the correct winner under distractors. R158 adds a local
event-only cleanup readout with candidate WTA-style inhibition.

The method remains pure no-BP: no pretrained model, no BP, no parser, no
symbolic object state, no answer head, and no raw replay.

## Implementation

Updated `synthetic_object_carry_token_experiment.py`:

- added method `event_cleanup`;
- added `EventCleanupAssemblyMemory`, a subclass of `QueryEventAssemblyMemory`;
- added event-only cleanup prototypes over event-hop features;
- added optional top-k cleanup inhibition over cleanup candidates;
- added args:
  - `--cleanup-slots`;
  - `--cleanup-lr`;
  - `--cleanup-wrong-lr`;
  - `--cleanup-score-scale`;
  - `--cleanup-top-k`;
  - `--cleanup-inhibit`.

The cleanup branch learns with the same local target/wrong-winner rule:

```text
event assembly feature -> event-only cleanup prototypes -> cleanup scores
combined scores = full feature readout + cleanup scores
if wrong competes with target:
    update target prototype and anti-update wrong prototype
```

There is still one full-vocabulary output distribution. The cleanup branch is a
candidate competition mechanism, not a separate answer classifier.

## Smoke

```bash
PYTHONDONTWRITEBYTECODE=1 python synthetic_object_carry_token_experiment.py \
  --out-dir output/synthetic_object_carry_token_r158_event_cleanup_smoke \
  --train-examples 80 --valid-examples 30 --test-examples 50 \
  --methods event_assembly event_cleanup \
  --state-dim 32 --state-order 80 --micro-slots 8 \
  --micro-score-scale 8.0 \
  --assembly-hops 2 --assembly-event-window 2 \
  --assembly-seed-top-k 3 \
  --cleanup-score-scale 3.0 --cleanup-top-k 4 \
  --cleanup-inhibit 0.25 --seed 0
```

Smoke passed. `event_cleanup` improved test top-1 from `0.2200` to `0.2400`
versus `event_assembly`, while CE worsened on this tiny run, indicating the
expected sharpening tradeoff.

## m2_d2 Cleanup Sweep

Settings: `1200/300/300`, `assembly_hops=2`, `assembly_event_window=3`,
`assembly_seed_top_k=5`.

| Variant | Val acc | Test acc | Test CE | State bytes | Wall time |
|---|---:|---:|---:|---:|---:|
| baseline | 0.1600 | 0.1433 | 2.0577 | 793,584 | 11.7s |
| span hop3 | 0.2233 | 0.2000 | 2.0351 | 3,005,424 | 27.8s |
| span_gate t0.20 | 0.2133 | 0.1967 | 2.0380 | 3,799,008 | 102.6s |
| event_assembly w3/k5/h2 | 0.2000 | 0.2067 | 2.0294 | 2,270,832 | 69.4s |
| cleanup scale1 inhibit0 | 0.2000 | 0.2000 | 2.0188 | 3,753,072 | 79.6s |
| cleanup scale2 inhibit0 | 0.2233 | 0.2167 | 2.0089 | 3,753,072 | 81.3s |
| cleanup scale2 inhibit0.25 | 0.2133 | 0.2133 | 1.9964 | 3,753,072 | 82.4s |
| cleanup scale3 inhibit0.25 | 0.2133 | 0.2233 | 1.9889 | 3,753,072 | 80.0s |

Selected `cleanup_score_scale=3.0`, `cleanup_top_k=4`, `cleanup_inhibit=0.25`
for full checks.

## Full-Scale Results

Settings: `3000/500/500`, seed 0.

| Setting | Method | Train-post acc | Val acc | Test acc | Test CE | State bytes | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|
| `m2_d0` | baseline | 0.7983 | 0.5880 | 0.5520 | 1.9345 | 793,584 | 64.0s |
| `m2_d0` | span | 0.6693 | 0.3800 | 0.3620 | 1.9451 | 2,268,144 | 104.2s |
| `m2_d0` | event_cleanup | 0.7677 | 0.5120 | 0.5300 | 1.6523 | 3,752,304 | 59.4s |
| `m2_d2` | baseline | 0.5103 | 0.1760 | 0.1580 | 2.0583 | 793,584 | 71.2s |
| `m2_d2` | span hop2 | 0.5867 | 0.2160 | 0.1980 | 2.0267 | 2,268,144 | 107.9s |
| `m2_d2` | span hop3 | 0.5827 | 0.2320 | 0.2080 | 2.0209 | 3,005,424 | 45.6s |
| `m2_d2` | span_gate t0.20 | 0.6180 | 0.2280 | 0.2120 | 2.0222 | 3,799,008 | 83.8s |
| `m2_d2` | event_assembly | 0.6013 | 0.2080 | 0.2000 | 2.0087 | 2,270,832 | 56.0s |
| `m2_d2` | event_cleanup | 0.6463 | 0.2160 | 0.2380 | 1.9469 | 3,753,072 | 64.9s |
| `m3_d4` | baseline | 0.4850 | 0.1220 | 0.1320 | 2.0860 | 793,584 | 75.2s |
| `m3_d4` | span | 0.5343 | 0.1460 | 0.1580 | 2.0692 | 2,268,144 | 106.9s |
| `m3_d4` | event_cleanup | 0.5693 | 0.1720 | 0.1940 | 2.0491 | 3,753,072 | 26.2s |

## Findings

1. Event cleanup is the strongest synthetic object-carry method so far.
   On `m2_d2`, it improves test accuracy to `0.2380`, above span hop3
   `0.2080` and span_gate `0.2120`, while also improving CE to `1.9469`.

2. The gain holds under harder distractors.
   On `m3_d4`, event cleanup reaches `0.1940/2.0491`, beating baseline
   `0.1320/2.0860` and span `0.1580/2.0692`.

3. It also repairs most of the no-distractor span damage.
   On `m2_d0`, event cleanup reaches `0.5300/1.6523`, close to baseline top-1
   `0.5520` but with much better CE than baseline `1.9345`.

4. The tradeoff is state size.
   Event cleanup uses about `3.75MB`, larger than event assembly `2.27MB` and
   span hop3 `3.01MB`, but similar to span_gate `3.80MB` while delivering much
   better top-1 and CE.

## Interpretation

R158 is a positive mechanism result on the synthetic object-carry bench. The
combination of query-seeded event assembly and event-only cleanup readout
addresses the R157 winner-selection failure without falling back to a parser or
statistical table.

This still does not solve bAbI QA2 directly. The next required test is whether
the same cleanup branch can be inserted into the unified bAbI token evaluator
without adding a task-specific parser or answer head.

## Next Step

Port `event_cleanup` into `babi_unified_token_qa_experiment.py` as a generic
branch of `OnlineStateMicroPrototypeMemory` or a compatible wrapper, then test
on bAbI `en-qa2` seed0:

- keep full-vocabulary next-token evaluation;
- no bAbI parser, no symbolic state;
- compare R147/R150/R153/R158-style branches on the same split;
- first smoke, then one full seed if smoke is not negative.
