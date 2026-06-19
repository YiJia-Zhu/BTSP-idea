# R157 Query Event Assembly

**Date**: 2026-06-19

## Goal

R156 showed that scalar score-margin arbitration is too weak to decide when
span binding should be trusted. R157 replaces scalar arbitration with a local
query-seeded event assembly: start from query tokens that reappear in the prompt
prefix, assemble local neighbor events, select next-hop seeds by neural
similarity, and feed the resulting event states into the same full-vocabulary
micro-prototype readout.

The method remains pure no-BP: no pretrained model, no BP, no parser, no
symbolic object state, no task-specific answer head, and no raw replay.

## Implementation

Updated `synthetic_object_carry_token_experiment.py`:

- added method `event_assembly`;
- added `QueryEventAssemblyMemory`;
- added args:
  - `--assembly-hops`;
  - `--assembly-event-window`;
  - `--assembly-seed-top-k`;
  - `--assembly-recency-decay`;
  - `--assembly-locality-decay`.

Mechanism:

1. Split the prompt into prefix and query tail using the existing query-order
   setting.
2. Choose initial seeds from query tokens that also occur in the prefix.
3. For each hop, build a local event state from seed-neighbor windows using
   fixed random token and relative-position codes.
4. Select next-hop seeds by token-code similarity to the event state.
5. Concatenate recurrent state plus event-hop states, then use local
   micro-prototype target/wrong-winner updates.

## Smoke

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile synthetic_object_carry_token_experiment.py
PYTHONDONTWRITEBYTECODE=1 python synthetic_object_carry_token_experiment.py \
  --out-dir output/synthetic_object_carry_token_r157_event_assembly_smoke \
  --train-examples 80 --valid-examples 30 --test-examples 50 \
  --methods baseline span event_assembly \
  --state-dim 32 --state-order 80 --micro-slots 8 \
  --micro-score-scale 8.0 --span-binding-hops 2 \
  --assembly-hops 2 --assembly-event-window 2 \
  --assembly-seed-top-k 3 --seed 0
```

Smoke test passed. `event_assembly` reached test acc/CE `0.2200/2.1953`
versus baseline `0.2000/2.2071` and span `0.1400/2.1588`.

## Medium Comparison

Settings: `1200/300/300`, `state_dim=96`, `state_order=96`, `micro_slots=64`,
`micro_score_scale=9.0`.

| Setting | Method | Source | Val acc | Test acc | Test CE | State bytes | Wall time |
|---|---|---|---:|---:|---:|---:|---:|
| `m2_d0` | baseline | R156 refs | 0.3867 | 0.3967 | 1.9484 | 793,584 | 8.3s |
| `m2_d0` | span hop3 | R156 refs | 0.3767 | 0.3667 | 1.9293 | 3,005,424 | 28.0s |
| `m2_d0` | span_gate t0.20 | R156 | 0.4000 | 0.4200 | 1.9135 | 3,799,008 | 102.0s |
| `m2_d0` | event_assembly w2/k3/h2 | R157 | 0.4533 | 0.4533 | 1.8092 | 2,270,064 | 20.1s |
| `m2_d2` | baseline | R156 refs | 0.1600 | 0.1433 | 2.0577 | 793,584 | 11.7s |
| `m2_d2` | span hop3 | R156 refs | 0.2233 | 0.2000 | 2.0351 | 3,005,424 | 27.8s |
| `m2_d2` | span_gate t0.20 | R156 | 0.2133 | 0.1967 | 2.0380 | 3,799,008 | 102.6s |
| `m2_d2` | event_assembly w2/k3/h2 | R157 | 0.2067 | 0.1700 | 2.0245 | 2,270,064 | 20.7s |

## m2_d2 Event Sweep

| Variant | Val acc | Test acc | Test CE |
|---|---:|---:|---:|
| w2/k3/h2 | 0.2067 | 0.1700 | 2.0245 |
| w3/k3/h2 | 0.1700 | 0.1733 | 2.0344 |
| w2/k5/h2 | 0.1900 | 0.1733 | 2.0191 |
| w3/k5/h2 | 0.2000 | 0.2067 | 2.0294 |
| w2/k3/h3 | 0.1667 | 0.1800 | 2.0015 |

## Full-Scale Check

Settings: `3000/500/500`, seed 0.

| Setting | Method | Source | Train-post acc | Val acc | Test acc | Test CE | State bytes | Wall time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `m2_d0` | baseline | R155 | 0.7983 | 0.5880 | 0.5520 | 1.9345 | 793,584 | 64.0s |
| `m2_d0` | span hop2 | R155 | 0.6693 | 0.3800 | 0.3620 | 1.9451 | 2,268,144 | 104.2s |
| `m2_d0` | span_gate t0.20/hop3 | R156 | 0.7633 | 0.4560 | 0.4600 | 1.9108 | 3,799,008 | 81.0s |
| `m2_d0` | event_assembly w2/k3/h2 | R157 | 0.7433 | 0.5080 | 0.5160 | 1.7958 | 2,270,064 | 49.1s |
| `m2_d2` | baseline | R155 | 0.5103 | 0.1760 | 0.1580 | 2.0583 | 793,584 | 71.2s |
| `m2_d2` | span hop2 | R155 | 0.5867 | 0.2160 | 0.1980 | 2.0267 | 2,268,144 | 107.9s |
| `m2_d2` | span hop3 | R155 | 0.5827 | 0.2320 | 0.2080 | 2.0209 | 3,005,424 | 45.6s |
| `m2_d2` | span_gate t0.20/hop3 | R156 | 0.6180 | 0.2280 | 0.2120 | 2.0222 | 3,799,008 | 83.8s |
| `m2_d2` | event_assembly w3/k5/h2 | R157 | 0.6013 | 0.2080 | 0.2000 | 2.0087 | 2,270,832 | 56.0s |

## Findings

1. Event assembly is a real CE improvement.
   On full `m2_d0`, it improves CE from baseline `1.9345` and span_gate
   `1.9108` to `1.7958`. On full `m2_d2`, it improves CE from span hop3
   `2.0209` and span_gate `2.0222` to `2.0087`.

2. Top-1 is still not solved under distractors.
   On full `m2_d2`, event assembly test acc is `0.2000`, below span_gate
   `0.2120` and span hop3 `0.2080`, despite better CE. This means the event
   state improves probability mass but does not yet cleanly select the winner.

3. The clean-transition result is better than scalar arbitration.
   On full `m2_d0`, event assembly reaches `0.5160`, recovering much more of
   the baseline `0.5520` than span_gate `0.4600`, while keeping much lower CE.

4. State efficiency improves versus span_gate.
   Full event assembly uses about `2.27MB`, compared with `3.80MB` for
   span_gate. It is close to span hop2 in state size but produces much better CE
   on both tested full settings.

## Interpretation

R157 is a partial positive. Query-seeded local event assembly is a better
mechanistic direction than scalar margin arbitration because it creates useful
event features and improves CE without requiring two full memories. However,
the winner selection problem remains: distractors still cause the final
argmax answer token to be unstable.

The next missing component is not a wider local window; the sweep shows that
larger windows and more seeds trade CE against top-1. The next step should be a
local inhibitory cleanup or WTA candidate layer over location-like answer
candidates, driven by the event assembly state, before the full-vocab readout.

## Next Step

Add an event-assembly candidate cleanup branch:

- derive a small set of candidate output tokens from event-state/token-code
  similarity;
- apply local WTA or anti-winner inhibition among candidates;
- train with the same target/wrong local third-factor update;
- keep the output full-vocabulary, with no parser and no answer head.

Only if this improves `m2_d2` top-1 without losing the CE gain should it be
ported back to bAbI QA2.
