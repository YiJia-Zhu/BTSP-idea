# R155 Synthetic Object-Carry Ablation

**Date**: 2026-06-19

## Goal

R154 found a controlled positive signal for query-focused sparse span binding on
a parser-free synthetic object-carry next-token QA task. R155 tests whether that
signal is robust across task difficulty and whether it depends on the number of
span-binding hops.

This remains a diagnostic bench for a pure no-BP neural mechanism. It is not a
statistical n-gram baseline, does not use CLUTRR, does not use pretrained
weights, and evaluates the answer as a full-vocabulary next-token prediction.

## Local Data Status

CLUTRR is not present under `data/`. The currently available local datasets are:

| Dataset | Local path | Current use |
|---|---|---|
| TinyStories | `data/TinyStories/` | token-level language modeling / online learning |
| bAbI processed | `data/babi_qa_processed/` | QA evaluation and training; configs `en-qa1/2/3/14/15/16/17/18/19` |
| GSM8k-Aug | `data/GSM8k-Aug/` | later-stage math QA pressure test, not suitable as the immediate no-BP training target |

The synthetic object-carry task is generated locally by
`synthetic_object_carry_token_experiment.py` so that the object-carrier-location
transition can be isolated before returning to full bAbI QA2.

## Implementation

Added `--span-binding-hops` to `synthetic_object_carry_token_experiment.py`.
For `span` and `span_event_cell` variants this controls the number of sparse
span-binding hops passed into `OnlineStateMicroPrototypeMemory`; baseline and
event-only variants keep binding disabled.

Smoke command passed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile synthetic_object_carry_token_experiment.py
PYTHONDONTWRITEBYTECODE=1 python synthetic_object_carry_token_experiment.py \
  --out-dir output/synthetic_object_carry_token_r155_hops_smoke \
  --train-examples 40 --valid-examples 20 --test-examples 40 \
  --methods baseline span --span-binding-hops 1 \
  --state-dim 24 --state-order 80 --micro-slots 4 --seed 0
```

## Main Settings

All difficulty runs used:

```bash
PYTHONDONTWRITEBYTECODE=1 python synthetic_object_carry_token_experiment.py \
  --train-examples 3000 --valid-examples 500 --test-examples 500 \
  --methods baseline span \
  --state-dim 96 --state-order 96 --micro-slots 64 \
  --micro-score-scale 9.0 \
  --span-binding-hops 2 --seed 0
```

The four difficulty settings were:

| Setting | Carrier moves | Extra distractors |
|---|---:|---:|
| `m1_d0` | 1 | 0 |
| `m2_d0` | 2 | 0 |
| `m2_d2` | 2 | 2 |
| `m3_d4` | 3 | 4 |

## Raw Results

| Setting | Method | Train-post acc | Val acc | Test acc | Test CE | State bytes | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|
| `m1_d0` | baseline | 0.7663 | 0.5440 | 0.5600 | 1.9280 | 793,584 | 43.6s |
| `m1_d0` | span | 0.7057 | 0.4080 | 0.4200 | 1.9064 | 2,268,144 | 97.1s |
| `m2_d0` | baseline | 0.7983 | 0.5880 | 0.5520 | 1.9345 | 793,584 | 64.0s |
| `m2_d0` | span | 0.6693 | 0.3800 | 0.3620 | 1.9451 | 2,268,144 | 104.2s |
| `m2_d2` | baseline | 0.5103 | 0.1760 | 0.1580 | 2.0583 | 793,584 | 71.2s |
| `m2_d2` | span | 0.5867 | 0.2160 | 0.1980 | 2.0267 | 2,268,144 | 107.9s |
| `m3_d4` | baseline | 0.4850 | 0.1220 | 0.1320 | 2.0860 | 793,584 | 75.2s |
| `m3_d4` | span | 0.5343 | 0.1460 | 0.1580 | 2.0692 | 2,268,144 | 106.9s |

Paired test deltas for span minus baseline:

| Setting | Test acc delta | Test CE delta |
|---|---:|---:|
| `m1_d0` | -0.1400 | -0.0216 |
| `m2_d0` | -0.1900 | +0.0105 |
| `m2_d2` | +0.0400 | -0.0316 |
| `m3_d4` | +0.0260 | -0.0167 |

Hop sweep on the default hard setting `m2_d2`:

| Span hops | Train-post acc | Val acc | Test acc | Test CE | State bytes | Wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5550 | 0.1840 | 0.2040 | 2.0472 | 1,530,864 | 31.9s |
| 2 | 0.5867 | 0.2160 | 0.1980 | 2.0267 | 2,268,144 | 107.9s |
| 3 | 0.5827 | 0.2320 | 0.2080 | 2.0209 | 3,005,424 | 45.6s |

## Findings

1. Span binding is not a universal improvement. On the no-distractor settings
   it lowers top-1 accuracy sharply: `0.5600 -> 0.4200` for `m1_d0` and
   `0.5520 -> 0.3620` for `m2_d0`. The baseline can exploit the short, clean
   carrier-location trace; span reads add competing candidates.

2. Span binding becomes useful when distractors are present. On `m2_d2`, test
   accuracy improves `0.1580 -> 0.1980` and CE improves `2.0583 -> 2.0267`.
   On `m3_d4`, test accuracy improves `0.1320 -> 0.1580` and CE improves
   `2.0860 -> 2.0692`.

3. More hops help only modestly on the hard setting. Hop3 gives the best tested
   `m2_d2` result, with test acc/CE `0.2080/2.0209`, but this is still far from
   solving the transition.

## Interpretation

Sparse span binding is best interpreted as an anti-distractor context-binding
branch, not as a solved object-state transition mechanism. It helps when the
prompt contains misleading later events, but hurts when a simpler recurrent
trace is already sufficient.

The mechanism therefore needs a local arbitration/gating circuit before being
ported back to full bAbI QA2. A plausible next step is not another fixed span
variant, but a learned local confidence gate or WTA event assembly that decides
when to trust carrier/object/location binding versus the base recurrent trace.

## Next Step

Stay on the synthetic object-carry bench and test:

- local confidence arbitration between baseline trace and span branch;
- carrier-object-location event assemblies with winner-take-all cleanup;
- held-out carrier/object/location combinations to separate memorization from
  compositional transition learning;
- only after a robust synthetic result, port the mechanism back to bAbI QA2.
