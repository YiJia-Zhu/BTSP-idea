# R156 Local Span Arbitration

**Date**: 2026-06-19

## Goal

R155 showed that sparse span binding behaves like an anti-distractor branch:
it helps when distractors are present but hurts clean no-distractor settings.
R156 tests a narrow local arbitration mechanism that decides when to use span
binding from branch confidence, rather than always concatenating span features.

The mechanism remains pure no-BP. It trains two existing no-BP memories online
in parallel, a baseline recurrent trace and a span-binding branch. At readout,
a scalar gate derived from local score margins mixes the two score vectors. The
gate does not use labels at test time, does not use BP, and stores no raw text.

## Implementation

Updated `synthetic_object_carry_token_experiment.py`:

- added method `span_gate`;
- added `LocalSpanArbitrationMemory`;
- added gate args:
  - `--arbitration-gate-mode {hard_low_margin,soft_low_margin,soft_span_advantage}`;
  - `--arbitration-margin-threshold`;
  - `--arbitration-temperature`;
  - `--arbitration-span-gain`;
- added CSV diagnostics: `gate_rate`, `mean_gate_weight`,
  `mean_base_margin`, and `mean_span_margin`.

The tested gate is:

```text
if base_top1_minus_top2_margin < threshold:
    use span branch
else:
    use baseline branch
```

for `hard_low_margin`. Soft variants are implemented but were not expanded
after the hard gate showed the main boundary.

## Smoke

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile synthetic_object_carry_token_experiment.py
PYTHONDONTWRITEBYTECODE=1 python synthetic_object_carry_token_experiment.py \
  --out-dir output/synthetic_object_carry_token_r156_gate_smoke \
  --train-examples 50 --valid-examples 20 --test-examples 40 \
  --methods baseline span span_gate \
  --state-dim 24 --state-order 80 --micro-slots 4 \
  --micro-score-scale 8.0 --span-binding-hops 1 \
  --arbitration-margin-threshold 0.5 \
  --arbitration-temperature 0.25 --seed 0
```

The smoke run passed and wrote gate diagnostics to `summary.csv`.

## Small Threshold Sweep

Settings:

```bash
--train-examples 1200 --valid-examples 300 --test-examples 300
--state-dim 96 --state-order 96 --micro-slots 64
--micro-score-scale 9.0 --span-binding-hops 3
--arbitration-gate-mode hard_low_margin
```

| Setting | Method | Threshold | Val acc | Test acc | Test CE | Test gate rate |
|---|---|---:|---:|---:|---:|---:|
| `m2_d0` | baseline | - | 0.3867 | 0.3967 | 1.9484 | - |
| `m2_d0` | span | - | 0.3767 | 0.3667 | 1.9293 | - |
| `m2_d0` | span_gate | 0.02 | 0.4100 | 0.4267 | 1.9352 | 0.1867 |
| `m2_d0` | span_gate | 0.05 | 0.4400 | 0.4367 | 1.9255 | 0.4367 |
| `m2_d0` | span_gate | 0.20 | 0.4000 | 0.4200 | 1.9135 | 0.8900 |
| `m2_d0` | span_gate | 0.50 | 0.3767 | 0.3667 | 1.9293 | 1.0000 |
| `m2_d2` | baseline | - | 0.1600 | 0.1433 | 2.0577 | - |
| `m2_d2` | span | - | 0.2233 | 0.2000 | 2.0351 | - |
| `m2_d2` | span_gate | 0.02 | 0.1600 | 0.1567 | 2.0539 | 0.2100 |
| `m2_d2` | span_gate | 0.05 | 0.1833 | 0.1567 | 2.0526 | 0.3967 |
| `m2_d2` | span_gate | 0.20 | 0.2133 | 0.1967 | 2.0380 | 0.8900 |
| `m2_d2` | span_gate | 0.50 | 0.2233 | 0.2000 | 2.0351 | 1.0000 |

## Full-Scale Check

Selected threshold `0.20` as a compromise and reran the two key R155 settings
at `3000/500/500` examples with hop3 span.

| Setting | Method | Source | Train-post acc | Val acc | Test acc | Test CE | Gate rate | State bytes | Wall time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `m2_d0` | baseline | R155 | 0.7983 | 0.5880 | 0.5520 | 1.9345 | - | 793,584 | 64.0s |
| `m2_d0` | span | R155 hop2 | 0.6693 | 0.3800 | 0.3620 | 1.9451 | - | 2,268,144 | 104.2s |
| `m2_d0` | span_gate | R156 t0.20 hop3 | 0.7633 | 0.4560 | 0.4600 | 1.9108 | 0.9160 | 3,799,008 | 81.0s |
| `m2_d2` | baseline | R155 | 0.5103 | 0.1760 | 0.1580 | 2.0583 | - | 793,584 | 71.2s |
| `m2_d2` | span | R155 hop2 | 0.5867 | 0.2160 | 0.1980 | 2.0267 | - | 2,268,144 | 107.9s |
| `m2_d2` | span | R155 hop3 | 0.5827 | 0.2320 | 0.2080 | 2.0209 | - | 3,005,424 | 45.6s |
| `m2_d2` | span_gate | R156 t0.20 hop3 | 0.6180 | 0.2280 | 0.2120 | 2.0222 | 0.9480 | 3,799,008 | 83.8s |

## Findings

1. Margin arbitration is a mixed boundary, not a solved gate.
   On `m2_d2`, full-scale `span_gate` slightly improves top-1 over R155 hop3
   span (`0.2120` vs `0.2080`) but CE is slightly worse (`2.0222` vs `2.0209`).

2. The gate partially reduces the no-distractor span damage but does not protect
   the baseline. On `m2_d0`, full-scale `span_gate` improves over span
   (`0.4600` vs `0.3620`) but remains far below baseline (`0.5520`).

3. A single score-margin threshold cannot select the right branch across both
   regimes. In the small sweep, low thresholds protect `m2_d0` better but lose
   the `m2_d2` span gain; high thresholds recover `m2_d2` but collapse into
   always-span behavior. At threshold `0.20`, test gate rates are already
   `0.8900` on `m2_d0` and `0.9480` on full `m2_d2`.

4. The state/cost tradeoff is unfavorable for this gate. `span_gate` stores both
   memories and reaches about `3.80MB`, compared with `0.79MB` baseline and
   `3.01MB` hop3 span. The extra branch is only justified if arbitration is
   materially better, which this margin gate is not.

## Interpretation

The R155 hypothesis is partly confirmed: some arbitration can recover part of
the no-distractor damage while preserving part of the distractor benefit. But
top1-top2 margin is too weak as the biological control signal. The model is
low-margin in both regimes, so a scalar margin threshold cannot reliably tell
whether the prompt needs span binding.

Keep `span_gate` as a diagnostic mechanism, not as the next main architecture.
The next branch should use richer local signals: branch disagreement, recency
structure, event-cell competition, or an inhibitory event assembly that
explicitly detects carrier/object/location transitions.

## Next Step

Move from scalar margin gating to learned local event assemblies on the same
synthetic object-carry bench:

- WTA cells over local token windows;
- separate carrier-object and carrier-location event codes;
- eligibility traces from query object tokens;
- local inhibitory cleanup before full-vocab readout.

Do not port this scalar margin gate back to bAbI QA2 as the main solution.
