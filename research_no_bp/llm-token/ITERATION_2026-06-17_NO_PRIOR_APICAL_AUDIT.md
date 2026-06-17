# 2026-06-17 No-Direct-Prior Audit For Apical No-BP Learner

The project goal changed: BP-pretrained models cannot be the method backbone, and statistical token-probability methods can only be diagnostic auxiliaries. R057 made the variable-type no-BP state loadable. This round asks whether the current best phase/trace/apical learner depends on direct unigram/log-prior scoring.

## Audit

Existing knob:

- `--phase-bias-weight 0.0`

This disables direct addition of the learned `output_bias` term to model scores. The phase branches, trace branch, WTA readout, apical error gating, and adaptive inhibition remain active and locally updated.

Two runs were executed:

1. full precision bias-free audit
2. variable-type 8-bit row checkpoint audit with bias-free scoring

## Commands

Full precision:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_random_bias0_medium \
  --method-filter phase_trace_inhib phase_trace_apical_inhib \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 --phase-bias-weight 0.0 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --apical-gating-branch --apical-error-mode random_feedback \
  --apical-decay 0.85 --apical-strength 0.15 \
  --apical-margin 0.0 --apical-min-gate 0.8 \
  --apical-max-gate 1.25 --apical-error-clip 1.0 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1
```

Low-precision checkpoint:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_random_lowp8_varstate_bias0_medium \
  --method-filter phase_trace_inhib phase_trace_apical_inhib \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 --phase-bias-weight 0.0 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --apical-gating-branch --apical-error-mode random_feedback \
  --apical-decay 0.85 --apical-strength 0.15 \
  --apical-margin 0.0 --apical-min-gate 0.8 \
  --apical-max-gate 1.25 --apical-error-clip 1.0 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 1000
```

## Results

| setting | method | post CE / acc | state bytes | serialized bytes | checkpoint bytes | parity |
|---|---|---:|---:|---:|---:|---:|
| full precision, bias=1 | trace+inhib | 2.359 / 0.432 | 2,761,728 | 2,761,728 | n/a | n/a |
| full precision, bias=1 | apical+inhib | 2.289 / 0.437 | 2,761,752 | 2,761,752 | n/a | n/a |
| variable 8-bit, bias=1 | trace+inhib | 2.359 / 0.432 | 696,576 | 706,820 | 629,074 | 1.000 |
| variable 8-bit, bias=1 | apical+inhib | 2.295 / 0.438 | 696,585 | 706,841 | 635,091 | 1.000 |
| full precision, bias=0 | trace+inhib | 2.643 / 0.453 | 2,761,728 | 2,761,728 | n/a | n/a |
| full precision, bias=0 | apical+inhib | 2.513 / 0.463 | 2,761,752 | 2,761,752 | n/a | n/a |
| variable 8-bit, bias=0 | trace+inhib | 2.651 / 0.453 | 696,576 | 706,820 | 641,713 | 1.000 |
| variable 8-bit, bias=0 | apical+inhib | 2.523 / 0.462 | 696,585 | 706,841 | 642,633 | 1.000 |

Low-precision bias-free checkpoint parity:

- trace+inhib: 1000 contexts, pred match `1.000`, max score diff `0.0`, loss diff `0.0`
- apical+inhib: 1000 contexts, pred match `1.000`, max score diff `0.0`, loss diff `0.0`

## Interpretation

Positive:

- Removing direct log-prior scoring does not collapse the phase/trace/WTA/apical learner.
- Top-1 accuracy improves under bias-free scoring: apical+inhib `0.438 -> 0.462` in the variable 8-bit run.
- The apical mechanism remains useful and its CE advantage over trace+inhib grows under bias-free scoring: `0.064` CE with bias=1 vs `0.128` CE with bias=0.
- The loadable checkpoint path remains exact for bias-free scoring.

Boundary:

- CE calibration worsens when direct prior scoring is removed: variable 8-bit apical CE `2.295 -> 2.523`.
- This means the current learner's probability calibration still relies on a statistical prior-like component, even though winner selection can be driven by the no-BP neural state.
- Keeping `output_bias` as a disabled or diagnostic state is acceptable for analysis, but it cannot be the final method's calibration mechanism.

## Next Step

Replace direct token-frequency prior scoring with a neural/homeostatic alternative:

1. Add local output excitability or adaptive threshold dynamics that are updated from neuron activity rather than interpreted as token probability counts.
2. Compare `bias=0`, direct `output_bias`, and homeostatic calibration under the same apical random-feedback setup.
3. Keep sparse/context statistical methods only in auxiliary diagnostic tables.

This keeps the route aligned with the updated goal: a new no-BP biomimetic model framework, not a BP-pretrained model and not a final statistical token-probability model.
