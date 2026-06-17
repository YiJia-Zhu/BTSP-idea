# 2026-06-17 Homeostatic Output Calibration Audit

R058 showed that direct token-prior scoring is not acceptable as a final mechanism: it improves CE calibration, but it is a statistical token-probability component. This round tests a simple neural alternative: bounded output-neuron excitability.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper:

- `OutputHomeostasisMemory`

New CLI:

- `--output-homeostasis`
- `--homeostasis-strength`
- `--homeostasis-lr`
- `--homeostasis-decay`
- `--homeostasis-clip`

Rule:

- score adds `strength * excitability`
- on a current WTA error, target excitability increases and wrong-winner excitability decreases
- excitability decays and is clipped

This stores one local state per output neuron. It is not a token-probability table and does not store context counts or raw text.

## Results

All runs use `--phase-bias-weight 0.0` so direct log-prior scoring is disabled.

| setting | post CE / acc | state bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|
| bias-free apical, no homeostasis | 2.513 / 0.463 | 2,761,752 | n/a | n/a |
| homeostasis s=0.25 lr=0.005 | 2.513 / 0.463 | 2,762,776 | n/a | n/a |
| homeostasis s=0.50 lr=0.010 | 2.513 / 0.463 | 2,762,776 | n/a | n/a |
| homeostasis s=1.00 lr=0.020 | 2.513 / 0.465 | 2,762,776 | n/a | n/a |
| variable 8-bit homeostasis s=1.00 lr=0.020 | 2.524 / 0.468 | 696,841 | 643,258 | 1.000 |

Representative command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_homeo_s100_lr020_lowp8_checkpoint_medium \
  --method-filter phase_trace_apical_inhib \
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
  --output-homeostasis --homeostasis-strength 1.0 \
  --homeostasis-lr 0.02 --homeostasis-decay 0.995 \
  --homeostasis-clip 2.0 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 1000
```

## Interpretation

Negative:

- Simple global output excitability does not recover the CE gap left by removing direct `output_bias`.
- It slightly improves top-1 accuracy in the stronger setting, but calibration stays near the bias-free baseline.
- Therefore direct token prior cannot be replaced by a one-vector target-up/wrong-down homeostatic state.

Positive boundary:

- The state is small and survives the variable 8-bit checkpoint path.
- Loaded checkpoint parity remains exact over 1000 validation contexts.

## Next Step

The next non-statistical calibration mechanism should be context/feature-conditioned rather than global:

1. feature-conditioned adaptive thresholds over a compact random gate
2. local energy normalization across WTA competitors
3. confidence-temperature calibration from branch agreement or inhibition pressure

Do not return to sparse/continuation token probabilities as the final method. They remain diagnostics only.
