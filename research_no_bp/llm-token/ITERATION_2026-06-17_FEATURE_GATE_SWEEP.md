# 2026-06-17 Feature-Calibration Gate Sweep

R062 hardened the derived-code checkpoint with a config signature. This round sweeps the feature-calibration gate structure while keeping direct token-prior scoring disabled. The goal is to improve CE/byte using only local neural state, not statistical token probabilities.

## Setup

Base configuration:

- `phase_trace_apical_inhib_competitive_online`
- `--phase-bias-weight 0.0`
- random-feedback apical error
- adaptive inhibition
- feature calibration with derived codes
- 8-bit row low-precision wrapper over vector/matrix/dynamic state

The baseline for this sweep is R062:

- dim `64`
- threshold `0.0`
- gate decay `0.0`
- post CE / acc: `2.478 / 0.474`
- serialized bytes: `724,317`

## Sweep Results

| setting | changed parameter | post CE / acc | state bytes | serialized bytes | interpretation |
|---|---|---:|---:|---:|---|
| R062 baseline | dim64, thr0, gate_decay0 | 2.478 / 0.474 | 713,033 | 724,317 | signed derived checkpoint baseline |
| dim32 | `--feature-calibration-dim 32` | 2.480 / 0.476 | 704,809 | 716,093 | best bytes, small CE regression |
| dim128 | `--feature-calibration-dim 128` | 2.477 / 0.479 | 729,481 | 740,765 | best acc, small CE gain, more state |
| threshold0.10 | `--feature-calibration-threshold 0.10` | 2.478 / 0.477 | 713,033 | 724,317 | no meaningful CE gain |
| gate_decay0.50 | `--feature-calibration-gate-decay 0.50` | 2.475 / 0.476 | 713,033 | 724,317 | best CE without extra bytes |

## Checkpointed Best Point

Command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_gatedecay050_checkpoint_medium \
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
  --feature-calibration --feature-calibration-derived-codes \
  --feature-calibration-strength 1.5 \
  --feature-calibration-lr 0.03 --feature-calibration-decay 1.0 \
  --feature-calibration-clip 2.0 --feature-calibration-dim 64 \
  --feature-calibration-gate-decay 0.50 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 1000
```

Result:

| setting | post CE / acc | state bytes | serialized bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|---:|
| gate_decay0.50 signed checkpoint | 2.475 / 0.476 | 713,033 | 724,317 | 660,552 | 1.000 |

Checkpoint metadata:

- `config_signature`: `1871571c8218cf5a2eed455551dedb76c0195630104fde31ed732d6d94d327e2`
- `gate_decay`: `0.5`
- derived state: `calibration_codes`
- `calibration_codes` absent from checkpoint entries

## Interpretation

Positive:

- Gate decay gives a small CE improvement at no extra serialized-state cost.
- Dim32 is a reasonable lower-byte alternative: it saves about `8KB` serialized vs dim64 while losing only `0.002` CE.
- Dim128 gives the best top-1 accuracy (`0.479`) but costs about `16KB` extra serialized state.
- All tested points stay within the no-BP/no-raw-data route and keep statistical token probabilities disabled.

Boundary:

- The CE gain is small: `2.478 -> 2.475`.
- Direct-prior calibration remains much better on CE (`2.295`), so non-statistical calibration is still the bottleneck.
- Generation quality was not re-audited here.

## Next Step

The feature-calibration route is improving slowly. The next distinct mechanism should be tested rather than only sweeping this one:

1. local energy/temperature normalization from WTA score margins or branch agreement
2. optional combination with gate_decay0.50 feature calibration
3. generation/repetition rerun for the best bias-free checkpoint once CE improves more materially

Sparse/continuation token statistics remain diagnostics only.
