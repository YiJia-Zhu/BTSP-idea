# 2026-06-17 Local Adaptive Readout Gain

R065 moved the best global temperature audit into a checkpointed fixed readout gain. This round tests whether that gain can become a context-local, online plastic neural mechanism rather than a fixed scalar.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper:

- `LocalAdaptiveReadoutGainMemory`

New CLI:

- `--local-readout-gain`
- `--local-readout-base-gain`
- `--local-readout-gain-strength`
- `--local-readout-gain-lr`
- `--local-readout-gain-decay`
- `--local-readout-gain-clip`
- `--local-readout-gain-min`
- `--local-readout-gain-max`
- `--local-readout-gain-dim`
- `--local-readout-gain-gate-decay`
- `--local-readout-gain-threshold`
- `--local-readout-gain-correct-margin`
- `--local-readout-gain-mistake-margin`
- `--local-readout-gain-derived-codes`

Mechanism:

- fixed random context gate, optionally derived from seed/architecture and not stored in checkpoint
- scalar gain synapse `gain_weights`
- correct WTA decision increases local energy; wrong WTA decision decreases local energy
- scores are multiplied by `base_gain * (1 + strength * tanh(gain_weights @ gate))`

This stores no token counts, token probabilities, or raw text. It is a local readout-energy modulation layer over the no-BP phase/trace/apical/inhibition learner.

## Results

Base learner:

- random-feedback apical + adaptive inhibition
- feature calibration with derived codes
- gate decay `0.50`
- direct token prior disabled: `--phase-bias-weight 0.0`
- low-precision 8-bit row state for neural matrices/vectors

| setting | post CE / acc | serialized bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|
| fixed gain 1.428571 | 2.262 / 0.476 | 724,317 | 660,634 | 1.000 |
| local gain base1.4286 strength0.15 lr0.005 | 2.276 / 0.476 | 724,389 | 661,627 | 1.000 |
| local gain base1.4286 strength0.25 lr0.010 | 2.326 / 0.476 | 724,389 | n/a | n/a |
| local gain base1.4286 strength0.50 lr0.020 | 2.593 / 0.476 | 724,389 | n/a | n/a |
| local gain base1.0 strength0.50 lr0.020 | 2.820 / 0.476 | 724,389 | n/a | n/a |

Best checkpoint command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_localgain_s015_lr005_checkpoint_medium \
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
  --local-readout-gain --local-readout-gain-derived-codes \
  --local-readout-base-gain 1.4285714286 \
  --local-readout-gain-dim 32 --local-readout-gain-strength 0.15 \
  --local-readout-gain-lr 0.005 --local-readout-gain-gate-decay 0.50 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 1000
```

Checkpoint metadata:

- `config_signature`: `ca7da2dd5c6c2bdb6141cb09dd7390a255a600e5136d7d3e338aaa5dab9d9372`
- derived state: `calibration_codes`, `gain_codes`
- quantized/raw arrays: `18 / 8`
- `LocalAdaptiveReadoutGainMemory(base_gain=1.4285714286, strength=0.15, lr=0.005, gate_dim=32, gate_decay=0.5)`

Generation/repetition:

- greedy post repeat-2 is unchanged versus fixed gain: `0.383`
- controlled post repeat-2 is also essentially unchanged: fixed `0.128`, local `0.135`

## Interpretation

Positive:

- A local plastic gain can closely approach the fixed global-gain calibration: CE `2.276` vs `2.262`.
- The extra deployable state is tiny: `+72` serialized bytes over fixed gain.
- Derived gain codes and exact checkpoint parity work.

Negative boundary:

- Stronger scalar local gain degrades CE, and learning from `base_gain=1.0` is much worse.
- This scalar gain cannot improve top-1 accuracy or greedy repetition, because multiplying all token scores by one local scalar does not change argmax order.

## Next Step

Stop tuning scalar gain as the main route. Keep fixed gain as the current best model-side energy calibration, and move to a local mechanism that can change winner ordering:

- branch-agreement gated readout boosts tokens supported by multiple dendritic branches
- inhibition-pressure conditioned anti-winner updates
- local calibration that uses recent winner errors per branch/output row, not global score scaling

Sparse/continuation token probabilities remain diagnostics only.
