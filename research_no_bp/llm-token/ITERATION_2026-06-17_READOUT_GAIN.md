# 2026-06-17 Readout Gain As Model-Side Energy Calibration

R064 showed that global temperature `0.7` sharply improves CE for the bias-free feature-calibrated learner. This round moves that audit from the evaluator into the model readout by adding a no-BP readout gain wrapper.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper:

- `ReadoutGainMemory`

New CLI:

- `--readout-gain`
- `--readout-gain-mode {fixed,margin}`
- `--readout-gain-margin-center`
- `--readout-gain-margin-scale`
- `--readout-gain-min`
- `--readout-gain-max`

Modes:

- `fixed`: multiply scores by a constant gain. This is equivalent to lowering softmax temperature, but it is stored as model readout config and included in checkpoint signature.
- `margin`: compute a dynamic gain from current top1-top2 WTA score separation.

This stores no token counts or raw text. It is an energy/readout mechanism, not a statistical token-prior model.

## Results

Base learner:

- random-feedback apical + adaptive inhibition
- feature calibration with derived codes
- gate decay `0.50`
- direct token prior disabled: `--phase-bias-weight 0.0`
- eval temperature back to default `1.0`

| setting | post CE / acc | serialized bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|
| R064 evaluator temp0.7 | 2.262 / 0.476 | 724,317 | 660,552 | 1.000 |
| fixed readout gain 1.428571 | 2.262 / 0.476 | 724,317 | 660,634 | 1.000 |
| margin dynamic gain | 2.742 / 0.476 | 724,317 | n/a | n/a |

Fixed gain command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_fixedgain1429_checkpoint_medium \
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
  --readout-gain 1.4285714286 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 1000
```

Checkpoint metadata:

- `config_signature`: `3666e808f2753c6a8e8a6bf05778d4adfea24cbd2017dcfe07936c28dfeea904`
- state config includes:
  - `ReadoutGainMemory(gain=1.4285714286, mode=fixed)`
  - `FeatureConditionedCalibrationMemory(gate_decay=0.5, derived_codes=True)`

## Interpretation

Positive:

- The R064 temperature audit is now represented as model-side readout gain with identical CE/acc.
- It remains checkpointable with exact parity.
- Bias-free no-BP CE remains better than the direct-prior result: `2.262` vs `2.295`.
- The gain config adds negligible checkpoint overhead.

Negative boundary:

- The simple top1-top2 margin dynamic gain is harmful in this form: CE `2.742`.
- Current margin-based formula likely over-amplifies already confident wrong winners or under-amplifies ambiguous correct contexts.

## Next Step

Two routes are now clear:

1. Rerun generation/repetition for the best bias-free signed checkpoint.
2. Replace naive margin gain with a learned/local gain signal based on branch agreement, inhibition pressure, or recent calibration error rather than raw top1-top2 separation.

Sparse/continuation token probabilities remain diagnostics only.
