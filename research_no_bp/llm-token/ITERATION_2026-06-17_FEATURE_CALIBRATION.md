# 2026-06-17 Feature-Conditioned Calibration For Bias-Free Apical Learner

R058 showed that disabling direct token-prior scoring improves top-1 accuracy but hurts CE calibration. R059 showed that a global output excitability vector is too weak. This round tests a context/feature-conditioned neural calibration state instead of returning to statistical token probabilities.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper:

- `FeatureConditionedCalibrationMemory`

New CLI:

- `--feature-calibration`
- `--feature-calibration-strength`
- `--feature-calibration-lr`
- `--feature-calibration-decay`
- `--feature-calibration-clip`
- `--feature-calibration-dim`
- `--feature-calibration-gate-decay`
- `--feature-calibration-threshold`

Mechanism:

- a fixed random context encoder maps the current token context to a compact gate
- on a current WTA mistake, only target and wrong-winner calibration rows are updated from the gate
- scores add `strength * (calibration @ gate)`
- state is synaptic and local: `calibration_codes`, `calibration`, `calibration_gate`

This is not a context count table or token-probability prior. It stores fixed random neural codes and locally updated output calibration synapses.

## Commands

Representative full-precision positive point:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_s150_lr030_d64_medium \
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
  --feature-calibration --feature-calibration-strength 1.5 \
  --feature-calibration-lr 0.03 --feature-calibration-decay 1.0 \
  --feature-calibration-clip 2.0 --feature-calibration-dim 64
```

Low-precision checkpoint:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_s150_lr030_lowp8_checkpoint_medium \
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
  --feature-calibration --feature-calibration-strength 1.5 \
  --feature-calibration-lr 0.03 --feature-calibration-decay 1.0 \
  --feature-calibration-clip 2.0 --feature-calibration-dim 64 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 1000
```

## Results

All feature-calibration runs use `--phase-bias-weight 0.0`; direct token-prior scoring is disabled.

| setting | post CE / acc | state bytes | serialized bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|---:|
| R057 direct prior, variable 8-bit | 2.295 / 0.438 | 696,585 | 706,841 | 635,091 | 1.000 |
| R058 bias-free, variable 8-bit | 2.523 / 0.462 | 696,585 | 706,841 | 642,633 | 1.000 |
| R059 global homeostasis, variable 8-bit | 2.524 / 0.468 | 696,841 | 707,101 | 643,258 | 1.000 |
| feature calib s=0.5 lr=0.01 full precision | 2.507 / 0.464 | 3,876,120 | 3,876,120 | n/a | n/a |
| feature calib s=1.0 lr=0.02 full precision | 2.491 / 0.471 | 3,876,120 | 3,876,120 | n/a | n/a |
| feature calib s=1.5 lr=0.03 full precision | 2.473 / 0.476 | 3,876,120 | 3,876,120 | n/a | n/a |
| feature calib s=1.5 lr=0.03 variable 8-bit | 2.478 / 0.474 | 975,177 | 986,465 | 888,491 | 1.000 |

Checkpoint parity for the low-precision feature-calibration point:

- contexts: `1000`
- pred match: `1.000`
- max score diff: `0.0`
- loss diff: `0.0`

Serialized-state additions in the low-precision checkpoint:

| state | serialized bytes | raw bytes |
|---|---:|---:|
| `calibration_codes` | 262,148 | 1,048,576 |
| `calibration` | 17,408 | 65,536 |
| `calibration_gate` | 68 | 256 |

## Interpretation

Positive:

- Feature-conditioned neural calibration recovers part of the CE lost by removing direct `output_bias`: variable 8-bit CE improves `2.523 -> 2.478`.
- It also improves top-1 accuracy beyond both bias-free and direct-prior settings: variable 8-bit acc `0.474` vs bias-free `0.462` and direct-prior `0.438`.
- The effect scales monotonically over the tested strength/lr sweep.
- The state remains loadable through the mixed int8/float32 checkpoint path with exact parity.

Boundary:

- CE is still worse than the direct-prior run (`2.478` vs `2.295`), so probability calibration is not solved.
- State grows from about `0.707MB` serialized to about `0.986MB` serialized because fixed calibration codes are stored. A deterministic-code variant could avoid storing those codes.
- This only evaluates TinyStories 50k/10k medium; no broad generation-quality claim is made.

## Next Step

The next calibration step should keep the feature-conditioned route but reduce state and improve CE:

1. regenerate `calibration_codes` from seed instead of storing them, cutting about `262KB` from the checkpoint
2. sweep calibration dim/threshold/gate decay for better CE per byte
3. test energy/temperature calibration from branch agreement as an orthogonal non-statistical path

Sparse/continuation token statistics remain diagnostic baselines only, not the final method.
