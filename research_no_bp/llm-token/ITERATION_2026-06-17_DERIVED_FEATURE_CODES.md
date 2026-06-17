# 2026-06-17 Derived Feature-Calibration Codes

R060 showed that feature-conditioned neural calibration recovers part of the CE loss caused by removing direct token-prior scoring. The main cost was storing fixed random `calibration_codes` in the checkpoint. This round makes those codes derived from architecture and seed rather than stored state.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New option:

- `--feature-calibration-derived-codes`

When enabled:

- `FeatureConditionedCalibrationMemory` marks `calibration_codes` as derived state
- `LowPrecisionStateWrapper` excludes derived arrays from `state_bytes`, `serialized_state_bytes`, manifest, and `.npz` checkpoint entries
- fresh model construction regenerates the same codes from seed before loading learned state
- checkpoint metadata records `derived_state_names=["calibration_codes"]`

This keeps fixed random neural codes as part of the architecture, not deployable learned memory.

## Command

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_s150_lr030_derived_lowp8_checkpoint_medium \
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
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 1000
```

## Results

All rows use direct prior disabled (`--phase-bias-weight 0.0`) except R057.

| setting | post CE / acc | state bytes | serialized bytes | checkpoint bytes | q/raw arrays | parity |
|---|---:|---:|---:|---:|---:|---:|
| R057 direct prior, variable 8-bit | 2.295 / 0.438 | 696,585 | 706,841 | 635,091 | 14 / 8 | 1.000 |
| R058 bias-free, variable 8-bit | 2.523 / 0.462 | 696,585 | 706,841 | 642,633 | 14 / 8 | 1.000 |
| R060 feature calibration, stored codes | 2.478 / 0.474 | 975,177 | 986,465 | 888,491 | 17 / 8 | 1.000 |
| R061 feature calibration, derived codes | 2.478 / 0.474 | 713,033 | 724,317 | 660,259 | 16 / 8 | 1.000 |

Parity for R061:

- contexts: `1000`
- pred match: `1.000`
- max score diff: `0.0`
- loss diff: `0.0`

Manifest/checkpoint audit:

- `calibration_codes` is absent from the serialized manifest
- checkpoint metadata reports `derived_state_names=["calibration_codes"]`
- `calibration` remains stored as learned local synaptic state

## Interpretation

Positive:

- Derived fixed codes reduce serialized state by `262,148` bytes with no metric or parity change.
- The feature-calibration improvement over bias-free scoring is retained: CE `2.523 -> 2.478`, acc `0.462 -> 0.474`.
- The deployable checkpoint is now `660,259` bytes, close to the bias-free checkpoint `642,633` bytes while preserving the added neural calibration benefit.
- This keeps random context features as architecture/seed, not learned memory.

Boundary:

- CE is still behind the direct-prior version (`2.478` vs `2.295`), so calibration remains incomplete.
- The approach assumes deterministic seed/architecture compatibility at load time; checkpoint metadata should be extended with a config hash before a true standalone deployment artifact.
- Generation quality was not re-audited in this round.

## Next Step

1. Add config/hash metadata to the checkpoint so derived-state loading can detect seed or architecture mismatch.
2. Sweep feature-calibration dimension, threshold, and gate decay now that fixed-code storage no longer dominates bytes.
3. Test local energy/temperature normalization as a second non-statistical calibration route.

Sparse/continuation token statistics remain diagnostic baselines only, not the final method.
