# 2026-06-17 Checkpoint Config Signature For Derived State

R061 removed fixed `calibration_codes` from the deployable checkpoint by regenerating them from seed/architecture. That makes the checkpoint smaller, but it also creates a new safety requirement: loading must reject seed or architecture mismatches instead of silently using incompatible derived codes.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

`LowPrecisionStateWrapper` now writes:

- `state_config`: model-provided structural metadata
- `config_signature`: SHA-256 over low-precision settings, derived state names, state config, and checkpoint entry schema

`FeatureConditionedCalibrationMemory` now exposes `state_config_metadata()` with:

- strength/lr/decay/clip
- gate decay/threshold
- seed offset
- derived-code flag
- max order, vocab size, gate dim
- calibration-code shape

`load_serialized_state()` recomputes the signature on the fresh model and rejects mismatch before loading arrays.

## Smoke Checks

Positive smoke:

- output: `output/phase_binding_online_stream_feature_calib_signature_smoke/summary.csv`
- parity contexts: `80`
- pred match: `1.000`
- score diff: `0.0`

Negative smoke:

The checkpoint metadata was deliberately modified to set:

```text
config_signature = 0000000000000000000000000000000000000000000000000000000000000000
```

Loading into an otherwise matching fresh model raises:

```text
ValueError: checkpoint config signature mismatch
```

## Medium Result

Command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_signature_lowp8_checkpoint_medium \
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

| setting | post CE / acc | state bytes | serialized bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|---:|
| R061 derived feature calibration | 2.478 / 0.474 | 713,033 | 724,317 | 660,259 | 1.000 |
| R062 signed checkpoint | 2.478 / 0.474 | 713,033 | 724,317 | 660,561 | 1.000 |

R062 metadata:

- `config_signature`: `6ad1a0423986f5c9eddb8a32a75a5e3b1983b5167dadb1ce39a9110ae2fcf6bb`
- `derived_state_names`: `["calibration_codes"]`
- `state_config`: feature-calibration config with `seed_offset=32452843`, `gate_dim=64`, `max_order=16`, `vocab_size=256`
- `calibration_codes` is absent from checkpoint entries

## Interpretation

Positive:

- Derived-state checkpoint loading is now guarded against architecture/seed mismatch.
- The signature adds negligible size: checkpoint `660,259 -> 660,561` bytes.
- Prediction parity remains exact over 1000 validation contexts.
- The checkpoint remains no-raw-data and no-BP: only learned no-BP state plus metadata is stored.

Boundary:

- The signature protects exact config compatibility, not semantic versioning across future code changes.
- It does not solve CE calibration beyond R061; it hardens deployment correctness.

## Next Step

Now that derived-code checkpoints are safer, the next algorithmic step should be a sweep of feature-calibration gate structure:

1. gate dimension vs CE/bytes
2. gate threshold sparsification
3. gate decay for short-term contextual calibration
4. local energy/temperature normalization as a separate non-statistical calibration path

Sparse/continuation token statistics remain diagnostic baselines only, not the final method.
