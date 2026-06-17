# 2026-06-17 Loadable Integer Checkpoint For Apical No-BP Learner

R056 showed that the useful low-precision point is variable-type state: vectors/matrices use 8-bit row-scaled state, while count/prior-like arrays remain float32. This round turns that accounting result into a loadable checkpoint and verifies prediction parity.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New CLI:

- `--save-serialized-checkpoint`
- `--checkpoint-parity-limit N`

`LowPrecisionStateWrapper` now supports:

- `save_serialized_state(path)`: writes `.npz` checkpoint with `metadata_json`
- `load_serialized_state(path)`: restores the checkpoint into a fresh same-architecture memory object
- selected floating arrays: uniform symmetric integer codes plus scale metadata
- row-scaled 2D arrays: one float32 scale per row
- unselected count/prior arrays: raw float32 arrays

The parity check builds a fresh model with the same builder, loads the checkpoint, then compares reference-vs-loaded scores and argmax predictions on real validation next-token targets. No BP model, pretrained backbone, API output, or statistical memory baseline is used for the parity target.

## Command

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_random_lowp8_checkpoint_medium \
  --method-filter phase_trace_inhib phase_trace_apical_inhib \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 \
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

| method | post CE / acc | state bytes | serialized bytes | `.npz` bytes | q arrays / raw arrays | parity contexts | pred match | max score diff | loss diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace+inhib | 2.359 / 0.432 | 696,576 | 706,820 | 629,074 | 11 / 8 | 1000 | 1.000 | 2.38e-7 | -1.53e-10 |
| apical+inhib | 2.295 / 0.438 | 696,585 | 706,841 | 635,091 | 14 / 8 | 1000 | 1.000 | 2.38e-7 | 2.30e-11 |

Checkpoint paths:

- `output/phase_binding_online_stream_apical_random_lowp8_checkpoint_medium/phase_trace_inhib_competitive_online_serialized_state.npz`
- `output/phase_binding_online_stream_apical_random_lowp8_checkpoint_medium/phase_trace_apical_inhib_competitive_online_serialized_state.npz`

Smoke check:

- output: `output/phase_binding_online_stream_checkpoint_smoke/summary.csv`
- apical checkpoint file: 83,938 bytes
- parity over 80 contexts: pred match `1.000`, max score diff `7.15e-7`, loss diff `0.0`

## Interpretation

Positive:

- R056 is now a loadable state path, not only an accounting estimate.
- A fresh same-architecture no-BP learner restored from integer/float mixed checkpoint produces prediction-equivalent scores on held-out stream contexts.
- The checkpoint file is smaller than the raw serialized-byte estimate because `.npz` compression exploits structure in integer codes.
- The apical random-feedback advantage is preserved under this deployable path: post CE `2.295` vs trace+inhib `2.359`.

Boundary:

- `output_bias`, `prototype_counts`, and `unigram_counts` are still stored as float32 because R055 showed naive quantization damages CE. They are small, but direct log-prior use is still a statistical component that should not become the final method.
- The in-memory experiment wrapper still computes with float arrays after projection; the checkpoint proves loadability and equivalence, not a native integer inference kernel.
- This run does not improve generation semantics; it only hardens the storage/deployment story.

## Next Step

The next audit should directly address the updated final-goal constraint: reduce or remove direct token-probability priors from the main method. Concretely:

1. Run `--phase-bias-weight 0.0` on trace+inhib and apical+inhib to quantify dependence on unigram/log-prior bias.
2. If performance collapses, replace direct prior scoring with a neural/homeostatic alternative such as local output excitability, adaptive threshold, or inhibitory balance.
3. Keep sparse/continuation token statistics as diagnostic baselines only.

This remains within the pure no-BP route: no BP-pretrained backbone, no API learner, no raw-text replay, and no statistical token-probability model as the final method.
