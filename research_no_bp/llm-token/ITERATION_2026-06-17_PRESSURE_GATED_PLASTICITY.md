# 2026-06-17 Pressure-Gated Plastic Branch Agreement

R068 showed that plastic branch-agreement improves seed0 top-1 accuracy but does not improve generation loops. This round gates the plastic target/wrong update with local inhibition pressure and adds loop-pressure diagnostics to generation metrics.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

Extended wrapper:

- `PlasticBranchAgreementReadoutMemory`

New CLI:

- `--plastic-branch-agreement-pressure-mode {none,inhibition,context_loop,either,both}`
- `--plastic-branch-agreement-pressure-threshold`
- `--plastic-branch-agreement-pressure-gain`
- `--plastic-branch-agreement-loop-window`

New generation diagnostics:

- `max_run_fraction`
- `loop_pressure_mean`
- `loop_pressure_max`

Mechanism:

- `inhibition` mode finds the existing adaptive inhibitory interneuron state and gates plastic branch updates when the wrong winner is under learned inhibition pressure.
- `context_loop` mode gates updates when the current context window is locally repetitive.
- The default `pressure_mode=none` exactly preserves R068 behavior.

No raw text, replay buffer, context-count table, or statistical token prior is added.

## Results

Base learner:

- random-feedback apical + adaptive inhibition
- feature calibration with derived codes
- fixed readout gain `1.428571`
- fixed low-variance branch-agreement strength `0.10`
- plastic branch-agreement strength `0.02`, lr `0.002`
- direct token prior disabled: `--phase-bias-weight 0.0`
- low-precision 8-bit row state for neural matrices/vectors

Seed0:

| setting | post CE / acc | serialized bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|
| ungated plastic branch agreement | 2.254 / 0.481 | 725,853 | 662,607 | 1.000 |
| inhibition pressure t0.00 | 2.254 / 0.481 | 725,853 | n/a | n/a |
| inhibition pressure t0.02 | 2.254 / 0.482 | 725,853 | 662,662 | 1.000 |
| context-loop pressure t0.50 | 2.253 / 0.478 | 725,853 | n/a | n/a |

Seed repeats for inhibition pressure t0.02:

| seed | ungated plastic | pressure-gated plastic |
|---:|---:|---:|
| 1 | 2.369 / 0.462 | 2.368 / 0.462 |
| 2 | 2.257 / 0.474 | 2.256 / 0.475 |

Best checkpoint command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_pressure_plastic_inhib_t002_checkpoint_medium \
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
  --branch-agreement-readout --branch-agreement-strength 0.10 \
  --branch-agreement-mode low_variance --branch-agreement-clip 3.0 \
  --branch-agreement-variance-penalty 0.25 \
  --plastic-branch-agreement --plastic-branch-agreement-strength 0.02 \
  --plastic-branch-agreement-lr 0.002 --plastic-branch-agreement-top-k 1 \
  --plastic-branch-agreement-pressure-mode inhibition \
  --plastic-branch-agreement-pressure-threshold 0.02 \
  --plastic-branch-agreement-pressure-gain 1.0 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 1000
```

Checkpoint metadata:

- `config_signature`: `ff974435d6933525107813a9bcc2f81b6abe279b86de6f2fe86e8f26ab83a406`
- derived state: `calibration_codes`
- quantized/raw arrays: `17 / 8`
- exact checkpoint parity over 1000 contexts
- `PlasticBranchAgreementReadoutMemory(pressure_mode=inhibition, pressure_threshold=0.02, pressure_gain=1.0)`

Generation diagnostics:

| seed | decode | repeat-2 | distinct-2 | loop mean | loop max |
|---:|---|---:|---:|---:|---:|
| 0 | controlled | 0.085 | 0.915 | 0.180 | 0.500 |
| 0 | greedy | 0.383 | 0.617 | 0.196 | 0.500 |
| 1 | controlled | 0.163 | 0.837 | 0.177 | 0.500 |
| 1 | greedy | 0.355 | 0.645 | 0.191 | 0.500 |
| 2 | controlled | 0.149 | 0.851 | 0.182 | 0.500 |
| 2 | greedy | 0.376 | 0.624 | 0.198 | 0.500 |

## Interpretation

Positive:

- Inhibition-pressure gating gives the current best seed0 accuracy: `0.482`.
- CE also improves slightly over ungated plastic: `2.2545 -> 2.2537`.
- Seed1/seed2 retain the same direction, but the gain is very small.
- Checkpoint parity remains exact, with no extra serialized bytes over R068.
- Loop-pressure diagnostics are now emitted in generation summaries.

Boundary:

- Generation repetition is still not improved. Greedy repeat-2 stays high around `0.35-0.38`.
- Context-loop gated plasticity collapses back to fixed branch-agreement behavior: CE `2.253`, acc `0.478`.
- The pressure gate currently improves token ranking, not free-running language quality.

## Next Step

The next mechanism should act during generation dynamics, not only during online learning:

1. Add a dynamic loop-pressure inhibitory wrapper that suppresses recently repeated winners during generation and observation.
2. Keep it as local neural state, not a decoding-only statistical penalty.
3. Test whether dynamic loop inhibition improves greedy repetition without damaging CE/checkpoint parity.

Sparse/continuation token probabilities remain diagnostics only.
