# 2026-06-17 Dynamic Loop-Pressure Inhibition

R069 added loop-pressure diagnostics, but its pressure gate acted only on plastic branch-agreement learning. This round tests whether a transient local inhibitory state can act directly during scoring and generation to suppress repeated-output attractors.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper:

- `LoopPressureInhibitionMemory`

New CLI:

- `--loop-inhibition`
- `--loop-inhibit-strength`
- `--loop-inhibit-activity-decay`
- `--loop-inhibit-pressure-decay`
- `--loop-inhibit-threshold`
- `--loop-inhibit-clip`
- `--loop-inhibit-repeat-gain`
- `--loop-inhibit-transition-strength`
- `--loop-inhibit-transition-decay`
- `--loop-inhibit-transition-threshold`
- `--loop-inhibit-transition-clip`
- `--loop-inhibit-transition-gain`

Mechanism:

- A decaying `loop_activity[token]` trace and `loop_pressure[token]` state estimate repeated winners.
- The score path subtracts `strength * max(loop_pressure - threshold, 0)`.
- A second transient state, `loop_transition_pressure[prev, token]`, tracks repeated local output transitions and subtracts transition-specific inhibition from the current context's last token row.
- During free generation the state is updated from predicted tokens; during stream evaluation it is updated from observed targets.
- Low-precision target group `dynamic` now includes `loop_activity`, `loop_pressure`, `loop_prev_output`, and `loop_transition_pressure`.

This is a local neural-state wrapper. It adds no replay buffer, raw-text storage, n-gram count table, statistical token prior, BP-trained backbone, or API/frozen-model dependence.

## Results

Base learner:

- random-feedback apical + adaptive inhibition
- feature calibration with derived codes
- fixed readout gain `1.428571`
- fixed low-variance branch agreement strength `0.10`
- plastic branch agreement strength `0.02`, lr `0.002`
- inhibition-pressure gate threshold `0.02`
- direct token prior disabled: `--phase-bias-weight 0.0`
- low-precision 8-bit row state for neural matrices/vectors

Medium seed0 sweep:

| setting | post CE | post acc | serialized bytes | greedy repeat-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|---:|
| R069 pressure-gated plastic | 2.253651762 | 0.481766821 | 725,853 | 0.382978723 | 0.085106383 |
| token loop inhibition s0.05 | 2.253777552 | 0.481766821 | 726,373 | 0.382978723 | 0.085106383 |
| token loop inhibition s0.10 | 2.253906513 | 0.481766821 | 726,373 | 0.382978723 | 0.085106383 |
| token loop inhibition s0.20 | 2.254173766 | 0.481766821 | 726,373 | 0.382978723 | 0.085106383 |
| transition loop inhibition s0.25 | 2.253640539 | 0.481766821 | 793,193 | 0.382978723 | 0.085106383 |
| transition loop inhibition s0.50 | 2.253632180 | 0.481766821 | 793,193 | 0.382978723 | 0.085106383 |
| transition loop inhibition s1.00 | 2.253623384 | 0.481766821 | 793,193 | 0.382978723 | 0.085106383 |

Smoke checkpoint for transition state:

| setting | post CE / acc | serialized bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|
| transition loop inhibition smoke | 1.817552 / 0.523013 | 347,465 | 320,110 | 1.000 |

The smoke checkpoint serialized `21` quantized arrays and `8` raw arrays, including the new loop pressure states, and restored exact predictions over 100 contexts.

Representative medium command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_transition_loop_s100_medium \
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
  --loop-inhibition --loop-inhibit-strength 0.0 \
  --loop-inhibit-transition-strength 1.0 \
  --loop-inhibit-transition-decay 0.80 \
  --loop-inhibit-transition-threshold 0.5 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic
```

## Interpretation

Positive:

- The implementation is fully online, local, checkpointable, and no-raw-data.
- The transition-state smoke checkpoint verifies that the new dynamic matrices can be serialized and reloaded with exact prediction parity.
- Transition inhibition produces a tiny CE improvement over R069: `2.253651762 -> 2.253623384`.

Boundary:

- Single-token loop inhibition is slightly harmful for CE and does not affect accuracy or generation repetition.
- Transition inhibition increases serialized state by about `67 KB` on the medium run while leaving greedy and controlled repeat-2 exactly unchanged.
- The mechanism mainly adjusts logit calibration around already-selected winners; it does not alter the free-running attractor enough to change generated token order.

## Next Step

Do not continue broad sweeps of token-level loop inhibition. The next mechanism should detect and act on higher-level context dynamics:

1. Context-feature-gated loop inhibition: inhibit only when branch agreement, inhibition pressure, and context features jointly predict a loop.
2. Segment-level attractor detector: maintain a compact recurrent state over recent generated features, not just tokens or token transitions.
3. Keep controlled decoding and statistical loop metrics as diagnostics only; the method itself must remain a pure no-BP neural/dynamic model.
