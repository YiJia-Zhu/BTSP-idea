# 2026-06-17 Temperature / Energy-Scale Audit For Bias-Free Apical Learner

R063 showed that feature-calibration gate decay gives a small no-statistical-prior CE gain. The next question is whether the remaining CE gap is mainly a score-scale problem. This round sweeps global readout temperature while keeping direct token-prior scoring disabled.

Important: this is an audit of energy scale, not a statistical token-probability model. It does not store token counts or raw text and does not use a BP-pretrained model. The next model step should replace this global temperature with a local/adaptive neural gain mechanism.

## Setup

Base method:

- `phase_trace_apical_inhib_competitive_online_feature_calib`
- random-feedback apical error
- adaptive inhibition
- feature calibration with derived fixed codes
- `--feature-calibration-gate-decay 0.50`
- direct prior disabled: `--phase-bias-weight 0.0`
- low precision: 8-bit row wrapper over vector/matrix/dynamic state

## Temperature Sweep

| temperature | post CE / acc | state bytes |
|---:|---:|---:|
| 0.5 | 2.456 / 0.476 | 713,033 |
| 0.6 | 2.302 / 0.476 | 713,033 |
| 0.7 | 2.262 / 0.476 | 713,033 |
| 0.8 | 2.294 / 0.476 | 713,033 |
| 1.0 | 2.475 / 0.476 | 713,033 |
| 1.3 | 2.846 / 0.476 | 713,033 |
| 1.6 | 3.202 / 0.476 | 713,033 |
| 2.0 | 3.589 / 0.476 | 713,033 |

Temperature does not change argmax predictions, so accuracy stays constant. CE has a clear optimum around `0.7`.

## Checkpointed Best Point

Command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_temp070_checkpoint_medium \
  --method-filter phase_trace_apical_inhib \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 --temperature 0.7 \
  --phase-bias-weight 0.0 \
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

| setting | post CE / acc | serialized bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|
| R063 temp1.0 gate_decay0.50 | 2.475 / 0.476 | 724,317 | 660,552 | 1.000 |
| R064 temp0.7 gate_decay0.50 | 2.262 / 0.476 | 724,317 | 660,552 | 1.000 |

Checkpoint metadata:

- `config_signature`: `1871571c8218cf5a2eed455551dedb76c0195630104fde31ed732d6d94d327e2`
- derived state: `calibration_codes`
- parity contexts: `1000`
- max score diff: `0.0`
- loss diff: `0.0`

## Interpretation

Positive:

- The bias-free no-BP learner beats the previous direct-prior CE point after energy-scale calibration:
  - direct-prior variable 8-bit R057: CE `2.295`, acc `0.438`
  - bias-free feature-calibrated R064: CE `2.262`, acc `0.476`
- This shows that the remaining CE issue was largely score scale, not a need for statistical token priors.
- The deployable state remains small: signed checkpoint `660,552` bytes.
- No raw text, no replay, no BP-pretrained backbone, and no token count probability table is used as the method.

Boundary:

- Global temperature is a readout calibration hyperparameter, not yet a local biological learning mechanism.
- Retention metrics at temperature `0.7` are mixed; this run optimizes stream CE, not all secondary metrics.
- Generation quality was not re-audited.

## Next Step

Turn this audit into a local neural mechanism:

1. adaptive output gain from WTA margin, branch agreement, or inhibition pressure
2. per-context or per-segment gain learned by local errors
3. generation/repetition rerun for the best bias-free checkpoint

Statistical token-probability methods remain only diagnostics, not final methods.
