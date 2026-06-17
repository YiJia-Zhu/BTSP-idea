# 2026-06-17 Branch-Agreement Readout

R066 showed that scalar local gain is useful for calibration but cannot change argmax. This round adds a token-wise dendritic branch-agreement readout so the model can change winner ordering without statistical token probabilities or BP.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper:

- `BranchAgreementReadoutMemory`

New CLI:

- `--branch-agreement-readout`
- `--branch-agreement-strength`
- `--branch-agreement-mode {mean_min,positive_fraction,low_variance,min}`
- `--branch-agreement-clip`
- `--branch-agreement-threshold`
- `--branch-agreement-variance-penalty`

Mechanism:

- locate the wrapped phase `branch_model`
- compute each branch's current prototype support for every token
- normalize each branch support vector by its own mean/std
- add a token-wise agreement signal to final scores
- keep fixed readout gain as the model-side energy calibration from R065

This uses existing neural branch state only. It stores no raw text, no context-count rows, and no token probability tables.

## Results

Base learner:

- random-feedback apical + adaptive inhibition
- feature calibration with derived codes
- fixed readout gain `1.428571`
- direct token prior disabled: `--phase-bias-weight 0.0`
- low-precision 8-bit row state for neural matrices/vectors

| setting | post CE / acc | serialized bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|
| fixed gain baseline | 2.262 / 0.476 | 724,317 | 660,634 | 1.000 |
| mean_min strength0.05 | 2.258 / 0.476 | 724,317 | n/a | n/a |
| mean_min strength0.15 | 2.254 / 0.479 | 724,317 | n/a | n/a |
| positive_fraction strength0.10 | 2.254 / 0.477 | 724,317 | n/a | n/a |
| low_variance strength0.10 | 2.253 / 0.478 | 724,317 | 660,721 | 1.000 |

Seed1 repeat:

| setting | post CE / acc | serialized bytes |
|---|---:|---:|
| fixed gain baseline seed1 | 2.372 / 0.460 | 724,317 |
| low_variance strength0.10 seed1 | 2.370 / 0.462 | 724,317 |

Best CE checkpoint command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_branchagree_lowvar_s010_checkpoint_medium \
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
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 1000
```

Checkpoint metadata:

- `config_signature`: `98d3d0b5ce84b81f49d8ecb59d680509a5616f287a596e7d335d4b6273b0847a`
- derived state: `calibration_codes`
- quantized/raw arrays: `16 / 8`
- `BranchAgreementReadoutMemory(strength=0.1, mode=low_variance, branch_orders=[1,2])`

Generation/repetition:

| setting | post controlled repeat-2 | post controlled distinct-2 | post greedy repeat-2 |
|---|---:|---:|---:|
| fixed gain baseline | 0.128 | 0.872 | 0.383 |
| branch-agreement low_variance | 0.085 | 0.915 | 0.383 |
| branch-agreement mean_min s0.15 | 0.121 | 0.879 | 0.383 |

Seed1 generation boundary:

- controlled repeat-2 is unchanged: fixed `0.163`, branch-agreement `0.163`
- greedy repeat-2 worsens on the three prompt audit: fixed `0.092`, branch-agreement `0.135`

## Interpretation

Positive:

- First bias-free checkpointed result in this calibration line that improves both CE and argmax accuracy over fixed gain.
- No extra serialized state over fixed gain: same `724,317` serialized bytes; checkpoint only rises `660,634 -> 660,721`.
- Controlled generation repetition improves: repeat-2 `0.128 -> 0.085`, distinct-2 `0.872 -> 0.915`.
- Parity remains exact over 1000 contexts.
- Seed1 repeats the CE/acc direction, but with a smaller gain: CE `2.372 -> 2.370`, acc `0.460 -> 0.462`.

Boundary:

- Greedy generation for the three audit prompts is unchanged, so branch agreement helps measured token decisions and controlled decoding but does not yet solve free-running loops.
- Mean-min gives the best observed accuracy (`0.479`), while low-variance gives the best CE (`2.253`); the difference is small and needs seed repeat.
- Generation repetition improvement is not seed-robust yet; seed1 greedy repeat-2 worsens on the three-prompt audit.

## Next Step

Treat branch agreement as the current best no-BP winner-ordering mechanism. Next:

1. Repeat the mean_min accuracy-best setting on seed1 and add seed2 for both CE-best and acc-best settings.
2. Add a plastic version where local mistakes strengthen branch-agreement support for the target and weaken the wrong winner.
3. Combine branch agreement with inhibition-pressure diagnostics for generation-loop reduction.

Sparse/continuation token probabilities remain diagnostics only.
