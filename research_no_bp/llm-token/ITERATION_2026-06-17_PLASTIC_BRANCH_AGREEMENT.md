# 2026-06-17 Plastic Branch-Agreement Readout

R067 showed that fixed dendritic branch agreement improves winner ordering over fixed readout gain, but the generation benefit is not seed-robust. This round adds a small local plastic branch-agreement correction.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper:

- `PlasticBranchAgreementReadoutMemory`

New CLI:

- `--plastic-branch-agreement`
- `--plastic-branch-agreement-strength`
- `--plastic-branch-agreement-lr`
- `--plastic-branch-agreement-decay`
- `--plastic-branch-agreement-clip`
- `--plastic-branch-agreement-support-clip`
- `--plastic-branch-agreement-top-k`
- `--plastic-branch-agreement-margin`

Mechanism:

- each output token owns a small vector over dendritic branches
- branch support vectors come from the existing phase branch prototypes
- on a local WTA error, the target row moves toward its current branch-support vector
- the wrong-winner row moves away from its current branch-support vector
- scores add `plastic_strength * dot(plastic_branch_weights[token], branch_supports[:, token])`

This stores only a small synaptic matrix of shape `vocab_size x branch_count`; it stores no raw text, no context-count rows, and no token probability table.

## Results

Base learner:

- random-feedback apical + adaptive inhibition
- feature calibration with derived codes
- fixed readout gain `1.428571`
- fixed low-variance branch-agreement strength `0.10`
- direct token prior disabled: `--phase-bias-weight 0.0`
- low-precision 8-bit row state for neural matrices/vectors

Seed0:

| setting | post CE / acc | serialized bytes | checkpoint bytes | parity |
|---|---:|---:|---:|---:|
| fixed gain baseline | 2.262 / 0.476 | 724,317 | 660,634 | 1.000 |
| fixed branch agreement | 2.253 / 0.478 | 724,317 | 660,721 | 1.000 |
| plastic agreement s0.02 lr0.002 | 2.254 / 0.481 | 725,853 | 662,607 | 1.000 |
| plastic agreement s0.05 lr0.005 | 2.254 / 0.479 | 725,853 | n/a | n/a |
| plastic-only s0.05 lr0.005 | 2.267 / 0.477 | 725,853 | n/a | n/a |

Seed repeats:

| seed | fixed gain | fixed branch agreement | plastic agreement s0.02 lr0.002 |
|---:|---:|---:|---:|
| 1 | 2.372 / 0.460 | 2.370 / 0.462 | 2.369 / 0.462 |
| 2 | 2.264 / 0.467 | 2.256 / 0.473 | 2.257 / 0.474 |

Best-accuracy checkpoint command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_branchagree_plastic_s002_lr002_checkpoint_medium \
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
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 1000
```

Checkpoint metadata:

- `config_signature`: `fcc3143d72443df0b0625ae74e3f91351c6537fa7e69733f465c64e08b00dea6`
- derived state: `calibration_codes`
- quantized/raw arrays: `17 / 8`
- `PlasticBranchAgreementReadoutMemory(strength=0.02, lr=0.002, branch_orders=[1,2])`
- exact checkpoint parity over 1000 contexts

Generation/repetition:

| seed | setting | controlled repeat-2 | controlled distinct-2 | greedy repeat-2 |
|---:|---|---:|---:|---:|
| 0 | fixed branch agreement | 0.085 | 0.915 | 0.383 |
| 0 | plastic agreement | 0.085 | 0.915 | 0.383 |
| 1 | plastic agreement | 0.163 | 0.837 | 0.355 |
| 2 | plastic agreement | 0.149 | 0.851 | 0.376 |

## Interpretation

Positive:

- Plastic branch agreement gives the current best seed0 accuracy: `0.481`, above fixed gain `0.476` and fixed branch agreement `0.478`.
- Seed1 and seed2 keep the CE/acc direction over fixed gain, but gains are small.
- Checkpointing remains exact; the extra serialized state is small: `724,317 -> 725,853` bytes.

Boundary:

- CE-best remains fixed low-variance branch agreement: `2.253`, slightly better than plastic `2.254`.
- Generation-loop reduction is not improved beyond fixed branch agreement. Greedy repeat-2 remains high, and seed1/seed2 controlled repeat-2 are worse than seed0.
- The current plastic update helps winner selection more than language quality.

## Next Step

Move from output-row branch weights to inhibition-pressure-aware plasticity:

1. Update plastic branch rows only when the wrong winner is also under strong learned inhibition.
2. Add a local loop-pressure metric during generation to detect repeated-token attractors.
3. Test whether loop pressure can gate anti-winner updates without hurting CE.

Sparse/continuation token probabilities remain diagnostics only.
