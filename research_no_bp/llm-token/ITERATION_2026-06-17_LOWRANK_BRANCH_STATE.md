# 2026-06-17 Low-Rank Branch-State Projection

R075 showed that a full feature-space branch-state projection can improve smoke CE/accuracy, but it is too large and does not survive medium. This round compresses the representation-level residual into a low-rank projection and tests a feature-state novelty gate.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

Extended wrapper:

- `BranchStateStabilizerMemory`

New CLI:

- `--branch-state-projection-rank`
- `--branch-state-novelty-slots`
- `--branch-state-novelty-threshold`
- `--branch-state-novelty-strength`

Mechanism:

- If `projection_rank > 0`, a fixed derived encoder maps `branch_state` into a low-dimensional latent.
- Only the decoder `branch_state_projection[feature_dim, rank]` is plastic.
- The score residual remains representation-level: `W @ (decoder @ encoder @ branch_state)`.
- Novelty slots keep recent feature states and downweight the residual when the current state is too similar to recent states.
- No raw text, BP, pretrained backbone, API model, replay buffer, or statistical token-probability table is used.

## Results

Base learner:

- R069 pressure-gated plastic branch-agreement base
- direct token prior disabled
- low-precision 8-bit row state
- feature-calibration derived codes

Smoke sweep on 10k/3k:

| setting | post CE | post acc | serialized bytes | greedy repeat-2 | greedy distinct-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|---:|---:|
| same-data R069 base | 2.218069850 | 0.501923077 | 725,853 | 0.439716312 | 0.560283688 | 0.070921986 |
| R075 full branch-state apical | 2.101052590 | 0.523076923 | 1,060,513 | 0.510638298 | 0.489361702 | 0.106382979 |
| rank16 branch-state apical | 2.129223634 | 0.513461538 | 737,957 | 0.567375887 | 0.432624113 | 0.141843972 |
| rank16 + novelty threshold0.92 | 2.129223634 | 0.513461538 | 747,253 | 0.567375887 | 0.432624113 | 0.141843972 |
| rank16 + novelty threshold0.50 | 2.168664949 | 0.503846154 | 747,253 | 0.567375887 | 0.432624113 | 0.141843972 |

Medium seed0:

| setting | post CE | post acc | serialized bytes | checkpoint bytes | parity | greedy repeat-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R069 pressure-gated plastic | 2.253651762 | 0.481766821 | 725,853 | 662,662 | 1.000 | 0.382978723 | 0.085106383 |
| R075 full branch-state apical | 2.261931934 | 0.481766821 | 1,060,513 | 965,651 | 1.000 | 0.382978723 | 0.120567376 |
| R076 rank16 branch-state apical | 2.261385503 | 0.483307653 | 737,957 | 676,280 | 1.000 | 0.382978723 | 0.085106383 |

Representative medium command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_branch_state_lowrank16_medium \
  --method-filter phase_trace_apical_inhib \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 --phase-bias-weight 0.0 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --apical-gating-branch --apical-error-mode random_feedback \
  --adaptive-inhibition --feature-calibration --feature-calibration-derived-codes \
  --readout-gain 1.4285714286 \
  --branch-agreement-readout --branch-agreement-mode low_variance \
  --plastic-branch-agreement --plastic-branch-agreement-pressure-mode inhibition \
  --branch-state-stabilizer --branch-state-strength 0.10 \
  --branch-state-lr 0.001 --branch-state-gate-mode apical \
  --branch-state-projection-rank 16 --branch-state-derived-codes \
  --low-precision-bits 8 --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 200
```

## Interpretation

Positive:

- Low-rank projection preserves much of the smoke CE/accuracy gain: full `2.101/0.523`, rank16 `2.129/0.513`, base `2.218/0.502`.
- State compression works. Medium serialized state drops from R075 `1,060,513` to `737,957` bytes, close to R069 `725,853`.
- Checkpoint parity remains exact enough: 200-context pred match `1.000`, max score diff `1.19e-7`, loss diff `0.0`.
- Medium rank16 slightly improves accuracy over R069 (`0.4833` vs `0.4818`) but not CE.

Boundary:

- Medium CE still does not beat R069 (`2.261` vs `2.254`).
- Greedy repetition is not improved. Medium greedy repeat-2 remains `0.383`.
- On smoke, rank16 actually worsens greedy repeat-2 to `0.567`.
- The simple feature-similarity novelty multiplier changes CE but not greedy argmax, so it is not sufficient for free-running loop control.

## Next Step

The low-rank representation path is now deployable but not yet useful for generation. Next change should alter the generated state trajectory itself:

1. Learn a low-rank state-space anti-attractor update that penalizes repeated feature-state direction before scoring.
2. Gate it on generated/observed branch-state recurrence, not only current-state similarity.
3. Keep the correction in feature space; do not revert to output-token suppression or statistical token probabilities as the final method.
