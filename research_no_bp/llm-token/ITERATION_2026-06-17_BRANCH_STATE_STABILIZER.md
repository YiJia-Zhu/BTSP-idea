# 2026-06-17 Branch-State Stabilizer

R074 showed that output-side loop escape is not stable enough. This round moves the intervention into representation state: a recurrent branch-state residual is read through the existing WTA weights before the final score.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper:

- `BranchStateStabilizerMemory`

New CLI:

- `--branch-state-stabilizer`
- `--branch-state-strength`
- `--branch-state-lr`
- `--branch-state-decay`
- `--branch-state-projection-decay`
- `--branch-state-clip`
- `--branch-state-target-mix`
- `--branch-state-gate-mode {none,margin,branch,inhibition,apical,any,all}`
- `--branch-state-margin-threshold`
- `--branch-state-branch-threshold`
- `--branch-state-inhibition-threshold`
- `--branch-state-apical-threshold`
- `--branch-state-gate-gain`
- `--branch-state-top-k`
- `--branch-state-support-clip`
- `--branch-state-input-mode {feature,target,mixed}`
- `--branch-state-derived-codes`

Mechanism:

- The wrapper keeps a dynamic feature-space state `branch_state`.
- A local projection `branch_state_projection` maps the state into a feature residual.
- Scores add `W @ (branch_state_projection @ branch_state)`, where `W` is the existing learned no-BP readout matrix.
- Projection updates are local target-vs-wrong corrections: the current state is mapped toward `W[target] - W[wrong]`.
- Gates can depend on local top-2 margin, branch disagreement, adaptive inhibition pressure, and apical error trace.
- `branch_state_codes` can be derived from seed/config and omitted from checkpoint storage.

This is not a BP-trained model, pretrained backbone, API method, replay buffer, or statistical token-probability table.

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
| branch-state any s0.10 lr0.001 | 2.099480224 | 0.521153846 | 1,060,513 | 0.510638298 | 0.489361702 | 0.106382979 |
| branch-state any s0.05 lr0.0005 | 2.156554597 | 0.511538462 | 1,060,513 | 0.510638298 | 0.489361702 | 0.092198582 |
| branch-state apical s0.10 lr0.001 | 2.101052590 | 0.523076923 | 1,060,513 | 0.510638298 | 0.489361702 | 0.106382979 |

Medium seed0:

| setting | post CE | post acc | serialized bytes | checkpoint bytes | parity | greedy repeat-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R069 pressure-gated plastic | 2.253651762 | 0.481766821 | 725,853 | 662,662 | 1.000 | 0.382978723 | 0.085106383 |
| R075 branch-state apical | 2.261931934 | 0.481766821 | 1,060,513 | 965,651 | 1.000 | 0.382978723 | 0.120567376 |

Representative medium command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_branch_state_apical_medium \
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
  --branch-state-derived-codes \
  --low-precision-bits 8 --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic \
  --save-serialized-checkpoint --checkpoint-parity-limit 200
```

## Interpretation

Positive:

- The representation-level residual is functional. On smoke it improves post CE/acc from `2.218/0.502` to about `2.10/0.52`.
- The checkpoint path works with derived codes: 200-context parity is `1.000`, max score diff `0.0`.
- The mechanism remains no-BP and no-raw-data; it stores feature-state/projection arrays, not text or token-count probabilities.

Boundary:

- The smoke gain does not survive medium. Medium CE is `2.262`, slightly worse than R069 `2.254`, with the same acc.
- Full-matrix projection is expensive: serialized state grows from `725,853` to `1,060,513` bytes, and runtime roughly doubles.
- Free-running repetition is not solved. Medium greedy repeat-2 stays at `0.383`, and controlled repeat-2 worsens to `0.121`.
- On smoke, the feature residual actually strengthens greedy repetition (`0.440 -> 0.511`), suggesting the residual can sharpen local ranking while also deepening attractors.

## Next Step

Keep the representation-level route, but do not keep the full-matrix version as final:

1. Replace full `feature_dim x feature_dim` projection with a low-rank state encoder/decoder.
2. Add a novelty or anti-attractor gate on generated state, not a direct output-token penalty.
3. Evaluate whether low-rank projection preserves the smoke CE gain without increasing repetition.
