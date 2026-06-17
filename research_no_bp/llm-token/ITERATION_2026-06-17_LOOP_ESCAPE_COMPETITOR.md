# 2026-06-17 Learned Loop-Escape Competitor

R072 showed that scalar segment-pressure gates are not selective enough. This round changes the action taken after detecting a loop: instead of suppressing recent tokens, learn a local escape competitor over branch supports.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper:

- `LoopEscapeCompetitorMemory`

New CLI:

- `--loop-escape`
- `--loop-escape-strength`
- `--loop-escape-lr`
- `--loop-escape-decay`
- `--loop-escape-clip`
- `--loop-escape-support-clip`
- `--loop-escape-top-k`
- `--loop-escape-margin`
- `--loop-escape-gate-mode {pressure,pressure_and_margin,pressure_or_margin}`
- `--loop-escape-pressure-threshold`
- `--loop-escape-pressure-gain`
- `--loop-escape-margin-threshold`

Mechanism:

- `SegmentAttractorInhibitionMemory` supplies segment pressure, usually with `--segment-attractor-strength 0.0` so it acts only as a detector.
- `LoopEscapeCompetitorMemory` learns `loop_escape_weights[token, branch]`.
- Scores add `strength * gate * einsum(loop_escape_weights, branch_supports)`.
- Online updates are local target-up / wrong-winner-down corrections when the segment-pressure gate fires.
- The update stores only synaptic state; no raw text, replay, statistical token-probability table, BP backbone, or API model is used.

## Results

Base learner:

- R069 pressure-gated plastic branch-agreement base
- direct token prior disabled
- low-precision 8-bit row state
- segment attractor detector enabled with scoring strength `0.0`

Smoke sweep on 10k/3k:

| setting | post CE | post acc | greedy repeat-2 | greedy distinct-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|---:|
| same-data R069 base | 2.218069850 | 0.501923077 | 0.439716312 | 0.560283688 | 0.070921986 |
| R071 always-on segment s2 | 2.306256788 | 0.496153846 | 0.283687943 | 0.716312057 | 0.070921986 |
| pressure s1.0 lr0.01 | 2.672932096 | 0.461538462 | 0.290780142 | 0.709219858 | 0.106382979 |
| pressure+margin s1.0 lr0.01 | 2.243855182 | 0.492307692 | 0.808510638 | 0.191489362 | 0.141843972 |
| pressure-or-margin s1.0 lr0.01 | 2.670562376 | 0.463461538 | 0.560283688 | 0.439716312 | 0.056737589 |
| pressure strong s2.0 lr0.02 | 3.327079187 | 0.423076923 | 0.631205674 | 0.368794326 | 0.163120567 |
| pressure s0.25 lr0.002 | 2.234810839 | 0.500000000 | 0.439716312 | 0.560283688 | 0.070921986 |
| pressure s0.35 lr0.003 | 2.256718116 | 0.498076923 | 0.439716312 | 0.560283688 | 0.056737589 |
| pressure s0.40 lr0.004 | 2.281007597 | 0.503846154 | 0.439716312 | 0.560283688 | 0.085106383 |
| pressure s0.50 lr0.005 | 2.318054541 | 0.500000000 | 0.170212766 | 0.829787234 | 0.085106383 |
| pressure s0.75 lr0.005 | 2.374572556 | 0.501923077 | 0.134751773 | 0.865248227 | 0.049645390 |

Medium seed0:

| setting | post CE | post acc | serialized bytes | greedy repeat-2 | greedy distinct-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|---:|---:|
| R069 pressure-gated plastic | 2.253651762 | 0.481766821 | 725,853 | 0.382978723 | 0.617021277 | 0.085106383 |
| R071 always-on segment s2 | 2.310237883 | 0.475089882 | 726,818 | 0.304964539 | 0.695035461 | 0.156028369 |
| R072 event-gated segment | 2.344551575 | 0.466872111 | 726,818 | 0.333333333 | 0.666666667 | 0.148936170 |
| R073 escape s0.35 lr0.003 | 2.397795844 | 0.472521828 | 728,354 | 0.241134752 | 0.758865248 | 0.085106383 |
| R073 escape s0.50 lr0.005 | 2.589518440 | 0.447868516 | 728,354 | 0.361702128 | 0.638297872 | 0.099290780 |

Representative medium command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_loop_escape_s035_lr003_medium \
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
  --segment-attractor-inhibition --segment-attractor-strength 0.0 \
  --segment-attractor-threshold 0.50 --segment-attractor-gain 2.0 \
  --segment-attractor-dim 32 --segment-attractor-slots 16 \
  --segment-attractor-lag 4 --segment-attractor-stride 2 \
  --loop-escape --loop-escape-strength 0.35 --loop-escape-lr 0.003 \
  --loop-escape-gate-mode pressure --loop-escape-pressure-threshold 0.20 \
  --loop-escape-pressure-gain 1.0 --loop-escape-margin-threshold 0.35 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic
```

## Interpretation

Positive:

- The learned escape competitor can change generation more strongly than scalar segment gates.
- Medium greedy repeat-2 improves from R069 `0.383` and R071 `0.305` to `0.241` at s0.35/lr0.003.
- Controlled decoding is not harmed at the s0.35 medium point: repeat-2 returns to `0.085`.
- Serialized state only rises to `728,354` bytes because the escape matrix is tiny.

Boundary:

- CE/accuracy cost is too high. The best medium generation point degrades CE/acc from R069 `2.254/0.482` to `2.398/0.473`.
- Stronger escape settings are unstable: s0.50 medium gives CE/acc `2.590/0.448` and loses much of the repetition gain.
- Pressure+margin protects CE in smoke but worsens greedy loops, suggesting the escape update is learning a bad attractor when the gate is too selective.

## Next Step

The escape idea has leverage, but the update rule is too broad. Next candidate:

1. Constrain escape learning to the current wrong winner and a small candidate set, not every target/wrong branch support.
2. Add a margin-preserving update: only accept escape synapse changes that do not lower held-out next-token CE on the immediate local context.
3. Alternatively learn a local anti-loop projection over the current winner only, then let existing branch agreement choose the replacement.

Keep statistical token probabilities and controlled decoding as diagnostics only.
