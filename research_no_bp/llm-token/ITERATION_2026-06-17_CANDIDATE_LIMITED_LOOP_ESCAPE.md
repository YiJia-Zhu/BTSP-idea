# 2026-06-17 Candidate-Limited Loop Escape

R073 showed that learned loop escape can reduce free-running repetition, but the broad target/wrong update damages CE and ranking. This round constrains where the escape signal can act.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

Extended wrapper:

- `LoopEscapeCompetitorMemory`

New CLI:

- `--loop-escape-score-mode {all,base_topk,winner_suppress}`
- `--loop-escape-score-top-k`
- `--loop-escape-update-mode {target_wrong,wrong_only}`
- `--loop-escape-learn-candidate-k`

Mechanism:

- `base_topk` applies escape scores only to the base model's top-K candidates.
- `winner_suppress` applies only a nonpositive escape signal to the current base winner.
- `learn_candidate_k` limits wrong-candidate learning to the base top-K candidates.
- `wrong_only` removes the target-up half of the update and keeps only wrong-candidate suppression.
- The state remains local synaptic state: `loop_escape_weights[token, branch]`. It stores no raw text, replay data, statistical token-probability table, BP backbone, or API model.

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
| R073 broad escape s0.50 lr0.005 | 2.318054541 | 0.500000000 | 0.170212766 | 0.829787234 | 0.085106383 |
| base_topk k8 learn_k8 s0.50 lr0.005 | 2.322758378 | 0.500000000 | 0.170212766 | 0.829787234 | 0.085106383 |
| base_topk k8 wrong_only s0.50 lr0.005 | 2.470553797 | 0.494230769 | 0.333333333 | 0.666666667 | 0.070921986 |
| winner_suppress learn_k8 s0.50 lr0.005 | 2.216029615 | 0.482692308 | 0.276595745 | 0.723404255 | 0.092198582 |
| winner_suppress wrong_only s0.50 lr0.005 | 2.282292988 | 0.467307692 | 0.695035461 | 0.304964539 | 0.042553191 |
| winner_suppress s0.25 lr0.002 | 2.209122004 | 0.492307692 | 0.588652482 | 0.411347518 | 0.070921986 |
| winner_suppress s0.35 lr0.003 | 2.206003304 | 0.484615385 | 0.588652482 | 0.411347518 | 0.063829787 |

Medium seed0:

| setting | post CE | post acc | serialized bytes | greedy repeat-2 | greedy distinct-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|---:|---:|
| R069 pressure-gated plastic | 2.253651762 | 0.481766821 | 725,853 | 0.382978723 | 0.617021277 | 0.085106383 |
| R073 broad escape s0.35 lr0.003 | 2.397795844 | 0.472521828 | 728,354 | 0.241134752 | 0.758865248 | 0.085106383 |
| R074 winner_suppress s0.50 lr0.005 | 2.480947097 | 0.385208012 | 728,354 | 0.205673759 | 0.794326241 | 0.049645390 |

Representative medium command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_loop_escape_winner_suppress_medium \
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
  --loop-escape --loop-escape-strength 0.50 --loop-escape-lr 0.005 \
  --loop-escape-gate-mode pressure --loop-escape-pressure-threshold 0.20 \
  --loop-escape-pressure-gain 1.0 --loop-escape-margin-threshold 0.35 \
  --loop-escape-score-mode winner_suppress --loop-escape-learn-candidate-k 8 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic
```

## Interpretation

Boundary result:

- Top-K candidate limiting does not fix broad escape. It reproduces the same smoke repetition gain as R073 s0.50, with slightly worse CE.
- `wrong_only` is not enough. Both top-K wrong-only and winner-suppress wrong-only degrade ranking or generation.
- Weak winner-suppress settings can preserve smoke CE, but they fail the actual generation objective: greedy repeat-2 rises to `0.589`.
- Strong winner-suppress scales poorly. Medium repeat-2 improves to `0.206`, but CE/acc collapse to `2.481/0.385`.

Conclusion:

Candidate-limited output-layer escape is still an output-side patch. It can move generation metrics, but it is not a stable route to a GPT-like no-BP learner because it breaks the learned next-token ranking when applied strongly enough.

## Next Step

Move the loop fix down into the representation/state level instead of adding more output penalties:

1. Learn a recurrent/trace branch-state stabilizer that changes branch features before WTA readout.
2. Gate that state update with local branch disagreement, inhibition pressure, and apical error.
3. Keep statistical token probabilities and controlled decoding as diagnostics only.
