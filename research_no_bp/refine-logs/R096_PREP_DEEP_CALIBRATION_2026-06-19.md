# R096 Prep: Deep Local Calibration

**Date**: 2026-06-19

## Question

R092 DLL and R093 NoProp showed that deep local target features can learn useful winners but fail CE calibration. Can existing local calibration/arbitration wrappers rescue the deep NoProp branch without BP, pretrained weights, raw replay, or statistical n-gram memory?

## Setup

No new model code was needed. Existing wrappers already apply after `--method-filter`, so they can wrap the NoProp builder directly.

Best candidate:

- base: R093 `phase_trace_noprop_local_inhib_competitive_online`;
- local calibration: `FeatureConditionedCalibrationMemory`;
- fixed readout energy gain: `ReadoutGainMemory` with gain `1.15`;
- feature calibration uses fixed random context gates and local target/wrong-winner updates;
- no raw text or context-count table is stored.

Command template:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_r096prep_noprop_feature_gain_seed${SEED} \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --phase-dim 128 --branch-orders 1 2 --branch-weights 0.5 0.5 \
  --trace-branch --trace-order 16 --trace-dim 128 \
  --noprop-depth-branch --noprop-hidden-dims 768 --noprop-label-dim 128 \
  --noprop-alpha-start 0.95 --noprop-alpha-end 0.50 \
  --noprop-lr 0.02 --noprop-denoise-lr 0.02 \
  --noprop-bias-lr 0.002 --noprop-delta-clip 1.0 \
  --adaptive-inhibition --inhibit-strength 0.15 --inhibit-lr 0.005 --inhibit-top-k 1 \
  --feature-calibration --feature-calibration-strength 1.0 \
  --feature-calibration-lr 0.02 --feature-calibration-gate-decay 0.50 \
  --feature-calibration-derived-codes \
  --readout-gain 1.15 \
  --method-filter phase_trace_noprop_local_inhib_competitive_online \
  --completion-count 1 --prompt-tokens 16 --completion-tokens 48 \
  --seed ${SEED}
```

Outputs:

- `output/phase_binding_online_stream_r096prep_deep_calib/summary.csv`
- `output/phase_binding_online_stream_r096prep_deep_calib/per_seed_summary.csv`
- `output/phase_binding_online_stream_r096prep_deep_calib/comparison_summary.csv`
- `output/phase_binding_online_stream_r096prep_deep_calib/generation_summary_aggregate.csv`
- `output/phase_binding_online_stream_r096prep_deep_calib/probe_summary.csv`

## Results

| Method | Seeds | Post CE | Post acc | Retention CE | State bytes |
|---|---:|---:|---:|---:|---:|
| `phase_trace_noprop_local_inhib_competitive_online_feature_calib_gain` | 3 | `2.2108 +/- 0.0198` | `0.4749 +/- 0.0067` | `3.3023 +/- 0.0283` | `11,746,564` |

Per seed:

| Seed | Post CE | Post acc |
|---:|---:|---:|
| 0 | 2.201 | 0.482 |
| 1 | 2.238 | 0.466 |
| 2 | 2.193 | 0.477 |

Comparison:

| Reference | Post CE | Post acc | State |
|---|---:|---:|---:|
| R134 apical+inhib baseline | `2.2791 +/- 0.0217` | `0.4580 +/- 0.0066` | 2.89M |
| R134 KV no-gate | `2.2668 +/- 0.0234` | `0.4557 +/- 0.0041` | 3.35M |
| R092 DLL 1x768+inhib | `2.2807 +/- 0.0099` | `0.4720 +/- 0.0059` | 6.30M |
| R093 NoProp 1x768+inhib | `2.2782 +/- 0.0115` | `0.4672 +/- 0.0074` | 10.63M |
| R096-prep NoProp+calib+gain | `2.2108 +/- 0.0198` | `0.4749 +/- 0.0067` | 11.75M |

## Generation

| Decode | Stage | Repeat-2 | Distinct-2 |
|---|---|---:|---:|
| greedy | post_online | `0.496 +/- 0.223` | `0.504 +/- 0.223` |
| controlled | post_online | `0.121 +/- 0.040` | `0.879 +/- 0.040` |

## Verdict

This is a **positive R096-prep result**, not full R096 completion.

What passed:

- Deep NoProp + local calibration beats the R092/R093 CE gate on all three seeds.
- It improves mean CE from R093 `2.2782` to `2.2108`.
- It beats the prior R134 apical+inhib and KV CE baselines by a large margin.
- It keeps top-1 near the best prior line at `0.4749`.

What did not pass:

- R096 target mean CE `<2.20` is not met yet (`2.2108`).
- Greedy repeat-2 remains high (`0.496`), although controlled decoding is good (`0.121`).
- State grows to `11.75MB`; compression/derived-state treatment is needed before hardware/privacy claims.

## Next

R096 should proceed, but not as a blind stack of R092/R093 + R094 + R095. The viable integration path is:

1. Keep NoProp+feature-calibration+gain as the deep calibrated backbone.
2. Add R095 e-prop only as a candidate/tie-break or repetition-control branch, not as direct CE feature.
3. Add R094 KV only behind a learned/local arbitration gate because the hard KV margin gate was too coarse.
4. Add a repetition-specific local state mechanism; CE is now close enough that greedy generation is the next bottleneck.
