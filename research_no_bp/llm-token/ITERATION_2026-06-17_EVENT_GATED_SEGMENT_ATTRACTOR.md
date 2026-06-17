# 2026-06-17 Event-Gated Segment Attractor

R071 showed that always-on segment attractor inhibition can reduce greedy repetition, but only by damaging CE/accuracy. This round gates the segment pressure with local event signals so it fires only in suspected loop states.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

Extended wrapper:

- `SegmentAttractorInhibitionMemory`

New CLI:

- `--segment-attractor-gate-mode {none,margin,inhibition,branch,margin_or_inhibition,margin_and_inhibition,either,both}`
- `--segment-attractor-gate-margin-threshold`
- `--segment-attractor-gate-inhibition-threshold`
- `--segment-attractor-gate-branch-threshold`
- `--segment-attractor-gate-gain`

Event signals:

- `margin`: fires when the current top1-top2 score margin is low.
- `inhibition`: fires when the current winner has learned adaptive-inhibition pressure.
- `branch`: fires when branch support for the current winner is weak.
- `margin_or_inhibition` and `margin_and_inhibition` isolate the two most useful non-statistical event signals without forcing branch gating.

The mechanism still stores only local neural/dynamic state. It adds no raw text, replay, n-gram table, statistical token prior, BP-pretrained backbone, or API model.

## Results

Base learner is unchanged from R069/R071:

- random-feedback apical + adaptive inhibition
- feature calibration with derived codes
- fixed readout gain `1.428571`
- fixed low-variance branch agreement strength `0.10`
- plastic branch agreement strength `0.02`, lr `0.002`
- inhibition-pressure plasticity threshold `0.02`
- direct token prior disabled
- low-precision 8-bit row state

Smoke sweep on 10k/3k:

| setting | post CE | post acc | greedy repeat-2 | greedy distinct-2 |
|---|---:|---:|---:|---:|
| same-data R069 base | 2.218069850 | 0.501923077 | 0.439716312 | 0.560283688 |
| R071 always-on s2/t0.50 | 2.306256788 | 0.496153846 | 0.283687943 | 0.716312057 |
| margin gate t0.35 | 2.232585688 | 0.498076923 | 0.510638298 | 0.489361702 |
| inhibition gate t0.02 | 2.318668404 | 0.494230769 | 0.276595745 | 0.723404255 |
| margin_or_inhibition | 2.322759324 | 0.496153846 | 0.276595745 | 0.723404255 |
| margin_and_inhibition | 2.232698607 | 0.498076923 | 0.510638298 | 0.489361702 |
| inhibition gate t0.05 | 2.310198894 | 0.500000000 | 0.191489362 | 0.808510638 |
| inhibition gate t0.10 | 2.282963684 | 0.500000000 | 0.439716312 | 0.560283688 |
| inhibition gate t0.20 | 2.242740196 | 0.500000000 | 0.439716312 | 0.560283688 |
| inhibition gate s1.0/t0.05 | 2.258497316 | 0.500000000 | 0.439716312 | 0.560283688 |
| inhibition gate s1.5/t0.05 | 2.283121302 | 0.503846154 | 0.439716312 | 0.560283688 |

Medium seed0 check:

| setting | post CE | post acc | serialized bytes | greedy repeat-2 | greedy distinct-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|---:|---:|
| R069 pressure-gated plastic | 2.253651762 | 0.481766821 | 725,853 | 0.382978723 | 0.617021277 | 0.085106383 |
| R071 always-on segment s2/t0.50 | 2.310237883 | 0.475089882 | 726,818 | 0.304964539 | 0.695035461 | 0.156028369 |
| R072 event inhibition s2/t0.05 | 2.344551575 | 0.466872111 | 726,818 | 0.333333333 | 0.666666667 | 0.148936170 |

Representative medium command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_event_segment_inhib_s200_t005_medium \
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
  --segment-attractor-inhibition --segment-attractor-strength 2.0 \
  --segment-attractor-threshold 0.50 --segment-attractor-gain 2.0 \
  --segment-attractor-dim 32 --segment-attractor-slots 16 \
  --segment-attractor-lag 4 --segment-attractor-stride 2 \
  --segment-attractor-gate-mode inhibition \
  --segment-attractor-gate-inhibition-threshold 0.05 \
  --segment-attractor-gate-gain 1.0 \
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic
```

## Interpretation

Positive:

- The event-gated implementation is wired into the checkpoint metadata/config path and reuses the same no-raw-data dynamic state as R071.
- Inhibition pressure is the only tested event signal that can lower greedy repetition in smoke.
- Higher inhibition thresholds protect CE, showing that the gate is actually controlling the pressure path.

Boundary:

- The useful repetition regime remains too costly. Medium event-gated inhibition is worse than always-on segment inhibition: CE `2.310 -> 2.345`, acc `0.475 -> 0.467`, and greedy repeat-2 is worse `0.305 -> 0.333`.
- Margin gating protects CE but worsens repetition in smoke.
- The adaptive-inhibition pressure is not selective enough to identify only harmful generation loops.

## Next Step

Do not keep tuning scalar segment-pressure gates. The next mechanism should change what the model does after detecting a loop:

1. Learn a local `loop_escape` competitor over branch/trace features, instead of only suppressing recent tokens.
2. Fire it only when segment pressure is high and current winner confidence is weak.
3. Update escape synapses from target-vs-wrong local feedback during online observation, preserving no raw data and no statistical token probabilities.
