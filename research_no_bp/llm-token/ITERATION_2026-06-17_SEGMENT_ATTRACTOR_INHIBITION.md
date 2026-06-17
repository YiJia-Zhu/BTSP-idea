# 2026-06-17 Segment Attractor Inhibition

R070 showed that token-level and transition-level loop inhibition do not change medium greedy generation. This round tests a higher-level dynamic state: a compact segment attractor detector over recent generated/observed output-code trajectories.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper:

- `SegmentAttractorInhibitionMemory`

New CLI:

- `--segment-attractor-inhibition`
- `--segment-attractor-strength`
- `--segment-attractor-dim`
- `--segment-attractor-state-decay`
- `--segment-attractor-trace-decay`
- `--segment-attractor-pressure-decay`
- `--segment-attractor-threshold`
- `--segment-attractor-gain`
- `--segment-attractor-clip`
- `--segment-attractor-slots`
- `--segment-attractor-lag`
- `--segment-attractor-stride`

Mechanism:

- Derived random output codes form a compact decaying `segment_state`.
- Older segment states are stored in a small slot bank.
- If the current state revisits an older slot above a similarity threshold, `segment_attractor_pressure` rises.
- Scores subtract `strength * segment_attractor_pressure * recent_output_trace`.
- The wrapper updates from observed targets during stream evaluation and from predicted tokens during free generation.
- Checkpoints store only dynamic arrays; `segment_attractor_codes` are derived from seed/config.

No raw text, replay buffer, statistical token-probability table, BP-pretrained backbone, or API model is added.

## Results

Base learner is the R069 pressure-gated plastic branch-agreement model:

- random-feedback apical + adaptive inhibition
- feature calibration with derived codes
- fixed readout gain `1.428571`
- fixed low-variance branch agreement strength `0.10`
- plastic branch agreement strength `0.02`, lr `0.002`
- inhibition-pressure gate threshold `0.02`
- direct token prior disabled: `--phase-bias-weight 0.0`
- low-precision 8-bit row state for neural matrices/vectors

Smoke sweep on the same 10k/3k split:

| setting | post CE | post acc | serialized bytes | greedy repeat-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|---:|
| same-data R069 base | 2.218069850 | 0.501923077 | 725,853 | 0.439716312 | 0.070921986 |
| segment s0.10 t0.80 | 2.218239507 | 0.501923077 | 726,818 | 0.439716312 | 0.070921986 |
| segment s0.25 t0.80 | 2.218496815 | 0.501923077 | 726,818 | 0.439716312 | 0.070921986 |
| segment s0.25 t0.90 | 2.218145344 | 0.501923077 | 726,818 | 0.439716312 | 0.070921986 |
| segment s0.75 t0.80 | 2.219378549 | 0.501923077 | 726,818 | 0.439716312 | 0.070921986 |
| segment s2.00 t0.50 | 2.306256788 | 0.496153846 | 726,818 | 0.283687943 | 0.070921986 |

Smoke checkpoint:

| setting | post CE / acc | checkpoint bytes | quantized/raw arrays | parity |
|---|---:|---:|---:|---:|
| segment s0.75 t0.80 | 2.219379 / 0.501923 | 675,363 | 21 / 12 | 1.000 |

Medium seed0 check:

| setting | post CE | post acc | serialized bytes | greedy repeat-2 | greedy distinct-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|---:|---:|
| R069 pressure-gated plastic | 2.253651762 | 0.481766821 | 725,853 | 0.382978723 | 0.617021277 | 0.085106383 |
| R070 transition loop s1.00 | 2.253623384 | 0.481766821 | 793,193 | 0.382978723 | 0.617021277 | 0.085106383 |
| segment s1.00 t0.50 | 2.277690313 | 0.478685157 | 726,818 | 0.382978723 | 0.617021277 | 0.120567376 |
| segment s2.00 t0.50 | 2.310237883 | 0.475089882 | 726,818 | 0.304964539 | 0.695035461 | 0.156028369 |

Representative medium command:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_feature_calib_segment_attractor_s200_t050_medium \
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
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic
```

## Interpretation

Positive:

- Unlike R070, aggressive segment attractor pressure can change greedy output ordering.
- Medium greedy repeat-2 improves from `0.383` to `0.305` and distinct-2 from `0.617` to `0.695`.
- The new dynamic state is small: serialized bytes rise only from `725,853` to `726,818` because the output codes are derived.
- Smoke checkpoint parity is exact.

Boundary:

- The repetition gain is bought with a large CE/accuracy cost: medium CE `2.254 -> 2.310`, acc `0.482 -> 0.475`.
- Moderate settings preserve CE but do not change greedy generation.
- Controlled decoding gets worse under aggressive segment inhibition: repeat-2 `0.085 -> 0.156`.
- Continuous segment inhibition is too blunt; it penalizes useful recent motifs as well as bad attractor loops.

## Next Step

Keep segment attractor pressure as a diagnostic signal, but do not use it as an always-on scoring term. The next candidate should be event-gated:

1. Open segment inhibition only when branch agreement is low or inhibition pressure is high.
2. Gate it by a local confidence margin so high-confidence grammatical continuations are not penalized.
3. Evaluate whether event-gated segment pressure keeps greedy repeat-2 below R069 while preserving CE near `2.254`.

Statistical token-probability methods and controlled decoding remain diagnostics only, not the final no-BP mechanism.
