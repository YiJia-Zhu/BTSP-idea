# U001: Unified NoProp Core

**Date**: 2026-06-20

## Question

Can the strongest R096-prep line be converted from a wrapper stack into one unified no-BP model core without losing the TinyStories online token result?

R096-prep used:

- NoProp local denoising depth;
- adaptive output inhibition;
- feature-conditioned local calibration;
- fixed readout gain.

U001 folds these into `OnlineUnifiedNoPropCalibrationMemory` so the model has one core `scores/update/observe/state_bytes` lifecycle instead of separate post-hoc branches.

## Contract Check

| Check | Answer |
|---|---|
| Same core can be used by multiple adapters | Yes. The core is still next-token/full-vocab `scores(context)` plus online `update(context, target)`. TinyStories is only the first adapter. |
| New mechanism is core learning rule, not task patch | Yes. The changes are NoProp hidden state, output inhibition, calibration, gain, and optional eligibility pressure inside one token learner. |
| Local neural interpretation | Yes. All updates are target/wrong local WTA, inhibitory trace, fixed random context gates, and local denoising targets. |
| Complexity reduced or held | Partially. External wrappers are unified, but state size remains 11.75MB. |
| No dataset labels as internal state | Yes. No answer slots, parser, path graph, raw replay, or pretrained model. |
| Statistical methods only baseline/diagnostic | Yes. No n-gram/backoff table is used. |

## Implementation

Added to `phase_binding_online_stream_experiment.py`:

- `OnlineUnifiedNoPropCalibrationMemory`;
- `--unified-noprop-core`;
- `--unified-calibration-*`;
- `--unified-readout-gain`;
- `--unified-eligibility-*`.

The main U001 score path is:

```text
phase/trace feature
  -> NoProp local denoising feature
  -> local output inhibition
  -> context-gated local calibration
  -> fixed readout gain
  -> full-vocab next-token scores
```

Optional eligibility score pressure was implemented but is not part of the positive U001 result.

## Main Run

Command template:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/u001_unified_noprop_medium_v2_seed${SEED} \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --phase-dim 128 --branch-orders 1 2 --branch-weights 0.5 0.5 \
  --trace-branch --trace-order 16 --trace-dim 128 \
  --unified-noprop-core --noprop-hidden-dims 768 --noprop-label-dim 128 \
  --noprop-alpha-start 0.95 --noprop-alpha-end 0.50 \
  --noprop-lr 0.02 --noprop-denoise-lr 0.02 \
  --noprop-bias-lr 0.002 --noprop-delta-clip 1.0 \
  --inhibit-strength 0.15 --inhibit-lr 0.005 --inhibit-top-k 1 \
  --unified-calibration-strength 1.0 --unified-calibration-lr 0.02 \
  --unified-calibration-gate-decay 0.50 --unified-readout-gain 1.15 \
  --method-filter u001 --completion-count 1 \
  --prompt-tokens 16 --completion-tokens 48 --seed ${SEED}
```

Outputs:

- `output/u001_unified_noprop_medium_v2_seed0/`
- `output/u001_unified_noprop_medium_v2_seed1/`
- `output/u001_unified_noprop_medium_v2_seed2/`

## Results

| Seed | Post CE | Post acc | Retention CE | Top-4 acc | Target rank | Greedy repeat-2 | Controlled repeat-2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.2014 | 0.4818 | 3.3162 | 0.7417 | 6.48 | 0.383 | 0.149 |
| 1 | 2.2383 | 0.4658 | 3.3278 | 0.7196 | 6.31 | 0.298 | 0.149 |
| 2 | 2.1927 | 0.4771 | 3.2629 | 0.7298 | 6.28 | 0.809 | 0.064 |

Aggregate:

| Metric | U001 |
|---|---:|
| Post CE | `2.2108 +/- 0.0242` |
| Post acc | `0.4749 +/- 0.0082` |
| Retention CE | `3.3023 +/- 0.0346` |
| Top-4 acc | `0.7304 +/- 0.0111` |
| Target rank | `6.3573 +/- 0.1114` |
| State bytes | `11,746,564` |
| Greedy repeat-2 | `0.4965 +/- 0.2736` |
| Controlled repeat-2 | `0.1206 +/- 0.0491` |

Reference R096-prep:

| Metric | R096-prep |
|---|---:|
| Post CE | `2.2108 +/- 0.0198` |
| Post acc | `0.4749 +/- 0.0067` |
| Retention CE | `3.3023 +/- 0.0283` |
| Greedy repeat-2 | `0.496 +/- 0.223` |
| Controlled repeat-2 | `0.121 +/- 0.040` |
| State bytes | `11,746,564` |

The U001 v2 result intentionally matches R096-prep after aligning the internal calibration-code seed with the original wrapper implementation.

## Eligibility Probe

Direct finite-window eligibility score pressure was tested on the smoke setting:

| Variant | Smoke post CE | Smoke post acc |
|---|---:|---:|
| U001 no eligibility | 2.089 | 0.485 |
| U001 eligibility score weight 0.05 | 2.095 | 0.483 |

This repeats the R136 pattern: eligibility carries useful temporal signal, but direct score mixing hurts CE calibration. It should not be added to the main core as a raw probability feature.

## Verdict

U001 is **DONE-POSITIVE-STRUCTURAL**, not a new metric breakthrough.

What passed:

- R096-prep is now expressible as one unified no-BP model core.
- The result preserves the strongest known deep no-BP TinyStories line.
- The implementation avoids pretrained models, BP/BPTT, raw replay, n-gram memory, answer slots, parser state, and dataset-specific controllers.

What did not pass:

- Mean post CE is still `2.2108`, so the target CE `< 2.20` is not met.
- Greedy repeat-2 remains high and high-variance.
- State remains `11.75MB`; no compression/hardware claim is supported yet.
- Direct eligibility score pressure is negative in smoke.

## Next

U002 should modify the unified core rather than add a new branch. The highest-priority directions are:

1. rank/margin-weighted calibration update inside U001, using the existing top-k and margin diagnostics;
2. anti-loop learning through inhibitory pressure or calibration update gating, not direct eligibility score mixing;
3. state compression of NoProp targets/projections after metric parity is established;
4. adapter test on a second task only after the core change is task-agnostic and still exposes full-vocab next-token scores.
