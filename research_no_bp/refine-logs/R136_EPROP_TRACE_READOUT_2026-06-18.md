# R136: E-Prop Eligibility Trace Readout

**Date**: 2026-06-18

## Purpose

Test R095: replace the current WTA/apical readout feature with a local e-prop-style eligibility trace:

```text
e_t = decay * e_{t-1} + feature_t
readout_feature = normalize((1 - weight) * feature_t + weight * e_t)
```

The implementation uses a finite input-context eligibility window so `scores()` remains pure and does not mutate hidden state during repeated evaluation/generation calls. It uses no BP, BPTT, pretrained model, API, or raw-text replay.

Implementation additions in `phase_binding_online_stream_experiment.py`:

- `--eprop-trace-readout`
- `--eprop-order`
- `--eprop-decay`
- `--eprop-weight`

## Runs

Smoke sweeps:

- full replacement:
  - `output/phase_binding_online_stream_r136_eprop_smoke_order8/`
  - `output/phase_binding_online_stream_r136_eprop_smoke_order16/`
- mixed weak trace:
  - `output/phase_binding_online_stream_r136_eprop_smoke_order8_w025/`
  - `output/phase_binding_online_stream_r136_eprop_smoke_order8_w050/`
  - `output/phase_binding_online_stream_r136_eprop_smoke_order4_w025/`

Medium seed repeat:

- `output/phase_binding_online_stream_r136_eprop_medium_seed0_order4_w025/`
- `output/phase_binding_online_stream_r136_eprop_medium_seed1_order4_w025/`
- `output/phase_binding_online_stream_r136_eprop_medium_seed2_order4_w025/`
- aggregate: `output/phase_binding_online_stream_r136_eprop_trace/`

Chosen medium probe:

```bash
--eprop-trace-readout --eprop-order 4 --eprop-decay 0.90 --eprop-weight 0.25
```

## Results

| Method | post CE mean | post acc mean | retention CE mean | greedy repeat-2 |
|---|---:|---:|---:|---:|
| phase_trace_apical_inhib | 2.2791 +/- 0.0217 | 0.4580 +/- 0.0066 | 3.0382 +/- 0.0211 | 0.5059 |
| phase_eprop_trace_apical_inhib | 2.3565 +/- 0.0164 | 0.4730 +/- 0.0082 | 3.0680 +/- 0.0177 | 0.4255 |

Paired e-prop minus baseline:

| seed | CE delta | acc delta | retention CE delta |
|---:|---:|---:|---:|
| 0 | +0.08560 | +0.01746 | +0.03715 |
| 1 | +0.06733 | +0.01181 | +0.02093 |
| 2 | +0.07914 | +0.01592 | +0.03139 |

## Interpretation

R136 is a clear tradeoff, not a CE improvement.

The weak eligibility trace improves top-1 accuracy on all three seeds and reduces greedy repeat-2 (`0.5059 -> 0.4255`), so it is changing winner selection in a useful direction. But CE worsens substantially on every seed (`+0.067` to `+0.086`), and retention also worsens. Full replacement by eligibility was much worse in smoke; only a weak mix preserved enough of the original feature.

Mechanistically, this suggests that local eligibility traces carry useful winner-selection evidence but are poorly calibrated as probability features. Treating the same eligibility vector as both a CE-calibrated feature and a WTA correction is too blunt.

## Verdict

R095 status: **DONE / TRADEOFF-POSITIVE**.

Supported:

- e-prop-style finite eligibility readout is implemented;
- accuracy improves reproducibly across seeds;
- greedy repetition decreases;
- no raw text is stored in model state.

Not supported:

- CE improvement;
- retention-safe integration;
- direct stacking into R096 as the main readout feature.

Next:

1. Use eligibility as a candidate/winner correction branch, not as a full probability feature.
2. Test a dual-head readout: CE head remains phase/apical; eligibility head only breaks low-margin ties or supplies anti-loop candidates.
3. Compare with R134 KV: KV is CE-positive but acc/retention-negative; e-prop is acc/repetition-positive but CE-negative. A future R096 candidate should arbitrate between them rather than simply concatenate both.
