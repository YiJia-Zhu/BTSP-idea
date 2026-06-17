# 2026-06-17 Selective Quantization Audit For Apical No-BP State

R054 showed that per-row scaling improves 8-bit quantization but still leaves a large CE gap to full precision. This round localizes which state variables cause the damage by adding selective low-precision targets to `LowPrecisionStateWrapper`.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New CLI:

- `--low-precision-targets`
- `--low-precision-bias-clip`

Target groups:

| group | Main arrays |
|---|---|
| `all` | previous behavior: all floating NumPy state |
| `plastic` | phase state, readout weights, recurrent transition, inhibition, output bias |
| `readout` | WTA weights, inhibition, output bias |
| `phase` | phase code banks, prototypes, counts, output bias |
| `fixed` | fixed/random code tables and apical feedback vectors |
| `dynamic` | short-lived activity, fatigue, apical error trace, eligibility |
| `bias` | output log-prior / `output_bias` |
| `counts` | `prototype_counts` and `unigram_counts` |
| `phase_codes` | phase code banks only |
| `phase_prototypes` | phase prototypes only |
| `phase_banks` | phase code banks, prototypes, and counts |
| `readout_weights` | WTA/readout `weights` only |
| `inhibition` | adaptive inhibition matrix only |

The wrapper now estimates mixed-precision state bytes by counting quantized and unquantized arrays separately. It also caches target array references after initialization, so repeated projection no longer recursively scans the whole object graph. It is still a projection audit, not true serialized integer storage.

## Selective 8-Bit Row Results

Common setup: TinyStories tokenizer, 50k/10k chars, vocab 256, random-feedback apical error, trace+adaptive inhibition, `--low-precision-bits 8 --low-precision-scale-mode row --low-precision-clip 1.0`.

| target group | method | post CE / acc | state bytes | interpretation |
|---|---|---:|---:|---|
| full precision | trace+inhib | 2.358 / 0.429 | 2,761,728 | reference |
| full precision | apical+inhib | 2.289 / 0.437 | 2,761,752 | reference |
| all | trace+inhib | 2.651 / 0.453 | 690,432 | R054 all-state row |
| all | apical+inhib | 2.523 / 0.462 | 690,438 | R054 all-state row |
| plastic | trace+inhib | 2.653 / 0.452 | 1,133,568 | nearly same as all-state |
| plastic | apical+inhib | 2.521 / 0.462 | 1,133,604 | nearly same as all-state |
| readout | trace+inhib | 2.653 / 0.453 | 2,119,680 | nearly same as all-state |
| readout | apical+inhib | 2.520 / 0.462 | 2,119,716 | nearly same as all-state |
| phase | trace+inhib | 2.644 / 0.451 | 1,772,544 | also damaged |
| phase | apical+inhib | 2.512 / 0.460 | 1,772,580 | also damaged |
| fixed | trace+inhib | 2.358 / 0.429 | 2,319,360 | no meaningful CE damage |
| fixed | apical+inhib | 2.289 / 0.436 | 2,319,378 | no meaningful CE damage |
| dynamic | trace+inhib | 2.359 / 0.431 | 2,760,960 | no meaningful CE damage |
| dynamic | apical+inhib | 2.289 / 0.437 | 2,760,987 | no meaningful CE damage |
| bias | trace+inhib | 2.643 / 0.453 | 2,758,656 | reproduces most CE damage |
| bias | apical+inhib | 2.513 / 0.463 | 2,758,692 | reproduces most CE damage |
| readout_weights | trace+inhib | 2.364 / 0.432 | 2,319,360 | close to full precision |
| readout_weights | apical+inhib | 2.295 / 0.439 | 2,319,396 | close to full precision |
| inhibition | trace+inhib | 2.354 / 0.429 | 2,565,120 | close to full precision |
| inhibition | apical+inhib | 2.289 / 0.439 | 2,565,156 | close to full precision |

## Generation Check

8-bit row readout-targeted quantization was rerun with generation enabled.

Greedy post-online repetition:

| setting | trace+inhib repeat-2 | apical+inhib repeat-2 |
|---|---:|---:|
| full precision | 0.473 | 0.383 |
| 8-bit all row | 0.569 | 0.388 |
| 8-bit readout row | 0.559 | 0.388 |

Controlled post-online repetition:

| setting | trace+inhib repeat-2 | apical+inhib repeat-2 |
|---|---:|---:|
| full precision | 0.144 | 0.106 |
| 8-bit all row | 0.048 | 0.122 |
| 8-bit readout row | 0.032 | 0.106 |

The generation behavior of readout-targeted quantization matches all-state row closely. The apical repetition benefit survives, but CE calibration is not full precision.

## Wider Clip Sanity

Hypothesis: `output_bias` stores log-priors, so fixed `clip=1.0` clips values around `[-5, 0]`. A quick sanity test widened the global clip to `8.0`:

| setting | method | post CE / acc |
|---|---|---:|
| 8-bit tensor clip8 all | trace+inhib | 2.684 / 0.446 |
| 8-bit tensor clip8 all | apical+inhib | 2.570 / 0.455 |
| 8-bit row clip8 all | trace+inhib | 2.662 / 0.451 |
| 8-bit row clip8 all | apical+inhib | 2.539 / 0.466 |

Widening the global clip does not fix the all-state audit. It prevents bias clipping but lowers resolution for small floating state under tensor mode and still leaves cumulative projection/update noise under row mode.

## Dedicated Bias Clip Sanity

R055 localized much of the damage to `output_bias`, so the wrapper was extended with `--low-precision-bias-clip`. This uses a separate clip only when quantizing `output_bias`; all other arrays keep `--low-precision-clip`.

Command change:

```bash
--low-precision-bits 8 \
--low-precision-clip 1.0 \
--low-precision-bias-clip 8.0 \
--low-precision-scale-mode row \
--low-precision-targets all
```

Result:

| setting | method | post CE / acc |
|---|---|---:|
| 8-bit tensor + bias clip8 | trace+inhib | 2.672 / 0.441 |
| 8-bit tensor + bias clip8 | apical+inhib | 2.555 / 0.458 |
| 8-bit row + bias clip8 | trace+inhib | 2.654 / 0.450 |
| 8-bit row + bias clip8 | apical+inhib | 2.525 / 0.462 |

This does not recover full precision. It is nearly identical to all-state row without the dedicated bias clip (`2.523`). Therefore, `output_bias` is a sufficient damage source when quantized alone, but the all-state projection also has cumulative multi-array perturbation. A single special clip is not the correct final fix.

## Phase-Side Isolates

To separate vector state from count-like state, the phase group was split into `phase_codes`, `phase_prototypes`, `counts`, and `phase_banks`. All use 8-bit row scaling with `--low-precision-clip 1.0`.

| target group | method | post CE / acc | interpretation |
|---|---|---:|---|
| phase_codes | trace+inhib | 2.358 / 0.428 | close to full precision |
| phase_codes | apical+inhib | 2.290 / 0.437 | close to full precision |
| phase_prototypes | trace+inhib | 2.358 / 0.429 | close to full precision |
| phase_prototypes | apical+inhib | 2.289 / 0.437 | close to full precision |
| counts | trace+inhib | 2.645 / 0.450 | reproduces most CE damage |
| counts | apical+inhib | 2.513 / 0.460 | reproduces most CE damage |
| phase_banks | trace+inhib | 2.646 / 0.451 | same as counts |
| phase_banks | apical+inhib | 2.513 / 0.459 | same as counts |

Conclusion: unit-normalized vector/matrix state is mostly 8-bit row tolerant. Frequency/count-like state (`prototype_counts`, `unigram_counts`) behaves like `output_bias`: it needs a different representation.

## Interpretation

Positive:

- The low-precision damage is not caused by fixed random codes, apical feedback, short-term activity, or inhibition alone.
- The core WTA/readout weights alone tolerate 8-bit row quantization well: apical CE `2.295` vs full `2.289`.
- Phase code banks and phase prototypes also tolerate 8-bit row quantization well.
- The apical advantage survives every selective quantization condition.
- Dedicated `output_bias` clip is now implemented and verified as a negative/insufficient fix, which narrows the next step.

Boundary:

- `output_bias` quantization alone reproduces most of the CE gap. It is a log-prior calibration variable, not a unit-norm neural weight vector, so it needs a separate scale/range policy.
- `counts` quantization also reproduces most of the CE gap. Count-like accumulators are not unit-normalized neural vectors and should not share the same 8-bit row rule.
- Quantizing broad groups that include `output_bias` or count-like arrays makes `phase`, `readout`, and `plastic` appear damaged; the fine split shows this was mostly a calibration/bias/count issue.
- Simply widening `--low-precision-clip` or adding `--low-precision-bias-clip` is not enough for all-state quantization.
- The cached wrapper is still a projection audit over float arrays; it is not yet true serialized integer storage.

Conclusion:

The next no-BP compression step should use variable-type-aware quantization:

1. Keep `output_bias`, `prototype_counts`, and `unigram_counts` high precision or quantize them with dedicated log/count-scale ranges.
2. Quantize WTA/readout weights and inhibition with per-row 8-bit scales.
3. Move from projection audit to true serialized integer arrays plus scale metadata.

This remains a pure no-BP framework result. No BP-pretrained model, API backbone, or statistical token-probability method is used as the learning method.
