# 2026-06-17 Low-Precision And Generation Audit For Dynamic Apical Error

R052 identified dynamic target-vs-wrong apical error as the useful signal, with `random_feedback` and `global_margin` slightly beating branch-local segment margins. This round checks two practical constraints:

1. Does the `random_feedback` candidate keep the R051 generation/repetition improvement?
2. Does a simple low-precision projection preserve the learned behavior enough for a hardware-friendly route?

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New wrapper and CLI:

- `LowPrecisionStateWrapper`
- `--low-precision-bits`
- `--low-precision-clip`

The wrapper recursively quantizes floating NumPy state after each online update and reports compressed state bytes as `base_state_bytes * bits / 32`. It skips statistical memory rows and is intended as an audit/simulation layer, not an optimized quantized implementation.

No BP/BPTT, pretrained backbone, API main model, or raw-text replay is introduced.

## Full-Precision Generation Rerun

Command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_random_generation_medium \
  --method-filter phase_trace_inhib phase_trace_apical_inhib \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --apical-gating-branch --apical-error-mode random_feedback \
  --apical-decay 0.85 --apical-strength 0.15 \
  --apical-margin 0.0 --apical-min-gate 0.8 \
  --apical-max-gate 1.25 --apical-error-clip 1.0 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --completion-count 4 --prompt-tokens 16 --completion-tokens 48
```

Result:

| method | online CE / acc | post CE / acc | greedy repeat-2 | controlled repeat-2 | state bytes |
|---|---:|---:|---:|---:|---:|
| `phase_trace_inhib_competitive_online` | 3.114 / 0.335 | 2.358 / 0.429 | 0.473 | 0.144 | 2,761,728 |
| `phase_trace_apical_inhib_competitive_online` | 3.179 / 0.336 | 2.289 / 0.437 | 0.383 | 0.106 | 2,761,752 |

Interpretation: `random_feedback` preserves the R051 generation benefit. Greedy repetition improves and controlled decoding also improves, while CE/acc remain the strongest pure no-BP point so far.

## Low-Precision Audit

8-bit command adds:

```bash
--low-precision-bits 8 --low-precision-clip 1.0
```

16-bit command adds:

```bash
--low-precision-bits 16 --low-precision-clip 1.0 --completion-count 0
```

Results:

| setting | method | online CE / acc | post CE / acc | greedy repeat-2 | controlled repeat-2 | reported state bytes |
|---|---|---:|---:|---:|---:|---:|
| full | trace+inhib | 3.114 / 0.335 | 2.358 / 0.429 | 0.473 | 0.144 | 2,761,728 |
| full | apical+inhib | 3.179 / 0.336 | 2.289 / 0.437 | 0.383 | 0.106 | 2,761,752 |
| 8-bit | trace+inhib | 3.452 / 0.314 | 2.728 / 0.427 | 0.404 | 0.106 | 690,432 |
| 8-bit | apical+inhib | 3.447 / 0.317 | 2.713 / 0.431 | 0.404 | 0.122 | 690,438 |
| 16-bit | trace+inhib | 3.397 / 0.319 | 2.635 / 0.455 | n/a | n/a | 1,380,864 |
| 16-bit | apical+inhib | 3.387 / 0.333 | 2.504 / 0.460 | n/a | n/a | 1,380,876 |

## Interpretation

Positive:

- The full-precision `random_feedback` candidate keeps the CE gain and improves repetition:
  - CE: `2.358 -> 2.289`
  - acc: `0.429 -> 0.437`
  - greedy repeat-2: `0.473 -> 0.383`
  - controlled repeat-2: `0.144 -> 0.106`
- Low precision preserves the relative apical advantage over trace+inhibition:
  - 8-bit: `2.728 -> 2.713`
  - 16-bit: `2.635 -> 2.504`
- Reported state bytes drop by the expected factors:
  - 8-bit: about 2.76MB -> 0.69MB
  - 16-bit: about 2.76MB -> 1.38MB

Boundary:

- The naive quantization wrapper is not lossless. CE calibration degrades substantially at 8-bit and still degrades at 16-bit.
- 16-bit improves top-1 accuracy but worsens CE, so quantization changes calibration rather than simply compressing the same model.
- `pickle_state_bytes` remains full-size because the wrapper simulates quantization in float arrays; true storage compression would require serialized integer/low-precision arrays.
- 8-bit generation repetition is similar between trace+inhib and apical+inhib, so low precision weakens the generation advantage even though it preserves a small CE/acc advantage.

Conclusion:

Dynamic apical error with random feedback is now the best pure no-BP online token candidate, and it improves both CE and repetition in full precision. Low precision is promising for relative robustness but needs a better quantization-aware update rule or per-state scaling before it can be claimed as hardware-ready compression.

Next step: implement quantization-aware scaling for the dominant matrices (`weights`, `inhibition`, phase prototypes) or sparse/low-rank inhibition, then rerun 8/12/16-bit audits with generation.
