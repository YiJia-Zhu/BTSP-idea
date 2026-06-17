# 2026-06-17 Quantization-Aware Scaling For Dynamic Apical Error

R053 showed that fixed-clip low-precision projection preserves the relative apical advantage but damages CE calibration. This round adds per-tensor and per-row scaling modes to the low-precision audit wrapper and reruns the `random_feedback` apical candidate.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New CLI:

- `--low-precision-scale-mode {fixed,tensor,row}`

Modes:

| mode | Meaning |
|---|---|
| `fixed` | R053 behavior: symmetric quantization with global clip `--low-precision-clip`. |
| `tensor` | Symmetric quantization using each array's current max absolute value, capped by `--low-precision-clip`. |
| `row` | For 2D arrays, use per-row max absolute value; for vectors, use tensor scaling. |

The wrapper still simulates quantization in floating arrays. It is an audit layer, not final compressed serialization.

## Medium Results

Common command template:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_random_lowp<...>_medium \
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
  --low-precision-bits <bits> --low-precision-clip 1.0 \
  --low-precision-scale-mode <mode>
```

| setting | method | online CE / acc | post CE / acc | state bytes |
|---|---|---:|---:|---:|
| full precision | trace+inhib | 3.114 / 0.335 | 2.358 / 0.429 | 2,761,728 |
| full precision | apical+inhib | 3.179 / 0.336 | 2.289 / 0.437 | 2,761,752 |
| 8-bit fixed | trace+inhib | 3.452 / 0.314 | 2.728 / 0.427 | 690,432 |
| 8-bit fixed | apical+inhib | 3.447 / 0.317 | 2.713 / 0.431 | 690,438 |
| 8-bit tensor | trace+inhib | 3.429 / 0.315 | 2.670 / 0.443 | 690,432 |
| 8-bit tensor | apical+inhib | 3.392 / 0.330 | 2.550 / 0.459 | 690,438 |
| 8-bit row | trace+inhib | 3.411 / 0.316 | 2.651 / 0.453 | 690,432 |
| 8-bit row | apical+inhib | 3.405 / 0.333 | 2.523 / 0.462 | 690,438 |
| 16-bit row | trace+inhib | 3.397 / 0.319 | 2.635 / 0.455 | 1,380,864 |
| 16-bit row | apical+inhib | 3.387 / 0.334 | 2.504 / 0.461 | 1,380,876 |

## Generation Check

8-bit row was the best low-precision CE point, so it was rerun with generation enabled:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_random_lowp8_row_generation_medium \
  ... \
  --low-precision-bits 8 --low-precision-scale-mode row \
  --completion-count 4 --prompt-tokens 16 --completion-tokens 48
```

Greedy generation:

| setting | method | repeat-2 | distinct-2 | max token fraction |
|---|---|---:|---:|---:|
| full | trace+inhib | 0.473 | 0.527 | 0.156 |
| full | apical+inhib | 0.383 | 0.617 | 0.104 |
| 8-bit fixed | trace+inhib | 0.404 | 0.596 | 0.089 |
| 8-bit fixed | apical+inhib | 0.404 | 0.596 | 0.089 |
| 8-bit row | trace+inhib | 0.569 | 0.431 | 0.141 |
| 8-bit row | apical+inhib | 0.388 | 0.612 | 0.104 |

Controlled decoding:

| setting | method | repeat-2 | distinct-2 | max token fraction |
|---|---|---:|---:|---:|
| full | trace+inhib | 0.144 | 0.856 | 0.099 |
| full | apical+inhib | 0.106 | 0.894 | 0.089 |
| 8-bit fixed | trace+inhib | 0.106 | 0.894 | 0.068 |
| 8-bit fixed | apical+inhib | 0.122 | 0.878 | 0.068 |
| 8-bit row | trace+inhib | 0.048 | 0.952 | 0.062 |
| 8-bit row | apical+inhib | 0.122 | 0.878 | 0.083 |

## Interpretation

Positive:

- Per-state scaling substantially improves 8-bit CE over fixed clipping:
  - apical+inhib fixed 8-bit CE `2.713`
  - tensor 8-bit CE `2.550`
  - row 8-bit CE `2.523`
- The apical advantage survives all low-precision modes.
- 8-bit row restores the greedy repetition benefit:
  - trace+inhib repeat-2 `0.569`
  - apical+inhib repeat-2 `0.388`
- 8-bit row gives the best low-precision top-1 so far: apical+inhib acc `0.462`.

Boundary:

- Low precision still does not recover full-precision CE `2.289`; even 16-bit row is CE `2.504`.
- The wrapper projects every floating state every update, so it is slower than the base method and not an optimized hardware implementation.
- Controlled decoding metrics are mixed: 8-bit row trace+inhib is more diverse under the external no-repeat/repetition controls, even though apical+inhib is better on CE and greedy repetition.
- State bytes are estimated from bit width, but actual pickle size remains full because arrays are stored as float32.

Conclusion:

Per-row/tensor scaling fixes much of the naive fixed-clip quantization damage, but the no-BP apical learner still needs quantization-aware update dynamics. The next compression step should be targeted, not recursive: quantize only the dominant matrices (`weights`, `inhibition`, phase prototypes) with per-row scales and adjust local learning rates to quantization steps.
