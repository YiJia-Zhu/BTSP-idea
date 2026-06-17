# 2026-06-17 Serialized Variable-Type State For Apical No-BP Learner

R055 localized the 8-bit failure to count/prior-like state (`output_bias`, `prototype_counts`, `unigram_counts`) rather than unit-normalized vector/matrix state. This round moves the low-precision audit closer to deployable state accounting: vector/matrix arrays are projected as 8-bit row-scaled state, while count/prior arrays remain float32.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New output fields:

- `serialized_state_bytes`
- `serialized_bytes_per_target`
- `serialized_manifest`

`LowPrecisionStateWrapper` now exports a serialized-state manifest for selected arrays. The estimate uses:

- selected float arrays: int8 data bytes plus float32 scale metadata
- row mode on 2D arrays: one float32 scale per row
- selected vectors: one float32 scale
- unselected count/prior arrays: raw float32 bytes

This is still an accounting/export manifest, not yet a custom binary checkpoint loader.

## Variable-Type Candidate

Command pattern:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_random_lowp8_varstate_medium \
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
  --low-precision-bits 8 --low-precision-clip 1.0 \
  --low-precision-scale-mode row \
  --low-precision-targets phase_codes phase_prototypes readout_weights inhibition fixed dynamic
```

The target list deliberately excludes:

- `bias`
- `counts`

So `output_bias`, `prototype_counts`, and `unigram_counts` stay float32.

## Medium Results

| setting | method | post CE / acc | state bytes | serialized bytes | pickle bytes |
|---|---|---:|---:|---:|---:|
| full precision | trace+inhib | 2.358 / 0.429 | 2,761,728 | 2,761,728 | 2,764,112 |
| full precision | apical+inhib | 2.289 / 0.437 | 2,761,752 | 2,761,752 | 2,764,568 |
| all-state 8-bit row | trace+inhib | 2.651 / 0.453 | 690,432 | n/a | 2,764,206 |
| all-state 8-bit row | apical+inhib | 2.523 / 0.462 | 690,438 | n/a | 2,764,662 |
| variable-type 8-bit row | trace+inhib | 2.359 / 0.432 | 696,576 | 706,820 | 2,764,580 |
| variable-type 8-bit row | apical+inhib | 2.295 / 0.438 | 696,585 | 706,841 | 2,765,072 |

The variable-type candidate recovers most of the all-state 8-bit CE loss:

- apical full precision CE: `2.289`
- apical all-state 8-bit row CE: `2.523`
- apical variable-type 8-bit row CE: `2.295`

It also preserves the apical advantage:

- trace+inhib variable-type CE: `2.359`
- apical+inhib variable-type CE: `2.295`

## Manifest Breakdown

For `phase_trace_apical_inhib_competitive_online`, the serialized manifest reports:

| array group | serialized bytes | raw bytes | arrays |
|---|---:|---:|---:|
| code_banks | 199,680 | 786,432 | 3 |
| weights | 148,480 | 589,824 | 1 |
| target_anchors | 133,120 | 524,288 | 2 |
| prototypes | 133,120 | 524,288 | 2 |
| inhibition | 66,560 | 262,144 | 1 |
| trace_codes | 17,408 | 65,536 | 1 |
| output_bias | 4,096 | 4,096 | 4 |
| prototype_counts | 2,048 | 2,048 | 2 |
| unigram_counts | 2,048 | 2,048 | 2 |

Manifest path:

`output/phase_binding_online_stream_apical_random_lowp8_varstate_medium/phase_trace_apical_inhib_competitive_online_serialized_state_manifest.json`

## Interpretation

Positive:

- This is the first low-precision point that is close to full precision on CE while keeping an approximately 4x smaller deployable state estimate.
- Unit-normalized vector/matrix state can be row-scaled int8 without major CE loss.
- Count/prior-like state should not share the vector/matrix quantization rule.
- The apical random-feedback mechanism remains useful under this variable-type state design.

Boundary:

- The in-memory Python object still stores float arrays; `pickle_state_bytes` stays about `2.76MB`.
- The new `serialized_state_bytes` is an export/accounting estimate plus manifest, not a loadable integer checkpoint yet.
- Generation was not rerun for this exact variable-type candidate; R054/R055 already showed apical repetition benefit survives related 8-bit row settings.

Conclusion:

The next implementation step should replace the audit wrapper with a true serialized state path:

1. Store selected vectors/matrices as int8 arrays plus row-scale metadata.
2. Store `output_bias`, `prototype_counts`, and `unigram_counts` as float32 or dedicated log/count-scale arrays.
3. Add a loadable checkpoint format and verify prediction parity against the current projection wrapper.

This remains a pure no-BP result: no BP-pretrained model, no API backbone, and no statistical token-count model is used as the final learning method.
