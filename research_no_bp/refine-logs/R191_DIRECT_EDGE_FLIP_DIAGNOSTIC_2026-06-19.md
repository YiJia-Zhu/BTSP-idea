# R191: Direct Edge-Path Flip Diagnostic

Date: 2026-06-19

## Purpose

R190 showed that `edge_path_soft_direct` changes QA19 behavior but does not
beat R186 on full-answer exact match. R191 asks whether the direct channel has
localized helpful cases that can be separated from harmful cases.

This is a diagnostic, not a new learning method.

## Implementation

Added two reproducibility utilities:

- `--prediction-row-limit`: controls how many prediction rows are written per
  eval split; `0` writes all evaluated rows.
- `babi_prediction_flip_diagnostic.py`: compares two prediction CSVs by
  `(config, split, example_index, target_answer)` and writes exact helpful /
  harmful flip summaries.

The diagnostic still uses dataset ground truth. It does not compare against
another model as a label source.

## Runs

Full-prediction reruns:

- R186 baseline: `output/babi_unified_qa19_r191_r186_soft_fullpred_s0`
- R190 direct scale 1.0: `output/babi_unified_qa19_r191_r190_direct_s10_fullpred_s0`

Diagnostic output:

- `output/babi_unified_qa19_r191_direct_flip_diagnostic/flip_summary.csv`
- `output/babi_unified_qa19_r191_direct_flip_diagnostic/flip_rows.csv`
- `output/babi_unified_qa19_r191_direct_flip_diagnostic/flip_by_target.csv`
- `output/babi_unified_qa19_r191_direct_flip_diagnostic/flip_top_changes.csv`

## Results

R190 direct scale 1.0 versus R186 soft:

| split | examples | R186 full acc | R190 full acc | delta | helpful | harmful | net | token acc delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_post | 300 | 0.2567 | 0.3767 | +0.1200 | 43 | 7 | +36 | +0.0833 |
| validation | 100 | 0.1300 | 0.1400 | +0.0100 | 4 | 3 | +1 | +0.0050 |
| test | 300 | 0.1267 | 0.1233 | -0.0033 | 14 | 15 | -1 | +0.0050 |

Test target-level pattern:

- helpful exact concentrates on `north west` (+3 net), `south west` (+2), and
  some `west west` cases.
- harmful exact concentrates on `north north` (-6 net) and `east north` (-2).
- Overall test exact is nearly one-for-one: 14 helpful versus 15 harmful.

## Interpretation

The direct path channel is not merely underpowered. It overfits train-post
examples strongly, but on held-out test its exact-answer gains and losses cancel.

Direct evidence still improves token-level accuracy slightly on test
(`+0.0050`), which means it often fixes one of the two direction tokens without
fixing the whole answer sequence. QA19 therefore needs a mechanism that can
protect already-correct full paths or coordinate both answer slots, not just
push independent token scores.

## Boundary

Supported claim:

> The R190 direct channel produces strong train-post gains but no positive
> held-out full-answer exact delta versus R186 on QA19 medium seed0.

Unsupported claims:

- Direct score scale 1.0 is a robust QA19 improvement.
- The direct channel failures are only a calibration issue.
- First-token gains imply full-answer success.

## Next

The next mechanism should be a local inhibitory/arbitration circuit. The most
minimal test is to protect high-margin pre-direct answer readouts and require a
positive direct-delta margin before adding direct scores.
