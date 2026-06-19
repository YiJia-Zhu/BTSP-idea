# R189: QA19 Full-Limit Soft Edge Boundary

Date: 2026-06-19

## Purpose

R187 found a small three-seed test improvement for fixed
`edge_path_soft t0.20 c0.00` in the medium QA19 setting. R188 showed that naive
validation selection is unreliable. R189 tests whether the fixed soft-edge
setting scales to a larger QA19 run.

This is not a new mechanism. It is a full-limit sanity check for the existing
soft multi-candidate edge routing.

## Setup

Dataset/config:

- bAbI `en-qa19`
- train examples: 900
- validation examples: 100
- test examples: 1000
- seed: 0

Compared systems:

- R184 `edge_path`
- R186 `edge_path_soft t0.20 c0.00`

Both use the same R174-style role-transition backbone, answer-slot readout, and
full multi-token QA19 evaluation. Neither stores raw prompt text or uses BP.

## Runs

- `output/babi_unified_qa19_r189_r184_edge_path_full_s0`
- `output/babi_unified_qa19_r189_r186_soft_t020_c000_full_s0`
- comparison: `output/babi_unified_qa19_r189_full_limit_comparison`

## Results

`output/babi_unified_qa19_r189_full_limit_comparison/comparison_summary.csv`:

| variant | split | evaluated | first acc | full acc | full-token acc | full CE |
|---|---|---:|---:|---:|---:|---:|
| R184 edge | validation | 100 | 0.2300 | 0.1000 | 0.2850 | 1.2966 |
| R186 soft | validation | 100 | 0.2600 | 0.1000 | 0.3000 | 1.2939 |
| delta | validation | 100 | +0.0300 | +0.0000 | +0.0150 | -0.0028 |
| R184 edge | test | 1000 | 0.2700 | 0.0980 | 0.3135 | 1.2837 |
| R186 soft | test | 1000 | 0.2570 | 0.0990 | 0.3065 | 1.2905 |
| delta | test | 1000 | -0.0130 | +0.0010 | -0.0070 | +0.0068 |

Medium seed0 reference:

| setting | split | full acc delta | full CE delta |
|---|---|---:|---:|
| R186 soft minus R184 | validation | +0.0200 | +0.0020 |
| R186 soft minus R184 | test | +0.0067 | +0.0016 |

## Interpretation

R189 is a full-limit boundary result.

The fixed soft edge setting does not scale cleanly:

- Full test exact improves only `0.0980 -> 0.0990`.
- Full test CE worsens `1.2837 -> 1.2905`.
- Full-token accuracy worsens `0.3135 -> 0.3065`.

Validation is mixed:

- Full exact is unchanged.
- CE improves slightly.
- First-token accuracy improves, but first-token accuracy is not a valid QA19
  success criterion.

This means the medium-set exact gain from R187 does not become a meaningful
full-test improvement. Soft multi-candidate routing remains a useful mechanism
direction, but this feature-only implementation is not enough.

## Boundary

Supported claim:

> On full-limit QA19 seed0, `edge_path_soft t0.20 c0.00` is roughly neutral on
> exact answer accuracy and worse on full-answer CE compared with R184.

Unsupported claims:

- Soft edge feature alone solves QA19.
- Medium-set exact improvements scale to full test.
- Larger training data automatically improves the current no-BP QA19 circuit.

## Next

The next QA19 mechanism should not be another temperature-only feature sweep.
More plausible directions:

1. Add a local consistency score channel directly into the answer-slot readout.
2. Add train-only counterfactual path candidates to teach the slot readout which
   soft edges are misleading.
3. Investigate why full train lowers test exact relative to the 300-example
   medium setting.
