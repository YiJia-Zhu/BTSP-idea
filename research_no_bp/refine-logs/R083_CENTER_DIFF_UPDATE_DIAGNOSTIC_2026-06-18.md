# R083 Center-Difference Update Diagnostic

**Date**: 2026-06-18

## Purpose

Test whether the current no-BP local update direction is at least locally aligned with a loss-reducing direction. This is diagnostic only: center difference is not used as the optimizer or as the final learning rule.

## Implementation

Added:

- `no_bp_update_alignment_diagnostic.py`

Diagnostics:

- `compositional_cue / target_only_phase_codes`: compare one target-only phase factorization update against a center-difference negative-gradient direction over cue phase codes.
- `babi_qa / en-qa1`: compare one local WTA target/wrong-winner update against a center-difference negative-gradient direction over the phase/dendritic readout weights.

Metrics:

- cosine between local update and center-difference negative-gradient direction;
- sign agreement over nonzero components;
- one-step CE change from the local update;
- one-step CE change from same-norm center-difference direction;
- same-norm random direction control.

## Commands

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 python no_bp_update_alignment_diagnostic.py \
  --out-dir output/r083_update_alignment_smoke \
  --tasks all --seeds 0 1 --k-values 4 \
  --batch-size 8 --qa-phase-dim 4 \
  --qa-pretrain-rows 40 --qa-pretrain-epochs 1 \
  --center-eps 0.001
```

Medium:

```bash
PYTHONDONTWRITEBYTECODE=1 python no_bp_update_alignment_diagnostic.py \
  --out-dir output/r083_update_alignment_medium \
  --tasks all --seeds 0 1 2 --k-values 4 8 \
  --batch-size 16 --qa-phase-dim 8 \
  --qa-pretrain-rows 120 --qa-pretrain-epochs 1 \
  --center-eps 0.001
```

## Results

Medium aggregate:

| Task | Variant | Runs | Cosine | Sign agreement | Local dCE | Center-diff dCE | Random dCE |
|---|---|---:|---:|---:|---:|---:|---:|
| bAbI QA | en-qa1 | 3 | 0.559 | 0.655 | -0.402 | -0.510 | +0.013 |
| compositional cue | target_only_phase_codes | 6 | 0.721 | 0.768 | -2.534 | -2.023 | +0.086 |

Per-run medium details:

| Task | Seed | K/classes | Cosine | Sign | Local dCE | Center-diff dCE | Random dCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| compositional | 0 | 4 | 0.718 | 0.844 | -2.157 | -1.516 | +0.331 |
| compositional | 1 | 4 | 0.794 | 0.688 | -2.873 | -2.328 | +0.463 |
| compositional | 2 | 4 | 0.709 | 0.719 | -3.382 | -2.767 | -0.563 |
| compositional | 0 | 8 | 0.680 | 0.773 | -2.112 | -1.762 | +0.235 |
| compositional | 1 | 8 | 0.696 | 0.795 | -2.118 | -1.820 | -0.373 |
| compositional | 2 | 8 | 0.729 | 0.787 | -2.562 | -1.945 | +0.420 |
| bAbI QA1 | 0 | 6 | 0.591 | 0.649 | -0.749 | -0.908 | +0.046 |
| bAbI QA1 | 1 | 6 | 0.505 | 0.653 | -0.246 | -0.353 | +0.021 |
| bAbI QA1 | 2 | 6 | 0.583 | 0.662 | -0.211 | -0.268 | -0.028 |

## Interpretation

R083 gives a positive but bounded diagnostic:

- The no-BP local update is consistently loss-reducing on these local batches.
- The direction has moderate-to-strong alignment with center-difference negative-gradient estimates.
- Random same-norm directions do not show the same average loss reduction.
- This supports the local rule as a plausible learning signal, but it does not solve the R082 multi-hop failure.

The combined R082/R083 reading is therefore precise: the problem is less likely to be "the local rule has no relation to loss reduction" and more likely to be "the current state/role-binding architecture cannot represent and update multi-hop relational state."

## Artifacts

- `output/r083_update_alignment_smoke/summary.csv`
- `output/r083_update_alignment_smoke/aggregate.csv`
- `output/r083_update_alignment_smoke/config.json`
- `output/r083_update_alignment_medium/summary.csv`
- `output/r083_update_alignment_medium/aggregate.csv`
- `output/r083_update_alignment_medium/config.json`
