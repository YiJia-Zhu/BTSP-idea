# R162 Role-Gate Center-Difference Diagnostic

**Date**: 2026-06-19

## Goal

R161 improved unified bAbI QA2/QA3 with a parser-free local role-transition
branch. R162 checks whether the local role-gate credit update is actually close
to a center-difference negative-gradient direction, or whether it is better
understood as a useful but non-gradient local rule.

This is a diagnostic only. Center difference is not used for training, model
selection, or update application.

## Implementation

Added `babi_role_gate_alignment_diagnostic.py` and refactored
`OnlineLocalRoleTransitionMemory` so the role gate update can be inspected
without changing the training rule:

- `compute_role_gate_delta(...)` returns the local no-BP gate delta.
- `apply_role_gate_delta(...)` applies an already computed local delta.
- `update_role_gate(...)` preserves the original training behavior.

The diagnostic samples bAbI QA rows, computes the local role-gate delta, then
estimates selected gate dimensions with centered finite differences on answer
loss. It reports cosine alignment, sign agreement, and one-step loss change for
the local delta, the scaled center-difference direction, and a random direction.

## Commands

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_role_gate_alignment_diagnostic.py \
  --out-dir output/babi_role_gate_alignment_r162_qa2_smoke \
  --configs en-qa2 --train-limit 80 --diagnostic-limit 8 \
  --max-diff-dims 48 --center-eps 0.001 --seed 0
```

Larger diagnostic:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_role_gate_alignment_diagnostic.py \
  --out-dir output/babi_role_gate_alignment_r162_qa2_20 \
  --configs en-qa2 --train-limit 160 --diagnostic-limit 20 \
  --max-diff-dims 64 --center-eps 0.001 --seed 0
```

## Results

| Run | Rows | Cosine mean | Sign agreement | Local loss change | Center-diff loss change | Random loss change | Local improves | Center-diff improves | Random improves |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 8 | 0.0734 | 0.5506 | -0.0116 | -0.0290 | -0.0000 | 6/8 | 8/8 | 4/8 |
| qa2_20 | 20 | 0.1707 | 0.5768 | -0.0083 | -0.0324 | -0.0039 | 16/20 | 20/20 | 12/20 |

Output files:

- `output/babi_role_gate_alignment_r162_qa2_smoke/role_gate_alignment_summary.csv`
- `output/babi_role_gate_alignment_r162_qa2_smoke/role_gate_alignment_rows.csv`
- `output/babi_role_gate_alignment_r162_qa2_20/role_gate_alignment_summary.csv`
- `output/babi_role_gate_alignment_r162_qa2_20/role_gate_alignment_rows.csv`

## Findings

1. The local role-gate update is useful: it reduces answer loss on `16/20`
   diagnostic rows in the larger run, much more consistently than the random
   direction baseline.

2. It is not a strong gradient approximation. Mean cosine alignment with the
   center-difference negative-gradient direction is only `0.1707`, and several
   rows have weak or negative alignment.

3. Center difference remains a stronger oracle on this diagnostic: it improves
   `20/20` rows and has a larger mean one-step loss decrease. This is expected
   because it directly probes the current scalar loss.

4. The result supports a conservative interpretation of R160/R161: the
   role-transition circuit has a locally useful third-factor gate, but the
   current mechanism should not be claimed as a BP-equivalent or close finite
   difference approximation.

## Interpretation

R162 gives a boundary around the "does it approximate BP?" question. The local
rule is partially aligned and behaviorally helpful, but it is qualitatively
different from the finite-difference loss gradient. That is acceptable for the
project goal because the final route is a pure no-BP biomimetic framework, not a
hidden BP estimator.

Next steps should focus on mechanisms, not closer finite-difference matching:

- make role-transition runtime cheaper with sparse event caching;
- repeat QA3 seeds after runtime is under control;
- add state/channel separation that can represent object-carrier-location
  updates more explicitly while preserving the unified next-token interface;
- keep center-difference diagnostics as an audit tool only.
