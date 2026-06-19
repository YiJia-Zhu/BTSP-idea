# R179 Conflict Feature Separability

**Date**: 2026-06-19

## Goal

R178 showed that a disagreement-only binary prototype arbiter still over-selects
`base_only` on QA14/QA17. R179 asks whether the conflict cases are separable by
simple local inference-time features before adding another controller.

The tested features are available at inference:

- `base_only` top-2 margin;
- `base_plus_direct_joint` top-2 margin;
- margin difference `base_margin - joint_margin`.

No task labels or raw text are used by the scanned rules.

## Inputs

R179 uses the single-task R177 diagnostic rows:

- `output/babi_branch_arbitration_r177_qa14_joint_agreement/branch_rows.csv`
- `output/babi_branch_arbitration_r177_qa17_joint_agreement/branch_rows.csv`
- `output/babi_branch_arbitration_r177_qa18_joint_agreement/branch_rows.csv`

It filters to base-vs-joint conflicts where:

`base_only_pred != base_plus_direct_joint_pred`

and scans threshold rules over margins.

## Outputs

- `output/babi_branch_arbitration_r179_conflict_feature_scan/conflict_feature_summary.csv`
- `output/babi_branch_arbitration_r179_conflict_feature_scan/threshold_scan.csv`
- `output/babi_branch_arbitration_r179_conflict_feature_scan/best_aggregate_rules.csv`

## Conflict Summary

Single-task conflicts:

| Task | Conflicts | Base wins | Joint wins | Both wrong | Mean base margin | Mean joint margin |
|---|---:|---:|---:|---:|---:|---:|
| QA14 | 668 | 90 | 266 | 312 | low | higher |
| QA17 | 228 | 93 | 135 | 0 | moderate | higher |
| QA18 | 34 | 24 | 10 | 0 | higher | higher |

QA18 is the only task where base wins more often than joint on conflicts.
QA14 has many both-wrong conflicts, and when exactly one side wins, joint wins
far more often.

## Threshold Scan

Rules scanned:

- `base_margin >= t`
- `margin_diff >= d`
- `base_margin >= t and margin_diff >= 0`
- `base_margin >= t and margin_diff >= d`

Aggregate result across QA14/QA17/QA18: no simple margin rule with at least 10
selected conflicts has positive net accuracy if it selects `base_only`.

Best aggregate rules are still negative. For example:

| Rule | Selected | Base wins | Joint wins | Precision | Net delta |
|---|---:|---:|---:|---:|---:|
| `base_margin >= 0.25` | 14 | 4 | 10 | 0.286 | -6 |
| `base_margin >= 0.25 and margin_diff >= -0.50` | 14 | 4 | 10 | 0.286 | -6 |

Per-task best rules confirm the issue:

| Task | Best simple-rule behavior |
|---|---|
| QA14 | all threshold rules with coverage >=5 are negative for base fallback |
| QA17 | all threshold rules with coverage >=5 are negative for base fallback |
| QA18 | selecting all conflicts gives `+14` net accuracy, but this is task-specific |
| All | every simple margin rule with meaningful coverage is negative |

## Interpretation

Simple confidence geometry is not enough. The same local margin features that
would help QA18 select base also select harmful base fallbacks in QA14/QA17.
This explains why R176/R177 hand-written thresholds and R178 conflict prototypes
failed: the currently exposed feature space does not cleanly separate the
desired high-precision base fallback cases across tasks.

The next useful step is not another scalar threshold. It should either:

1. add richer conflict features, such as relation between predicted token codes,
   role-event evidence distribution, and suppression trace activity; or
2. require repeated evidence for the same conflict pattern before overriding
   the rescue path, making base fallback high precision and low recall.

## Boundary

R179 rejects simple margin-only separability for base-vs-rescue conflict
arbitration. Future branch controllers should treat base fallback as a rare,
high-precision exception, not a broad confidence-threshold rule.
