# R194 QA19 Component-Margin Diagnostic

**Date**: 2026-06-19  
**Status**: DONE-DIAGNOSTIC  
**Question**: R193 answer-slot coupling improves QA19 full-answer CE but not held-out full exact. Is it raising the target evidence without changing the winner, or is it introducing confident wrong winners?

## Implementation

- Added `--prediction-component-margins` to `babi_unified_token_qa_experiment.py`.
- When enabled, the evaluator writes `prediction_components.csv` with one row per answer token, for both `teacher_forced` and `greedy` decode phases.
- Each row decomposes answer-slot scoring into `base`, `slot_delta`, `after_slot`, `coupling_delta`, `after_coupling`, `direct_delta`, and `final` summaries: top id, margin, target score, target-vs-best-wrong margin, target rank, active count, and target probability where meaningful.
- Added `babi_component_margin_diagnostic.py` to align two component CSVs and write:
  - `component_comparison_rows.csv`
  - `component_summary.csv`
  - `missing_rows.csv`

## Reproduction

Exact R186/R193 reruns were required because an initial R194 rerun omitted the R174/R186 role-transition branch config and was discarded for conclusions. The accepted reruns match the old configs except for `out_dir`, `prediction_component_margins`, and default-off new args in the R186 config.

Outputs:

- R186 component run: `output/babi_unified_qa19_r194_r186_soft_components_s0_exact`
- R193 component run: `output/babi_unified_qa19_r194_r193_coupling_components_s0_exact`
- Component diagnostic: `output/babi_unified_qa19_r194_component_margin_diagnostic`
- Sequence flip diagnostic: `output/babi_unified_qa19_r194_sequence_flip_diagnostic`

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  babi_unified_token_qa_experiment.py \
  babi_component_margin_diagnostic.py
```

## Sequence-Level Result

R193 coupling remains exact-neutral on held-out QA19:

| Split | R186 full exact | R193 full exact | Delta | R186 token acc | R193 token acc | Helpful/Harmful exact |
|---|---:|---:|---:|---:|---:|---:|
| train_post | 0.2567 | 0.2733 | +0.0167 | 0.5283 | 0.5300 | 6 / 1 |
| validation | 0.1300 | 0.1300 | 0.0000 | 0.3100 | 0.3000 | 1 / 1 |
| test | 0.1267 | 0.1267 | 0.0000 | 0.3300 | 0.3250 | 1 / 1 |

Slot0 is unchanged by design, so first-token flip rate is 0. The effect is entirely in slot1.

## Slot1 Component Findings

Held-out test, slot1:

| Decode phase | R186 slot1 acc | R193 slot1 acc | Delta | Mean target prob delta | Mean target-vs-best delta | Helpful/Harmful token flips | High-margin wrong >=0.20 / >=0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher_forced | 0.3633 | 0.3533 | -0.0100 | +0.00866 | -0.01905 | 7 / 10 | 114 / 40 |
| greedy | 0.3100 | 0.3133 | +0.0033 | +0.00162 | -0.15978 | 3 / 2 | 123 / 39 |

Detailed failure pattern:

- In test teacher-forced slot1, among 184 examples where both R186 and R193 are wrong, R193 raises the target probability on 125 examples, but only improves target-vs-best margin on 60. There are 69 cases where probability improves while the target loses more badly relative to the winner.
- In test greedy slot1, among 204 both-wrong rows, target probability rises on 78, but target-vs-best margin improves on only 42.
- Coupling is active on all slot1 rows with mean 4 active coupling candidates, so this is not an inactive-channel problem.
- The coupling delta target-vs-best signal is negative on average on held-out test (`-0.0361` teacher-forced, `-0.1833` greedy), meaning the coupling bank often adds score to plausible but wrong second-direction tokens more than to the target.

## Interpretation

R193 is CE-positive because it moves probability mass toward the true second token under teacher-forced scoring. It is exact-neutral because the same local coupling also strengthens wrong high-margin winners. The mechanism has a calibration/winner-selection failure, not a lack of slot1 activity.

This also explains why scaling direct or coupling evidence is risky: more evidence can improve CE while widening the wrong top-1 gap. The next change should be a local slot1 cleanup/suppression mechanism trained on wrong winners, not another score-scale sweep.

## Next Step

R195 should test a default-off slot1 near-miss cleanup:

- During online answer-slot updates, when the slot1 target loses, store a local inhibitory prototype for the wrong winner under the coupling feature.
- At inference, apply the suppression only for slot1 when coupling evidence is active and the wrong-winner prototype matches.
- Evaluate whether high-margin wrong rows decrease without hurting the small number of helpful coupling flips.

This remains a pure no-BP local plasticity mechanism and does not use raw-text storage or pretrained models.
