# R170 Branch Arbiter Boundary

**Date**: 2026-06-19

## Goal

R169 showed that branch-separated role-transition readouts contain useful but
task-dependent evidence: QA14 benefits from direct role drive, QA18 should
mostly protect the base branch, and QA17 needs a better role candidate route.
R170 turns that diagnostic into two local no-BP arbitration mechanisms and
checks whether either resolves the QA14/QA17/QA18 tradeoff.

## Implementation

Extended `babi_unified_token_qa_experiment.py` with optional branch arbiters:

- `--role-branch-arbiter local_proto`: a small prototype/WTA circuit over local
  branch margins, branch agreement flags, and candidate token random codes.
- `--role-branch-arbiter base_margin_adaptive`: a scalar inhibitory threshold
  that chooses `base_only` when base margin is high enough, otherwise allows
  `base_plus_direct`.

Both are default-off. They do not use BP, pretrained models, task-specific QA
heads, raw replay, or decoded prompt text. The bottom role-transition memory is
still trained with the original non-arbitrated branch scores; the arbiter gets
its own local answer-token target/wrong signal.

## Runs

Medium `train-limit=300 eval-limit=300`, seed0:

- `output/babi_unified_role_transition_r170_qa14_medium_arb`
- `output/babi_unified_role_transition_r170_qa17_medium_arb`
- `output/babi_unified_role_transition_r170_qa18_medium_arb`
- `output/babi_unified_role_transition_r170_qa14_medium_base_adapt`
- `output/babi_unified_role_transition_r170_qa17_medium_base_adapt`
- `output/babi_unified_role_transition_r170_qa18_medium_base_adapt`
- `output/babi_unified_role_transition_r170_qa14_medium_base_adapt_m02`
- `output/babi_unified_role_transition_r170_qa17_medium_base_adapt_m02`
- `output/babi_unified_role_transition_r170_qa18_medium_base_adapt_m02`

Full seed0 adaptive m0.20:

- `output/babi_unified_role_transition_r170_qa14_full_base_adapt_m02`
- `output/babi_unified_role_transition_r170_qa17_full_base_adapt_m02`
- `output/babi_unified_role_transition_r170_qa18_full_base_adapt_m02`

## Medium Results

Test split:

| Task | Variant | Accuracy | CE |
|---|---|---:|---:|
| QA14 | R167 branch r8 | 0.340 | 1.6808 |
| QA14 | local proto arbiter | 0.313 | 1.7317 |
| QA14 | base adaptive m0.05 | 0.327 | 1.6948 |
| QA14 | base adaptive m0.20 | 0.330 | 1.6854 |
| QA17 | R167 branch r8 | 0.527 | 0.7001 |
| QA17 | local proto arbiter | 0.533 | 0.6964 |
| QA17 | base adaptive m0.05 | 0.527 | 0.6951 |
| QA17 | base adaptive m0.20 | 0.527 | 0.6961 |
| QA18 | R167 branch r8 | 0.890 | 0.4969 |
| QA18 | local proto arbiter | 0.703 | 0.5282 |
| QA18 | base adaptive m0.05 | 0.890 | 0.4973 |
| QA18 | base adaptive m0.20 | 0.890 | 0.4971 |

## Full Seed0 Results

Test split:

| Task | Variant | Accuracy | CE |
|---|---|---:|---:|
| QA14 | R167 branch r8 | 0.398 | 1.6529 |
| QA14 | R168 branch r4 | 0.385 | 1.6560 |
| QA14 | R170 base adaptive m0.20 | 0.378 | 1.6617 |
| QA17 | R165 role concat | 0.578 | 0.6734 |
| QA17 | R167 branch r8 | 0.512 | 0.7076 |
| QA17 | R170 base adaptive m0.20 | 0.513 | 0.7062 |
| QA18 | R165 microproto | 0.912 | 0.4674 |
| QA18 | R167 branch r8 | 0.909 | 0.4764 |
| QA18 | R168 branch r4 | 0.911 | 0.4759 |
| QA18 | R170 base adaptive m0.20 | 0.909 | 0.4750 |

## Findings

1. `local_proto` is not acceptable as the next main path. It gives a small QA17
   medium gain, but it catastrophically mis-arbitrates QA18 by selecting role or
   base+role paths too often.

2. `base_margin_adaptive` is stable but not a breakthrough. It protects QA18 and
   slightly improves QA17/QA18 CE relative to branch r8 on full seed0, but it
   loses QA14 accuracy and CE.

3. A one-dimensional base-protection threshold is too weak. It cannot express
   the difference between QA14 cases where direct role drive should override
   base and QA18 cases where base should dominate.

## Interpretation

R170 supports the R169 diagnosis but rejects two simple arbiter designs. The
next arbiter should not be a generic branch classifier or a single scalar gate.
It needs a richer local circuit that separately detects:

- direct-role rescue evidence for QA14-like temporal transitions;
- role-only candidate evidence for QA17-like positional yes/no cases;
- high-confidence base protection for QA18-like containment/size comparisons.

The strongest current relation-task point remains:

- QA14: R167 branch r8;
- QA17: R165 role concat;
- QA18: R165 microproto or R168 branch r4.

R170 code remains useful as a default-off branch-arbitration harness, but the
result is **DONE-BOUNDARY**, not a new main result.
