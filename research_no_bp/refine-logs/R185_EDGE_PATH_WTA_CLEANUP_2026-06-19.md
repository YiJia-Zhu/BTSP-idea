# R185: Edge-Path WTA Cleanup Boundary

Date: 2026-06-19

## Purpose

R184 showed that parser-free edge-path answer-slot features improve QA19 CE and
slightly improve exact full-answer accuracy, but the result remains weak. R185
tests whether replacing shared-neighbor averaging with a local winner-take-all
candidate cleanup over intermediate path tokens improves path selection.

This remains a pure no-BP mechanism probe. It uses compact token IDs, local
edge-event features, recurrent/role traces, and local prototype updates only. It
does not use a parser, symbolic graph labels, raw prompt replay, pretrained
weights, or BP.

## Implementation

Changed `babi_unified_token_qa_experiment.py`.

Added `--answer-slot-feature-mode edge_path_wta`.

New local pieces inside `OnlineLocalRoleTransitionMemory`:

- `anchor_edge_bundle_map`: local edge map that preserves support strength.
- `edge_path_candidates`: builds candidate intermediate path states.
- `edge_path_wta_state`: selects one or a few intermediate candidates with
  support plus optional learned cleanup score.
- `edge_path_cleanup_*`: bounded local prototype bank for candidate
  intermediate token cleanup.
- `update_answer_slot_feature`: answer-slot training hook for local
  reward/modulatory cleanup updates.

The original R184 `edge_path` mode is unchanged and remains reproducible.

## Runs

Smoke:

- `output/babi_unified_qa19_r185_edge_path_wta_aslot_smoke`

Medium, same R184 QA19 setting:

- `output/babi_unified_qa19_r185_edge_path_wta_aslot_medium_s2`
- `output/babi_unified_qa19_r185_edge_path_wta_supportonly_medium_s2`
- `output/babi_unified_qa19_r185_edge_path_wta_support_top2_medium_s2`
- comparison: `output/babi_unified_qa19_r185_edge_path_wta_comparison`

All medium runs use `train-limit=300`, `eval-limit=300`, `state_dim=64`,
`state_order=224`, R174-style role-transition, answer-slot readout, and full
multi-token QA19 evaluation.

## Results

`output/babi_unified_qa19_r185_edge_path_wta_comparison/comparison_summary.csv`:

| variant | split | first acc | full acc | full-token acc | full CE | state MB |
|---|---|---:|---:|---:|---:|---:|
| R184 edge average | validation | 0.2700 | 0.1100 | 0.2900 | 1.3179 | 35.39 |
| R184 edge average | test | 0.2867 | 0.1200 | 0.3300 | 1.2802 | 35.39 |
| R185 learned WTA | validation | 0.2600 | 0.0900 | 0.2950 | 1.3423 | 36.40 |
| R185 learned WTA | test | 0.2900 | 0.1133 | 0.3283 | 1.2876 | 36.40 |
| R185 support WTA | validation | 0.3100 | 0.1300 | 0.3000 | 1.3268 | 36.40 |
| R185 support WTA | test | 0.2967 | 0.1033 | 0.3267 | 1.2914 | 36.40 |
| R185 support top2 | validation | 0.2300 | 0.1000 | 0.2650 | 1.3184 | 36.40 |
| R185 support top2 | test | 0.2800 | 0.1133 | 0.3300 | 1.2865 | 36.40 |

Cleanup stats:

| variant | wall s | cleanup score scale | top k | active cleanup slots | avg candidates | raw text stored |
|---|---:|---:|---:|---:|---:|---|
| R185 learned WTA | 104.32 | 0.75 | 1 | 56 | 5.85 | false |
| R185 support WTA | 109.55 | 0.00 | 1 | 72 | 5.85 | false |
| R185 support top2 | 104.43 | 0.00 | 2 | 72 | 5.85 | false |

## Interpretation

R185 is a boundary/negative result relative to R184.

The learned cleanup score is not helpful in the current feature space:

- validation full CE worsens `1.3179 -> 1.3423`
- test full CE worsens `1.2802 -> 1.2876`
- validation exact drops `0.1100 -> 0.0900`
- test exact drops `0.1200 -> 0.1133`

Support-only WTA gives a tempting validation exact score:

- validation full exact reaches `0.1300`, matching the R181 validation majority
  baseline.

But it does not generalize:

- test full exact drops `0.1200 -> 0.1033`, below the R181 test majority
  baseline `0.1100`.
- test full CE worsens `1.2802 -> 1.2914`.

Top-2 support mixing recovers CE close to R184 but still does not improve exact
answer accuracy.

## Boundary

This does not support the hypothesis that hard WTA over a single intermediate
token is the right next step for QA19. The evidence points the other way:
R184's softer shared-edge composition is still the best current QA19 role
variant.

The most plausible failure mode is that compact-token edge candidates are too
ambiguous for one intermediate winner. A useful mechanism likely needs soft
multi-candidate routing or a separate local consistency signal, not a hard
candidate cleanup selected before answer-token readout is reliable.

## Next

Do not continue with hard `edge_path_wta` as the default QA19 route.

Next mechanism candidates:

1. Soft multi-candidate edge mixture with learned reliability weights, not hard
   top-1 WTA.
2. Local consistency/closure signal over source, intermediate, and destination
   edge bundles before answer-slot readout.
3. Contrastive candidate cleanup trained from train-only counterfactual path
   alternatives, if it can be generated without parser labels or raw replay.
