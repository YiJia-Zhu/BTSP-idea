# R183: Role-Hop Answer-Slot Feature

Date: 2026-06-19

## Purpose

R182 showed that answer-slot readout improves full-answer CE but does not solve
QA19 exact match. R183 tests whether answer slot 0/1 should read separate
role-transition hops instead of the full concatenated role feature.

The motivation is QA19's two-direction answer format: slot 0 and slot 1 may
benefit from different local transition hops. This is still parser-free and
no-BP: features come from token/position codes and local role-transition traces.

## Implementation

Changed `babi_unified_token_qa_experiment.py`.

New CLI:

- `--answer-slot-feature-mode {base,role_hop}`

Default is `base`, preserving R182 behavior. `role_hop` is only active when the
wrapped memory exposes `answer_slot_feature`.

New role-transition method:

- `OnlineLocalRoleTransitionMemory.answer_slot_feature(context, slot)`

For `role_hop`, the answer-slot wrapper uses:

- slot 0: base recurrent state + role hop 0 state
- slot 1: base recurrent state + role hop 1 state
- slots beyond available hops: last available hop

## Runs

Smoke:

- `output/babi_unified_qa19_r183_role_hop_aslot_smoke`

Medium:

- `output/babi_unified_qa19_r183_role_hop_aslot_medium_s2`
- comparison: `output/babi_unified_qa19_r183_role_hop_comparison`

The medium run uses the same R174-style role-transition setting as R181/R182,
with `--answer-slot-readout --answer-slot-feature-mode role_hop`.

## Results

`output/babi_unified_qa19_r183_role_hop_comparison/comparison_summary.csv`:

| variant | split | first acc | first CE | full acc | full-token acc | full CE | state MB |
|---|---|---:|---:|---:|---:|---:|---:|
| role no-slot | validation | 0.2600 | 1.4465 | 0.0700 | 0.2650 | 1.3649 | 27.32 |
| role aslot base | validation | 0.2700 | 1.4064 | 0.0700 | 0.2800 | 1.3207 | 39.39 |
| role aslot hop | validation | 0.2700 | 1.4066 | 0.0700 | 0.2800 | 1.3193 | 35.39 |
| role no-slot | test | 0.2800 | 1.4295 | 0.1167 | 0.3350 | 1.3317 | 27.32 |
| role aslot base | test | 0.2767 | 1.3914 | 0.1167 | 0.3283 | 1.2882 | 39.39 |
| role aslot hop | test | 0.2733 | 1.3920 | 0.1133 | 0.3333 | 1.2872 | 35.39 |

Slot-feature stats:

| variant | wall s | feature mode | feature dim | active slots | raw text stored |
|---|---:|---|---:|---:|---|
| role aslot base | 155.84 | base | 192 | 256 | false |
| role aslot hop | 104.32 | role_hop | 128 | 256 | false |

## Interpretation

`role_hop` is a useful efficiency/CE boundary, not an exact-answer improvement.

Compared with R182's base answer-slot feature:

- validation full CE improves slightly: `1.3207 -> 1.3193`
- test full CE improves slightly: `1.2882 -> 1.2872`
- state drops by about 4 MB because slot feature dim is `128` instead of `192`
- wall time drops from `155.84s` to `104.32s`

But full-answer exact match does not improve:

- validation exact: unchanged at `0.0700`
- test exact: `0.1167 -> 0.1133`

Thus, slot-specific role-hop features are cleaner and cheaper, but still do not
perform the missing path composition needed by QA19.

## Boundary

R183 should not be treated as solving QA19. It supports a narrower mechanism
claim: answer slots can read hop-local no-BP traces with lower memory/runtime and
slightly better CE, but exact path inference remains absent.

## Next

The next step needs an edge-event composition memory rather than a different
readout feature. Specifically, learn local edge traces keyed by source/location,
direction, and destination token neighborhoods, then perform a query-seeded
two-hop rollout before answer-slot readout.
