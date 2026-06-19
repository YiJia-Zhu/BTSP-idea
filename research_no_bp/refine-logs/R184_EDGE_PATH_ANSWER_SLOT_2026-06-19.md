# R184: Edge-Path Answer-Slot Feature

Date: 2026-06-19

## Purpose

R182/R183 showed that answer-position readout and role-hop features improve CE
but do not solve QA19 full-answer exact match. R184 adds a parser-free local
edge-path feature to test whether explicit two-anchor edge composition moves the
model in the right direction.

The mechanism does not parse text or store raw prompt strings. It uses token
identity, query/prefix repetition, relative-position event features, and local
eligibility-style prototype updates.

## Implementation

Changed `babi_unified_token_qa_experiment.py`.

Extended `--answer-slot-feature-mode`:

- `base`: R182 behavior, full base/role feature.
- `role_hop`: R183 behavior, slot-specific role hop.
- `edge_path`: new R184 behavior.

New local helpers inside `OnlineLocalRoleTransitionMemory`:

- `ordered_query_anchors`
- `anchor_edge_map`
- `edge_path_state`

The edge-path feature:

1. Splits the prompt into prefix/query using the existing query-order window.
2. Selects two low-frequency query tokens that also occur in the prefix,
   preserving query order. This is a parser-free source/destination anchor
   heuristic.
3. Builds local edge maps around each anchor using token codes and relative
   position codes.
4. Finds shared local-neighbor tokens between the two anchor edge maps.
5. Builds slot 0 from source-side edge events and slot 1 from destination-side
   edge events.

## Runs

Smoke:

- `output/babi_unified_qa19_r184_edge_path_aslot_smoke`

Medium:

- `output/babi_unified_qa19_r184_edge_path_aslot_medium_s2`
- comparison: `output/babi_unified_qa19_r184_edge_path_comparison`

The medium run uses the same R174-style role-transition setting as R181-R183,
with `--answer-slot-readout --answer-slot-feature-mode edge_path`.

## Results

`output/babi_unified_qa19_r184_edge_path_comparison/comparison_summary.csv`:

| variant | split | first acc | first CE | full acc | full-token acc | full CE | state MB |
|---|---|---:|---:|---:|---:|---:|---:|
| role no-slot | validation | 0.2600 | 1.4465 | 0.0700 | 0.2650 | 1.3649 | 27.32 |
| role aslot base | validation | 0.2700 | 1.4064 | 0.0700 | 0.2800 | 1.3207 | 39.39 |
| role aslot hop | validation | 0.2700 | 1.4066 | 0.0700 | 0.2800 | 1.3193 | 35.39 |
| role aslot edge | validation | 0.2700 | 1.4028 | 0.1100 | 0.2900 | 1.3179 | 35.39 |
| role no-slot | test | 0.2800 | 1.4295 | 0.1167 | 0.3350 | 1.3317 | 27.32 |
| role aslot base | test | 0.2767 | 1.3914 | 0.1167 | 0.3283 | 1.2882 | 39.39 |
| role aslot hop | test | 0.2733 | 1.3920 | 0.1133 | 0.3333 | 1.2872 | 35.39 |
| role aslot edge | test | 0.2867 | 1.3866 | 0.1200 | 0.3300 | 1.2802 | 35.39 |

Slot-feature stats:

| variant | wall s | feature mode | feature dim | active slots | raw text stored |
|---|---:|---|---:|---:|---|
| role aslot base | 155.84 | base | 192 | 256 | false |
| role aslot hop | 104.32 | role_hop | 128 | 256 | false |
| role aslot edge | 105.42 | edge_path | 128 | 256 | false |

## Interpretation

`edge_path` is the best role-transition QA19 variant in this local sequence, but
it is still only weak-positive.

Compared with R182 base answer-slot:

- validation full-answer exact: `0.0700 -> 0.1100`
- validation full CE: `1.3207 -> 1.3179`
- test full-answer exact: `0.1167 -> 0.1200`
- test full CE: `1.2882 -> 1.2802`
- state: `39.39MB -> 35.39MB`
- wall time: `155.84s -> 105.42s`

Compared with R181 no-slot:

- validation exact improves `0.0700 -> 0.1100`
- test exact improves only `0.1167 -> 0.1200`
- test full CE improves `1.3317 -> 1.2802`

The direction is encouraging: local edge composition improves validation exact
and consistently lowers CE. But QA19 remains unsolved. The R181 majority
baselines were validation `0.1300` and test `0.1100`, so R184 is still below
validation majority and only slightly above test majority.

## Boundary

R184 supports a narrow mechanism claim: parser-free local edge-path features are
more useful than pure answer-slot or hop-slot readouts for QA19 CE and validation
exact match. It does not support a claim that the model performs robust path
reasoning.

## Next

The next mechanism should make the edge path state less heuristic:

1. Learn local edge compatibility/prototype slots for candidate intermediate
   tokens instead of using raw shared-neighbor overlap.
2. Add an inhibitory WTA cleanup over candidate intermediate tokens.
3. Feed slot 0/slot 1 from the selected intermediate path state.
4. Validate on QA19 with majority baselines and full-answer exact match only.
