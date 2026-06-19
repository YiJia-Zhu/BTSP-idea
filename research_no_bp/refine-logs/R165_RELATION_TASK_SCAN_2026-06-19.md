# R165 Relation Task Scan

**Date**: 2026-06-19

## Goal

After R163/R164, the strongest current bAbI path is the parser-free
channel-final role-transition circuit. R165 tests whether that circuit transfers
beyond QA2/QA3 object-location tasks into relation-style tasks that are closer
to the human-cognition examples we want to explain later.

This scan uses the current single-token answer evaluator. QA19 is excluded for
now because its answers are two-token directions such as `south east`; it needs
a multi-token answer evaluator before it should be used as a main result.

## Tasks

| Config | Informal type | Answer space |
|---|---|---|
| `en-qa14` | temporal before/after location | 6 locations |
| `en-qa17` | positional relation yes/no | yes/no |
| `en-qa18` | size/comparison yes/no | yes/no |

The yes/no tasks are not strongly label-imbalanced:

- QA17 test: `no=480`, `yes=520`;
- QA18 test: `no=531`, `yes=469`.

## Setup

All runs use full local bAbI splits, seed 0, `max_vocab=256`, unified
next-token answer prediction, no parser, no answer head, no BP, no pretrained
backbone, and no raw replay.

Baseline:

- `state_microproto_online`
- `state_dim=64`
- `state_order=224`
- `micro_slots=64`

Role-transition:

- `state_role_transition_online`
- `role_hops=2`
- `role_window=4`
- `role_top_k=6`
- `role_channel_gates=true`
- `role_final_score_only=true`
- `role_event_cache_size=4096`

Aggregate output:

- `output/babi_unified_role_transition_r165_relation_scan/comparison_summary.csv`

## Results

| Task | Baseline test acc | Role test acc | Acc delta | Baseline CE | Role CE | CE delta | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| QA14 | 0.211 | 0.351 | +0.140 | 1.7269 | 1.7037 | -0.0232 | positive |
| QA17 | 0.482 | 0.578 | +0.096 | 0.6994 | 0.6734 | -0.0261 | positive |
| QA18 | 0.912 | 0.834 | -0.078 | 0.4674 | 0.6023 | +0.1349 | negative |

Validation mirrors the mixed picture:

- QA14 improves `0.130 -> 0.300`;
- QA17 improves `0.510 -> 0.573`;
- QA18 accuracy barely improves `0.811 -> 0.821`, but CE worsens
  `0.5105 -> 0.6025`, and test collapses relative to the strong baseline.

Cache behavior:

| Task | Role wall seconds | Cache entries | Hit rate | Cache bytes |
|---|---:|---:|---:|---:|
| QA14 | 40.93 | 1714 | 0.9990 | 438,784 |
| QA17 | 27.53 | 572 | 0.9990 | 146,432 |
| QA18 | 34.65 | 473 | 0.9997 | 121,088 |

## Findings

1. Role-transition transfers positively to QA14 temporal-before reasoning and
   QA17 positional yes/no reasoning. These are useful signals because they are
   not the same object-location target as QA2/QA3.

2. QA18 is a clear negative boundary. The microproto baseline already reaches
   `0.912` test accuracy, while role-transition drops to `0.834` and sharply
   worsens CE.

3. QA18 failure is not explained by a trivial test-label imbalance. The test
   split is only moderately skewed (`no=531`, `yes=469`). A more likely
   explanation is that QA18 has local lexical/size-comparison cues that the
   microproto readout exploits well, while the role-transition direct score
   injects distracting candidates.

4. R164 caching is useful for scanning: all three role runs have hit rates
   above `0.999`, with small transient derived-feature cache sizes.

## Interpretation

R165 supports a more precise claim than "role-transition helps relation
reasoning." It helps some relation tasks, especially temporal/positional ones,
but it is not universally safe. The next mechanism should be a local arbiter
that decides when role-transition scores should influence the full-vocabulary
readout.

This is also closer to the human-problem framing: the same extra transition
signal that helps one relational setting can interfere with a simpler learned
comparison habit, analogous to interference between learned mappings.

## Next Steps

1. Add a parser-free local role-score arbitration gate using only train-split
   confidence/margin signals, then retest QA14/17/18.
2. Extend the evaluator to multi-token answer scoring before using QA19 as a
   main spatial-navigation result.
3. Repeat QA14/17/18 on seeds 1/2 only after the arbiter is defined; otherwise
   seed repeats would mostly confirm the current mixed boundary.
