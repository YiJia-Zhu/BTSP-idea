# R164 Role Event Feature Cache

**Date**: 2026-06-19

## Goal

R163 confirmed that the channel-final role-transition circuit gives a stable
bAbI QA3 signal, but runtime is now a bottleneck. R164 adds a transient cache
for fixed role-event features so later QA3/QA14/QA17/QA18/QA19 runs are cheaper
without changing the learning rule or evaluation.

This is an engineering acceleration, not a new learning mechanism.

## Implementation

Added `--role-event-cache-size` to `babi_unified_token_qa_experiment.py`.

The cache stores only fixed derived event features keyed by
`(seed_token, neighbor_token, relative_position)`. It does not store raw prompt
text, original examples, labels, answer strings, or replay buffers. The cached
feature is:

```text
normalize(token_code[seed] * token_code[neighbor] * relative_code[rel])
```

These codes are fixed random anchors, so caching them is behavior-preserving.
The cache is optional and defaults to `0`, so all previous commands preserve
the old behavior.

Added `event_cache_stats` to `config.json` with entries, hits, misses, hit
rate, and transient cache bytes.

## Runs

Smoke pair:

- no cache: `output/babi_unified_role_transition_r164_qa3_smoke_nocache`
- cache 2048: `output/babi_unified_role_transition_r164_qa3_smoke_ecache2048`

Full seed0 pair:

- no cache reference: `output/babi_unified_role_transition_r161_qa3_channel_final_seed0`
- cache 4096: `output/babi_unified_role_transition_r164_qa3_full_seed0_ecache4096`

Aggregate comparison:

- `output/babi_unified_role_transition_r164_event_cache/comparison_summary.csv`

## Results

| Run | Cache | Metric deltas | Wall seconds | Speedup | Hit rate | Cache bytes |
|---|---:|---|---:|---:|---:|---:|
| QA3 smoke | 2048 | all split acc/CE deltas exactly 0 | 8.15 -> 5.13 | 1.59x | 0.9975 | 358,656 |
| QA3 full seed0 | 4096 | all split acc/CE deltas exactly 0 | 106.01 -> 79.32 | 1.34x | 0.9997 | 407,040 |

Full seed0 cached reproduces R161 exactly:

| Split | R161 acc | Cached acc | R161 CE | Cached CE |
|---|---:|---:|---:|---:|
| train_online | 0.4156 | 0.4156 | 1.6880 | 1.6880 |
| train_post | 0.7978 | 0.7978 | 1.4689 | 1.4689 |
| validation | 0.3100 | 0.3100 | 1.6810 | 1.6810 |
| test | 0.3440 | 0.3440 | 1.7041 | 1.7041 |

## Findings

1. The cache is behavior-preserving on both smoke and full QA3 seed0. Accuracy
   and CE match exactly across all evaluated splits.

2. Runtime improves meaningfully. The full seed0 QA3 run drops from `106.01s`
   to `79.32s`, a `1.34x` speedup.

3. The repeated event feature space is small for bAbI QA3. A cache of 4096
   slots ends with only 1590 entries and a `0.9997` hit rate.

4. This does not address the algorithmic quality bottleneck. It only makes the
   current role-transition circuit cheaper to run.

## Interpretation

R164 removes part of the runtime friction identified by R161/R163 while keeping
the no-BP learning mechanism unchanged. It is now practical to run more
relation-style bAbI tasks and seed repeats with the role-transition branch.

Recommended next runs:

1. Use `--role-event-cache-size 4096` for future full bAbI role-transition
   repeats.
2. Test role-transition on `en-qa14`, `en-qa17`, `en-qa18`, and `en-qa19` to
   connect the mechanism to temporal, size/comparison, and spatial relation
   settings.
3. Add a real local candidate-cleanup mechanism after the relation-task scan;
   do not treat this cache as a model-quality improvement.
