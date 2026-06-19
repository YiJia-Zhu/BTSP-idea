# R099 Cached QA-Credit Scaling

**Date**: 2026-06-18

## Purpose

R098 found a positive delayed answer-credit signal on QA2 but the exact candidate search was too slow for longer QA3 stories. R099 adds a cached candidate scoring mode so each training row first emits current event/query traces, then candidate credit scoring reuses those traces and only replaces one local event or query candidate.

This keeps the learning rule no-BP: the training signal is still final answer log-probability; the update is still local target/wrong-winner plasticity on detector weights and slot prototypes. The cache is an engineering approximation to an eligibility trace, not a statistical memory or raw replay table.

## Implementation

Script: `babi_delayed_credit_experiment.py`

New option:

```bash
--candidate-cache-mode exact|cached
```

- `exact` is the default and preserves R098 reproducibility by replaying the current detector for every candidate.
- `cached` precomputes the current event sequence and query once per training row, then scores candidate replacements against that cached trace.

## Commands

QA2 full cached:

```bash
time PYTHONDONTWRITEBYTECODE=1 python babi_delayed_credit_experiment.py \
  --out-dir output/babi_delayed_credit_qa2_cached_full \
  --configs en-qa2 \
  --seed-mode seeded \
  --candidate-cache-mode cached \
  --credit-epochs 1 \
  --max-event-updates-per-row 1 \
  --max-credit-sentences-per-row 8 \
  --event-eval-limit 3000 --query-eval-limit 1000
```

QA3 cached event-credit:

```bash
time PYTHONDONTWRITEBYTECODE=1 python babi_delayed_credit_experiment.py \
  --out-dir output/babi_delayed_credit_qa3_cached_200 \
  --configs en-qa3 \
  --seed-mode seeded \
  --candidate-cache-mode cached \
  --train-limit 200 --eval-limit 200 \
  --credit-epochs 1 \
  --max-event-updates-per-row 1 \
  --max-credit-sentences-per-row 8 \
  --event-eval-limit 2000 --query-eval-limit 600
```

QA3 cached query-only:

```bash
time PYTHONDONTWRITEBYTECODE=1 python babi_delayed_credit_experiment.py \
  --out-dir output/babi_delayed_credit_qa3_cached_query_200 \
  --configs en-qa3 \
  --seed-mode seeded \
  --candidate-cache-mode cached \
  --train-limit 200 --eval-limit 200 \
  --credit-epochs 1 \
  --max-event-updates-per-row 0 \
  --max-credit-sentences-per-row 4 \
  --event-eval-limit 2000 --query-eval-limit 600
```

## Results

All evaluations use strong paraphrased inputs.

| Task / size | Method | Test acc | Test CE | Wall time |
|---|---|---:|---:|---:|
| QA2 full 900/1000 | majority | 0.187 | 1.811 | - |
| QA2 full 900/1000 | R097 original structural seed | 0.409 | 4.457 | - |
| QA2 full 900/1000 | cached seeded answer-credit | 0.834 | 1.346 | 2m57s |
| QA2 300 | R097 original structural seed | 0.447 | 4.357 | - |
| QA2 300 | cached seeded answer-credit | 0.807 | 1.716 | 1m09s |
| QA3 80 | R097 original structural seed | 0.562 | 2.402 | - |
| QA3 80 | cached seeded answer-credit | 0.600 | 3.187 | 1m13s |
| QA3 200 | R097 original structural seed | 0.785 | 1.100 | - |
| QA3 200 | cached query-only answer-credit | 0.645 | 2.817 | 1m58s |
| QA3 200 | cached query+event answer-credit | 0.250 | 5.413 | 2m30s |

QA2 full detector probes:

- Event accuracy improves from 0.773 to 0.891 on the 3000-sentence test probe.
- Query subject accuracy improves from 0.357 to 1.000.
- Training used 130 query updates and 235 event updates over 900 training rows.

## Interpretation

R099 separates two conclusions:

- Positive scaling result: cached candidate scoring makes QA2 full split feasible and preserves a large improvement over the R097 original-grammar seed: 0.409 -> 0.834 test accuracy without local structural labels on the strong paraphrase stream.
- Negative QA3 result: the same answer-credit update is not stable for before-location reasoning. On QA3 200, query-only credit already hurts the seed, and query+event credit collapses to 0.250. This points to bad credit assignment/prototype drift rather than just slow implementation.
- Approximation boundary: cached QA2 300 is weaker than exact R098 QA2 300 (0.807 vs 0.863). Cached traces are useful for scale, but exact prefix/suffix state caching or a learned recurrent parser state is still needed.

## Next Step

The next useful step is not more brute-force paraphrase runs. It should add gated answer-credit updates:

- update only when the answer-margin improvement is above a stronger threshold and the current seed confidence is low;
- separate query-subject repair from QA3 destination/before-location repair;
- add rollback or validation-gated prototype updates to prevent QA3 drift;
- eventually replace candidate search with a dendritic/recurrent parser that emits local eligibility traces.
