# R150 Sparse Span Binding for Unified QA

**Date**: 2026-06-19

## Goal

R149 showed that focused prompt-local binding reduced runtime versus dense
all-pair binding, but still did not improve held-out QA2.  R150 tests a more
selective no-BP binding path:

- keep the unified next-token QA evaluator;
- keep full-vocabulary answer-token prediction;
- do not add a bAbI parser, QA head, symbolic state, or raw replay;
- replace all-pair prompt binding with sparse span binding around query-matched
  prompt tokens.

## Implementation

Updated `babi_unified_token_qa_experiment.py`:

- added `--binding-mode {pair_apply,span_sparse}`;
- kept `pair_apply` as the default to preserve R148/R149 behavior;
- added `--binding-span-window`, `--binding-span-top-k`, and
  `--binding-span-decay`;
- for `span_sparse`, seeds are selected from the question tail using the
  existing `prefix_overlap` focus rule; each hop reads only local token spans
  around seed occurrences and then selects top prompt tokens for the next hop.

This is still a neural feature path into the same micro-prototype full-vocab
readout.  It stores no raw text and uses no task-specific answer classifier.

## Main Command

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --out-dir output/babi_unified_token_qa_enqa2_state_microproto_span_seed0 \
  --configs en-qa2 --max-vocab 512 \
  --method state_microproto_online \
  --state-dim 128 --state-order 128 --state-decay 0.90 \
  --micro-slots 64 --micro-lr 0.35 --micro-wrong-lr 0.02 \
  --micro-score-scale 9.0 --micro-margin 0.0 \
  --binding-hops 2 --binding-query-order 8 \
  --binding-query-mode prefix_overlap --binding-focus-k 2 \
  --binding-mode span_sparse --binding-span-window 6 \
  --binding-span-top-k 4 --binding-span-decay 0.95 \
  --phase-bias-weight 1.0 \
  --answer-only-train --train-epochs 1 --seed 0
```

## Raw Results

| Run | Binding | Train-post acc | Val acc | Test acc | Test CE | State bytes | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|
| R147 seed0 | none | 0.4100 | 0.2100 | 0.1980 | 1.8026 | 17,240,064 | 4.3s |
| R148 seed0 | dense bind2 recent trace | 0.5011 | 0.2000 | 0.1980 | 1.8015 | 50,794,496 | 201.9s |
| R149 seed0 | low-rank bind2 prefix overlap | 0.5167 | 0.1900 | 0.1990 | 1.8013 | 50,794,496 | 107.9s |
| R150 seed0 | sparse span bind2 prefix overlap | 0.6622 | 0.1700 | 0.1980 | 1.7988 | 50,794,496 | 9.8s |

Small seed0 span probes:

| Variant | Train-post acc | Val acc | Test acc | Test CE |
|---|---:|---:|---:|---:|
| window6 top4 focus2 decay0.95 | 0.6622 | 0.1700 | 0.1980 | 1.7988 |
| window6 top4 focus1 decay1.00 | 0.6100 | 0.1700 | 0.1870 | 1.8110 |
| window6 top4 focus2 decay1.00 | 0.5844 | 0.1200 | 0.1990 | 1.8084 |
| window12 top6 focus2 decay1.00 | 0.5678 | 0.1200 | 0.1860 | 1.8092 |
| window4 top2 focus2 decay0.99 | 0.6000 | 0.1100 | 0.1900 | 1.8071 |

## Key Findings

1. Sparse span binding is much faster than R149.
   The main seed0 runtime drops from `107.9s` to `9.8s`, because R150 avoids
   applying all prompt token pairs.

2. Sparse spans increase training recall but not held-out reasoning.
   Train-post accuracy rises to `0.6622`, above R149's `0.5167`, but validation
   falls to `0.1700` and test returns to `0.1980`.

3. Removing recency decay or widening spans does not fix the failure.
   The best probe reaches only `0.1990` test accuracy, still indistinguishable
   from R149 and the no-binding seed0 baseline.

## Interpretation

R150 rules out a simple explanation that QA2 failed because R148/R149 were too
dense or too slow.  Even when binding is focused around question-overlap token
spans and runtime is low, the model mostly memorizes train prompt surfaces.  The
missing mechanism is not just retrieval of nearby token spans; it is a local
state-transition/write gate that learns object ownership and movement updates
from answer-token credit.

## Next Step

Do not expand R150 to more seeds.  The next useful experiment should add a
learned local write gate over binding events, driven by answer-token eligibility
or apical error, so that the model can suppress irrelevant prompt co-occurrence
and update an object-location state without a task-specific parser.
