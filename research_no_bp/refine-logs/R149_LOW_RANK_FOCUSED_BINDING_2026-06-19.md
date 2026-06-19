# R149 Low-Rank Focused Binding for Unified QA

**Date**: 2026-06-19
**Status**: DONE-NEGATIVE
**Task**: bAbI `en-qa2` unified token QA

## Purpose

R148's prompt-local binding hops were too slow and too diffuse.  They improved
QA2 training recall but did not improve held-out test accuracy.  R149 tests two
fixes:

1. Replace explicit dense binding matrix construction with low-rank direct
   application of token-pair writes.
2. Replace the diffuse recent-question query trace with a prompt-local focus
   heuristic: use question-tail tokens that also appeared in the prompt prefix,
   preferring lower-frequency prefix matches.  This is intended to focus on
   tokens such as `football` rather than averaging `Where/is/the/Answer`.

The method remains parser-free and full-vocabulary: no QA head, no entity table,
no raw replay.

## Implementation

Updated `OnlineStateMicroPrototypeMemory` in
`babi_unified_token_qa_experiment.py`:

- `apply_binding(tokens, state)` computes `M @ state` from local token-pair
  writes without materializing `M`;
- `--binding-query-mode prefix_overlap`;
- `--binding-focus-k`.

The default query mode remains `recent_trace`, so previous commands still work.

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --out-dir output/babi_unified_token_qa_enqa2_state_microproto_bind2_focus_seed0 \
  --configs en-qa2 --max-vocab 512 \
  --method state_microproto_online \
  --state-dim 128 --state-order 128 --state-decay 0.90 \
  --micro-slots 64 --micro-lr 0.35 --micro-wrong-lr 0.02 \
  --micro-score-scale 9.0 --micro-margin 0.0 \
  --binding-hops 2 --binding-window 12 --binding-query-order 8 \
  --binding-query-mode prefix_overlap --binding-focus-k 2 \
  --binding-decay 0.95 --binding-bidirectional \
  --phase-bias-weight 1.0 \
  --answer-only-train --train-epochs 1 --seed 0
```

## Raw Data Table

QA2 seed0:

| Method | Train post acc | Val acc | Test acc | Test CE | State bytes | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| R147 no binding | 0.4100 | 0.2100 | 0.1980 | 1.8026 | 17,240,064 | ~3s |
| R148 dense bind2 recent trace | 0.5011 | 0.2000 | 0.1980 | 1.8015 | 50,794,496 | 202s |
| R149 low-rank bind2 prefix overlap | 0.5167 | 0.1900 | 0.1990 | 1.8013 | 50,794,496 | 108s |

## Findings

1. Low-rank application roughly halves runtime versus dense binding, but the
   method remains much slower than no-binding micro-prototypes.

2. Prefix-overlap focus increases train recall slightly (`0.5011 -> 0.5167`)
   but does not improve held-out QA2 (`0.1980 -> 0.1990`, not meaningful).

3. The failure mode persists: token-pair prompt binding adds capacity but does
   not learn the selective object-carrier-location transition.

## Decision

Do not continue tuning all-pair prompt-local binding.  The next attempt should
change the write mechanism, not the query heuristic:

- use sparse candidate spans rather than all token pairs;
- learn local write gates from answer-token eligibility;
- keep the state low-rank;
- report train_post and held-out QA2 together to avoid mistaking memorization
  for state-transition reasoning.
