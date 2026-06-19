# R148 Prompt-Local Binding Hops for Unified QA

**Date**: 2026-06-19
**Status**: DONE-NEGATIVE
**Task**: bAbI `en-qa2` unified token QA

## Purpose

R147 showed that `state_microproto_online` substantially improves unified-token
QA1 but fails QA2.  The likely missing mechanism is state transition: QA2 needs
the model to bind an object to a carrier and then update the carrier's location.

R148 tests a parser-free prompt-local binding circuit:

- while reading the current prompt, build a temporary Hebbian token-token
  association matrix from nearby compact tokens;
- query that matrix from the recent question tokens for two hops;
- concatenate the hop states to the recurrent prompt state;
- feed the combined feature to the same full-vocabulary micro-prototype readout.

This uses only the current input prompt as fast state.  It stores no raw training
text and adds no bAbI parser or QA head.

## Implementation

Extended `OnlineStateMicroPrototypeMemory` in
`babi_unified_token_qa_experiment.py` with optional binding features:

- `--binding-hops`
- `--binding-window`
- `--binding-query-order`
- `--binding-decay`
- `--binding-bidirectional`

The default remains `--binding-hops 0`, preserving R147 behavior.

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --out-dir output/babi_unified_token_qa_enqa2_state_microproto_bind2_seed0 \
  --configs en-qa2 --max-vocab 512 \
  --method state_microproto_online \
  --state-dim 128 --state-order 128 --state-decay 0.90 \
  --micro-slots 64 --micro-lr 0.35 --micro-wrong-lr 0.02 \
  --micro-score-scale 9.0 --micro-margin 0.0 \
  --binding-hops 2 --binding-window 12 --binding-query-order 8 \
  --binding-decay 0.95 --binding-bidirectional \
  --phase-bias-weight 1.0 \
  --answer-only-train --train-epochs 1 --seed 0
```

Runtime was about `202s` for one full QA2 seed, much slower than R147 slots64
without binding (`~3s` per QA1 seed and similar order for QA2).

## Raw Data Table

QA2 seed0:

| Method | Train post acc | Val acc | Test acc | Test CE | State bytes |
|---|---:|---:|---:|---:|---:|
| R147 slots64 no binding | 0.4100 | 0.2100 | 0.1980 | 1.8026 | 17,240,064 |
| R148 bind2 bidirectional | 0.5011 | 0.2000 | 0.1980 | 1.8015 | 50,794,496 |

## Findings

1. Prompt-local binding hops increase training recall but not held-out QA2
   reasoning.  `train_post` improves from `0.4100` to `0.5011`, while test stays
   exactly `0.1980`.

2. The added binding state is expensive.  State bytes rise from `17.24MB` to
   `50.79MB`, and runtime rises to about `202s` for one full seed.

3. The current all-token-pair Hebbian matrix is too diffuse.  It creates more
   capacity but does not isolate the object-carrier-location transition needed
   for QA2.

## Decision

Do not expand this exact bind2 implementation to three seeds.  It is a useful
negative boundary: naive prompt-local token-token Hebbian hops are not enough.

Next mechanism should be more selective and lower-rank:

- learn local write gates from answer-position credit;
- bind candidate key/value spans instead of all token pairs;
- use eligibility traces from prompt tokens to answer token error;
- keep the output as full-vocabulary next-token scores, not a QA classifier.
