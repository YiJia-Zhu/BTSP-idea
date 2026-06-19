# R163 QA3 Role-Transition Seed Repeat

**Date**: 2026-06-19

## Goal

R161 showed a strong seed0 gain on bAbI QA3, but it was only a pressure test.
R163 repeats QA3 on seeds 1 and 2, pairing the same-setting microproto baseline
against the R161 channel-final role-transition circuit.

The evaluation remains the unified full-vocabulary next-token QA setup:

- no BP;
- no pretrained LLM backbone;
- no task-specific QA head;
- no parser or symbolic state;
- no raw replay;
- answer is evaluated as the next token after the serialized prompt.

## Setup

Dataset: `data/babi_qa_processed/en-qa3`, full split
`900/100/1000`.

Shared settings:

- `max_vocab=256`
- `state_dim=64`
- `state_order=224`
- `state_decay=0.9`
- `micro_slots=64`
- `micro_lr=0.35`
- `micro_wrong_lr=0.02`
- `micro_score_scale=8.0`
- `answer_only_train=true`
- `require_single_token_answer=true`

Role-transition settings:

- `role_hops=2`
- `role_window=4`
- `role_top_k=6`
- `role_query_order=16`
- `role_gate_lr=0.08`
- `role_gate_wrong_lr=0.04`
- `role_score_scale=1.5`
- `role_downstream_bonus=0.75`
- `role_channel_gates=true`
- `role_final_score_only=true`

## Output Files

Aggregate:

- `output/babi_unified_role_transition_r163_qa3_seed_repeat/per_seed_summary.csv`
- `output/babi_unified_role_transition_r163_qa3_seed_repeat/aggregate_summary.csv`
- `output/babi_unified_role_transition_r163_qa3_seed_repeat/paired_deltas.csv`

New seed runs:

- `output/babi_unified_role_transition_r163_qa3_ref_base_seed1/summary.csv`
- `output/babi_unified_role_transition_r163_qa3_ref_base_seed2/summary.csv`
- `output/babi_unified_role_transition_r163_qa3_channel_final_seed1/summary.csv`
- `output/babi_unified_role_transition_r163_qa3_channel_final_seed2/summary.csv`

Seed0 is reused from R161:

- `output/babi_unified_role_transition_r161_qa3_ref_base_seed0/summary.csv`
- `output/babi_unified_role_transition_r161_qa3_channel_final_seed0/summary.csv`

## Results

### Per Seed

| Seed | Microproto test acc | Microproto CE | Role-transition test acc | Role-transition CE | Acc delta | CE delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.219 | 1.7202 | 0.344 | 1.7041 | +0.125 | -0.0161 |
| 1 | 0.221 | 1.7193 | 0.307 | 1.7165 | +0.086 | -0.0028 |
| 2 | 0.224 | 1.7242 | 0.305 | 1.7136 | +0.081 | -0.0106 |

### Aggregate

| Variant | Test acc mean | Test acc std | Test CE mean | Test CE std | Val acc mean | State bytes |
|---|---:|---:|---:|---:|---:|---:|
| microproto | 0.2213 | 0.0021 | 1.7212 | 0.0021 | 0.1367 | 4,384,768 |
| role channel+final | 0.3187 | 0.0179 | 1.7114 | 0.0053 | 0.3633 | 12,776,192 |

Paired deltas:

- test accuracy: `+0.125`, `+0.086`, `+0.081`;
- test CE: `-0.0161`, `-0.0028`, `-0.0106`;
- validation accuracy: `+0.18`, `+0.23`, `+0.27`;
- state cost: `+8,391,424` bytes.

## Findings

1. The R161 QA3 seed0 improvement reproduces on seeds 1 and 2.  All three
   seeds improve both top-1 answer accuracy and CE over the microproto
   baseline.

2. The QA3 gain is larger than the QA2 gain.  QA2 R161 improved mean test
   accuracy from `0.2337` to `0.2427` over R160, while QA3 improves the
   microproto baseline from `0.2213` to `0.3187`.

3. The method is still far from solving QA3.  Accuracy around `0.319` is a real
   transition-circuit signal, not a solved multi-hop QA model.

4. The cost tradeoff remains significant: state grows from `4.38MB` to
   `12.78MB`, and R161 seed0 runtime was about `106s` versus `20s` for the
   same-setting microproto.

## Interpretation

R163 turns the R161 QA3 pressure test into a reproducible three-seed result.
The parser-free channel-final role-transition circuit is now the strongest
available bAbI QA3 no-BP path in this repository.  The improvement is not
explained by a single lucky initialization.

The next bottleneck is not whether the role-transition path has signal; it does.
The bottleneck is turning that signal into a reusable state-update mechanism
with better candidate cleanup and lower runtime/state cost.

Recommended next runs:

1. Add runtime-aware sparse event caching for role traversal, then rerun QA3 to
   verify metrics are unchanged.
2. Add a local candidate cleanup mechanism that is not the harmful R161 direct
   top-k inhibition.
3. Test the same role-transition circuit on `en-qa14`, `en-qa17`, `en-qa18`,
   and `en-qa19` to connect the mechanism to temporal, visual-size, and spatial
   relation reasoning.
