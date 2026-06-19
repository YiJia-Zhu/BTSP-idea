# R160 Local Role Transition for Unified QA2

**Date**: 2026-06-19

## Goal

R159 showed that event cleanup over unordered query-neighborhood events does
not transfer from the synthetic object-carry task to full bAbI `en-qa2`.
R160 tests a more explicit parser-free transition circuit:

- start from query-overlap seed tokens in the raw prompt;
- walk local token neighborhoods for a small number of hops;
- expose the resulting transition states to the same full-vocabulary next-token
  micro-prototype readout;
- add a direct full-vocabulary role score over traversed prompt tokens;
- learn a local event gate only from answer-token target/wrong credit.

The method still uses no BP, no pretrained model, no frozen LLM backbone, no
bAbI parser, no symbolic object state, no answer head, no structure labels, and
no raw replay.

## Implementation

Updated `babi_unified_token_qa_experiment.py`:

- added `OnlineLocalRoleTransitionMemory`;
- added method choice `state_role_transition_online`;
- added `state_role_transition_cfg` to `config.json`;
- added CLI args:
  - `--role-query-order`;
  - `--role-hops`;
  - `--role-window`;
  - `--role-top-k`;
  - `--role-recency-decay`;
  - `--role-locality-decay`;
  - `--role-gate-lr`;
  - `--role-gate-wrong-lr`;
  - `--role-gate-strength`;
  - `--role-score-scale`;
  - `--role-downstream-bonus`.

Mechanism:

```text
compact prompt tokens
-> split recent query tail from prefix
-> choose query tokens that also occur in prefix as seeds
-> for each hop:
     scan local windows around seed occurrences
     build seed-token * neighbor-token * relative-position event features
     weight events by recency, locality, and learned role gate
     accumulate next-token activation scores
     choose top-k neighbor tokens as next seeds
-> concatenate recurrent state + hop states
-> full-vocabulary micro-prototype readout
-> add direct full-vocabulary role scores
-> target/wrong local updates to prototypes and role gate
```

The direct role score is still a full-vocabulary next-token score, not an
answer-class head.  It can score any compact token, although bAbI QA2 answers
are single-token locations.

## Prompt Length Check

Raw tokenizer prompt lengths for `en-qa2`:

| Split | N | Min | Max | Mean | P90 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| train | 900 | 23 | 379 | 112.39 | 175 | 208 |
| validation | 100 | 25 | 171 | 98.74 | 149 | 163 |
| test | 1000 | 23 | 608 | 111.36 | 170 | 197 |

Full R160 uses `state_order=224`, covering about 95% of prompts without
truncation while keeping runtime manageable.

## Smoke

Settings: QA2, `60/60/60`, `max_vocab=256`, `state_dim=48`,
`state_order=128`, `micro_slots=8`, `role_hops=2`, `role_window=4`,
`role_top_k=6`, seed 0.

| Method | Train-post acc | Val acc | Test acc | Test CE | State bytes |
|---|---:|---:|---:|---:|---:|
| microproto, same order | 0.6667 | 0.0667 | 0.1667 | 1.9173 | 477,184 |
| role-transition | 0.8333 | 0.1167 | 0.2667 | 1.8014 | 1,265,536 |
| role-transition, no gate | 0.8333 | 0.1000 | 0.2500 | 1.8068 | 1,265,536 |
| role-transition, no direct role score | 0.8167 | 0.0667 | 0.1500 | 1.8724 | 1,265,536 |

Smoke interpretation: the direct prompt-local role score is necessary.  The
learned answer-credit gate gives a smaller positive increment.

## Full Three-Seed Result

Settings: QA2 full `900/100/1000`, `max_vocab=256`, answer-only train,
`state_dim=64`, `state_order=224`, `micro_slots=64`, `micro_score_scale=8.0`,
`role_query_order=16`, `role_hops=2`, `role_window=4`, `role_top_k=6`,
`role_score_scale=1.5`, `role_gate_strength=1.0`, `role_gate_lr=0.08`,
`role_gate_wrong_lr=0.04`, `role_downstream_bonus=0.75`.

| Seed | Method | Train-post acc | Val acc | Test acc | Test CE | State bytes |
|---:|---|---:|---:|---:|---:|---:|
| 0 | microproto | 0.3644 | 0.1700 | 0.1960 | 1.8048 | 4,384,768 |
| 0 | role-transition | 0.6533 | 0.1600 | 0.2380 | 1.7612 | 12,775,936 |
| 1 | microproto | 0.3800 | 0.1600 | 0.1990 | 1.8060 | 4,384,768 |
| 1 | role-transition | 0.6433 | 0.1500 | 0.2290 | 1.7574 | 12,775,936 |
| 2 | microproto | 0.3844 | 0.1400 | 0.2060 | 1.8028 | 4,384,768 |
| 2 | role-transition | 0.6622 | 0.1500 | 0.2340 | 1.7572 | 12,775,936 |

Aggregate:

| Method | Test acc mean | Test acc std | Test CE mean | Test CE std |
|---|---:|---:|---:|---:|
| microproto | 0.2003 | 0.0042 | 1.8045 | 0.0013 |
| role-transition | 0.2337 | 0.0037 | 1.7586 | 0.0019 |

Paired mean deltas: `+0.0333` test accuracy and `-0.0459` test CE.

## Full Seed0 Ablation

| Variant | Test acc | Test CE | Interpretation |
|---|---:|---:|---|
| microproto | 0.1960 | 1.8048 | same-setting baseline |
| role-transition | 0.2380 | 1.7612 | full R160 |
| no gate | 0.2310 | 1.7642 | most gain comes from prompt-local traversal/direct scoring |
| no direct role score | 0.1890 | 1.8145 | transition states alone are not enough |

## Findings

1. R160 is the first clear positive result on the unified bAbI QA2 next-token
   line after R147.  It improves three-seed test accuracy from `0.2003` to
   `0.2337` and CE from `1.8045` to `1.7586`.

2. The gain is reproducible across seeds 0/1/2, with paired accuracy deltas
   `+0.042`, `+0.030`, and `+0.028`.

3. Direct prompt-local role scoring is the main contributor.  Removing it drops
   seed0 test accuracy to `0.1890`, worse than microproto.

4. The learned answer-credit gate is a small positive increment, not the whole
   result.  Seed0 `no_gate` reaches `0.2310`, while the gated version reaches
   `0.2380`.

5. The method is still far from solving QA2.  It improves object-location
   transition evidence, but it remains a shallow prompt-local traversal rather
   than a stable reusable object/carrier/location state.

## Interpretation

R160 supports the R159 diagnosis: bAbI QA2 needs an explicit local transition
path.  Unordered event cleanup overfits prompt surfaces, but query-seeded local
role traversal exposes useful answer evidence and generalizes modestly.

The next useful step is not another scalar gate sweep.  R161 should make the
transition state more state-like:

- separate seed-to-carrier and carrier-to-location channels without hard-coded
  parser labels;
- add local inhibitory competition over traversed candidate tokens;
- add a center-difference diagnostic on the role gate update direction;
- test QA2 plus QA3 before claiming broader multi-hop reasoning.
