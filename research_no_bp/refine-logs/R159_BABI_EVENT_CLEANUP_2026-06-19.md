# R159 bAbI Event Cleanup Port

**Date**: 2026-06-19

## Goal

R158 found a positive parser-free `event_cleanup` mechanism on the synthetic
object-carry token QA task.  R159 ports the same mechanism into the unified
bAbI token evaluator and tests whether the synthetic gain transfers to full
bAbI `en-qa2`.

The constraints are unchanged: no BP, no pretrained model, no frozen LLM
backbone, no bAbI parser, no symbolic object state, no answer head, and no raw
replay.  Evaluation remains full-vocabulary next-token prediction of the first
answer token from a raw prompt:

```text
Context:
...
Question: ...
Answer:
```

## Local Data Inventory

The current local data tree does not contain CLUTRR.  Available data is:

| Path | Size | Use |
|---|---:|---|
| `data/TinyStories` | 6.2GB | token-stream language modeling |
| `data/babi_qa_processed` | 16MB | current unified QA benchmark |
| `data/babi_qa` | 561KB | bAbI dataset loader/cache material |
| `data/GSM8k-Aug` | 95MB | later arithmetic/chain pressure test |
| `data/babi_qa_raw` | 4KB | empty/raw placeholder |

Processed bAbI configs currently present: `en-qa1`, `en-qa2`, `en-qa3`,
`en-qa14`, `en-qa15`, `en-qa16`, `en-qa17`, `en-qa18`, `en-qa19`.

## Implementation

Updated `babi_unified_token_qa_experiment.py`:

- added `OnlineQueryEventCleanupMemory`;
- added method choice `state_event_cleanup_online`;
- added the `state_event_cleanup_cfg` output config block;
- added CLI args for event assembly and cleanup:
  - `--assembly-hops`;
  - `--assembly-event-window`;
  - `--assembly-seed-top-k`;
  - `--assembly-recency-decay`;
  - `--assembly-locality-decay`;
  - `--cleanup-slots`;
  - `--cleanup-lr`;
  - `--cleanup-wrong-lr`;
  - `--cleanup-score-scale`;
  - `--cleanup-top-k`;
  - `--cleanup-inhibit`.

Mechanism summary:

```text
compact prompt tokens
-> query-overlap seed tokens
-> local event-neighborhood states
-> hop-wise seed reselection
-> recurrent state + event states
-> full-vocab micro-prototype readout
-> event-only cleanup prototypes
-> top-k cleanup inhibition
-> local target / wrong-winner updates
```

This is a candidate competition/readout mechanism, not a separate QA classifier.

## Smoke Result

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --configs en-qa2 \
  --out-dir output/babi_unified_event_cleanup_r159_smoke \
  --method state_event_cleanup_online \
  --train-limit 60 --eval-limit 60 --max-vocab 256 \
  --answer-only-train \
  --state-dim 48 --state-order 96 --micro-slots 8 \
  --micro-score-scale 8.0 \
  --binding-query-order 16 \
  --assembly-hops 2 --assembly-event-window 3 --assembly-seed-top-k 5 \
  --cleanup-score-scale 3.0 --cleanup-top-k 4 --cleanup-inhibit 0.25 \
  --seed 0
```

Same-size references were run with ordinary `state_microproto_online` and
`span_sparse`.

| Method | Train-post acc | Val acc | Test acc | Test CE | State bytes |
|---|---:|---:|---:|---:|---:|
| microproto | 0.7500 | 0.1000 | 0.1667 | 1.9376 | 471,040 |
| span_sparse | 0.8833 | 0.0667 | 0.1167 | 1.9101 | 1,257,472 |
| event_cleanup | 1.0000 | 0.1167 | 0.2667 | 1.9058 | 2,053,440 |

Smoke was positive, but the high train-post accuracy already suggested a risk
of prompt-surface memorization.

## Full QA2 Result

Selected full run:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --configs en-qa2 \
  --out-dir output/babi_unified_event_cleanup_r159_full \
  --method state_event_cleanup_online \
  --max-vocab 256 --answer-only-train \
  --state-dim 96 --state-order 96 --micro-slots 64 \
  --micro-score-scale 9.0 \
  --binding-query-order 16 \
  --assembly-hops 2 --assembly-event-window 3 --assembly-seed-top-k 5 \
  --cleanup-score-scale 3.0 --cleanup-top-k 4 --cleanup-inhibit 0.25 \
  --seed 0
```

Same-setting microproto reference:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --configs en-qa2 \
  --out-dir output/babi_unified_event_cleanup_r159_full_ref_base \
  --method state_microproto_online \
  --max-vocab 256 --answer-only-train \
  --state-dim 96 --state-order 96 --micro-slots 64 \
  --micro-score-scale 9.0 --seed 0
```

| Method | Train-post acc | Val acc | Test acc | Test CE | State bytes |
|---|---:|---:|---:|---:|---:|
| microproto same-setting | 0.3789 | 0.1400 | 0.2010 | 1.8051 | 6,494,208 |
| event_cleanup | 0.8844 | 0.1600 | 0.1880 | 1.8580 | 31,728,256 |

Historical seed0 references from the same unified QA2 evaluator:

| Run | Method | Test acc | Test CE | State bytes |
|---|---|---:|---:|---:|
| R147 | microproto, vocab512 | 0.1980 | 1.8026 | 17,240,064 |
| R150 | sparse span, vocab512 | 0.1980 | 1.7988 | 50,794,496 |
| R153 | event-cell branch, vocab512 | 0.1890 | 1.8012 | 34.1MB |
| R159 | event_cleanup, vocab256 | 0.1880 | 1.8580 | 31,728,256 |

## Findings

1. The port is technically functional and remains within the no-BP unified
   next-token setup.

2. The smoke gain does not scale.  Full QA2 `event_cleanup` reaches train-post
   accuracy `0.8844`, but held-out test accuracy falls to `0.1880`, below the
   same-setting microproto baseline `0.2010`.

3. R158's synthetic object-carry positive result does not transfer directly to
   bAbI QA2.  The bAbI prompt distribution appears to require a more explicit
   local state-write mechanism, not only query-neighborhood event cleanup.

4. The failure is useful: the mechanism can sharpen memorized train prompt
   surfaces, but it is not yet learning the object-carrier/location transition
   rule in a way that generalizes across bAbI stories.

## Interpretation

R159 is a negative transfer result.  Event cleanup is still valuable as a
synthetic anti-distractor and winner-selection mechanism, but the full bAbI
QA2 task exposes a missing piece: local transition writes should be gated by
event role and answer-token eligibility, instead of treating event neighborhoods
as mostly unordered token clouds.

The next step should not be another cleanup-strength sweep.  A better R160
candidate is a parser-free local role/transition circuit:

- learn event-role detectors from raw token windows using target/wrong answer
  credit;
- use those detectors to write a compact object-carrier/location state;
- keep the output interface as full-vocab next-token prediction;
- use center-difference alignment only as a diagnostic of whether the local
  update direction resembles useful error descent, not as a BP substitute.
