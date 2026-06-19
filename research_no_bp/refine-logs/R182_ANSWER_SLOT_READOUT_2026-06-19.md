# R182: Answer-Slot Local Readout

Date: 2026-06-19

## Purpose

R181 showed that QA19 first-token accuracy is misleading and that full-answer
exact match remains near the majority-answer baseline. R182 adds a default-off,
parser-free local answer-slot readout so multi-token answers can learn separate
prototype banks for answer slot 0 and answer slot 1.

This is not a task parser and does not store raw prompt text. The wrapped no-BP
memory still learns the normal online next-token stream. At answer-token
positions only, the wrapper adds a slot-specific prototype score and performs a
local target/wrong update.

## Implementation

Changed `babi_unified_token_qa_experiment.py`.

New wrapper:

- `AnswerSlotReadoutMemory`

New CLI flags:

- `--answer-slot-readout`
- `--answer-slot-count`
- `--answer-slot-slots`
- `--answer-slot-lr`
- `--answer-slot-wrong-lr`
- `--answer-slot-score-scale`
- `--answer-slot-margin`
- `--answer-slot-no-base-update`

New config output:

- `answer_slot_cfg`
- `answer_slot_stats`

Training/evaluation changes:

- Full-answer evaluation uses `scores_with_answer_slot(context, slot)` when the
  wrapped memory supports it.
- Full-sequence answer training uses `update_answer_slot(context, target, slot)`
  only for answer-token positions; prompt-token updates remain unchanged.
- Default behavior is unchanged unless `--answer-slot-readout` is passed.

## Runs

Smoke:

- `output/babi_unified_qa19_r182_aslot_smoke`

Medium:

- `output/babi_unified_qa19_r182_microproto_aslot_medium_s2`
- `output/babi_unified_qa19_r182_role_r174_aslot_medium_s2`
- comparison: `output/babi_unified_qa19_r182_answer_slot_comparison`

All runs use `--allow-multi-token-answer`, full prompt+answer sequence online
updates, and no pretrained model backbone.

## Results

`output/babi_unified_qa19_r182_answer_slot_comparison/comparison_summary.csv`:

| variant | split | first acc | first CE | full acc | full-token acc | full CE | state MB |
|---|---|---:|---:|---:|---:|---:|---:|
| microproto no-slot | validation | 0.2300 | 1.4277 | 0.0900 | 0.2900 | 1.3476 | 4.18 |
| microproto answer-slot | validation | 0.2500 | 1.3982 | 0.0800 | 0.2950 | 1.3061 | 8.24 |
| delta | validation | +0.0200 | -0.0294 | -0.0100 | +0.0050 | -0.0415 | +4.06 |
| microproto no-slot | test | 0.2733 | 1.4136 | 0.1100 | 0.3083 | 1.3311 | 4.18 |
| microproto answer-slot | test | 0.2733 | 1.3855 | 0.1200 | 0.3067 | 1.2920 | 8.24 |
| delta | test | +0.0000 | -0.0281 | +0.0100 | -0.0017 | -0.0391 | +4.06 |
| role no-slot | validation | 0.2600 | 1.4465 | 0.0700 | 0.2650 | 1.3649 | 27.32 |
| role answer-slot | validation | 0.2700 | 1.4064 | 0.0700 | 0.2800 | 1.3207 | 39.39 |
| delta | validation | +0.0100 | -0.0401 | +0.0000 | +0.0150 | -0.0442 | +12.06 |
| role no-slot | test | 0.2800 | 1.4295 | 0.1167 | 0.3350 | 1.3317 | 27.32 |
| role answer-slot | test | 0.2767 | 1.3914 | 0.1167 | 0.3283 | 1.2882 | 39.39 |
| delta | test | -0.0033 | -0.0381 | +0.0000 | -0.0067 | -0.0435 | +12.06 |

Slot stats:

| variant | wall s | feature dim | active slots | slot0 updates | slot1 updates | wrong updates slot0/slot1 | raw text stored |
|---|---:|---:|---:|---:|---:|---|---|
| microproto answer-slot | 71.01 | 64 | 256 | 300 | 300 | 235 / 193 | false |
| role answer-slot | 155.84 | 192 | 256 | 300 | 300 | 226 / 180 | false |

## Interpretation

The answer-slot readout is CE-positive but not an exact-answer solution.

It improves full-answer CE consistently on validation and test:

- microproto test full CE: `1.3311 -> 1.2920`
- role test full CE: `1.3317 -> 1.2882`

It also improves train-post exact match, showing the slot banks are active and
learn local answer-position evidence. However, validation/test full-answer exact
match remains near the majority baseline from R181. The best medium test exact
match is `0.1200`, only `+0.0100` over the test majority answer baseline
`0.1100`.

Therefore, answer-slot separation is useful as a probability calibration and
mechanistic component, but it is not sufficient for QA19 path composition. The
model still lacks a prompt-local edge/path composition mechanism that can infer
both answer directions.

## Boundary

Do not claim QA19 is solved. R182 supports a narrower claim: local answer-slot
readout can reduce full-answer CE without storing raw text or using BP, but
exact full-answer generalization remains essentially unsolved.

## Next

The next mechanism should learn local edge-event composition, not just answer
position separation. A plausible no-BP route is:

1. Write edge-event traces from local co-occurrences of location tokens and
   direction tokens.
2. Use query source/destination tokens to seed a two-hop local path rollout.
3. Feed separate answer-slot readouts from the inferred slot-1 and slot-2 path
   states.
4. Keep the mechanism parser-free by deriving all features from token/position
   codes and local eligibility traces.
