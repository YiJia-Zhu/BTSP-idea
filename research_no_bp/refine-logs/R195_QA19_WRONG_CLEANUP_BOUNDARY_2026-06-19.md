# R195 QA19 Wrong-Winner Cleanup Boundary

**Date**: 2026-06-19  
**Status**: DONE-NEGATIVE-BOUNDARY  
**Question**: R194 showed that R193 coupling raises slot1 target probability but often leaves a wrong high-margin winner. Can a local inhibitory wrong-winner cleanup turn the CE gain into full-answer exact gain?

## Implementation

Added a default-off local inhibitory prototype bank inside `AnswerSlotReadoutMemory`:

- New args:
  - `--answer-slot-wrong-cleanup-slots`
  - `--answer-slot-wrong-cleanup-lr`
  - `--answer-slot-wrong-cleanup-disinhibit-lr`
  - `--answer-slot-wrong-cleanup-score-scale`
  - `--answer-slot-wrong-cleanup-min-slot`
- Feature: same local slot1 coupling feature, `slot_feature * previous_answer_token_code`.
- Update rule:
  - If a slot loses during online answer-slot training, store the wrong winner in an inhibitory prototype bank.
  - If that token later appears as the correct target, apply a small disinhibition update.
  - At inference, matching wrong-cleanup prototypes add a negative score delta.
- Defaults preserve prior runs: `wrong_cleanup_score_scale=0.0`.
- Component logging now includes `wrong_cleanup_delta` and `after_cleanup` fields.

This remains no-BP/local-plasticity and stores no raw text.

## Runs

All medium runs use QA19 `300/100/300`, seed0, exact R193 coupling config plus wrong cleanup.

| Run | Cleanup scale | Val exact | Val CE | Test exact | Test CE | Test token acc | Active cleanup slots |
|---|---:|---:|---:|---:|---:|---:|---:|
| R186 soft | 0.00 | 0.1300 | 1.3200 | 0.1267 | 1.2818 | 0.3300 | 0 |
| R193 coupling | 0.00 | 0.1300 | 1.3128 | 0.1267 | 1.2714 | 0.3250 | 0 |
| R195 cleanup s0.10 | 0.10 | 0.1400 | 1.3132 | 0.1200 | 1.2724 | 0.3217 | 33 |
| R195 cleanup s0.25 | 0.25 | 0.1300 | 1.3135 | 0.1200 | 1.2741 | 0.3167 | 33 |
| R195 cleanup s0.50 | 0.50 | 0.1300 | 1.3146 | 0.1233 | 1.2771 | 0.3150 | 33 |

Cleanup update stats:

- s0.10: updates `[0,178]`, disinhibit `[0,292]`
- s0.25/s0.50: updates `[0,180]`, disinhibit `[0,295]`

## Flip Results vs R193

| Run | Split | Full exact delta | Token acc delta | Helpful/Harmful exact |
|---|---|---:|---:|---:|
| s0.10 | train_post | +0.0033 | +0.0067 | 1 / 0 |
| s0.10 | validation | +0.0100 | +0.0050 | 1 / 0 |
| s0.10 | test | -0.0067 | -0.0033 | 0 / 2 |
| s0.25 | test | -0.0067 | -0.0083 | 1 / 3 |
| s0.50 | test | -0.0033 | -0.0100 | 2 / 3 |

## Component Diagnosis for s0.10

Against R193 coupling, held-out test slot1:

| Decode phase | R193 slot1 acc | R195 slot1 acc | Delta | Target prob delta | Target-vs-best delta | Cleanup target score | High-margin wrong >=0.20 / >=0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher_forced | 0.3533 | 0.3467 | -0.0067 | -0.00080 | +0.00098 | -0.0794 | 118 / 40 |
| greedy | 0.3133 | 0.3100 | -0.0033 | -0.00018 | +0.01169 | -0.0685 | 126 / 38 |

The cleanup channel is active on all slot1 rows, with about 4.9 active cleanup candidates per row. It slightly improves target-vs-best margin, but it also suppresses the target itself on average. High-margin wrong rows do not reliably decrease, and full-answer exact drops on test.

## Interpretation

Naive wrong-winner suppression is too blunt. It learns local wrong-token contexts, but QA19 direction tokens are reused across many valid answers, so the same inhibitory prototype often fires when that token is actually correct. The disinhibition update is not enough to separate "wrong in this context" from "correct in a similar context."

R195 therefore rejects generic slot1 wrong cleanup as the next default. The useful result is the boundary: a successful cleanup needs evidence-conditioned inhibition, not a single wrong-token memory.

## Next Step

R196 should test a narrower evidence-conditioned cleanup:

- Suppress a wrong candidate only when its inhibitory prototype is active and its current positive slot/coupling support is weak.
- Protect candidates that are strongly supported by current coupling or slot readout, even if they appeared as wrong winners before.
- Track whether the intervention reduces high-margin wrong rows without suppressing target tokens.

This is still local/no-BP, but it conditions inhibition on current evidence instead of treating wrong-token memory as globally inhibitory.
