# R196 QA19 Evidence-Protected Cleanup Boundary

**Date**: 2026-06-19  
**Status**: DONE-BOUNDARY  
**Question**: R195's wrong-winner cleanup failed because it also suppressed target tokens. Can current positive slot/coupling evidence protect well-supported candidates while keeping useful inhibition?

## Implementation

Extended R195 cleanup with default-off evidence protection:

- New args:
  - `--answer-slot-wrong-cleanup-protect-mode {none,positive_delta}`
  - `--answer-slot-wrong-cleanup-protect-threshold`
- If `protect-mode=positive_delta`, cleanup suppression is skipped for a token when:

```text
max(slot_delta[token], 0) + max(coupling_delta[token], 0) >= threshold
```

This keeps the mechanism local/no-BP: inhibition is still from a learned local wrong-winner prototype, but current local positive evidence can gate it off.

Defaults preserve R195/R193 behavior: `protect-mode=none`.

## Runs

All runs use QA19 `300/100/300`, seed0, exact R193 config plus R195 cleanup scale `0.10`.

| Run | Protect threshold | Protected / checks | Val exact | Val CE | Test exact | Test CE | Test token acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| R193 coupling | off | - | 0.1300 | 1.3128 | 0.1267 | 1.2714 | 0.3250 |
| R195 cleanup s0.10 | off | - | 0.1400 | 1.3132 | 0.1200 | 1.2724 | 0.3217 |
| R196 protect t0.50 | 0.50 | 7892 / 9890 | 0.1300 | 1.3128 | 0.1267 | 1.2713 | 0.3250 |
| R196 protect t1.00 | 1.00 | 7866 / 9890 | 0.1300 | 1.3128 | 0.1267 | 1.2713 | 0.3250 |
| R196 protect t2.00 | 2.00 | 6108 / 9942 | 0.1300 | 1.3130 | 0.1267 | 1.2713 | 0.3250 |

Sequence flip vs R193:

- t0.50: no full-answer flips on train_post/validation/test.
- t1.00: no full-answer flips on train_post/validation/test.
- t2.00: no full-answer flips on train_post/validation/test.

## Component Diagnosis for t2.00

Held-out test slot1 versus R193:

| Decode phase | R193 slot1 acc | R196 slot1 acc | Target prob delta | Target-vs-best delta | Mean cleanup target score | High-margin wrong >=0.20 / >=0.50 |
|---|---:|---:|---:|---:|---:|---:|
| teacher_forced | 0.3533 | 0.3533 | +0.00015 | -0.00087 | -0.00135 | 114 / 40 |
| greedy | 0.3133 | 0.3133 | -0.000006 | -0.00775 | -0.00814 | 123 / 39 |

Compared with R195 s0.10, the target-suppression problem is mostly removed:

- R195 teacher-forced cleanup target score: `-0.0794`
- R196 t2.00 teacher-forced cleanup target score: `-0.00135`
- R195 greedy cleanup target score: `-0.0685`
- R196 t2.00 greedy cleanup target score: `-0.00814`

But the protection also removes practical top-1 impact: exact and token accuracy are identical to R193, and high-margin wrong counts do not improve.

## Interpretation

Evidence protection solves the R195 failure mode, but it protects too much of the candidate set. The current positive-delta rule acts like a safety guard: it prevents harm and preserves the R193 CE level, but does not create a selective enough intervention to flip wrong winners.

This rules out both extremes:

- Unprotected wrong cleanup: active but harmful.
- Positive-delta protected cleanup: safe but top-1 inert.

## Next Step

R197 should move from token-level protection to conflict-local arbitration:

- Apply cleanup only on final-score conflicts where the wrong winner has cleanup evidence and the target-like alternative has independent edge/coupling support.
- Use local candidate-pair features rather than per-token global protection.
- Evaluate whether it changes the same high-margin wrong rows identified in R194 without suppressing broadly reused direction tokens.

The next mechanism should decide among a small local candidate set, not globally inhibit direction words.
