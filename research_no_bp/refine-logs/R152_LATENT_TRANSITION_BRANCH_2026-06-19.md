# R152 Latent Transition Branch for Unified QA

**Date**: 2026-06-19

## Goal

R151 showed that scalar answer-credit gates over sparse span events do not learn
a reusable QA2 object-location transition.  R152 tests a parser-free latent
transition branch: start from a query-seeded token state, spread activation over
nearby prompt tokens for multiple passes, and expose the resulting state vector
to the same full-vocabulary next-token readout.

The branch does not add a bAbI parser, symbolic state table, QA head, raw replay,
or BP.  It is a fixed local recurrent/spreading dynamic plus the existing
no-BP micro-prototype readout.

## Implementation

Updated `babi_unified_token_qa_experiment.py` with:

- `--latent-transition-branch`
- `--transition-window`
- `--transition-passes`
- `--transition-decay`
- `--transition-threshold`
- `--transition-strength`

For each prompt, the branch:

1. builds a query state from either recent prompt tokens or the existing
   `prefix_overlap` query focus;
2. scans the prompt prefix left-to-right;
3. activates local neighborhoods when a token matches the current state;
4. normalizes the accumulated neighborhood vector;
5. repeats for a small number of passes;
6. concatenates the final latent transition state to the recurrent/binding
   feature vector before the full-vocab micro-prototype readout.

## Commands

Latent transition with prefix-overlap query:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_unified_token_qa_experiment.py \
  --out-dir output/babi_unified_token_qa_enqa2_state_microproto_latent_transition_prefix_seed0 \
  --configs en-qa2 --max-vocab 512 \
  --method state_microproto_online \
  --state-dim 128 --state-order 128 --state-decay 0.90 \
  --micro-slots 64 --micro-lr 0.35 --micro-wrong-lr 0.02 \
  --micro-score-scale 9.0 --micro-margin 0.0 \
  --binding-query-order 8 --binding-query-mode prefix_overlap \
  --binding-focus-k 2 \
  --latent-transition-branch --transition-window 6 \
  --transition-passes 2 --transition-decay 0.98 \
  --transition-threshold 0.10 --transition-strength 0.85 \
  --phase-bias-weight 1.0 \
  --answer-only-train --train-epochs 1 --seed 0
```

## Raw Results

| Run | Train-post acc | Val acc | Test acc | Test CE | State bytes | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| R147 no binding | 0.4100 | 0.2100 | 0.1980 | 1.8026 | 17,240,064 | 4.3s |
| R150 sparse span | 0.6622 | 0.1700 | 0.1980 | 1.7988 | 50,794,496 | 9.8s |
| R151 learned span gate | 0.6556 | 0.1400 | 0.1960 | 1.7977 | 50,798,080 | 40.3s |
| R152 transition, recent trace | 0.5411 | 0.1600 | 0.1930 | 1.8032 | 34,017,280 | 14.1s |
| R152 transition, prefix overlap | 0.5467 | 0.1900 | 0.1900 | 1.8054 | 34,017,280 | 15.3s |
| R152 sparse span + transition | 0.6444 | 0.1500 | 0.1970 | 1.8024 | 67,571,712 | 23.1s |

## Key Findings

1. Fixed latent spreading does not solve QA2.
   The best R152 test accuracy is `0.1970`, below the R147/R150 seed0 level
   of `0.1980` and below R149's `0.1990`.

2. Prefix-overlap query focus helps validation slightly but hurts test.
   It raises validation from `0.1600` to `0.1900`, but test falls from `0.1930`
   to `0.1900`.

3. Combining fixed transition with sparse span adds state but not generalization.
   The combination uses `67.6MB` state and still reaches only `0.1970` test
   accuracy.

## Interpretation

R152 rules out a fixed query-seeded spreading dynamic as the missing QA2
mechanism.  The branch can create additional prompt-dependent state, but without
learned event cells or a gated state-write rule it still behaves like a richer
surface-memory feature rather than a true object-location transition model.

The useful next direction is not more fixed diffusion.  It should be a learned
latent transition circuit: candidate event cells compete over local token spans,
eligibility traces bind query object/carrier/location roles, and answer-token
apical error updates the write/read gates.  The output should remain the same
full-vocabulary next-token distribution.

## Next Step

Stop expanding fixed transition/spreading branches.  Implement a small learned
event-cell layer with local WTA competition and eligibility-based state writes,
then test whether it can beat the `0.198-0.202` QA2 held-out band without adding
a parser or answer head.
