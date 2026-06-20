# U003: Error-Neuron Microcircuit Core

**Date**: 2026-06-20

## Purpose

Try the biologically motivated error-neuron route inspired by Max et al.'s
error-neuron cortical microcircuits, while keeping the unified next-token
interface from U002.

The key question is not just whether weights change.  It is whether the final
token error reaches early layers in a direction that resembles the true final
CE-loss descent direction.

## Implementation

Added `u003_error_microcircuit_no_bp_experiment.py`.

Model flow:

```text
context tokens
  -> token codes + sinusoidal position codes
  -> optional fixed causal attention read
  -> representation FF blocks
  -> token-neuron logits / softmax
  -> output error neurons
  -> hidden error neurons
  -> local block update: hidden_error outer local_input
```

Learning remains no-BP:

- no autograd, BP, or BPTT;
- output token neurons use the local softmax error;
- hidden blocks receive separate error-neuron signals;
- each block updates from only its local input and local error-neuron activity;
- center-difference is diagnostic only.

Deep-support design:

- `feedback_topology=direct`: every layer receives a direct output-error projection;
- `feedback_topology=layered`: error moves one layer at a time;
- `feedback_topology=hybrid`: direct plus layered error signals;
- residual representation updates are used to avoid forward depth collapse.

Large-config estimate:

```text
vocab=50,000, d_model=1536, blocks=16
estimated params = 1,606,533,120
```

`feedback_init=output_transpose` is included only as a diagnostic upper bound.
It is not a final acceptable no-BP mechanism because it injects weight-transport
information into the feedback path.

## Results

TinyStories smoke uses `train_chars=12000`, `valid_chars=3000`,
`max_vocab=128`, `context_len=32`, `d_model=64`, `eval_token_limit=600`,
`diag_batch=4`, seed 0.

| Variant | Blocks | Feedback init | Feedback learning | Valid post CE | Valid post acc | Output cosine | Hidden mean cosine |
|---|---:|---|---:|---:|---:|---:|---:|
| random hybrid | 4 | random | 0 | 3.796 | 0.132 | 0.983 | 0.065 |
| random hybrid + feedback lr | 4 | random | 0.001 | 3.796 | 0.132 | 0.983 | 0.065 |
| transpose upper bound | 4 | output transpose | 0 | 3.659 | 0.155 | 0.978 | 0.783 |
| random hybrid deep | 8 | random | 0 | 3.956 | 0.086 | 0.991 | -0.034 |
| transpose upper bound deep | 8 | output transpose | 0 | 3.868 | 0.098 | 0.988 | 0.667 |

Temporal toy sanity, repeated 4-state sequence, `context_len=8`, `d_model=32`,
`blocks=4`:

| Task | Valid post CE | Valid post acc | Output cosine | Hidden mean cosine |
|---|---:|---:|---:|---:|
| temporal | 0.151 | 1.000 | 0.755 | 0.012 |

## Interpretation

U003 separates two issues that were previously mixed together.

What worked:

- The output update is again well aligned with center-difference.
- The error-neuron pathway can support hidden-layer gradient-like updates if
  feedback is already aligned: 4-layer transpose upper bound gives hidden cosine
  `0.783`, and 8-layer transpose upper bound remains positive at `0.667`.
- This means the forward residual block stack is not the main blocker.

What failed:

- Random error-neuron feedback does not self-align in this small online setting.
- The 4-layer random model has hidden cosine `0.065`, essentially the same
  failure mode as U002.
- The 8-layer random model is worse: hidden cosine `-0.034`, valid acc `0.086`.
- A naive local feedback Hebbian update with `feedback_lr=0.001` did not improve
  the result.

## Verdict

U003 is a **positive structural probe but negative current algorithm**.

The architecture can carry error signals through many layers when feedback is
aligned, but the current random-feedback microcircuit does not learn that
alignment.  Therefore, "use error neurons" is not enough.  The next model must
implement a real local mechanism for aligning the error pathway.

## Next

U004 should keep the error-neuron representation/error-population split but add
one of the following:

1. Max-style self-predicting lateral circuit: learn `WIP`, `BPI`, and `BII`
   so error neurons predict and cancel representation activity, leaving a
   usable mismatch signal.
2. Direct feedback alignment with a measurable alignment objective, then test
   whether hidden cosine rises from near zero without using transpose feedback.
3. Low-dimensional learned feedback subspace to avoid huge `d_model x vocab`
   feedback per layer at 1B scale.

No medium TinyStories run is justified until hidden cosine improves under a
non-transport feedback rule.
