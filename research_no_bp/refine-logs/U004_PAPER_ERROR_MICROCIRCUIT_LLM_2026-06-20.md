# U004: Paper-Faithful Error-Microcircuit LLM Adapter

**Date**: 2026-06-20

## Purpose

Refactor U003 toward the actual error-neuron microcircuit formulation instead
of adding extra feedback modules.  The goal is a general next-token LLM model
that can use TinyStories, GSM8k-Aug, or a mixture as plain language modeling
data, while preserving the paper-level mechanism.

## Implementation

Added `u004_paper_error_microcircuit_llm_experiment.py`.

The model uses the paper variables:

- `WPP`: representation-to-representation forward weights;
- `BII`: error-neuron top-down feedback weights;
- `WIP`: representation-to-error lateral connection, identity and implicit;
- `BPI`: error-to-representation lateral connection, identity and implicit;
- `varphi_transfer`: derivative transfer from representation units to error
  neurons;
- local `post_diff outer pre_rate` updates for `WPP`.

The LLM adaptation is deliberately narrow:

```text
token ids
  -> token code + sinusoidal position
  -> residual representation stack
  -> token-neuron output layer
  -> softmax next-token loss
```

Optional standard causal attention context can be injected through
`--attention-scale`, but Q/K/V/O are not given a custom learning rule because
the paper does not define one.  This keeps attention as a structural adapter,
not a new patched mechanism.

Data interface:

- `--task tinystories`: TinyStories next-token stream;
- `--task gsm8k`: GSM8k-Aug question/reasoning/answer text as next-token data;
- `--task mix`: concatenates TinyStories text and GSM8k-Aug text into one
  next-token stream.

No answer-slot parser or math-specific evaluator is used.

## Scalability Fix

The first U004 draft stored dense identity `WIP/BPI` and dense upper-triangular
`BII_skip`, which would explode for a 50k-token vocabulary.  This was corrected:

- `WIP/BPI` identity connections are implicit;
- `BII_skip` is stored as sparse upper-triangular blocks;
- the model only initializes the active `BII` topology, not both `layered` and
  `skip`.

Large parameter estimates for `vocab=50,000`, `d_model=1536`, `blocks=16`:

| Feedback topology | Estimated params |
|---|---:|
| layered `BII` | 454,533,120 |
| skip/block-sparse `BII` | 1,854,259,200 |

## Results

All runs are small smoke diagnostics with `max_vocab=128`, `context_len=32`,
`d_model=64`, `blocks=4`, `eval_token_limit=600`, `diag_batch=4`, seed 0.

| Task | Model | Feedback | Attention | Valid post CE | Valid post acc | Output cosine | Hidden mean cosine |
|---|---|---|---:|---:|---:|---:|---:|
| TinyStories | FA | skip | 0.0 | 4.063 | 0.114 | 1.000 | -0.007 |
| TinyStories | FA | layered | 0.0 | 4.061 | 0.114 | 1.000 | -0.024 |
| TinyStories | FA | skip | 0.5 | 4.080 | 0.107 | 1.000 | -0.021 |
| TinyStories | BP diagnostic | layered | 0.0 | 4.015 | 0.114 | 1.000 | 0.415 |
| TinyStories + GSM8k-Aug mix | FA | skip | 0.0 | 4.219 | 0.085 | 1.000 | -0.042 |

## Interpretation

What worked:

- The output layer update is almost exactly aligned with center-difference.
- The paper-style `post_diff outer pre_rate` path is implemented and runs.
- `model_type=BP` diagnostic gives hidden cosine `0.415`, showing the paper-style
  update can point in a useful hidden-layer direction when `BII` is aligned.
- TinyStories/GSM8k-Aug can be mixed as ordinary next-token data.
- The LLM-scale state issue from dense identity/skip matrices was fixed.

What failed:

- Strict fixed-random `BII` feedback alignment is still not enough.
- Both paper topologies fail in FA mode: skip hidden cosine `-0.007`, layered
  hidden cosine `-0.024`.
- Attention context injection does not fix feedback alignment.
- Mixed TinyStories + GSM8k-Aug has the same failure mode.

## Verdict

U004 is stricter than U003 and removes the ad hoc feedback modules, but the
current FA version is not yet a useful deep no-BP LLM training algorithm.

The main blocker is now cleanly isolated:

```text
paper-style local WPP update works if BII is aligned;
fixed-random BII does not align by itself in this online next-token setting.
```

## Next

The next step should not add a new auxiliary module.  It should implement the
paper's missing alignment machinery more faithfully:

1. self-predicting/lateral training of the representation-error circuit;
2. local `WIP/BPI` adaptation if dimensions are not identity-compatible;
3. paper-style noise or settling dynamics only if needed to reproduce the
   self-predicting state;
4. no medium run until hidden cosine in FA mode is clearly above the current
   near-zero boundary.
