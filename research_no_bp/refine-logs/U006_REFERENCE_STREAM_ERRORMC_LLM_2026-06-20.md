# U006: Reference Error-Microcircuit Streaming LLM Adapter

**Date**: 2026-06-20

## Purpose

U006 keeps the cloned `Error-Neuron-Microcircuits` implementation as the model
core, but changes the LLM adapter from U005's context-average input to a real
token stream.

Per token:

```text
current token + sinusoidal position -> r0 input rate
next token -> output-layer target rate
reference errormc_model.evolve_system(...)
```

No parser, answer slot, dataset-specific controller, BP training, pretrained
backbone, or n-gram memory is used.  `model_type=BP` remains diagnostic only.

## Implementation

Added `u006_reference_stream_errormc_llm_adapter.py`.

The reference microcircuit is still imported through U005's direct reference
path:

- `init_MC`;
- `errormc_model.evolve_system`;
- reference dendritic update;
- reference synapse update.

Local adapter changes:

- token/document stream instead of one mean context vector;
- voltage reset at document boundaries by default;
- multiple reference `dt` steps per token via `--presentation-steps`;
- free prediction steps before target injection via `--predict-steps`;
- target encodings: `onehot`, `centered`, `smoothed`;
- readout modes: `rate` or `voltage`;
- optional unigram output bias as a readout bridge, not model core;
- TinyStories, GSM8k-Aug, and mixed document streams through the same next-token
  interface.

## Smoke Results

TinyStories smoke uses `train_chars=12000`, `valid_chars=3000`,
`max_vocab=128`, `d_model=64`, `blocks=4`, `eval_token_limit=600`, seed 0.

| Variant | Target | Bias | Present steps | Predict steps | Valid post CE | Valid post acc |
|---|---|---|---:|---:|---:|---:|
| FA/layered | smoothed | none | 5 | 1 | 6.599 | 0.013 |
| FA/layered | smoothed | unigram | 5 | 1 | 5.815 | 0.027 |
| FA/layered | smoothed | unigram | 5 | 5 | 5.739 | 0.021 |
| FA/layered | onehot | unigram | 5 | 5 | 5.730 | 0.025 |
| BP/layered diagnostic | onehot | unigram | 5 | 5 | 5.563 | 0.029 |
| FA/layered, no reset | onehot | unigram | 5 | 5 | 5.736 | 0.021 |
| FA/layered | onehot | unigram | 20 | 5 | 5.284 | 0.044 |
| BP/layered diagnostic | onehot | unigram | 20 | 5 | 4.539 | 0.067 |
| FA/layered + WIP/BPI local update | onehot | unigram | 20 | 5 | 5.284 | 0.044 |
| FA/layered voltage readout | onehot | unigram | 20 | 5 | 5.284 | 0.044 |

Mixed TinyStories+GSM8k-Aug adapter smoke also runs:

| Task | Config | Valid post CE | Valid post acc |
|---|---|---:|---:|
| mix | vocab64 d32 blocks2 steps5 pred2 | 3.734 | 0.083 |

## Interpretation

The direct reference stream now works mechanically and can train/evaluate on
plain next-token documents.

The important result is the step-length boundary.  U005 called
`evolve_system(...)` once per target.  U006 shows that using the paper-style
multiple `dt` steps matters: FA/layered improves from CE `5.730` at 5 steps to
CE `5.284` at 20 steps.

However, the main FA route is still weak.  The BP diagnostic reaches CE `4.539`
under the same stream adapter, so aligned top-down feedback remains much
stronger than fixed random feedback.  The current reference FA path still does
not provide a strong LLM learner.

The optional unigram bias is only a readout bridge.  With no output bias, the
same FA stream setup is CE `6.599`, so the reference rates alone are badly
calibrated for a token softmax.

A scaling issue is also explicit.  The direct reference implementation stores
WIP/BPI lateral identity matrices densely.  For a vocab50k, d_model1536,
blocks16 layered reference config, the dense parameter estimate is
`5,385,427,104`, dominated by output-layer lateral matrices.  This direct
reference form is not a practical 1B-parameter LLM layout unless the paper-level
identity/lateral structure is made implicit or factorized.

## Verdict

U006 is a better reference-faithful LLM adapter than U005 because it uses the
microcircuit as a token stream and respects multi-step presentation.  It is not
yet a successful no-BP LLM learner.

Next changes should stay paper-level:

1. make WIP/BPI identity-like lateral pathways implicit or factorized for LLM
   vocab scaling;
2. add hidden-layer center-difference diagnostics to U006 after the streaming
   adapter;
3. inspect the paper/reference self-predicting and alignment setup instead of
   adding new gates or suppression modules;
4. keep BP as a diagnostic upper bound only.

## Mixed One-Epoch Run

After adding token counts and greedy sample output, ran one full pass over a
joint TinyStories+GSM8k-Aug subset:

```text
task=mix
TinyStories train/valid chars = 50000 / 10000
GSM8k-Aug train/valid items = 128 / 32
max_vocab = 128
d_model = 64
blocks = 4
FA/layered
presentation_steps = 20
predict_steps = 5
eval_token_limit = 0
```

Output directory:

```text
output/u006_reference_stream_errormc_mix_epoch1_fa_b4_steps20/
```

Metrics:

| Metric | Value |
|---|---:|
| train docs / valid docs | 141 / 42 |
| train pairs / valid pairs | 13,347 / 3,345 |
| train seconds | 187.94 |
| train CE / acc | 6.575 / 0.042 |
| valid pre CE / acc | 4.619 / 0.075 |
| valid online CE / acc | 5.043 / 0.054 |
| valid post CE / acc | 6.889 / 0.030 |
| params / state bytes | 127,232 / 957,440 |

Here `valid_pre` is the relevant "after one training epoch" validation number.
`valid_online` and `valid_post` show that continuing to update online on the
validation stream destabilizes the model.

Same compact-vocab train unigram baseline:

| Baseline | Valid CE | Valid acc |
|---|---:|---:|
| train unigram | 4.334 | 0.077 |

So the current FA reference-stream learner does not beat a simple train-token
frequency baseline on the mixed one-epoch run.

Greedy samples collapse to high-frequency punctuation patterns.  Example:

```text
Prompt:
Question:  day. She for and for her friends day with. She the at the for $2. How much

Completion:
.,,..,,..,,..,,..,,..,,..,,........,,..,,..,,.,,..,,..,,..,,..,,..,,..,,...,,..,
```

This confirms the same failure mode as the metrics: output calibration and
winner dynamics are dominated by frequent tokens rather than useful sequence
state.
