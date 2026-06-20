# U017 Error-Population LLM Adapter

Date: 2026-06-21

## Goal

Fix the main replication gap in U012/U015: hidden layers previously shared one
global `code_error`.  The paper and official code instead use one error
population per layer and propagate layer-specific errors through backward
projection matrices.

## Source Mapping

Reference code:

- `Error-Neuron-Microcircuits/numpy_model/src/microcircuit.py`
- `Error-Neuron-Microcircuits/numpy_model/README.md`

Official variables:

- `uP`: representation neurons
- `uI`: error neurons
- `WPP`: representation-to-representation weights
- `BII` / `BPP`: error projection weights
- `dWPP = outer(error, previous_rate)`

Relevant official-code behavior:

- error neurons are layer-specific;
- output error is computed at the output population;
- hidden error is recursively projected through `BPP` / `BII`;
- representation weights update locally from error population activity and
  previous representation activity;
- for computational feasibility, the official code can set error projection
  matrices from forward weights in BP/noisy-transpose mode.

## Implementation

Added:

- `u017_error_population_llm.py`

The U017 LLM adapter keeps the U012 data, tokenizer, generation, and evaluation
pipeline, but replaces the hidden update rule.

Old U015 hidden update:

```text
one output-derived code_error -> every block
W_l += code_error x local_input_l
```

New U017 hidden update:

```text
delta_last = output_error @ output_weights
delta_l = phi'(h_l) * B_l(delta_{l+1})
W_l += delta_l^T @ local_input_l
```

Notes:

- In the row-vector torch implementation, the BP-aligned `B_l` stores the
  row-vector equivalent of the paper's column-vector transpose projection.
- `--error-mode transpose` refreshes `B_l` from the next forward matrix, matching
  the official code's computational-feasibility path.
- `--error-mode fixed` is implemented for feedback-alignment probes but was not
  the main run here.
- No autograd or PyTorch BP is used.

## Smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python u017_error_population_llm.py \
  --task tinystories --mode samplebatch --device cuda:0 \
  --out-dir output/u017_error_population_smoke \
  --train-chars 12000 --valid-chars 3000 --doc-chars 1200 \
  --d-model 128 --blocks 4 --attn-rank 32 \
  --context-len 16 --chunk-size 16 --chunk-tokens 500 \
  --sample-count 2 --sample-prompt-tokens 12 --sample-new-tokens 24 \
  --skip-train-probes --error-mode transpose
```

Result:

- Parameters: `49.69M`
- Train CE: `6.8735`
- Train chunks: `8.4523 -> 6.2338`
- Valid CE: `11.7642 -> 7.0522`
- Speed: `2896.16 tok/s`

Fixed-feedback smoke also runs:

- `--error-mode fixed`
- Train CE: `6.8740`
- Train chunks: `8.4542 -> 6.2324`
- Valid CE: `11.7642 -> 7.0491`
- Speed: `2975.65 tok/s`

## Full TinyStories 100k

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python u017_error_population_llm.py \
  --task tinystories --mode samplebatch --device cuda:0 \
  --out-dir output/u017_tinystories_errorpop_d2048_b16_r512_ctx64_full_epoch_100k \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --d-model 2048 --blocks 16 --attn-rank 512 \
  --context-len 64 --chunk-size 64 --chunk-tokens 1000 \
  --sample-count 4 --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --skip-train-probes --error-mode transpose
```

Data:

- TinyStories only
- Train docs: 58
- Valid docs: 10
- Train pairs: 23826
- Valid pairs: 2996
- Full tokenizer outputs: 128256

## Results

| model | hidden update | params | train CE | valid CE | speed | max GPU memory |
|---|---|---:|---:|---:|---:|---:|
| U015 | shared `code_error` | 922.64M | 6.4466 | 6.1326 | 222.87 tok/s | 3.79 GiB |
| U017 | layer-wise error populations | 985.56M | 6.4543 | 6.1534 | 194.93 tok/s | 4.27 GiB |

U017 exact state:

- Parameters: `985557504`
- State bytes: `3942230016`, about `3.67 GiB`
- Max CUDA memory: `4584352256`, about `4.27 GiB`

## Sample Quality

Greedy generation still collapses, now mostly to punctuation loops:

```text
......,,,,...,,...,,....,,....,,.......,,......,,,,,...........,
```

Sampling remains word-like but incoherent and still shows tokenizer artifacts.

## Interpretation

The replication gap is real and is now fixed structurally.  U015 was not the
paper's core error-neuron idea because all hidden blocks received the same
output-derived vector.  U017 implements layer-wise error projection and local
`delta_l x r_{l-1}` updates.

However, on this small 100k-character one-epoch TinyStories run, U017 is not yet
better than U015.  The likely reasons are:

1. the local Transformer-like residual/attention adapter is still not identical
   to the paper's clean `WPP` layered chain;
2. the BP-aligned `B_l` path now adds deeper credit assignment, but the old
   learning rates were tuned for the shared-error U015 rule;
3. samplebatch still suffers from document-round-robin tiny batches on this
   58-document prefix;
4. one epoch over 23826 token pairs is too little for a 986M-state learner.

Next experiments should keep the U017 core fixed and only test:

- lower hidden learning rate for layer-wise propagated errors;
- packed-sequence or true context-target sample batching;
- fixed-feedback `--error-mode fixed` as the feedback-alignment boundary;
- larger TinyStories prefix.
