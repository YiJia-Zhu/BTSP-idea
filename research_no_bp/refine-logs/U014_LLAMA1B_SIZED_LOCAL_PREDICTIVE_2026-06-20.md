# U014 Llama1B-Sized Local Predictive No-BP

Date: 2026-06-20

## Goal

Increase the U013 large-state no-BP model above the previous 964.76M state
parameters and roughly match the random-init Llama1B BP baseline parameter
scale, while keeping the same full-tokenizer local predictive learning rule.

## Change

No learning-rule change was made.

The run uses the existing U012/U013 torch implementation:

- `u012_u009_torch_fast_variants.py`

Configuration change:

- `d_model`: `2304 -> 2816`
- `blocks`: `12 -> 14`
- `attn_rank`: `128 -> 256`

The model still uses:

- full Llama tokenizer vocabulary, `128256` output ids;
- TinyStories-only data;
- sample-level batching, not same-sequence token-position chunking;
- local no-BP output/hidden/embedding updates.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 python u012_u009_torch_fast_variants.py \
  --task tinystories --mode samplebatch --device cuda:0 \
  --out-dir output/u014_tinystories_large_state_d2816_b14_r256_full_epoch_100k \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --d-model 2816 --blocks 14 --attn-rank 256 \
  --context-len 48 --chunk-size 64 --chunk-tokens 1000 \
  --sample-count 4 --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --skip-train-probes
```

Data:

- Train docs: 58
- Valid docs: 10
- Train pairs: 23826
- Valid pairs: 2996

## Results

| model | state params | train CE | valid CE | speed | max GPU memory |
|---|---:|---:|---:|---:|---:|
| U012 samplebatch small | 24.95M | 6.3067 | 5.7810 | 925.60 tok/s | 0.19 GiB |
| U013 large-state | 964.76M | 6.4381 | 6.1074 | 230.31 tok/s | 3.95 GiB |
| U014 Llama1B-sized no-BP | 1.235B | 6.4421 | 6.1203 | 190.03 tok/s | 5.13 GiB |
| Llama-1B BP matched token batch | 1.236B | nan | 6.2625 | 241.98 tok/s | 23.07 GiB |

U014 exact state:

- Parameters: `1235325952`
- State bytes: `4941303808`, about `4.60 GiB`
- Max CUDA memory: `5511159296`, about `5.13 GiB`

## Sample Quality

Greedy generation still degenerates into punctuation and `and` attractors.

Example:

```text
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.,,,,,,,,,,,,,,,,, and,,,,,,,,,,,,,,,
```

Sampling remains word-like but incoherent and still contains tokenizer artifacts.

## Interpretation

The parameter-size target is now met: U014 is effectively the same scale as the
Llama1B BP comparison. The memory result remains favorable for no-BP because
there is no autograd graph or optimizer state.

However, quality did not improve over U013 on this 100k-character one-epoch
TinyStories run. This strengthens the current diagnosis:

1. simply increasing state size is not enough;
2. 23826 token pairs is too little for a 1.2B-state online learner;
3. the larger model likely needs scale-aware learning rates;
4. repetition is still caused by the training objective/dynamics, not by
   insufficient parameter count alone.

Next useful experiment should keep U014 size fixed and change only training
regime, preferably a complete 1M-character TinyStories epoch plus a lower-LR
variant.
