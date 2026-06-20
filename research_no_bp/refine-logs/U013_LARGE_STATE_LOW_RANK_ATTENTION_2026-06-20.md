# U013 Large-State Low-Rank-Attention U009

Date: 2026-06-20

## Goal

Increase the U012/U009 no-BP model from about 25M state parameters to near 1B
while keeping full-tokenizer local learning and avoiding the prohibitive cost of
full `d x d` attention.

## Implementation

Implemented by extending:

- `u012_u009_torch_fast_variants.py`

New setting:

- `attn_rank`: low-rank attention bottleneck.

Instead of full attention projections:

```text
d -> d
```

the large model uses:

```text
d -> attn_rank -> d
```

This keeps the local predictive rule, output update, count bias, and hidden
local update unchanged, but makes a near-1B state feasible.

## Large Config

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python u012_u009_torch_fast_variants.py \
  --task tinystories --mode samplebatch --device cuda:0 \
  --out-dir output/u013_tinystories_large_state_d2304_b12_r128_full_epoch_100k \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --d-model 2304 --blocks 12 --attn-rank 128 \
  --context-len 48 --chunk-size 64 --chunk-tokens 1000 \
  --sample-count 4 --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --skip-train-probes
```

Data:

- TinyStories only
- Train pairs: 23826
- Valid pairs: 2996
- Full tokenizer outputs: 128256

Model:

- `d_model=2304`
- `blocks=12`
- `attn_rank=128`
- State parameters: 964756992
- State bytes: 3859027968 bytes, about 3.68 GiB

## Results

| model | state params | train CE | valid CE | speed | max GPU memory |
|---|---:|---:|---:|---:|---:|
| U012 samplebatch small | 24.95M | 6.3067 | 5.7810 | 925.60 tok/s | 0.19 GiB |
| U013 large-state | 964.76M | 6.4381 | 6.1074 | 230.31 tok/s | 3.95 GiB |
| Llama-1B BP matched token batch | 1.236B | nan | 6.2625 | 241.98 tok/s | 23.07 GiB |

Interpretation:

- The large no-BP state is feasible on a single L20.
- It uses far less memory than Llama-1B BP training.
- Speed is close to the matched-token-batch Llama baseline.
- However, quality did not improve on this small 100k-char TinyStories subset.

## Sample Quality

Greedy generation still degenerates.  The dominant attractor changes from
`the`/quote loops to comma/`and` loops.

Example pattern:

```text
,,,,,,,,,,,,,, and,,,,,, and,,,,,, and
```

Sampling is still more word-like than greedy but remains incoherent and includes
tokenizer artifacts.

## Diagnosis

Increasing state size alone is not enough on this small one-epoch corpus.

Likely reasons:

- 23826 training pairs is far too little for a near-1B state.
- The large model uses the same learning rates as the small model; scale-aware
  learning-rate tuning is needed.
- Low-rank attention changes representation dynamics and may need its own
  hyperparameters.
- Repetition degeneration remains an objective/training-regime issue, not just
  a parameter-count issue.

Next likely tests:

1. larger TinyStories prefix, e.g. 1M chars complete epoch;
2. lower output/hidden learning rates for the 1B state;
3. multi-epoch full TinyStories-only run;
4. compare `d_model=1536`, more local MLP capacity, and `attn_rank=128` as a
   trainable-parameter-efficient alternative.
