# U015 Llama-Dimension-Matched Local Predictive No-BP

Date: 2026-06-20

## Goal

Make the U012/U013/U014 no-BP model closer to the random-init Llama1B baseline
training shape by matching the main Llama1B body dimensions instead of using the
larger U014 hidden width.

## Dimension Match

Local Llama1B body config in `llama_torch_model.py`:

- `hidden_size=2048`
- `intermediate_size=8192`
- `num_hidden_layers=16`
- `num_attention_heads=32`
- `num_key_value_heads=8`

U015 uses:

- `d_model=2048`
- `blocks=16`
- `context_len=64`
- `attn_rank=512`

The `attn_rank=512` choice roughly follows the Llama1B grouped-query attention
key/value width: `8 kv heads * 64 head dim = 512`.

This is still not an identical Transformer. U015 keeps the same local no-BP
learning rule and does not add a Llama-style `2048 -> 8192 -> 2048` MLP.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 python u012_u009_torch_fast_variants.py \
  --task tinystories --mode samplebatch --device cuda:0 \
  --out-dir output/u015_tinystories_llama_dim_d2048_b16_r512_ctx64_full_epoch_100k \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --d-model 2048 --blocks 16 --attn-rank 512 \
  --context-len 64 --chunk-size 64 --chunk-tokens 1000 \
  --sample-count 4 --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --skip-train-probes
```

Data:

- TinyStories only
- Train pairs: 23826
- Valid pairs: 2996
- Full tokenizer outputs: 128256

## Results

| model | dimensions | state params | train CE | valid CE | speed | max GPU memory |
|---|---|---:|---:|---:|---:|---:|
| U013 large-state | d2304 b12 r128 ctx48 | 964.76M | 6.4381 | 6.1074 | 230.31 tok/s | 3.95 GiB |
| U014 Llama1B-sized | d2816 b14 r256 ctx48 | 1.235B | 6.4421 | 6.1203 | 190.03 tok/s | 5.13 GiB |
| U015 Llama-dim matched | d2048 b16 r512 ctx64 | 922.64M | 6.4466 | 6.1326 | 222.87 tok/s | 3.79 GiB |
| Llama-1B BP matched token batch | d2048 layers16 seq64 | 1.236B | nan | 6.2625 | 241.98 tok/s | 23.07 GiB |

U015 exact state:

- Parameters: `922642944`
- State bytes: `3690571776`, about `3.44 GiB`
- Max CUDA memory: `4073908736`, about `3.79 GiB`

## Interpretation

Matching Llama's hidden width makes the speed comparison cleaner. U015 reaches
`222.87 tok/s`, only about 8% slower than the matched-token-batch random-init
Llama1B BP baseline, while using far less memory.

The remaining speed gap is likely implementation-side and output-side:

1. U015 still computes full-vocabulary logits, dense output updates, and
   `probs @ target_codes`.
2. `samplebatch` uses independent contexts rather than one contiguous training
   sequence, so it has less regular memory access than Llama teacher forcing.
3. The current local learner is Python/PyTorch tensor code, not a fused
   Transformer training kernel.

Quality still does not improve on the small 100k-character one-epoch setting.
Greedy samples remain punctuation/`and` loops, so the immediate bottleneck is
not hidden width alignment.

## Boundary

U015 is the fairer speed comparison point. U014 is the fairer parameter-count
comparison point. Neither solves generation quality by scaling alone.
