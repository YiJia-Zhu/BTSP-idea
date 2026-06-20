# U012 U009 Torch Fast Variants

Date: 2026-06-20

## Goal

Accelerate U009 without changing the local no-BP learning rule.

Implemented modes:

- `exact`: torch.no_grad GPU backend with the same per-token update order as U009.
- `chunk`: same local update formulas, but updates are accumulated over a chunk
  and applied with GEMM.
- `samplebatch`: same local update formulas, but each batch takes at most one
  next-token pair from each document/sequence.  This avoids putting multiple
  token positions from the same sequence in one batch.

No autograd or BP is used.

Implemented in:

- `u012_u009_torch_fast_variants.py`

## Data

Same TinyStories-only configuration as U011:

- Train prefix: 100000 chars
- Valid prefix: 12000 chars
- Train next-token pairs: 23826
- Valid next-token pairs: 2996
- Full Llama tokenizer outputs: 128256

## Results

| model | backend | train CE | valid CE | speed |
|---|---:|---:|---:|---:|
| U009 NumPy | CPU exact online | 6.2762 | 5.8534 | 32.01 tok/s |
| U012 exact | GPU exact online | 6.2762 | 5.8533 | 845.26 tok/s |
| U012 chunk | GPU chunk local | 6.3663 | 5.8535 | 1205.53 tok/s |
| U012 samplebatch | GPU sample-batch local | 6.3067 | 5.7810 | 925.60 tok/s |
| Llama-1B BP | GPU bf16 BP | 7.6203 | 6.5243 | 979.15 tok/s |
| Llama-1B BP matched token batch | GPU fp32 BP | nan | 6.2625 | 241.98 tok/s |

Speedups:

- U012 exact vs U009 NumPy: about 26.4x
- U012 chunk vs U009 NumPy: about 37.7x
- U012 chunk vs Llama-1B BP: about 1.23x faster
- U012 samplebatch vs U009 NumPy: about 28.9x
- U012 samplebatch vs Llama-1B BP: about 0.95x
- U012 samplebatch vs Llama-1B matched token batch: about 3.82x

Memory:

- U012 exact max CUDA memory: 0.10 GiB
- U012 chunk max CUDA memory: 0.20 GiB
- U012 samplebatch max CUDA memory: 0.19 GiB
- Llama-1B BP max CUDA memory: 11.57 GiB
- Llama-1B matched token batch fp32 max CUDA memory: 23.07 GiB

## Interpretation

The slowdown was mostly implementation/backend, not the local learning rule.

The exact GPU version matches U009 NumPy almost exactly:

- same train CE to 4 decimals;
- same valid CE to 4 decimals;
- same greedy repetition behavior.

The chunk version changes online timing but not the local update formula.  It is
faster than Llama-1B BP on this run and uses much less GPU memory, while keeping
valid CE essentially unchanged.

The `samplebatch` version is the more conservative speed comparison after
rejecting same-sequence token chunks.  It is slightly slower than Llama-1B BP in
tokens/s on this run, but uses about 60x less peak GPU memory and reaches lower
valid CE on the same TinyStories-only subset.

After matching Llama's effective token batch to about 64 token predictions per
optimizer step (`seq_len=64`, `batch_size=1`, `grad_accum=1`), Llama speed drops
to 241.98 tok/s in fp32 and the run reports a NaN train chunk.  Its post-train
valid CE remains finite at 6.2625, but the train CE is not clean enough to use
as a primary quality comparison.  This suggests the earlier 979.15 tok/s Llama
number benefited from a larger effective token batch.

## Sample Quality

Generation quality did not improve from acceleration alone.

Both exact and chunk still show the TinyStories-only repetition attractors:

- repeated `the`;
- repeated `He and said`;
- quote loops.

`samplebatch` also repeats, mostly comma/quote loops.

So U012 solves the speed problem, not the generation-degeneration problem.

## Commands

Exact:

```bash
CUDA_VISIBLE_DEVICES=0 python u012_u009_torch_fast_variants.py \
  --task tinystories --mode exact --device cuda:0 \
  --out-dir output/u012_tinystories_u009_torch_exact_full_epoch_100k \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --d-model 64 --blocks 3 --context-len 48 \
  --chunk-tokens 1000 --sample-count 4 \
  --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --skip-train-probes
```

Chunk:

```bash
CUDA_VISIBLE_DEVICES=1 python u012_u009_torch_fast_variants.py \
  --task tinystories --mode chunk --device cuda:0 \
  --out-dir output/u012_tinystories_u009_torch_chunk_full_epoch_100k \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --d-model 64 --blocks 3 --context-len 48 \
  --chunk-size 64 --chunk-tokens 1000 --sample-count 4 \
  --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --skip-train-probes
```

Samplebatch:

```bash
CUDA_VISIBLE_DEVICES=0 python u012_u009_torch_fast_variants.py \
  --task tinystories --mode samplebatch --device cuda:0 \
  --out-dir output/u012_tinystories_u009_torch_samplebatch_full_epoch_100k \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --d-model 64 --blocks 3 --context-len 48 \
  --chunk-size 64 --chunk-tokens 1000 --sample-count 4 \
  --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --skip-train-probes
```

Llama matched token batch:

```bash
CUDA_VISIBLE_DEVICES=0 python u010_llama1b_bp_same_data_baseline.py \
  --task tinystories --model-scale llama1b --device cuda:0 \
  --out-dir output/u012_tinystories_llama1b_bp_matched_tokenbatch64_fp32_lr1e5 \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --gsm-train-items 0 --gsm-valid-items 0 \
  --seq-len 64 --batch-size 1 --grad-accum 1 \
  --lr 1e-5 --dtype fp32 \
  --chunk-tokens 1000 --sample-count 4 \
  --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --sample-temperature 1.0
```
