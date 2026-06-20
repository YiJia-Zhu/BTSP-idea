# U019 Packed Pretraining Batching

Date: 2026-06-21

## Goal

Match standard GPT/DeepSeek-style causal-LM pretraining data construction for
both the no-BP learner and the Llama BP baseline.

The previous no-BP `samplebatch` mode used document-round-robin batching, where
each document contributed at most one next-token pair per batch.  On the 100k
TinyStories prefix, this collapsed to tiny batches because there were only 58
documents and one document had 8735 tokens.

## Implementation

Added:

- `packed_lm_data.py`

Changed:

- `u012_u009_torch_fast_variants.py`
- `u017_error_population_llm.py`
- `u010_llama1b_bp_same_data_baseline.py`

New data pipeline:

```text
tokenized docs + EOS boundaries
-> one continuous packed token stream
-> fixed-length sequences of seq_len/context_len + 1 tokens
-> every position predicts the next token
```

No cross-document attention mask is used inside a packed sequence, matching the
DeepSeek-V3 report's packed pretraining setup.

For U/no-BP:

- `--mode packed`
- `context_len` is the packed sequence length.
- `chunk_size` is the sequence batch size.
- With `context_len=64, chunk_size=1`, each update has up to 64 next-token
  targets, matching Llama `seq_len=64, batch_size=1`.

For Llama:

- `--sequence-mode packed`
- `--seq-len 64`
- `--batch-size 1`

## Packed Dataset Stats

TinyStories 100k chars:

- Raw docs: 58
- Raw next-token pairs: 23826
- Packed train sequences: 375
- Packed train targets: 23941
- Packed valid sequences: 48
- Packed valid targets: 3015
- EOS id: `128009`

The target count is slightly larger than raw pairs because EOS boundary tokens
are included as training targets.

## Smoke

U017 packed smoke:

- `d_model=128`, `blocks=4`, `context_len=16`, `chunk_size=2`
- Train CE: `6.9840`
- Valid CE: `11.7617 -> 7.0571`
- Speed: `5395.65 tok/s`

Llama debug packed smoke:

- `seq_len=16`, `batch_size=2`
- Train CE: `9.1165`
- Valid CE: `11.7867 -> 7.2497`
- Speed: `939.47 tok/s`

## Full 100k Results

All full runs below use TinyStories only, full Llama tokenizer, `seq_len/context_len=64`,
and one packed sequence per training step.

| model | learning path | params | train CE | valid CE | speed | max GPU memory |
|---|---|---:|---:|---:|---:|---:|
| U018 packed | output-only | 922.64M | 6.5021 | 6.2218 | 3513.35 tok/s | 3.81 GiB |
| U015 packed | direct shared hidden `code_error` | 922.64M | 6.4965 | 6.2271 | 3175.36 tok/s | 3.81 GiB |
| U017 packed | layer-wise error populations | 985.56M | 6.5019 | 6.2347 | 2789.93 tok/s | 4.28 GiB |
| Llama1B BP packed | AdamW BP, fp32 lr=1e-5 | 1.236B | 8.3432 | 6.2916 | 261.12 tok/s | 23.07 GiB |

## Interpretation

The batching bug is fixed.  With standard packed pretraining data, no-BP updates
are now much faster than the Llama BP baseline at the same token target count:

```text
U017 packed: 2789.93 tok/s
Llama1B BP packed: 261.12 tok/s
```

However, representation learning is still not proven.  The output-only no-BP
ablation is best on this small run:

```text
U018 output-only valid CE = 6.2218
U015 direct hidden valid CE = 6.2271
U017 error-pop valid CE = 6.2347
```

Therefore:

1. standard packed batching resolves the speed comparison problem;
2. low memory and high throughput are real in the current implementation;
3. hidden no-BP learning is still not adding value over output readout learning;
4. claims must continue to use output-only as the minimum ablation.

## Next

The next fair experiments should keep packed batching fixed and test only hidden
learning improvements:

- U017 lower `hidden_lr`;
- fixed-feedback vs transpose-feedback under packed data;
- larger TinyStories prefix;
- packed-sequence exactness with `chunk_size > 1` compared to Llama batch-size
  scaling.
