# U010 Llama-1B BP Same-Data Baseline

Date: 2026-06-20

## Goal

Check whether the U009 repetition/format-attractor problem is specific to the
no-BP local learner, or whether a standard Llama-1B decoder trained from random
initialization on the same small mixed corpus also shows it.

This is a BP baseline only.  It is not part of the no-BP method.

## Setup

Implemented in:

- `u010_llama1b_bp_same_data_baseline.py`

Model:

- Llama-3.2-1B config loaded locally.
- Random initialization only.
- No pretrained weights loaded.
- Full Llama tokenizer: 128256 token outputs.
- Parameters: 1235814400.
- Optimizer: AdamW, BP.

Data matches the U009 complete epoch run:

- TinyStories prefix: 100000 chars
- GSM8k-Aug train records: 128
- Valid TinyStories prefix: 12000 chars
- GSM8k valid records: 32
- Train docs: 186
- Valid docs: 42
- Train next-token pairs: 32670
- Valid next-token pairs: 5889

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python u010_llama1b_bp_same_data_baseline.py \
  --model-scale llama1b --device cuda:0 \
  --out-dir output/u010_llama1b_bp_same_data_full_epoch_100k_128gsm \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --gsm-train-items 128 --gsm-valid-items 32 \
  --seq-len 128 --batch-size 1 --grad-accum 4 \
  --lr 3e-4 --dtype bf16 \
  --chunk-tokens 1000 --sample-count 4 \
  --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --sample-temperature 1.0
```

## Results

- Train CE: 7.3509
- Train accuracy: 0.0380
- Train chunk CE: 11.6875 -> 5.3777
- Valid CE: 12.1457 -> 6.9088
- Valid accuracy: 0.0219
- Speed: 844.10 tokens/s
- Max CUDA memory: 11.57 GiB

For comparison, U009 on the same loaded corpus:

- U009 valid CE: 11.8024 -> 6.2538
- U009 valid unigram CE: 7.8460
- U009 train chunk CE: 7.5711 -> 5.7404

## Sample Quality

The Llama-1B BP baseline also fails generation after one epoch.

Observed greedy/sample attractors:

- repeated `Answer`;
- repeated `Reason:`;
- repeated digit runs such as many `5`s;
- GSM arithmetic fragments leaking into TinyStories prompts;
- `<<+=>>`-style format fragments.

Example failure pattern:

```text
Reason:555555555555555...
```

## Interpretation

The generation failure is not unique to U009.

With this small mixed corpus and full tokenizer, a standard 1.236B-parameter
Llama trained by BP also lowers loss but still collapses during self-generated
decoding.  This points to a shared data/training-regime issue:

- full tokenizer output is very large relative to the number of training tokens;
- TinyStories and GSM formats are mixed in one stream;
- one epoch is enough to learn strong local format markers but not enough for
  stable free-running generation;
- teacher-forced CE can improve while autoregressive samples remain poor.

For the no-BP line, the next comparison should use a cleaner data protocol:
separate TinyStories-only and GSM-only complete epochs, plus a mixed run with
explicit text-format balancing, before attributing repetition to the local
learning rule itself.
