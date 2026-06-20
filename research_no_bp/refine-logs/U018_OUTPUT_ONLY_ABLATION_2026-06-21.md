# U018 Output-Only Ablation

Date: 2026-06-21

## Goal

Test whether U015's apparent success came from the shared direct hidden
`code_error`, or whether the full-tokenizer output readout and count bias were
doing almost all of the learning.

## Command

```bash
CUDA_VISIBLE_DEVICES=0 python u012_u009_torch_fast_variants.py \
  --task tinystories --mode samplebatch --device cuda:0 \
  --out-dir output/u018_tinystories_u015_output_only_d2048_b16_r512_ctx64_full_epoch_100k \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --d-model 2048 --blocks 16 --attn-rank 512 \
  --context-len 64 --chunk-size 64 --chunk-tokens 1000 \
  --sample-count 4 --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --skip-train-probes \
  --hidden-lr 0 --hidden-bias-lr 0 --embedding-lr 0
```

## Results

| model | hidden update | train CE | valid CE | speed | max GPU memory |
|---|---|---:|---:|---:|---:|
| U015 | shared direct `code_error` | 6.4466 | 6.1326 | 222.87 tok/s | 3.79 GiB |
| U017 | layer-wise error populations | 6.4543 | 6.1534 | 194.93 tok/s | 4.27 GiB |
| U018 | output-only, no hidden/embedding update | 6.4526 | 6.1332 | 235.53 tok/s | 3.79 GiB |

## Interpretation

The user's objection is correct.  Directly broadcasting one output-derived error
to every hidden layer has no solid deep-credit-assignment theory, and this
ablation shows it was not the real source of U015's CE improvement.

U015 and U018 have essentially identical validation CE:

```text
U015 valid CE = 6.1326
U018 valid CE = 6.1332
delta = 0.0006
```

Therefore, most of the improvement in U015 came from:

1. full-vocabulary output readout learning;
2. count-bias / unigram calibration;
3. possibly minor incidental representation drift, but not a meaningful hidden
   no-BP learning effect.

This changes the interpretation:

- U015 is not evidence that shared direct hidden error works.
- U017 is the structurally correct error-population path, but it needs its own
  learning-rate/data/batching treatment.
- Future claims must compare against output-only before attributing gains to
  no-BP hidden learning.
