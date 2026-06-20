# U002: Transformer-Inspired No-BP Attention Core

**Date**: 2026-06-20

## Purpose

Test whether the main route should move away from the short-context U001 design toward a Transformer-like no-BP core with:

- token output neurons;
- sinusoidal position codes;
- causal attention-like context reads;
- multiple local no-BP feed-forward blocks;
- scalable parameter estimate near 1B;
- TinyStories and temporal adapters;
- center-difference diagnostic against the local output update.

This is a small executable prototype, not a 1B run.  Directly instantiating 1B parameters in the current NumPy setting would be memory-heavy and scientifically unhelpful on 50k-character data.

## Implementation

Added `u002_attention_no_bp_experiment.py`.

Model flow:

```text
context tokens
  -> token codes + sinusoidal position codes
  -> fixed random causal attention read
  -> local feed-forward block trained toward token target codes
  -> token-neuron logits
  -> softmax probabilities
```

Learning remains no-BP:

- block maps update only from their local input, local hidden output, and current target token code;
- output token neurons use target-up / wrong-down local updates;
- center-difference is diagnostic only and is never used for training.

Default large-configuration estimate:

```text
vocab=50,000, d_model=1536, blocks=8, ff_mult=4
estimated params = 994,492,416
```

## Smoke Results

### TinyStories

TinyStories smoke uses `train_chars=12000`, `valid_chars=3000`, `max_vocab=128`, `eval_token_limit=600`, seed 0.

| Variant | Context | d_model | Blocks | Attention scale | Params | Valid post CE | Valid post acc | Center-diff cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 32 | 64 | 2 | 0.5 | 76,160 | 3.872 | 0.214 | 0.916 |
| no attention | 32 | 64 | 2 | 0.0 | 76,160 | 3.777 | 0.245 | 0.934 |
| deeper | 32 | 64 | 4 | 0.5 | 133,632 | 3.977 | 0.164 | 0.881 |
| longer context | 64 | 64 | 2 | 0.5 | 78,208 | 3.890 | 0.206 | 0.945 |
| wider | 32 | 128 | 2 | 0.5 | 233,984 | 3.817 | 0.257 | 0.908 |

### Temporal

Temporal toy uses a repeated 4-state sequence, context 8, d_model 32, blocks 2.

| Task | Params | Valid post CE | Valid post acc | Center-diff cosine |
|---|---:|---:|---:|---:|
| temporal | 11,080 | 0.253 | 1.000 | 0.065 |

## Interpretation

U002 is a **negative architectural boundary for the current attention design**.

What worked:

- The script closes the requested evaluation loop: TinyStories, temporal, parameter estimate near 1B, and center-difference diagnostic.
- The output-neuron local update aligns well with center-difference on TinyStories smoke (`0.88-0.95` cosine).
- The model solves the simple temporal sequence task.

What failed:

- TinyStories accuracy is far below U001.  The best smoke variant reaches only `0.257` acc.
- Fixed random attention hurts in this setup: attention scale 0.5 is worse than 0.0.
- Naive depth hurts: 4 blocks is worse than 2 blocks.
- Longer context without learned/gated attention hurts.

## Verdict

U002 should not replace U001 as the main model core.

The useful lesson is not "use this attention block"; it is:

1. token-neuron output and center-difference diagnostics are now in place;
2. the local output update is not the main problem;
3. the current attention/block representation is too noisy;
4. a Transformer-inspired no-BP model needs trainable/gated attention-like state, not fixed random attention reads.

## Next

U003 should keep the U002 evaluation harness but change the context mechanism:

1. replace fixed random attention with locally learned Hebbian key-value state;
2. feed the retrieved value into the hidden state, not directly into logits;
3. gate the read by uncertainty or novelty so longer context does not always inject noise;
4. compare against U001 on the same TinyStories smoke before any medium run.
