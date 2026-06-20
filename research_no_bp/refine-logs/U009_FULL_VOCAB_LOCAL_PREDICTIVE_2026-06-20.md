# U009 Full-Vocab Local Predictive Stream

Date: 2026-06-20

## Goal

Test a pure no-BP next-token learner with the full Llama tokenizer output space.
This run does not compact the vocabulary: `len(tokenizer)=128256`, and each token
id has an output neuron/logit.

## Model

- Input: raw Llama tokenizer ids.
- Context: causal attention over the recent token window with sinusoidal position
  codes.
- Blocks: residual tanh feed-forward blocks.
- Output: full-vocabulary softmax over 128256 token neurons.
- Learning:
  - output weights update from local softmax error;
  - hidden blocks update from a token-population error represented in fixed token
    code space;
  - output bias uses local target counts as each output neuron's baseline firing
    rate.
- No autograd, no BP, no BPTT, no pretrained model weights, no n-gram memory.

Implemented in:

- `u009_full_vocab_local_predictive_stream.py`

## Complete Epoch Run

Command:

```bash
python u009_full_vocab_local_predictive_stream.py \
  --task mix \
  --out-dir output/u009_mix_full_vocab_d64_countbias_full_epoch_100k_128gsm \
  --train-chars 100000 --valid-chars 12000 --doc-chars 1200 \
  --gsm-train-items 128 --gsm-valid-items 32 \
  --d-model 64 --blocks 3 --context-len 48 \
  --train-token-limit 0 --eval-token-limit 0 \
  --chunk-tokens 1000 --sample-count 4 \
  --sample-prompt-tokens 20 --sample-new-tokens 64 \
  --bias-mode count --bias-alpha 0.01 \
  --output-lr 0.030 --hidden-lr 0.004 --embedding-lr 0.001 \
  --logit-scale 2.0
```

Loaded data:

- Train docs: 186
- Valid docs: 42
- Train next-token pairs: 32670
- Valid next-token pairs: 5889

Model size:

- Parameters/state elements: 24946368
- State bytes: 99785472 bytes, about 95.2 MiB

## Results

Main metrics:

- Train online CE: 6.3764, accuracy 0.0919
- Train chunk CE: 7.5711 -> 5.7404
- Train probe CE: 11.7976 -> 5.7786
- Valid CE: 11.8024 -> 6.2538
- Valid accuracy: 0.1167
- Valid unigram CE: 7.8460
- Speed: 33.09 token updates/s

Interpretation:

- Loss reduction is real on a full-tokenizer output layer.
- Complete valid CE beats the unigram baseline on the same loaded corpus.
- The no-BP local hidden/output update is doing more than just learning global
  frequency.

## Sample Quality

Generation is still not acceptable.

Greedy decoding collapses into format/repetition attractors, for example:

- repeated `.` in shorter runs;
- after full mixed training, repeated `<<`, `=`, and `Answer:` fragments;
- TinyStories prompts are contaminated by GSM-style answer markers.

Sampling produces more word-like output than before count bias, but remains
incoherent and includes tokenizer/code fragments.

## Current Diagnosis

U009 passes the narrow loss test but fails the generation test.

The main issue is now not "no learning"; it is self-generated context stability.
The model can lower CE under teacher-forced evaluation, but once its own token is
fed back as context, high-frequency punctuation and mixed-format markers become
attractors.

The next change should stay inside the unified no-BP core.  The likely direction
is to improve context/state dynamics and mixed-data conditioning, not to add a
decode-time repetition penalty or n-gram backoff.
