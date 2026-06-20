# U011 TinyStories-Only Re-Evaluation

Date: 2026-06-20

## Goal

Remove GSM8k-Aug from the experiment and re-evaluate:

- U009 full-tokenizer local no-BP learner;
- random-init Llama-3.2-1B BP baseline.

Both use TinyStories only, the full Llama tokenizer, and complete one-epoch
training over the loaded corpus.

## Data

- Task: TinyStories only
- Train prefix: 100000 chars
- Valid prefix: 12000 chars
- Train docs: 58
- Valid docs: 10
- Train next-token pairs: 23826
- Valid next-token pairs: 2996
- Tokenizer outputs: 128256

## U009 No-BP Result

Output:

- `output/u011_tinystories_u009_full_vocab_full_epoch_100k/`

Metrics:

- Train online CE: 6.2762
- Train online accuracy: 0.0852
- Train chunk CE: 7.5711 -> 6.3484
- Train probe CE: 11.7868 -> 5.5684
- Valid CE: 11.7884 -> 5.8534
- Valid accuracy: 0.1375
- Valid unigram CE: 7.6688
- Speed: 32.01 tokens/s

Sample behavior:

- GSM-style `Answer`/`Reason`/number attractors disappear.
- Greedy still repeats TinyStories surface fragments such as `the`, `He and said`,
  and quotes.
- Sampling becomes more English-like than mixed-data runs but remains incoherent
  and includes occasional tokenizer artifacts.

## Llama-1B BP Result

Output:

- `output/u011_tinystories_llama1b_bp_full_epoch_100k/`

Metrics:

- Parameters: 1235814400
- Train CE: 7.6203
- Train accuracy: 0.0382
- Train chunk CE: 11.6875 -> 7.2021
- Valid CE: 12.1341 -> 6.5243
- Valid accuracy: 0.0664
- Speed: 979.15 tokens/s
- Max CUDA memory: 11.57 GiB

Sample behavior:

- GSM-style `Answer`/`Reason`/number attractors disappear.
- Greedy collapses to repeated periods.
- Sampling is TinyStories-like at the word level but remains incoherent and
  repetitive.

## Interpretation

Removing GSM fixes the mixed-format contamination but not the autoregressive
repetition problem.

This separates the issues:

1. `Answer`/`Reason`/number loops were caused by GSM/TinyStories mixing.
2. Repetition under greedy decoding remains even on TinyStories only.
3. The remaining issue is shared by random-init Llama-1B BP and U009 no-BP under
   this small-data, one-epoch, full-tokenizer setting.

On this TinyStories-only run, U009 achieves lower valid CE than the random-init
Llama-1B BP baseline:

- U009 valid CE: 5.8534
- Llama-1B BP valid CE: 6.5243

This does not prove U009 is better at scale, but it means the current repetition
failure should not be attributed solely to the no-BP learning rule.
