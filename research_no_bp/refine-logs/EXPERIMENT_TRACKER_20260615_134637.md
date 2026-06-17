# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | sanity delayed cue | bigram/reservoir/eprop/BPTT | delay=5 hidden=48 seed=0 | target_acc | MUST | DONE | reservoir also solved, task too easy |
| R002 | M1-old | hard delayed cue | bigram/reservoir/eprop/resampled/BPTT | delay=12 hidden=12 seeds=0,1,2 | target_acc | MUST | DONE | eprop=1.0, reservoir=0, resampled=0 |
| R003 | M1-old | tuned BPTT baseline | BPTT lr=0.02 epochs=36 | delay=12 hidden=12 seeds=0,1,2 | target_acc | MUST | DONE | BPTT=1.0 |
| R004 | M2-old | feedback ablation | eprop fixed vs resampled | delay=12 hidden=12 | target_acc | MUST | DONE | fixed feedback necessary |
| R005 | M4 | delay sweep | reservoir/eprop/BPTT | delay=4,8,12,16,20 | target_acc curve | MUST | TODO | tests fixed-position memorization |
| R006 | M4 | trace decay sweep | eprop_3factor | trace=0.5..0.98 | target_acc | MUST | TODO | tests eligibility mechanism |
| R007 | M4 | compositional cue | eprop/BPTT/reservoir | held-out cue pairs | target_acc | MUST | TODO | checks composition vs lookup |
| R008 | M0 | TinyStories recurrent three-factor | recurrent_3factor vs STDP/BTSP/dendritic | TinyStories tokenizer smoke/medium | CE, acc | MUST | DONE | negative on real text; medium CE=5.055 acc=0.068 |
| R009 | M0 | TinyStories sparse Hebbian context | sparse context memory vs no-BP baselines and low-budget tiny Llama | TinyStories tokenizer smoke/medium | CE, acc, tok/s | MUST | DONE | positive signal; medium CE=4.113 acc=0.361 |
| R010 | M0 | normalized sparse memory | additive vs normalized backoff | TinyStories tokenizer medium | CE, acc, sample quality | MUST | DONE | normalized CE=4.151, acc=0.357; not better than additive |
| R011 | M0 | linear hybrid backoff | sparse memory + dendritic/Llama logits | TinyStories tokenizer medium | CE, acc | MUST | DONE | negative; hybrid Llama CE=4.537 vs sparse CE=4.113 |
| R012 | M0 | sparse memory reproducibility | sparse_hebbian_context repeats | TinyStories tokenizer medium seeds/slices=3 | CE mean/std, acc mean/std | MUST | TODO | first next run; confirm current best is stable |
| R013 | M1 | confidence features | memory row count, entropy, max prob, highest order exists | TinyStories valid | confidence bucket CE/acc | MUST | TODO | diagnostic before gated fusion |
| R014 | M1 | gated dendritic adapter | sparse memory with dendritic backoff only on low-confidence contexts | TinyStories tokenizer medium | CE, acc, gate rate, low-conf CE | MUST | TODO | success if CE < 4.1126 or acc > 0.361 |
| R015 | M1 | gated Llama adapter | sparse memory with low-budget Llama backoff only on low-confidence contexts | TinyStories tokenizer medium | CE, acc, gate rate, eval tok/s | MUST | TODO | compare to failed linear hybrid |
| R016 | M2 | context order ablation | max_order=1..6 | TinyStories tokenizer medium | CE, acc, active contexts | MUST | TODO | identify useful order and memory growth |
| R017 | M2 | memory pruning/decay | sparse memory prune threshold + decay | TinyStories tokenizer medium | CE, acc, bytes/token | MUST | TODO | required for online deployment |
| R018 | M3 | continual online stream | sparse/gated adapter on segmented TinyStories | style/story stream | prequential CE, adaptation speed, forgetting | MUST | TODO | no raw text storage after update |
| R019 | M5 | API-compatible prototype | frozen/API base + no-BP memory adapter | small personalization/FAQ stream | task accuracy, storage, cost | NICE | TODO | only after R012-R018 pass |
| R020 | M5 | replay association | eprop with replay | A->B->C chain | two-step association | NICE | TODO | neuroscience motivation |
