# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | sanity delayed cue | bigram/reservoir/eprop/BPTT | delay=5 hidden=48 seed=0 | target_acc | MUST | DONE | reservoir also solved, task too easy |
| R002 | M1 | hard delayed cue | bigram/reservoir/eprop/resampled/BPTT | delay=12 hidden=12 seeds=0,1,2 | target_acc | MUST | DONE | eprop=1.0, reservoir=0, resampled=0 |
| R003 | M1 | tuned BPTT baseline | BPTT lr=0.02 epochs=36 | delay=12 hidden=12 seeds=0,1,2 | target_acc | MUST | DONE | BPTT=1.0 |
| R004 | M2 | feedback ablation | eprop fixed vs resampled | delay=12 hidden=12 | target_acc | MUST | DONE | fixed feedback necessary |
| R005 | M3 | delay sweep | reservoir/eprop/BPTT | delay=4,8,12,16,20 | target_acc curve | MUST | TODO | tests fixed-position memorization |
| R006 | M3 | trace decay sweep | eprop_3factor | trace=0.5..0.98 | target_acc | MUST | TODO | tests eligibility mechanism |
| R007 | M4 | compositional cue | eprop/BPTT/reservoir | held-out cue pairs | target_acc | MUST | TODO | checks composition vs lookup |
| R008 | M4 | frozen adapter | frozen embedding + eprop adapter | char-level or grammar | next-token loss | MUST | TODO | LLM compatibility prototype |
| R009 | M5 | replay association | eprop with replay | A->B->C chain | two-step association | NICE | TODO | neuroscience motivation |

