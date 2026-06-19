# R181: QA19 Full-Answer Boundary

Date: 2026-06-19

## Purpose

R180 added full-answer metrics for multi-token bAbI QA answers. R181 uses those
metrics to test whether the current no-BP unified token models actually solve
QA19 path finding, rather than only predicting the first direction token.

This is a medium local run, not a final full-split claim.

## Data

- Dataset: `data/babi_qa_processed/en-qa19`
- Train limit: 300
- Validation limit: 100
- Test limit: 300
- Answers: two-token directions, 12 classes in the evaluated subsets
- Evaluation: first-token accuracy plus full-answer greedy exact match

Majority/random baselines on the same evaluated rows:

| split | majority full answer | majority full acc | majority first direction | majority first acc | uniform full random | uniform first random |
|---|---|---:|---|---:|---:|---:|
| validation | east north | 0.1300 | east | 0.3100 | 0.0833 | 0.2500 |
| test | north north | 0.1100 | north | 0.2667 | 0.0833 | 0.2500 |

## Variants

1. `microproto_fullseq`: `state_microproto_online`, full prompt+answer sequence
   online updates.
2. `role_r174_fullseq`: R174-style role-transition branch with channel/final
   role gates, branch readout, joint rescue, and protect-direct suppression,
   also using full prompt+answer sequence online updates.

Both variants use `--allow-multi-token-answer`, `max_vocab=256`, `state_dim=64`,
`state_order=224`, `micro_slots=64`, seed 0, and no pretrained model backbone.
The tokenizer is local only; the learned model stores no raw text.

## Results

`output/babi_unified_qa19_r181_medium_comparison/comparison_summary.csv`:

| variant | split | first-token acc | first CE | full-answer acc | full-token acc | full CE | state MB |
|---|---|---:|---:|---:|---:|---:|---:|
| microproto_fullseq | validation | 0.2300 | 1.4277 | 0.0900 | 0.2900 | 1.3476 | 4.18 |
| role_r174_fullseq | validation | 0.2600 | 1.4465 | 0.0700 | 0.2650 | 1.3649 | 27.32 |
| delta role-micro | validation | +0.0300 | +0.0188 | -0.0200 | -0.0250 | +0.0174 | +23.14 |
| microproto_fullseq | test | 0.2733 | 1.4136 | 0.1100 | 0.3083 | 1.3311 | 4.18 |
| role_r174_fullseq | test | 0.2800 | 1.4295 | 0.1167 | 0.3350 | 1.3317 | 27.32 |
| delta role-micro | test | +0.0067 | +0.0159 | +0.0067 | +0.0267 | +0.0007 | +23.14 |

Wall time:

- microproto: 58.79s
- role R174: 138.30s

Role suppression stats:

- candidates: 5536
- updates: 2223
- skipped by direct evidence: 3313

## Interpretation

The current role-transition branch does not solve QA19 under the full-answer
metric. On test, it only improves exact full-answer accuracy by `+0.0067` over
microproto, while CE is slightly worse. On validation, it is worse on
full-answer accuracy, full-token accuracy, and CE.

The first-token metric is misleading here. Test first-token accuracy around
`0.27-0.28` is close to the majority first-direction baseline `0.2667`, while
full-answer accuracy is only `0.1100-0.1167`, essentially at the test majority
full-answer baseline `0.1100`.

This makes QA19 a useful harder benchmark: it exposes that the current local
role/branch mechanisms can weakly bias direction tokens, but do not yet perform
robust two-step path composition.

## Next

Do not report QA19 first-token accuracy as a task success metric. The next
model-side step should add an explicitly sequential answer-slot / path
composition circuit, still no-BP and parser-free: a local edge-event memory that
can write source-location, relation-direction, and destination-location traces,
then read out answer slot 1 and answer slot 2 with separate eligibility traces.
