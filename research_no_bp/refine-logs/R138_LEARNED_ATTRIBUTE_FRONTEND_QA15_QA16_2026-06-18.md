# R138 Learned Attribute Front-End QA15/QA16

**Date**: 2026-06-18  
**Task**: bAbI `en-qa15` and `en-qa16`, full `900/100/1000` split  
**Goal**: Remove the task-specific regex attribute/query parser from R137 and replace it with local no-BP learned statement/query detectors.

## Setup

R138 adds two learned front-ends to `babi_no_bp_qa_experiment.py`:

- `LearnedAttributeStatementDetector`: local perceptron event classifier over fixed random token/position features, plus local entity/value slot prototypes.
- `LearnedAttributeQueryDetector`: local query-type classifier plus subject prototype readout.

Both are trained from sentence/question-local structure labels only. They do not read final QA answer labels, use BP, use pretrained encoders, use API calls, or store raw examples. The downstream `AttributeBindingStateQALearner` then reads context through:

- `--attr-statement-mode learned`
- `--attr-query-mode learned`

Outputs:

- `output/babi_attr_learned_qa15_seed0/`
- `output/babi_attr_learned_qa15_seed1/`
- `output/babi_attr_learned_qa15_seed2/`
- `output/babi_attr_learned_qa16_seed0/`
- `output/babi_attr_learned_qa16_seed1/`
- `output/babi_attr_learned_qa16_seed2/`
- `output/babi_attr_learned_qa15_qa16_seed_repeat/aggregate_summary.csv`
- `output/babi_attr_learned_qa15_qa16_seed_repeat/detector_metric_summary.csv`

## Raw Data Table

Three-seed test aggregate:

| Task | Method | Mean acc | Std acc | Mean CE | Std CE | State bytes | Raw text |
|---|---|---:|---:|---:|---:|---:|---|
| QA15 | learned attribute binding | 1.000000 | 0.000000 | 0.003610 | 0.000332 | 72,773 | false |
| QA15 | phase dendritic | 0.325667 | 0.008957 | 1.362482 | 0.003042 | 6,549 | false |
| QA15 | raw lexical retrieval | 0.213000 | 0.000000 | 1.386034 | 0.000000 | 300,471 | true |
| QA16 | learned attribute binding | 0.995000 | 0.000000 | 0.041655 | 0.003839 | 73,365 | false |
| QA16 | phase dendritic | 0.459667 | 0.009463 | 1.287332 | 0.011044 | 6,549 | false |
| QA16 | raw lexical retrieval | 0.438000 | 0.000000 | 1.361147 | 0.000000 | 261,991 | true |

Detector test aggregate:

| Task | Statement event | Statement entity | Statement value | Query type | Query subject |
|---|---:|---:|---:|---:|---:|
| QA15 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| QA16 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Findings

1. Learned local front-ends preserve R137's QA performance. QA15 remains perfect and QA16 remains `0.995` across all three seeds.

2. The learned detectors are not the bottleneck on canonical bAbI QA15/QA16. Statement event/entity/value and query type/subject metrics are all `1.000` on test for both tasks.

3. State cost is small. Learned front-ends raise total state from R137's ~65.7KB to ~72.8KB on QA15 and ~73.4KB on QA16. The detector states themselves are ~4.8-5.1KB for statements and ~2.2-2.5KB for queries.

4. This result removes the regex-parser boundary for canonical surface forms, but not the paraphrase boundary. Like R091 before R097, the next meaningful test is strong surface-form stress and delayed answer-credit adaptation.

## Decision

R138 is a positive result. The bAbI QA15/QA16 path now has:

- pure local learned front-end parsing;
- pure local associative state binding;
- no raw text replay;
- no BP, pretrained model, or API backbone.

Next step:

- create QA15/QA16 paraphrase stress analogous to R097;
- then test delayed final-answer credit for adapting the attribute/query front-end without local structure labels on the new surface forms.
