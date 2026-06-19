# R137 Attribute Binding QA15/QA16

**Date**: 2026-06-18  
**Task**: bAbI `en-qa15` basic deduction and `en-qa16` basic induction, full `900/100/1000` split  
**Goal**: Extend the bAbI QA evaluation beyond QA1/QA2/QA3 location/object movement into category-to-attribute and category-to-relation binding.

## Setup

R137 adds `attribute_binding_state_no_bp` to `babi_no_bp_qa_experiment.py`.

The model is a pure local associative state learner:

- entity -> category;
- category -> afraid-of answer;
- entity -> color answer;
- category -> color answer.

Updates are local delta-Hebbian writes over fixed random entity/category/answer codes. The state is rebuilt per story and stores no raw training examples. QA16 needed two local safeguards against random-code crosstalk:

- only entities that have observed color events can backfill entity -> color into category -> color;
- direct entity-color lookup is used only for entities that actually had a color event, otherwise the model uses category induction.

The front-end is still a task-specific grammar parser. This is analogous to the early QA1/QA2/QA3 role-binding baseline before learned event/query front-ends, so the claim is about the state-binding mechanism, not yet about learned natural-language parsing.

Outputs:

- `output/babi_attr_binding_qa15_seed0/`
- `output/babi_attr_binding_qa15_seed1/`
- `output/babi_attr_binding_qa15_seed2/`
- `output/babi_attr_binding_qa16_seed0/`
- `output/babi_attr_binding_qa16_seed1/`
- `output/babi_attr_binding_qa16_seed2/`
- `output/babi_attr_binding_qa15_qa16_seed_repeat/aggregate_summary.csv`
- `output/babi_attr_binding_qa15_qa16_seed_repeat/seed_repeat.csv`

## Raw Data Table

Three-seed test aggregate:

| Task | Method | Mean acc | Std acc | Mean CE | Std CE | State bytes | Raw text |
|---|---|---:|---:|---:|---:|---:|---|
| QA15 | `attribute_binding_state_no_bp` | 1.000000 | 0.000000 | 0.003610 | 0.000332 | 65,717 | false |
| QA15 | `phase_dendritic_no_bp` | 0.325667 | 0.008957 | 1.362482 | 0.003042 | 6,549 | false |
| QA15 | `raw_lexical_retrieval` | 0.213000 | 0.000000 | 1.386034 | 0.000000 | 300,471 | true |
| QA15 | `majority_no_memory` | 0.213000 | 0.000000 | 1.415665 | 0.000000 | 154 | false |
| QA16 | `attribute_binding_state_no_bp` | 0.995000 | 0.000000 | 0.041655 | 0.003839 | 65,720 | false |
| QA16 | `phase_dendritic_no_bp` | 0.459667 | 0.009463 | 1.287332 | 0.011044 | 6,549 | false |
| QA16 | `raw_lexical_retrieval` | 0.438000 | 0.000000 | 1.361147 | 0.000000 | 261,991 | true |
| QA16 | `majority_no_memory` | 0.236000 | 0.000000 | 1.391019 | 0.000000 | 154 | false |

## Findings

1. Attribute/category state binding solves QA15 and nearly solves QA16. QA15 reaches perfect test accuracy on all seeds; QA16 has 5 residual errors per 1000 examples on all seeds.

2. Generic phase readout has real but limited signal on QA16. It reaches mean `0.459667`, slightly above raw lexical retrieval `0.438000`, but far below the structured local state model.

3. QA16 exposes a useful mechanism boundary. Additive category-color memory was wrong because the task is state-like induction with recency/overwrite, not majority voting. The working version needs local state guards to avoid reading nonexistent entity colors from noisy associative matrices.

4. This is not a final learned-language result. The current attribute front-end is regex/task-specific, so the next fair step is a learned local attribute/query detector or delayed answer-credit adaptation, following the earlier R090/R091/R098 path for QA1/QA2/QA3.

## Decision

R137 is a positive expansion of the bAbI QA benchmark set. It shows the no-BP local state-binding route can cover:

- QA1/QA2/QA3: location/object movement and before-location relation;
- QA15: category relation deduction;
- QA16: category attribute induction.

Next step:

- implement learned local attribute/query detectors for QA15/QA16 using fixed random token/position features and local perceptron/prototype updates;
- then test paraphrase stress and delayed QA-level credit for these new attribute tasks.
