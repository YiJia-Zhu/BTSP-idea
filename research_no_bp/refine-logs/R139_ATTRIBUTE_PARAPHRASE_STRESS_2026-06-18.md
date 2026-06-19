# R139 Attribute Paraphrase Stress QA15/QA16

**Date**: 2026-06-18  
**Task**: bAbI `en-qa15` and `en-qa16`, full `900/100/1000` split  
**Goal**: Stress the R138 learned attribute/query front-end under surface paraphrase, using the currently available local data instead of CLUTRR.

## Setup

Current local data does not include CLUTRR. R139 therefore extends the existing bAbI QA path with QA15/QA16 paraphrase stress:

- statement rewrites:
  - `Mice are afraid of wolves.` -> `Mice fear wolves .`
  - `Gertrude is a mouse.` -> `Gertrude is classified as a mouse .`
  - `Bernhard is green.` -> `Bernhard is colored green .`
- question rewrites:
  - `What is gertrude afraid of?` -> `What is gertrude scared of ?`
  - `What color is Greg?` -> `Which color is Greg ?`

Compared systems:

- `learned_attribute_binding`: claimable pure no-BP path. Fixed random token/position features, local perceptron target-vs-wrong updates for statement/query type, local slot prototypes, then no-BP attribute/category binding state.
- `aware_attribute_binding`: diagnostic normalized front-end only. It maps paraphrase strings back to canonical strings before feature extraction and is not a main-method claim.

Runs:

```bash
python babi_attribute_paraphrase_stress_experiment.py \
  --out-dir output/babi_attribute_paraphrase_stress_seed0 \
  --configs en-qa15 en-qa16 \
  --train-strengths none strong \
  --eval-strength strong \
  --methods learned aware \
  --seed 0
```

Repeated for seeds `1` and `2`.

Outputs:

- `output/babi_attribute_paraphrase_stress_seed0/`
- `output/babi_attribute_paraphrase_stress_seed1/`
- `output/babi_attribute_paraphrase_stress_seed2/`
- `output/babi_attribute_paraphrase_stress_r139/aggregate_summary.csv`
- `output/babi_attribute_paraphrase_stress_r139/detector_summary.csv`
- `output/babi_attribute_paraphrase_stress_r139/seed_repeat.csv`
- `output/babi_attribute_paraphrase_stress_r139/selection.json`

## Raw Data Table

Three-seed test aggregate:

| Task | Train surface | Method | Mean acc | Std acc | Mean CE | State bytes | Raw text |
|---|---|---|---:|---:|---:|---:|---|
| QA15 | original | learned attribute binding | 0.567333 | 0.059744 | 2.346010 | 72,765 | false |
| QA15 | strong paraphrase | learned attribute binding | 1.000000 | 0.000000 | 0.003610 | 72,765 | false |
| QA15 | original | aware diagnostic | 1.000000 | 0.000000 | 0.003610 | 72,765 | false |
| QA16 | original | learned attribute binding | 0.332000 | 0.100464 | 2.255045 | 73,357 | false |
| QA16 | strong paraphrase | learned attribute binding | 0.995000 | 0.000000 | 0.041655 | 73,357 | false |
| QA16 | original | aware diagnostic | 0.995000 | 0.000000 | 0.041655 | 73,357 | false |

Learned front-end detector test aggregate:

| Task | Train surface | Statement event | Statement entity | Statement value | Query type | Query subject |
|---|---|---:|---:|---:|---:|---:|
| QA15 | original | 0.732167 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| QA15 | strong paraphrase | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| QA16 | original | 0.552333 | 1.000000 | 0.943593 | 1.000000 | 1.000000 |
| QA16 | strong paraphrase | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Findings

1. R138 is surface-form fragile. Training the learned front-end on canonical QA15/QA16 and evaluating on strong paraphrases drops QA15 to `0.567` accuracy and QA16 to `0.332`.

2. The failure localizes mainly to statement event recognition, not query parsing. Under original-train strong-test, QA15 statement event accuracy is `0.732` while entity/value/query/subject are all `1.000`; QA16 statement event accuracy is `0.552`, with value also weakened to `0.944`.

3. The same local no-BP front-end can adapt if the new surface forms are observed during training. Strong-paraphrase training restores QA15 to `1.000` and QA16 to `0.995`, matching R138's canonical performance.

4. The aware normalized path confirms the downstream attribute/category binding state is not the bottleneck. With diagnostic normalization, original-train strong-test reaches QA15 `1.000` and QA16 `0.995`; this is an upper-bound diagnostic, not the final method.

## Decision

R139 is a boundary result, not a new performance win. It shows that the current local learned attribute front-end behaves like a narrow grammar learner: it can learn a new paraphrase distribution locally, but it does not yet generalize from canonical forms to unseen paraphrases.

Next step:

- implement delayed QA-level credit for QA15/QA16 paraphrases, analogous to R098-R104, so final answer error can update statement-event and value-slot detectors without local structure labels on the paraphrased stream;
- add paraphrase-diverse training as a controlled positive condition, but keep diagnostic normalization out of the main method claim.
