# R140 Attribute Delayed QA-Credit

**Date**: 2026-06-19  
**Task**: bAbI `en-qa15` and `en-qa16`, full `900/100/1000` split  
**Goal**: Adapt the R139 paraphrase-fragile attribute front-end from final QA answer credit, without using local structure labels on the strong-paraphrase credit stream.

## Setup

R140 adds `babi_attribute_delayed_credit_experiment.py`.

Training stages:

1. Seed the statement/query detectors on canonical train rows, matching R139's `seeded_pre_credit` condition.
2. Rewrite the same train rows to strong paraphrases.
3. For train rows where the current model is wrong, enumerate local statement candidates from the paraphrased sentence surface only:
   - `fear/afraid` cues propose `class_afraid`;
   - `classified` / article cues propose `entity_class`;
   - `colored` / color-word cues propose `entity_color`.
4. Accept a candidate only if replacing that one detected event improves the final answer log-probability for the train label.
5. Use that improvement as a third-factor credit signal for local target-vs-wrong event updates and slot prototype writes.

Default R140 is conservative:

- statement credit only;
- query credit disabled, because R139 already had query accuracy `1.000`;
- error-only credit;
- two credit epochs, selected by validation behavior, not test labels.

No BP, pretrained model, API, raw replay, n-gram table, or test-label update is used.

Command pattern:

```bash
python babi_attribute_delayed_credit_experiment.py \
  --out-dir output/babi_attribute_delayed_credit_e2_seed0 \
  --configs en-qa15 en-qa16 \
  --credit-epochs 2 \
  --seed 0
```

Repeated for seeds `1` and `2`.

Outputs:

- `output/babi_attribute_delayed_credit_e2_seed0/`
- `output/babi_attribute_delayed_credit_e2_seed1/`
- `output/babi_attribute_delayed_credit_e2_seed2/`
- `output/babi_attribute_delayed_credit_r140/aggregate_summary.csv`
- `output/babi_attribute_delayed_credit_r140/detector_summary.csv`
- `output/babi_attribute_delayed_credit_r140/credit_summary.csv`
- `output/babi_attribute_delayed_credit_r140/selection.json`

## Raw Data Table

Three-seed test aggregate:

| Task | Method | Mean acc | Std acc | Mean CE | Std CE | Raw text |
|---|---|---:|---:|---:|---:|---|
| QA15 | seeded pre-credit | 0.567333 | 0.059744 | 2.346010 | 0.507952 | false |
| QA15 | answer credit | 1.000000 | 0.000000 | 0.003610 | 0.000407 | false |
| QA15 | strong structural upper | 1.000000 | 0.000000 | 0.002293 | 0.001647 | false |
| QA16 | seeded pre-credit | 0.332000 | 0.100464 | 2.255045 | 0.727103 | false |
| QA16 | answer credit | 0.970000 | 0.023643 | 0.132053 | 0.074650 | false |
| QA16 | strong structural upper | 0.995000 | 0.000000 | 0.042041 | 0.002174 | false |

Answer-credit detector test aggregate:

| Task | Statement event | Statement entity | Statement value | Query type | Query subject |
|---|---:|---:|---:|---:|---:|
| QA15 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| QA16 | 1.000000 | 1.000000 | 0.985481 | 1.000000 | 1.000000 |

Credit update aggregate:

| Task | Epoch | Statement updates | Query updates | Skipped correct | Mean gain / row |
|---|---:|---:|---:|---:|---:|
| QA15 | 0 | 4.33 | 0.00 | 895.67 | 0.0283 |
| QA15 | 1 | 0.00 | 0.00 | 900.00 | 0.0000 |
| QA16 | 0 | 69.33 | 0.00 | 821.00 | 0.2526 |
| QA16 | 1 | 35.00 | 0.00 | 859.67 | 0.1246 |

## Findings

1. Final-answer credit closes the R139 paraphrase gap for QA15. Accuracy improves from `0.567` to `1.000`, matching the strong-structure-label upper bound.

2. QA16 also improves sharply, from `0.332` to `0.970`, but remains below the structural upper bound `0.995`. The residual bottleneck is statement value coverage: answer-credit detector value accuracy is `0.985`, while event/entity/query metrics are `1.000`.

3. Query credit is unnecessary and harmful in this setting. A smoke run with query credit degraded query accuracy despite answer gains, so the claimable R140 default is statement-only credit.

4. The useful learning signal is sparse. QA15 needs only about 4-5 statement updates per seed after canonical seeding; QA16 needs more, about 69 updates in epoch 0 and 35 in epoch 1 on average.

## Decision

R140 is a positive result. It shows that the paraphrase adaptation problem from R139 can be mostly solved by local detector updates gated by final QA answer improvement, without local structure labels on the paraphrased stream.

Boundary:

- candidate generation is still surface-cue constrained, not open-domain semantic parsing;
- QA16 residual errors show that final-answer credit does not always cover every value-slot prototype;
- next step should add an eligibility trace over all answer-relevant statements so one final answer error can distribute credit to multiple class/color facts, instead of only the best one-sentence replacement.
