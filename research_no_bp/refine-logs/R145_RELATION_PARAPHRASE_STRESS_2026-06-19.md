# R145 Relation Paraphrase Stress

**Date**: 2026-06-19  
**Task**: bAbI `en-qa18` size reasoning and `en-qa19` path finding  
**Goal**: Stress R144 learned relation front-ends under surface rewrites and localize whether errors come from relation-state matrices or from unseen relation words.

## Setup

R145 adds `babi_relation_paraphrase_stress_experiment.py`.

Strong rewrites:

- QA18 statements:
  - `The X fits inside the Y.` -> `The X can fit within the Y.`
  - `The X is bigger than the Y.` -> `The X is larger than the Y.`
- QA18 questions:
  - `Does the X fit in the Y?` -> `Can the X fit inside the Y?`
  - `Is the X bigger than the Y?` -> `Is the X larger than the Y?`
- QA19 statements:
  - `north` -> `above`
  - `south` -> `below`
  - `east` -> `to the right`
  - `west` -> `to the left`
- QA19 questions:
  - `How do you go from the X to the Y?` -> `What path takes you from the X to the Y?`

Compared conditions:

- `learned_original_train_strong_test`: train local detectors on canonical train rows, evaluate on strong paraphrased validation/test.
- `learned_strong_train_strong_test`: train local detectors on strong paraphrased train rows, evaluate on strong paraphrased validation/test.

The stress detectors use paraphrased text features and source-text labels only for constructing the local structure target, matching the R139/R144 style. No test labels are used for updates.

Claimable command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_relation_paraphrase_stress_experiment.py \
  --out-dir output/babi_relation_paraphrase_stress_r145_seed0 \
  --configs en-qa18 en-qa19 \
  --relation-dim 128 \
  --detector-dim 64 \
  --detector-epochs 3 \
  --seed 0
```

Repeated for seeds `1` and `2`.

Outputs:

- `output/babi_relation_paraphrase_stress_r145/aggregate_summary.csv`
- `output/babi_relation_paraphrase_stress_r145/detector_summary.csv`
- `output/babi_relation_paraphrase_stress_r145/seed_test_rows.json`

## Raw Data Table

Three-seed strong-test aggregate:

| Config | Condition | Mean acc | Std acc | Mean CE | Std CE |
|---|---|---:|---:|---:|---:|
| QA18 | original-train strong-test | 0.586333 | 0.049775 | 1.743335 | 0.338121 |
| QA18 | strong-train strong-test | 0.930667 | 0.024459 | 0.155576 | 0.008733 |
| QA19 | original-train strong-test | 0.060000 | 0.026870 | 8.017557 | 0.676406 |
| QA19 | strong-train strong-test | 0.946000 | 0.004320 | 0.301004 | 0.023879 |

Detector test aggregate:

| Config | Condition | Main failing detector metric |
|---|---|---:|
| QA18 | original-train strong-test | statement event acc `0.607310`; query acc `0.940333`; all slots `1.000` |
| QA18 | strong-train strong-test | statement/query type and slots all `1.000` |
| QA19 | original-train strong-test | statement direction acc `0.165533`; source/target/query slots `1.000` |
| QA19 | strong-train strong-test | direction/source/target/query slots all `1.000` |

## Findings

1. R144 relation front-ends are surface-fragile under unseen relation words. QA18 drops from `0.9307/0.1556` to `0.5863/1.7433`; QA19 drops from `0.9460/0.3010` to `0.0600/8.0176`.

2. The failures are localized to relation-type classification, not entity/place slot extraction. QA18 statement event accuracy drops to `0.6073`, while left/right slots remain `1.000`; QA19 direction accuracy drops to `0.1655`, while source/target and query slots remain `1.000`.

3. Local strong-surface training fully restores the R144 relation-state performance. QA18 returns to `0.9307/0.1556`; QA19 returns to `0.9460/0.3010`.

4. This is the right boundary for delayed answer-credit work. The relation-state matrices can still solve the task if the local detector maps the new relation words into the correct event/direction channels.

## Decision

R145 is a stress boundary, not a method regression. It shows that canonical learned front-ends are not semantic enough to transfer `bigger -> larger`, `fits inside -> fit within`, or cardinal directions to relative aliases.

Next:

- adapt QA18 statement event and QA19 statement direction detectors from final-answer credit;
- keep slot prototypes mostly unchanged because slots already transfer;
- use answer-error third-factor updates over local relation-word candidates, mirroring R140-R142.
