# R144 Learned Relation Front-End for QA18/QA19

**Date**: 2026-06-19  
**Task**: bAbI `en-qa18` size reasoning and `en-qa19` path finding  
**Goal**: Remove R143's canonical regex statement/query front-end from the main relation-state path, using local no-BP detectors.

## Setup

R144 extends `babi_relation_state_experiment.py` with learned local front-ends:

- QA18 statement detector: `fit_inside` vs `bigger_than`, plus left/right entity slot prototypes.
- QA18 query detector: `fit_in` vs `bigger_than`, plus left/right entity slot prototypes.
- QA19 statement detector: local direction classifier over `north/south/east/west`, plus source/target place slot prototypes.
- QA19 query detector: source/target place slot prototypes.

All detectors use:

- fixed random token/position features;
- perceptron-style target/wrong-winner updates for relation/query/direction type;
- prototype averaging for slots;
- no BP/BPTT, pretrained encoder, API, raw-example replay, or test-answer update.

Claimable command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_relation_state_experiment.py \
  --out-dir output/babi_relation_state_r144_learned_seed0 \
  --configs en-qa18 en-qa19 \
  --relation-dim 128 \
  --phase-dim 64 \
  --phase-epochs 8 \
  --detector-dim 64 \
  --detector-epochs 3 \
  --seed 0
```

Repeated for seeds `1` and `2`.

Outputs:

- `output/babi_relation_state_r144_learned/aggregate_summary.csv`
- `output/babi_relation_state_r144_learned/detector_summary.csv`
- `output/babi_relation_state_r144_learned/seed_test_rows.json`
- `output/babi_relation_state_r144_learned_seed{0,1,2}/summary.csv`
- `output/babi_relation_state_r144_learned_seed{0,1,2}/detector_metrics.csv`

## Raw Data Table

Three-seed test aggregate:

| Config | Method | Mean acc | Std acc | Mean CE | Std CE | State bytes |
|---|---|---:|---:|---:|---:|---:|
| QA18 | phase_dendritic_no_bp | 0.486333 | 0.035311 | 0.702355 | 0.010359 | 3470 |
| QA18 | size_relation_state_no_bp | 0.930667 | 0.024459 | 0.155576 | 0.008733 | 65536 |
| QA18 | size_learned_relation_state_no_bp | 0.930667 | 0.024459 | 0.155576 | 0.008733 | 74118 |
| QA19 | phase_dendritic_no_bp | 0.098000 | 0.003266 | 2.483551 | 0.000731 | 18870 |
| QA19 | path_relation_state_no_bp | 0.946000 | 0.004320 | 0.301004 | 0.023879 | 262144 |
| QA19 | path_learned_relation_state_no_bp | 0.946000 | 0.004320 | 0.301004 | 0.023879 | 270653 |

Detector test aggregate:

| Config | Detector | Key metrics |
|---|---|---|
| QA18 | size_statement | event/left/right acc all `1.000` |
| QA18 | size_query | query/left/right acc all `1.000` |
| QA19 | path_statement | direction/source/target acc all `1.000` |
| QA19 | path_query | source/target acc all `1.000` |

## Findings

1. Learned front-ends exactly preserve R143 relation-state performance on canonical QA18/QA19. The learned variants match the regex-front-end variants in both accuracy and CE across all three seeds.

2. The local detectors solve the canonical statement/query parsing subproblem. Every reported detector slot/type metric is `1.000` on held-out test rows.

3. The remaining QA18/QA19 gap to symbolic upper is now localized to the relation-state matrix/recurrent readout, not to the canonical front-end. This makes the next mechanism choice sharper: branch separation, cleanup/inhibition, or candidate arbitration should target matrix superposition errors.

4. State overhead is modest. QA18 learned front-end adds about `8.6KB`; QA19 adds about `8.5KB`.

## Decision

R144 supersedes R143 as the canonical QA18/QA19 relation-state baseline because it removes the regex front-end from the claimable path without changing metrics.

Boundary:

- the learned front-end is validated on canonical bAbI grammar only;
- it still uses sentence-local structure labels for training, not delayed answer credit;
- paraphrase robustness is untested.

Next:

- add QA18/QA19 paraphrase stress;
- adapt relation detectors with final-answer delayed credit;
- add cleanup/inhibitory arbitration for relation matrix superposition before trying broader QA17/QA14 transfer.
