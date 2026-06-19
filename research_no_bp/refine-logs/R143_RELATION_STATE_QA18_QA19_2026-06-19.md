# R143 Relation-State QA18/QA19 Expansion

**Date**: 2026-06-19  
**Task**: bAbI `en-qa18` size reasoning and `en-qa19` path finding, full local splits  
**Goal**: Use the newly materialized bAbI configs to test relation reasoning beyond QA1/2/3 movement and QA15/16 attribute induction.

## Setup

Data materialized:

```bash
PYTHONDONTWRITEBYTECODE=1 python export_babi_qa_jsonl.py \
  --configs en-qa1 en-qa2 en-qa3 en-qa15 en-qa16 en-qa14 en-qa17 en-qa18 en-qa19
```

The official bAbI tarball URL returned 404, so the exporter used the existing HuggingFace fallback `Muennighoff/babi`. New local splits:

- `en-qa14`: train 900 / validation 100 / test 1000
- `en-qa17`: train 904 / validation 96 / test 1000
- `en-qa18`: train 905 / validation 95 / test 1000
- `en-qa19`: train 900 / validation 100 / test 1000

Implementation:

- `babi_relation_state_experiment.py`
- QA18 `size_relation_state_no_bp`: one fixed-random-code relation matrix maps smaller-object codes to larger-object codes. Transitive reasoning uses finite recurrent retrieval through the matrix.
- QA19 `path_relation_state_no_bp`: one transition matrix per direction. Each statement writes forward and reverse local delta-Hebbian transitions. Candidate answer strings are scored by applying their direction sequence to the source code.
- No BP/BPTT, pretrained model, API, raw-example replay, or test-answer update.
- Regex parsers are only the first local sensory front-end, analogous to R137 before learned front-ends were added.

Claimable command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 python babi_relation_state_experiment.py \
  --out-dir output/babi_relation_state_r143_final_seed0 \
  --configs en-qa18 en-qa19 \
  --relation-dim 128 \
  --phase-dim 64 \
  --phase-epochs 8 \
  --seed 0
```

Repeated for seeds `1` and `2`.

Outputs:

- `output/babi_relation_state_r143_final/aggregate_summary.csv`
- `output/babi_relation_state_r143_final/seed_test_rows.json`
- `output/babi_relation_state_r143_final_seed{0,1,2}/summary.csv`

## Raw Data Table

Three-seed test aggregate:

| Config | Method | Mean acc | Std acc | Mean CE | Std CE | State bytes |
|---|---|---:|---:|---:|---:|---:|
| QA18 | majority_no_memory | 0.469000 | 0.000000 | 0.694139 | 0.000000 | 146 |
| QA18 | raw_lexical_retrieval | 0.529000 | 0.000000 | 0.692892 | 0.000000 | 381245 |
| QA18 | hashed_lookup_diagnostic | 0.469000 | 0.000000 | 11.004051 | 0.000000 | 2623 |
| QA18 | phase_dendritic_no_bp | 0.486333 | 0.035311 | 0.702355 | 0.010359 | 3470 |
| QA18 | symbolic_size_graph_upper | 1.000000 | 0.000000 | 0.000335 | 0.000000 | 0 |
| QA18 | size_relation_state_no_bp | 0.930667 | 0.024459 | 0.155576 | 0.008733 | 65536 |
| QA19 | majority_no_memory | 0.081000 | 0.000000 | 2.484038 | 0.000000 | 186 |
| QA19 | raw_lexical_retrieval | 0.081000 | 0.000000 | 2.484829 | 0.000000 | 333926 |
| QA19 | hashed_lookup_diagnostic | 0.081000 | 0.000000 | 19.044681 | 0.000000 | 9604 |
| QA19 | phase_dendritic_no_bp | 0.098000 | 0.003266 | 2.483551 | 0.000731 | 18870 |
| QA19 | symbolic_path_graph_upper | 1.000000 | 0.000000 | 0.003683 | 0.000000 | 0 |
| QA19 | path_relation_state_no_bp | 0.946000 | 0.004320 | 0.301004 | 0.023879 | 262144 |

## Findings

1. Relation-state matrices give a strong positive result on new bAbI tasks. QA18 improves over majority `0.469` and raw retrieval `0.529` to `0.931`; QA19 improves over majority/raw `0.081` and phase `0.098` to `0.946`.

2. The result is not a statistical lookup win. Hashed lookup stays at majority on both tasks and has very poor CE, while the relation-state model uses fixed random neural codes plus local matrix writes.

3. The structural upper bound is still `1.000`, so the remaining gap is not data ambiguity. It likely comes from matrix superposition and finite recurrent retrieval, especially when several relations share the same entity code in one story.

4. A quick `relation_dim=256` rerun did not improve the default: QA18 dropped to about `0.911` mean acc and QA19 to about `0.933`. The current bottleneck is not solved by simply increasing dimensionality.

## Decision

R143 is a positive expansion result: the no-BP state-binding line now covers location/object movement, attribute/category induction, size transitivity, and two-hop path finding.

Boundary:

- the sensory front-end is still regex/grammar-specific;
- QA18/QA19 do not yet use delayed final-answer credit;
- relation matrices are lossy under superposition, so the next mechanism should add role/branch separation, inhibitory cleanup, or learned candidate arbitration before claiming structural upper-bound reasoning.

Next:

- add learned local statement/query detectors for QA18/QA19, mirroring R138;
- test answer-credit adaptation for size/path paraphrases, mirroring R140-R142;
- optionally materialize QA17 positional reasoning and QA14 time reasoning as the next relation-state tasks.
