# R081 Prep: bAbI QA Materialization

**Date**: 2026-06-18

## Purpose

Materialize bAbI QA into project-local JSONL files for pure no-BP QA training. This avoids depending on implicit Hugging Face cache paths during R081/R082.

## Source Resolution

The local `data/babi_qa/` directory contains the Hugging Face loader metadata, but no sample files. The original URL embedded in the loader is currently unavailable:

```text
http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
https://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz
```

Both returned 404 during export.

The exporter therefore uses the standard-format Hugging Face fallback:

```text
Muennighoff/babi
```

This fallback provides `passage`, `question`, `answer`, and `task` fields. It does not include explicit supporting-fact ids, so `supporting_ids` and `supporting_context` are empty in the processed JSONL. That is acceptable for first-stage answer classification; supporting-fact supervision is not used by the planned pure no-BP learner.

## Export Utility

Added:

- `export_babi_qa_jsonl.py`

The script:

- reads local bAbI loader metadata for official config naming;
- tries the official archive when requested;
- falls back to `Muennighoff/babi` in `--source auto` or `--source hf-fallback`;
- exports one JSONL per config/split under `data/babi_qa_processed/`.

## Command

```bash
PYTHONDONTWRITEBYTECODE=1 python export_babi_qa_jsonl.py \
  --source hf-fallback \
  --configs en-qa1 en-qa2 en-qa3 en-qa15 en-qa16
```

## Output

| Config | Train | Validation | Test | Answer count | Answers |
|---|---:|---:|---:|---:|---|
| en-qa1 | 900 | 100 | 1000 | 6 | bathroom, bedroom, garden, hallway, kitchen, office |
| en-qa2 | 900 | 100 | 1000 | 6 | bathroom, bedroom, garden, hallway, kitchen, office |
| en-qa3 | 900 | 100 | 1000 | 6 | bathroom, bedroom, garden, hallway, kitchen, office |
| en-qa15 | 900 | 100 | 1000 | 4 | cat, mouse, sheep, wolf |
| en-qa16 | 900 | 100 | 1000 | 4 | gray, green, white, yellow |

Files:

- `data/babi_qa_processed/en-qa1/{train,validation,test}.jsonl`
- `data/babi_qa_processed/en-qa2/{train,validation,test}.jsonl`
- `data/babi_qa_processed/en-qa3/{train,validation,test}.jsonl`
- `data/babi_qa_processed/en-qa15/{train,validation,test}.jsonl`
- `data/babi_qa_processed/en-qa16/{train,validation,test}.jsonl`
- `data/babi_qa_processed/summary.json`

## Verification

- `python -m py_compile export_babi_qa_jsonl.py` passed.
- JSONL files were generated and sampled successfully.
- `data/babi_qa_processed/` is about 11 MB.

## Next Step

Implement R081: a pure no-BP bAbI QA1 answer selector with no-memory, raw retrieval, statistical lookup, reservoir/e-prop, and phase/dendritic/apical no-BP comparisons. Test answers must not be used for updates.
