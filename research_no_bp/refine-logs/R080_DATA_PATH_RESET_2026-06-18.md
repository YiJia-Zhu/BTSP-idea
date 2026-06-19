# R080 Data Path Reset

**Date**: 2026-06-18

## Purpose

Reset the local data assumptions after the dataset directory was rebuilt. This run checks which datasets are actually present and whether the TinyStories scripts can run with the new default paths.

## Local Data Inventory

| Dataset | Local path | Status | First-stage use |
|---|---|---|---|
| TinyStories | `data/TinyStories/` | Full text files present | Yes, token/generation diagnostics |
| bAbI QA | `data/babi_qa/` | Hugging Face loader metadata present, no actual sample files cached/exported | Yes after sample download/export |
| GSM8k-Aug | `data/GSM8k-Aug/` | JSON files present with `question`, `cot`, `answer` keys | Later stress test |
| CLUTRR | n/a | Not present locally | Optional later only |

## Code Changes

Updated default TinyStories paths:

- `phase_binding_token_experiment.py`
- `tinystories_llama_token_experiment.py`

Downstream scripts inherit these defaults:

- `tinystories_online_stream_experiment.py`
- `phase_binding_online_stream_experiment.py`

## Verification

Syntax check:

```bash
python -m py_compile phase_binding_token_experiment.py tinystories_llama_token_experiment.py tinystories_online_stream_experiment.py phase_binding_online_stream_experiment.py
```

Passed.

TinyStories path smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 python phase_binding_token_experiment.py \
  --out-dir output/r080_phase_path_smoke \
  --train-chars 2000 --valid-chars 800 --max-vocab 64 \
  --phase-dim 8 --eval-token-limit 100 \
  --phase-epochs 1 --competitive-epochs 1
```

Result:

| method | loss | acc |
|---|---:|---:|
| phase_binding_token_no_bias | 4.365 | 0.116 |
| phase_binding_token | 3.751 | 0.163 |
| unigram_aux | 3.520 | 0.151 |
| sparse_context_aux | 3.630 | 0.233 |

Artifact:

- `output/r080_phase_path_smoke/metrics.csv`

## Interpretation

R080 passes for TinyStories path compatibility. bAbI is not ready for offline training yet because only loader metadata is local. The next step is to materialize bAbI QA samples into a project-local processed JSONL directory before implementing R081.
