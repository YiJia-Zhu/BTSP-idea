# Repository Guidelines

## 最终目的
构建一个替代BP的更加高效的仿生物神经的学习算法。最终可以实现在线学习，例如自动浏览网页学习与视频流在线学习，无需保存数据直接在实际中学习。

## Project Structure & Module Organization

This repository contains lightweight NumPy experiments for biologically inspired, no-backprop learning.

- `temporal_association_experiment.py`: toy temporal association experiment, including the `00/01` sequence setup.
- `tinystories_llama_token_experiment.py`: TinyStories next-token experiment comparing STDP, STDP-Bio, BTSP, BTSP-Bio, DendriticError-1810-lite, and a random-init Llama-style torch baseline.
- `llama_torch_model.py`: standalone Llama-style causal decoder implementation. It defines architecture only and must not load pretrained weights.
- `data/`: local datasets, including TinyStories text files and small `.npy` assets.
- `output/`: generated metrics, checkpoints, figures, and samples. Treat as disposable experiment output.
- `*.md`, `*.pdf`, `*.ipynb`: research notes, source papers, and analysis notebooks.

## Build, Test, and Development Commands

There is no build step. Run experiments directly with Python.

```bash
python -m py_compile temporal_association_experiment.py tinystories_llama_token_experiment.py llama_torch_model.py
```

Checks syntax for the two maintained Python scripts.

```bash
python temporal_association_experiment.py
```

Runs the temporal association experiment and writes figures/results.

```bash
python tinystories_llama_token_experiment.py --out-dir output/tinystories_llama_token
```

Runs the TinyStories comparison. Use a smaller smoke test during development:

```bash
python tinystories_llama_token_experiment.py \
  --out-dir output/tinystories_llama_token_smoke \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --embed-dim 8 --hidden-dim 16 --seq-len 8 \
  --lstm-updates 2 --eval-token-limit 200 --sample-len 8
```

Add `--no-progress` when redirecting logs.

## Coding Style & Naming Conventions

Use Python 3.12-compatible code, 4-space indentation, type hints for public helpers, and concise comments only where the update rule is non-obvious. Keep no-BP methods in NumPy; keep supervised Llama-style code isolated in `llama_torch_model.py`. Do not load pretrained Llama weights. Keep method names explicit, for example `train_btsp_bio_matrix` or `evaluate_dendritic_error`.

## Testing Guidelines

No formal test framework is configured. Before committing, run `py_compile` and at least one smoke command. For algorithm changes, inspect `metrics.csv` and `greedy_completions.txt` under the chosen `output/` directory.

## Commit & Pull Request Guidelines

Git history currently uses very short messages, but future commits should be descriptive, e.g. `add progress reporting to TinyStories experiment`. Pull requests should include the changed experiment, command used, key metrics, and any generated figures or sample output relevant to the claim.

## Data & Output Hygiene

Do not commit large datasets, checkpoints, cache files, or `__pycache__/`. Keep reusable datasets in `data/`; write new generated artifacts under `output/<experiment_name>/`.
