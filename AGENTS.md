# Repository Guidelines

## 最终目的
构建一个**纯 no-BP** 的仿生物神经学习算法。核心方向必须从头使用局部可塑性、资格迹、反馈对齐、树突/抑制性回路等机制来学习，不能把“已经用 BP 预训练好的开源 LLM / 冻结大模型 / API 主干”作为主方法再外挂局部记忆来充数。

允许把预训练 LLM 仅当作后续对照、应用接口或工程上界，但**不能**作为论文主线、机制主张或最终硬件路线的基础。若要面向在线学习、硬件实现或类脑结构，必须优先做纯 no-BP 原型，再逐步扩展到更大规模模型。

`continuation_backoff`、n-gram、Kneser-Ney-style sparse memory 等方法只能作为调试工具、统计下界或 sanity baseline；它们不属于最终可接受的仿生神经结构路线。

最终候选必须是**神经网络类 no-BP 方法**，例如 dendritic/DEN1810-like 双室或树突误差模型、STDP/BTSP 及其调制版本、e-prop/三因子 eligibility、反馈对齐或抑制性回路模型。统计表格方法可以帮助定位数据、评测和记忆瓶颈，但不能替代神经结构本身。


## API地址与密钥
注意：API耗费金钱，你最好使用本地数据测试正确后才开始大规模测试
API: https://yzhanghmeng.com
KEY: sk-6d999a81b2692a0b25d41a3d942deaa5b58c8e8098011ae5f2992591e2383555

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
