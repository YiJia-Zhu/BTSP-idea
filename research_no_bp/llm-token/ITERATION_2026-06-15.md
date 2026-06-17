# Iteration Report: TinyStories Token-Level No-BP Online Memory

日期：2026-06-15

## 1. 本轮目标

上一阶段已经证明：

- 三因子 DFA 在小规模 MNIST 上可以接近 BP。
- e-prop 风格 eligibility trace 在合成 delayed next-token 任务上可以学会跨时间记忆。

本轮转向更贴近最终目标的 TinyStories tokenizer-level next-token 任务：

```text
过去 token context -> 下一个 Llama tokenizer token
```

目标不是直接宣称替代 GPT，而是低成本验证：

1. 非 BP / 局部在线规则是否能在真实文本 next-token 上超过当前 STDP/BTSP/dendritic baselines。
2. 哪条路线值得进入下一轮更大规模实验。

## 2. 文献依据和设计选择

本轮参考三条路线：

| 路线 | 依据 | 本轮实现判断 |
|---|---|---|
| eligibility trace + fixed feedback | Bellec et al. 2020 e-prop；Lillicrap et al. 2016 random feedback | 实现为 `recurrent_3factor`，但 TinyStories 上不稳定 |
| reservoir / fixed dynamics + readout | reservoir computing language model 方向；上一轮 delayed task 中 reservoir 在简单设置有效 | 做了 hidden-update-off 消融，未超过 dendritic baseline |
| Hebbian associative context memory | Hebbian/SoftHebb/local statistics 思路；语言局部 n-gram 统计很强 | 实现为 `sparse_hebbian_context`，得到本轮正信号 |

关键现实判断：

截至本轮调研，没有发现可信的、严格 no-BP、端到端训练且达到现代主流 GPT 级别的大语言模型。因此本项目应先走：

```text
冻结/弱训练 backbone + no-BP online memory/adapter
```

而不是一开始声称完全替代大规模 Transformer BP 预训练。

## 3. 新增方法

### 3.1 RecurrentThreeFactor

位置：`tinystories_llama_token_experiment.py`

形式：

```text
h_t = tanh(W_in x_t + W_rec h_{t-1})
e_out = softmax(W_out h_t) - onehot(y_t)
m_h = B e_out
eligibility <- decay * eligibility + local_derivative * pre_activity
Delta W_hidden = -lr * m_h * eligibility
```

不使用 BPTT，不使用 `W_rec.T` 反传未来误差。

结论：在 TinyStories token task 上是负结果。小规模 smoke 中 final CE 接近 dendritic，但中等规模明显退化。

### 3.2 SparseHebbianContext

位置：`tinystories_llama_token_experiment.py`

形式：

```text
for context ngram c and next token y:
    M[c, y] += alpha
score(y | c) = unigram_score + sum_order M[c_order, y] * order_weight
```

特点：

- 完全在线。
- 不做 BP/BPTT。
- 不保存原始文本样本，只保存稀疏 context-to-token 统计。
- 可作为 frozen LLM 的外部在线 memory/adapter 原型。

局限：

- 本质接近稀疏 n-gram/Hebbian associative memory。
- 能提高局部 top-1 prediction，但仍会生成重复短语。
- 还不是抽象组合推理或 GPT 级语言能力。

## 4. 关键实验

### 4.1 Smoke

命令：

```bash
python tinystories_llama_token_experiment.py \
  --out-dir output/sparse_hebbian_context_smoke \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --hidden-dim 16 --seq-len 8 --llama-updates 2 \
  --eval-token-limit 200 --sample-len 8 --curve-points 3 \
  --curve-probe-tokens 32 --skip-llama --no-progress
```

结果：

| method | CE | PPL | acc | train tok/s |
|---|---:|---:|---:|---:|
| dendritic_error_1810_lite | 4.1976 | 66.53 | 0.090 | 14,188 |
| sparse_hebbian_context | 4.2010 | 66.75 | 0.240 | 83,050 |
| recurrent_3factor | 4.2063 | 67.11 | 0.100 | 12,668 |
| btsp_trace | 4.5258 | 92.37 | 0.150 | 56,174 |
| stdp_trace | 4.7664 | 117.49 | 0.190 | 76,684 |

解释：

- `sparse_hebbian_context` 的 CE 与 dendritic baseline 接近，但 top-1 accuracy 明显更高。
- greedy samples 不再像 BTSP/dendritic 那样快速陷入逗号循环。

### 4.2 Medium

命令：

```bash
python tinystories_llama_token_experiment.py \
  --out-dir output/sparse_hebbian_context_medium \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --hidden-dim 32 --seq-len 16 \
  --llama-hidden-dim 64 --llama-intermediate-dim 128 \
  --llama-layers 2 --llama-heads 4 --llama-kv-heads 1 \
  --llama-batch-size 4 --llama-updates 80 --eval-every 40 \
  --eval-batches 8 --eval-token-limit 1000 --sample-len 24 \
  --curve-points 5 --curve-probe-tokens 64 \
  --context-max-order 4 --plastic-temperature 0.8 --no-progress
```

结果：

| method | CE | PPL | acc | train seconds | train tok/s |
|---|---:|---:|---:|---:|---:|
| sparse_hebbian_context | 4.1126 | 61.11 | 0.361 | 0.104 | 92,204 |
| dendritic_error_1810_lite | 4.5777 | 97.29 | 0.118 | 2.691 | 3,565 |
| btsp_trace | 4.6779 | 107.54 | 0.147 | 0.320 | 29,974 |
| torch_llama, 80 updates | 4.8748 | 130.95 | 0.117 | 1.389 | 3,687 |
| recurrent_3factor | 5.0553 | 156.85 | 0.068 | 1.946 | 4,930 |
| stdp_trace | 5.2052 | 182.22 | 0.240 | 0.167 | 57,338 |

`sparse_hebbian_context` 使用 17,638 个 active contexts。

### 4.3 Normalized Backoff Check

为了改善 CE 校准，新增：

```bash
--context-score-mode normalized --plastic-temperature 1.0
```

同等 medium 设置下：

| variant | CE | PPL | acc | note |
|---|---:|---:|---:|---|
| additive score | 4.1126 | 61.11 | 0.361 | 主结果，CE 最好 |
| normalized backoff | 4.1510 | 63.50 | 0.357 | 生成样本略自然，但 CE 未超过 additive |

结论：normalized backoff 是有价值的生成/校准方向，但本轮主指标仍采用 additive score。

## 5. 支持的结论

本轮支持一个有限但有价值的 claim：

> 在 TinyStories tokenizer-level 小规模 next-token 设置中，稀疏 Hebbian context-to-token online memory 明显强于当前 STDP/BTSP/dendritic no-BP baselines，并且在低训练预算下优于一个随机初始化小 Llama-style BP baseline。

具体证据：

- medium run 中 `sparse_hebbian_context` 相比 `dendritic_error_1810_lite`：
  - CE 从 4.5777 降到 4.1126
  - accuracy 从 0.118 提到 0.361
  - CPU 训练速度约 92k tok/s，而 dendritic baseline 约 3.6k tok/s
- 相比 `torch_llama` 80 updates：
  - CE 4.1126 vs 4.8748
  - accuracy 0.361 vs 0.117

## 6. 不支持的结论

本轮不支持：

- 已经接近主流 GPT 模型。
- 已经可以替代大规模 BP 预训练。
- sparse Hebbian context 具备长程抽象推理能力。
- recurrent three-factor 在真实文本上已经有效。

特别注意：

`torch_llama` 这里只训练了 80 updates，属于低预算随机初始化 baseline，不是充分训练的 Transformer，更不是 GPT 主流模型。

## 7. 负结果与删除策略

保留的负结果：

- `recurrent_3factor`：smoke 接近 dendritic，但 medium 明显退化。
- hidden update off / reservoir readout 消融：未超过 dendritic baseline。

删除的中间输出：

- 临时 sweep 目录已删除，只保留关键结果目录：
  - `output/tinystories_recurrent3factor_smoke`
  - `output/sparse_hebbian_context_smoke`
  - `output/sparse_hebbian_context_medium`
  - `output/sparse_hebbian_context_normalized_medium`

这样避免把未成体系的调参日志误当作有效结论。

## 8. 下一轮计划

下一轮不应继续微调 recurrent_3factor。更有价值的方向是：

1. **Sparse Hebbian Memory + Neural Backoff**
   - 当前 sparse memory 对已见局部上下文强，但未见 context 退化。
   - 加一个 frozen/random reservoir 或 dendritic readout 做 backoff。

2. **Online Adapter Prototype**
   - 冻结小 Llama embeddings 或 logits。
   - sparse Hebbian memory 学 residual logits：
     ```text
     final_logits = backbone_logits + lambda * memory_logits
     ```
   - 测在线学习是否能快速适应新故事风格。

3. **Continual / No Raw Data Test**
   - 逐段在线喂入 TinyStories。
   - 不保存文本，只保存 context memory。
   - 测旧风格遗忘、新风格适应和 memory size。

4. **Calibration**
   - 已试 `normalized` backoff，medium CE 未超过 additive。
   - 下一步应尝试 count normalization、Kneser-Ney style continuation backoff、repetition penalty。

## 9. 本轮文件

| file/dir | purpose |
|---|---|
| `tinystories_llama_token_experiment.py` | 新增 `recurrent_3factor` 与 `sparse_hebbian_context` |
| `output/sparse_hebbian_context_smoke/metrics.csv` | smoke result |
| `output/sparse_hebbian_context_medium/metrics.csv` | main result |
| `output/sparse_hebbian_context_normalized_medium/metrics.csv` | normalized backoff calibration check |
| `output/sparse_hebbian_context_medium/greedy_completions.txt` | generation samples |
| `research_no_bp/llm-token/ITERATION_2026-06-15.md` | 本迭代报告 |
