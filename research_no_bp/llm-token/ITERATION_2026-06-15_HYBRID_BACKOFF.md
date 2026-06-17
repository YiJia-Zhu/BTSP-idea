# Iteration Report: Hybrid Backoff for No-BP Online Memory

日期：2026-06-15

## 1. 大方向规划

最终目标不是在第一步就从零训练一个 GPT 级 no-BP 模型，而是构建：

```text
冻结/弱训练主干 + no-BP 在线记忆 adapter
```

理由：

- 主干负责未见上下文的平滑泛化。
- Hebbian memory 负责现场快速写入和局部精确回忆。
- 未来接 GPT/API 时，可以不保存原始数据，只保存局部统计或压缩记忆。

本轮测试最小形式：

```text
final_logits = neural_weight * neural_logits + memory_weight * hebbian_memory_logits
```

其中 neural logits 来自两种 no/low-BP backoff：

1. `dendritic_error_1810_lite`
2. 低预算 `torch_llama`

## 2. 新增实现

位置：`tinystories_llama_token_experiment.py`

新增：

- `HybridBackoffConfig`
- `evaluate_hybrid_dendritic_context`
- `evaluate_hybrid_llama_context`
- `sample_hybrid_dendritic_context`
- `sample_hybrid_llama_context`
- CLI:
  - `--hybrid-memory-weight`
  - `--hybrid-neural-weight`

这些 fusion 只发生在 logits 层，不对 memory 做 BP，也不额外训练 Llama。

## 3. Smoke 结果

设置：

```bash
--train-chars 10000 --valid-chars 3000 --max-vocab 128
--hybrid-memory-weight 1.0 --hybrid-neural-weight 1.0
```

结果：

| method | CE | PPL | acc |
|---|---:|---:|---:|
| sparse_hebbian_context | 4.201 | 66.75 | 0.240 |
| hybrid_dendritic_context | 3.668 | 39.18 | 0.200 |

解释：

- CE 明显下降，说明 neural logits 可改善概率校准。
- accuracy 和 greedy generation 变差，说明 dendritic logits 会把 argmax 往逗号/重复 token 拉。

小 sweep 后选中：

```text
memory_weight=2.0, neural_weight=0.5
```

## 4. Medium 结果

主设置和上一轮 medium 一致：

```bash
python tinystories_llama_token_experiment.py \
  --out-dir output/hybrid_llama_context_medium \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --hidden-dim 32 --seq-len 16 \
  --llama-hidden-dim 64 --llama-intermediate-dim 128 \
  --llama-layers 2 --llama-heads 4 --llama-kv-heads 1 \
  --llama-batch-size 4 --llama-updates 80 --eval-every 40 \
  --eval-batches 8 --eval-token-limit 1000 --sample-len 24 \
  --curve-points 5 --curve-probe-tokens 64 \
  --context-max-order 4 --plastic-temperature 0.8 \
  --hybrid-memory-weight 2.0 --hybrid-neural-weight 0.5 --no-progress
```

结果：

| method | CE | PPL | acc |
|---|---:|---:|---:|
| sparse_hebbian_context | 4.1126 | 61.11 | 0.361 |
| hybrid_llama_context | 4.5369 | 93.40 | 0.357 |
| hybrid_dendritic_context | 4.5630 | 95.87 | 0.342 |
| dendritic_error_1810_lite | 4.5777 | 97.29 | 0.118 |
| torch_llama | 4.8748 | 130.95 | 0.117 |

## 5. 结论

本轮是负结果，但有用：

> 线性 logits fusion 不是正确 adapter 形式。它能在小 smoke 上改善 CE，但在 medium 上破坏 sparse memory 的强局部统计，使 CE 从 4.1126 退化到 4.5369。

因此，不应继续扫线性权重。正确下一步是：

```text
gated adapter:
    if memory context confident:
        use memory logits
    else:
        back off to neural/base logits
```

也就是让主干只处理 memory 不确定或未见上下文，而不是始终线性混入。

## 6. 下一轮计划

实现 `gated_context_adapter`：

1. 为每个 context 计算 confidence：
   - max row count
   - row entropy
   - total context count
   - whether high-order context exists
2. 若 confidence 高，使用 memory logits。
3. 若 confidence 低，使用 normalized lower-order backoff 或 Llama logits。
4. 对比：
   - sparse memory only
   - linear hybrid
   - gated hybrid
5. 成功标准：
   - CE < 4.1126 或 accuracy > 0.361，且 greedy repetition 不恶化。

## 7. 本轮文件

| file/dir | purpose |
|---|---|
| `output/hybrid_dendritic_context_smoke/metrics.csv` | smoke positive CE / worse argmax |
| `output/hybrid_dendritic_context_medium/metrics.csv` | dendritic linear hybrid negative medium |
| `output/hybrid_llama_context_smoke/metrics.csv` | Llama hybrid smoke |
| `output/hybrid_llama_context_medium/metrics.csv` | Llama linear hybrid negative medium |
| `research_no_bp/llm-token/ITERATION_2026-06-15_HYBRID_BACKOFF.md` | 本轮报告 |

