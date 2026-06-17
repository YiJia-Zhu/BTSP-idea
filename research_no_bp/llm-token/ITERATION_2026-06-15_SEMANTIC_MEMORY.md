# Iteration Report: Semantic-Key Hebbian Memory

日期：2026-06-15

## 1. 目的

本轮回答一个更关键的问题：

```text
exact n-gram Hebbian memory 到底还有没有低成本可挖的泛化空间？
```

对应探索目标 `R023` / `R024`。

## 2. 方法

新增了一个纯 no-BP 的 semantic-key memory：

- context token 先经过固定随机 embedding；
- 再用固定随机投影 + SimHash 生成 bucket key；
- 每个 bucket 在线累积 next-token 分布；
- 可与 exact context memory 线性组合；
- 不训练 embedding，不用 BP，不保存原文样本。

## 3. 主要实验

### 3.1 Smoke

设置：

- `train_chars=10000`
- `valid_chars=3000`
- `max_vocab=128`
- `context_max_order=3`
- `semantic_hash_bits=10`

结果：

| method | CE | acc |
|---|---:|---:|
| sparse_hebbian_context | 4.242 | 0.250 |
| semantic_hebbian | 4.797 | 0.110 |
| combined_context | 4.224 | 0.230 |

结论：semantic-only 很弱，但 combined 比 exact-only 略好。

### 3.2 Medium

设置：

- `train_chars=50000`
- `valid_chars=10000`
- `max_vocab=256`
- `context_max_order=3`

结果：

| method | CE | acc | 说明 |
|---|---:|---:|---|
| sparse_hebbian_context | 4.0796 | 0.359 | exact-only baseline |
| semantic_hebbian | 5.3597 | 0.127 | semantic-only 很弱 |
| combined_context | 4.0658 | 0.358 | 比 exact-only 略好 |

### 3.3 Sweep

扫 `hash_bits in {10,12,14}`，`combine_weight in {0.2,0.5,1.0}`。

最佳 CE 组合：

| hash_bits | combine_weight | CE | acc | low-conf CE |
|---:|---:|---:|---:|---:|
| 10 | 1.0 | 4.0368 | 0.344 | 4.7086 |
| 12 | 1.0 | 4.0529 | 0.351 | 4.7291 |
| 10 | 0.5 | 4.0571 | 0.352 | 4.7351 |

## 4. 结论

这是一个小但明确的正结果：

1. `combined_context` 一直优于 `semantic_hebbian` 单独使用，说明 semantic bucket 不是噪声。
2. `combined_context` 在 medium 上可把 CE 从 `4.0796` 降到 `4.0368`。
3. 但 accuracy 并未同步提升，最佳 CE 点还牺牲了 accuracy。
4. low-confidence CE 仍在 `4.71~4.76`，离真正解决泛化还很远。

因此：

```text
semantic-key memory 是有效方向，但当前这版还只是“辅助 exact memory 的微增益”，不是最终解。
```

## 5. 对探索的意义

这说明：

- exact n-gram memory 还有一点可挖空间；
- 但要接近 GPT-like 样本，不能只靠 exact context 和记忆混合；
- 下一步应转向更强的 context 表示，或真实任务评估；
- 如果继续沿记忆路线，最好引入更稳的 frozen representation key，而不是纯随机 hash。

## 6. 文件

| file/dir | purpose |
|---|---|
| `output/semantic_memory_medium/` | medium run |
| `output/semantic_memory_sweep/summary.csv` | sweep summary |
| `output/semantic_memory_sweep/b*/metrics.csv` | sweep metrics |

