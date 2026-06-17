# Iteration Report: Continuation / Kneser-Ney-Style No-BP Memory

日期：2026-06-15

## 1. 目的

上一轮 `semantic-key memory` 只把 CE 从 4.0796 降到 4.0368，增益很小。根据自动探索策略，本轮转向更强的统计平滑：

```text
exact context counts + continuation counts + discount backoff
```

这仍然是 no-BP、在线、局部统计更新，不保存原始文本。

## 2. 方法

新增：

- `ContinuationBackoffConfig`
- `ContinuationBackoffMemory`
- `train_continuation_backoff_memory`
- `evaluate_continuation_backoff_memory`

核心思想：

- 每个 context 记录 next-token count；
- 每个 token 记录它可跟随的不同 context 数，即 continuation count；
- prediction 时用 discounted exact counts + continuation distribution 做 backoff；
- 没有梯度，没有 BP，没有保存原文样本。

## 3. Medium 结果

设置：

- `train_chars=50000`
- `valid_chars=10000`
- `max_vocab=256`
- `context_max_order=3`
- `continuation_max_order=3`

主结果：

| method | CE | PPL | acc | train tok/s |
|---|---:|---:|---:|---:|
| sparse_hebbian_context, order=3 | 4.0796 | 59.12 | 0.359 | 130k |
| combined_context semantic best prior | 4.0368 | 56.65 | 0.344 | ~70k |
| continuation_backoff | 3.3304 | 27.95 | 0.365 | 143k |

这是一轮明显正结果：

- 相比 exact sparse memory，CE 改善 `0.7492`。
- 相比 semantic combined，CE 改善 `0.7064`。
- accuracy 也小幅提高到 `0.365`。
- 训练速度仍然很高，约 `140k tok/s`。

## 4. Sweep

扫：

- `discount in {0.25, 0.5, 0.75, 1.0}`
- `exact_backoff in {0.2, 0.4, 0.6}`

最佳 CE：

| discount | exact_backoff | CE | acc |
|---:|---:|---:|---:|
| 0.75 | 0.2 | 3.3254 | 0.360 |
| 0.75 | 0.4 | 3.3304 | 0.365 |
| 0.5 | 0.4 | 3.3454 | 0.363 |

最佳 accuracy：

| discount | exact_backoff | CE | acc |
|---:|---:|---:|---:|
| 0.25 | 0.6 | 3.3860 | 0.366 |
| 0.75 | 0.4 | 3.3304 | 0.365 |

默认推荐：

```text
discount=0.75, exact_backoff=0.4
```

理由：CE 接近最优，accuracy 更高。

## 5. 判断

这是目前最强本地结果。

但需要保持边界：

- 它仍然是统计 language model / cache-LM 风格，不是 GPT-like 语义生成。
- 样本仍会出现短语拼接和不连贯。
- 目前是在离线 train-prefix 后 eval，不是严格 online prequential。

因此下一步必须验证：

1. **Online no-raw-data stream**
   - prequential eval: before update -> update -> after update。
2. **更大 valid / 多 split**
   - 防止 TinyStories 前缀切片偶然性。
3. **任务数据集 accuracy**
   - 不能只看 CE。

## 6. 文件

| file/dir | purpose |
|---|---|
| `output/continuation_medium/metrics.csv` | main continuation result |
| `output/continuation_sweep/summary.csv` | continuation sweep summary |
| `output/continuation_sweep/d*_b*/metrics.csv` | per-run metrics |

