# Iteration Report: Stream Memory Pruning and Capacity Control

日期：2026-06-15

## 1. 目的

上一轮已经确认 strict prequential no-raw-data stream 是可行的，但 memory state 仍会继续增长。为了更接近最终目标里的“可控存储”，本轮对在线 memory 做容量约束：

- `prune_min_count`
- `prune_decay`
- `prune_max_contexts`

目标不是把 CE 压到最好，而是得到可解释的质量-存储折中。

## 2. 方法

实验脚本：

```text
tinystories_online_stream_experiment.py
```

对以下 memory 做裁剪：

- `sparse_hebbian_context`
- `combined_context`
- `continuation_backoff`

裁剪时机：

- `prune_every_segments=4`

裁剪方式：

- `min_count` 低于阈值的条目删除
- `decay` 对所有计数做整体衰减
- `max_contexts` 保留总计数最高的 context

## 3. 结果

### 3.1 激进 min-count pruning

设置：

```text
prune_every_segments=1
prune_min_count=2
```

结果：

| method | CE | acc | state bytes |
|---|---:|---:|---:|
| sparse_hebbian_context | 4.852 | 0.084 | 983 |
| combined_context | 4.852 | 0.084 | 37,759 |
| continuation_backoff | 5.943 | 0.178 | 6,290 |

结论：太激进，质量崩溃，不能用作最终设置。

### 3.2 decay pruning

设置：

```text
prune_every_segments=4
prune_min_count=0.25
prune_decay=0.98
```

结果：

| method | stream post CE | stream post acc | state bytes |
|---|---:|---:|---:|
| sparse_hebbian_context | 4.526 | 0.286 | 6,046 |
| combined_context | 4.520 | 0.285 | 76,281 |
| continuation_backoff | 1.598 | 0.789 | 604,975 |

This mainly shows that decay alone is not the right lever for continuation memory if the goal is to preserve quality.

### 3.3 context cap pruning

设置：

```text
prune_every_segments=4
prune_max_contexts=5000
```

结果：

| method | stream post CE | stream post acc | state bytes |
|---|---:|---:|---:|
| sparse_hebbian_context | 3.765 | 0.431 | 200,553 |
| combined_context | 3.837 | 0.406 | 321,385 |
| continuation_backoff | 1.957 | 0.599 | 351,859 |

再压到 `prune_max_contexts=2000` 后：

| method | stream post CE | stream post acc | state bytes |
|---|---:|---:|---:|
| sparse_hebbian_context | 3.897 | 0.396 | 124,723 |
| combined_context | 3.968 | 0.366 | 246,767 |
| continuation_backoff | 2.252 | 0.454 | 215,542 |

## 4. 结论

这是一个有用的 tradeoff 结果：

```text
continuation_backoff 可以在无原文 replay 的前提下在线适应，
而且可以通过 context cap 把状态从 ~618KB 压到 ~352KB 或 ~216KB，
代价是 CE 从 1.542 退到 1.957 / 2.252。
```

这足以支持下一步做 API-compatible prototype 时的存储预算设计，但还不足以直接宣称已经满足最终隐私与长期存储约束。

## 5. 下一步

1. 继续把 `continuation_backoff` 作为本地 online adapter 基线。
2. 在更像实际应用的数据上做小规模 API-compatible prototype。
3. 继续测 retention，避免只会记最近 segment。

