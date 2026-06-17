# Iteration Report: Prequential No-Raw-Data Online Stream

日期：2026-06-15

## 1. 目的

上一轮 `continuation_backoff` 在离线 train-prefix 后 eval 的设置下取得当前最强 CE。这个结果还没有证明最终目标里的关键约束：

```text
无需保存原始数据，边接收 token 流边学习，并在后续推理中立即变好。
```

本轮实现 `R026`，把 TinyStories tokenizer-level 任务改成严格 prequential stream：

- 先用 train prefix 做在线 warmup；
- valid prefix 按连续 segment 流入；
- 每段先用当前 memory 评估，再在线更新 memory；
- 更新后不保存原始文本，只保留 memory state；
- 记录 segment CE、accuracy、active contexts、state bytes。

## 2. 新增实现

新增脚本：

```text
tinystories_online_stream_experiment.py
```

复用现有实现：

- `SparseHebbianContextMemory`
- `SemanticHebbianMemory`
- `ContinuationBackoffMemory`
- tokenizer / compact vocab / decoding metrics helper

比较方法：

| method | 说明 |
|---|---|
| `sparse_hebbian_context` | exact context Hebbian memory |
| `combined_context` | exact context + random-projection semantic bucket |
| `continuation_backoff` | discounted exact counts + continuation distribution |

输出：

| file | purpose |
|---|---|
| `summary.csv` | method-level warmup / stream pre / online / post metrics |
| `segment_metrics.csv` | segment-level CE、acc、memory size |
| `segment_curve.png` | segment CE curve |
| `generation_metrics.csv` | held-out prompt repetition/diversity |
| `greedy_completions.txt` | qualitative samples |

## 3. Smoke

命令：

```bash
python tinystories_online_stream_experiment.py \
  --out-dir output/online_stream_smoke \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --segment-tokens 128 --sample-len 8 \
  --context-max-order 3 --continuation-max-order 3 \
  --stream-token-limit 500 --warmup-token-limit 1500 \
  --no-progress
```

结果：

| method | stream pre CE | stream post CE | stream post acc | state bytes |
|---|---:|---:|---:|---:|
| sparse_hebbian_context | 4.126 | 3.434 | 0.711 | 71,316 |
| combined_context | 4.104 | 3.388 | 0.718 | 132,045 |
| continuation_backoff | 3.365 | 1.469 | 0.881 | 132,325 |

Smoke 通过，说明脚本路径、stream chunk、CSV/figure/generation 输出正常。

## 4. Medium

命令：

```bash
python tinystories_online_stream_experiment.py \
  --out-dir output/online_stream_medium \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 --sample-len 24 \
  --context-max-order 3 --semantic-order 8 --semantic-hash-bits 12 \
  --continuation-max-order 3 \
  --stream-token-limit 5000 --warmup-token-limit 20000 \
  --no-progress
```

结果：

| method | warmup CE | stream pre CE | stream online CE | stream post CE | stream post acc | active contexts | state bytes | bytes / target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_hebbian_context | 4.701 | 4.136 | 4.109 | 3.580 | 0.512 | 10,973 | 336,343 | 29.1 |
| combined_context | 4.687 | 4.125 | 4.098 | 3.547 | 0.511 | 13,205 | 511,912 | 44.4 |
| continuation_backoff | 4.055 | 3.395 | 3.332 | 1.542 | 0.791 | 10,973 | 617,631 | 53.5 |

关键结论：

- `continuation_backoff` 在 strict stream 下仍是最强方法。
- valid 流更新前 CE 为 `3.395`，更新后同一 valid prefix 的 post CE 为 `1.542`。
- online-pass CE 为 `3.332`，说明边预测边更新时已经优于 update-free stream pre。
- 更新过程没有 replay 原始文本，只保存 context/continuation 统计状态。

## 5. 生成质量

Medium controlled generation 指标：

| method | distinct-2 | repeat-2 rate | repeat-3 rate |
|---|---:|---:|---:|
| sparse_hebbian_context | 0.849 | 0.151 | 0.056 |
| combined_context | 0.882 | 0.098 | 0.033 |
| continuation_backoff | 1.000 | 0.000 | 0.000 |

样本仍然不接近 GPT：`continuation_backoff` 重复少，但有拼接、语法缺失和 `<|endoftext|>` 泄出。它证明的是快速局部适应，不证明 GPT-like 语义生成。

## 6. 判断

这是一个强正结果，但 claim 仍需收窄：

支持：

```text
no-BP 统计/Hebbian online memory 可以在不保存原始训练段的情况下对当前 token 流快速适应，并显著降低后续 next-token CE。
```

不支持：

- 已经接近 GPT/API 输出质量；
- memory state 一定满足强隐私要求；
- memory size 已经可长期受控；
- 已经验证跨域/旧任务 retention。

## 7. 下一步

立即下一步应做：

1. `R017` memory pruning / decay：控制 continuation memory 增长，报告质量-存储 tradeoff。
2. `R018` retention / style stream：在线学新段后检查旧段是否崩。
3. `R019` API-compatible prototype：先构造小规模 personalization/FAQ，本地验证任务 accuracy，再小预算接 API。

## 8. 文件

| file/dir | purpose |
|---|---|
| `tinystories_online_stream_experiment.py` | strict prequential stream experiment |
| `output/online_stream_smoke/summary.csv` | smoke method summary |
| `output/online_stream_medium/summary.csv` | medium method summary |
| `output/online_stream_medium/segment_metrics.csv` | segment-level stream metrics |
| `output/online_stream_medium/generation_metrics.csv` | generation repetition/diversity |
| `output/online_stream_medium/greedy_completions.txt` | qualitative completions |
