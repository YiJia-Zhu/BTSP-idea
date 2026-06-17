# Iteration Report: Confidence-Gated No-BP Context Adapter

日期：2026-06-15

## 1. 本轮目标

本轮按新的最终目标推进：

```text
no-BP 在线学习方法推理样本看起来接近主流 GPT 输出，同时本地 CE/accuracy/速度过关。
```

本轮不使用付费 API，先在 TinyStories tokenizer-level medium 设置上验证：

1. confidence-gated adapter 是否比 always-on linear hybrid 更好；
2. 生成样本的重复问题能否被量化和缓解；
3. 下一步是否继续调 gate，还是需要换思路。

## 2. 新增实现

位置：`tinystories_llama_token_experiment.py`

新增：

- `GatedBackoffConfig`
- `SparseHebbianContextMemory.confidence`
- `evaluate_gated_dendritic_context`
- `evaluate_gated_llama_context`
- `sample_gated_dendritic_context`
- `sample_gated_llama_context`
- `DecodingConfig`
- controlled decoding:
  - repetition penalty
  - no-repeat ngram
  - optional top-k sampling
- `generation_metrics.csv`:
  - distinct-1 / distinct-2
  - repeat-2 / repeat-3 rate
  - longest repeated token run

门控逻辑：

```text
if high-order context exists and row count/probability/entropy indicate confidence:
    use sparse memory logits
else:
    use neural backoff + memory logits
```

## 3. 关键命令

Medium run:

```bash
python tinystories_llama_token_experiment.py \
  --out-dir output/gated_context_decode_medium \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --hidden-dim 32 --seq-len 16 \
  --llama-hidden-dim 64 --llama-intermediate-dim 128 \
  --llama-layers 2 --llama-heads 4 --llama-kv-heads 1 \
  --llama-batch-size 4 --llama-updates 80 --eval-every 40 \
  --eval-batches 8 --eval-token-limit 1000 --sample-len 32 \
  --curve-points 5 --curve-probe-tokens 64 \
  --context-max-order 4 --plastic-temperature 0.8 \
  --hybrid-memory-weight 2.0 --hybrid-neural-weight 0.5 \
  --gate-min-order 4 --gate-min-row-total 0.20 \
  --gate-min-max-prob 0.55 --gate-max-entropy 1.50 \
  --repetition-penalty 0.45 --no-repeat-ngram 4 \
  --no-progress
```

## 4. Token-Level 结果

| method | CE | PPL | acc | train tok/s | 判断 |
|---|---:|---:|---:|---:|---|
| sparse_hebbian_context | 4.1126 | 61.11 | 0.361 | 91.6k | 仍是本轮最强 CE/acc |
| hybrid_dendritic_context | 4.5630 | 95.87 | 0.342 | 7.2k | 线性融合负结果 |
| gated_dendritic_context | 4.4624 | 86.70 | 0.344 | 7.2k | 优于线性融合，但输给 sparse-only |
| torch_llama, 80 updates | 4.8748 | 130.95 | 0.117 | 3.4k | 低预算 BP baseline 仍弱 |
| hybrid_llama_context | 4.5369 | 93.40 | 0.357 | 9.2k | 线性融合负结果 |
| gated_llama_context | 4.4457 | 85.26 | 0.358 | 9.2k | 优于线性融合，但输给 sparse-only |

门控诊断：

| subset | CE | acc | rate |
|---|---:|---:|---:|
| high-confidence memory | 2.1966 | 0.696 | 0.204 |
| gated dendritic backoff positions | 5.0431 | 0.254 | 0.796 |
| gated llama backoff positions | 5.0222 | 0.271 | 0.796 |

结论：

```text
gate 本身有效地找到了强 memory 区域，但当前 neural backoff 在低置信区域太弱。
继续扫 gate 阈值意义有限；需要更强的 no-BP 表示/semantic key/backoff。
```

## 5. 生成质量结果

Controlled decoding 不改变 CE，只用于推理样本质量。

| method | distinct-2 | repeat-2 rate | repeat-3 rate | 判断 |
|---|---:|---:|---:|---|
| sparse_hebbian_context | 0.427 | 0.573 | 0.553 | 重复严重 |
| sparse_hebbian_context_controlled | 0.827 | 0.173 | 0.062 | 重复大幅下降 |
| gated_llama_context | 0.333 | 0.667 | 0.658 | 重复严重 |
| gated_llama_context_controlled | 0.819 | 0.181 | 0.071 | 重复大幅下降 |
| LlamaTorch 80 updates | 0.271 | 0.729 | 0.667 | 极弱 baseline |

样本判断：

- controlled decoding 能消除大量循环，如 `He wanted to the`。
- 但语义仍然不像 GPT：常出现拼接式 story fragment 和不连贯短语。
- 因此推理层控制只能修外观，不能解决核心泛化能力。

## 6. 本轮结论

本轮是部分正结果、机制负反馈：

1. `gated_*` 比 `linear hybrid` 明显更合理，说明门控方向正确。
2. `gated_*` 仍输给 `sparse_hebbian_context`，说明当前 backoff 模型弱，不能提升低置信 context。
3. high-confidence memory 很强，low-confidence context 是主要瓶颈。
4. controlled decoding 明显改善重复率，但不能让输出接近 GPT。

## 7. 下一步

不继续大扫 gated 阈值。下一步应转向：

1. **Context order / memory size ablation**
   - 明确 sparse memory 的收益来自几阶 context。
2. **Semantic-key no-BP memory**
   - 用 frozen embedding/API embedding 或本地 embedding 做 key。
   - value 仍用 no-BP Hebbian/statistical update。
   - 目标是让未见 exact n-gram 也能泛化。
3. **wwwy4/huggingface_datasets 任务评估**
   - 选择 next-token / QA / classification 子任务。
   - 报告 accuracy、CE/proxy CE、训练速度、存储。
4. **API 只用于最终小预算评估**
   - 本地验证 semantic memory 有收益后，再比较 API/no-memory vs API+no-BP-memory。

## 8. 文件

| file/dir | purpose |
|---|---|
| `output/gated_context_smoke/` | first gated smoke |
| `output/gated_context_medium/` | gated medium without generation metrics |
| `output/gated_context_decode_smoke/` | controlled decoding smoke |
| `output/gated_context_decode_medium/metrics.csv` | main token metrics |
| `output/gated_context_decode_medium/generation_metrics.csv` | generation repetition/diversity metrics |
| `output/gated_context_decode_medium/greedy_completions.txt` | qualitative samples |
