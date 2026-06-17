# Iteration Report: Context Order Ablation

日期：2026-06-15

## 1. 目的

验证 sparse Hebbian context memory 的收益来自几阶 context，并估计 memory size 增长。该实验对应 `R016`。

## 2. 命令

每个 order 跑一次 TinyStories medium，跳过 Llama 以节省时间：

```bash
python tinystories_llama_token_experiment.py \
  --out-dir output/context_order_ablation/order_${order} \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --hidden-dim 32 --seq-len 16 \
  --llama-updates 2 --eval-token-limit 1000 --sample-len 16 \
  --curve-points 0 --curve-probe-tokens 64 \
  --context-max-order ${order} --plastic-temperature 0.8 \
  --hybrid-memory-weight 2.0 --hybrid-neural-weight 0.5 \
  --repetition-penalty 0.45 --no-repeat-ngram 4 \
  --skip-llama --no-progress
```

## 3. 结果

| max_order | CE | acc | active contexts | train tok/s | controlled repeat-2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.5608 | 0.300 | 256 | 184k | 0.106 |
| 2 | 4.1406 | 0.339 | 3,404 | 151k | 0.107 |
| 3 | 4.0796 | 0.359 | 9,778 | 62k | 0.102 |
| 4 | 4.1126 | 0.361 | 17,638 | 106k | 0.074 |
| 5 | 4.1539 | 0.370 | 26,054 | 94k | 0.016 |
| 6 | 4.2757 | 0.368 | 34,744 | 80k | 0.016 |

## 4. 结论

这是一个小正结果：

- CE 最优是 `max_order=3`: CE 4.0796，优于此前主结果 `max_order=4` 的 CE 4.1126。
- accuracy 最优是 `max_order=5`: acc 0.370，但 CE 退化到 4.1539。
- order 越高 active contexts 越多，泛化更差，CE 在 order 5/6 明显退化。
- 对当前 CE 主指标，下一步默认应使用 `context_max_order=3`。

## 5. 对自动探索的影响

exact n-gram memory 还有一点低成本收益，但不是最终解：

```text
order=3 提升 CE，但没有解决 GPT-like 语义连贯性。
```

下一步继续 `R023 semantic-key Hebbian memory`，目标是改善 low-confidence / unseen exact-context 泛化。

## 6. 文件

| file/dir | purpose |
|---|---|
| `output/context_order_ablation/summary.csv` | order sweep summary |
| `output/context_order_ablation/order_*/metrics.csv` | per-order token metrics |
| `output/context_order_ablation/order_*/generation_metrics.csv` | per-order generation metrics |
