# 2026-06-16 Context-Gated Output Inhibition

本轮继续沿纯 no-BP token learner 主线，尝试把上一轮 adaptive output inhibition 从“全局输出抑制”改成“上下文门控抑制”。目标是只在相似上下文里压制错误 winner，减少全局 flatten logits 带来的 CE 损害。

## 方法

代码：`phase_binding_online_stream_experiment.py`

新增 wrapper：`ContextGatedOutputInhibitionMemory`

机制：

```text
gate = fixed_random_context_encoder(context)
scores = base_scores - strength * inhibition @ gate

if wrong output beats target:
  inhibition[wrong, :] += lr * gate
  inhibition[target, :] -= disinhibit_lr * gate
```

它只使用当前 context、当前 target 和局部 winner 竞争信号；没有 BP/BPTT、预训练模型、API 主干或原始文本 replay。`sparse_context_aux` 仍只是统计辅助对照。

实现注意：

- 初版 `gate_decay` 只更新了 `dynamic_gate` 但没有进入 score path；已修成 `effective_gate(context) = normalize(context_gate + decay * dynamic_gate)`。
- 修复后 `gate_decay=0.8` 在 smoke 上退化，说明短时 gate 混合会把抑制扩散过宽。
- 最终 medium 使用 `gate_decay=0.0`，即纯 context-gated inhibition。

## Smoke

Best smoke config:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_gate_inhib_tracefix_smoke/decay0 \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --warmup-token-limit 1200 --stream-token-limit 600 \
  --segment-tokens 128 \
  --trace-branch --trace-order 8 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --context-gated-inhibition \
  --gate-inhibit-strength 2.0 --gate-inhibit-lr 0.02 \
  --gate-inhibit-disinhibit-lr 0.004 --gate-inhibit-top-k 4 \
  --gate-dim 64 --gate-decay 0.0 \
  --completion-count 2 --prompt-tokens 12 --completion-tokens 24
```

Smoke result:

| method | post CE | post acc | greedy repeat-2 |
|---|---:|---:|---:|
| `phase_competitive_online` | 2.055 | 0.485 | 0.543 |
| `phase_trace_fatigue_competitive_online` | 2.030 | 0.492 | 0.152 |
| `phase_trace_gate_inhib_competitive_online` | 2.002 | 0.490 | 0.022 |
| `phase_trace_fatigue_gate_inhib_competitive_online` | 1.994 | 0.508 | 0.022 |

Smoke was strongly positive, but medium is the deciding test.

## Medium

Command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_gate_inhib_medium_s200_lr020_k4_decay0 \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --context-gated-inhibition \
  --gate-inhibit-strength 2.0 --gate-inhibit-lr 0.02 \
  --gate-inhibit-disinhibit-lr 0.004 --gate-inhibit-top-k 4 \
  --gate-dim 64 --gate-decay 0.0 \
  --completion-count 4 --prompt-tokens 16 --completion-tokens 48
```

Medium stream results:

| method | online CE / acc | post CE / acc | greedy repeat-2 | state bytes |
|---|---:|---:|---:|---:|
| `phase_competitive_online` | 3.229 / 0.335 | 2.427 / 0.422 | 0.771 | 2,367,488 |
| `phase_trace_competitive_online` | 3.157 / 0.335 | 2.389 / 0.427 | 0.681 | 2,498,560 |
| `phase_trace_fatigue_competitive_online` | 3.141 / 0.339 | 2.382 / 0.429 | 0.606 | 2,499,584 |
| `phase_trace_gate_inhib_competitive_online` | 3.181 / 0.330 | 2.413 / 0.434 | 0.388 | 3,612,928 |
| `phase_trace_fatigue_gate_inhib_competitive_online` | 3.178 / 0.331 | 2.416 / 0.435 | 0.346 | 3,613,952 |
| `sparse_context_aux` | 3.721 / 0.334 | 1.297 / 0.571 | 0.234 | 175,568 |

## 判断

正信号：

- Context-gated inhibition improves greedy diversity much more than trace/fatigue:
  - trace+fatigue repeat-2 `0.606`
  - trace+fatigue+gate repeat-2 `0.346`
- It also improves top-1 acc over trace/fatigue:
  - post acc `0.429 -> 0.435`

边界：

- CE worsens relative to weak adaptive inhibition and trace/fatigue:
  - previous best `phase_trace_inhib_competitive_online`: post CE `2.358`
  - trace+fatigue baseline: post CE `2.382`
  - trace+fatigue+gate: post CE `2.416`
- Extra state is nontrivial: trace+fatigue+gate is `3.61MB` vs trace+fatigue `2.50MB`.
- Text samples still contain template loops; the model is less repetitive but not close to GPT/API quality.

Conclusion:

Context-gated inhibition is a useful quality/top-1 tradeoff, not the current best CE model. Do not spend more turns only tuning gate strength. The next meaningful step is a plastic recurrent/SSM branch whose state participates directly in local WTA learning.
