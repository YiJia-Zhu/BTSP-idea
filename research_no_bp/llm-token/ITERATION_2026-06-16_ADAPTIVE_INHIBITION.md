# 2026-06-16 Adaptive Output Inhibition

本轮在 phase/WTA、trace branch 和 output fatigue 之上加入局部可塑输出抑制，目标是把“抑制哪些错误赢家”写入模型状态，而不是只用固定 fatigue 分数偏置。

## 方法

代码：`phase_binding_online_stream_experiment.py`

新增 wrapper：`AdaptiveOutputInhibitionMemory`

机制：

```text
activity[target] += 1
activity <- decay * activity
scores <- base_scores - strength * inhibition @ activity

if wrong output beats target:
  inhibition[wrong, :] += lr * activity
  inhibition[target, :] -= disinhibit_lr * activity
```

其中 `activity` 是最近输出神经元迹，`inhibition` 是局部可塑抑制边。更新只使用当前 target token 和局部输出活动，不使用 BP/BPTT、预训练模型、API 主干或原始文本 replay。`sparse_context_aux` 仍只作为统计辅助对照。

## Smoke And Tuning

初始 smoke 使用较强抑制 `strength=0.35, lr=0.02, top_k=4`：

- trace+inhibition acc 小幅提升，但 CE 变差。
- greedy repeat-2 明显下降，说明机制能打断短循环，但强度过大时校准受损。

随后小网格显示 `strength=0.30, lr=0.010, top_k=2` 在 smoke 上最好：

| method | post CE | post acc | greedy repeat-2 |
|---|---:|---:|---:|
| `phase_competitive_online` | 2.055 | 0.485 | 0.543 |
| `phase_trace_competitive_online` | 2.050 | 0.490 | 0.217 |
| `phase_trace_inhib_competitive_online` | 1.994 | 0.508 | 0.022 |

但 medium 上这个强度损害 trace 的 CE，因此继续调弱。

## Medium Best Config

Command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_inhibition_combo_medium_s015_lr005_k1 \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --completion-count 4 --prompt-tokens 16 --completion-tokens 48
```

Seed 0 stream results:

| method | online CE / acc | post CE / acc | greedy repeat-2 | state bytes |
|---|---:|---:|---:|---:|
| `phase_competitive_online` | 3.229 / 0.335 | 2.427 / 0.422 | 0.771 | 2,367,488 |
| `phase_trace_competitive_online` | 3.157 / 0.335 | 2.389 / 0.427 | 0.681 | 2,498,560 |
| `phase_trace_fatigue_competitive_online` | 3.141 / 0.339 | 2.382 / 0.429 | 0.606 | 2,499,584 |
| `phase_trace_inhib_competitive_online` | 3.114 / 0.335 | 2.358 / 0.429 | 0.473 | 2,761,728 |
| `phase_trace_fatigue_inhib_competitive_online` | 3.113 / 0.341 | 2.363 / 0.432 | 0.436 | 2,762,752 |
| `sparse_context_aux` | 3.721 / 0.334 | 1.297 / 0.571 | 0.234 | 175,568 |

Seed 1 repeat check:

| method | post CE / acc |
|---|---:|
| `phase_competitive_online` | 2.497 / 0.416 |
| `phase_trace_competitive_online` | 2.450 / 0.420 |
| `phase_trace_fatigue_competitive_online` | 2.440 / 0.426 |
| `phase_trace_inhib_competitive_online` | 2.410 / 0.431 |
| `phase_trace_fatigue_inhib_competitive_online` | 2.414 / 0.434 |

## 判断

正信号：

- Adaptive inhibition gives a stable within-seed CE/acc gain over trace-only:
  - seed 0: `2.389 -> 2.358`, acc `0.427 -> 0.429`
  - seed 1: `2.450 -> 2.410`, acc `0.420 -> 0.431`
- The balanced trace+fatigue+inhibition point improves top-1 further:
  - seed 0: post CE `2.363`, acc `0.432`
  - seed 1: post CE `2.414`, acc `0.434`
- The method remains pure no-BP and stores only learned neural state.

边界：

- Generation remains weak. Seed 0 repeat improves, but seed 1 shows that fatigue alone can be better for repetition. The mechanism is a stream CE/acc improvement, not yet a GPT-like generation solution.
- Strong inhibition improves diversity but damages CE; the useful region is weak, sparse inhibition.
- `sparse_context_aux` remains much stronger on post-online CE/acc, but it is a statistical cache and is not a final method candidate.

Next step:

1. Make inhibition state context-gated or branch-gated so it suppresses wrong local attractors without globally flattening useful logits.
2. Add a plastic recurrent/SSM branch whose state participates in WTA learning, instead of relying only on fixed trace codes.
3. Run low-precision/sparse-state audit for phase/trace/inhibition matrices.
