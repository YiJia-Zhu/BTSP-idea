# 2026-06-16 Output Fatigue Dynamics

本轮在 phase/WTA 与 trace branch 之上加入局部输出疲劳/抑制动态，目标是让模型状态本身减少重复循环，而不是只靠解码时的 repetition penalty。

## 方法

代码：`phase_binding_online_stream_experiment.py`

新增 wrapper：`OutputFatigueMemory`

动态：

```text
fatigue[target] += 1
fatigue <- decay * fatigue
scores <- base_scores - strength * fatigue
```

在严格在线评估中：

- `stream_pre` / `stream_post`: 预测后用真实 target 更新疲劳，但不更新权重。
- `stream_online`: 预测后先做局部权重更新，再用真实 target 更新疲劳。
- generation: prompt token 和预测 token 都更新疲劳。

为避免 counterfactual 评估污染训练状态，pre/post/retention/generation 都使用 `deepcopy` 出来的模型副本。模型状态不保存原始文本；生成审计文件保存 decoded prompt/reference/generated text 只作为 artifact。

## Smoke

命令：

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_fatigue_smoke \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --warmup-token-limit 1200 --stream-token-limit 600 \
  --segment-tokens 128 \
  --trace-branch --trace-order 8 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --completion-count 2 --prompt-tokens 12 --completion-tokens 24
```

Smoke result:

| method | post CE | post acc | greedy repeat-2 | greedy distinct-2 |
|---|---:|---:|---:|---:|
| `phase_competitive_online` | 2.055 | 0.485 | 0.543 | 0.457 |
| `phase_fatigue_competitive_online` | 2.039 | 0.479 | 0.478 | 0.522 |
| `phase_trace_competitive_online` | 2.050 | 0.490 | 0.217 | 0.783 |
| `phase_trace_fatigue_competitive_online` | 2.030 | 0.492 | 0.152 | 0.848 |

Smoke showed a strong repetition reduction, especially when fatigue is combined with trace.

## Medium

Command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_fatigue_medium \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --completion-count 4 --prompt-tokens 16 --completion-tokens 48
```

Medium stream results:

| method | pre CE / acc | online CE / acc | post CE / acc | state bytes |
|---|---:|---:|---:|---:|
| `phase_competitive_online` | 3.425 / 0.303 | 3.229 / 0.335 | 2.427 / 0.422 | 2,367,488 |
| `phase_fatigue_competitive_online` | 3.415 / 0.305 | 3.215 / 0.337 | 2.419 / 0.422 | 2,368,512 |
| `phase_trace_competitive_online` | 3.339 / 0.303 | 3.157 / 0.335 | 2.389 / 0.427 | 2,498,560 |
| `phase_trace_fatigue_competitive_online` | 3.330 / 0.303 | 3.141 / 0.339 | 2.382 / 0.429 | 2,499,584 |
| `sparse_context_aux` | 3.835 / 0.323 | 3.721 / 0.334 | 1.297 / 0.571 | 175,568 |

Medium generation results:

| method | greedy repeat-2 | greedy distinct-2 | controlled repeat-2 | controlled distinct-2 | first-token | prefix tokens |
|---|---:|---:|---:|---:|---:|---:|
| `phase_competitive_online` | 0.771 | 0.229 | 0.170 | 0.830 | 0.750 | 4.75 |
| `phase_fatigue_competitive_online` | 0.612 | 0.388 | 0.176 | 0.824 | 0.750 | 4.75 |
| `phase_trace_competitive_online` | 0.681 | 0.319 | 0.176 | 0.824 | 0.750 | 5.00 |
| `phase_trace_fatigue_competitive_online` | 0.606 | 0.394 | 0.144 | 0.856 | 0.750 | 5.00 |

## 判断

正信号：

- Output fatigue is a real dynamic-state mechanism and improves both CE/acc and repetition when combined with trace.
- `phase_trace_fatigue_competitive_online` is the current strongest pure no-BP token learner:
  - post CE `2.382`, acc `0.429`
  - online CE `3.141`, acc `0.339`
  - greedy repeat-2 `0.606` vs phase baseline `0.771`
- It costs only one float fatigue vector: state increases from `2,498,560` to `2,499,584` bytes for trace+fatigue.

边界：

- The text is still weak. Fatigue changes a tight loop into a longer template loop, e.g. "She was a big, and. They the...".
- Controlled decoding is still much more effective for suppressing repetition, but relying on it alone would not be a method contribution.
- `sparse_context_aux` remains a stronger post-online baseline, but it is still a statistical cache and only an auxiliary comparator.
- This is not yet near GPT/API quality; it is a concrete step toward a biologically plausible online state mechanism.

Next step:

1. Make fatigue adaptive or class-specific instead of fixed global strength.
2. Add local inhibitory competition inside the readout update, not only as a score offset.
3. Test seed/data-slice stability for trace+fatigue.
4. Begin low-precision state audit for phase/trace/fatigue states.
