# 2026-06-16 Phase Trace Branch

本轮针对上一轮暴露的生成循环问题，给在线 phase/WTA 模型增加一个纯 no-BP 的 leaky trace / SSM-like 分支。目标不是调解码，而是把短程动态状态加入模型特征本身。

## 方法

代码：`phase_binding_online_stream_experiment.py`

新增方法：`phase_trace_competitive_online`

结构：

```text
branch_1: token_{t-1} phase -> target
branch_2: token_{t-2}, token_{t-1} phase -> target
trace: h_t = decay * h_{t-1} + code[token_t]
readout: local WTA/perceptron over [branch_1, branch_2, trace]
```

约束：

- trace token codes 随机初始化并固定。
- phase branch 和 WTA readout 仍用局部更新。
- 不使用 BP/BPTT/API/预训练模型。
- 模型状态只保存相位码、trace codes、原型和读出权重，不保存原始文本。

## Smoke Sweep

命令模板：

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_trace_sweep_smoke/<config> \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --warmup-token-limit 1200 --stream-token-limit 600 \
  --segment-tokens 128 --trace-branch \
  --trace-order 8 --trace-dim 64 \
  --completion-count 2 --prompt-tokens 12 --completion-tokens 24
```

Smoke trace results:

| config | post CE | post acc | greedy repeat-2 | controlled repeat-2 | greedy distinct-2 |
|---|---:|---:|---:|---:|---:|
| phase baseline | 2.055 | 0.485 | 0.543 | 0.152 | 0.457 |
| `w=0.25, decay=0.85` | 2.050 | 0.487 | 0.217 | 0.065 | 0.783 |
| `w=0.50, decay=0.85` | 2.050 | 0.490 | 0.217 | 0.065 | 0.783 |
| `w=0.50, decay=0.50` | 2.064 | 0.475 | 0.174 | 0.065 | 0.826 |
| `w=1.00, decay=0.50` | 2.128 | 0.452 | 0.196 | 0.043 | 0.804 |

Smoke gave a positive signal: light/moderate trace weight sharply reduced greedy repetition while keeping CE/acc roughly unchanged.

## Medium

Best medium command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_trace_sweep_medium/w050_o16_d085 \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 --trace-branch \
  --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --completion-count 4 --prompt-tokens 16 --completion-tokens 48
```

Medium stream results:

| method | pre CE / acc | online CE / acc | post CE / acc | state bytes |
|---|---:|---:|---:|---:|
| `phase_competitive_online` | 3.425 / 0.303 | 3.229 / 0.335 | 2.427 / 0.422 | 2,367,488 |
| `phase_trace_competitive_online` | 3.339 / 0.303 | 3.157 / 0.335 | 2.389 / 0.427 | 2,498,560 |
| `sparse_context_aux` | 3.835 / 0.323 | 3.721 / 0.334 | 1.297 / 0.571 | 175,568 |

Medium generation audit:

| method | mode | first-token | prefix tokens | distinct-2 | repeat-2 |
|---|---|---:|---:|---:|---:|
| `phase_competitive_online` post | greedy | 0.750 | 4.75 | 0.229 | 0.771 |
| `phase_trace_competitive_online` post | greedy | 0.750 | 5.00 | 0.319 | 0.681 |
| `phase_competitive_online` post | controlled | 0.750 | 4.75 | 0.830 | 0.170 |
| `phase_trace_competitive_online` post | controlled | 0.750 | 5.00 | 0.824 | 0.176 |

## 判断

正信号：

- trace branch gives a real model-state improvement, not only a decoding trick.
- Medium post CE improves `2.427 -> 2.389`, acc `0.422 -> 0.427`.
- Online CE improves `3.229 -> 3.157`.
- Greedy distinct-2 improves `0.229 -> 0.319`, and repeat-2 drops `0.771 -> 0.681`.

边界：

- The generation loop is not solved. Text still repeats fragments like "She was a big, and".
- Controlled decoding is still much more effective than trace alone for repetition.
- The trace feature is fixed random and only gives short-term dynamics; it does not yet provide semantic planning or robust variable-length memory.
- `sparse_context_aux` remains a stronger post-online baseline but is still a statistical cache, not the final method.

Next step:

1. Add local inhibitory/fatigue dynamics inside model state, evaluated during both stream scoring and generation.
2. Try a plastic trace code update or gated trace readout so the recurrent branch learns useful attractors instead of fixed random codes.
3. Run seed/data-slice repeats for `trace_order=16, trace_weight=0.5, decay=0.85`.
4. Combine trace branch with low-precision/sparse-state audit.
