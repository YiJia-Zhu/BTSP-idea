# 2026-06-16 Phase-Binding Online Stream

本轮把 `phase_binding_token_experiment.py` 里的竞争分支相位模型迁到严格在线流评估。主方法仍是纯 no-BP：随机/局部初始化，相位绑定分支、target-gated Hebbian 原型、局部 winner-take-all 读出；没有 BP/BPTT、预训练 LLM 或 API。`sparse_context_aux` 只作为统计辅助基线，不进入最终方法。

## 方法

新脚本：`phase_binding_online_stream_experiment.py`

主方法 `phase_competitive_online`：

```text
branch_1: token_{t-1} -> target
branch_2: token_{t-2}, token_{t-1} -> target
readout: local WTA/perceptron over branch features
```

每个 token 的流程是严格 prequential：

```text
predict next token -> score loss/acc -> observe target -> local update
```

模型状态只包含相位码、目标锚点、分支原型、输出 bias 和 WTA 读出权重；不保存原始训练文本。脚本同时跑 `sparse_context_aux`，但它是二阶上下文计数缓存，只用于判断数据/评估和在线适应上界。

## Smoke

命令：

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_smoke_fixed \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --warmup-token-limit 1200 --stream-token-limit 600 \
  --segment-tokens 128
```

结果：

| method | stream pre CE / acc | stream online CE / acc | stream post CE / acc |
|---|---:|---:|---:|
| `phase_competitive_online` | 3.436 / 0.264 | 3.365 / 0.278 | 2.055 / 0.485 |
| `sparse_context_aux` | 4.109 / 0.255 | 4.085 / 0.255 | 0.852 / 0.646 |

## Medium

命令：

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_medium_fixed \
  --train-chars 50000 --valid-chars 10000 \
  --max-vocab 256 --segment-tokens 256
```

默认候选配置：

```text
phase_dim=128
phase_lr=0.10
branch_orders=1,2
branch_weights=0.5,0.5
competitive_lr=0.16
competitive_neg_k=8
competitive_score_scale=10.0
```

结果：

| method | warmup CE / acc | stream pre CE / acc | stream online CE / acc | stream post CE / acc | retention after CE / acc | state bytes |
|---|---:|---:|---:|---:|---:|---:|
| `phase_competitive_online` | 3.825 / 0.277 | 3.425 / 0.303 | 3.229 / 0.335 | 2.427 / 0.422 | 3.272 / 0.326 | 2,367,488 |
| `sparse_context_aux` | 4.408 / 0.252 | 3.835 / 0.323 | 3.721 / 0.334 | 1.297 / 0.571 | 1.229 / 0.553 | 175,568 |

解读：

- 纯 phase/WTA 主方法在线后相对在线前改善 `0.998` CE，acc 从 `0.303` 到 `0.422`。
- 在线中 prequential acc `0.335` 已略高于统计辅助的 `0.334`；在线后统计缓存仍明显更强，因为它直接保存上下文计数。
- retention 从 `3.187/0.343` 到 `3.272/0.326`，有轻微遗忘但没有崩塌。
- phase/WTA 是固定容量神经状态，当前状态约 `2.37MB`；统计辅助是按上下文增长的缓存，当前约 `176KB`。容量对比不能直接解释为最终可扩展性，需要后续做低精度和压缩。

## Local Sweep

在同一 medium 设置下，读出层学习率是主要增益来源：

| config | stream pre CE / acc | stream online CE / acc | stream post CE / acc | retention after CE / acc |
|---|---:|---:|---:|---:|
| `competitive_lr=0.02, score=8` | 3.728 / 0.279 | 3.598 / 0.311 | 3.190 / 0.351 | 3.530 / 0.309 |
| `competitive_lr=0.05, score=8` | 3.544 / 0.301 | 3.383 / 0.324 | 2.857 / 0.386 | 3.346 / 0.325 |
| `competitive_lr=0.08, score=8` | 3.471 / 0.305 | 3.286 / 0.329 | 2.699 / 0.395 | 3.280 / 0.330 |
| `competitive_lr=0.16, score=8` | 3.392 / 0.305 | 3.186 / 0.332 | 2.521 / 0.414 | 3.259 / 0.320 |
| `competitive_lr=0.16, score=10` | 3.425 / 0.303 | 3.229 / 0.335 | 2.427 / 0.422 | 3.272 / 0.326 |

`competitive_lr=0.16, score=10` 是当前默认候选：CE 最低、acc 最高，retention 代价可接受。

## Generation Audit

`phase_binding_online_stream_experiment.py` 现在保存：

- `generation_metrics.csv`: 每个 prompt 的 first-token match、prefix match、distinct-n、repeat-n 和 max-token fraction。
- `generation_summary.csv`: 按 method/stage/decode mode 聚合。
- `greedy_completions.txt`: decoded prompt / generated / reference 文本，作为人工审计材料。

注意：这些输出文件为了审计会保存 decoded prompt/reference/generated 文本；模型状态本身仍不保存原始训练文本。生成 CSV 因此区分 `model_stores_raw_text=false` 和 `artifact_contains_decoded_text=true`。

命令：

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_generation_medium \
  --train-chars 50000 --valid-chars 10000 \
  --max-vocab 256 --segment-tokens 256 \
  --completion-count 4 --prompt-tokens 16 --completion-tokens 48
```

Medium 生成摘要：

| method | stage | mode | first-token | prefix tokens | distinct-2 | repeat-2 | max-token frac |
|---|---|---|---:|---:|---:|---:|---:|
| `phase_competitive_online` | pre | greedy | 0.500 | 4.50 | 0.271 | 0.729 | 0.141 |
| `phase_competitive_online` | pre | controlled | 0.500 | 4.50 | 0.787 | 0.213 | 0.125 |
| `phase_competitive_online` | post | greedy | 0.750 | 4.75 | 0.229 | 0.771 | 0.151 |
| `phase_competitive_online` | post | controlled | 0.750 | 4.75 | 0.830 | 0.170 | 0.094 |
| `sparse_context_aux` | post | greedy | 0.750 | 5.25 | 0.766 | 0.234 | 0.083 |
| `sparse_context_aux` | post | controlled | 1.000 | 5.75 | 0.968 | 0.032 | 0.068 |

解读：

- Online phase/WTA improves first-token match from `0.500` to `0.750`, consistent with the CE/acc gain.
- Greedy phase/WTA still collapses into local loops: post `repeat-2=0.771`, with examples like repeated short fragments.
- Controlled decoding reduces phase/WTA post `repeat-2` to `0.170` and raises `distinct-2` to `0.830`, but generated text is still semantically thin and not close to GPT/API quality.
- The next method step should improve state dynamics and local inhibition, not rely on decoding controls as the main contribution.

## 判断

正信号：

- 这是第一次把纯 phase-binding token learner 跑通严格在线流：预测在更新前发生，不保存原始文本，不用 replay。
- 在线学习对 phase/WTA 主方法是强正结果：post CE `2.427`、acc `0.422`，超过此前静态 competitive WTA 的 CE `3.195` / acc `0.322`。
- 在线中 prequential acc 与统计辅助基本持平，说明局部竞争更新能快速吸收新流。
- Controlled generation shows the online signal is usable, but greedy generation exposes a severe repetition bottleneck.

边界：

- `sparse_context_aux` post CE `1.297` / acc `0.571` 更强，但这是 token 上下文统计缓存，只能作为分析辅助和上界，不能作为最终仿生模型框架。
- phase/WTA 仍缺少可变长语义状态和低精度/硬件友好压缩；当前生成审计已经显示循环和语义薄弱是主要瓶颈。
- 生成质量仍远低于最终目标：CE/acc 改善尚未自动转化为自然 completion，受控解码只能缓解重复。
- 当前状态矩阵预分配容量，下一步需要报告有效激活、低精度量化和可剪枝读出，而不是只看 bytes。

下一步：

1. 把 WTA 竞争信号反馈到相位分支状态，而不只是输出读出。
2. 做低精度/稀疏化审计：phase code 8-bit/ternary、读出 top-k rows、原型剪枝。
3. 引入可变长状态或短程 SSM/reservoir branch，解决 “She was ...” / “was a ...” 循环。
4. 做 seed/data-slice 复跑，确认 `competitive_lr=0.16, score=10` 稳定。
