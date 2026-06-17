# 2026-06-16 Phase-Binding Token Learner

本轮把 target-only phase binding 从受控组合 cue 迁到 TinyStories next-token 原型，目标是开始补 `pure no-BP token learner 主线`，不依赖 BP 预训练模型/API，也不把 n-gram 统计表作为最终方法。

## 方法

新脚本：`phase_binding_token_experiment.py`

主方法 `phase_binding_token`：

```text
code_left[token_{t-2}] * code_right[token_{t-1}] -> target_token_t
```

训练时只使用 next-token target class：

```text
code_left  <- target_anchor * conj(code_right)
code_right <- conj(code_left) * target_anchor
```

读出是 target-gated Hebbian prototype readout，加一个本地输出神经元 bias。`phase_binding_token_no_bias` 单独报告，用于拆出 bias 的贡献。

辅助基线：

- `unigram_aux`: 只看输出频率。
- `sparse_context_aux`: 二阶上下文计数，只作为统计辅助基线，不作为最终方法。

## Smoke

命令：

```bash
python phase_binding_token_experiment.py \
  --out-dir output/phase_binding_token_smoke \
  --train-chars 10000 --valid-chars 3000 \
  --max-vocab 128 --eval-token-limit 1000
```

结果：

| method | CE | acc |
|---|---:|---:|
| `phase_binding_token_no_bias` | 4.123 | 0.199 |
| `phase_binding_token` | 3.583 | 0.254 |
| `unigram_aux` | 4.232 | 0.084 |
| `sparse_context_aux` | 4.119 | 0.254 |

## Medium Tuned

命令：

```bash
python phase_binding_token_experiment.py \
  --out-dir output/phase_binding_token_medium_tuned \
  --train-chars 50000 --valid-chars 10000 \
  --max-vocab 256 --eval-token-limit 5000 \
  --phase-dim 128 --phase-lr 0.10 \
  --phase-logit-scale 10.0 --phase-bias-weight 1.0
```

结果：

| method | CE | acc |
|---|---:|---:|
| `phase_binding_token_no_bias` | 4.204 | 0.229 |
| `phase_binding_token` | 3.551 | 0.284 |
| `unigram_aux` | 4.652 | 0.080 |
| `sparse_context_aux` | 3.833 | 0.323 |

## EMA Readout Variant

命令：

```bash
python phase_binding_token_experiment.py \
  --out-dir output/phase_binding_token_ema_medium \
  --train-chars 50000 --valid-chars 10000 \
  --max-vocab 256 --eval-token-limit 5000 \
  --phase-dim 128 --phase-lr 0.10 \
  --phase-logit-scale 10.0 --phase-bias-weight 1.0 \
  --prototype-lr 0.10
```

结果：

| method | CE | acc |
|---|---:|---:|
| `phase_binding_token_no_bias` | 3.985 | 0.260 |
| `phase_binding_token` | 3.614 | 0.298 |
| `unigram_aux` | 4.652 | 0.080 |
| `sparse_context_aux` | 3.833 | 0.323 |

EMA prototype readout 改善 top-1，但牺牲一部分 CE。临时 anti-Hebbian winner-take-all sweep 没有稳定超过 EMA-only，因此暂不作为默认主方法。

## Context Order Sweep

`phase_binding_token_experiment.py` 现在支持 `--context-order`。这不是统计 n-gram 主方法，而是把多个最近 token 的局部相位码做复数乘法绑定：

```text
code_0[token_{t-n}] * ... * code_{n-1}[token_{t-1}] -> target_token_t
```

汇总文件：`output/phase_binding_token_order_sweep/summary.csv`

| order | phase CE | phase acc | same-order sparse CE | same-order sparse acc |
|---:|---:|---:|---:|---:|
| 1 | 3.603 | 0.261 | 3.310 | 0.292 |
| 2 | 3.551 | 0.284 | 3.833 | 0.323 |
| 3 tuned | 4.199 | 0.180 | 4.080 | 0.274 |
| 4 | 10.462 | 0.077 | 4.132 | 0.227 |

结论：order=2 是当前相位绑定 token learner 的最好点。直接把更多 token 做同一个相位乘积会退化；order=3 调小学习率后仍明显弱于 order=2。下一步不应继续堆 context order，而应改成分支化 dendritic/SSM 状态或门控组合。

## Branched Phase State

`phase_binding_token_experiment.py` 现在支持 `--branch-orders` 和 `--branch-weights`。分支模型不是把长上下文压成单一相位乘积，而是保留多个局部分支，再固定加权求和 logits：

```text
branch_1: token_{t-1} -> target
branch_2: token_{t-2}, token_{t-1} -> target
output = w1 * branch_1 + w2 * branch_2 + output_bias
```

所有分支仍是 target-only local phase binding，没有 BP/BPTT/API/预训练主干。

汇总文件：`output/phase_binding_token_branch_sweep/summary.csv`

| branch weights `(order1, order2)` | CE | acc |
|---|---:|---:|
| `(0.50, 0.50)` | 3.282 | 0.295 |
| `(0.40, 0.60)` | 3.291 | 0.299 |
| `(0.25, 0.75)` | 3.350 | 0.304 |
| single order-2 phase | 3.551 | 0.284 |
| `sparse_context_aux` | 3.833 | 0.323 |

结论：分支化相位状态是当前最强纯 phase token learner。它把 CE 从 `3.551` 降到 `3.282`，也超过旧 continuation 统计基线 CE `3.325`，但 top-1 仍未超过 sparse context。

## Readout Calibration

汇总文件：`output/phase_binding_token_branch_calibration/summary.csv`

| variant | CE | acc |
|---|---:|---:|
| base branch `(0.50, 0.50)`, bias 1.0, temp 1.0 | 3.282 | 0.295 |
| balanced calibrated `(0.40, 0.60)`, bias 0.8, temp 1.0 | 3.285 | 0.306 |
| max-acc calibrated `(0.10, 0.90)`, bias 0.6, temp 0.7 | 3.599 | 0.312 |
| `sparse_context_aux` | 3.833 | 0.323 |

结论：读出校准能把 acc 从 `0.295` 提到 `0.312`，但仍低于 sparse context `0.323`。这说明瓶颈不是单纯温度/先验问题，需要训练期或状态期的局部竞争/抑制机制。

## Competitive WTA Readout

`phase_binding_token_experiment.py` 现在支持 `--competitive-readout`。该读出不是 BP：它固定已学习的相位分支特征，用局部 winner-take-all/perceptron 规则训练输出原型：

```text
target row <- target row + lr * feature
top wrong winner rows <- wrong rows - lr/k * feature
```

汇总文件：`output/phase_binding_token_competitive_sweep/summary.csv`

| variant | CE | acc |
|---|---:|---:|
| branch baseline | 3.282 | 0.295 |
| WTA neg_k=1, lr=0.02 | 3.276 | 0.308 |
| WTA neg_k=8, lr=0.02 | 3.230 | 0.322 |
| WTA neg_k=8, lr=0.05 | 3.195 | 0.320 |
| `sparse_context_aux` | 3.833 | 0.323 |

结论：训练期局部竞争显著改善 winner selection，把 acc 从 `0.295` 提到 `0.322`，几乎追平 sparse context `0.323`；同时 CE 仍显著优于 sparse。当前差距已经从结构性大差距变成极小 top-1 差距。

## 判断

正信号：

- `phase_binding_token` 在 medium 上 CE `3.551`，低于辅助 `sparse_context_aux` 的 `3.833`。
- EMA readout 把 acc 提到 `0.298`，但仍低于 `sparse_context_aux` 的 `0.323`。
- Branched phase state 进一步把 CE 降到 `3.282`，这是当前最强纯 no-BP token learner CE 结果。
- Readout calibration 把 acc 提到 `0.312`，接近但仍未超过 sparse context。
- Competitive WTA readout 把 acc 提到 `0.322`，几乎追平 sparse context `0.323`，并把 CE 进一步降到 `3.195`。
- `phase_binding_token_no_bias` 明显优于 `unigram_aux`，说明相位绑定状态本体有上下文信号，而不是全靠输出频率。
- 训练和读出全程无 BP/BPTT、无 API、无预训练主干。

短板：

- Top-1 acc 仍低于 sparse context：`0.284` vs `0.323`。
- 当前模型只用二阶相位绑定，缺少可变长上下文、抑制性 winner-take-all、树突分支和 SSM/recurrent 状态。
- Naive order>2 相位乘积是负结果，不应作为下一步主路线。
- 分支化改善 CE 明显，但 acc 最高仍只有 `0.304`，说明 winner selection/局部竞争仍是瓶颈。
- 仅调分支权重、bias 和 temperature 不能解决 winner selection。
- WTA 读出仍略低于 sparse top-1，下一步需要把竞争信号反馈到分支状态或引入更稳定的抑制性微回路。
- 输出 bias 有明显贡献，后续必须报告 no-bias/bias 拆分，避免把频率偏置误判为结构泛化。

下一步：把 phase-binding feature 接入局部竞争读出或 dendritic/SSM 状态，目标是同时提高 CE 和 top-1，而不是回退到 token 统计表或 BP 预训练模型。
