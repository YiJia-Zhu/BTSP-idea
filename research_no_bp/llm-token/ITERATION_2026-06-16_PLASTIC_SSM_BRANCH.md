# 2026-06-16 Plastic Recurrent/SSM Branch

本轮沿纯 no-BP token learner 主线实现 plastic recurrent/SSM branch，目标是让 recurrent state 本身参与局部 WTA 学习，而不是只使用固定 leaky trace。

## 方法

代码：`phase_binding_online_stream_experiment.py`

新增类：

- `OnlinePlasticSSMCompetitivePhaseMemory`
- `OnlineTracePlasticSSMCompetitivePhaseMemory`

机制：

```text
h_t = normalize(tanh(decay * h_{t-1} + scale * W_rec h_{t-1} + input_code[token_t]))
feature = concat(phase_branch_features, h_t)

W_rec <- W_rec + lr * (target_modulated_post - W_rec pre) outer pre
W_readout[target] += lr * feature
W_readout[top_wrong] -= lr/k * feature
```

这是局部 target-modulated Hebbian/Oja 式 recurrent update，不使用 BP/BPTT、预训练模型、API 主干或原始文本 replay。`sparse_context_aux` 仍只是统计辅助对照。

## Smoke

Default plastic SSM smoke:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_plastic_ssm_smoke \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --warmup-token-limit 1200 --stream-token-limit 600 \
  --segment-tokens 128 \
  --trace-branch --trace-order 8 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --plastic-ssm-branch --ssm-order 8 --ssm-dim 64 \
  --ssm-decay 0.80 --ssm-recurrent-scale 0.40 \
  --ssm-weight 0.5 --ssm-lr 0.01 --ssm-target-mix 0.25 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --completion-count 2 --prompt-tokens 12 --completion-tokens 24
```

Smoke result:

| method | post CE | post acc | greedy repeat-2 |
|---|---:|---:|---:|
| `phase_trace_competitive_online` | 2.050 | 0.490 | 0.217 |
| `phase_trace_fatigue_competitive_online` | 2.030 | 0.492 | 0.152 |
| `phase_plastic_ssm_competitive_online` | 2.072 | 0.479 | 0.217 |
| `phase_plastic_ssm_fatigue_competitive_online` | 2.053 | 0.477 | 0.065 |
| `phase_trace_plastic_ssm_fatigue_competitive_online` | 2.041 | 0.490 | 0.152 |

Two stronger SSM updates were also tested:

- `lr=0.02, target_mix=0.50, recurrent_scale=0.20`
- `lr=0.05, target_mix=1.00, recurrent_scale=0.20`

Both worsened CE relative to the default smoke and did not improve the fixed trace baseline.

## Medium

Command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_plastic_ssm_medium \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --plastic-ssm-branch --ssm-order 16 --ssm-dim 64 \
  --ssm-decay 0.80 --ssm-recurrent-scale 0.40 \
  --ssm-weight 0.5 --ssm-lr 0.01 --ssm-target-mix 0.25 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --completion-count 4 --prompt-tokens 16 --completion-tokens 48
```

Medium result:

| method | online CE / acc | post CE / acc | greedy repeat-2 | state bytes |
|---|---:|---:|---:|---:|
| `phase_trace_competitive_online` | 3.157 / 0.335 | 2.389 / 0.427 | 0.681 | 2,498,560 |
| `phase_trace_fatigue_competitive_online` | 3.141 / 0.339 | 2.382 / 0.429 | 0.606 | 2,499,584 |
| `phase_trace_inhib_competitive_online` | 3.114 / 0.335 | 2.358 / 0.429 | 0.473 | 2,761,728 |
| `phase_trace_fatigue_inhib_competitive_online` | 3.113 / 0.341 | 2.363 / 0.432 | 0.436 | 2,762,752 |
| `phase_plastic_ssm_competitive_online` | 3.172 / 0.333 | 2.409 / 0.426 | 0.697 | 2,580,480 |
| `phase_plastic_ssm_fatigue_competitive_online` | 3.158 / 0.336 | 2.402 / 0.427 | 0.644 | 2,581,504 |
| `phase_plastic_ssm_inhib_competitive_online` | 3.129 / 0.333 | 2.377 / 0.425 | 0.590 | 2,843,648 |
| `sparse_context_aux` | 3.721 / 0.334 | 1.297 / 0.571 | 0.234 | 175,568 |

## 判断

正信号：

- The branch is functional and fully local/no-BP.
- It can reduce repetition in very small smoke settings when combined with fatigue.
- On medium, plastic SSM + inhibition is close to trace+fatigue CE but still below the current best.

边界：

- It does not beat fixed leaky trace on CE/acc.
- The trace+plastic SSM combination is also negative in smoke: it adds state but does not improve trace+fatigue.
- Stronger target-modulated recurrent updates hurt CE, suggesting the current Oja-style transition update is not the right credit assignment mechanism for token modeling.

Conclusion:

Current plastic SSM is a useful boundary result, not a new best method. The next recurrent attempt should use eligibility-gated transition writes or dendritic/apical error gating, rather than only increasing Hebbian/Oja recurrent strength.
