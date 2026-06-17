# 2026-06-17 Eligibility-Gated Recurrent/SSM Branch

本轮沿纯 no-BP token learner 主线实现 eligibility-gated recurrent/SSM transition writes。目标是检验：比 R049 的 Hebbian/Oja transition 更接近三因子 eligibility 的局部写入，是否能让 recurrent state 成为有效的在线 token 学习状态。

## 方法

代码：`phase_binding_online_stream_experiment.py`

新增类：

- `EligibilitySSMTransitionMixin`
- `OnlineEligibilitySSMCompetitivePhaseMemory`
- `OnlineTraceEligibilitySSMCompetitivePhaseMemory`

新增 CLI：

- `--eligibility-ssm-branch`
- `--ssm-eligibility-decay`
- `--ssm-eligibility-clip`
- `--method-filter`

机制：

```text
h_t = normalize(tanh(decay * h_{t-1} + scale * W_rec h_{t-1} + input_code[token_t]))
e_t = decay_e * e_{t-1} + outer(h_t, h_{t-1})
W_rec <- W_rec + lr * outer(target_modulated_h_t - W_rec h_{t-1}, h_{t-1}) * abs(e_t)
feature = concat(phase_branch_features, optional_trace, h_t)
W_readout[target] += lr_readout * feature
W_readout[top_wrong] -= lr_readout/k * feature
```

更新仍是局部三因子式 no-BP：只使用当前 context state、target token 和局部 pre/post eligibility；不使用 BP/BPTT、预训练模型、API 主干、原始文本 replay。`sparse_context_aux` 继续只是统计辅助对照，不是最终方法候选。

## Smoke

Initial eligibility SSM smoke:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_elig_ssm_smoke \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --warmup-token-limit 1200 --stream-token-limit 600 \
  --segment-tokens 128 \
  --trace-branch --trace-order 8 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --eligibility-ssm-branch --ssm-order 8 --ssm-dim 64 \
  --ssm-decay 0.80 --ssm-recurrent-scale 0.40 \
  --ssm-weight 0.5 --ssm-lr 0.01 --ssm-target-mix 0.25 \
  --ssm-eligibility-decay 0.90 --ssm-eligibility-clip 2.0 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --completion-count 2 --prompt-tokens 12 --completion-tokens 24
```

Initial smoke result:

| method | post CE | post acc | greedy repeat-2 |
|---|---:|---:|---:|
| `phase_trace_competitive_online` | 2.050 | 0.490 | 0.217 |
| `phase_trace_fatigue_competitive_online` | 2.030 | 0.492 | 0.152 |
| `phase_trace_fatigue_inhib_competitive_online` | 2.020 | 0.483 | 0.152 |
| `phase_trace_elig_ssm_competitive_online` | 2.052 | 0.483 | 0.174 |
| `phase_trace_elig_ssm_fatigue_competitive_online` | 2.032 | 0.492 | 0.065 |
| `phase_trace_elig_ssm_inhib_competitive_online` | 2.038 | 0.490 | 0.174 |

Then a narrow smoke sweep tested:

- low write strength: `ssm_weight=0.2`, `ssm_lr=0.003`, `target_mix=0.10`, `elig_decay=0.95`, `elig_clip=1.0`
- fixed reservoir control: same but `ssm_lr=0.0`
- clipped write: `ssm_lr=0.01`, `target_mix=0.15`, `elig_decay=0.97`, `elig_clip=0.5`

Best smoke was effectively the fixed/very-low-write setting:

| method | fixed reservoir post CE / acc | low eligibility post CE / acc |
|---|---:|---:|
| `phase_trace_fatigue_competitive_online` | 2.030 / 0.492 | 2.030 / 0.492 |
| `phase_trace_fatigue_inhib_competitive_online` | 2.020 / 0.483 | 2.020 / 0.483 |
| `phase_trace_elig_ssm_fatigue_competitive_online` | 2.017 / 0.494 | 2.018 / 0.494 |
| `phase_trace_elig_ssm_inhib_competitive_online` | 2.022 / 0.494 | 2.023 / 0.494 |

This suggests the small gain comes from adding a fixed recurrent reservoir feature, not from eligibility-gated transition plasticity.

## Medium

Fixed reservoir control:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_elig_ssm_medium_fixed \
  --method-filter phase_trace_competitive_online phase_trace_fatigue phase_trace_inhib phase_trace_elig_ssm \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --eligibility-ssm-branch --ssm-order 16 --ssm-dim 64 \
  --ssm-decay 0.80 --ssm-recurrent-scale 0.40 \
  --ssm-weight 0.2 --ssm-lr 0.0 --ssm-target-mix 0.10 \
  --ssm-eligibility-decay 0.95 --ssm-eligibility-clip 1.0 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --completion-count 4 --prompt-tokens 16 --completion-tokens 48
```

Seed 0 result:

| method | online CE / acc | post CE / acc | greedy repeat-2 | state bytes |
|---|---:|---:|---:|---:|
| `phase_trace_competitive_online` | 3.157 / 0.335 | 2.389 / 0.427 | 0.681 | 2,498,560 |
| `phase_trace_fatigue_competitive_online` | 3.141 / 0.339 | 2.382 / 0.429 | 0.606 | 2,499,584 |
| `phase_trace_inhib_competitive_online` | 3.114 / 0.335 | 2.358 / 0.429 | 0.473 | 2,761,728 |
| `phase_trace_fatigue_inhib_competitive_online` | 3.113 / 0.341 | 2.363 / 0.432 | 0.436 | 2,762,752 |
| `phase_trace_elig_ssm_competitive_online` | 3.153 / 0.336 | 2.386 / 0.430 | 0.697 | 2,727,936 |
| `phase_trace_elig_ssm_fatigue_competitive_online` | 3.137 / 0.339 | 2.379 / 0.431 | 0.617 | 2,728,960 |
| `phase_trace_elig_ssm_inhib_competitive_online` | 3.109 / 0.335 | 2.355 / 0.431 | 0.527 | 2,991,104 |

Weak eligibility write (`ssm_lr=0.003`) was slightly worse than fixed reservoir on the same seed:

| method | online CE / acc | post CE / acc | greedy repeat-2 |
|---|---:|---:|---:|
| `phase_trace_elig_ssm_competitive_online` | 3.152 / 0.335 | 2.388 / 0.429 | 0.681 |
| `phase_trace_elig_ssm_fatigue_competitive_online` | 3.136 / 0.340 | 2.381 / 0.430 | 0.617 |
| `phase_trace_elig_ssm_inhib_competitive_online` | 3.109 / 0.334 | 2.357 / 0.431 | 0.489 |

Seed 1 fixed-reservoir repeat:

| method | post CE / acc |
|---|---:|
| `phase_trace_competitive_online` | 2.450 / 0.420 |
| `phase_trace_fatigue_competitive_online` | 2.440 / 0.426 |
| `phase_trace_inhib_competitive_online` | 2.410 / 0.431 |
| `phase_trace_fatigue_inhib_competitive_online` | 2.414 / 0.434 |
| `phase_trace_elig_ssm_competitive_online` | 2.450 / 0.421 |
| `phase_trace_elig_ssm_fatigue_competitive_online` | 2.440 / 0.421 |
| `phase_trace_elig_ssm_inhib_competitive_online` | 2.412 / 0.430 |

## 判断

正信号：

- A small fixed reservoir feature can marginally improve seed 0 CE when paired with adaptive inhibition: `2.358 -> 2.355`.
- Fixed reservoir also improves trace/fatigue CE on seed 0: `2.382 -> 2.379`.
- The mechanism remains pure no-BP and stores no raw text.

边界：

- The seed 0 CE gain is tiny and not reproduced on seed 1: `phase_trace_elig_ssm_inhib_competitive_online` is `2.412`, while `phase_trace_inhib_competitive_online` is `2.410`.
- Weak eligibility transition plasticity is not better than `ssm_lr=0.0`; the current gain is best explained as a fixed random reservoir feature plus WTA readout, not as successful recurrent credit assignment.
- It costs extra state: seed 0 trace+elig+inhib uses `2.99MB` vs trace+inhib `2.76MB`.
- Greedy generation remains repetitive; fixed reservoir+inhibition repeat-2 `0.527` is worse than trace+fatigue+inhibition `0.436` on seed 0.

Conclusion:

Eligibility-gated transition writes are a negative/neutral boundary result in the current form. A small fixed reservoir branch is worth keeping as an optional feature, but it is not yet a new best method and should not be treated as evidence that recurrent transition learning has been solved.

Next step should shift from transition-matrix plasticity to dendritic/apical local prediction-error gating: use a basal phase/trace/reservoir feature, an apical target/error trace, and local branch-wise gating of readout updates. The target is to improve winner selection without adding a global statistical cache or relying on BP-pretrained backbones.
