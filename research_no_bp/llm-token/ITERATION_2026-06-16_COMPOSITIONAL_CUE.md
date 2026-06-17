# Iteration 2026-06-16: Compositional Cue And Phase-Binding Hebbian Prototype

## 目的

本轮回到纯 no-BP 主线，补 R007 compositional cue task。任务是：

```text
C_a, F, C_b, F, ..., Q -> T_((a + b) mod K)
```

训练时留出一部分 `(a, b)` cue pairs，只在 held-out pairs 上评估 target-position accuracy。这个 split 用来区分：

- 记住 seen pair 的查表/统计方法；
- 能利用结构 bias 对未见 cue pair 做组合泛化的仿生 no-BP 结构。

## 代码

- `../compositional_cue_experiment.py`

比较方法：

| method | 说明 |
|---|---|
| `pair_lookup_hebbian` | 纯 Hebbian pair prototype lookup；非组合查表对照 |
| `phase_binding_hebbian` | 新增候选：环形相位绑定 + target-gated Hebbian prototype readout |
| `learned_phase_binding_hebbian` | 局部 target-gated plasticity 学 cue-to-phase code，再做复数相位绑定 |
| `phase_binding_scrambled_control` | 打乱第二个 cue 的相位码，检验相位结构是否必要 |
| `reservoir_readout` | 固定随机 reservoir + ridge readout |
| `eprop_3factor` | fixed-feedback eligibility trace recurrent learner |
| `eprop_resampled_feedback` | resampled feedback ablation |
| `tuned_bptt` | BPTT 上界/对照，不作为最终方法 |

本轮不使用 API、不使用预训练模型、不使用 token 统计 probability memory。

## 命令

```bash
python -m py_compile compositional_cue_experiment.py

python compositional_cue_experiment.py \
  --out-dir output/compositional_cue_phase_smoke \
  --k-values 4 --seeds 0 \
  --epochs 40 --bptt-epochs 60 \
  --hidden-dim 24 --reservoir-hidden-dim 48

python compositional_cue_experiment.py \
  --out-dir output/compositional_cue_learned_phase_r007 \
  --k-values 4 8 --seeds 0 1 2
```

## 结果

| K | method | seen acc | held-out acc | held-out loss |
|---:|---|---:|---:|---:|
| 4 | `learned_phase_binding_hebbian` | 1.000 | 1.000 | 0.000 |
| 4 | `phase_binding_hebbian` | 1.000 | 1.000 | 0.000 |
| 4 | `pair_lookup_hebbian` | 1.000 | 0.417 | 1.386 |
| 4 | `reservoir_readout` | 1.000 | 0.000 | 2.197 |
| 4 | `eprop_3factor` | 0.972 | 0.000 | 15.461 |
| 4 | `tuned_bptt` | 1.000 | 0.000 | 16.575 |
| 8 | `learned_phase_binding_hebbian` | 1.000 | 1.000 | 0.001 |
| 8 | `phase_binding_hebbian` | 1.000 | 1.000 | 0.001 |
| 8 | `pair_lookup_hebbian` | 1.000 | 0.146 | 2.079 |
| 8 | `reservoir_readout` | 1.000 | 0.021 | 2.357 |
| 8 | `eprop_3factor` | 1.000 | 0.021 | 15.374 |
| 8 | `tuned_bptt` | 1.000 | 0.000 | 15.117 |

`phase_binding_scrambled_control` drops to held-out `0.500` for K=4 and `0.188` for K=8, so the result depends on the compositional phase code rather than generic prototype lookup.

## 判断

This is the first clean positive signal for a pure no-BP compositional mechanism in this branch:

- Ordinary e-prop RNN, BPTT RNN, reservoir readout, and pair lookup all fit seen pairs but fail held-out composition.
- The phase-binding Hebbian prototype gets held-out `1.000` for K=4 and K=8 across 3 seeds.
- The learned-phase variant also gets held-out `1.000` for K=4 and K=8 after learning cue-to-phase codes with local target-gated attraction and complex phase multiplication.
- The scrambled phase-code control is much weaker, which supports the claim that the structured phase representation is doing real work.

Important boundary:

- This is not yet a general language learner. It solves a controlled algebraic cue task because the representation has a ring-composition inductive bias.
- The next step is to embed the learned phase-binding module inside a recurrent/SSM/dendritic no-BP token learner. It should not be replaced by token-frequency statistics or API-generated text.

## Next Step

Turn learned phase-binding into a harder neural module:

1. add noise, variable delay, distractor cues, and larger K;
2. make target-gated phase attraction less supervised, e.g. neuromodulated by delayed reward or error cells;
3. test whether the same mechanism can support TinyStories-style latent state shifts without using n-gram backoff as the method.
