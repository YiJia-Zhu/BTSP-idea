# 2026-06-16 Target-Only Phase Binding

本轮继续纯 no-BP 主线，修正上一版 compositional cue 的一个潜在手工化问题：`learned_phase_binding_hebbian` 虽然不使用 BP，但训练时仍显式使用了 `C_a -> phase a` 这类 cue 相位教师信号。

新的 `target_only_phase_binding_hebbian` 只使用最终 target class 作为三因子调制信号：

```text
C_a, F, C_b, ..., Q -> T_((a+b) mod K)
```

## 方法

每个 cue 维护一个局部复数相位码。若当前样本的目标相位为 `target_phase`，局部绑定关系要求：

```text
code_a * code_b ~= target_phase
```

因此更新只需要局部共轭绑定：

```text
code_a <- target_phase * conj(code_b)
code_b <- conj(code_a) * target_phase
```

读出仍是 target-gated Hebbian prototype readout。整个方法不使用 BP、BPTT、预训练模型、API 或 token 统计表。

## 命令

```bash
python compositional_cue_experiment.py \
  --out-dir output/compositional_cue_targetonly_r008 \
  --k-values 4 8 12 \
  --seeds 0 1 2 \
  --methods pair_lookup_hebbian phase_binding_hebbian learned_phase_binding_hebbian \
    target_only_phase_binding_hebbian phase_binding_scrambled_control \
  --target-phase-epochs 500
```

## 结果

| K | method | held-out acc | held-out loss |
|---:|---|---:|---:|
| 4 | `target_only_phase_binding_hebbian` | 1.000 | 0.173 |
| 4 | `pair_lookup_hebbian` | 0.417 | 1.386 |
| 4 | `phase_binding_scrambled_control` | 0.500 | 3.672 |
| 8 | `target_only_phase_binding_hebbian` | 1.000 | 0.022 |
| 8 | `pair_lookup_hebbian` | 0.146 | 2.079 |
| 8 | `phase_binding_scrambled_control` | 0.188 | 4.948 |
| 12 | `target_only_phase_binding_hebbian` | 1.000 | 0.156 |
| 12 | `pair_lookup_hebbian` | 0.093 | 2.485 |
| 12 | `phase_binding_scrambled_control` | 0.213 | 3.732 |

## 判断

这是比 R007 更强的纯 no-BP 机制证据：

- 不再需要直接告诉模型 `C_a` 对应哪个环相位。
- held-out cue-pair 泛化在 K=4/8/12 都达到 1.000。
- pair lookup 随 K 增大快速退化，说明不是训练 pair 查表。
- scrambled second-cue control 明显失败，说明结果依赖一致的相位绑定结构。

边界仍然明确：这是受控组合任务，不是语言模型。下一步应把 target-only phase factorization 接入纯 no-BP token learner 的 recurrent/SSM/dendritic 状态，而不是回到预训练 LLM 或 n-gram 统计方法。
