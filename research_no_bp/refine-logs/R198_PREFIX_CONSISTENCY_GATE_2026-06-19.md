# R198 QA19 Prefix-Consistency Gate

**日期**: 2026-06-19
**状态**: DONE-NEGATIVE-BOUNDARY
**问题**: R197 的 conflict-local rescue 在 teacher-forced slot1 上有用，但 greedy 前缀下破坏 full-answer exact。一个更简洁的 prefix-consistency / prefix-confidence gate 能否保留 CE/token 正信号，同时避免 greedy 前缀不稳定？

## 实现

在 `AnswerSlotReadoutMemory` 中加入默认关闭的 conflict rescue prefix gate：

- 新参数：
  - `--answer-slot-conflict-rescue-prefix-gate {none,observed_pred,margin,observed_pred_margin}`
  - `--answer-slot-conflict-rescue-prefix-margin`
- wrapper 记录上一 answer slot 的预测、score margin，以及随后 observe/update 的 token 是否等于该预测。
- `observed_pred`：只有上一 slot 的 observed token 等于模型当时预测时，slot1 conflict rescue 才启用。
- `margin`：只有上一 slot 的 score margin 达到阈值时启用。
- `observed_pred_margin`：同时要求 observed-prefix agreement 和 margin。
- gate 同时影响 rescue score 和 rescue prototype update；默认 `none` 完全复现 R197/R193 旧路径。

这个实现不新增记忆表，只增加少量局部状态与计数器，保持 no-BP 和不保存原始文本。

## 运行

所有 medium run 使用 QA19 `300/100/300`、seed0、R197 s0.25 配置：R193 coupling + conflict rescue scale `0.25`。

| Run | Prefix gate | Gate applied / checks | Conflict updates | Val exact | Val CE | Val token | Test exact | Test CE | Test token |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R186 soft | none | - | - | 0.1300 | 1.3200 | 0.3100 | 0.1267 | 1.2818 | 0.3300 |
| R193 coupling | none | - | - | 0.1300 | 1.3128 | 0.3000 | 0.1267 | 1.2714 | 0.3250 |
| R197 conflict s0.25 | none | - | `[0,181]` | 0.1100 | 1.3068 | 0.3050 | 0.1200 | 1.2664 | 0.3400 |
| R198 observed_pred | observed_pred | 1089 / 2000 | `[0,35]` | 0.1300 | 1.3124 | 0.3000 | 0.1167 | 1.2717 | 0.3200 |
| R198 observed_pred_margin0.05 | observed_pred_margin 0.05 | 889 / 2000 | `[0,29]` | 0.1300 | 1.3117 | 0.3000 | 0.1167 | 1.2708 | 0.3200 |
| R198 margin0.05 | margin 0.05 | 1630 / 2000 | `[0,152]` | 0.1200 | 1.3065 | 0.3250 | 0.1133 | 1.2688 | 0.3233 |
| R198 margin0.20 | margin 0.20 | 860 / 2000 | `[0,84]` | 0.1300 | 1.3107 | 0.2950 | 0.1200 | 1.2701 | 0.3333 |

## Flip 诊断

相对 R193 coupling：

| Candidate | Split | Full exact delta | Token acc delta | Helpful/Harmful exact |
|---|---|---:|---:|---:|
| observed_pred | test | -0.0100 | -0.0050 | 7 / 10 |
| observed_pred_margin0.05 | test | -0.0100 | -0.0050 | 6 / 9 |
| margin0.05 | test | -0.0133 | -0.0017 | 7 / 11 |
| margin0.20 | test | -0.0067 | +0.0083 | 3 / 5 |

最佳 R198 点 `margin0.20` 相对 R197 s0.25：

| Split | Full exact delta | Token acc delta | Helpful/Harmful exact |
|---|---:|---:|---:|
| validation | +0.0200 | -0.0100 | 3 / 1 |
| test | 0.0000 | -0.0067 | 5 / 5 |
| train_post | -0.0233 | -0.0050 | 11 / 18 |

## 解释

R198 否定了一个看似简洁的修复：只在推理/更新时加 prefix gate，不足以把 R197 的局部 rescue 变成 full-answer exact 正收益。

具体边界：

1. `observed_pred` 类 gate 太强。它把 conflict updates 从 R197 的 `[0,181]` 压到 `[0,35]` 或 `[0,29]`，held-out CE/token 信号基本丢失，test exact 仍下降到 `0.1167`。
2. `margin` 类 gate 保留更多 rescue，但仍无法超过 R193/R186 的 exact。`margin0.20` 是最稳点：test exact 保持 R197 的 `0.1200`，token acc `0.3333`，但 CE `1.2701` 明显弱于 R197 的 `1.2664`，也低于 R193/R186 的 exact `0.1267`。
3. 这说明问题不只是“什么时候启用 rescue”。R197 的核心失败更像是训练分布问题：rescue prototype 主要在 teacher-forced prefix 下写入，而 greedy 生成进入的是模型自身 prefix 分布。

## 结论

R198 不进入主方法默认配置。它的价值是缩小下一步方向：

- 不再继续堆推理时 prefix gate。
- 下一步应做 predicted-prefix eligibility training：训练时显式让 slot1 rescue 在模型自己预测的 slot0 prefix 下形成资格迹和局部修正。
- 保留 R186 soft edge-path + R193 coupling 作为简洁核心；R197/R198 作为诊断性边界与设计依据。

如果按“简洁、优雅、允许小幅误差”的标准，R198 也不如 R193：它更复杂，却没有更好的 exact，也没有稳定保留 R197 的 CE/token 优势。

## 下一步

R199 建议实现 predicted-prefix rescue training：

- 在训练 answer sequence 时，除 teacher-forced path 外，额外生成一个 local predicted-prefix context。
- 只对 slot1/后续 slot 的 rescue/coupling eligibility 做局部更新，不把错误预测当成标签。
- 比较三条路径：R193 coupling、R197 teacher-forced rescue、R199 predicted-prefix rescue。
- 主指标仍是 QA19 full-answer exact、full-token acc、CE；first-token accuracy 继续只作为辅助。
