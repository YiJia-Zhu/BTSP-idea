# R197 QA19 Conflict-Local Rescue

**日期**: 2026-06-19
**状态**: DONE-CE-POSITIVE-EXACT-NEGATIVE-BOUNDARY
**问题**: R196 证明 evidence-protected cleanup 可以避免 R195 的目标自抑制，但几乎不改变 top-1。能否改成只在当前 winner/challenger 冲突局部触发的有序候选救援，把 R193 的 CE 增益转成 QA19 full-answer exact 增益？

## 实现

在 `AnswerSlotReadoutMemory` 中加入默认关闭的 ordered-pair conflict rescue：

- 新参数：
  - `--answer-slot-conflict-rescue-slots`
  - `--answer-slot-conflict-rescue-lr`
  - `--answer-slot-conflict-rescue-score-scale`
  - `--answer-slot-conflict-rescue-top-k`
  - `--answer-slot-conflict-rescue-min-slot`
- 局部特征：`coupling_feature * token_code(winner) * token_code(candidate)`。
- 更新规则：如果 slot 预测失败，把 `(wrong winner, target challenger)` pair feature 存到目标 token 的 rescue prototype bank。
- 推理规则：先用 pre-conflict score 找当前 winner；候选集合来自 pre-score top-k 与正支持 top-k 的并集；若 pair prototype 匹配，则给 challenger 加 rescue score。
- Component logging 新增 `conflict_rescue_delta` 与 `after_conflict` 字段。

这仍然是局部/no-BP 更新：没有反向传播，没有预训练主干，也不保存原始文本。

## 原始结果表

所有 medium run 使用 QA19 `300/100/300`、seed0、R193 coupling 配置。R197 在此基础上打开 conflict rescue。

| Run | Conflict scale | Val exact | Val CE | Val token acc | Test exact | Test CE | Test token acc | Active slots | Updates | Score applied |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R186 soft | 0.00 | 0.1300 | 1.3200 | 0.3100 | 0.1267 | 1.2818 | 0.3300 | 0 | - | - |
| R193 coupling | 0.00 | 0.1300 | 1.3128 | 0.3000 | 0.1267 | 1.2714 | 0.3250 | 0 | - | - |
| R197 conflict s0.10 | 0.10 | 0.1100 | 1.3083 | 0.3050 | 0.1233 | 1.2684 | 0.3333 | 32 | `[0,176]` | 5956 |
| R197 conflict s0.25 | 0.25 | 0.1100 | 1.3068 | 0.3050 | 0.1200 | 1.2664 | 0.3400 | 32 | `[0,181]` | 5958 |
| R197 conflict s0.50 | 0.50 | 0.0900 | 1.3044 | 0.3150 | 0.1067 | 1.2659 | 0.3367 | 32 | `[0,197]` | 5966 |

## Flip 结果 vs R193

| Run | Split | Full exact delta | Token acc delta | Helpful/Harmful exact |
|---|---|---:|---:|---:|
| s0.10 | validation | -0.0200 | +0.0050 | 1 / 3 |
| s0.10 | test | -0.0033 | +0.0083 | 5 / 6 |
| s0.25 | validation | -0.0200 | +0.0050 | 2 / 4 |
| s0.25 | test | -0.0067 | +0.0150 | 8 / 10 |
| s0.50 | validation | -0.0400 | +0.0150 | 4 / 8 |
| s0.50 | test | -0.0200 | +0.0117 | 16 / 22 |

## Component 诊断：s0.25

R197 s0.25 相对 R193 的主要信号来自 slot1。它在 teacher-forced 条件下确实减少了高置信 wrong winner，但在 greedy 条件下破坏了自回归一致性。

| Split | Decode phase | Slot | R193 acc | R197 acc | Acc delta | Target-vs-best delta | High-margin wrong >=0.20 / >=0.50 |
|---|---|---:|---:|---:|---:|---:|---:|
| test | teacher_forced | 1 | 0.3533 | 0.3833 | +0.0300 | +0.0400 | `114/40 -> 68/15` |
| test | greedy | 1 | 0.3133 | 0.2800 | -0.0333 | +0.0365 | `123/39 -> 72/14` |
| validation | teacher_forced | 1 | 0.3100 | 0.3200 | +0.0100 | +0.0363 | `29/9 -> 23/4` |
| validation | greedy | 1 | 0.3000 | 0.2600 | -0.0400 | +0.0335 | `33/11 -> 26/5` |

关键细节：

- Teacher-forced test slot1 helpful/harmful token flips 为 `44/35`，slot1 accuracy `+0.0300`。
- Greedy test slot1 helpful/harmful token flips 为 `20/30`，slot1 accuracy `-0.0333`。
- High-margin wrong rows 显著减少，但 sequence exact 仍下降，说明局部 rescue 在正确前缀下有用，在模型自己生成的错误前缀下不稳定。

## 结论

R197 是机制上有信号、任务指标上未成功的边界结果：

1. **正向部分**：conflict-local rescue 比 R195/R196 更接近所需机制。它能降低 held-out CE，提升 held-out token accuracy，并在 teacher-forced slot1 上减少高置信错误。
2. **负向部分**：它没有提升 full-answer exact；scale 越强，exact 下降越明显。s0.25 的 test CE/token acc 优于 R193，但 full exact 从 `0.1267` 降到 `0.1200`。
3. **失败原因**：局部救援学到了正确前缀条件下的 near-miss 修正，但评估是 greedy two-token answer；第一 token 或前缀一旦偏离，slot1 的 rescue feature 分布改变，导致有用的 pairwise rescue 变成不稳定干预。

因此，下一步不应该继续加大 rescue scale。问题已经不是“没有 rescue 信号”，而是“rescue 没有 prefix-consistency / predicted-prefix 适配”。

## 下一步

R198 应该测试 prefix-consistency gated rescue：

- 在 teacher-forced 与 greedy prefix 的 score 分布一致时才启用 rescue。
- 或者在训练时把 rescue eligibility 也暴露给 predicted-prefix context，而不是只从 teacher-forced target loss 写入。
- 继续保留 full-answer exact、token acc、CE、component margin 四类指标；QA19 不能回退到 first-token accuracy。

目标是保留 R197 的 CE/token 正信号，同时避免它在 greedy answer 生成中破坏 full-answer exact。
