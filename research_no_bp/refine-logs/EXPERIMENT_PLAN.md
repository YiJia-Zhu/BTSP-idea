# Experiment Plan

**Problem**: 能否用类脑 no-BP 规则学习时序关系，并作为 tiny LLM / recurrent memory adapter 的训练机制？
**Method Thesis**: 固定反馈调制的三因子 eligibility trace 可以在 next-token 任务中训练隐藏状态，不需要 BPTT。
**Date**: 2026-06-03

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: no-BP 三因子规则能学习延迟时序记忆 | 直接对应“看到 A 想到 B” | 在 bigram/reservoir 失败的低维长延迟任务上，eprop_3factor target_acc 接近 BPTT | B1, B2 |
| C2: 该规则有走向 LLM/现有模型兼容的路线 | 用户关心能否实现 LLM 或兼容已有训练 | next-token objective 下可训练 recurrent/SSM adapter，且不需要训练完整 backbone | B3, B4 |

## Paper Storyline

Main paper must prove:

- 静态分类不是核心，时序 next-token 才是核心任务
- 固定反馈和 eligibility trace 缺一不可
- 方法可以作为 recurrent memory adapter，而不是一开始声称替代完整 Transformer 预训练

Appendix can support:

- MNIST 静态结果
- 零阶优化和 STDP 的负对照
- 更多 seed / delay sweep

Experiments intentionally cut:

- 直接训练大 LLM
- 大规模 SNN
- 完整 Transformer no-BP 预训练

## Experiment Blocks

### Block 1: Delayed Cue Next-Token

- Claim tested: C1
- Why this block exists: 最小化验证隐藏状态能否跨时间保存 cue
- Dataset / split / task: synthetic `C_k, F...F, T_k, SEP`
- Compared systems: bigram, reservoir, eprop_3factor, eprop_resampled, BPTT
- Metrics: target-position accuracy, overall accuracy, loss
- Setup details: hidden_dim=12, delay=12, seeds=3
- Success criterion: eprop_3factor target_acc >= 0.95, reservoir/resampled target_acc <= 0.1
- Failure interpretation: 如果 eprop 失败，当前资格迹或反馈规则不足以训练时序记忆
- Table / figure target: Main Table 1
- Priority: MUST-RUN

### Block 2: Delay Sweep

- Claim tested: C1
- Why this block exists: 排除只记固定位置的可能
- Dataset / split / task: train/test delay in {4, 8, 12, 16, 20}
- Compared systems: reservoir, eprop_3factor, BPTT
- Metrics: target-position accuracy vs delay
- Setup details: hidden_dim in {12, 24}, seeds=3
- Success criterion: eprop 在更长 delay 下比 reservoir 更稳
- Failure interpretation: 如果只在固定 delay 成功，说明泛化不足
- Table / figure target: Main Figure 2
- Priority: MUST-RUN

### Block 3: Compositional Cue Task

- Claim tested: C2
- Why this block exists: LLM 不只是记忆单 cue，还要组合上下文
- Dataset / split / task: `C_a, F, C_b, F...F, T_(a+b mod 4)`
- Compared systems: reservoir, eprop_3factor, BPTT
- Metrics: target accuracy, unseen-pair split accuracy
- Setup details: train 部分 cue pairs，test held-out pairs
- Success criterion: eprop 超过 reservoir，并接近 BPTT
- Failure interpretation: 如果 held-out pair 失败，说明只是查表，不是组合
- Table / figure target: Main Table 2
- Priority: MUST-RUN

### Block 4: Frozen Backbone + no-BP Memory Adapter

- Claim tested: C2
- Why this block exists: 连接到已有模型兼容路线
- Dataset / split / task: character-level next-token 或 synthetic grammar
- Compared systems: frozen embedding + linear readout, frozen embedding + reservoir adapter, frozen embedding + eprop adapter, BPTT adapter
- Metrics: next-token loss, target accuracy, online adaptation speed
- Setup details: 冻结 token embedding/backbone，只训练 recurrent adapter 和 readout
- Success criterion: eprop adapter 明显强于 frozen readout/reservoir
- Failure interpretation: 如果没有增益，说明 no-BP 规则不适合做兼容 adapter
- Table / figure target: Main Figure 3
- Priority: MUST-RUN

### Block 5: Replay / Association Extension

- Claim tested: C1/C2
- Why this block exists: 连接到人脑 memory linking 和 replay
- Dataset / split / task: A->B->C 链式序列，测试 A 是否唤起 C
- Compared systems: eprop no replay, eprop + replay, shuffled replay
- Metrics: one-step accuracy, two-step association accuracy, conflict cue behavior
- Setup details: replay 比例和 trace decay sweep
- Success criterion: 正确 replay 提升远距离联想，shuffled replay 不提升
- Failure interpretation: 如果 replay 无效，说明当前机制只能学一阶预测
- Table / figure target: Appendix or Main Figure 4
- Priority: NICE-TO-HAVE

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | sanity | quick delayed cue | bigram target=0, BPTT target=1 | CPU minutes | 任务太简单 |
| M1 | main result | hard delayed cue | eprop target>=0.95, reservoir target<=0.1 | CPU < 30 min | 超参敏感 |
| M2 | novelty isolation | resampled feedback / trace decay ablation | fixed feedback wins | CPU < 1 h | 结果只靠读出层 |
| M3 | generalization | delay sweep | eprop better than reservoir across delay | CPU 1-2 h | 只学固定时长 |
| M4 | LLM compatibility | frozen adapter | eprop adapter improves next-token | CPU/GPU small | adapter 太弱 |
| M5 | brain association | replay chain | replay improves long association | CPU < 1 h | replay 设计不合理 |

## Compute and Data Budget

- Total estimated GPU-hours: 0 for current synthetic tasks
- Data preparation needs: synthetic generator, optional small text corpus
- Human evaluation needs: none
- Biggest bottleneck: Python loops and hyperparameter sweep

## Risks and Mitigations

- Risk: reservoir 在简单任务上太强
- Mitigation: 使用低维长延迟、组合任务、held-out pairs

- Risk: BPTT baseline 未调好
- Mitigation: 单独调 BPTT lr/epochs，报告 tuned baseline

- Risk: eprop 只是记固定位置
- Mitigation: delay sweep 和 variable-delay train/test

- Risk: 和 LLM 连接太弱
- Mitigation: 明确定位为 next-token recurrent memory adapter，而不是完整 LLM 替代

## Final Checklist

- [x] Main temporal sanity covered
- [x] Novelty isolated by resampled feedback
- [x] BPTT tuned baseline included
- [ ] Delay sweep
- [ ] Compositional cue task
- [ ] Frozen backbone + no-BP adapter
- [ ] Replay association extension

