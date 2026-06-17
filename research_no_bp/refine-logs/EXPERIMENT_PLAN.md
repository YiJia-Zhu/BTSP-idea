# Experiment Plan

**Problem**: 构建替代 BP 的高效仿生物神经学习算法，最终支持无原始数据保存的在线学习，并能面向纯仿生结构、在线学习硬件或 neuromorphic 实现。
**Hard Constraint**: 主方法必须是**纯 no-BP 神经网络类方法**：从随机/局部初始化开始训练，使用局部可塑性、资格迹、固定反馈、树突误差信号、抑制性回路、STDP/BTSP、e-prop、replay 或 reservoir/SSM 动态；不得依赖已经由 BP 训练好的开源 LLM、冻结大模型或 API 主干作为核心性能来源。预训练 LLM/API 只能作为外部上界、后续应用接口或自然语言评测器，不得作为论文主线。
**Method Thesis**: 当前路线需要从“冻结 LLM + 外挂记忆”和“统计 n-gram memory”转向“纯 no-BP 可训练神经结构”。优先级应回到 DEN1810/dendritic-like 树突双室误差、STDP/BTSP 调制版本、e-prop/三因子 eligibility、feedback alignment、抑制性微回路、recurrent/SSM/reservoir 动态。`continuation_backoff` / n-gram-style sparse memory 只能作为调试工具、统计下界或 sanity baseline，不能作为最终方法或核心 claim；TinyStories/API memory 结果只能作为应用侧参考。
**Date**: 2026-06-17

## 当前结果汇总

| 阶段 | 任务 | 最强 no-BP 结果 | 关键对照 | 判断 |
|---|---|---:|---:|---|
| 静态分类 | MNIST 14x14, hidden=128, seeds=3 | `dfa_3factor` acc 0.9042 | BP acc 0.9128 | 支持三因子固定反馈可接近 BP，但任务太简单 |
| 时序 toy | delayed cue next-token, delay=12 | `eprop_3factor` target acc 1.000 | tuned BPTT target acc 1.000; reservoir 0.000 | 支持 eligibility trace + 固定反馈能训练隐藏记忆 |
| 真实文本 token | TinyStories tokenizer medium | `sparse_hebbian_context` CE 4.1126, acc 0.361, 92k tok/s | dendritic CE 4.5777 acc 0.118; torch Llama 80 updates CE 4.8748 acc 0.117 | 当前最强正信号 |
| hybrid backoff | TinyStories tokenizer medium | sparse memory 单独 CE 4.1126 | `hybrid_llama_context` CE 4.5369; `hybrid_dendritic_context` CE 4.5630 | always-on 线性 logits fusion 是负结果 |
| context order ablation | TinyStories tokenizer medium | `sparse_hebbian_context` order=3 CE 4.0796 | order=4 CE 4.1126; order=5 acc 0.370 | order=3 是新的 CE 默认点 |
| semantic-key memory | TinyStories tokenizer medium | combined_context CE 4.0368 | semantic-only CE 5.36；exact-only CE 4.0796 | semantic bucket 只带来小幅增益 |
| compositional cue task | held-out `(a,b)` pairs, target `(a+b) mod K` | `target_only_phase_binding_hebbian` held-out acc 1.000 for K=4/8/12 | pair lookup 0.417/0.146/0.093; scrambled phase control 0.500/0.188/0.213; eprop/reservoir/BPTT fail on K=4/8 | 更强的纯 no-BP 正信号：不再使用 cue->phase 直接教师，只用最终 target 三因子信号学习局部相位码 + 复数绑定 + Hebbian 原型 |
| phase-binding token learner | TinyStories tokenizer 50k/10k, vocab=256 | trace+apical(random-feedback)+inhibition WTA online full precision post CE 2.289, acc 0.437, greedy repeat-2 0.383; variable-type 8-bit row CE 2.295 acc 0.438, serialized state 706,841 bytes; 8-bit all-state row CE 2.523 acc 0.462; 8-bit readout-weight-only CE 2.295, inhibition-only CE 2.289, phase_codes-only CE 2.290, phase_prototypes-only CE 2.289, bias-only CE 2.513, counts-only CE 2.513 | phase WTA post CE 2.427 acc 0.422; trace+inhib post CE 2.358 acc 0.429 repeat-2 0.473; fixed-random gate post CE 2.316; 8-bit fixed clip CE 2.713; 16-bit row CE 2.504; online `sparse_context_aux` post CE 1.297 acc 0.571 | 当前最强纯 no-BP CE 是弱 dynamic apical error modulation + inhibition；随机反馈可承载 apical error。variable-type quantization 已显示向量/矩阵 int8 row + count/prior float32 可接近 full precision，同时 deployable state 约 0.707MB；下一步是 loadable integer checkpoint 和预测一致性验证；统计 auxiliary 仍只作调试/上界 |
| phase token order sweep | TinyStories tokenizer 50k/10k, vocab=256 | order=2 phase CE 3.551 acc 0.284 | order=1 CE 3.603; order=3 tuned CE 4.199; order=4 CE 10.462 | 负结果：naive 多 token 相位乘法退化，下一步应做分支化 dendritic/SSM 或门控组合，而不是简单堆 context order |
| continuation backoff | TinyStories tokenizer medium | continuation_backoff CE 3.3254, acc 0.360 | sparse order=3 CE 4.0796; combined_context CE 4.0368 | 仅作为强统计基线/调试工具；n-gram/Kneser-Ney 风格，不能作为最终路线 |
| online no-raw stream | TinyStories segmented stream | phase_competitive_online post CE 2.427, acc 0.422; continuation_backoff post CE 1.542, acc 0.791 | phase stream pre CE 3.425, online CE 3.229; sparse aux post CE 1.297 | 纯 phase/WTA 在线学习已跑通且有强正增益；统计 memory/continuation 只保留为适应上界和调试基线 |
| memory cap pruning | TinyStories segmented stream | cap=5000: continuation post CE 1.957, 352KB | uncapped CE 1.542, 618KB; cap=2000 CE 2.252, 216KB | 只用于存储-质量调试，不作为核心方法 |
| API-compatible local prototype | synthetic personalization/FAQ | hashed_memory acc 1.000 smoke / 1.000 medium | no-memory acc 0.025 / 0.017; raw retrieval 0.800 | 本地 schema adapter 形态有效 |
| real API schema QA smoke | synthetic personalization/FAQ | API+memory_hint acc 1.000 on 10 questions | API no-memory acc 0.200 | 真实 API 调用链闭合，小样本强正结果 |
| natural FAQ API demo | synthetic customer FAQ | API+memory_hint acc 1.000 on 8 questions | API no-memory acc 0.000 | API 能用 no-BP memory hint 生成自然 FAQ 答案 |
| generated FAQ API scale | generated customer FAQ | API+memory_hint acc 1.000 on 16 questions; local 256-fact acc 1.000 | API no-memory acc 0.000 | 扩展到模板生成多事实空间后仍闭合，但还不是开放域 dialogue |
| dialogue paraphrase FAQ API | generated dialogue-style FAQ | API+memory_hint semantic acc 1.000 on 16 paraphrased questions; local 256-fact acc 1.000 | API no-memory acc 0.000 | 支持从对话式指令更新 memory 后回答改写问句；仍依赖 schema/alias |
| FAQ revision/delete API | generated dialogue-style FAQ revision | local overwrite/delete all 1.000; API overwrite 1.000/delete suppression 1.000 on 4 facts | old-value leak 0.000 | 支持在线覆盖和遗忘，不靠保存旧 raw dialogue；仍存 canonical answer values |
| semantic FAQ router API | generated dialogue-style FAQ | semantic router local 256-fact acc 1.000; API+memory_hint acc 1.000 on 8 questions | API no-memory acc 0.000 | 去掉 alias 必需性，使用 hashed semantic prototypes；代价是 256 facts state 约 825KB |
| compressed semantic FAQ router | generated dialogue-style FAQ | sparse-only cap12 local 256-fact acc 1.000; revision/delete all 1.000 | uncompressed semantic state 824,791 bytes | 256 facts state 降到 292,592 bytes，保留 no-raw semantic routing |
| compressed semantic FAQ router API | generated dialogue-style FAQ | sparse-only cap12 API memory-hint acc 1.000 on 8 questions; local 64-fact acc 1.000 | API no-memory acc 0.000 | 压缩 router 仍能闭合真实 API 链路，state 79,931 bytes on 64 facts |
| value sketch FAQ memory | generated dialogue-style FAQ | sketch store local 256-fact acc 1.000; API memory-hint acc 1.000 on 8 questions | full answer store state 292,614 bytes | 不保存完整 answer text，改存结构化 answer sketch；state 284,248 bytes on 256 facts |
| multi-turn FAQ API session | generated dialogue FAQ support session | semantic sketch memory local acc 1.000 on 14 turns; API acc 1.000 on 10 turns | API no-memory acc 0.200; raw retrieval acc 1.000 | learn/query/revise/delete/query 闭环完成，并产出 human-readable transcript；仍非开放域 GPT-like final eval |
| personalized style API | generated writing preferences | API+style_sketch all-pass acc 1.000 on 8 natural writing prompts; deleted suppression 1.000 | API no-memory all-pass acc 0.000; raw profile acc 1.000 | 从事实召回推进到自然短文生成约束遵守和删后遗忘；仍是规则化偏好，不是开放域人类偏好评测 |
| personalized style judge | blinded API ranker on style outputs | style-sketch best 0.500 on 8 prompts; beats raw-profile 0.750 | no-memory best 0.500; raw-profile best 0.000 | 自然度/可用性评测显示 style-sketch 有价值但并未压倒 no-memory，说明还需更柔和的 hint/render 路线 |
| preference-aware style judge | generated writing preferences | strict style-sketch best 0.875 with learned preference context; soft style-sketch best 0.625 | request-only judge preferred no-memory 1.000 on soft outputs | 个性化评审必须显式给出 learned preference；当前 strict sketch 比 soft sketch 更强，下一步应修偏好语义而不是弱化约束 |

## 最终验收标准

最终结果以“no-BP 在线学习方法在推理样本上接近主流 GPT 输出”为目标，但每轮实验必须先通过本地定量门槛，避免只看个别生成样本。

| 维度 | 验收方式 | 当前门槛 | 最终门槛 |
|---|---|---|---|
| 生成质量 | held-out prompts 的 greedy/sample completion，人眼检查是否自然、少重复、能延续语义 | 明显优于 STDP/BTSP/dendritic/recurrent_3factor | 在目标任务上接近 GPT5.5/API 样本的可读性和任务完成度 |
| CE / PPL | TinyStories 或 `wwwy4/huggingface_datasets` 中合适 next-token/QA 数据集 | 先超过 sparse-only medium CE 4.1126 或至少不退化 | 接近强 frozen/API baseline 的 proxy CE 或偏好分数 |
| Accuracy | next-token top-1、QA/分类准确率、个性化事实召回准确率 | 超过 sparse-only acc 0.361 或在低置信 subset 明显提升 | 在任务准确率上接近主流模型，同时训练更快 |
| 训练速度 | train tokens/s 和 wall-clock 适应时间 | 保持 no-BP memory 的高吞吐优势，避免为了小提升牺牲 10x 速度 | 在线更新显著快于 BP/微调 |
| 在线学习 | prequential / streaming evaluation | 不保存原始文本，只保存统计/压缩 memory state | 实际交互中边用边学，不依赖 replay 原始数据 |
| 存储与隐私 | active contexts、bytes/token、是否可还原原文 | 报告 memory size，并实现 pruning/decay | 可控存储，不保存原始训练样本 |

在把任何结果解释为“可替代 BP 的主方法”之前，必须先在**纯 no-BP 本地结构**上完成 smoke/medium 验证；所有 API / 冻结主干 / 预训练 LLM 只允许出现在对照、接口或后验应用示例中，不允许反向定义主线方法。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: 局部/结构化 no-BP 神经机制能在受控时序和组合任务上替代查表 | 证明算法机制不是普通 n-gram 统计或 pair lookup | delayed cue 中 eprop 成功；组合 cue 中 target-only phase-binding Hebbian 只用最终 target 信号对 held-out pairs 达到 1.000，而 pair lookup/eprop/BPTT/reservoir 失败 | B1, B2 |
| C2: 稀疏/continuation memory 是强统计基线而非最终方法 | 防止把 n-gram/Kneser-Ney 风格记忆误判为可替代 BP 的仿生神经结构 | 用它们作为调试下界，并要求纯 recurrent/SSM/dendritic 主体在关键任务上超过或解释这些基线 | B3, B4 |
| C3: 未来接 GPT/API 应采用置信度门控 adapter，而不是 always-on 线性融合 | 线性融合已被 medium 结果否定 | gated adapter 在低置信 context 才调用 backoff，CE 或 acc 超过 sparse-only，且生成重复不恶化 | B5, B6 |

## Paper Storyline

Main paper must prove:

- “替代 BP”必须先限定为**纯 no-BP 神经网络类小/中规模结构**，不依赖任何 BP 预训练主干。
- 最终候选应是 DEN1810/dendritic-like、STDP/BTSP 调制、e-prop/三因子 eligibility、反馈对齐、抑制性回路或 recurrent/SSM/reservoir 动态这类神经结构。
- 三因子固定反馈、eligibility trace、树突/抑制性局部误差、相位/振荡绑定或 reservoir/SSM 动态必须在受控任务中学习时序 credit assignment 和组合泛化。
- 在真实 token 数据上，稀疏 Hebbian / continuation memory 只能作为调试基线、数据 sanity check 和统计下界；最终主线必须是可训练的纯 no-BP 神经/动态主体，而不是 n-gram/Kneser-Ney-style 表格记忆。
- 与 GPT/API 的连接只能是外部上界或应用展示；论文核心 claim 应围绕纯仿生结构、局部更新和硬件可实现性。

Appendix can support:

- MNIST DFA 接近 BP 的早期结果。
- recurrent_3factor TinyStories 负结果。
- normalized backoff 和 linear hybrid 负结果。
- 更多 context order / memory size / seed sweep。

Experiments intentionally cut:

- 不做 GPT 级规模的从零训练，但必须做纯 no-BP 神经网络类小/中规模 token learner；不能用“预训练 LLM + 外挂 memory”或“n-gram/backoff 统计表”替代这个主线。
- 大规模 API 消耗测试；API 只作为后验对照或评测工具。
- 继续盲目刷 API/FAQ demo；除非它服务于纯 no-BP 主体的评估。
- 继续大规模调 recurrent_3factor 前，应先完成 delay sweep、compositional cue、tree/dendritic error、phase-binding token learner 和可硬件化 memory 的关键判别实验。

## Experiment Blocks

### Block 1: Delayed And Variable Cue Sanity

- Claim tested: C1
- Why this block exists: 防止三因子结果只是在固定 delay 上记位置。
- Dataset / split / task: synthetic cue/filler/target，delay in {4, 8, 12, 16, 20}，再加 variable-delay train/test。
- Compared systems: bigram, reservoir, eprop_3factor, eprop_resampled, tuned BPTT。
- Metrics: target-position accuracy, overall accuracy, loss, train time。
- Setup details: hidden_dim in {12, 24}, seeds=3。
- Success criterion: eprop_3factor 在长 delay 和 variable delay 上显著超过 reservoir/resampled，并接近 tuned BPTT。
- Failure interpretation: 若只在固定 delay 成功，则当前规则更像位置模板而不是稳健时序记忆。
- Table / figure target: Appendix or method sanity figure。
- Priority: MUST-RUN, but lower than B5 because main token result has moved forward。

### Block 2: Compositional Cue Task

- Claim tested: C1
- Why this block exists: 区分联想记忆和简单查表。
- Dataset / split / task: `C_a, F, C_b, F...F, T_(a+b mod K)`，包含 held-out cue pairs。
- Compared systems: pair lookup, hand-coded phase binding, learned cue-to-phase binding, target-only phase binding, scrambled phase control, reservoir, eprop_3factor, eprop_resampled, tuned BPTT。
- Metrics: seen-pair target acc, held-out-pair target acc, loss。
- Setup details: K in {4, 8}, held-out pair split, seeds=3。
- Success criterion: target-only phase binding 在 held-out pairs 上稳定超过 pair lookup 和 scrambled control；若普通 eprop/RNN 失败，则说明需要显式结构化绑定而不是泛型 recurrent credit assignment。
- Failure interpretation: seen 成功 held-out 失败意味着机制更偏局部关联，暂不能宣称组合推理；若只有 hand-coded phase 成功，则说明手工结构过强，需要继续去教师化。
- Table / figure target: Main or appendix depending on outcome。
- Priority: MUST-RUN。

### Block 3: TinyStories Statistical Baselines For Debugging

- Claim tested: C2
- Why this block exists: 固化 n-gram / Kneser-Ney-style 统计基线，作为纯 no-BP 神经主体必须超过或解释的下界；不把它当最终路线。
- Dataset / split / task: TinyStories tokenizer-level next-token，medium 设置。
- Compared systems: STDP, STDP-Bio, BTSP, BTSP-Bio, dendritic_error_1810_lite, recurrent_3factor, phase_binding_token, sparse_hebbian_context, continuation_backoff, low-budget torch_llama upper-bound/control。
- Metrics: CE, ppl, top-1 acc, train tok/s, active contexts, greedy sample repetition。
- Setup details: train_chars=50k, valid_chars=10k, max_vocab=256, context_max_order=4, seeds/config repeats=3 if script supports seeds；phase-binding order sweep 已显示 order=2 最稳。
- Success criterion: 统计基线稳定可复现，并用于暴露数据切片、tokenizer、评测和生成重复问题；pure phase-binding token learner 至少在 CE 或低置信 subset 上超过统计辅助基线，同时明确 top-1 差距。
- Failure interpretation: 如果纯神经 no-BP 主体无法超过这些统计基线，需要重新设计 credit assignment / state dynamics，而不是把统计基线包装成最终方法；若 naive order>2 退化，则改用 dendritic branches、gating 或 SSM state。
- Table / figure target: Baseline / Appendix table。
- Priority: DEBUG-BASELINE。

### Block 4: Context Order, Calibration, And Memory Size Ablation

- Claim tested: C2
- Why this block exists: 找到 sparse memory 的真实收益来源和规模边界。
- Dataset / split / task: TinyStories tokenizer medium。
- Compared systems: max_order in {1, 2, 3, 4, 5, 6}; additive vs normalized; memory pruning; alpha/temperature sweep。
- Metrics: CE, acc, active contexts, bytes per token, repetition rate。
- Setup details: 固定数据和 eval；先单 seed 小 sweep，再复跑最优三组。
- Success criterion: 证明 high-order context 提供主要收益，并得到可控 memory-size/quality tradeoff。
- Failure interpretation: 如果 order>2 没收益，方法更像普通 bigram/trigram，需要重新定位。
- Table / figure target: Main ablation table。
- Priority: MUST-RUN。

### Block 5: Confidence-Gated Online Adapter

- Claim tested: C3
- Why this block exists: 当前 linear hybrid medium 负结果说明不能始终混入 neural logits；需要让 backoff 只处理 memory 低置信场景。
- Dataset / split / task: TinyStories tokenizer medium，先本地 low-budget Llama/dendritic backoff，之后再考虑 API。
- Compared systems: sparse memory only, normalized sparse memory, linear hybrid, gated+dendritic backoff, gated+Llama backoff。
- Metrics: CE, acc, low-confidence subset CE, high-confidence subset CE, gate usage rate, repetition rate, train/eval tok/s。
- Setup details: confidence = highest-order context exists + total row count + max row probability/entropy；threshold sweep。
- Success criterion: gated adapter CE < 4.1126 or acc > 0.361，且 greedy repetition 不差于 sparse-only；低置信 subset 上 backoff 有明确收益。
- Failure interpretation: 若 gated 仍退化，说明当前 neural backoff 太弱或 logits 标度不兼容，先换更强 frozen backbone 再测。
- Table / figure target: Main Table 2 and confidence calibration plot。
- Priority: MUST-RUN FIRST。

### Block 6: Continual No-Raw-Data Online Learning

- Claim tested: C2/C3
- Why this block exists: 直接验证最终目标中“无需保存数据直接在实际中学习”的核心约束。
- Dataset / split / task: TinyStories 按 story/domain/style 分段在线流式输入；训练后丢弃原文，只保留 memory/adapter state。
- Compared systems: phase_competitive_online, sparse memory auxiliary, continuation auxiliary, replay-free vs compressed-stat replay。
- Metrics: prequential CE, adaptation tokens-to-improvement, old-style retention, new-style adaptation, memory growth, deletion/pruning 后性能。
- Setup details: online train/eval interleaving；不得使用原始训练段回放。
- Success criterion: 新风格 CE 在少量 token 内下降，旧风格不过度遗忘，memory size 可通过 pruning 控制；纯 phase/WTA 已满足第一版在线正结果，下一版看低精度和生成质量。
- Failure interpretation: 若只能记住最近局部 n-gram，需引入聚类/semantic key 或 API embedding key。
- Table / figure target: Main online learning figure。
- Priority: MUST-RUN after B5。

### Block 7: Pure No-BP Application Bridge

- Claim tested: C3 and long-term goal
- Why this block exists: 只在纯 no-BP 主线稳定后，再把机制映射到应用接口；不是依赖预训练 LLM 来定义方法，而是用它们做后验接口或评测上界。
- Dataset / split / task: 小规模个性化写作/FAQ/客服风格流，但主方法必须先用纯 no-BP 主体完成；如果接入 API，只能作为对照评估，不得作为核心训练骨干。
- Compared systems: pure no-BP model, raw retrieval baseline, optional API or frozen backbone upper bound。
- Metrics: next-token/proxy CE if logits available, task accuracy, exact recall, hallucination rate, storage bytes, privacy constraint。
- Setup details: 只保存 hashed/compressed context statistics or embedding-keyed memory；若使用 API，仅用于评测，不用于训练主方法。
- Success criterion: 纯 no-BP 主体在个性化/新事实任务上能自洽学习，并满足 no-raw-data storage；若 API 只提供上界，则不影响主结论。
- Failure interpretation: 若纯 no-BP 主体必须借助预训练 LLM 才能有效，则该路线不满足当前项目约束。
- Table / figure target: Demo / application section。
- Priority: NICE-TO-HAVE after pure no-BP core passes。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 固化纯 no-BP 证据 | 复跑 eprop_3factor、dendritic_error；sparse/continuation memory 仅作调试基线 | 至少一个纯 no-BP 神经/动态结构在受控任务稳定强于 reservoir/output-only，并明确区分统计基线 | CPU/GPU minutes | 单次结果偶然 |
| M1 | 机制判别 | delay sweep、trace decay、feedback fixed/resampled、dendritic apical/basal ablation | 证明性能来自 eligibility/feedback/dendritic mechanism，而不是 n-gram 或随机特征 | CPU < 1h | 机制只是查表 |
| M2 | 纯 no-BP token learner | phase/trace/WTA + output inhibition + dendritic/apical local error gating on TinyStories tokenizer | 不依赖 Llama/API 的 CE/acc 明显优于 STDP/BTSP/dendritic 旧 baseline，并持续缩小与统计辅助基线的差距 | CPU/GPU 1-2h | 长文本生成弱 |
| M3 | 可硬件化在线学习 | memory cap、局部更新、整数/低精度状态、no-raw replay | 存储、更新和读出都能解释为局部/硬件友好操作 | CPU/GPU 1-2h | memory 无界增长 |
| M4 | 组合与泛化 | compositional cue、held-out pair、style/fact shift 的纯 no-BP 版本 | 证明不是只记训练 n-gram 或固定位置 | CPU 1-2h | 组合泛化失败 |
| M5 | 外部对照/应用展示 | 可选 API/frozen baseline 只作上界或评测器 | 不改变主结论；若纯 no-BP 不成立则不得用 API 结果补洞 | API 成本受控 | API 结果掩盖方法本体弱点 |

## Compute and Data Budget

- Total estimated GPU-hours before any external-model comparison: < 5 小时，小模型即可。
- API budget rule: API 不再是路线推进条件；只有纯 no-BP 主体已经形成可解释结果后，才可作为评测器或上界对照使用。
- Data preparation needs: TinyStories 分段流式切片；可选构造 style/fact shift、composition split、delayed association split。
- Human evaluation needs: 只有纯 no-BP 模型能生成可读样本后才需要少量人工检查。
- Biggest bottleneck: 纯 no-BP 主体的长程 credit assignment、context 泛化、memory size 控制和硬件可实现性。

## Risks and Mitigations

- Risk: 依赖 BP 预训练 LLM / API 会污染主张，使方法变成“BP 模型外挂记忆”，不适合纯仿生结构或硬件路线。
- Mitigation: 所有主表、主结论、路线图必须以纯 no-BP 从头训练或局部在线更新结构为核心；API/预训练模型只能放在附录、上界或应用展示。

- Risk: sparse / continuation memory 只是 n-gram 或 Kneser-Ney-style 统计计数，不能支撑“类脑学习算法”叙事。
- Mitigation: 明确标记为 DEBUG-BASELINE；只用于调试、下界和 sanity check。最终 claim 必须来自 DEN1810/dendritic-like、STDP/BTSP、e-prop、feedback alignment、recurrent/SSM/reservoir 等纯神经或动态结构。

- Risk: 把“无梯度统计方法”误认为“神经网络类 no-BP 方法”。
- Mitigation: 主方法必须包含可解释的神经状态、突触更新或局部可塑性机制；没有神经状态/突触结构的表格统计方法只能辅助分析。

- Risk: 线性 hybrid 已失败，gated 也可能失败。
- Mitigation: 分 high/low-confidence subset 报告；若 backoff 只在 low-confidence 子集有效，仍有价值。

- Risk: low-budget Llama 或 API baseline 会把研究重心带回 BP 训练模型，而不是纯 no-BP 结构。
- Mitigation: Llama/API 只能作为对照或评测上界；真正 claim 必须在纯 no-BP 主体上提出。

- Risk: no-raw-data memory 仍可能泄漏原文。
- Mitigation: 报告 memory schema、存储 token IDs/hashed keys/统计值，增加 pruning 和 privacy audit。

- Risk: 直接追求 GPT 级别会烧钱且不可证。
- Mitigation: 阶段性目标先定义为在线个性化 adapter，只有在 adapter 明显提升后再扩大。

## Final Checklist

- [x] MNIST DFA 接近 BP 的早期证据已覆盖
- [x] delayed hard toy task 已显示 eprop_3factor 可达 tuned BPTT
- [x] TinyStories sparse Hebbian context 有 medium 正结果
- [x] linear hybrid backoff 已判定为负结果
- [x] confidence-gated adapter
- [x] context order / calibration / memory size ablation
- [x] semantic-key Hebbian memory
- [x] continuation / Kneser-Ney-style backoff memory 已降级为调试基线，不作为最终方法
- [x] continual no-raw-data online learning
- [x] memory cap / pruning tradeoff
- [x] compositional cue task
- [ ] pure no-BP token learner 主线
  - [x] dendritic apical/basal error variant 第一版：branch-wise apical local error gate，seed0/seed1 均改善 CE
  - [x] apical gate ablation: global margin and fixed random feedback match or beat branch-local margin; fixed random gate is weaker
  - [x] low-precision / sparse-state audit 第一轮：8/16-bit projection 保留相对 apical 增益但 CE 校准退化
  - [x] generation rerun for random-feedback apical error: greedy repeat-2 0.473->0.383, controlled repeat-2 0.144->0.106
  - [x] quantization-aware updates / per-state scaling 第一轮：8-bit row scaling 将 apical CE 从 fixed-clip 2.713 改到 2.523，并保留 greedy repetition benefit
  - [x] selective quantization localization: CE 退化主要来自 `output_bias`/log-prior 和 count-like arrays；readout weights、inhibition、phase codes/prototypes 单独 8-bit row 接近 full precision
  - [x] dedicated bias clip sanity: `--low-precision-bias-clip 8.0` 未恢复 all-state row CE，说明不能只靠单独 bias clip
  - [x] cached target-array projection for low-precision wrapper
  - [x] serialized-state manifest/accounting: int8 vectors/matrices + float32 count/prior state recovers apical CE 2.295 with serialized bytes 706,841
  - [x] loadable integer checkpoint and prediction parity check: mixed int8/float32 `.npz` restores into a fresh no-BP learner with 1000-context pred_match 1.000 and max score diff 2.38e-7
  - [x] direct prior ablation: `--phase-bias-weight 0.0` raises variable 8-bit apical CE 2.295->2.523 but improves acc 0.438->0.462 and preserves apical advantage, showing winner selection can be neural while calibration still needs replacement
  - [x] simple global output homeostasis audit: target-up/wrong-down excitability does not recover CE calibration, though it is small and checkpointable
  - [x] feature-conditioned neural calibration first pass: fixed random gate + local target/wrong-winner calibration improves variable 8-bit bias-free CE 2.523->2.478 and acc 0.462->0.474, with checkpoint parity 1.000
  - [x] derived feature-calibration codes: regenerate fixed random codes from seed/architecture, preserving CE/acc/parity while reducing serialized bytes 986,465->724,317
  - [x] checkpoint config/hash validation: derived-state checkpoint stores SHA-256 signature and rejects corrupted/incompatible signatures before loading
  - [x] feature-calibration gate sweep: gate_decay=0.50 improves signed derived checkpoint CE 2.478->2.475 at unchanged 724,317 serialized bytes; dim32 offers smaller 716,093-byte variant with CE 2.480
  - [x] temperature / energy-scale audit: global temperature 0.7 improves bias-free feature-calibrated checkpoint CE 2.475->2.262 at unchanged bytes, beating direct-prior CE 2.295 without statistical token probabilities
  - [ ] replace global temperature with local adaptive neural gain from WTA margin/branch agreement/inhibition pressure; rerun generation/repetition for best bias-free checkpoint
  - [x] recurrent/SSM/reservoir + 三因子 eligibility on TinyStories 已跑；fixed reservoir 有边际信号，eligibility transition 写入未通过 seed 复核
  - [x] target-only phase-binding token learner CE signal on TinyStories
  - [x] branched phase token learner improves CE to 3.282
  - [x] branch readout calibration improves acc to 0.312 but remains below sparse 0.323
  - [x] competitive WTA readout improves acc to 0.322 and CE to 3.195
  - [x] strict online phase/WTA stream improves pre CE 3.425 acc 0.303 to post CE 2.427 acc 0.422
  - [x] generation audit: online phase improves first-token match but greedy repeat-2 remains high; controlled decoding mitigates loops
  - [x] leaky trace/SSM branch improves online phase post CE to 2.389 and acc to 0.427, with modest greedy repetition reduction
  - [x] output fatigue dynamics improves trace+phase post CE to 2.382 and acc to 0.429, with greedy repeat-2 down to 0.606
  - [x] adaptive output inhibition improves trace post CE to 2.358 and seed1 trace CE 2.450->2.410, but generation loops remain
  - [x] context-gated output inhibition improves trace+fatigue acc to 0.435 and repeat-2 to 0.346, but CE worsens to 2.416
  - [x] plastic recurrent/SSM branch implemented; medium plastic SSM+inhib reaches CE 2.377 acc 0.425 but remains below trace+inhib
  - [x] phase-binding context-order sweep; naive order>2 negative
  - [x] composition / held-out cue 泛化
  - [x] target-only phase factorization without cue->phase teacher
  - [ ] hardware-friendly sparse memory / low-precision state audit
- [ ] optional API / pretrained-model upper-bound appendix
  - [x] local synthetic QA proxy
  - [x] real API schema QA smoke
  - [x] natural FAQ API demo
  - [x] generated natural FAQ API scale check
  - [x] dialogue/paraphrase API smoke
  - [x] overwrite/delete revision audit
  - [x] learned semantic routing
  - [x] semantic memory compression
  - [x] compressed API smoke
  - [x] value compression
  - [x] multi-turn human-readable qualitative demo
  - [x] personalized style API generation benchmark
  - [x] personalized style judge benchmark
  - [x] preference-aware style judge benchmark
- [ ] pure no-BP qualitative and quantitative final evaluation
