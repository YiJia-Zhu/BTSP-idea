# Auto Exploration Strategy

**Date**: 2026-06-15  
**Long Goal**: 自动探索一条 no-BP / 仿生物神经在线学习路线，使其在推理样本和定量指标上尽量接近主流 GPT/API 输出，同时保持训练快、在线更新、无原始数据保存。

## 1. 总目标拆解

最终不是证明某一个固定算法必然正确，而是持续搜索可行路线：

```text
no-BP online learner
  -> fast local update
  -> no raw data storage
  -> improves with interaction
  -> generated samples approach GPT-like quality
  -> quantitative metrics approach strong baseline
```

当前最强起点：

- `sparse_hebbian_context`: TinyStories medium CE 4.1126, acc 0.361, train speed ~90k tok/s。
- high-confidence memory bucket: CE 2.1966, acc 0.696。
- 当前瓶颈: low-confidence / unseen context 泛化弱，exact n-gram memory 不能产生 GPT-like 语义连贯输出。

## 2. 自动探索循环

每轮必须形成闭环：

1. **Hypothesis**
   - 写清楚本轮假设，例如 “semantic key 能让未见 context 泛化”。
2. **Implementation**
   - 优先改现有脚本，保持可复现 CLI。
   - no-BP 路线必须只做局部/统计/闭式/在线更新；不能把核心学习偷换成 BP 微调。
3. **Local Smoke**
   - 小数据快速验证不崩，输出 metrics 和 samples。
4. **Medium Eval**
   - 与当前 best `sparse_hebbian_context` 对比。
5. **Decision**
   - 正结果：扩展到 ablation / larger / dataset eval。
   - 负结果：记录失败原因，修改路线。
   - 连续 2 轮同类负结果：停止该方向，转文献或新机制。
6. **Documentation**
   - 更新 tracker、manifest、iteration report。

## 3. 成功判据

### 本地短期成功

任一方向进入下一阶段，需要至少满足其中之一：

- CE < 4.1126 on TinyStories medium；
- accuracy > 0.361；
- low-confidence subset CE 明显低于当前 ~5.02；
- generation repeat-2 rate < 0.20 且样本语义更连贯；
- online/prequential setting 中少量 token 后明显改善，同时不保存原文。

### 中期成功

- 在 `wwwy4/huggingface_datasets` 或等价任务上，no-BP learner 的 accuracy / CE 明显优于无记忆 baseline。
- 训练速度显著快于 BP 微调。
- 存储可控，能报告 bytes/token 或 active memory size。

### 最终成功

- no-BP adapter 推理样本接近 GPT/API 样本的可读性和任务完成度。
- 定量上在目标数据集接近强 baseline。
- 在线学习过程中不保存原始训练样本。

## 4. 路线优先级

### Priority A: Semantic-Key Hebbian Memory

目的：解决 exact n-gram 未见 context 泛化弱。

候选实现：

- 用 frozen local Llama embedding / hidden state 做 key。
- 用 random projection / SimHash / product quantization 把 context 压缩到 buckets。
- 每个 bucket 用 Hebbian/statistical update 存 next-token 分布。
- 可组合 exact n-gram memory + semantic bucket memory。

成功标准：

- low-confidence subset CE < current ~5.02。
- overall CE 接近或优于 4.1126。
- generation 不更差。

### Priority B: Context Order / Memory Size Ablation

目的：明确当前 sparse memory 的收益来自几阶 context，以及成本如何增长。

需要跑：

- max_order = 1..6。
- additive vs normalized。
- pruning/decay threshold。

成功标准：

- 找到最小有效 order。
- 报告 active contexts 和 approximate memory bytes。

### Priority C: Continual No-Raw-Data Stream

目的：贴近最终在线学习。

设置：

- 把 TinyStories 或 QA 数据按 segment 流式输入。
- 每段先 eval，再 online update，再 eval。
- 更新后丢弃原文，只保留 memory state。

成功标准：

- 新 segment adaptation CE 降低。
- 旧 segment retention 不崩。
- memory size 可控。

### Priority D: Dataset Task Evaluation

目的：不用只看 TinyStories next-token。

候选：

- `wwwy4/huggingface_datasets` 中可本地加载的数据。
- QA / classification / next-token style。
- 如果远端 dataset 不可用，先用本地 TinyStories 构造 QA/fact recall。

成功标准：

- accuracy 明确可比较。
- no-BP update 后超过 no-memory baseline。

### Priority E: Literature Pivot

触发条件：

- semantic-key memory 连续 2 轮不能改善 low-confidence subset；
- context ablation 证明 exact memory 已到上限；
- generation 仍无法改善。

调研方向：

- hippocampal indexing / complementary learning systems；
- Kanerva sparse distributed memory；
- modern kNN-LM / cache-LM；
- reservoir computing + online readout；
- feedback alignment / e-prop for recurrent adapters；
- predictive coding / equilibrium propagation 的局部替代训练。

调研输出必须转成可实现实验，不写空泛综述。

## 5. Stop / Go 规则

继续当前方向：

- 指标改善，或诊断显示明确可修复瓶颈。

停止当前方向：

- 只改善样本外观但 CE/accuracy/online 指标无改善；
- 训练速度损失超过 10x 且质量无明显提升；
- 需要保存原始数据才能有效；
- 需要 BP 微调核心模块才有效。

允许阶段性负结果：

- 负结果要保留，因为它能缩小搜索空间。
- 每个负结果必须写清“为什么负、下一步怎么变”。

## 6. 当前下一步

立即执行顺序：

1. `R016`: context order / memory size ablation。
2. `R023`: implement semantic-key Hebbian memory using frozen local representations or hashed context features。
3. `R024`: compare exact memory vs semantic memory vs combined memory on low-confidence subset。
4. `R018`: continual no-raw-data stream。
5. 若 R023/R024 失败，进入 literature pivot。

