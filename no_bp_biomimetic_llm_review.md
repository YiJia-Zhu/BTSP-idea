# 类脑、完全不用 BP 的语言模型 / LLM 检索笔记

更新时间：2026-06-02

## 1. 我这里的筛选标准

我把“完全不用 BP”分成三档：

1. **严格 no-BP**
   - 训练时不用反向传播
   - 也不用 BPTT
   - 也不用 surrogate gradient / local gradient approximation / distillation 作为主训练机制
2. **弱 no-BP**
   - 不用标准 BP，但仍然在做梯度近似，或者本质上仍在优化一个梯度目标
   - 例如 predictive coding, equilibrium propagation, surrogate gradient SNN
3. **伪 no-BP**
   - 推理时有 Hebbian / STDP / fast weights / online plasticity
   - 但预训练主体仍然靠 BP/BPTT

你现在要找的，是第 1 类。

---

## 2. 结论先说

**截至 2026-06-02，我没有找到一个主流、可信、端到端预训练的大语言模型，是用 STDP / Hebbian / 纯局部塑性、并且完全不用 BP/BPTT/surrogate gradient 训练出来的。**

目前最接近的有两条线：

- **Reservoir / Echo State**：确实可以做到几乎不用 BP，甚至只训练读出层
- **Hebbian / STDP 小模型**：确实可以完全局部学习，但规模和任务还远没有到现代 LLM 水平

所以如果你的问题是：

> “有没有真正类脑、完全不用 BP 的 LLM？”

当前更准确的回答是：

**还没有成熟答案。**

---

## 3. 真正接近“完全不用 BP”的语言相关工作

### 3.1 Reservoir Computing as a Language Model (2025)

- 论文：Reservoir Computing as a Language Model
- 结论：这是我找到的最干净的“语言建模 + 基本不用 BP”候选之一
- 关键信息：论文明确写了是两种 reservoir computing 方法，**只有输出层可训练**

这类方法的本质是：

- 大的递归动力系统是固定的
- 文本序列只是在 reservoir 里激发状态轨迹
- 学习主要发生在读出层

优点：

- 很省算力
- 很接近“固定动力系统 + 局部读出”的神经科学风格

缺点：

- 表达能力和可扩展性目前还是明显不如 Transformer

链接：

- https://arxiv.org/abs/2507.15779

---

### 3.2 Syntactic Learnability of Echo State Neural Language Models at Scale (2025)

- 论文：Syntactic Learnability of Echo State Neural Language Models at Scale
- 结论：继续说明 **ESN 这种几乎不训练内部递归权重** 的方案，在某些语言结构任务上 surprisingly strong
- 关键信息：论文把 ESN 当作一种“最小复杂度”的 neural language model，并报告在 grammaticality judgment 上可与 Transformer 比较

这说明：

- 语言中的一部分结构学习，未必需要完整 end-to-end BP
- 固定高维动力系统 + 简单读出，有时已经能抓住不少句法规律

但它不是现代生成式 LLM 的替代品。

链接：

- https://arxiv.org/abs/2503.01724

---

### 3.3 Echo State Neural Machine Translation (2020)

- 论文：Echo State Neural Machine Translation
- 关键信息：论文明确写 encoder 和 decoder 的 layer weights **随机生成后在训练中保持固定**
- 结果：即使这么极端，机器翻译也能达到 fully trainable baseline 的 70-80%

这很重要，因为它说明：

- 对序列建模来说，“固定动态系统 + 少量可训练映射”并不是完全没用
- 但它依然离今天的 LLM 很远

链接：

- https://arxiv.org/abs/2002.11847

---

### 3.4 Hebbian learning the local structure of language (2025)

- 论文：Hebbian learning the local structure of language
- 方向：更像一个“语言微观起源 / Hebbian tokenizer + embedding 绑定”的理论模型
- 关键信息：作者明确把学习设为 local and unsupervised Hebbian

这篇更偏理论和机制构想，不是已经成熟的 LLM 训练范式。

链接：

- https://arxiv.org/abs/2503.02057

---

## 4. 真正 no-BP，但目前主要还在视觉 / 小规模任务的类脑工作

### 4.1 SoftHebb (2021) / Hebbian Deep Learning Without Feedback (2022)

- SoftHebb: https://arxiv.org/abs/2107.05747
- Hebbian Deep Learning Without Feedback: https://arxiv.org/abs/2209.11883

这条线非常重要，因为它不是“近似 BP”，而是直接走另一条路：

- soft winner-take-all
- Hebbian 局部更新
- 不要 feedback target / error signal

其中 `Hebbian Deep Learning Without Feedback` 甚至明确写：

- without any feedback, target, or error signals

这说明：

- “完全不用 BP”并非只能做单层模型
- 多层局部 Hebbian 网络确实可以学到有用表征

但问题仍然是：

- 任务主要是视觉
- 深度和规模还远未到 LLM

---

### 4.2 STDP-based supervised / unsupervised SNN

代表：

- An STDP-Based Supervised Learning Algorithm for Spiking Neural Networks
  - https://arxiv.org/abs/2203.03379
- An Unsupervised STDP-based Spiking Neural Network Inspired By Biologically Plausible Learning Rules and Connections
  - https://arxiv.org/abs/2207.02727

意义：

- 说明 STDP 的确可以支持分类学习
- 但通常需要特定编码、竞争机制、阈值调节、侧抑制、homeostasis 等配套结构

所以不能把它理解成：

- “只要有 STDP，就自然得到 LLM 级别学习能力”

更准确地说：

- STDP 能学习时序相关和特征探测器
- 要变成复杂认知系统，还要加网络结构、记忆机制、重放、调制信号

---

## 5. 那些“看起来类脑”，但其实不是严格 no-BP 的

### 5.1 SpikeGPT / SpikeLLM / 其他 spiking LLM

代表：

- SpikeGPT
  - https://arxiv.org/abs/2302.13939

这类工作很容易让人误以为：

- “用了 spike，就是不用 BP”

但 SpikeGPT 的摘要直接写了：

- **largest backpropagation-trained SNN model**

也就是说：

- 架构是脉冲神经网络
- 训练仍然靠 BP

所以它不属于你要找的第 1 类。

---

### 5.2 NeuronSpark (2026)

- 论文：NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics
- 链接：https://arxiv.org/abs/2603.16148

这篇摘要明确写了：

- trained with next-token prediction and surrogate gradients

所以它也不是严格 no-BP。

---

### 5.3 BiSpikCLM (2026)

- 论文：BiSpikCLM: A Spiking Language Model integrating Softmax-Free Spiking Attention and Spike-Aware Alignment Distillation
- 链接：https://arxiv.org/abs/2605.13859

这篇是一个很新的反例，因为它会让人误以为：

- fully binary
- spiking
- MatMul-free

就已经接近“完全类脑 + 完全不用 BP”。

但摘要里写得很清楚：

- `For efficient training, we introduce Spike-Aware Alignment Distillation`

也就是说：

- 它虽然在表示和推理形式上更像脉冲系统
- 训练上仍然依赖 teacher-student 蒸馏

所以它依旧不属于严格 no-BP。

---

### 5.4 Dragon Hatchling / BDH (2025)

- 论文：The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain
- 链接：https://arxiv.org/abs/2509.26507

这篇很容易被误解，因为它确实很“脑启发”：

- 推理时 working memory 依赖 synaptic plasticity / Hebbian learning
- 结构上也在强调 local graph dynamics

但是它**不是严格 no-BP 训练**。

论文正文甚至专门有一节：

- `7.2 Training without backpropagation through time`

并且在正文讨论里明确写到了：

- during training with backpropagation ...

所以更准确的定位是：

- **推理和状态更新很像类脑**
- **训练主体仍未摆脱 BP/BPTT**

---

## 6. 神经科学这边给你的真正启发

如果你的终极目标是：

> 人脑为什么能建立时序联系，为什么看到 A 会想到 B？

那目前神经科学文献给出的方向，和“纯大模型训练技巧”不太一样。

更核心的机制是：

1. **局部时序塑性**
   - STDP / anti-Hebbian STDP 可以直接学习谁先谁后
2. **序列表征**
   - 海马体会显式编码事件序列关系
3. **memory linking**
   - 时间上接近的记忆会共享部分 engram，从而更容易互相唤起
4. **offline replay**
   - 休息 / 睡眠时重放，把短时序联系变成长期联想结构

代表文献：

- Engram mechanisms of memory linking and identity
  - https://www.nature.com/articles/s41583-024-00814-0
- Replay, the default mode network and the cascaded memory systems model
  - https://www.nature.com/articles/s41583-022-00620-6
- Hippocampal ensembles represent sequential relationships among an extended sequence of nonspatial events
  - https://www.nature.com/articles/s41467-022-28057-6
- Anti-Hebbian plasticity drives sequence learning in striatum
  - https://www.nature.com/articles/s42003-024-06203-8

这几篇合起来给出的图景更像是：

- **A 激活了一个隐状态/engram_A**
- 因为 A 和 B 在时间上反复相邻出现，局部塑性把 `engram_A -> engram_B` 这条边加强
- 之后看到 A，只需部分激活，就可能通过重放/补全/吸引子动力学把系统推向 B

这和 BP 的区别是：

- BP 学的是“全局 loss 怎么减”
- 脑更像是在学“哪些局部状态经常先后共现、哪些回路值得保留和重放”

---

## 7. 最后一句判断

如果你坚持两个条件同时成立：

- **类脑**
- **完全不用 BP**

那么今天最靠谱的技术路线还不是 Transformer-LLM，而是：

- **reservoir / ESN**
- **Hebbian / SoftHebb**
- **STDP + recurrent associative memory**
- **再加 replay / consolidation / neuromodulation**

如果你要的是“真正解释人脑时序联想”，这条路线比“纯 STDP 直接训练现代 LLM”更可信。
