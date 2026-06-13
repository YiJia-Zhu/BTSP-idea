# 替代反向传播算法：文献调研与类脑动机

日期：2026-06-03

## 研究问题

目标不是简单做一个 MNIST 分类器，而是寻找一种有可能替代反向传播的学习机制：

- 不依赖逐层误差反传和精确权重转置
- 尽量局部、在线、样本逐个到达时可更新
- 能从神经科学机制获得动机，而不是只做工程近似
- 至少在小规模任务上接近 BP，并能解释为什么有效

本文把候选方法分成四类：

1. 零阶优化
2. 生物可行的 BP 近似
3. STDP / Hebbian / 三因子学习规则
4. 神经科学中的记忆、时序关联、信用分配机制

---

## 1. 零阶优化：完全不用梯度，但不够类脑

代表思路：

- SPSA
- Evolution Strategies
- MeZO / zeroth-order LLM fine-tuning

核心机制：

给参数一个随机扰动 `u`，只看前向 loss：

```text
g_hat = (L(theta + eps u) - L(theta - eps u)) / (2 eps) * u
theta <- theta - eta g_hat
```

优点：

- 不需要反向传播
- 只要能计算前向 loss，就能优化
- 对黑盒模型、量化模型、不可微系统有价值

缺点：

- 方差高
- 维度越高越低效
- 类脑解释弱，因为它仍然是全局参数扰动和全局 loss

代表文献：

- MeZO: Fine-Tuning Language Models with Just Forward Passes
  - https://arxiv.org/abs/2305.17333
- Evolution Strategies as a Scalable Alternative to Reinforcement Learning
  - https://arxiv.org/abs/1703.03864

对我们的启发：

零阶优化适合作为“完全不用 BP”的负对照。它证明不用梯度也能学习，但不太像脑，因为脑不太可能对所有突触做全局随机扰动再计算一个整体 loss。

---

## 2. Feedback Alignment / Direct Feedback Alignment

代表思路：

- Feedback Alignment
- Direct Feedback Alignment

核心观点：

BP 需要把误差乘以每层前向权重的转置。Feedback Alignment 发现，即使用固定随机反馈矩阵，网络也可能学会让前向权重逐渐对齐这个随机反馈方向。

代表文献：

- Random synaptic feedback weights support error backpropagation for deep learning
  - Nature Communications, 2016
  - https://www.nature.com/articles/ncomms13276
- Direct Feedback Alignment Provides Learning in Deep Neural Networks
  - https://arxiv.org/abs/1609.01596

意义：

这条线很关键，因为它去掉了 BP 最不生物合理的部分之一：

- 不要求反馈权重等于前向权重转置
- 不要求精确逐层链式法则

但它仍然需要一个输出误差信号。所以它不是纯 STDP，也不是完全无监督。

---

## 3. STDP / Hebbian / 三因子学习规则

### 3.1 STDP

经典 STDP 可以写成：

```text
if pre before post:  delta_w > 0
if post before pre:  delta_w < 0
```

代表实验：

- Bi and Poo, Synaptic modifications in cultured hippocampal neurons
  - Journal of Neuroscience, 1998
  - https://www.jneurosci.org/content/18/24/10464

它解释了为什么局部时间顺序能改变突触强度。

但只靠 pair-based STDP 很难直接解决分类，因为它主要学习：

- 哪些输入共同激活
- 哪些输入先于某个神经元激活
- 哪些状态在时间上相邻

这更像特征学习 / 关联学习，不像监督 loss 优化。

### 3.2 三因子学习规则

更接近监督学习和强化学习的类脑规则是：

```text
delta_w_ij = learning_rate * pre_i * post_j_eligibility * modulatory_signal_j
```

三因子分别是：

- `pre_i`：突触前活动
- `post_j_eligibility`：突触后活动或资格迹
- `modulatory_signal_j`：多巴胺、奖励、错误、注意或随机反馈调制

这比纯 STDP 多了一个“这次变化是否有用”的调制项。

代表文献：

- e-prop: A biologically plausible learning rule for recurrent neural networks
  - https://www.nature.com/articles/s41467-020-17236-y
- Surrogate gradients / eligibility traces for SNN
  - https://arxiv.org/abs/1901.09948

对我们的启发：

如果把 Direct Feedback Alignment 写成三因子局部规则：

```text
delta_w_hidden = - eta * x_pre * relu_gate_post * (B * output_error)_post
```

那么它既不是标准 BP，又能利用输出层错误作为全局调制。这个形式比零阶优化更类脑，比纯 STDP 更能做任务学习。

---

## 4. Target Propagation / Equilibrium Propagation / Predictive Coding

这些方法都试图把 BP 改写成更局部或更动态的过程。

代表：

- Target Propagation
  - https://arxiv.org/abs/1412.7525
- Equilibrium Propagation
  - https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full
- Forward-Forward Algorithm
  - https://arxiv.org/abs/2212.13345

它们的共同问题是：

- 理论上更接近局部学习
- 但很多实现仍然有非局部目标、迭代平衡过程或工程复杂度
- 不一定比 DFA/三因子规则更适合作为第一轮可复现实验

因此本轮实验先不把它们作为主方法。

---

## 5. 神经科学动机：脑可能不做 BP，但也不是纯无监督 STDP

### 5.1 脑中的局部可塑性

STDP 给出一个局部时序机制：

- 谁先激活
- 谁后激活
- 两者时间差决定突触增强或减弱

这可以解释“时序关联”的底层边：

```text
h_A -> h_B
```

但它不能单独解释复杂任务目标。

### 5.2 调制信号和资格迹

神经系统里有大量调制信号：

- 多巴胺
- 去甲肾上腺素
- 乙酰胆碱
- 皮层反馈
- 奖励预测误差

这些信号可以把局部可塑性从“看到什么就连什么”提升到“对任务有用的连接被保留”。

这就是三因子学习规则的意义。

### 5.3 记忆链接与 replay

和用户长期目标直接相关的文献：

- Engram mechanisms of memory linking and identity
  - Nature Reviews Neuroscience, 2024
  - https://www.nature.com/articles/s41583-024-00814-0
- Replay, the default mode network and the cascaded memory systems model
  - Nature Reviews Neuroscience, 2022
  - https://www.nature.com/articles/s41583-022-00620-6
- Hippocampal ensembles represent sequential relationships among an extended sequence of nonspatial events
  - Nature Communications, 2022
  - https://www.nature.com/articles/s41467-022-28057-6

这些工作支持一个关键动机：

脑中的学习可能不是“每层精确反传 loss”，而是：

```text
局部资格迹 + 全局调制 + replay/巩固 + 时序状态图
```

---

## 6. 本轮选择的研究 idea

本轮选择：

**把 Direct Feedback Alignment 重写为三因子局部学习规则，并和 BP、零阶优化、纯 STDP、固定随机隐层读出进行比较。**

核心假设：

```text
局部资格迹 x 全局/随机反馈调制 可以接近 BP 的任务学习能力，
同时比纯 STDP 和零阶优化更有效。
```

这不是最终答案，但它是一个可验证的中间目标：

- 如果 DFA/三因子远弱于 BP，则方向不够好
- 如果只和 output-only 差不多，则隐层局部更新没有贡献
- 如果明显强于 STDP/SPSA，并接近 BP，则值得继续加 replay、时序任务和神经科学机制

