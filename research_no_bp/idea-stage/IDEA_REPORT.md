# IDEA_REPORT: 替代反向传播的类脑学习算法

日期：2026-06-03

## 目标

探索一种替代 BP 的学习算法，偏好类脑机制。第一轮不追求 SOTA，而追求一个可验证的最小命题：

> 不使用反向传播穿过隐层，是否仍能通过局部资格迹和调制信号学到接近 BP 的表征？

---

## 候选 idea

### Idea 1: 三因子 Direct Feedback Alignment

核心规则：

```text
h = relu(W1 x + b1)
y_hat = softmax(W2 h + b2)
e_out = y_hat - y
m_h = B e_out
delta_W1 = - eta * (m_h * relu_gate) outer x
delta_W2 = - eta * e_out outer h
```

其中：

- `B` 是固定随机反馈矩阵
- `m_h` 是每个隐藏神经元接收到的调制信号
- `x` 和 `relu_gate` 是局部可获得的资格迹
- 隐层更新不使用 `W2.T`
- 不做逐层链式误差反传

为什么像类脑：

- 突触更新只需要 pre、post gate、调制信号
- 调制信号可以类比皮层反馈、奖励误差、多巴胺样误差信号
- 避免 BP 的 weight transport 问题

为什么值得做：

- 理论上比纯 STDP 更能利用任务信号
- 工程上比零阶优化方差低
- 可直接和 BP 做同构网络对比

风险：

- 它仍然需要输出误差信号，所以不是纯无监督
- 单隐层 MNIST 成功不代表深层网络或时序任务成功

状态：

- **选为本轮主 idea**

---

### Idea 2: 零阶 SPSA / ES 作为 BP 替代

规则：

```text
g_hat = (L(theta + eps u) - L(theta - eps u)) / (2 eps) * u
theta <- theta - eta g_hat
```

优点：

- 完全不用梯度
- 能处理黑盒目标

缺点：

- 参数维度高时方差大
- 不够类脑
- 不解释神经元局部可塑性

状态：

- 保留为负对照

---

### Idea 3: 纯 STDP + label voting

规则：

```text
pre before post -> strengthen
post before pre -> weaken
```

优点：

- 生物动机最直接
- 能学习局部特征和时序关联

缺点：

- 单独做监督分类很弱
- 需要竞争、homeostasis、readout、调制信号才能稳定完成任务

状态：

- 保留为局部无监督下界

---

### Idea 4: Predictive Coding / Equilibrium Propagation

优点：

- 理论上可把误差传播转成局部动态推断

缺点：

- 实验复杂度高
- 很多实现仍然等价或近似梯度优化
- 第一轮不适合直接作为主线

状态：

- 第二阶段再考虑

---

## 自动选择

根据调研和 pilot 可验证性，本轮自动选择：

**Idea 1: 三因子 Direct Feedback Alignment**

选择理由：

- 与 STDP / Hebbian 的三因子形式一致
- 与 BP 有同构网络可比性
- 可以定量比较：
  - BP 上界
  - output-only 随机特征对照
  - STDP 局部无监督下界
  - SPSA 完全无梯度对照

---

## 实验假设

### H1: 接近 BP

在小规模 MNIST 上，三因子 DFA 的准确率应接近同结构 BP。

判据：

```text
acc(dfa_3factor) >= acc(bp) - 0.03
```

### H2: 强于固定随机隐层

如果三因子隐层更新有意义，它应强于只训练输出层的随机特征模型。

判据：

```text
acc(dfa_3factor) > acc(output_only) + 0.02
```

### H3: 强于纯 STDP 和零阶优化

三因子 DFA 应强于纯 STDP 和 SPSA。

判据：

```text
acc(dfa_3factor) > acc(stdp)
acc(dfa_3factor) > acc(zo_spsa)
```

---

## 实现

脚本：

- `research_no_bp/no_bp_mnist_experiment.py`

输出：

- `research_no_bp/results/pilot_v1`
- `research_no_bp/results/full_v1`
- `research_no_bp/results/full_v2`

