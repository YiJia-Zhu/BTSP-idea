# NARRATIVE_REPORT: 三因子局部学习作为 BP 替代候选

日期：2026-06-03

## 1. 核心问题

我们想找一种替代反向传播的学习算法，并且偏向类脑机制。

第一轮研究把问题收敛为：

> 一个不通过 `W2.T` 反传误差、不执行逐层 BP 的局部三因子学习规则，能否在简单视觉分类上接近 BP？

这里的三因子是：

```text
突触前活动 x 突触后资格迹 x 调制信号
```

这和 STDP/Hebbian 的局部可塑性兼容，但比纯 STDP 多了一个任务相关调制项。

---

## 2. 方法

使用单隐层网络：

```text
x -> h -> y_hat
```

BP 更新隐层时需要：

```text
delta_h = (y_hat - y) @ W2
```

三因子 DFA 不使用 `W2` 的转置或链式反传，而使用固定随机反馈矩阵 `B`：

```text
e_out = y_hat - y
m_h = e_out @ B.T
delta_W1 = - eta * (m_h * relu_gate) outer x
delta_W2 = - eta * e_out outer h
```

解释：

- `x` 是突触前输入
- `relu_gate` 是突触后资格迹
- `m_h` 是隐藏神经元局部接收的调制信号
- `B` 是固定反馈通道

因此它不是标准 BP，但也不是纯无监督 STDP。更准确的定位是：

```text
局部监督/调制学习规则
```

---

## 3. 对照方法

实验包含六个方法：

| method | 作用 |
|---|---|
| BP | 同结构反向传播上界 |
| dfa_3factor | 主方法 |
| output_only | 固定随机隐层，只训练输出层 |
| dfa_resampled | 每个 batch 重采样反馈矩阵 |
| STDP | 最小无监督局部 STDP baseline |
| zo_spsa | 朴素零阶优化负对照 |

关键设计：

- 如果 `dfa_3factor` 只比 BP 差很少，说明它有替代 BP 的潜力
- 如果 `dfa_3factor` 高于 `output_only`，说明隐层局部更新有贡献
- 如果 `dfa_3factor` 高于 `dfa_resampled`，说明固定反馈通道有意义

---

## 4. 实验设置

脚本：

- `research_no_bp/no_bp_mnist_experiment.py`

主结果：

- `research_no_bp/results/full_v3/results.json`
- `research_no_bp/results/full_v3/summary.csv`
- `research_no_bp/results/full_v3/summary.png`

设置：

- MNIST
- 训练样本：5000
- 测试样本：2000
- 图像下采样：28x28 -> 14x14
- seeds：0, 1, 2
- BP/DFA/output-only：hidden_dim=128
- 训练轮数：12

---

## 5. 结果

| method | mean test acc | std |
|---|---:|---:|
| BP | 0.9128 | 0.0072 |
| dfa_3factor | 0.9042 | 0.0099 |
| output_only | 0.8245 | 0.0150 |
| dfa_resampled | 0.7970 | 0.0215 |
| STDP | 0.2865 | 0.0349 |
| SPSA zero-order | 0.1688 | 0.0088 |

结论：

1. `dfa_3factor` 与 BP 的差距约 `0.87%`
2. `dfa_3factor` 比 `output_only` 高约 `7.97%`
3. `dfa_3factor` 比 `dfa_resampled` 高约 `10.72%`
4. 当前 minimal STDP 和朴素 SPSA 明显不适合作为直接分类主方法

---

## 6. 这证明了什么

本轮结果支持：

> 在简单视觉分类上，固定随机反馈调制的三因子局部学习规则可以接近 BP，并且隐层局部更新确实贡献了性能。

更具体地说：

```text
pre/post eligibility + stable modulatory feedback
```

比下面两种都更有效：

```text
只训练输出层的随机特征读出
每步重采样的随机反馈噪声
```

这说明“固定反馈通道”可能是从局部可塑性走向任务学习的关键条件。

---

## 7. 没有证明什么

本轮结果不证明：

- 人脑就是这样学习
- STDP 本身足以替代 BP
- 零阶优化整体不可行
- 该方法能扩展到 LLM
- 该方法已经能解释复杂时序联想

这些都需要下一轮实验。

---

## 8. 与神经科学的连接

本轮方法和神经科学的连接点是：

1. STDP/Hebbian 提供局部资格迹
2. 多巴胺、皮层反馈、注意等信号可以作为调制项
3. 固定反馈通道可能允许网络形成反馈对齐
4. replay 可以把短期资格迹巩固成长期时序联想

关键文献：

- Random synaptic feedback weights support error backpropagation for deep learning
  - https://www.nature.com/articles/ncomms13276
- e-prop: A biologically plausible learning rule for recurrent neural networks
  - https://www.nature.com/articles/s41467-020-17236-y
- Engram mechanisms of memory linking and identity
  - https://www.nature.com/articles/s41583-024-00814-0
- Replay, the default mode network and the cascaded memory systems model
  - https://www.nature.com/articles/s41583-022-00620-6

---

## 9. 下一步最有价值的实验

不要继续只刷 MNIST。下一轮应转向用户的核心问题：

> 为什么看到 A 会想到 B？

建议实验：

```text
三因子 DFA + eligibility trace + replay 的时序联想任务
```

最小设计：

1. 输入 `x1 -> x2 -> x3 -> x4`
2. 网络先学 `x -> h`
3. 用资格迹学习 `h_t -> h_{t+1}`
4. 加 replay 后测试 `x1` 是否能唤起 `h2/h3/h4`
5. 加冲突 cue，测试当前输入和时间先验如何竞争

这会把本轮“分类学习”推进到真正的“时序联想”。

