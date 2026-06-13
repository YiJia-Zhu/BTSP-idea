# AUTO_REVIEW: 第一轮 no-BP 类脑学习实验审查

日期：2026-06-03

## 被审查对象

- 方法：三因子 Direct Feedback Alignment
- 脚本：`research_no_bp/no_bp_mnist_experiment.py`
- 主结果：`research_no_bp/results/full_v3/results.json`

## 结果摘要

设置：

- 数据：MNIST
- 训练样本：5000
- 测试样本：2000
- 图像：14x14，下采样自 28x28
- seeds：0, 1, 2
- 网络：单隐层 MLP，hidden_dim=128
- 主训练轮数：12 epochs

结果：

| method | mean test acc | std |
|---|---:|---:|
| BP | 0.9128 | 0.0072 |
| dfa_3factor | 0.9042 | 0.0099 |
| output_only | 0.8245 | 0.0150 |
| dfa_resampled | 0.7970 | 0.0215 |
| STDP | 0.2865 | 0.0349 |
| SPSA zero-order | 0.1688 | 0.0088 |

## Claim 审查

### Claim 1: 三因子 DFA 接近 BP

支持程度：强

证据：

```text
BP          = 0.9128 +/- 0.0072
dfa_3factor = 0.9042 +/- 0.0099
gap         = 0.0087
```

gap 小于预设阈值 0.03。

### Claim 2: 隐层局部更新有贡献

支持程度：强

证据：

```text
dfa_3factor = 0.9042
output_only = 0.8245
gain        = 0.0797
```

说明结果不是单纯随机特征 + 输出层读出造成的。

### Claim 3: 固定反馈通道是关键

支持程度：中到强

证据：

```text
dfa_3factor   = 0.9042
dfa_resampled = 0.7970
gain          = 0.1072
```

每个 batch 重采样反馈矩阵后性能显著下降，说明稳定反馈通道有助于网络形成对齐。

### Claim 4: 三因子机制优于纯 STDP 和零阶优化

支持程度：强，但限定在本实验设置

证据：

```text
dfa_3factor = 0.9042
STDP        = 0.2865
SPSA        = 0.1688
```

## 主要问题

### 问题 1: 这还不是完全无监督的脑模型

三因子 DFA 仍然使用 label 产生的输出误差信号。它不是纯 STDP，也不是人脑的完整学习机制。

修正：

报告中必须表述为：

```text
替代 BP 的局部监督/调制学习规则
```

不要表述为：

```text
完全解释人脑学习
```

### 问题 2: MNIST 太简单

单隐层 MNIST 只能证明学习规则可行，不能说明可扩展到深层网络、LLM 或真实时序认知。

下一步：

- 两层隐藏网络
- Permuted MNIST / sequential MNIST
- 时序关联任务
- replay / eligibility trace 消融

### 问题 3: STDP baseline 偏弱

当前 STDP 是教育型最小实现，不是优化过的 SNN 分类器。

修正：

结果中只能说：

```text
优于当前 minimal STDP baseline
```

不能说：

```text
优于所有 STDP/SNN 方法
```

### 问题 4: 零阶优化 baseline 不充分

SPSA 在高维网络上很弱，但 MeZO/ES 在其他设置可能更强。

修正：

将其定位为：

```text
朴素零阶负对照
```

不能泛化为：

```text
零阶优化不可行
```

## Review 评分

当前研究闭环评分：6.5 / 10

理由：

- 有清晰问题和类脑动机
- 有可复现实验
- 有关键对照和改进复跑
- 结果支持一个有限但清楚的 claim
- 还缺少时序任务和更强神经科学连接

## 已根据 review 完成的改进

1. 增加 `output_only` 对照，证明隐层局部更新有贡献
2. 增加 `dfa_resampled` 消融，证明固定反馈通道有贡献
3. 保存多 seed 结果、history、summary 和可视化图
4. 明确 STDP/SPSA 的 baseline 限制

## 下一轮建议

下一轮最应该做：

```text
三因子 DFA + eligibility trace + replay 的时序预测任务
```

目标不是继续刷 MNIST，而是贴近用户终极问题：

```text
看到 A 为什么想到 B？
```

具体实验：

1. 输入序列 `x1 -> x2 -> x3 -> x4`
2. 网络形成隐藏状态 `h1, h2, h3, h4`
3. 用三因子资格迹学习 `h_t -> h_{t+1}`
4. 加 replay 后测试远距离联想 `x1 -> x3 / x4`
5. 和纯 STDP、BP-BPTT、随机 replay 对照

