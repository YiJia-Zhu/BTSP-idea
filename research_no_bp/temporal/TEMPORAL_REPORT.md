# TEMPORAL_REPORT: no-BP 规则能否学习 next-token 时序记忆？

日期：2026-06-03

## 1. 为什么要从 MNIST 转向时序任务

MNIST 分类只能说明：

```text
x -> y
```

但你的核心问题更接近：

```text
A -> hidden_state_A -> hidden_state_B -> B
```

LLM 的训练目标也是 next-token prediction：

```text
x_1, x_2, ..., x_t -> x_{t+1}
```

所以第二轮实验改成时序任务，而不是继续做静态分类。

---

## 2. 任务设计

构造一个 delayed cue-to-target next-token 任务：

```text
C_k, F, F, ..., F, T_k, SEP
```

其中：

- `C_k` 是 cue，有 4 种：`C0, C1, C2, C3`
- `F` 是 filler，所有 episode 中都一样
- `T_k` 是 cue 对应的 target：`T0, T1, T2, T3`
- `SEP` 是结束符

关键测试点：

```text
在最后一个 F 处预测下一个 token T_k
```

这一步不能靠 bigram 完成，因为所有 filler token 都长一样。模型必须把最早的 cue 存进隐藏状态，隔多个时间步后再读出来。

---

## 3. 方法

比较方法：

| method | 说明 |
|---|---|
| bigram | 只统计当前 token 到下一个 token |
| reservoir | 固定随机 RNN，只训练输出读出层 |
| eprop_3factor | 在线三因子 eligibility trace，无 BPTT |
| eprop_resampled | 每步重采样反馈矩阵的破坏性对照 |
| bptt_rnn | 标准 RNN + BPTT |

主方法 `eprop_3factor`：

```text
h_t = tanh(W_in x_t + W_rec h_{t-1})
e_out = y_hat_t - y_t
m_h = B e_out
eligibility <- decay * eligibility + local_derivative * pre_activity
delta_W <- - eta * m_h * eligibility
```

这不是 BPTT，因为：

- 不把误差沿时间反传
- 不使用 `W_rec.T` 传播未来梯度
- 每一步只用当前输出误差、局部 eligibility trace 和固定反馈矩阵

---

## 4. 快速实验：任务太简单时 reservoir 也能解

设置：

- delay=5
- hidden_dim=48
- train_episodes=1000
- epochs=6

结果：

| method | target accuracy |
|---|---:|
| bigram | 0.0 |
| reservoir | 1.0 |
| eprop_3factor | 1.0 |
| eprop_resampled | 0.0 |
| bptt_rnn | 1.0 |

解释：

这个任务对随机 reservoir 太容易。固定随机动态系统本身已经能把 cue 保留到目标位置，只训练读出层就够了。

这不是坏事，它说明：

```text
reservoir / ESN 是实现 tiny language model 的重要 no-BP 路线
```

但它不能证明三因子隐层更新有必要。

---

## 5. Hard setting：低维长延迟

为了测试隐层训练是否必要，降低隐藏维度并拉长延迟：

- delay=12
- hidden_dim=12
- train_episodes=2000
- test_episodes=800
- seeds=0,1,2

主结果：

| method | overall acc | target acc |
|---|---:|---:|
| bigram | 0.9286 | 0.0000 |
| reservoir | 0.9286 | 0.0000 |
| eprop_resampled | 0.8868 | 0.0000 |
| eprop_3factor | 1.0000 | 1.0000 |
| bptt_rnn, default lr | 0.9710 | 0.5942 |
| bptt_rnn, tuned lr | 1.0000 | 1.0000 |

关键结论：

1. bigram 的 overall 很高但 target=0，因为它只会预测 filler/SEP，不能记 cue
2. reservoir 在低维长延迟下 target=0，固定随机动态系统容量不够
3. resampled feedback target=0，说明不是随机调制噪声就能学
4. eprop_3factor target=1，说明固定反馈 + eligibility trace 能训练隐藏状态保留 cue
5. 调好参的 BPTT 也 target=1，所以结论不是 no-BP 超越 BPTT，而是 no-BP 可以在该时序任务上达到同样功能

---

## 6. 这和 LLM 有什么关系

这个任务不是 LLM，但它和 LLM 共享一个核心形式：

```text
next-token prediction
```

区别是：

- LLM 是大规模 Transformer
- 本实验是 tiny recurrent next-token model

本实验能证明的只是一个机制级雏形：

```text
不用 BPTT，也可以用局部 eligibility trace + 固定反馈调制训练出跨时间隐藏状态
```

这对 LLM 的启发是：

### 路线 1: Reservoir / ESN-like language model

固定大动态系统，只训练读出层或少量 adapter。

优点：

- 完全兼容 next-token 数据
- 训练成本低
- 不需要 BP 穿过主体

缺点：

- 容量和可扩展性可能受限
- hard setting 中固定 reservoir 会失败

### 路线 2: Recurrent / SSM / RWKV / Mamba 上的三因子 eligibility

这比 Transformer 更适合 no-BP，因为它天然有状态：

```text
h_t -> h_{t+1}
```

可以把 eligibility trace 放在状态更新权重上。

优点：

- 与时序联想目标一致
- 比 Transformer 的全局 attention 更接近神经动态系统

缺点：

- 需要解决长序列稳定性、归一化和大规模并行训练问题

### 路线 3: 兼容已有 LLM 的 no-BP 记忆层

最现实的路线可能不是完全替代 LLM 预训练，而是：

```text
冻结已有 LLM
外挂 no-BP associative memory / recurrent adapter
用三因子 eligibility 在线更新 adapter
```

这样可以避免一开始就挑战完整 Transformer BP 预训练。

---

## 7. 和人脑时序联想的关系

这个 hard setting 对应一个简化脑机制：

```text
C_k 激活隐状态 h_k
filler 期间 h_k 被保持
最后 h_k 触发 T_k
```

换成你的问题：

```text
看到 A -> 激活 h_A
经过一段时间/上下文 -> h_A 仍影响当前状态
于是想到 B
```

三因子规则的意义：

- eligibility trace 负责“刚才哪些连接参与了”
- 固定反馈调制负责“这次预测错在哪里”
- recurrent state 负责“把过去带到现在”

这比纯 STDP 更接近任务学习，比 BP 更接近在线局部学习。

---

## 8. 下一步实验

下一轮最应该做三件事：

1. **多 token 组合任务**
   - 目标依赖两个 cue，例如 `T_(a+b mod 4)`
   - 测试组合泛化

2. **变长 delay 任务**
   - train delay=4..12
   - test delay=16/20
   - 测试是否只是记住固定时间位置

3. **外挂 LLM adapter 原型**
   - 不训练完整 LLM
   - 用字符级 tiny transformer/RNN 或 frozen embedding
   - 只用三因子规则训练 recurrent memory adapter

---

## 9. 文件

代码：

- `research_no_bp/temporal_sequence_experiment.py`

结果：

- `research_no_bp/temporal/results/delayed_quick_v1`
- `research_no_bp/temporal/results/delayed_hard_v1`
- `research_no_bp/temporal/results/bptt_tuned_v2`

图：

- `research_no_bp/temporal/results/delayed_hard_v1/temporal_summary.png`
- `research_no_bp/temporal/results/bptt_tuned_v2/temporal_summary.png`

