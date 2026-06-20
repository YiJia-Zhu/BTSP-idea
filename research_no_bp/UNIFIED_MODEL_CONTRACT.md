# Unified No-BP Model Contract

日期：2026-06-19

## 目标

最终目标是构建一个**纯 no-BP 的仿生物神经学习模型**。交付物应当是一个统一模型框架，而不是面向某个数据集或指标堆出来的模块集合。

模型必须从随机或显式初始化开始，通过局部、在线、神经网络类学习规则更新。预训练 LLM、冻结大模型、API 主干、BP 训练模型只能作为对照、接口或工程上界，不能作为主方法。

## 统一接口

主线模型应暴露任务无关接口：

- `observe(x, target=None, modulatory_signal=None)`：在线观察输入并执行 no-BP 更新。
- `predict_next(context)`：给出下一 token / 下一状态分布或分数。
- `generate(context, max_steps)`：用同一预测机制生成序列。
- `state_dict()` / `load_state_dict(...)`：保存和恢复模型状态，不保存原始训练样本。

不同任务只能通过 adapter 完成：

- 文本续写 adapter：把文本转成 token stream，并用 next-token CE/accuracy/sample 评价。
- QA adapter：把故事和问题转成同一 token/event stream，并用答案准确率评价。
- 时序联想 adapter：把符号序列转成同一事件流，并用目标预测评价。

adapter 可以处理数据格式和指标，但不能把任务结构塞进模型核心。

## 允许机制

允许作为统一模型核心的机制包括：

- recurrent / working state；
- STDP、BTSP、Hebbian/Oja 类局部可塑性；
- eligibility trace 和三因子调制学习；
- dendritic / apical / 双室误差信号；
- feedback alignment 或随机反馈调制；
- 局部竞争、WTA、抑制性回路、稳态调节；
- 通用记忆压缩或稀疏激活，只要它是神经状态/突触结构，而不是任务答案表。

这些机制必须尽量少而清晰。若性能问题可以通过调整核心状态表示、核心局部更新、核心竞争/抑制或核心调制规则解决，优先修改这些核心机制，不新增任务专用模块。

## 禁止作为模型核心的机制

以下内容不能作为最终模型核心：

- 预训练 LLM / 冻结大模型 / API 模型主干；
- BP、BPTT 或通过 BP 预训练得到的主干；
- n-gram、Kneser-Ney、continuation backoff、纯统计 token 表作为最终方法；
- QA-specific answer slots；
- edge-path parser、candidate path graph、手写关系跳转；
- 数据集专用 controller、router、gate、oracle、candidate rescoring；
- 针对某个 benchmark 失败样例临时添加的补丁模块；
- 存储原始训练样本并检索答案的方案。

这些机制可以作为 diagnostic、baseline 或上界，但报告必须明确标记为“not unified model core”。

## 实验准入检查

一个主线实验开始前必须回答：

1. 同一模型核心是否能不改结构地跑至少两个任务？
2. 新增部分是否是核心学习规则的一部分，而不是任务 adapter 内的补丁？
3. 这个规则能否用局部神经机制解释？
4. 是否减少或保持复杂度，而不是继续堆模块？
5. 是否避免使用数据集标签结构作为内部状态？
6. 统计方法是否只用于 baseline / sanity check / diagnostic？

如果任一答案不满足，实验只能作为诊断实验，不能命名为 `Uxxx`，也不能作为最终模型路线。

## 旧结果解释原则

`Rxxx` 记录中包含大量探索性和数据集特异诊断，尤其是 bAbI/QA19 answer-slot、edge-path、candidate-arbiter、homeostasis gate 等实验。这些结果可以帮助理解失败模式，但不能直接决定最终架构。

从现在起，最终路线只接受通过本契约的 `Uxxx` 统一模型实验。
