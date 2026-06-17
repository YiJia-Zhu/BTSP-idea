# Iteration Report: Real API Call for No-Raw-Data QA Adapter

日期：2026-06-15

## 1. 目的

本轮把本地 `API-compatible` 原型接到真实 OpenAI-compatible endpoint，验证调用链和提示设计是否闭合。

目标不是做大规模评测，只做最小可验证调用：

- base model no memory
- base model + hashed memory hint

## 2. 实现

新增脚本：

```text
online_memory_qa_api_experiment.py
```

特性：

- 默认 dry-run，不发送请求；
- `--run-api` 才调用真实 endpoint；
- 通过 `https://yzhanghmeng.com/v1/chat/completions`；
- 模型：`gpt-5.5`；
- 只发送结构化 question / attribute / candidate values / memory hint；
- 本地 memory 仍只保存哈希特征与计数，不保存原始问答文本。

## 3. Dry-run

默认模式会写：

- `api_requests.jsonl`
- `api_eval.csv`
- `summary.csv`

Dry-run 主要用于确认 request payload 格式和输出文件完整性。

## 4. Real API 小样本

命令：

```bash
API_KEY=... python online_memory_qa_api_experiment.py \
  --out-dir output/online_memory_qa_api_run \
  --num-people 8 --seed 0 --api-limit 3 --run-api
```

结果：

| method | accuracy | note |
|---|---:|---|
| local_base_no_memory | 0.025 | 本地固定猜测 |
| local_hashed_memory | 0.750 | 本地 no-raw adapter |
| local_raw_retrieval | 0.800 | raw upper bound |
| api_no_memory | 0.000 | 3/3 错 |
| api_memory_hint | 1.000 | 3/3 对 |

`api_memory_hint` 的三条样本里，API 返回的答案与 memory hint 一致，说明 prompt + hint 结构有效。

## 5. Real API 10-question 对照

改进本地 memory gate 后，新增：

```text
如果 name+attribute fact feature 存在，只使用该局部事实行；
否则才退回全局 n-gram / 属性统计。
```

命令：

```bash
API_KEY=... python online_memory_qa_api_experiment.py \
  --out-dir output/online_memory_qa_api_run10 \
  --num-people 8 --seed 0 --api-limit 10 --run-api
```

结果：

| method | accuracy | note |
|---|---:|---|
| local_base_no_memory | 0.025 | 本地固定猜测 |
| local_hashed_memory | 1.000 | 本地 no-raw adapter |
| local_raw_retrieval | 0.800 | raw lexical retrieval baseline |
| api_no_memory | 0.200 | 2/10 对，主要是猜中常见值 |
| api_memory_hint | 1.000 | 10/10 对 |

错因检查：

- `api_memory_hint` 无错例；
- `api_no_memory` 正确的两题是 lucky guess；
- 所有 API 记忆答案都来自本地 hashed memory hint。

## 6. 证据边界

这是一个真实 API 小样本强正信号，但还不是最终目标完成：

- 样本数仍只有 10；
- 任务仍是合成 schema QA，不是自然对话或长文本生成；
- 仍未对比更强 frozen backbone 的大样本长期表现；
- 没有做 token-level/生成质量的人眼评估。

## 7. 下一步

1. 扩大到更大一点的 synthetic QA / personalization 样本；
2. 做 API no-memory vs API+hashed-memory 的稳定对照；
3. 再把同一套路迁移到更像 FAQ / customer support 的自然问答；
4. 最终才接 `R021` 的 GPT-like qualitative evaluation。
