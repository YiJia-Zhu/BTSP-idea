# Iteration Report: Local API-Compatible Online Memory QA Prototype

日期：2026-06-15

## 1. 目的

本轮做 `R019` 的本地版，不调用付费 API：

```text
冻结/API base model + no-BP online memory adapter
```

目标是验证 adapter 形态是否能在小规模 personalization / FAQ 场景中工作：

- base system 没有新用户事实；
- online memory 逐条吸收事实；
- memory state 不保存原始问句/答案文本；
- 用 paraphrased question 测试新事实召回。

## 2. 新增实现

新增脚本：

```text
online_memory_qa_experiment.py
```

比较：

| method | 说明 |
|---|---|
| `base_no_memory` | 无记忆固定猜测 |
| `hashed_memory` | hashed sparse features -> answer-count memory，不保存原文 |
| `raw_retrieval` | 保存原始文本的检索上界，违反 no-raw-data 约束 |

`hashed_memory` 存储：

- hashed n-gram feature ids；
- normalized name/attribute feature ids；
- gated name+attribute fact feature ids；
- answer id 计数。

不存储：

- raw question text；
- raw statement text；
- raw answer text。

## 3. 结果

### Smoke

命令：

```bash
python online_memory_qa_experiment.py \
  --out-dir output/online_memory_qa_smoke \
  --num-people 8 --eval-every 5 --seed 0
```

| method | accuracy | state bytes | raw text stored |
|---|---:|---:|---|
| base_no_memory | 0.025 | 0 | no |
| hashed_memory | 1.000 | 7,102 | no |
| raw_retrieval | 0.800 | 3,928 | yes |

### Medium

命令：

```bash
python online_memory_qa_experiment.py \
  --out-dir output/online_memory_qa_medium \
  --num-people 12 --eval-every 10 --seed 1
```

| method | accuracy | state bytes | raw text stored |
|---|---:|---:|---|
| base_no_memory | 0.017 | 0 | no |
| hashed_memory | 1.000 | 9,276 | no |
| raw_retrieval | 0.800 | 5,820 | yes |

### Deletion audit

Smoke:

| split | before forget acc | after forget acc |
|---|---:|---:|
| deleted facts | 1.000 | 0.100 |
| retained facts | 1.000 | 1.000 |

Medium:

| split | before forget acc | after forget acc |
|---|---:|---:|
| deleted facts | 1.000 | 0.200 |
| retained facts | 1.000 | 1.000 |

The state byte count serializes only learned hashed feature tables and answer-count arrays. The fixed answer schema / parser is treated as external adapter code, not online learned memory.

## 4. 结论

这是 API-compatible prototype 的本地强正信号：

```text
hashed no-raw memory 在 schema personalization QA 上显著超过无记忆 base，
并且通过 name+attribute gated feature 超过保存原文的朴素 lexical retrieval。
```

但仍需保持边界：

- 当前数据是合成 schema QA，不是自然客服/写作任务；
- 没有调用 GPT/API，所以不能宣称接近 GPT 输出；
- hashed state 仍可能有隐私风险，需要后续做 membership audit 和真实删除测试。

## 5. 下一步

1. 增强 hashed memory 的泛化：多模板 attribute parser、embedding key、本地 frozen representation。
2. 做 retention / deletion：删除某个用户事实后能否忘记。
3. 小预算 API demo：只在本地原型稳定后，用少量个性化事实比较 API no-memory vs API+hashed-memory adapter。
