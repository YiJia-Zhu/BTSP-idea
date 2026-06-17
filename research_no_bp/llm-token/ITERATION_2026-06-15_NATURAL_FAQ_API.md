# Iteration Report: Natural FAQ API Demo with No-Raw-Example Memory

日期：2026-06-15

## 1. 目的

前一轮 schema QA 已经证明 API + hashed memory hint 可以召回个性化事实。本轮推进到更像客服/FAQ 的自然问答：

```text
online memory -> compressed FAQ hint -> API natural answer
```

目标是验证：

- API 不看原始训练句；
- memory 不保存原始训练样本；
- API 能根据 memory hint 生成自然回答；
- no-memory API 在同一问题上无法回答。

## 2. 新增实现

新增脚本：

```text
online_memory_faq_api_experiment.py
```

比较：

| method | 说明 |
|---|---|
| `local_hashed_faq_memory` | 本地 FAQ intent memory，只存 hashed features、answer values 和 counts |
| `api_no_memory` | API 只看问题，无 memory hint |
| `api_memory_hint` | API 看问题和本地 memory top hint |

注意：本轮不保存 raw training examples，但 memory state 保存 canonical answer values。这比保存原始问答样本更可控，但还不是强隐私保证。

## 3. Dry-run

命令：

```bash
python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_api_dry \
  --fact-limit 8
```

结果：

| method | accuracy | state bytes |
|---|---:|---:|
| local_hashed_faq_memory | 1.000 | 4,726 |

## 4. Real API Run

命令：

```bash
API_KEY=... python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_api_run \
  --fact-limit 8 --api-limit 8 --run-api
```

结果：

| method | accuracy | state bytes |
|---|---:|---:|
| local_hashed_faq_memory | 1.000 | 4,726 |
| api_no_memory | 0.000 | 0 |
| api_memory_hint | 1.000 | 4,726 |

代表回答：

| question | api_no_memory | api_memory_hint |
|---|---|---|
| What is the return window after delivery? | I do not know. | Unused items can be returned within 45 days of delivery. |
| What email handles billing support? | I do not know. | Billing questions should be sent to billing-help@example.test. |
| What is the cancellation deadline before an appointment? | I do not know. | Appointments can be cancelled up to 24 hours before the start time. |

## 5. 判断

这是 R021 的部分正结果：

```text
真实 API 可以作为 frozen language layer，
本地 no-BP memory 提供现场事实 hint，
从而在不保存原始训练样本的情况下完成自然 FAQ 回答。
```

仍不能宣布最终目标完成：

- FAQ 集合只有 8 条；
- memory 保存 canonical answer values；
- 还没有开放域自然对话、写作风格、长上下文生成；
- 还没有人工偏好评估或强 baseline 大样本对照。

## 6. 下一步

1. 扩展 FAQ 到 50-100 条，多 paraphrase。
2. 增加 answer-value privacy audit：删除 answer value 后 API 不能恢复。
3. 做开放式 held-out prompts 的 qualitative comparison：API no-memory vs API+memory vs raw retrieval。

