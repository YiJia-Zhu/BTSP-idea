# Iteration 2026-06-15: Multi-turn FAQ Session Demo

## 目的

前面的 FAQ API 结果已经证明了单轮问答、修订、删除、semantic routing 和 answer sketch。  
本轮把这些能力串成一个真实的多轮支持会话：

1. learn 新事实
2. query 事实
3. revise 事实
4. 再 query
5. delete 部分事实
6. 再 query deleted / retained facts

目标不是开放域 GPT 评测，而是验证一个更像真实客服交互的在线学习闭环。

## 代码

- `../online_memory_faq_multiturn_experiment.py`
- 复用 `../online_memory_faq_api_experiment.py` 的事实生成、FAQ memory、API client、correctness checker

比较的模式：

| method | 说明 |
|---|---|
| `local_raw_retrieval` | 本地 raw answer baseline，保留当前 raw answer 文本，仅作上界对照 |
| `local_semantic_sketch_memory` | sparse-only semantic router + structured answer sketch |
| `api_no_memory` | API 只看问题 |
| `api_raw_retrieval` | API 看问题和 raw retrieval hint |
| `api_semantic_sketch_memory` | API 看问题和 semantic sketch hint |

## 命令

```bash
python -m py_compile online_memory_faq_api_experiment.py online_memory_faq_multiturn_experiment.py

python online_memory_faq_multiturn_experiment.py \
  --out-dir output/online_memory_faq_multiturn_dry \
  --fact-limit 16 --session-api-limit 0

API_KEY=... python online_memory_faq_multiturn_experiment.py \
  --out-dir output/online_memory_faq_multiturn_api \
  --fact-limit 16 --session-api-limit 10 --run-api
```

## 结果

| method | accuracy | query_count | state bytes |
|---|---:|---:|---:|
| local_raw_retrieval | 1.000 | 14 | 1,812 |
| local_semantic_sketch_memory | 1.000 | 14 | 13,556 |
| api_no_memory | 0.200 | 10 | 0 |
| api_raw_retrieval | 1.000 | 10 | 1,812 |
| api_semantic_sketch_memory | 1.000 | 10 | 13,556 |

### Transcript notes

- 修订前 query 全部正确。
- 修订后 query 全部正确。
- 删除后 query 对 deleted facts 返回 `I don’t know`。
- retained facts 在 delete 之后仍然正确。
- API no-memory 仅在 10 个 query 中命中 2 个，说明没有 hint 时仍然弱。

## 判断

这个 demo 把 FAQ 原型从单轮问答推进到多轮交互闭环，能展示：

- 在线 learn / revise / delete；
- no-memory API、raw retrieval、compressed memory 三方对照；
- human-readable transcript；
- no-raw-example 的 semantic sketch state 仍可稳定工作。

但它仍然是受控 FAQ 支持会话，不是开放域 GPT-like final evaluation。

