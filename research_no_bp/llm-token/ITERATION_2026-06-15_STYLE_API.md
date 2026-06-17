# Iteration 2026-06-15: Personalized Style API Benchmark

## 目的

FAQ 实验验证了事实召回。本轮推进到更接近 GPT/API 输出质量的自然写作任务：

```text
online style preference -> compact no-raw sketch memory -> API customer-facing draft
```

每个 profile 有格式、必含短语、签名、禁用词、长度上限。API 需要生成自然短文，同时满足这些在线学习到的偏好。

## 代码

- `../online_memory_style_api_experiment.py`

比较：

| method | 说明 |
|---|---|
| `api_no_memory` | API 只看写作任务，不看用户偏好 |
| `api_raw_profile` | API 看 raw style profile 文本，上界 baseline |
| `api_style_sketch_memory` | API 看由 no-raw sketch memory 渲染出的 style hint |

`local_style_sketch_memory` 只检查 memory 是否携带正确约束，不代表生成模型。

## 命令

```bash
python -m py_compile online_memory_style_api_experiment.py online_memory_faq_api_experiment.py

python online_memory_style_api_experiment.py \
  --out-dir output/online_memory_style_dry \
  --api-limit 0

API_KEY=... python online_memory_style_api_experiment.py \
  --out-dir output/online_memory_style_delete_api \
  --api-limit 9 --run-api
```

## 结果

| method | all-pass acc | format | signoff | required | avoid | length | state bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_style_sketch_memory | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6,135 |
| api_no_memory | 0.000 | 0.000 | 0.000 | 0.000 | 0.750 | 0.750 | 0 |
| api_raw_profile | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1,021 |
| api_style_sketch_memory | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6,135 |
| local_deleted_suppression | 1.000 | - | - | - | - | - | 6,135 |
| api_deleted_suppression | 1.000 | - | - | - | - | - | 6,135 |

The latest API run covers 8 active generation prompts and 1 deleted-profile query. The session includes revise/delete/retention checks:

- revised profiles update to new format/signoff/required phrase;
- deleted profile returns no hint;
- deleted-profile API query receives no sketch hint;
- retained profiles remain queryable after delete.

## 判断

This is a stronger API-side result than FAQ recall: the API generates natural customer-facing text and only satisfies the full style constraint set when it receives either raw profile text or the no-raw sketch hint. The latest run also verifies deletion in the same natural-generation setting: after `profile_000_mira` is forgotten, the sketch-memory path sends no profile hint.

The current sketch memory still stores compact symbolic preference values. It does not store raw profile/instruction text, but the rendered hint is explicit at inference time. This is acceptable for the adapter prototype, but not yet a proof of open-domain GPT-level online learning.

## 下一步

1. Expand style benchmark to more profiles and more prompts per profile.
2. Add semantic/no-alias routing for style profiles.
3. Add a small preference-judge pass or human review for naturalness beyond rule compliance.
