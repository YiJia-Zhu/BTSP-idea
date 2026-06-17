# Iteration 2026-06-16: Soft Style Hint And Preference-Aware Judge

## 目的

上一轮 blind judge 暴露了一个问题：style sketch 满足规则，但自然度/可用性没有稳定压过 no-memory API。  
本轮测试两个点：

1. 把 no-raw style sketch 渲染成更柔和的 style card；
2. 区分“普通写作质量盲评”和“知道用户已学习偏好后的个性化质量评审”。

## 代码

- `../online_memory_style_api_experiment.py`
  - 新增 `--hint-style strict|soft`
  - `soft` 模式把 compact payload 渲染为较自然的短 style card
- `../online_memory_style_judge_experiment.py`
  - 新增 `--judge-context request_only|style_memory|raw_profile`
  - `style_memory` 模式把 learned style sketch 作为评审上下文，但候选答案仍盲化为 A/B/C

## 命令

```bash
python -m py_compile online_memory_style_api_experiment.py online_memory_style_judge_experiment.py

python online_memory_style_api_experiment.py \
  --out-dir output/online_memory_style_soft_dry \
  --hint-style soft --api-limit 0

API_KEY=... python online_memory_style_api_experiment.py \
  --out-dir output/online_memory_style_soft_api \
  --hint-style soft --api-limit 9 --run-api

API_KEY=... python online_memory_style_judge_experiment.py \
  --source-dir output/online_memory_style_soft_api \
  --out-dir output/online_memory_style_soft_judge_api \
  --case-limit 8 --run-api

API_KEY=... python online_memory_style_judge_experiment.py \
  --source-dir output/online_memory_style_soft_api \
  --out-dir output/online_memory_style_soft_judge_context_api \
  --case-limit 8 --judge-context style_memory --run-api

API_KEY=... python online_memory_style_judge_experiment.py \
  --source-dir output/online_memory_style_delete_api \
  --out-dir output/online_memory_style_strict_judge_context_api \
  --case-limit 8 --judge-context style_memory --run-api
```

## 结果

Soft style hint rule compliance:

| method | all-pass acc | format | signoff | required | avoid | length | state bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| local_style_sketch_memory | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6,155 |
| api_no_memory | 0.000 | 0.000 | 0.000 | 0.000 | 0.625 | 0.500 | 0 |
| api_raw_profile | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1,021 |
| api_style_sketch_memory | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6,155 |
| api_deleted_suppression | 1.000 | - | - | - | - | - | 6,155 |

Blind request-only judge on soft outputs:

| metric | value |
|---|---:|
| cases | 8 |
| no_memory_best_rate | 1.000 |
| raw_profile_best_rate | 0.000 |
| style_sketch_best_rate | 0.000 |
| style_sketch_beats_no_memory_rate | 0.000 |
| style_sketch_beats_raw_profile_rate | 0.500 |

Preference-aware judge on soft outputs:

| metric | value |
|---|---:|
| cases | 8 |
| no_memory_best_rate | 0.000 |
| raw_profile_best_rate | 0.375 |
| style_sketch_best_rate | 0.625 |
| style_sketch_beats_no_memory_rate | 1.000 |
| style_sketch_beats_raw_profile_rate | 0.625 |

Preference-aware judge on previous strict outputs:

| metric | value |
|---|---:|
| cases | 8 |
| no_memory_best_rate | 0.000 |
| raw_profile_best_rate | 0.125 |
| style_sketch_best_rate | 0.875 |
| style_sketch_beats_no_memory_rate | 1.000 |
| style_sketch_beats_raw_profile_rate | 0.875 |

## 判断

This is a useful split result, not a simple win:

- Soft rendering preserved the no-raw memory property and kept API rule compliance at `1.000`.
- In request-only blind judging, no-memory won `8/8`. The judge rewarded general customer-support completeness and penalized the learned constraints because it did not know those constraints were part of the user preference.
- In preference-aware judging, soft style sketch was best on `5/8` and beat no-memory on `8/8`.
- Strict style sketch was even stronger in preference-aware judging: best on `7/8`, beating raw-profile on `7/8` and no-memory on `8/8`.

The immediate conclusion is that the benchmark needs two separated scores:

1. **Base naturalness** without personalization context.
2. **Personalized usefulness** with learned user preference context.

For the current generated style profiles, stricter hints are better than the new soft hint. The next engineering step is not to make hints softer by deleting constraints; it is to make constraints less awkward semantically, for example separating sender identity from signoff name and using customer-facing required fields that do not read like internal labels.

