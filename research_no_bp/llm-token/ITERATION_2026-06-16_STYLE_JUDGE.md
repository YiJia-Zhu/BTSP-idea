# Iteration 2026-06-16: Personalized Style Judge

## 目的

上一轮 style API benchmark 主要用规则打分：格式、签名、必含短语、禁用词、长度。  
本轮补一个小规模偏好式 judge，评估自然度和可用性：

```text
same prompt -> no-memory answer / raw-profile answer / style-sketch answer
          -> blinded API judge ranks A/B/C
```

这不是新的生成实验；它复用 `output/online_memory_style_delete_api/session_turns.csv` 中已经生成的候选答案。

## 代码

- `../online_memory_style_judge_experiment.py`
- Input: `../output/online_memory_style_delete_api/session_turns.csv`
- Output: `../output/online_memory_style_judge_api_fixed/`

## 命令

```bash
python -m py_compile online_memory_style_judge_experiment.py online_memory_faq_api_experiment.py

API_KEY=... python online_memory_style_judge_experiment.py \
  --source-dir output/online_memory_style_delete_api \
  --out-dir output/online_memory_style_judge_api \
  --case-limit 8 --run-api

python online_memory_style_judge_experiment.py \
  --reuse-existing --reuse-dir output/online_memory_style_judge_api \
  --out-dir output/online_memory_style_judge_api_fixed
```

The second command fixes a label-decoding issue without re-calling the API; it reuses the saved raw judge outputs.

## 结果

| metric | value |
|---|---:|
| cases | 8 |
| no_memory_best_rate | 0.500 |
| raw_profile_best_rate | 0.000 |
| style_sketch_best_rate | 0.500 |
| style_sketch_beats_no_memory_rate | 0.500 |
| style_sketch_beats_raw_profile_rate | 0.750 |
| raw_profile_beats_no_memory_rate | 0.375 |

## 判断

This is a mixed but useful result:

- Positive: style-sketch is best on 4/8 cases and beats raw-profile on 6/8, so the compact no-raw memory can produce useful customer-facing drafts.
- Boundary: no-memory is also best on 4/8 cases. Rule compliance is strong, but naturalness/usefulness is not yet consistently better than the base API.
- Negative for raw profile: raw-profile is never best in this judge pass, suggesting raw text hints can overconstrain or produce awkward support drafts.

The next technical target should be better hint rendering or a two-stage generator that converts compact memory into softer style guidance without exposing raw examples.

