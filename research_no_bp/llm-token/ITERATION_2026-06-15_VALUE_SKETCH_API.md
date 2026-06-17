# Iteration 2026-06-15: Structured Answer Sketch Memory

## 目的

前面的 API-compatible FAQ memory 已经不保存 raw training examples，但仍保存完整 canonical answer strings。本轮把 answer storage 改成可选结构化 sketch：

- `--answer-store full`: 保存完整答案字符串。
- `--answer-store sketch`: 保存短 code tuple，例如 `(return_window, subject, days)` 的紧凑形式。

API hint 仍由 memory 临时 render 成自然短句，但 learned state 不保存完整 answer text。

## 代码

- `online_memory_faq_api_experiment.py`
- 新增 `--answer-store {full,sketch}`
- 新增 `answer_sketch`
- 新增 `render_answer_payload`
- Summary 增加：
  - `stores_answer_values`
  - `stores_answer_text`
  - `answer_store`

## 命令

```bash
python -m py_compile online_memory_faq_api_experiment.py

python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_value_sketch_256_dry \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --router semantic --semantic-dim 0 --semantic-feature-cap 12 \
  --answer-store sketch --fact-limit 256 --api-limit 16

python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_value_sketch_revision_256_dry \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --router semantic --semantic-dim 0 --semantic-feature-cap 12 \
  --answer-store sketch --fact-limit 256 --api-limit 0 \
  --run-revision-audit --revision-limit 32

API_KEY=... python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_value_sketch_api_run \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --router semantic --semantic-dim 0 --semantic-feature-cap 12 \
  --answer-store sketch --fact-limit 64 --api-limit 8 \
  --run-revision-audit --revision-limit 8 --revision-api-limit 4 \
  --run-api --api-timeout 90
```

## 结果

| Run | Accuracy | State bytes | Stores answer text | Answer store |
|---|---:|---:|---:|---|
| 256 full recheck | 1.000 | 292,614 | true | full |
| 256 sketch | 1.000 | 284,248 | false | sketch |
| 256 sketch revision | 1.000 overwrite/delete | 262,496 after audit | false | sketch |
| 64 sketch API | api_memory_hint 1.000 | 77,866 | false | sketch |

Revision audit with sketch store:

- `new_after_overwrite_correct`: `1.000`
- `old_value_leak_after_overwrite`: `0.000`
- `deleted_hint_suppressed`: `1.000`
- API overwrite/delete on 4 facts: both `1.000`

## 判断

This closes a privacy/storage gap: the learned state no longer has to retain full answer sentences. It still stores structured answer values, which are necessary for exact FAQ recall, but the representation is less raw and easier to audit/delete.

The byte reduction is modest because semantic routing prototypes dominate the current state size. On 256 facts, sketch storage reduces state from `292,614` to `284,248` bytes; on 64 API facts it reduces state to `77,866` bytes.

## 下一步

1. Build a multi-turn qualitative support-session benchmark.
2. Compare API no-memory, raw retrieval, full-answer memory, and sketch memory side by side.
3. Explore count-min/sketch compression for semantic prototypes, since they now dominate memory size.
