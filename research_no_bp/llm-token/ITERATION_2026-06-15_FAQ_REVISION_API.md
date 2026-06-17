# Iteration 2026-06-15: Natural FAQ Revision And Deletion API Audit

## 目的

验证 API-compatible no-BP memory adapter 是否支持更接近真实在线学习的状态操作：

- 先学习一个事实。
- 后续对话中覆盖这个事实的新值。
- 覆盖后不再泄漏旧值。
- 删除后不再给 API 发送该事实的 memory hint。
- 其他未修改事实保持可用。

## 代码

- `online_memory_faq_api_experiment.py`
- 新增 `FaqMemory.clear_intent_state`
- 新增 `FaqMemory.overwrite`
- 新增 `FaqMemory.forget`
- 新增 `--run-revision-audit`
- 新增 `--revision-limit`
- 新增 `--revision-api-limit`

核心约束：覆盖和删除不依赖保存旧 raw dialogue。`overwrite` 按 intent 清掉旧 hash rows，再写入新 answer value；`forget` 清掉该 intent 的 learned rows，并把 answer value 置空。

## 命令

```bash
python -m py_compile online_memory_faq_api_experiment.py

python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_revision_dry \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --fact-limit 64 --api-limit 0 \
  --run-revision-audit --revision-limit 16

python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_revision_256_dry \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --fact-limit 256 --api-limit 0 \
  --run-revision-audit --revision-limit 32

API_KEY=... python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_revision_api_run \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --fact-limit 64 --api-limit 0 \
  --run-revision-audit --revision-limit 8 --revision-api-limit 4 \
  --run-api --api-timeout 90
```

## 结果

| Run | Metric | Value |
|---|---:|---:|
| revision 64 dry | old_before_correct | 1.000 |
| revision 64 dry | new_after_overwrite_correct | 1.000 |
| revision 64 dry | old_value_leak_after_overwrite | 0.000 |
| revision 64 dry | deleted_hint_suppressed | 1.000 |
| revision 64 dry | retained_other_correct | 1.000 |
| revision 64 dry | state_bytes_after_revision_audit | 42,903 |
| revision 256 dry | old_before_correct | 1.000 |
| revision 256 dry | new_after_overwrite_correct | 1.000 |
| revision 256 dry | old_value_leak_after_overwrite | 0.000 |
| revision 256 dry | deleted_hint_suppressed | 1.000 |
| revision 256 dry | retained_other_correct | 1.000 |
| revision 256 dry | state_bytes_after_revision_audit | 185,022 |
| revision API, 4 facts | api_after_overwrite_correct | 1.000 |
| revision API, 4 facts | api_after_delete_suppressed | 1.000 |

API 样本：

- 覆盖 return window 后：API 回答 `32 days`。
- 删除 return window 后：API 回答 `I don't know.`
- 覆盖 support email 后：API 使用新的 `priority` 邮箱。
- 删除 support email 后：API 回答 `I don't know.`

## 判断

这是 M5 API 原型的一个关键正结果：memory adapter 不只是追加事实，还可以在线覆盖和遗忘事实；API 只负责自然语言生成，事实状态由 no-BP memory 控制。

边界仍然清楚：

- memory 仍存 canonical answer values，不是强隐私压缩。
- intent 路由仍依赖 schema/alias，不是 learned semantic routing。
- revision audit 是 generated FAQ，不是开放域多轮客服对话。

## 下一步

1. 把 alias router 换成 learned semantic key，减少 schema 依赖。
2. 做多轮 session：事实引入、追问、修订、删除、跨话题回访。
3. 加 value compression：存短摘要或可验证 sketch，而不是完整 answer string。
