# Iteration 2026-06-15: Learned Semantic FAQ Router API Prototype

## 目的

上一轮 FAQ memory 依赖 `aliases` 做 intent routing。这里实现一个更接近实际在线学习的 semantic-key router：训练时不保存 raw dialogue，只把训练语句和 query-like 文本写成 hashed sparse semantic prototypes；推理时用 hashed overlap + IDF-like weighting 选择 memory slot。

## 代码

- `online_memory_faq_api_experiment.py`
- 新增 `--router {alias,semantic,hybrid}`
- 新增 `--semantic-dim`
- 新增 hashed sparse semantic prototypes:
  - `semantic_sparse_prototypes`
  - `semantic_feature_df`
  - `tombstone_sparse_prototypes`
- 删除后保留 tombstone prototype；若查询更接近 tombstone，则不发送 memory hint。

注意：这仍不是深度语义 embedding。它是本地 no-BP / Hebbian-style hashed semantic key，用于减少手写 alias 路由依赖。

## 命令

```bash
python -m py_compile online_memory_faq_api_experiment.py

python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_semantic_256_dry \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --router semantic --semantic-dim 128 --fact-limit 256 --api-limit 16

python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_semantic_revision_256_dry \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --router semantic --semantic-dim 128 --fact-limit 256 --api-limit 0 \
  --run-revision-audit --revision-limit 32

API_KEY=... python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_semantic_api_run \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --router semantic --semantic-dim 128 --fact-limit 64 --api-limit 8 \
  --run-revision-audit --revision-limit 8 --revision-api-limit 4 \
  --run-api --api-timeout 90
```

## 结果

| Run | Metric | Value |
|---|---:|---:|
| semantic 64 dry | local accuracy | 1.000 |
| semantic 64 dry | state bytes | 212,954 |
| semantic 256 dry | local accuracy | 1.000 |
| semantic 256 dry | state bytes | 824,791 |
| semantic revision 64 dry | overwrite correctness | 1.000 |
| semantic revision 64 dry | old-value leak | 0.000 |
| semantic revision 64 dry | delete suppression | 1.000 |
| semantic revision 256 dry | overwrite correctness | 1.000 |
| semantic revision 256 dry | old-value leak | 0.000 |
| semantic revision 256 dry | delete suppression | 1.000 |
| semantic API, 8 questions | api_no_memory accuracy | 0.000 |
| semantic API, 8 questions | api_memory_hint accuracy | 1.000 |
| semantic API revision, 4 facts | api_after_overwrite_correct | 1.000 |
| semantic API revision, 4 facts | api_after_delete_suppressed | 1.000 |

## 判断

这是 M5 的重要正结果：FAQ prototype 不再必须靠 hand-authored alias routing 才能闭合。Semantic router 只保存 hashed sparse feature statistics、prototype vectors、answer values 和 tombstones；不保存 raw training statements or raw questions。

权衡：

- Semantic router 的存储成本明显高于 alias router：64 facts 约 `213KB`，256 facts 约 `825KB`。
- 仍然存 canonical answer values，不是强隐私压缩。
- Semantic key 是 hashed lexical/phrase overlap，不是强 embedding semantic generalization。

## 下一步

1. 压缩 semantic prototypes：feature cap、count-min sketch、低维 sign random projection。
2. 扩展到 multi-turn sessions，不只是一轮 fact/query/revision/delete。
3. 加 human qualitative comparison：base API vs memory API on held-out natural support prompts。
