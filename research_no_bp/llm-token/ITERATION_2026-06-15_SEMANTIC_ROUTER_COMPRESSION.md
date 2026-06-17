# Iteration 2026-06-15: Semantic Router Compression

## 目的

上一轮 semantic FAQ router 去掉了 alias 必需性，但状态较大：256 facts 约 `824,791` bytes。本轮测试更紧的 no-raw semantic memory：只保留 hashed sparse semantic features，并给每个 intent prototype 加 feature cap。

## 代码

- `online_memory_faq_api_experiment.py`
- 新增 `--semantic-feature-cap`
- 支持 `--semantic-dim 0`，即不保存 dense random-projection prototype，只保存 hashed sparse semantic prototype。
- Tombstone deletion 仍使用 hashed sparse prototype，删除后不发送 memory hint。

## 命令

```bash
python -m py_compile online_memory_faq_api_experiment.py

python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_semantic_sparseonly_cap12_256_dry \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --router semantic --semantic-dim 0 --semantic-feature-cap 12 \
  --fact-limit 256 --api-limit 16

python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_semantic_sparseonly_cap12_revision_256_dry \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --router semantic --semantic-dim 0 --semantic-feature-cap 12 \
  --fact-limit 256 --api-limit 0 \
  --run-revision-audit --revision-limit 32

API_KEY=... python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_semantic_sparseonly_cap12_api_run \
  --dataset generated --train-style dialogue --eval-style paraphrase \
  --router semantic --semantic-dim 0 --semantic-feature-cap 12 \
  --fact-limit 64 --api-limit 8 \
  --run-revision-audit --revision-limit 8 --revision-api-limit 4 \
  --run-api --api-timeout 90
```

## 结果

| Config | Accuracy | State bytes | Revision overwrite | Old-value leak | Delete suppression |
|---|---:|---:|---:|---:|---:|
| semantic dense+sparse, no cap | 1.000 | 824,791 | 1.000 | 0.000 | 1.000 |
| sparse-only cap 8 | 0.986 | 273,008 | not run | not run | not run |
| sparse-only cap 12 | 1.000 | 292,592 | 1.000 | 0.000 | 1.000 |
| sparse-only cap 16 | 1.000 | 312,048 | 1.000 | 0.000 | 1.000 |
| sparse-only cap 12 API, 64 facts | 1.000 | 79,931 | 1.000 | 0.000 | 1.000 |

The best current tradeoff is `--semantic-dim 0 --semantic-feature-cap 12`: it keeps 256-fact local QA and revision/delete perfect while reducing state from `824,791` to `292,592` bytes.

The compressed API smoke also passed: `api_no_memory` accuracy `0.000`, `api_memory_hint` accuracy `1.000` on 8 questions, and API revision overwrite/delete both `1.000` on 4 facts.

## 判断

This improves the practical online-learning story: semantic routing no longer needs raw examples or dense prototypes, and storage becomes much closer to the earlier alias-router size while preserving overwrite/delete behavior.

Remaining boundary:

- Memory still stores canonical answer values.
- Semantic features are hashed lexical/phrase features, not deep embeddings.
- The API smoke is still small; a larger qualitative support-session benchmark is needed.

## 下一步

1. Add value compression or structured answer sketches to reduce answer-value storage.
2. Build a multi-turn qualitative support session benchmark.
3. Compare compressed memory API answers against base API and raw retrieval in a human-readable report.
