# 2026-06-17 Dendritic/Apical Local Error Gating

本轮沿 R050 结论继续推进：不再放大 recurrent transition plasticity，而是在当前最强的 phase/trace/WTA online learner 上加入 dendritic/apical local prediction-error gating。目标是检验局部分支误差信号能否改善 winner selection，同时保持纯 no-BP、无预训练主干、无原始文本 replay。

## 方法

代码：`phase_binding_online_stream_experiment.py`

新增类：

- `OnlineTraceApicalGatedCompetitivePhaseMemory`

新增 CLI：

- `--apical-gating-branch`
- `--apical-decay`
- `--apical-strength`
- `--apical-margin`
- `--apical-min-gate`
- `--apical-max-gate`
- `--apical-error-clip`

机制：

```text
feature = concat(phase_branch_1, phase_branch_2, trace)

for each dendritic segment s:
  margin_s = dot(W[target, s], feature_s) - max_wrong dot(W[wrong, s], feature_s)
  apical_error_s = max(0, apical_margin - margin_s)
  trace_s = decay * trace_s + apical_error_s
  gate_s = clip(1 + strength * trace_s, min_gate, max_gate)

gated_feature_s = gate_s * feature_s
W[target] += lr * gated_feature
W[wrong]  -= lr/k * gated_feature
```

这是分支局部的三因子调制：basal feature 来自 phase/trace 分支，apical signal 来自当前 target-vs-wrong 局部预测误差，突触更新仍是 WTA/anti-winner 局部规则。不使用 BP/BPTT、预训练 LLM/API 主干或原始文本 replay。`sparse_context_aux` 仍只是统计辅助，不作为最终方法。

## Smoke

Default apical gate was too strong:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_smoke \
  --method-filter phase_trace_competitive_online phase_trace_fatigue phase_trace_inhib phase_trace_apical \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --warmup-token-limit 1200 --stream-token-limit 600 \
  --segment-tokens 128 \
  --trace-branch --trace-order 8 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --apical-gating-branch --apical-decay 0.90 \
  --apical-strength 0.75 --apical-margin 0.0 \
  --apical-min-gate 0.5 --apical-max-gate 2.0 \
  --apical-error-clip 2.0 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1
```

Default result:

| method | post CE | post acc |
|---|---:|---:|
| `phase_trace_competitive_online` | 2.050 | 0.490 |
| `phase_trace_fatigue_competitive_online` | 2.030 | 0.492 |
| `phase_trace_inhib_competitive_online` | 2.034 | 0.490 |
| `phase_trace_apical_competitive_online` | 2.045 | 0.462 |
| `phase_trace_apical_inhib_competitive_online` | 2.041 | 0.467 |

This showed that strong apical gates over-amplify local branch errors and hurt top-1.

Low-strength apical gate was strongly positive:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_sweep_s015 \
  --method-filter phase_trace_competitive_online phase_trace_fatigue phase_trace_inhib phase_trace_apical \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --warmup-token-limit 1200 --stream-token-limit 600 \
  --segment-tokens 128 \
  --trace-branch --trace-order 8 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --apical-gating-branch --apical-decay 0.85 \
  --apical-strength 0.15 --apical-margin 0.0 \
  --apical-min-gate 0.8 --apical-max-gate 1.25 \
  --apical-error-clip 1.0 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --completion-count 0
```

Smoke sweep result:

| method | post CE | post acc |
|---|---:|---:|
| `phase_trace_competitive_online` | 2.050 | 0.490 |
| `phase_trace_fatigue_competitive_online` | 2.030 | 0.492 |
| `phase_trace_fatigue_inhib_competitive_online` | 2.020 | 0.483 |
| `phase_trace_inhib_competitive_online` | 2.034 | 0.490 |
| `phase_trace_apical_competitive_online` | 1.969 | 0.513 |
| `phase_trace_apical_fatigue_competitive_online` | 1.972 | 0.506 |
| `phase_trace_apical_fatigue_inhib_competitive_online` | 1.971 | 0.508 |
| `phase_trace_apical_inhib_competitive_online` | 1.965 | 0.513 |

Two additional smoke checks were still positive but weaker:

- `apical_strength=0.30`, `max_gate=1.50`: best post CE `1.977`, acc `0.502`
- `apical_strength=0.30`, `apical_margin=-0.10`, `max_gate=1.35`: best post CE `1.978`, acc `0.502`

## Medium

Command:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_medium_s015 \
  --method-filter phase_trace_competitive_online phase_trace_fatigue phase_trace_inhib phase_trace_apical \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --apical-gating-branch --apical-decay 0.85 \
  --apical-strength 0.15 --apical-margin 0.0 \
  --apical-min-gate 0.8 --apical-max-gate 1.25 \
  --apical-error-clip 1.0 \
  --output-fatigue --fatigue-strength 0.75 --fatigue-decay 0.80 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --completion-count 4 --prompt-tokens 16 --completion-tokens 48
```

Seed 0 result:

| method | online CE / acc | post CE / acc | greedy repeat-2 | state bytes |
|---|---:|---:|---:|---:|
| `phase_trace_competitive_online` | 3.157 / 0.335 | 2.389 / 0.427 | 0.681 | 2,498,560 |
| `phase_trace_fatigue_competitive_online` | 3.141 / 0.339 | 2.382 / 0.429 | 0.606 | 2,499,584 |
| `phase_trace_inhib_competitive_online` | 3.114 / 0.335 | 2.358 / 0.429 | 0.473 | 2,761,728 |
| `phase_trace_fatigue_inhib_competitive_online` | 3.113 / 0.341 | 2.363 / 0.432 | 0.436 | 2,762,752 |
| `phase_trace_apical_competitive_online` | 3.207 / 0.336 | 2.318 / 0.431 | 0.771 | 2,498,572 |
| `phase_trace_apical_fatigue_competitive_online` | 3.198 / 0.340 | 2.317 / 0.430 | 0.771 | 2,499,596 |
| `phase_trace_apical_inhib_competitive_online` | 3.181 / 0.336 | 2.294 / 0.435 | 0.383 | 2,761,740 |
| `phase_trace_apical_fatigue_inhib_competitive_online` | 3.185 / 0.339 | 2.304 / 0.440 | 0.383 | 2,762,764 |

Seed 1 repeat:

| method | online CE / acc | post CE / acc |
|---|---:|---:|
| `phase_trace_competitive_online` | 3.194 / 0.330 | 2.450 / 0.420 |
| `phase_trace_fatigue_competitive_online` | 3.176 / 0.334 | 2.440 / 0.426 |
| `phase_trace_inhib_competitive_online` | 3.146 / 0.330 | 2.410 / 0.431 |
| `phase_trace_fatigue_inhib_competitive_online` | 3.144 / 0.329 | 2.414 / 0.434 |
| `phase_trace_apical_competitive_online` | 3.248 / 0.333 | 2.378 / 0.422 |
| `phase_trace_apical_fatigue_competitive_online` | 3.237 / 0.337 | 2.375 / 0.423 |
| `phase_trace_apical_inhib_competitive_online` | 3.215 / 0.333 | 2.350 / 0.431 |
| `phase_trace_apical_fatigue_inhib_competitive_online` | 3.217 / 0.333 | 2.359 / 0.436 |

## 判断

正信号：

- This is the strongest pure no-BP token learner so far.
- CE improves over the previous best on both seeds:
  - seed 0: trace+inhib `2.358` -> trace+apical+inhib `2.294`
  - seed 1: trace+inhib `2.410` -> trace+apical+inhib `2.350`
- Top-1 also improves on seed 0 and is preserved on seed 1:
  - seed 0: trace+fatigue+inhib acc `0.432` -> apical+fatigue+inhib acc `0.440`
  - seed 1: trace+fatigue+inhib acc `0.434` -> apical+fatigue+inhib acc `0.436`
- Greedy repetition improves with inhibition:
  - seed 0 trace+inhib repeat-2 `0.473`
  - seed 0 trace+fatigue+inhib repeat-2 `0.436`
  - seed 0 apical+inhib repeat-2 `0.383`
- Extra state is negligible: apical trace adds only `12` bytes in the current three-segment setup.

边界：

- Strong apical gating is harmful; the useful region is weak modulation (`strength=0.15`, `max_gate=1.25`).
- Apical without inhibition improves CE but still loops badly in greedy generation (`repeat-2=0.771`), so inhibition remains necessary for generation behavior.
- The method still trails statistical `sparse_context_aux` on post CE/acc, but that baseline remains a cache/count auxiliary and is not the final no-BP method.
- Generated text is still far from GPT/API quality; the gain is a strong local learning improvement, not a solved language model.

Conclusion:

Dendritic/apical local prediction-error gating is the first recurrent/dendritic-style branch after trace/fatigue/inhibition that gives a stable seed-level CE improvement. This should become the new main pure no-BP token learner candidate. Next steps should ablate the apical signal itself: frozen gate vs no trace, branch-wise errors vs random feedback errors, and low-precision/sparse-state audit.
