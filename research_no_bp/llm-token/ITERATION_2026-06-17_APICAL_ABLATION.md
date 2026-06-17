# 2026-06-17 Apical Error Gate Ablation

R051 showed a strong positive result for weak dendritic/apical local error gating. This round tests whether the gain specifically requires branch-local segment margins, or whether a simpler dynamic apical error signal is sufficient.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New CLI:

- `--apical-error-mode {segment_margin,global_margin,random_feedback,fixed_random}`

Modes:

| mode | Meaning |
|---|---|
| `segment_margin` | R051 default. Each feature segment gets its own target-vs-wrong local margin error. |
| `global_margin` | One global target-vs-wrong score margin drives all apical segments equally. |
| `random_feedback` | Global margin error is distributed across segments through fixed random positive feedback factors. |
| `fixed_random` | No target/error signal; a fixed random gate is applied every update. |

All modes keep the same no-BP constraints: local online updates, no BPTT/BP, no pretrained model/API backbone, no raw-text replay. `fixed_random` is the control for “extra gate/regularization only”.

## Smoke

Command template:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_ablate_<mode>_smoke \
  --method-filter phase_trace_apical_inhib \
  --train-chars 10000 --valid-chars 3000 --max-vocab 128 \
  --warmup-token-limit 1200 --stream-token-limit 600 \
  --segment-tokens 128 \
  --trace-branch --trace-order 8 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --apical-gating-branch --apical-error-mode <mode> \
  --apical-decay 0.85 --apical-strength 0.15 \
  --apical-margin 0.0 --apical-min-gate 0.8 \
  --apical-max-gate 1.25 --apical-error-clip 1.0 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --completion-count 0
```

Smoke results:

| mode | post CE | post acc | state bytes |
|---|---:|---:|---:|
| `segment_margin` | 1.965 | 0.513 | 1,315,340 |
| `global_margin` | 1.955 | 0.513 | 1,315,340 |
| `random_feedback` | 1.955 | 0.510 | 1,315,352 |
| `fixed_random` | 1.999 | 0.492 | 1,315,352 |

Smoke conclusion: fixed random gating is weaker. The useful signal is dynamic target-vs-wrong apical error; branch-local margins are not required in smoke.

## Medium

Command template:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_apical_ablate_<mode>_medium \
  --method-filter phase_trace_inhib phase_trace_apical_inhib \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --segment-tokens 256 \
  --trace-branch --trace-order 16 --trace-dim 64 \
  --trace-weight 0.5 --trace-decay 0.85 \
  --apical-gating-branch --apical-error-mode <mode> \
  --apical-decay 0.85 --apical-strength 0.15 \
  --apical-margin 0.0 --apical-min-gate 0.8 \
  --apical-max-gate 1.25 --apical-error-clip 1.0 \
  --adaptive-inhibition --inhibit-strength 0.15 \
  --inhibit-decay 0.85 --inhibit-lr 0.005 \
  --inhibit-disinhibit-lr 0.001 --inhibit-top-k 1 \
  --completion-count 0
```

Seed 0:

| mode | trace+inhib post CE / acc | apical+inhib post CE / acc | delta CE | state bytes |
|---|---:|---:|---:|---:|
| `segment_margin` | 2.358 / 0.429 | 2.294 / 0.435 | -0.064 | 2,761,740 |
| `global_margin` | 2.358 / 0.429 | 2.290 / 0.436 | -0.068 | 2,761,740 |
| `random_feedback` | 2.358 / 0.429 | 2.289 / 0.437 | -0.069 | 2,761,752 |
| `fixed_random` | 2.358 / 0.429 | 2.316 / 0.432 | -0.042 | 2,761,752 |

Seed 1 repeat for the two best dynamic modes:

| mode | trace+inhib post CE / acc | apical+inhib post CE / acc | delta CE |
|---|---:|---:|---:|
| `global_margin` | 2.410 / 0.431 | 2.346 / 0.437 | -0.064 |
| `random_feedback` | 2.410 / 0.431 | 2.345 / 0.436 | -0.065 |

## Interpretation

Positive:

- Dynamic apical error gating is robust across seeds.
- `global_margin` and `random_feedback` slightly beat the original `segment_margin` on seed 0 and seed 1.
- `fixed_random` improves CE a little but much less, so the R051/R052 gain cannot be explained only by a fixed gate or extra regularization.
- `random_feedback` matching `global_margin` is useful for the biological story: a fixed random feedback pathway can carry a global apical error signal without BP.

Boundary:

- Branch-local segment margin is not necessary for the current TinyStories token learner. The R051 claim should be revised from “branch-local error is essential” to “dynamic target-vs-wrong apical error is essential; random feedback can distribute it”.
- These runs disabled generation for speed, so repetition metrics should still use R051 seed0 generation summary until a full generation rerun is needed.
- The method remains far from GPT/API quality and still trails statistical `sparse_context_aux`; this is a mechanism improvement, not final language-model performance.

Conclusion:

R052 strengthens the apical route but changes the mechanism claim. The current best pure no-BP candidate should be described as:

> phase/trace WTA with weak dynamic apical error modulation and local output inhibition; the apical error can be global target-vs-wrong margin or fixed-random feedback, without BP or pretrained backbones.

Next step: use `random_feedback` or `global_margin` as the default candidate, then run low-precision/sparse-state audit and generation rerun to test whether the CE gain survives hardware-friendly compression and whether repetition remains improved.
