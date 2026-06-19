# R134: Hebbian KV Token Branch

**Date**: 2026-06-18

## Purpose

Test R094: add a pure no-BP Hebbian key-value associative branch to the TinyStories online phase/trace/apical/inhibition learner.

The mechanism is a compressed neural associative matrix, not a sparse context table:

- key/query: fixed random token codes accumulated with a local leaky trace over the current context
- write: `M <- (1 - decay) M + lr * outer(value_anchor[target], key(context))`
- read: `kv = normalize(M @ key(context))`
- optional direct local readout: `kv_score = value_anchor @ kv`
- no BP, no BPTT, no pretrained backbone, no raw-text replay

Implementation: `phase_binding_online_stream_experiment.py`

New CLI:

- `--hebbian-kv-branch`
- `--kv-order`
- `--kv-dim`
- `--kv-trace-decay`
- `--kv-weight`
- `--kv-score-weight`
- `--kv-lr`
- `--kv-decay`
- `--kv-clip`

## Runs

Medium three-seed run:

```bash
python phase_binding_online_stream_experiment.py \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --phase-dim 128 --branch-orders 1 2 --branch-weights 0.5 0.5 \
  --trace-branch --trace-order 16 --trace-dim 128 \
  --hebbian-kv-branch --kv-order 16 --kv-dim 128 \
  --kv-trace-decay 0.90 --kv-weight 0.0 --kv-score-weight 1.0 \
  --kv-lr 0.04 --kv-decay 0.002 \
  --apical-gating-branch --apical-strength 0.15 \
  --apical-max-gate 1.25 --apical-error-mode random_feedback \
  --adaptive-inhibition --inhibit-strength 0.15 --inhibit-lr 0.005 \
  --inhibit-top-k 1 \
  --method-filter phase_trace_apical_inhib_competitive_online \
                  phase_trace_kv_apical_inhib_competitive_online
```

Outputs:

- `output/phase_binding_online_stream_r134_kv_medium_seed0/`
- `output/phase_binding_online_stream_r134_kv_medium_seed1/`
- `output/phase_binding_online_stream_r134_kv_medium_seed2/`
- aggregate: `output/phase_binding_online_stream_r134_hebbian_kv/`

## Results

| Method | post CE mean | post acc mean | retention CE mean | state bytes |
|---|---:|---:|---:|---:|
| phase_trace_apical_inhib | 2.2791 +/- 0.0217 | 0.4580 +/- 0.0066 | 3.0382 +/- 0.0211 | 2,892,824 |
| phase_trace_kv_apical_inhib | 2.2668 +/- 0.0234 | 0.4557 +/- 0.0041 | 3.0866 +/- 0.0268 | 3,351,584 |

Paired KV - baseline post deltas:

| seed | CE delta | acc delta |
|---:|---:|---:|
| 0 | -0.01548 | +0.00308 |
| 1 | -0.00923 | -0.00103 |
| 2 | -0.01212 | -0.00873 |

Generation post-online greedy:

| Method | repeat-2 mean | distinct-2 mean | first-token match |
|---|---:|---:|---:|
| phase_trace_apical_inhib | 0.5059 | 0.4941 | 0.5556 |
| phase_trace_kv_apical_inhib | 0.4563 | 0.5437 | 0.5556 |

Controlled decoding:

| Method | repeat-2 mean | distinct-2 mean |
|---|---:|---:|
| phase_trace_apical_inhib | 0.1135 | 0.8865 |
| phase_trace_kv_apical_inhib | 0.1324 | 0.8676 |

## Interpretation

R134 is a partial positive result.

The Hebbian KV branch gives a reproducible CE improvement on all three medium seeds: mean post CE improves by `0.0123`, from `2.2791` to `2.2668`. This supports the claim that a rank-1 Hebbian associative matrix can add useful long-context probability mass to the pure no-BP token learner.

It is not a clean winner-selection improvement. Mean top-1 accuracy drops from `0.4580` to `0.4557`, with seed2 losing `0.0087`. Retention on the warmup slice also worsens (`3.0382 -> 3.0866`), so the KV matrix is currently acting as a helpful online calibration branch with interference cost.

Smoke sweeps showed that naive KV feature concatenation was consistently negative; the positive medium result requires direct local associative scoring with `kv_weight=0.0` and `kv_score_weight=1.0`. Therefore the current mechanism should not yet be blindly stacked into R096. It needs a confidence/novelty gate or apical arbitration so KV contributes where it is predictive and stays silent on retention-sensitive contexts.

## Verdict

R094 status: **DONE / PARTIAL-POSITIVE**.

Evidence supports:

- pure no-BP Hebbian KV implementation works end to end;
- CE improvement is reproducible across seeds;
- no raw text is stored in model state.

Evidence does not yet support:

- top-1 winner improvement;
- retention-safe integration;
- final R096 stacking without a gate.

Next:

1. Add a local confidence/novelty gate for `kv_score_weight`.
2. Compare against R095 e-prop trace on long-context buckets.
3. Only include KV in R096 if the gate preserves CE gain without acc/retention loss.
