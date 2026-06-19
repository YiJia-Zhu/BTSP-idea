# R092 DLL Local Depth

**Date**: 2026-06-19

## Question

Can a true intermediate no-BP layer improve the TinyStories online token learner without using BP/BPTT, pretrained weights, raw replay, or statistical n-gram memory?

## Implementation

Added `OnlineDLLDeepLocalMemory` to `phase_binding_online_stream_experiment.py` behind `--dll-depth-branch`.

Mechanism:

- basal input: existing phase + leaky trace feature;
- one or more hidden DLL layers;
- each layer has fixed random target projection `target_l = B_l @ label_embed[token]`;
- local update only: `(target_l - h_l) * activation'(z_l) outer local_input`;
- final WTA readout remains local target/wrong competition;
- optional adaptive output inhibition can wrap the DLL learner;
- model state stores learned arrays and fixed random projections, not raw text.

No cross-layer gradient, no BP, no BPTT, no pretrained model.

## Runs

Best candidate three-seed run:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_r092_dll_1x768_seed${SEED} \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --phase-dim 128 --branch-orders 1 2 --branch-weights 0.5 0.5 \
  --trace-branch --trace-order 16 --trace-dim 128 \
  --dll-depth-branch --dll-hidden-dims 768 --dll-label-dim 128 \
  --dll-lr 0.02 --dll-bias-lr 0.002 --dll-delta-clip 1.0 \
  --adaptive-inhibition --inhibit-strength 0.15 --inhibit-lr 0.005 --inhibit-top-k 1 \
  --method-filter phase_trace_dll_local_inhib_competitive_online \
  --completion-count 1 --prompt-tokens 16 --completion-tokens 48 \
  --seed ${SEED}
```

Aggregate:

- `output/phase_binding_online_stream_r092_dll/summary.csv`
- `output/phase_binding_online_stream_r092_dll/per_seed_summary.csv`
- `output/phase_binding_online_stream_r092_dll/variant_probe_summary.csv`
- `output/phase_binding_online_stream_r092_dll/generation_summary_aggregate.csv`

## Results

| Method | Seeds | Post CE | Post acc | Retention CE | State bytes |
|---|---:|---:|---:|---:|---:|
| `phase_trace_dll_local_inhib_competitive_online` | 3 | `2.2807 +/- 0.0099` | `0.4720 +/- 0.0059` | `3.2743 +/- 0.0148` | `6,303,744` |

Per seed:

| Seed | Post CE | Post acc |
|---:|---:|---:|
| 0 | 2.2791 | 0.4751 |
| 1 | 2.2936 | 0.4638 |
| 2 | 2.2695 | 0.4771 |

Reference points from prior reports:

| Reference | Post CE | Post acc | Note |
|---|---:|---:|---|
| R134/R136 `phase_trace_apical_inhib` | `2.2791 +/- 0.0217` | `0.4580 +/- 0.0066` | simpler apical+inhibition baseline |
| R136 e-prop trace | `2.3565 +/- 0.0164` | `0.4730 +/- 0.0082` | acc-positive, CE-negative |
| R069/R067 best line | about `2.253` | about `0.475-0.482` | stronger calibration/branch-agreement line |

## Probe Findings

Small and medium probes show a consistent pattern:

- two DLL layers are worse than one wide hidden layer;
- width helps monotonically up to the tested `1x768` point;
- adaptive output inhibition improves DLL CE and top-1;
- DLL improves top-1 compared with the simpler apical+inhibition baseline, but CE calibration remains worse than the R069/R067 best line.

Selected probe rows:

| Probe | Post CE | Post acc |
|---|---:|---:|
| smoke trace baseline | 2.5194 | 0.4331 |
| smoke DLL 2x256 | 2.5173 | 0.4268 |
| medium trace baseline seed0 | 2.3799 | 0.4402 |
| medium apical+inhib baseline seed0 | 2.2729 | 0.4561 |
| medium DLL 1x512+inhib seed0 | 2.3080 | 0.4674 |
| medium DLL 1x768+inhib seed0 | 2.2791 | 0.4751 |

## Verdict

R092 is **DONE-TRADEOFF**, not a success.

The success gate was post CE `< 2.253` on at least two seeds. The best DLL candidate reaches `2.2807 +/- 0.0099`, so it does not pass. It also uses substantially larger state than the best prior calibrated line.

However, the result is not a dead end: a single wide local DLL layer gives competitive top-1 (`0.4720 +/- 0.0059`) and beats the simpler apical+inhibition baseline on accuracy. The failure is mainly CE calibration and state cost, not winner selection.

## Next

Proceed to R093 NoProp. Do not keep widening DLL as the main path. If reused later, treat DLL as a candidate/top-1 branch that needs a local calibration mechanism, not as the CE-winning R096 backbone.
