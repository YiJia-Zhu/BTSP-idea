# R093 NoProp Local Denoising

**Date**: 2026-06-19

## Question

Can a NoProp-style decoupled local denoising stack improve the TinyStories online token learner after R092 DLL failed the CE gate?

## Implementation

Added `OnlineNoPropLocalDenoisingMemory` to `phase_binding_online_stream_experiment.py` behind `--noprop-depth-branch`.

Mechanism:

- basal input is the existing phase + trace feature;
- each layer has fixed random label target codes;
- update-time noisy target:
  `z_l = sqrt(alpha_l) * target_l + sqrt(1 - alpha_l) * noise_l(local_input)`;
- local input map learns `local_input -> z_l`;
- local denoiser learns `z_l -> target_l`;
- deeper layers train from the previous layer's clean target code, so training does not require a forward chain through lower learned layers;
- inference uses the standard feed-forward chain;
- final readout is the existing local target/wrong WTA update;
- no BP, no BPTT, no pretrained model, no raw text replay.

This is a practical online adaptation of the NoProp idea rather than a full reproduction of the original paper.

## Runs

Best candidate three-seed run:

```bash
python phase_binding_online_stream_experiment.py \
  --out-dir output/phase_binding_online_stream_r093_noprop_1x768_seed${SEED} \
  --train-chars 50000 --valid-chars 10000 --max-vocab 256 \
  --phase-dim 128 --branch-orders 1 2 --branch-weights 0.5 0.5 \
  --trace-branch --trace-order 16 --trace-dim 128 \
  --noprop-depth-branch --noprop-hidden-dims 768 --noprop-label-dim 128 \
  --noprop-alpha-start 0.95 --noprop-alpha-end 0.50 \
  --noprop-lr 0.02 --noprop-denoise-lr 0.02 \
  --noprop-bias-lr 0.002 --noprop-delta-clip 1.0 \
  --adaptive-inhibition --inhibit-strength 0.15 --inhibit-lr 0.005 --inhibit-top-k 1 \
  --method-filter phase_trace_noprop_local_inhib_competitive_online \
  --completion-count 1 --prompt-tokens 16 --completion-tokens 48 \
  --seed ${SEED}
```

Aggregate:

- `output/phase_binding_online_stream_r093_noprop/summary.csv`
- `output/phase_binding_online_stream_r093_noprop/per_seed_summary.csv`
- `output/phase_binding_online_stream_r093_noprop/variant_probe_summary.csv`
- `output/phase_binding_online_stream_r093_noprop/generation_summary_aggregate.csv`

## Results

| Method | Seeds | Post CE | Post acc | Retention CE | State bytes |
|---|---:|---:|---:|---:|---:|
| `phase_trace_noprop_local_inhib_competitive_online` | 3 | `2.2782 +/- 0.0115` | `0.4672 +/- 0.0074` | `3.2855 +/- 0.0222` | `10,632,196` |

Per seed:

| Seed | Post CE | Post acc |
|---:|---:|---:|
| 0 | 2.2690 | 0.4771 |
| 1 | 2.2940 | 0.4587 |
| 2 | 2.2714 | 0.4658 |

Comparison:

| Method | Post CE | Post acc | State bytes | Verdict |
|---|---:|---:|---:|---|
| R092 DLL 1x768+inhib | `2.2807 +/- 0.0099` | `0.4720 +/- 0.0059` | 6.30M | CE-similar, acc better, smaller |
| R093 NoProp 1x768+inhib | `2.2782 +/- 0.0115` | `0.4672 +/- 0.0074` | 10.63M | CE slightly better, acc/state worse |
| R069/R067 best line | about `2.253` | about `0.475-0.482` | < R093 | still stronger CE |

## Probe Findings

- One wide layer works better than two smaller layers, matching the R092 DLL pattern.
- Higher alpha (`0.95`) is slightly better in smoke than the noisier default.
- NoProp improves the best R092 CE by only about `0.0025`, while adding roughly `4.33MB` state and losing top-1 accuracy.
- The deep local target methods are not failing to choose winners entirely; they are failing CE calibration and efficient state use.

Selected probe rows:

| Probe | Post CE | Post acc |
|---|---:|---:|
| smoke NoProp 1x128+inhib | 2.591 | 0.425 |
| smoke NoProp 1x256+inhib | 2.528 | 0.416 |
| smoke NoProp 2x128+inhib | 2.727 | 0.385 |
| smoke NoProp 1x256 alpha0.95 | 2.524 | 0.423 |
| medium NoProp 1x512 seed0 | 2.294 | 0.464 |
| medium NoProp 1x768 seed0 | 2.269 | 0.477 |

## Verdict

R093 is **DONE-TRADEOFF**, not a success.

The success gate was post CE `< 2.253`. The best NoProp candidate reaches `2.2782 +/- 0.0115`, so it does not pass. It is also state-heavier than R092 and the older calibrated/apical line.

The mechanism is still informative: NoProp-style local denoising slightly improves CE over DLL, but the gain is too small and does not solve the core calibration gap.

## Next

Do not move directly to R096 as a claimed integration success. The next useful step is a small R096-prep arbitration/calibration experiment: keep R094 KV and R095 e-prop as candidate/tie-break branches, but first design a local calibration/readout mechanism for deep local-target features. A direct stack of R092/R093 + R094 + R095 is likely to inherit the CE calibration problem.
