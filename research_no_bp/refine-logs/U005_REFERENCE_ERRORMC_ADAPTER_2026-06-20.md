# U005: Direct Reference Error-Microcircuit Adapter

**Date**: 2026-06-20

## Purpose

Directly call the cloned `Error-Neuron-Microcircuits` implementation instead of
re-implementing the microcircuit update logic.

U005 imports the reference `init_MC` / `errormc_model` and uses the original:

- `evolve_system`;
- `evolve_voltages`;
- `calc_dendritic_updates`;
- `evolve_synapses`.

The only local code is the LLM adapter:

```text
context tokens -> r0 input vector
next token -> one-hot rate target u_tgt
reference errormc_model.evolve_system(...)
```

No parser, answer slot, BP, autograd, pretrained backbone, or n-gram table is
used.

## Implementation

Added `u005_reference_errormc_llm_adapter.py`.

The script builds reference params with:

- `mc_model=errormc`;
- `activation=[tanh ... tanh, linear]`;
- `error_activation=linear`;
- `init_WIP_identity=true`;
- `init_BPI_identity=true`;
- `varphi_transfer=true`;
- `fw_connection_mode=layered`;
- `bw_connection_mode=layered` or `skip`;
- `model_type=FA` or diagnostic `BP`.

TinyStories, GSM8k-Aug, and mixed next-token streams reuse the same adapter.

## Smoke Results

TinyStories smoke uses `train_chars=12000`, `valid_chars=3000`,
`max_vocab=128`, `context_len=32`, `d_model=64`, `blocks=4`,
`eval_token_limit=600`, seed 0.

| Model | Feedback | Logit scale | Valid post CE | Valid post acc |
|---|---|---:|---:|---:|
| reference FA | layered | 8 | 18.967 | 0.018 |
| reference BP diagnostic | layered | 8 | 19.116 | 0.020 |
| reference FA | layered | 1 | 5.451 | 0.032 |
| reference BP diagnostic | layered | 1 | 5.339 | 0.032 |

## Interpretation

Directly calling the reference code works mechanically, but the original
teacher-student / rate-target setup does not directly map to next-token softmax.

Two issues are clear:

1. Output scale matters.  Lowering `logit_scale` from 8 to 1 reduces CE from
   about 19 to about 5.4.
2. Even with BP diagnostic feedback, next-token accuracy remains poor, so the
   remaining mismatch is not just random feedback alignment.  The output target
   and readout adaptation need a paper-faithful rate/softmax bridge.

## Verdict

U005 is the strictest implementation path so far because the microcircuit
internals are directly from the cloned reference repository.  It is not yet a
usable LLM learner, but it establishes the correct integration point.

The next step should keep using the reference microcircuit and only adapt:

1. token output readout scale;
2. one-hot/rate target encoding;
3. context-to-`r0` encoding;
4. optional settling/reset policy.

The internal microcircuit logic should remain reference-owned.
