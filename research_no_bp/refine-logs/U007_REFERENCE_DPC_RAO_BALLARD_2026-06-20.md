# U007: Reference dPC / Rao-Ballard Predictive Coding Adapter

**Date**: 2026-06-20

## Purpose

Test the cloned source implementation of dendritic predictive coding (`dPC`) as
directly as possible, because the paper/source also compares against
Rao-Ballard-style predictive coding.

No gate, suppression module, repetition penalty, calibrated branch, parser, or
dataset-specific controller is added.  The local code only maps text documents
to the reference model interface:

```text
current token + position -> r0
next token -> u_tgt
reference dPC_model.evolve_system(...)
```

## Implementation

Added `u007_reference_dpc_llm_adapter.py`.

The script uses:

- `init_MC(..., mc_model="dPC")`;
- reference `dPC_model.evolve_system`;
- reference `dPC_model.evolve_voltages`;
- reference base predictive-coding `evolve_synapses`;
- optional source method `set_self_predicting_state`.

The default U007 path uses no unigram readout bias and no output patch.  A
unigram readout bridge remains available only as a diagnostic flag, but was not
used for the strict dPC smoke results below.

## Source Finding

The cloned reference dPC implementation has an important limitation:

```python
class dPC_model(base_model):
    ...
    super().__init__(..., BPP_init=WPP_init, ...)
```

For `model_type="FA"`, this leaves `BPP` with forward-weight shapes and causes a
matrix-shape failure on the first top-down prediction step for multilayer LLM
configs.

The reference experiment configs under
`Error-Neuron-Microcircuits/numpy_model/experiments/Fig3_multilayer_comparison/*/dPC/params.json`
use:

```text
mc_model = dPC
model_type = BP
init_in_SPS = true
```

So the source's working dPC comparison is a symmetric/aligned-feedback
predictive-coding diagnostic, not a runnable random-feedback FA no-BP path.

## Smoke Results

TinyStories smoke uses `train_chars=12000`, `valid_chars=3000`,
`max_vocab=128`, `d_model=64`, `blocks=4`, `presentation_steps=20`,
`predict_steps=5`, `eval_token_limit=600`, seed 0.

| Variant | Init / LR | Logit scale | Valid pre CE | Valid pre acc | Status |
|---|---|---:|---:|---:|---|
| dPC/FA | default | 1.0 | n/a | n/a | shape mismatch in source FA path |
| dPC/BP | default source scale | 1.0 | NaN | 0.160 | numerical overflow |
| dPC/BP + SPS | init0.02 eta0.0005/0.001 | 0.1 | 4.852 | 0.004 | stable uniform |
| dPC/BP + SPS | init0.02 eta0.0005/0.001 | 1.0 | 4.852 | 0.027 | stable uniform |
| dPC/BP + SPS | init0.1 eta0.001/0.002 | 1.0 | 4.852 | 0.017 | stable uniform |
| dPC/BP + SPS centered target | init0.1 eta0.001/0.002 | 1.0 | 4.852 | 0.017 | stable uniform |

`log(128) = 4.852`, so the stable dPC runs are effectively uniform softmax.

## Interpretation

The Rao-Ballard/dPC source path is now connected to the same LLM stream adapter,
but it does not yet learn useful next-token winners.

The strict result is negative:

1. dPC/FA cannot be run for deep LLM configs without changing the source's BPP
   initialization behavior.
2. dPC/BP is the source-supported Rao-Ballard diagnostic path, but default
   scales explode on token softmax.
3. Stable dPC/BP/SPS settings avoid overflow but remain uniform.

This does not support adding gates or suppression patches.  The next valid step,
if continuing this line, is a minimal source-level audit of dPC's `BPP_init`
shape handling and the source experiment's SPS/scale assumptions, not a new
model branch.
