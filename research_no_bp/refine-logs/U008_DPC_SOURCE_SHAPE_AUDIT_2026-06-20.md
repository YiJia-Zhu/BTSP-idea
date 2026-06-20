# U008: Minimal dPC Source Shape Audit

**Date**: 2026-06-20

## Purpose

Continue the Rao-Ballard/dPC line without adding gates, suppression, repetition
penalties, or new model modules.

U007 showed that the cloned source `dPC_model` could not run `model_type=FA` for
deep LLM configs because the top-down `BPP` matrices had forward-weight shapes.
U008 makes the smallest source-level shape correction needed to run the source
dPC/FA path.

## Source Changes

Changed `Error-Neuron-Microcircuits/numpy_model/src/init_MC.py`:

```python
if mc_model in ['sacramento2018', 'ann', 'dPC']:
    BPP_range = params['init_BPP_range']
...
if mc_model in ['sacramento2018', 'ann', 'dPC']:
    BPP_init = [...]
```

Changed `Error-Neuron-Microcircuits/numpy_model/src/microcircuit.py` inside
`dPC_model.__init__`:

```python
BPP_init=BPP_init
```

instead of:

```python
BPP_init=WPP_init
```

This is a source shape fix only.  No learning rule, gate, readout patch, or
dataset-specific logic was added.

## Shape Check

For a dPC/FA LLM config with layers `[16, 16, 16, 64]`, after the fix:

```text
WPP [(16, 16), (16, 16), (64, 16)]
BPP [(16, 16), (16, 64)]
WIP [(16, 16), (64, 16)]
BPI [(16, 16), (16, 16)]
```

The FA path now runs without the original matrix-shape failure.

## Smoke Results

### TinyStories tiny shape smoke

```text
task=tinystories
vocab=64
d_model=16
blocks=2
presentation_steps=2
predict_steps=1
eval_token_limit=50
model_type=FA
```

Result:

```text
valid_pre CE / acc = 4.159 / 0.000
```

`log(64) = 4.159`, so this is uniform.

### TinyStories d64/b4 smoke

```text
task=tinystories
vocab=128
d_model=64
blocks=4
presentation_steps=20
predict_steps=5
eval_token_limit=600
model_type=FA
```

| Variant | Valid pre CE | Valid pre acc |
|---|---:|---:|
| init0.02 eta0.0005/0.001 | 4.852 | 0.025 |
| init0.1 eta0.001/0.002 | 4.852 | 0.017 |
| init0.1 + SPS + BPI local update | 4.852 | 0.017 |

`log(128) = 4.852`, so all three remain uniform.

### Mixed TinyStories+GSM8k-Aug smoke

```text
task=mix
vocab=64
d_model=32
blocks=2
presentation_steps=5
predict_steps=2
eval_token_limit=120
model_type=FA
```

Result:

```text
valid_pre CE / acc = 4.159 / 0.008
```

Again, this is essentially uniform.

## Interpretation

The source-level dPC/FA shape problem is fixed, and the path can now run in the
LLM adapter.  But fixing shapes does not produce useful next-token learning.

The current dPC/Rao-Ballard path remains a negative boundary:

1. U007 dPC/BP/SPS was stable only at tiny scales and stayed uniform.
2. U008 dPC/FA now runs but also stays uniform.
3. Larger init, larger learning rate, centered target, and SPS do not create a
   next-token winner signal in these smoke tests.

The next valid step is not a gate.  It should be a source-faithful reproduction
of the original dPC Fig3 task after this shape fix, then a comparison of source
task behavior versus LLM token behavior.
