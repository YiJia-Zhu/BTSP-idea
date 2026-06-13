#!/usr/bin/env python3
"""Create an analysis notebook for TinyStories next-token distributions."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "tinystories_next_token_distribution_analysis.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    md(
        """# TinyStories Next-Token Distribution Analysis

This notebook inspects *why* the local STDP/BTSP predictors behave differently from the LSTM.

The key point is not only total accuracy. For the same context, we compare the full next-character distributions:

- **LSTM**: stateful nonlinear context model trained to reduce next-token cross entropy.
- **STDP**: linear readout from a decayed bag of recent characters.
- **BTSP**: broader, positive, plateau-gated association from a decayed bag of recent characters.

If a method cannot condition on order-sensitive context, its top next-token distribution should stay similar across contexts that contain similar characters but in different order.
"""
    ),
    code(
        """from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('/private/zhenningshi/idea')
RUN_DIR = ROOT / 'output' / 'tinystories_nlp_large'
CKPT = RUN_DIR / 'checkpoint.npz'
CONFIG = RUN_DIR / 'config.json'
print('checkpoint exists:', CKPT.exists(), CKPT)
print('config exists:', CONFIG.exists(), CONFIG)
"""
    ),
    code(
        """ckpt = np.load(CKPT, allow_pickle=True)
itos = [str(x) for x in ckpt['itos'].tolist()]
stoi = {ch: i for i, ch in enumerate(itos)}
vocab_size = len(itos)

params = {
    'wx': ckpt['lstm_wx'],
    'wh': ckpt['lstm_wh'],
    'b': ckpt['lstm_b'],
    'w_out': ckpt['lstm_w_out'],
    'b_out': ckpt['lstm_b_out'],
}
hidden_dim = int(ckpt['hidden_dim'][0])
stdp_w = ckpt['stdp_weights']
btsp_w = ckpt['btsp_weights']
stdp_decay = float(ckpt['stdp_trace_decay'][0])
btsp_decay = float(ckpt['btsp_trace_decay'][0])
plastic_temp = float(ckpt['plastic_temperature'][0])

with open(CONFIG, 'r', encoding='utf-8') as f:
    config = json.load(f)

print('vocab_size:', vocab_size)
print('hidden_dim:', hidden_dim)
print('STDP weights:', stdp_w.shape, 'BTSP weights:', btsp_w.shape)
print('plastic temperature:', plastic_temp)
"""
    ),
    code(
        """def encode(text):
    return np.array([stoi[ch] for ch in text if ch in stoi], dtype=np.int64)


def pretty_char(ch):
    if ch == ' ':
        return '<space>'
    if ch == '\\n':
        return '<newline>'
    if ch == '\\t':
        return '<tab>'
    return ch


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def softmax(logits, temperature=1.0):
    z = logits / max(temperature, 1e-6)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    return float(-np.sum(p * np.log(p + 1e-12)))


def lstm_next_probs(context):
    ids = encode(context)
    h = np.zeros(hidden_dim, dtype=np.float32)
    c = np.zeros(hidden_dim, dtype=np.float32)
    for token in ids:
        z = params['wx'][:, token] + params['wh'] @ h + params['b']
        i = sigmoid(z[:hidden_dim])
        f = sigmoid(z[hidden_dim:2*hidden_dim])
        o = sigmoid(z[2*hidden_dim:3*hidden_dim])
        g = np.tanh(z[3*hidden_dim:])
        c = f * c + i * g
        h = o * np.tanh(c)
    logits = params['w_out'] @ h + params['b_out']
    return softmax(logits)


def plastic_next_probs(context, weights, decay, temperature=1.0):
    ids = encode(context)
    trace = np.zeros(vocab_size, dtype=np.float32)
    for token in ids:
        trace *= decay
        trace[token] += 1.0
    scores = weights @ trace
    return softmax(scores, temperature)


def topk(probs, k=12):
    order = np.argsort(-probs)[:k]
    return pd.DataFrame({
        'rank': np.arange(1, len(order) + 1),
        'char': [pretty_char(itos[i]) for i in order],
        'prob': [float(probs[i]) for i in order],
    })
"""
    ),
    md(
        """## Compare Next-Token Distributions

Edit the contexts below to inspect specific cases. Good examples are phrase stems where order matters:

- `Once upon a tim`
- `The little girl said, "`
- `she was very happ`
- `he went to the `
"""
    ),
    code(
        """contexts = [
    'Once upon a tim',
    'Once upon a time, there was a little ',
    'The little girl said, \"',
    'she was very happ',
    'he went to the ',
    'the cat sat on the ',
]

for context in contexts:
    print('\\nCONTEXT:', repr(context))
    for name, probs in [
        ('LSTM', lstm_next_probs(context)),
        ('STDP', plastic_next_probs(context, stdp_w, stdp_decay, plastic_temp)),
        ('BTSP', plastic_next_probs(context, btsp_w, btsp_decay, plastic_temp)),
    ]:
        print(f'  {name:4s}: entropy={entropy(probs):.3f}, top={pretty_char(itos[int(np.argmax(probs))])!r}, p_top={float(np.max(probs)):.3f}')
        display(topk(probs, 10))
"""
    ),
    code(
        """def plot_distributions_for_context(context, k=15):
    probs_by_name = {
        'LSTM': lstm_next_probs(context),
        'STDP': plastic_next_probs(context, stdp_w, stdp_decay, plastic_temp),
        'BTSP': plastic_next_probs(context, btsp_w, btsp_decay, plastic_temp),
    }
    union = []
    for probs in probs_by_name.values():
        for idx in np.argsort(-probs)[:k]:
            if idx not in union:
                union.append(int(idx))
    labels = [pretty_char(itos[i]) for i in union]
    x = np.arange(len(union))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(max(10, len(union) * 0.45), 4.2))
    for offset, (name, probs) in zip([-width, 0, width], probs_by_name.items()):
        ax.bar(x + offset, [probs[i] for i in union], width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('next-character probability')
    ax.set_title(f'Next-token distribution after: {context!r}')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    return fig


plot_distributions_for_context('Once upon a tim')
plot_distributions_for_context('The little girl said, \"')
plot_distributions_for_context('he went to the ')
"""
    ),
    md(
        """## Context Order Sensitivity

This section checks whether the predictor changes when the same characters appear in a different order.

STDP/BTSP use a decayed character trace, so they retain **some recency**, but they still collapse context into a near bag-of-recent-characters. LSTM can map different orderings to different hidden states.
"""
    ),
    code(
        """pairs = [
    ('the ', 'hte '),
    ('said ', 'dias '),
    ('little ', 'tlelit '),
    ('Once upon ', 'upon Once '),
    ('girl said ', 'said girl '),
]

def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log((p + 1e-12) / (m + 1e-12))) + 0.5 * np.sum(q * np.log((q + 1e-12) / (m + 1e-12)))

rows = []
for a, b in pairs:
    for name, fn in [
        ('LSTM', lstm_next_probs),
        ('STDP', lambda x: plastic_next_probs(x, stdp_w, stdp_decay, plastic_temp)),
        ('BTSP', lambda x: plastic_next_probs(x, btsp_w, btsp_decay, plastic_temp)),
    ]:
        pa = fn(a)
        pb = fn(b)
        rows.append({
            'context_a': a,
            'context_b': b,
            'method': name,
            'JS_divergence': js_divergence(pa, pb),
            'top_a': pretty_char(itos[int(np.argmax(pa))]),
            'top_b': pretty_char(itos[int(np.argmax(pb))]),
        })

pd.DataFrame(rows)
"""
    ),
    md(
        """## Ground-Truth Validation Examples

For actual validation snippets, this cell compares whether each method assigns high probability to the real next character. Look for cases where the LSTM uses the phrase context, while STDP/BTSP prefer frequent local characters.
"""
    ),
    code(
        """valid_path = Path(config['valid_file'])
valid_text = valid_path.read_text(encoding='utf-8', errors='replace')[:int(config['valid_chars'])]

def score_context(context, target):
    rows = []
    for name, probs in [
        ('LSTM', lstm_next_probs(context)),
        ('STDP', plastic_next_probs(context, stdp_w, stdp_decay, plastic_temp)),
        ('BTSP', plastic_next_probs(context, btsp_w, btsp_decay, plastic_temp)),
    ]:
        if target not in stoi:
            continue
        target_id = stoi[target]
        rows.append({
            'method': name,
            'target': pretty_char(target),
            'p_target': float(probs[target_id]),
            'rank_target': int(np.where(np.argsort(-probs) == target_id)[0][0]) + 1,
            'top': pretty_char(itos[int(np.argmax(probs))]),
            'p_top': float(np.max(probs)),
            'entropy': entropy(probs),
        })
    return pd.DataFrame(rows)

examples = []
for start in [0, 200, 500, 900, 1500, 2500, 4000]:
    context = valid_text[start:start+60]
    target = valid_text[start+60]
    examples.append((context, target))

for context, target in examples:
    print('\\nCONTEXT:', repr(context))
    print('TARGET:', repr(pretty_char(target)))
    display(score_context(context, target))
"""
    ),
    md(
        """## Interpretation Checklist

When you inspect the plots/tables, look for these failure modes:

1. **Frequency bias**: STDP/BTSP often give high mass to common characters such as space, `e`, `t`, `a`, regardless of syntactic context.
2. **Order collapse**: Reordered contexts can produce similar STDP/BTSP distributions because the state is a decayed trace, not a learned nonlinear sequence representation.
3. **Wide-window interference**: BTSP's broader trace can help long association, but in natural text it also binds many irrelevant recent characters.
4. **Probability calibration**: STDP/BTSP weights are association strengths, not logits trained by cross entropy, so their softmax distributions can be over/under confident.
5. **LSTM hidden state**: LSTM can assign probability based on phrase-level state, e.g. after `Once upon a tim`, `e` should dominate.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(NOTEBOOK)
