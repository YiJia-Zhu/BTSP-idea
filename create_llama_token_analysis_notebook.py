#!/usr/bin/env python3
"""Create a notebook for Llama-token next-token distribution analysis."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "tinystories_llama_token_distribution_analysis.ipynb"


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
        """# TinyStories Llama-Tokenizer Next-Token Analysis

This notebook replaces the earlier character-level inspection with a token-level setup using the local Llama tokenizer.

Important caveat: the NumPy experiment uses a compact vocabulary of frequent Llama tokenizer IDs, so rare tokens are filtered out. This keeps STDP/BTSP matrices and the LSTM softmax tractable while preserving token-level next-token behavior.
"""
    ),
    code(
        """from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path('/private/zhenningshi/idea')
RUN_DIR = ROOT / 'output' / 'tinystories_llama_token'
CKPT = RUN_DIR / 'checkpoint.npz'
CONFIG = RUN_DIR / 'config.json'
ckpt = np.load(CKPT, allow_pickle=True)
with open(CONFIG, 'r', encoding='utf-8') as f:
    config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_dir'], local_files_only=True)
kept_raw = ckpt['kept_raw_ids'].astype(np.int64)
raw_to_compact = {int(raw): i for i, raw in enumerate(kept_raw)}
compact_to_raw = {i: int(raw) for i, raw in enumerate(kept_raw)}
vocab_size = len(kept_raw)

params = {
    'emb': ckpt['emb'],
    'wx': ckpt['wx'],
    'wh': ckpt['wh'],
    'b': ckpt['b'],
    'w_out': ckpt['w_out'],
    'b_out': ckpt['b_out'],
}
stdp_w = ckpt['stdp_weights']
btsp_w = ckpt['btsp_weights']
stdp_decay = float(ckpt['stdp_trace_decay'][0])
btsp_decay = float(ckpt['btsp_trace_decay'][0])
plastic_temp = float(ckpt['plastic_temperature'][0])
hidden_dim = params['wh'].shape[1]
print('compact vocab:', vocab_size)
print('hidden_dim:', hidden_dim)
print('metrics:')
display(pd.read_csv(RUN_DIR / 'metrics.csv'))
"""
    ),
    code(
        """def encode_compact(text):
    raw = tokenizer.encode(text, add_special_tokens=False)
    return np.array([raw_to_compact[t] for t in raw if t in raw_to_compact], dtype=np.int64)


def token_label(compact_id):
    raw_id = compact_to_raw[int(compact_id)]
    text = tokenizer.decode([raw_id])
    text = text.replace('\\n', '\\\\n')
    return f'{compact_id}:{text!r}'


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


def lstm_probs(context):
    ids = encode_compact(context)
    h = np.zeros(hidden_dim, dtype=np.float32)
    c = np.zeros(hidden_dim, dtype=np.float32)
    for token in ids:
        x = params['emb'][token]
        z = params['wx'] @ x + params['wh'] @ h + params['b']
        i = sigmoid(z[:hidden_dim])
        f = sigmoid(z[hidden_dim:2*hidden_dim])
        o = sigmoid(z[2*hidden_dim:3*hidden_dim])
        g = np.tanh(z[3*hidden_dim:])
        c = f * c + i * g
        h = o * np.tanh(c)
    return softmax(params['w_out'] @ h + params['b_out'])


def plastic_probs(context, weights, decay, temperature=1.0):
    ids = encode_compact(context)
    trace = np.zeros(vocab_size, dtype=np.float32)
    for token in ids:
        trace *= decay
        trace[token] += 1.0
    return softmax(weights @ trace, temperature)


def topk(probs, k=12):
    order = np.argsort(-probs)[:k]
    return pd.DataFrame({
        'rank': np.arange(1, len(order) + 1),
        'compact_id': order,
        'token': [token_label(i) for i in order],
        'prob': [float(probs[i]) for i in order],
    })
"""
    ),
    md(
        """## Same Context, Three Next-Token Distributions

Edit the contexts below. Unlike the char-level notebook, tokens are Llama tokenizer tokens, so top predictions are subwords/words/punctuation rather than characters.
"""
    ),
    code(
        """contexts = [
    'Once upon a time',
    'Once upon a time, there was a little',
    'The little girl said',
    'She was very happy',
    'He went to the',
    'The dog ran into the',
]

for context in contexts:
    print('\\nCONTEXT:', repr(context))
    print('compact ids:', encode_compact(context).tolist())
    for name, probs in [
        ('LSTM', lstm_probs(context)),
        ('STDP', plastic_probs(context, stdp_w, stdp_decay, plastic_temp)),
        ('BTSP', plastic_probs(context, btsp_w, btsp_decay, plastic_temp)),
    ]:
        print(f'  {name}: entropy={entropy(probs):.3f}, top={token_label(int(np.argmax(probs)))}, p_top={float(np.max(probs)):.3f}')
        display(topk(probs, 10))
"""
    ),
    code(
        """def plot_distributions_for_context(context, k=12):
    probs_by_name = {
        'LSTM': lstm_probs(context),
        'STDP': plastic_probs(context, stdp_w, stdp_decay, plastic_temp),
        'BTSP': plastic_probs(context, btsp_w, btsp_decay, plastic_temp),
    }
    union = []
    for probs in probs_by_name.values():
        for idx in np.argsort(-probs)[:k]:
            if int(idx) not in union:
                union.append(int(idx))
    labels = [token_label(i) for i in union]
    x = np.arange(len(union))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(12, len(union) * 0.7), 4.5))
    for offset, (name, probs) in zip([-width, 0, width], probs_by_name.items()):
        ax.bar(x + offset, [probs[i] for i in union], width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha='right')
    ax.set_ylabel('next-token probability')
    ax.set_title(f'Llama-token distribution after: {context!r}')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    return fig

plot_distributions_for_context('Once upon a time')
plot_distributions_for_context('Once upon a time, there was a little')
plot_distributions_for_context('The little girl said')
"""
    ),
    md(
        """## Greedy Continuation

This cell greedily continues English prompts with all three methods: LSTM, STDP and BTSP.

This run was trained on English TinyStories and keeps frequent English-heavy tokens, so the examples below use English prompts.
"""
    ),
    code(
        """def step_lstm_token(current, h, c):
    x = params['emb'][current]
    z = params['wx'] @ x + params['wh'] @ h + params['b']
    i = sigmoid(z[:hidden_dim])
    f = sigmoid(z[hidden_dim:2*hidden_dim])
    o = sigmoid(z[2*hidden_dim:3*hidden_dim])
    g = np.tanh(z[3*hidden_dim:])
    c = f * c + i * g
    h = o * np.tanh(c)
    probs = softmax(params['w_out'] @ h + params['b_out'])
    return int(np.argmax(probs)), h, c, probs


def decode_compact(compact_tokens):
    raw_tokens = [compact_to_raw[int(i)] for i in compact_tokens]
    return tokenizer.decode(raw_tokens, skip_special_tokens=True)


def greedy_lstm(prompt, steps=50):
    raw_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    compact_prompt = [raw_to_compact[t] for t in raw_prompt if t in raw_to_compact]
    h = np.zeros(hidden_dim, dtype=np.float32)
    c = np.zeros(hidden_dim, dtype=np.float32)
    if not compact_prompt:
        return '[prompt has no compact-vocab tokens]'
    generated = list(compact_prompt)
    for token in compact_prompt[:-1]:
        _, h, c, _ = step_lstm_token(int(token), h, c)
    current = int(compact_prompt[-1])
    for _ in range(steps):
        current, h, c, _ = step_lstm_token(current, h, c)
        generated.append(current)
    return decode_compact(generated)


def greedy_plastic(prompt, weights, decay, steps=50):
    raw_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    compact_prompt = [raw_to_compact[t] for t in raw_prompt if t in raw_to_compact]
    if not compact_prompt:
        return '[prompt has no compact-vocab tokens]'
    trace = np.zeros(vocab_size, dtype=np.float32)
    generated = list(compact_prompt)
    for token in compact_prompt:
        trace *= decay
        trace[int(token)] += 1.0
    for _ in range(steps):
        scores = weights @ trace
        current = int(np.argmax(scores))
        generated.append(current)
        trace *= decay
        trace[current] += 1.0
    return decode_compact(generated)


for prompt in ['Once upon a time', 'The little girl', 'Tom and Lily went to the']:
    print('\nPROMPT:', repr(prompt))
    raw_prompt = tokenizer.encode(prompt, add_special_tokens=False)
    print('raw prompt tokens:', [tokenizer.decode([t]) for t in raw_prompt])
    print('kept:', [t in raw_to_compact for t in raw_prompt])
    print('\nLSTM greedy:')
    print(greedy_lstm(prompt, steps=60))
    print('\nSTDP greedy:')
    print(greedy_plastic(prompt, stdp_w, stdp_decay, steps=60))
    print('\nBTSP greedy:')
    print(greedy_plastic(prompt, btsp_w, btsp_decay, steps=60))
"""
    ),
    md(
        """## Why STDP/BTSP Still Lose at Token-Level Next Prediction

The token-level setup fixes the tokenizer issue, but the structural mismatch remains:

- STDP/BTSP are linear association memories over a decayed token trace.
- LSTM maps ordered token sequences into a nonlinear hidden state.
- Cross entropy training teaches the LSTM which token distinctions matter for prediction.
- Plasticity matrices do not receive a corrective signal when they put probability mass on plausible-but-wrong high-frequency tokens.
"""
    ),
    code(
        """valid_path = Path(config['valid_file'])
valid_text = valid_path.read_text(encoding='utf-8', errors='replace')[:int(config['valid_chars'])]
valid_raw = tokenizer.encode(valid_text, add_special_tokens=False)
valid_compact = [raw_to_compact[t] for t in valid_raw if t in raw_to_compact]

def context_from_compact_window(end_idx, window=12):
    compact_context = valid_compact[max(0, end_idx-window):end_idx]
    raw_context = [compact_to_raw[i] for i in compact_context]
    return tokenizer.decode(raw_context)

def score_context(context, target_compact):
    rows = []
    for name, probs in [
        ('LSTM', lstm_probs(context)),
        ('STDP', plastic_probs(context, stdp_w, stdp_decay, plastic_temp)),
        ('BTSP', plastic_probs(context, btsp_w, btsp_decay, plastic_temp)),
    ]:
        order = np.argsort(-probs)
        rows.append({
            'method': name,
            'target': token_label(target_compact),
            'p_target': float(probs[target_compact]),
            'rank_target': int(np.where(order == target_compact)[0][0]) + 1,
            'top': token_label(int(order[0])),
            'p_top': float(probs[order[0]]),
            'entropy': entropy(probs),
        })
    return pd.DataFrame(rows)

for end_idx in [50, 120, 240, 500, 900, 1300]:
    context = context_from_compact_window(end_idx, 12)
    target = int(valid_compact[end_idx])
    print('\\nCONTEXT:', repr(context))
    display(score_context(context, target))
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
