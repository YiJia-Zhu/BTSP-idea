#!/usr/bin/env python3
"""
TinyStories token-level next-token experiment with the local Llama tokenizer.

This is closer to LLM training than the character-level sanity check:
  - text is encoded with /private/zhenningshi/model_weights/Llama-3.2-1B-Instruct
  - the experiment uses a compact vocab of frequent tokenizer IDs to keep NumPy
    plasticity matrices and the random-init Llama-style baseline small
  - STDP/BTSP and their bio-structured variants use no backpropagation
  - target is next tokenizer token, not next character
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from llama_torch_model import LlamaTorchCausalLM, LlamaTorchConfig, llama1b_body_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = SCRIPT_DIR / "data" / "TinyStories-train.txt"
DEFAULT_VALID = SCRIPT_DIR / "data" / "TinyStories-valid.txt"
DEFAULT_TOKENIZER = Path("/private/zhenningshi/model_weights/Llama-3.2-1B-Instruct")
FALLBACK_SAMPLE_PROMPTS = [
    "A bright spoon rolled beside the window",
    "The blue door opened under the chair",
    "A small painter carried a candle into the garden",
]


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--"
    seconds = int(max(seconds, 0.0))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


class ProgressBar:
    def __init__(self, label: str, total: int, enabled: bool = True, min_interval: float = 1.0) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.enabled = enabled
        self.min_interval = min_interval
        self.start = time.time()
        self.last_update = 0.0
        self.current = 0
        self.closed = False
        self.update(0, force=True)

    def update(self, current: int, force: bool = False) -> None:
        if not self.enabled or self.closed:
            return
        current = min(max(int(current), 0), self.total)
        now = time.time()
        if not force and current < self.total and now - self.last_update < self.min_interval:
            return
        self.current = current
        elapsed = max(now - self.start, 1e-9)
        rate = current / elapsed if current > 0 else 0.0
        eta = (self.total - current) / rate if rate > 0.0 and current < self.total else float("inf")
        percent = 100.0 * current / self.total
        sys.stderr.write(
            f"\r{self.label}: {current:,}/{self.total:,}"
            f" ({percent:5.1f}%) {rate:,.1f}/s ETA {format_duration(eta)}"
        )
        sys.stderr.flush()
        self.last_update = now

    def close(self) -> None:
        if self.closed:
            return
        if self.current < self.total:
            self.update(self.total, force=True)
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()
        self.closed = True


def sync_torch_device(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        sync_torch_device(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        sync_torch_device(self.device)
        self.elapsed = time.perf_counter() - self.start


def add_timing(
    metrics: dict,
    train_seconds: float,
    eval_seconds: float,
    train_tokens: int,
    eval_tokens: int,
    device: str,
) -> dict:
    row = dict(metrics)
    row["train_seconds"] = train_seconds
    row["eval_seconds"] = eval_seconds
    row["train_tokens"] = train_tokens
    row["eval_tokens"] = eval_tokens
    row["train_tokens_per_sec"] = train_tokens / max(train_seconds, 1e-9)
    row["eval_tokens_per_sec"] = eval_tokens / max(eval_seconds, 1e-9)
    row["device"] = device
    return row


@dataclass
class STDPConfig:
    a_plus: float = 0.008
    a_minus: float = 0.003
    trace_decay: float = 0.65
    row_decay: float = 0.9995
    epochs: int = 1


@dataclass
class BTSPConfig:
    potentiation: float = 0.008
    heterosynaptic_depression: float = 0.001
    trace_decay: float = 0.82
    row_decay: float = 0.9995
    max_weight: float = 8.0
    epochs: int = 1


@dataclass
class BioStructureConfig:
    """Shared sparse directed token-graph constraints for bio variants."""

    row_norm: float = 12.0
    max_active_inputs: int = 64
    trace_floor: float = 0.015


@dataclass
class DendriticErrorConfig:
    """Simplified 1810-style dendritic local error estimator."""

    trace_decay: float = 0.85
    lr_in: float = 0.020
    lr_out: float = 0.020
    lambda_apical: float = 0.35
    grad_clip: float = 2.0
    feedback_mode: str = "fixed"


@dataclass
class RecurrentThreeFactorConfig:
    """Online e-prop/DFA-style recurrent local learning rule."""

    eligibility_decay: float = 0.92
    lr_hidden: float = 0.003
    lr_out: float = 0.020
    rec_gain: float = 0.55
    input_scale: float = 0.35
    feedback_scale: float = 1.0
    grad_clip: float = 1.5
    weight_clip: float = 4.0
    feedback_mode: str = "fixed"


@dataclass
class SparseHebbianContextConfig:
    """Sparse online Hebbian context-to-next-token associative memory."""

    max_order: int = 4
    alpha: float = 0.05
    unigram_weight: float = 0.15
    order_weight: float = 1.0
    order_weight_growth: float = 1.6
    score_mode: str = "additive"
    smoothing: float = 0.05
    backoff: float = 0.35


@dataclass
class SemanticHebbianConfig:
    """Random-projection semantic buckets with Hebbian next-token counts."""

    order: int = 8
    dim: int = 64
    hash_bits: int = 12
    alpha: float = 0.05
    bucket_weight: float = 1.0
    unigram_weight: float = 0.15
    seed: int = 0


@dataclass
class HybridBackoffConfig:
    """No-BP hybrid adapter: neural backoff logits plus Hebbian memory logits."""

    memory_weight: float = 1.0
    neural_weight: float = 1.0


@dataclass
class GatedBackoffConfig:
    """Confidence gate for using neural backoff only when memory is uncertain."""

    min_order: int = 4
    min_row_total: float = 0.20
    min_max_prob: float = 0.55
    max_entropy: float = 1.50


@dataclass
class DecodingConfig:
    """Inference-time controls for reducing local associative-memory loops."""

    repetition_penalty: float = 0.0
    no_repeat_ngram: int = 0
    top_k: int = 0
    temperature: float = 1.0


@dataclass
class MethodTiming:
    train_seconds: float
    eval_seconds: float
    train_tokens: int
    eval_tokens: int
    device: str


def read_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


def normalize_text(text: str) -> str:
    return " ".join(text.split()).lower()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(temperature, 1e-6)
    z = z - float(np.max(z))
    exp_z = np.exp(z)
    return (exp_z / np.sum(exp_z)).astype(np.float32)


def cross_entropy_from_scores(scores: np.ndarray, target: int, temperature: float = 1.0) -> tuple[float, int]:
    probs = softmax(scores, temperature)
    return -math.log(float(probs[target]) + 1e-9), int(np.argmax(probs))


def encode_text(tokenizer, text: str) -> np.ndarray:
    return np.array(tokenizer.encode(text, add_special_tokens=False), dtype=np.int64)


def build_compact_vocab(train_raw: np.ndarray, valid_raw: np.ndarray, max_vocab: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(train_raw)
    max_vocab = min(max_vocab, counts.shape[0])
    kept_raw = np.argsort(-counts)[:max_vocab]
    kept_raw = kept_raw[np.argsort(kept_raw)]
    raw_to_compact = np.full(max(int(max(train_raw.max(), valid_raw.max())) + 1, kept_raw.max() + 1), -1, dtype=np.int64)
    raw_to_compact[kept_raw] = np.arange(len(kept_raw), dtype=np.int64)
    train_compact = raw_to_compact[train_raw]
    valid_compact = raw_to_compact[valid_raw]
    train_compact = train_compact[train_compact >= 0].astype(np.int64)
    valid_compact = valid_compact[valid_compact >= 0].astype(np.int64)
    return kept_raw.astype(np.int64), train_compact, valid_compact


def build_probe_traces(
    ids: np.ndarray,
    vocab_size: int,
    trace_decay: float,
    max_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    limit = min(len(ids) - 1, max_tokens)
    traces = np.zeros((limit, vocab_size), dtype=np.float32)
    targets = np.zeros(limit, dtype=np.int64)
    trace = np.zeros(vocab_size, dtype=np.float32)

    for idx in range(limit):
        current = int(ids[idx])
        targets[idx] = int(ids[idx + 1])
        trace *= trace_decay
        trace[current] += 1.0
        traces[idx] = trace

    return traces, targets


def batch_cross_entropy(scores: np.ndarray, targets: np.ndarray, temperature: float) -> tuple[float, float]:
    if scores.shape[0] == 0:
        return float("nan"), 0.0
    z = scores / max(temperature, 1e-6)
    z = z - np.max(z, axis=1, keepdims=True)
    log_denom = np.log(np.sum(np.exp(z), axis=1) + 1e-9)
    target_scores = z[np.arange(len(targets)), targets]
    loss = float(np.mean(log_denom - target_scores))
    acc = float(np.mean(np.argmax(z, axis=1) == targets))
    return loss, acc


def evaluate_plastic_probe(
    weights: np.ndarray,
    traces: np.ndarray,
    targets: np.ndarray,
    temperature: float,
) -> tuple[float, float]:
    scores = traces @ weights.T
    return batch_cross_entropy(scores, targets, temperature)


def evaluate_dendritic_probe(
    model: DendriticErrorNetwork,
    traces: np.ndarray,
    targets: np.ndarray,
    temperature: float,
) -> tuple[float, float]:
    hidden = np.tanh(traces @ model.w_in.T + model.b_h).astype(np.float32)
    scores = hidden @ model.w_out.T + model.b_out
    return batch_cross_entropy(scores, targets, temperature)


def should_record_curve(step: int, total_steps: int, every: int) -> bool:
    if every <= 0:
        return False
    if step == 0 or step == total_steps:
        return True
    return step % every == 0 and step + every < total_steps


def append_curve_row(
    rows: list[dict] | None,
    method: str,
    step: int,
    loss: float,
    accuracy: float,
) -> None:
    if rows is None:
        return
    rows.append({"method": method, "step": step, "loss": loss, "accuracy": accuracy})


def train_stdp_matrix(
    ids: np.ndarray,
    vocab_size: int,
    cfg: STDPConfig,
    show_progress: bool = True,
    curve_rows: list[dict] | None = None,
    curve_probe: tuple[np.ndarray, np.ndarray] | None = None,
    curve_every: int = 0,
    curve_temperature: float = 1.0,
    curve_method: str = "stdp_trace",
) -> np.ndarray:
    weights = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    trace = np.zeros(vocab_size, dtype=np.float32)
    total_steps = cfg.epochs * len(ids)
    progress = ProgressBar("STDP train", total_steps, show_progress)
    step = 0
    if curve_probe is not None and should_record_curve(step, total_steps, curve_every):
        loss, acc = evaluate_plastic_probe(weights, curve_probe[0], curve_probe[1], curve_temperature)
        append_curve_row(curve_rows, curve_method, step, loss, acc)

    for _ in range(cfg.epochs):
        trace.fill(0.0)
        for token in ids:
            token = int(token)
            weights[token] *= cfg.row_decay
            weights[token] += cfg.a_plus * trace
            weights[:, token] -= cfg.a_minus * trace
            trace *= cfg.trace_decay
            trace[token] += 1.0
            step += 1
            progress.update(step)
            if curve_probe is not None and should_record_curve(step, total_steps, curve_every):
                loss, acc = evaluate_plastic_probe(weights, curve_probe[0], curve_probe[1], curve_temperature)
                append_curve_row(curve_rows, curve_method, step, loss, acc)

    progress.close()
    return weights


def train_btsp_matrix(
    ids: np.ndarray,
    vocab_size: int,
    cfg: BTSPConfig,
    show_progress: bool = True,
    curve_rows: list[dict] | None = None,
    curve_probe: tuple[np.ndarray, np.ndarray] | None = None,
    curve_every: int = 0,
    curve_temperature: float = 1.0,
    curve_method: str = "btsp_trace",
) -> np.ndarray:
    weights = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    trace = np.zeros(vocab_size, dtype=np.float32)
    total_steps = cfg.epochs * len(ids)
    progress = ProgressBar("BTSP train", total_steps, show_progress)
    step = 0
    if curve_probe is not None and should_record_curve(step, total_steps, curve_every):
        loss, acc = evaluate_plastic_probe(weights, curve_probe[0], curve_probe[1], curve_temperature)
        append_curve_row(curve_rows, curve_method, step, loss, acc)

    for _ in range(cfg.epochs):
        trace.fill(0.0)
        for token in ids:
            token = int(token)
            weights[token] *= cfg.row_decay
            weights[token] += cfg.potentiation * trace
            inactive_trace = 1.0 - np.clip(trace, 0.0, 1.0)
            weights[token] -= cfg.heterosynaptic_depression * inactive_trace * weights[token]
            np.clip(weights[token], 0.0, cfg.max_weight, out=weights[token])
            trace *= cfg.trace_decay
            trace[token] += 1.0
            step += 1
            progress.update(step)
            if curve_probe is not None and should_record_curve(step, total_steps, curve_every):
                loss, acc = evaluate_plastic_probe(weights, curve_probe[0], curve_probe[1], curve_temperature)
                append_curve_row(curve_rows, curve_method, step, loss, acc)

    progress.close()
    return weights


def prune_row(row: np.ndarray, max_active: int) -> None:
    if max_active <= 0 or max_active >= row.shape[0]:
        return
    active = np.flatnonzero(np.abs(row) > 0.0)
    if len(active) <= max_active:
        return
    keep = np.argpartition(np.abs(row), -max_active)[-max_active:]
    mask = np.ones(row.shape[0], dtype=bool)
    mask[keep] = False
    row[mask] = 0.0


def normalize_row(row: np.ndarray, row_norm: float) -> None:
    norm = float(np.sum(np.abs(row)))
    if norm > row_norm:
        row *= row_norm / (norm + 1e-8)


def train_stdp_bio_matrix(
    ids: np.ndarray,
    vocab_size: int,
    stdp_cfg: STDPConfig,
    bio_cfg: BioStructureConfig,
    show_progress: bool = True,
    curve_rows: list[dict] | None = None,
    curve_probe: tuple[np.ndarray, np.ndarray] | None = None,
    curve_every: int = 0,
    curve_temperature: float = 1.0,
    curve_method: str = "stdp_bio",
) -> np.ndarray:
    weights = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    trace = np.zeros(vocab_size, dtype=np.float32)
    total_steps = stdp_cfg.epochs * len(ids)
    progress = ProgressBar("STDP-Bio train", total_steps, show_progress)
    step = 0
    if curve_probe is not None and should_record_curve(step, total_steps, curve_every):
        loss, acc = evaluate_plastic_probe(weights, curve_probe[0], curve_probe[1], curve_temperature)
        append_curve_row(curve_rows, curve_method, step, loss, acc)

    for _ in range(stdp_cfg.epochs):
        trace.fill(0.0)
        for token in ids:
            token = int(token)
            active = trace > bio_cfg.trace_floor
            weights[token] *= stdp_cfg.row_decay
            if np.any(active):
                weights[token, active] += stdp_cfg.a_plus * trace[active]
                weights[active, token] -= stdp_cfg.a_minus * trace[active]
            prune_row(weights[token], bio_cfg.max_active_inputs)
            normalize_row(weights[token], bio_cfg.row_norm)
            trace *= stdp_cfg.trace_decay
            trace[token] += 1.0
            step += 1
            progress.update(step)
            if curve_probe is not None and should_record_curve(step, total_steps, curve_every):
                loss, acc = evaluate_plastic_probe(weights, curve_probe[0], curve_probe[1], curve_temperature)
                append_curve_row(curve_rows, curve_method, step, loss, acc)

    progress.close()
    return weights


def train_btsp_bio_matrix(
    ids: np.ndarray,
    vocab_size: int,
    btsp_cfg: BTSPConfig,
    bio_cfg: BioStructureConfig,
    show_progress: bool = True,
    curve_rows: list[dict] | None = None,
    curve_probe: tuple[np.ndarray, np.ndarray] | None = None,
    curve_every: int = 0,
    curve_temperature: float = 1.0,
    curve_method: str = "btsp_bio",
) -> np.ndarray:
    weights = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    trace = np.zeros(vocab_size, dtype=np.float32)
    total_steps = btsp_cfg.epochs * len(ids)
    progress = ProgressBar("BTSP-Bio train", total_steps, show_progress)
    step = 0
    if curve_probe is not None and should_record_curve(step, total_steps, curve_every):
        loss, acc = evaluate_plastic_probe(weights, curve_probe[0], curve_probe[1], curve_temperature)
        append_curve_row(curve_rows, curve_method, step, loss, acc)

    for _ in range(btsp_cfg.epochs):
        trace.fill(0.0)
        for token in ids:
            token = int(token)
            active = trace > bio_cfg.trace_floor
            weights[token] *= btsp_cfg.row_decay
            if np.any(active):
                weights[token, active] += btsp_cfg.potentiation * trace[active]
            inactive_trace = 1.0 - np.clip(trace, 0.0, 1.0)
            weights[token] -= btsp_cfg.heterosynaptic_depression * inactive_trace * weights[token]
            np.clip(weights[token], 0.0, btsp_cfg.max_weight, out=weights[token])
            prune_row(weights[token], bio_cfg.max_active_inputs)
            normalize_row(weights[token], bio_cfg.row_norm)
            trace *= btsp_cfg.trace_decay
            trace[token] += 1.0
            step += 1
            progress.update(step)
            if curve_probe is not None and should_record_curve(step, total_steps, curve_every):
                loss, acc = evaluate_plastic_probe(weights, curve_probe[0], curve_probe[1], curve_temperature)
                append_curve_row(curve_rows, curve_method, step, loss, acc)

    progress.close()
    return weights


class DendriticErrorNetwork:
    """
    Minimal Sacramento-style local-error model for token prediction.

    Architecture:
        token trace -> basal hidden compartment -> output logits
        output error -> fixed feedback -> hidden apical compartment

    Hidden input weights are updated with a local dendritic error estimate:
        (phi(basal + lambda * feedback_error) - phi(basal)) outer trace
    Output weights use the local output error. No BPTT or backprop through hidden
    weights is used; the feedback path is the explicit credit signal.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, cfg: DendriticErrorConfig, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.cfg = cfg
        self.w_in = rng.normal(0.0, 1.0 / math.sqrt(vocab_size), (hidden_dim, vocab_size)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)
        self.w_out = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), (vocab_size, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        if cfg.feedback_mode == "symmetric":
            self.feedback = self.w_out.T.copy()
        else:
            self.feedback = rng.normal(0.0, 1.0 / math.sqrt(vocab_size), (hidden_dim, vocab_size)).astype(np.float32)

    def hidden(self, trace: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        basal = self.w_in @ trace + self.b_h
        return basal, np.tanh(basal).astype(np.float32)

    def scores_from_trace(self, trace: np.ndarray) -> np.ndarray:
        _, hidden = self.hidden(trace)
        return self.w_out @ hidden + self.b_out


def train_dendritic_error(
    ids: np.ndarray,
    vocab_size: int,
    hidden_dim: int,
    cfg: DendriticErrorConfig,
    seed: int,
    show_progress: bool = True,
    curve_rows: list[dict] | None = None,
    curve_probe: tuple[np.ndarray, np.ndarray] | None = None,
    curve_every: int = 0,
    curve_temperature: float = 1.0,
    curve_method: str = "dendritic_error_1810_lite",
) -> DendriticErrorNetwork:
    model = DendriticErrorNetwork(vocab_size, hidden_dim, cfg, seed)
    trace = np.zeros(vocab_size, dtype=np.float32)
    total_steps = len(ids) - 1
    progress = ProgressBar("DendriticError-1810-lite train", total_steps, show_progress)
    if curve_probe is not None and should_record_curve(0, total_steps, curve_every):
        loss, acc = evaluate_dendritic_probe(model, curve_probe[0], curve_probe[1], curve_temperature)
        append_curve_row(curve_rows, curve_method, 0, loss, acc)

    for idx in range(total_steps):
        current = int(ids[idx])
        target = int(ids[idx + 1])
        trace *= cfg.trace_decay
        trace[current] += 1.0

        basal, hidden = model.hidden(trace)
        logits = model.w_out @ hidden + model.b_out
        probs = softmax(logits)
        output_error = probs
        output_error[target] -= 1.0

        feedback = model.feedback @ output_error
        nudged_hidden = np.tanh(basal - cfg.lambda_apical * feedback)
        hidden_error_estimate = np.clip(nudged_hidden - hidden, -cfg.grad_clip, cfg.grad_clip)

        out_update = np.outer(output_error, hidden)
        in_update = np.outer(hidden_error_estimate, trace)
        np.clip(out_update, -cfg.grad_clip, cfg.grad_clip, out=out_update)
        np.clip(in_update, -cfg.grad_clip, cfg.grad_clip, out=in_update)

        model.w_out -= cfg.lr_out * out_update
        model.b_out -= cfg.lr_out * output_error
        model.w_in += cfg.lr_in * in_update
        model.b_h += cfg.lr_in * hidden_error_estimate

        if cfg.feedback_mode == "symmetric":
            model.feedback = model.w_out.T.copy()
        progress.update(idx + 1)
        if curve_probe is not None and should_record_curve(idx + 1, total_steps, curve_every):
            loss, acc = evaluate_dendritic_probe(model, curve_probe[0], curve_probe[1], curve_temperature)
            append_curve_row(curve_rows, curve_method, idx + 1, loss, acc)

    progress.close()
    return model


def evaluate_dendritic_error(
    model: DendriticErrorNetwork,
    ids: np.ndarray,
    temperature: float,
    max_tokens: int | None = None,
    label: str = "DendriticError-1810-lite eval",
    show_progress: bool = True,
) -> dict:
    trace = np.zeros(model.vocab_size, dtype=np.float32)
    losses: list[float] = []
    correct = 0
    total = 0
    limit = len(ids) - 1 if max_tokens is None else min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)

    for idx in range(limit):
        current = int(ids[idx])
        target = int(ids[idx + 1])
        trace *= model.cfg.trace_decay
        trace[current] += 1.0
        loss, pred = cross_entropy_from_scores(model.scores_from_trace(trace), target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1
        progress.update(idx + 1)

    progress.close()
    loss = float(np.mean(losses))
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / max(total, 1),
    }


class RecurrentThreeFactorNetwork:
    """
    Rate recurrent model trained by a local three-factor eligibility rule.

    The hidden recurrent/input weights receive no BPTT update.  At each token,
    the output error is projected through a fixed random feedback matrix to form
    a per-hidden-unit modulatory signal.  That signal gates eligibility traces
    that were built from local pre/post activity.
    """

    def __init__(self, vocab_size: int, hidden_dim: int, cfg: RecurrentThreeFactorConfig, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.cfg = cfg
        self.w_in = rng.normal(0.0, cfg.input_scale / math.sqrt(vocab_size), (hidden_dim, vocab_size)).astype(np.float32)
        self.w_rec = rng.normal(0.0, cfg.rec_gain / math.sqrt(hidden_dim), (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)
        self.w_out = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), (vocab_size, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        if cfg.feedback_mode == "symmetric":
            self.feedback = self.w_out.T.copy()
        else:
            self.feedback = (
                rng.normal(0.0, cfg.feedback_scale / math.sqrt(vocab_size), (hidden_dim, vocab_size))
                .astype(np.float32)
            )

    def step(self, token: int, hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        drive = self.w_in[:, int(token)] + self.w_rec @ hidden + self.b_h
        next_hidden = np.tanh(drive).astype(np.float32)
        return drive.astype(np.float32), next_hidden

    def scores_from_hidden(self, hidden: np.ndarray) -> np.ndarray:
        return self.w_out @ hidden + self.b_out


def evaluate_recurrent_probe(
    model: RecurrentThreeFactorNetwork,
    ids: np.ndarray,
    max_tokens: int,
    temperature: float,
) -> tuple[float, float]:
    hidden = np.zeros(model.hidden_dim, dtype=np.float32)
    losses: list[float] = []
    correct = 0
    total = 0
    limit = min(len(ids) - 1, max_tokens)
    for idx in range(limit):
        _, hidden = model.step(int(ids[idx]), hidden)
        loss, pred = cross_entropy_from_scores(model.scores_from_hidden(hidden), int(ids[idx + 1]), temperature)
        losses.append(loss)
        correct += int(pred == int(ids[idx + 1]))
        total += 1
    return float(np.mean(losses)), correct / max(total, 1)


def train_recurrent_three_factor(
    ids: np.ndarray,
    vocab_size: int,
    hidden_dim: int,
    cfg: RecurrentThreeFactorConfig,
    seed: int,
    show_progress: bool = True,
    curve_rows: list[dict] | None = None,
    curve_probe_ids: np.ndarray | None = None,
    curve_probe_tokens: int = 0,
    curve_every: int = 0,
    curve_temperature: float = 1.0,
    curve_method: str = "recurrent_3factor",
) -> RecurrentThreeFactorNetwork:
    model = RecurrentThreeFactorNetwork(vocab_size, hidden_dim, cfg, seed)
    hidden = np.zeros(hidden_dim, dtype=np.float32)
    elig_in = np.zeros((hidden_dim, vocab_size), dtype=np.float32)
    elig_rec = np.zeros((hidden_dim, hidden_dim), dtype=np.float32)
    total_steps = len(ids) - 1
    progress = ProgressBar("RecurrentThreeFactor train", total_steps, show_progress)

    if curve_probe_ids is not None and should_record_curve(0, total_steps, curve_every):
        loss, acc = evaluate_recurrent_probe(model, curve_probe_ids, curve_probe_tokens, curve_temperature)
        append_curve_row(curve_rows, curve_method, 0, loss, acc)

    for idx in range(total_steps):
        current = int(ids[idx])
        target = int(ids[idx + 1])
        prev_hidden = hidden
        drive, hidden = model.step(current, prev_hidden)
        local_deriv = (1.0 - np.square(hidden)).astype(np.float32)
        logits = model.scores_from_hidden(hidden)
        probs = softmax(logits)
        output_error = probs
        output_error[target] -= 1.0

        modulatory = np.clip(model.feedback @ output_error, -cfg.grad_clip, cfg.grad_clip).astype(np.float32)
        elig_in *= cfg.eligibility_decay
        elig_rec *= cfg.eligibility_decay
        elig_in[:, current] += local_deriv
        elig_rec += local_deriv[:, None] * prev_hidden[None, :]

        hidden_factor = np.clip(modulatory[:, None] * elig_in, -cfg.grad_clip, cfg.grad_clip)
        rec_factor = np.clip(modulatory[:, None] * elig_rec, -cfg.grad_clip, cfg.grad_clip)
        out_update = np.outer(output_error, hidden)
        np.clip(out_update, -cfg.grad_clip, cfg.grad_clip, out=out_update)

        model.w_in -= cfg.lr_hidden * hidden_factor
        model.w_rec -= cfg.lr_hidden * rec_factor
        model.b_h -= cfg.lr_hidden * modulatory * local_deriv
        model.w_out -= cfg.lr_out * out_update
        model.b_out -= cfg.lr_out * output_error
        np.clip(model.w_in, -cfg.weight_clip, cfg.weight_clip, out=model.w_in)
        np.clip(model.w_rec, -cfg.weight_clip, cfg.weight_clip, out=model.w_rec)
        np.clip(model.w_out, -cfg.weight_clip, cfg.weight_clip, out=model.w_out)
        if cfg.feedback_mode == "symmetric":
            model.feedback = model.w_out.T.copy()

        progress.update(idx + 1)
        if curve_probe_ids is not None and should_record_curve(idx + 1, total_steps, curve_every):
            loss, acc = evaluate_recurrent_probe(model, curve_probe_ids, curve_probe_tokens, curve_temperature)
            append_curve_row(curve_rows, curve_method, idx + 1, loss, acc)

    progress.close()
    return model


def evaluate_recurrent_three_factor(
    model: RecurrentThreeFactorNetwork,
    ids: np.ndarray,
    temperature: float,
    max_tokens: int | None = None,
    label: str = "RecurrentThreeFactor eval",
    show_progress: bool = True,
) -> dict:
    hidden = np.zeros(model.hidden_dim, dtype=np.float32)
    losses: list[float] = []
    correct = 0
    total = 0
    limit = len(ids) - 1 if max_tokens is None else min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)

    for idx in range(limit):
        current = int(ids[idx])
        target = int(ids[idx + 1])
        _, hidden = model.step(current, hidden)
        loss, pred = cross_entropy_from_scores(model.scores_from_hidden(hidden), target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1
        progress.update(idx + 1)

    progress.close()
    loss = float(np.mean(losses))
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / max(total, 1),
    }


class SparseHebbianContextMemory:
    """
    Sparse associative next-token memory.

    Each observed context tuple keeps a decayed score vector over next tokens.
    This is a local Hebbian update on active context features and the observed
    next token; it stores compressed statistics rather than raw examples.
    """

    def __init__(self, vocab_size: int, cfg: SparseHebbianContextConfig) -> None:
        self.vocab_size = vocab_size
        self.cfg = cfg
        self.tables: list[dict[tuple[int, ...], dict[int, float]]] = [dict() for _ in range(cfg.max_order + 1)]
        self.unigram = np.zeros(vocab_size, dtype=np.float32)

    def update(self, context: tuple[int, ...], target: int) -> None:
        target = int(target)
        self.unigram *= 1.0 - self.cfg.alpha * self.cfg.unigram_weight
        self.unigram[target] += self.cfg.alpha
        for order in range(1, self.cfg.max_order + 1):
            if len(context) < order:
                continue
            key = tuple(int(token) for token in context[-order:])
            row = self.tables[order].setdefault(key, {})
            row[target] = row.get(target, 0.0) + self.cfg.alpha * (self.cfg.order_weight_growth ** (order - 1))

    def scores(self, context: tuple[int, ...]) -> np.ndarray:
        if self.cfg.score_mode == "normalized":
            return self.normalized_scores(context)
        scores = self.cfg.unigram_weight * self.unigram.astype(np.float32, copy=True)
        for order in range(1, self.cfg.max_order + 1):
            if len(context) < order:
                continue
            row = self.tables[order].get(tuple(int(token) for token in context[-order:]))
            if not row:
                continue
            scale = self.cfg.order_weight * (self.cfg.order_weight_growth ** (order - 1))
            for token, value in row.items():
                scores[int(token)] += scale * float(value)
        return scores

    def confidence(self, context: tuple[int, ...]) -> dict:
        for order in range(min(len(context), self.cfg.max_order), 0, -1):
            row = self.tables[order].get(tuple(int(token) for token in context[-order:]))
            if not row:
                continue
            values = np.fromiter((float(value) for value in row.values()), dtype=np.float32)
            row_total = float(np.sum(values))
            if row_total <= 0.0:
                continue
            probs = values / row_total
            entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
            return {
                "memory_order": order,
                "row_entries": len(row),
                "row_total": row_total,
                "max_prob": float(np.max(probs)),
                "entropy": entropy,
            }
        return {
            "memory_order": 0,
            "row_entries": 0,
            "row_total": 0.0,
            "max_prob": 0.0,
            "entropy": float("inf"),
        }

    def normalized_scores(self, context: tuple[int, ...]) -> np.ndarray:
        probs = self.unigram.astype(np.float32, copy=True) + self.cfg.smoothing
        probs /= float(np.sum(probs) + 1e-9)
        for order in range(1, self.cfg.max_order + 1):
            if len(context) < order:
                continue
            row = self.tables[order].get(tuple(int(token) for token in context[-order:]))
            if not row:
                continue
            row_total = float(sum(row.values()) + self.cfg.smoothing * self.vocab_size)
            row_probs = np.full(self.vocab_size, self.cfg.smoothing / row_total, dtype=np.float32)
            for token, value in row.items():
                row_probs[int(token)] = (float(value) + self.cfg.smoothing) / row_total
            mix = 1.0 - self.cfg.backoff
            probs = mix * row_probs + self.cfg.backoff * probs
        return np.log(probs + 1e-9).astype(np.float32)

    def active_contexts(self) -> int:
        return sum(len(table) for table in self.tables)


def memory_is_confident(info: dict, cfg: GatedBackoffConfig) -> bool:
    return (
        int(info["memory_order"]) >= cfg.min_order
        and float(info["row_total"]) >= cfg.min_row_total
        and float(info["max_prob"]) >= cfg.min_max_prob
        and float(info["entropy"]) <= cfg.max_entropy
    )


def summarize_eval(losses: list[float], correct: int, total: int) -> dict:
    loss = float(np.mean(losses)) if losses else float("nan")
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))) if math.isfinite(loss) else float("nan"),
        "accuracy": correct / max(total, 1),
    }


def train_sparse_hebbian_context(
    ids: np.ndarray,
    vocab_size: int,
    cfg: SparseHebbianContextConfig,
    show_progress: bool = True,
    curve_rows: list[dict] | None = None,
    curve_probe_ids: np.ndarray | None = None,
    curve_probe_tokens: int = 0,
    curve_every: int = 0,
    curve_temperature: float = 1.0,
    curve_method: str = "sparse_hebbian_context",
) -> SparseHebbianContextMemory:
    memory = SparseHebbianContextMemory(vocab_size, cfg)
    total_steps = len(ids) - 1
    progress = ProgressBar("SparseHebbianContext train", total_steps, show_progress)
    if curve_probe_ids is not None and should_record_curve(0, total_steps, curve_every):
        loss, acc = evaluate_sparse_hebbian_context_probe(memory, curve_probe_ids, curve_probe_tokens, curve_temperature)
        append_curve_row(curve_rows, curve_method, 0, loss, acc)

    for idx in range(total_steps):
        start = max(0, idx + 1 - cfg.max_order)
        context = tuple(int(token) for token in ids[start : idx + 1])
        memory.update(context, int(ids[idx + 1]))
        progress.update(idx + 1)
        if curve_probe_ids is not None and should_record_curve(idx + 1, total_steps, curve_every):
            loss, acc = evaluate_sparse_hebbian_context_probe(
                memory,
                curve_probe_ids,
                curve_probe_tokens,
                curve_temperature,
            )
            append_curve_row(curve_rows, curve_method, idx + 1, loss, acc)

    progress.close()
    return memory


def evaluate_sparse_hebbian_context_probe(
    memory: SparseHebbianContextMemory,
    ids: np.ndarray,
    max_tokens: int,
    temperature: float,
) -> tuple[float, float]:
    losses: list[float] = []
    correct = 0
    total = 0
    limit = min(len(ids) - 1, max_tokens)
    for idx in range(limit):
        start = max(0, idx + 1 - memory.cfg.max_order)
        context = tuple(int(token) for token in ids[start : idx + 1])
        target = int(ids[idx + 1])
        loss, pred = cross_entropy_from_scores(memory.scores(context), target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1
    return float(np.mean(losses)), correct / max(total, 1)


def evaluate_sparse_hebbian_context(
    memory: SparseHebbianContextMemory,
    ids: np.ndarray,
    temperature: float,
    max_tokens: int | None = None,
    label: str = "SparseHebbianContext eval",
    show_progress: bool = True,
) -> dict:
    losses: list[float] = []
    correct = 0
    total = 0
    limit = len(ids) - 1 if max_tokens is None else min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)

    for idx in range(limit):
        start = max(0, idx + 1 - memory.cfg.max_order)
        context = tuple(int(token) for token in ids[start : idx + 1])
        target = int(ids[idx + 1])
        loss, pred = cross_entropy_from_scores(memory.scores(context), target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1
        progress.update(idx + 1)

    progress.close()
    loss = float(np.mean(losses))
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / max(total, 1),
        "active_contexts": memory.active_contexts(),
    }


class SemanticHebbianMemory:
    """
    Random-projection context buckets with local next-token statistics.

    Token embeddings and hash projections are fixed random features. Online
    learning only updates bucket-to-next-token counts, so this remains no-BP.
    """

    def __init__(self, vocab_size: int, cfg: SemanticHebbianConfig) -> None:
        self.vocab_size = vocab_size
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.token_embed = rng.normal(0.0, 1.0 / math.sqrt(cfg.dim), (vocab_size, cfg.dim)).astype(np.float32)
        self.hash_proj = rng.normal(0.0, 1.0 / math.sqrt(cfg.dim), (cfg.hash_bits, cfg.dim)).astype(np.float32)
        self.tables: dict[int, dict[int, float]] = {}
        self.unigram = np.zeros(vocab_size, dtype=np.float32)

    def key(self, context: tuple[int, ...]) -> int:
        if not context:
            return 0
        recent = context[-self.cfg.order :]
        vec = np.mean(self.token_embed[np.array(recent, dtype=np.int64)], axis=0)
        bits = (self.hash_proj @ vec) >= 0.0
        key = 0
        for idx, bit in enumerate(bits):
            if bool(bit):
                key |= 1 << idx
        return int(key)

    def update(self, context: tuple[int, ...], target: int) -> None:
        target = int(target)
        self.unigram *= 1.0 - self.cfg.alpha * self.cfg.unigram_weight
        self.unigram[target] += self.cfg.alpha
        key = self.key(context)
        row = self.tables.setdefault(key, {})
        row[target] = row.get(target, 0.0) + self.cfg.alpha

    def scores(self, context: tuple[int, ...]) -> np.ndarray:
        scores = self.cfg.unigram_weight * self.unigram.astype(np.float32, copy=True)
        row = self.tables.get(self.key(context))
        if row:
            for token, value in row.items():
                scores[int(token)] += self.cfg.bucket_weight * float(value)
        return scores

    def active_contexts(self) -> int:
        return len(self.tables)


@dataclass
class ContinuationBackoffConfig:
    """Kneser-Ney-style continuation counts mixed with exact context counts."""

    max_order: int = 4
    discount: float = 0.75
    unigram_weight: float = 0.10
    continuation_weight: float = 1.0
    exact_weight: float = 1.0
    exact_backoff: float = 0.4


class ContinuationBackoffMemory:
    """
    Online continuation / Kneser-Ney-style memory.

    Tracks exact context counts and continuation counts for next-token prediction
    using fixed local statistics only, no gradient updates.
    """

    def __init__(self, vocab_size: int, cfg: ContinuationBackoffConfig) -> None:
        self.vocab_size = vocab_size
        self.cfg = cfg
        self.exact_tables: list[dict[tuple[int, ...], dict[int, float]]] = [dict() for _ in range(cfg.max_order + 1)]
        self.continuation: dict[int, set[tuple[int, ...]]] = {token: set() for token in range(vocab_size)}
        self.prev_token_pairs: set[tuple[int, int]] = set()
        self.unigram = np.zeros(vocab_size, dtype=np.float32)
        self.context_totals: list[dict[tuple[int, ...], float]] = [dict() for _ in range(cfg.max_order + 1)]

    def update(self, context: tuple[int, ...], target: int) -> None:
        target = int(target)
        self.unigram[target] += 1.0
        for order in range(1, self.cfg.max_order + 1):
            if len(context) < order:
                continue
            key = tuple(int(token) for token in context[-order:])
            row = self.exact_tables[order].setdefault(key, {})
            row[target] = row.get(target, 0.0) + 1.0
            self.context_totals[order][key] = self.context_totals[order].get(key, 0.0) + 1.0
            self.continuation[target].add(key)
        if context:
            prev = int(context[-1])
            self.prev_token_pairs.add((prev, target))

    def continuation_scores(self, context: tuple[int, ...]) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        for token in range(self.vocab_size):
            scores[token] = float(len(self.continuation[token]))
        if np.sum(scores) <= 0.0:
            scores += 1.0
        return scores

    def scores(self, context: tuple[int, ...]) -> np.ndarray:
        probs = self.unigram.astype(np.float32, copy=True) + 1.0
        probs /= float(np.sum(probs) + 1e-9)
        lower = self.continuation_scores(context)
        lower /= float(np.sum(lower) + 1e-9)
        for order in range(1, self.cfg.max_order + 1):
            if len(context) < order:
                continue
            key = tuple(int(token) for token in context[-order:])
            row = self.exact_tables[order].get(key)
            if not row:
                continue
            total = float(self.context_totals[order].get(key, 0.0))
            if total <= 0.0:
                continue
            row_probs = np.zeros(self.vocab_size, dtype=np.float32)
            for token, count in row.items():
                row_probs[int(token)] = max(float(count) - self.cfg.discount, 0.0) / total
            remaining_mass = min(self.cfg.discount * len(row) / max(total, 1e-9), 1.0)
            probs = (1.0 - self.cfg.exact_backoff) * row_probs + self.cfg.exact_backoff * lower
            probs += remaining_mass * lower
        return np.log(probs + 1e-9).astype(np.float32)

    def active_contexts(self) -> int:
        return sum(len(table) for table in self.exact_tables)


def train_continuation_backoff_memory(
    ids: np.ndarray,
    vocab_size: int,
    cfg: ContinuationBackoffConfig,
    show_progress: bool = True,
) -> ContinuationBackoffMemory:
    memory = ContinuationBackoffMemory(vocab_size, cfg)
    total_steps = len(ids) - 1
    progress = ProgressBar("ContinuationBackoff train", total_steps, show_progress)
    for idx in range(total_steps):
        start = max(0, idx + 1 - cfg.max_order)
        context = tuple(int(token) for token in ids[start : idx + 1])
        memory.update(context, int(ids[idx + 1]))
        progress.update(idx + 1)
    progress.close()
    return memory


def evaluate_continuation_backoff_memory(
    memory: ContinuationBackoffMemory,
    ids: np.ndarray,
    temperature: float,
    max_tokens: int | None = None,
    label: str = "ContinuationBackoff eval",
    show_progress: bool = True,
) -> dict:
    losses: list[float] = []
    correct = 0
    total = 0
    limit = len(ids) - 1 if max_tokens is None else min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)
    for idx in range(limit):
        start = max(0, idx + 1 - memory.cfg.max_order)
        context = tuple(int(token) for token in ids[start : idx + 1])
        target = int(ids[idx + 1])
        loss, pred = cross_entropy_from_scores(memory.scores(context), target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1
        progress.update(idx + 1)
    progress.close()
    summary = summarize_eval(losses, correct, total)
    summary["active_contexts"] = memory.active_contexts()
    return summary


def train_semantic_hebbian_memory(
    ids: np.ndarray,
    vocab_size: int,
    cfg: SemanticHebbianConfig,
    show_progress: bool = True,
) -> SemanticHebbianMemory:
    memory = SemanticHebbianMemory(vocab_size, cfg)
    total_steps = len(ids) - 1
    progress = ProgressBar("SemanticHebbian train", total_steps, show_progress)
    for idx in range(total_steps):
        start = max(0, idx + 1 - cfg.order)
        context = tuple(int(token) for token in ids[start : idx + 1])
        memory.update(context, int(ids[idx + 1]))
        progress.update(idx + 1)
    progress.close()
    return memory


def evaluate_semantic_hebbian_memory(
    memory: SemanticHebbianMemory,
    ids: np.ndarray,
    temperature: float,
    max_tokens: int | None = None,
    label: str = "SemanticHebbian eval",
    show_progress: bool = True,
) -> dict:
    losses: list[float] = []
    correct = 0
    total = 0
    limit = len(ids) - 1 if max_tokens is None else min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)
    for idx in range(limit):
        start = max(0, idx + 1 - memory.cfg.order)
        context = tuple(int(token) for token in ids[start : idx + 1])
        target = int(ids[idx + 1])
        loss, pred = cross_entropy_from_scores(memory.scores(context), target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1
        progress.update(idx + 1)
    progress.close()
    summary = summarize_eval(losses, correct, total)
    summary["active_contexts"] = memory.active_contexts()
    summary["semantic_order"] = memory.cfg.order
    summary["semantic_hash_bits"] = memory.cfg.hash_bits
    return summary


def evaluate_combined_context_memory(
    exact: SparseHebbianContextMemory,
    semantic: SemanticHebbianMemory,
    ids: np.ndarray,
    semantic_weight: float,
    temperature: float,
    max_tokens: int | None = None,
    label: str = "CombinedContext eval",
    show_progress: bool = True,
) -> dict:
    losses: list[float] = []
    correct = 0
    total = 0
    low_conf_losses: list[float] = []
    low_conf_correct = 0
    low_conf_total = 0
    limit = len(ids) - 1 if max_tokens is None else min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)
    for idx in range(limit):
        exact_start = max(0, idx + 1 - exact.cfg.max_order)
        semantic_start = max(0, idx + 1 - semantic.cfg.order)
        exact_context = tuple(int(token) for token in ids[exact_start : idx + 1])
        semantic_context = tuple(int(token) for token in ids[semantic_start : idx + 1])
        target = int(ids[idx + 1])
        scores = exact.scores(exact_context) + semantic_weight * semantic.scores(semantic_context)
        loss, pred = cross_entropy_from_scores(scores, target, temperature)
        losses.append(loss)
        is_correct = int(pred == target)
        correct += is_correct
        total += 1
        if int(exact.confidence(exact_context)["memory_order"]) < exact.cfg.max_order:
            low_conf_losses.append(loss)
            low_conf_correct += is_correct
            low_conf_total += 1
        progress.update(idx + 1)
    progress.close()
    summary = summarize_eval(losses, correct, total)
    low_conf_summary = summarize_eval(low_conf_losses, low_conf_correct, low_conf_total)
    summary.update({
        "active_contexts": exact.active_contexts() + semantic.active_contexts(),
        "exact_active_contexts": exact.active_contexts(),
        "semantic_active_contexts": semantic.active_contexts(),
        "semantic_weight": semantic_weight,
        "low_conf_loss": low_conf_summary["loss"],
        "low_conf_accuracy": low_conf_summary["accuracy"],
        "low_conf_rate": low_conf_total / max(total, 1),
    })
    return summary


def evaluate_hybrid_dendritic_context(
    dendritic: DendriticErrorNetwork,
    memory: SparseHebbianContextMemory,
    ids: np.ndarray,
    cfg: HybridBackoffConfig,
    temperature: float,
    max_tokens: int | None = None,
    label: str = "HybridDendriticContext eval",
    show_progress: bool = True,
) -> dict:
    trace = np.zeros(dendritic.vocab_size, dtype=np.float32)
    losses: list[float] = []
    correct = 0
    total = 0
    limit = len(ids) - 1 if max_tokens is None else min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)

    for idx in range(limit):
        current = int(ids[idx])
        target = int(ids[idx + 1])
        trace *= dendritic.cfg.trace_decay
        trace[current] += 1.0
        start = max(0, idx + 1 - memory.cfg.max_order)
        context = tuple(int(token) for token in ids[start : idx + 1])
        scores = (
            cfg.neural_weight * dendritic.scores_from_trace(trace)
            + cfg.memory_weight * memory.scores(context)
        )
        loss, pred = cross_entropy_from_scores(scores, target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1
        progress.update(idx + 1)

    progress.close()
    loss = float(np.mean(losses))
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / max(total, 1),
        "active_contexts": memory.active_contexts(),
        "memory_weight": cfg.memory_weight,
        "neural_weight": cfg.neural_weight,
    }


def evaluate_gated_dendritic_context(
    dendritic: DendriticErrorNetwork,
    memory: SparseHebbianContextMemory,
    ids: np.ndarray,
    hybrid_cfg: HybridBackoffConfig,
    gate_cfg: GatedBackoffConfig,
    temperature: float,
    max_tokens: int | None = None,
    label: str = "GatedDendriticContext eval",
    show_progress: bool = True,
) -> dict:
    trace = np.zeros(dendritic.vocab_size, dtype=np.float32)
    losses: list[float] = []
    memory_losses: list[float] = []
    backoff_losses: list[float] = []
    correct = 0
    memory_correct = 0
    backoff_correct = 0
    total = 0
    memory_total = 0
    backoff_total = 0
    order_sum = 0.0
    row_total_sum = 0.0
    max_prob_sum = 0.0
    finite_entropy_sum = 0.0
    finite_entropy_count = 0
    limit = len(ids) - 1 if max_tokens is None else min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)

    for idx in range(limit):
        current = int(ids[idx])
        target = int(ids[idx + 1])
        trace *= dendritic.cfg.trace_decay
        trace[current] += 1.0
        start = max(0, idx + 1 - memory.cfg.max_order)
        context = tuple(int(token) for token in ids[start : idx + 1])
        memory_scores = memory.scores(context)
        info = memory.confidence(context)
        order_sum += float(info["memory_order"])
        row_total_sum += float(info["row_total"])
        max_prob_sum += float(info["max_prob"])
        if math.isfinite(float(info["entropy"])):
            finite_entropy_sum += float(info["entropy"])
            finite_entropy_count += 1

        if memory_is_confident(info, gate_cfg):
            scores = memory_scores
            bucket_losses = memory_losses
            memory_total += 1
        else:
            scores = hybrid_cfg.neural_weight * dendritic.scores_from_trace(trace) + hybrid_cfg.memory_weight * memory_scores
            bucket_losses = backoff_losses
            backoff_total += 1
        loss, pred = cross_entropy_from_scores(scores, target, temperature)
        losses.append(loss)
        bucket_losses.append(loss)
        is_correct = int(pred == target)
        correct += is_correct
        if bucket_losses is memory_losses:
            memory_correct += is_correct
        else:
            backoff_correct += is_correct
        total += 1
        progress.update(idx + 1)

    progress.close()
    summary = summarize_eval(losses, correct, total)
    memory_summary = summarize_eval(memory_losses, memory_correct, memory_total)
    backoff_summary = summarize_eval(backoff_losses, backoff_correct, backoff_total)
    summary.update({
        "active_contexts": memory.active_contexts(),
        "memory_weight": hybrid_cfg.memory_weight,
        "neural_weight": hybrid_cfg.neural_weight,
        "gate_min_order": gate_cfg.min_order,
        "gate_min_row_total": gate_cfg.min_row_total,
        "gate_min_max_prob": gate_cfg.min_max_prob,
        "gate_max_entropy": gate_cfg.max_entropy,
        "gate_memory_rate": memory_total / max(total, 1),
        "gate_backoff_rate": backoff_total / max(total, 1),
        "gate_memory_loss": memory_summary["loss"],
        "gate_memory_accuracy": memory_summary["accuracy"],
        "gate_backoff_loss": backoff_summary["loss"],
        "gate_backoff_accuracy": backoff_summary["accuracy"],
        "avg_memory_order": order_sum / max(total, 1),
        "avg_row_total": row_total_sum / max(total, 1),
        "avg_max_prob": max_prob_sum / max(total, 1),
        "avg_entropy": finite_entropy_sum / max(finite_entropy_count, 1),
    })
    return summary


def evaluate_plastic_matrix(
    weights: np.ndarray,
    ids: np.ndarray,
    trace_decay: float,
    temperature: float,
    max_tokens: int | None = None,
    label: str = "plastic eval",
    show_progress: bool = True,
) -> dict:
    trace = np.zeros(weights.shape[0], dtype=np.float32)
    losses: list[float] = []
    correct = 0
    total = 0
    limit = len(ids) - 1 if max_tokens is None else min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)

    for idx in range(limit):
        current = int(ids[idx])
        target = int(ids[idx + 1])
        trace *= trace_decay
        trace[current] += 1.0
        scores = weights @ trace
        loss, pred = cross_entropy_from_scores(scores, target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1
        progress.update(idx + 1)

    progress.close()
    loss = float(np.mean(losses))
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / max(total, 1),
    }


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_llama_config(args: argparse.Namespace, vocab_size: int) -> LlamaTorchConfig:
    max_position_embeddings = max(args.llama_max_position_embeddings, args.seq_len, args.curve_probe_tokens)
    if args.llama_preset == "llama1b":
        return llama1b_body_config(vocab_size=vocab_size, max_position_embeddings=max_position_embeddings)
    return LlamaTorchConfig(
        vocab_size=vocab_size,
        hidden_size=args.llama_hidden_dim,
        intermediate_size=args.llama_intermediate_dim,
        num_hidden_layers=args.llama_layers,
        num_attention_heads=args.llama_heads,
        num_key_value_heads=args.llama_kv_heads,
        max_position_embeddings=max_position_embeddings,
        dropout=args.llama_dropout,
    )


def make_torch_batch(
    ids: np.ndarray,
    seq_len: int,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(ids) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.stack([ids[start : start + seq_len] for start in starts]).astype(np.int64)
    y = np.stack([ids[start + 1 : start + seq_len + 1] for start in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate_llama_torch(
    model: LlamaTorchCausalLM,
    ids: np.ndarray,
    seq_len: int,
    batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    rng = np.random.default_rng(seed)
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0

    for _ in range(batches):
        x, y = make_torch_batch(ids, seq_len, batch_size, rng, device)
        logits, loss = model(x, y)
        losses.append(float(loss.item()))
        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    loss = float(np.mean(losses))
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate_llama_probe(
    model: LlamaTorchCausalLM,
    ids: np.ndarray,
    max_tokens: int,
    device: torch.device,
) -> tuple[float, float]:
    limit = min(len(ids) - 1, max_tokens, model.config.max_position_embeddings)
    x = torch.tensor(ids[:limit], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1 : limit + 1], dtype=torch.long, device=device).unsqueeze(0)
    model.eval()
    logits, loss = model(x, y)
    pred = torch.argmax(logits, dim=-1)
    acc = float((pred == y).float().mean().item())
    return float(loss.item()), acc


@torch.no_grad()
def evaluate_hybrid_llama_context(
    model: LlamaTorchCausalLM,
    memory: SparseHebbianContextMemory,
    ids: np.ndarray,
    cfg: HybridBackoffConfig,
    temperature: float,
    max_tokens: int,
    device: torch.device,
    label: str = "HybridLlamaContext eval",
    show_progress: bool = True,
) -> dict:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    limit = min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)

    for idx in range(limit):
        start = max(0, idx + 1 - model.config.max_position_embeddings)
        x = torch.tensor(ids[start : idx + 1], dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        llama_scores = logits[0, -1].detach().cpu().numpy().astype(np.float32)
        mem_start = max(0, idx + 1 - memory.cfg.max_order)
        context = tuple(int(token) for token in ids[mem_start : idx + 1])
        scores = cfg.neural_weight * llama_scores + cfg.memory_weight * memory.scores(context)
        target = int(ids[idx + 1])
        loss, pred = cross_entropy_from_scores(scores, target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1
        progress.update(idx + 1)

    progress.close()
    loss = float(np.mean(losses))
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / max(total, 1),
        "active_contexts": memory.active_contexts(),
        "memory_weight": cfg.memory_weight,
        "neural_weight": cfg.neural_weight,
    }


@torch.no_grad()
def evaluate_gated_llama_context(
    model: LlamaTorchCausalLM,
    memory: SparseHebbianContextMemory,
    ids: np.ndarray,
    hybrid_cfg: HybridBackoffConfig,
    gate_cfg: GatedBackoffConfig,
    temperature: float,
    max_tokens: int,
    device: torch.device,
    label: str = "GatedLlamaContext eval",
    show_progress: bool = True,
) -> dict:
    model.eval()
    losses: list[float] = []
    memory_losses: list[float] = []
    backoff_losses: list[float] = []
    correct = 0
    memory_correct = 0
    backoff_correct = 0
    total = 0
    memory_total = 0
    backoff_total = 0
    order_sum = 0.0
    row_total_sum = 0.0
    max_prob_sum = 0.0
    finite_entropy_sum = 0.0
    finite_entropy_count = 0
    limit = min(len(ids) - 1, max_tokens)
    progress = ProgressBar(label, limit, show_progress)

    for idx in range(limit):
        mem_start = max(0, idx + 1 - memory.cfg.max_order)
        context = tuple(int(token) for token in ids[mem_start : idx + 1])
        memory_scores = memory.scores(context)
        info = memory.confidence(context)
        order_sum += float(info["memory_order"])
        row_total_sum += float(info["row_total"])
        max_prob_sum += float(info["max_prob"])
        if math.isfinite(float(info["entropy"])):
            finite_entropy_sum += float(info["entropy"])
            finite_entropy_count += 1

        if memory_is_confident(info, gate_cfg):
            scores = memory_scores
            bucket_losses = memory_losses
            memory_total += 1
        else:
            start = max(0, idx + 1 - model.config.max_position_embeddings)
            x = torch.tensor(ids[start : idx + 1], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            llama_scores = logits[0, -1].detach().cpu().numpy().astype(np.float32)
            scores = hybrid_cfg.neural_weight * llama_scores + hybrid_cfg.memory_weight * memory_scores
            bucket_losses = backoff_losses
            backoff_total += 1
        target = int(ids[idx + 1])
        loss, pred = cross_entropy_from_scores(scores, target, temperature)
        losses.append(loss)
        bucket_losses.append(loss)
        is_correct = int(pred == target)
        correct += is_correct
        if bucket_losses is memory_losses:
            memory_correct += is_correct
        else:
            backoff_correct += is_correct
        total += 1
        progress.update(idx + 1)

    progress.close()
    summary = summarize_eval(losses, correct, total)
    memory_summary = summarize_eval(memory_losses, memory_correct, memory_total)
    backoff_summary = summarize_eval(backoff_losses, backoff_correct, backoff_total)
    summary.update({
        "active_contexts": memory.active_contexts(),
        "memory_weight": hybrid_cfg.memory_weight,
        "neural_weight": hybrid_cfg.neural_weight,
        "gate_min_order": gate_cfg.min_order,
        "gate_min_row_total": gate_cfg.min_row_total,
        "gate_min_max_prob": gate_cfg.min_max_prob,
        "gate_max_entropy": gate_cfg.max_entropy,
        "gate_memory_rate": memory_total / max(total, 1),
        "gate_backoff_rate": backoff_total / max(total, 1),
        "gate_memory_loss": memory_summary["loss"],
        "gate_memory_accuracy": memory_summary["accuracy"],
        "gate_backoff_loss": backoff_summary["loss"],
        "gate_backoff_accuracy": backoff_summary["accuracy"],
        "avg_memory_order": order_sum / max(total, 1),
        "avg_row_total": row_total_sum / max(total, 1),
        "avg_max_prob": max_prob_sum / max(total, 1),
        "avg_entropy": finite_entropy_sum / max(finite_entropy_count, 1),
    })
    return summary


def train_llama_torch(
    train_ids: np.ndarray,
    valid_ids: np.ndarray,
    args: argparse.Namespace,
    curve_rows: list[dict] | None = None,
    curve_probe_ids: np.ndarray | None = None,
    curve_probe_tokens: int = 0,
    curve_every_updates: int = 0,
) -> tuple[LlamaTorchCausalLM, list[dict], dict, torch.device, MethodTiming]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_torch_device(args.device)
    cfg = build_llama_config(args, args.max_vocab)
    model = LlamaTorchCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 17)
    history: list[dict] = []
    progress = ProgressBar("LlamaTorch train", args.llama_updates, not args.no_progress)
    print(
        "LlamaTorch config:"
        f" preset={args.llama_preset}"
        f" layers={cfg.num_hidden_layers}"
        f" hidden={cfg.hidden_size}"
        f" heads={cfg.num_attention_heads}"
        f" kv_heads={cfg.num_key_value_heads}"
        f" params={model.num_parameters:,}"
        f" device={device}"
    )

    if curve_probe_ids is not None and should_record_curve(0, args.llama_updates, curve_every_updates):
        loss, acc = evaluate_llama_probe(model, curve_probe_ids, curve_probe_tokens, device)
        append_curve_row(curve_rows, "torch_llama", 0, loss, acc)

    with Timer(device) as train_timer:
        for update in range(1, args.llama_updates + 1):
            model.train()
            x, y = make_torch_batch(train_ids, args.seq_len, args.llama_batch_size, rng, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            progress.update(update)

            processed_tokens = update * args.seq_len * args.llama_batch_size
            if curve_probe_ids is not None and should_record_curve(update, args.llama_updates, curve_every_updates):
                probe_loss, probe_acc = evaluate_llama_probe(model, curve_probe_ids, curve_probe_tokens, device)
                append_curve_row(curve_rows, "torch_llama", processed_tokens, probe_loss, probe_acc)

            if update == 1 or update % args.eval_every == 0 or update == args.llama_updates:
                valid_metrics = evaluate_llama_torch(
                    model,
                    valid_ids,
                    args.seq_len,
                    args.eval_batches,
                    args.llama_batch_size,
                    args.seed + update,
                    device,
                )
                row = {
                    "update": update,
                    "processed_tokens": processed_tokens,
                    "train_loss": float(loss.item()),
                    "valid_loss": valid_metrics["loss"],
                    "valid_ppl": valid_metrics["ppl"],
                    "valid_acc": valid_metrics["accuracy"],
                }
                history.append(row)
                print(
                    "LlamaTorch"
                    f" update={update:4d}"
                    f" tokens={processed_tokens:,}"
                    f" train_ce={row['train_loss']:.3f}"
                    f" valid_ce={row['valid_loss']:.3f}"
                    f" ppl={row['valid_ppl']:.2f}"
                    f" acc={row['valid_acc']:.3f}"
                )

    progress.close()
    with Timer(device) as eval_timer:
        final_metrics = evaluate_llama_torch(
            model,
            valid_ids,
            args.seq_len,
            max(args.eval_batches * 2, 16),
            args.llama_batch_size,
            args.seed + 1000,
            device,
        )
    timing = MethodTiming(
        train_seconds=train_timer.elapsed,
        eval_seconds=eval_timer.elapsed,
        train_tokens=args.llama_updates * args.seq_len * args.llama_batch_size,
        eval_tokens=max(args.eval_batches * 2, 16) * args.seq_len * args.llama_batch_size,
        device=str(device),
    )
    return model, history, final_metrics, device, timing


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_supervised_history(history: list[dict], path: Path) -> None:
    updates = [int(row["update"]) for row in history]
    valid_loss = [float(row["valid_loss"]) for row in history]
    valid_acc = [float(row["valid_acc"]) for row in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(updates, valid_loss, marker="o", color="firebrick")
    ax1.set_xlabel("LlamaTorch update")
    ax1.set_ylabel("token CE")
    ax2 = ax1.twinx()
    ax2.plot(updates, valid_acc, marker="s", color="royalblue")
    ax2.set_ylabel("top-1 token accuracy")
    ax1.set_title("TinyStories Llama-token LlamaTorch")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_online_loss_curves(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        steps = [int(row["step"]) for row in method_rows]
        losses = [float(row["loss"]) for row in method_rows]
        ax.plot(steps, losses, marker="o", linewidth=1.6, markersize=3.5, label=method)
    ax.set_xlabel("processed next-token training positions")
    ax.set_ylabel("probe token CE")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_token_maps(kept_raw: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    raw_to_compact = {int(raw): idx for idx, raw in enumerate(kept_raw)}
    compact_to_raw = {idx: int(raw) for idx, raw in enumerate(kept_raw)}
    return raw_to_compact, compact_to_raw


def compact_prompt_ids(tokenizer, kept_raw: np.ndarray, prompt: str) -> list[int]:
    raw_to_compact, _ = make_token_maps(kept_raw)
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = [raw_to_compact[token] for token in raw_ids if token in raw_to_compact]
    return ids if ids else [0]


def prompt_token_coverage(tokenizer, kept_raw: np.ndarray, prompt: str) -> tuple[int, int]:
    kept = {int(token) for token in kept_raw}
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    kept_count = sum(1 for token in raw_ids if int(token) in kept)
    return kept_count, len(raw_ids)


def read_prompt_file(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        prompt = line.strip()
        if prompt and not prompt.startswith("#"):
            prompts.append(prompt)
    return prompts


def build_heldout_prompts(
    train_text: str,
    valid_text: str,
    tokenizer,
    kept_raw: np.ndarray,
    count: int = 3,
) -> list[str]:
    train_norm = normalize_text(train_text)
    prompts: list[str] = []
    seen: set[str] = set()
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(valid_text.split()))

    for sentence in sentences:
        words = sentence.split()
        if len(words) < 6:
            continue
        for n_words in (8, 7, 9, 6, 10):
            if len(words) < n_words:
                continue
            prompt = " ".join(words[:n_words]).strip(" ,;:")
            prompt_norm = normalize_text(prompt)
            kept_count, total_count = prompt_token_coverage(tokenizer, kept_raw, prompt)
            enough_tokens = total_count > 0 and kept_count >= max(3, total_count // 2)
            if prompt_norm and prompt_norm not in train_norm and prompt_norm not in seen and enough_tokens:
                prompts.append(prompt)
                seen.add(prompt_norm)
                break
        if len(prompts) >= count:
            return prompts

    for prompt in FALLBACK_SAMPLE_PROMPTS:
        prompt_norm = normalize_text(prompt)
        if prompt_norm not in train_norm and prompt_norm not in seen:
            prompts.append(prompt)
            seen.add(prompt_norm)
        if len(prompts) >= count:
            break
    return prompts


def prompt_status(tokenizer, kept_raw: np.ndarray, prompt: str, train_text: str) -> dict:
    kept_count, total_count = prompt_token_coverage(tokenizer, kept_raw, prompt)
    return {
        "prompt": prompt,
        "in_train_prefix": normalize_text(prompt) in normalize_text(train_text),
        "compact_tokens": kept_count,
        "raw_tokens": total_count,
    }


def decode_compact_ids(tokenizer, kept_raw: np.ndarray, ids: list[int]) -> str:
    _, compact_to_raw = make_token_maps(kept_raw)
    raw_generated = [compact_to_raw[int(idx)] for idx in ids]
    return tokenizer.decode(raw_generated, skip_special_tokens=True)


def apply_decoding_controls(scores: np.ndarray, generated: list[int], cfg: DecodingConfig) -> np.ndarray:
    adjusted = scores.astype(np.float32, copy=True)
    if cfg.repetition_penalty > 0.0:
        window = generated[-32:]
        counts = Counter(int(token) for token in window)
        for token, count in counts.items():
            adjusted[token] -= cfg.repetition_penalty * float(count)
    if cfg.no_repeat_ngram > 1 and len(generated) >= cfg.no_repeat_ngram - 1:
        prefix = tuple(int(token) for token in generated[-(cfg.no_repeat_ngram - 1) :])
        banned: set[int] = set()
        for idx in range(len(generated) - cfg.no_repeat_ngram + 1):
            ngram = tuple(int(token) for token in generated[idx : idx + cfg.no_repeat_ngram])
            if ngram[:-1] == prefix:
                banned.add(ngram[-1])
        for token in banned:
            adjusted[int(token)] = -1e9
    return adjusted


def choose_next_token(scores: np.ndarray, generated: list[int], cfg: DecodingConfig, rng: np.random.Generator) -> int:
    adjusted = apply_decoding_controls(scores, generated, cfg)
    if cfg.top_k <= 1:
        return int(np.argmax(adjusted))
    k = min(cfg.top_k, adjusted.shape[0])
    top_idx = np.argpartition(adjusted, -k)[-k:]
    top_scores = adjusted[top_idx]
    probs = softmax(top_scores, cfg.temperature)
    return int(top_idx[int(rng.choice(len(top_idx), p=probs))])


def generation_quality_metrics(name: str, prompt: str, text: str) -> dict:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    continuation = text[len(prompt) :] if text.startswith(prompt) else text
    continuation_tokens = re.findall(r"\w+|[^\w\s]", continuation.lower())

    def distinct_n(n: int) -> float:
        if len(continuation_tokens) < n:
            return 0.0
        ngrams = [tuple(continuation_tokens[i : i + n]) for i in range(len(continuation_tokens) - n + 1)]
        return len(set(ngrams)) / max(len(ngrams), 1)

    def repeat_ngram_rate(n: int) -> float:
        if len(continuation_tokens) < n:
            return 0.0
        ngrams = [tuple(continuation_tokens[i : i + n]) for i in range(len(continuation_tokens) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return repeated / max(len(ngrams), 1)

    longest_run = 0
    current_run = 0
    prev = None
    for token in continuation_tokens:
        current_run = current_run + 1 if token == prev else 1
        longest_run = max(longest_run, current_run)
        prev = token

    return {
        "method": name,
        "prompt": prompt,
        "text_chars": len(text),
        "token_count": len(tokens),
        "continuation_token_count": len(continuation_tokens),
        "distinct_1": distinct_n(1),
        "distinct_2": distinct_n(2),
        "repeat_2_rate": repeat_ngram_rate(2),
        "repeat_3_rate": repeat_ngram_rate(3),
        "longest_token_run": longest_run,
    }


@torch.no_grad()
def sample_llama_torch(
    model: LlamaTorchCausalLM,
    tokenizer,
    kept_raw: np.ndarray,
    prompt: str,
    length: int,
    temperature: float,
    seed: int,
    device: torch.device,
    greedy: bool = False,
) -> str:
    ids = compact_prompt_ids(tokenizer, kept_raw, prompt)
    generated = list(ids)
    torch.manual_seed(seed)
    model.eval()

    for _ in range(length):
        context = generated[-model.config.max_position_embeddings :]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        next_logits = logits[0, -1] / max(temperature, 1e-6)
        if greedy:
            current = int(torch.argmax(next_logits).item())
        else:
            probs = torch.softmax(next_logits, dim=-1)
            current = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(current)

    return decode_compact_ids(tokenizer, kept_raw, generated)


def sample_plastic(
    weights: np.ndarray,
    tokenizer,
    kept_raw: np.ndarray,
    prompt: str,
    length: int,
    trace_decay: float,
) -> str:
    ids = compact_prompt_ids(tokenizer, kept_raw, prompt)
    generated = list(ids)
    trace = np.zeros(weights.shape[0], dtype=np.float32)
    for token in ids:
        trace *= trace_decay
        trace[int(token)] += 1.0

    for _ in range(length):
        current = int(np.argmax(weights @ trace))
        generated.append(current)
        trace *= trace_decay
        trace[current] += 1.0

    return decode_compact_ids(tokenizer, kept_raw, generated)


def sample_dendritic_error(
    model: DendriticErrorNetwork,
    tokenizer,
    kept_raw: np.ndarray,
    prompt: str,
    length: int,
) -> str:
    ids = compact_prompt_ids(tokenizer, kept_raw, prompt)
    generated = list(ids)
    trace = np.zeros(model.vocab_size, dtype=np.float32)
    for token in ids:
        trace *= model.cfg.trace_decay
        trace[int(token)] += 1.0

    for _ in range(length):
        current = int(np.argmax(model.scores_from_trace(trace)))
        generated.append(current)
        trace *= model.cfg.trace_decay
        trace[current] += 1.0

    return decode_compact_ids(tokenizer, kept_raw, generated)


def sample_recurrent_three_factor(
    model: RecurrentThreeFactorNetwork,
    tokenizer,
    kept_raw: np.ndarray,
    prompt: str,
    length: int,
) -> str:
    ids = compact_prompt_ids(tokenizer, kept_raw, prompt)
    generated = list(ids)
    hidden = np.zeros(model.hidden_dim, dtype=np.float32)
    for token in ids:
        _, hidden = model.step(int(token), hidden)

    for _ in range(length):
        current = int(np.argmax(model.scores_from_hidden(hidden)))
        generated.append(current)
        _, hidden = model.step(current, hidden)

    return decode_compact_ids(tokenizer, kept_raw, generated)


def sample_sparse_hebbian_context(
    memory: SparseHebbianContextMemory,
    tokenizer,
    kept_raw: np.ndarray,
    prompt: str,
    length: int,
    decode_cfg: DecodingConfig | None = None,
    seed: int = 0,
) -> str:
    ids = compact_prompt_ids(tokenizer, kept_raw, prompt)
    generated = list(ids)
    rng = np.random.default_rng(seed)

    for _ in range(length):
        context = tuple(generated[-memory.cfg.max_order :])
        scores = memory.scores(context)
        current = choose_next_token(scores, generated, decode_cfg, rng) if decode_cfg is not None else int(np.argmax(scores))
        generated.append(current)

    return decode_compact_ids(tokenizer, kept_raw, generated)


def sample_hybrid_dendritic_context(
    dendritic: DendriticErrorNetwork,
    memory: SparseHebbianContextMemory,
    cfg: HybridBackoffConfig,
    tokenizer,
    kept_raw: np.ndarray,
    prompt: str,
    length: int,
    decode_cfg: DecodingConfig | None = None,
    seed: int = 0,
) -> str:
    ids = compact_prompt_ids(tokenizer, kept_raw, prompt)
    generated = list(ids)
    rng = np.random.default_rng(seed)
    trace = np.zeros(dendritic.vocab_size, dtype=np.float32)
    for token in ids:
        trace *= dendritic.cfg.trace_decay
        trace[int(token)] += 1.0

    for _ in range(length):
        context = tuple(generated[-memory.cfg.max_order :])
        scores = cfg.neural_weight * dendritic.scores_from_trace(trace) + cfg.memory_weight * memory.scores(context)
        current = choose_next_token(scores, generated, decode_cfg, rng) if decode_cfg is not None else int(np.argmax(scores))
        generated.append(current)
        trace *= dendritic.cfg.trace_decay
        trace[current] += 1.0

    return decode_compact_ids(tokenizer, kept_raw, generated)


def sample_gated_dendritic_context(
    dendritic: DendriticErrorNetwork,
    memory: SparseHebbianContextMemory,
    hybrid_cfg: HybridBackoffConfig,
    gate_cfg: GatedBackoffConfig,
    tokenizer,
    kept_raw: np.ndarray,
    prompt: str,
    length: int,
    decode_cfg: DecodingConfig | None = None,
    seed: int = 0,
) -> str:
    ids = compact_prompt_ids(tokenizer, kept_raw, prompt)
    generated = list(ids)
    rng = np.random.default_rng(seed)
    trace = np.zeros(dendritic.vocab_size, dtype=np.float32)
    for token in ids:
        trace *= dendritic.cfg.trace_decay
        trace[int(token)] += 1.0

    for _ in range(length):
        context = tuple(generated[-memory.cfg.max_order :])
        memory_scores = memory.scores(context)
        if memory_is_confident(memory.confidence(context), gate_cfg):
            scores = memory_scores
        else:
            scores = hybrid_cfg.neural_weight * dendritic.scores_from_trace(trace) + hybrid_cfg.memory_weight * memory_scores
        current = choose_next_token(scores, generated, decode_cfg, rng) if decode_cfg is not None else int(np.argmax(scores))
        generated.append(current)
        trace *= dendritic.cfg.trace_decay
        trace[current] += 1.0

    return decode_compact_ids(tokenizer, kept_raw, generated)


@torch.no_grad()
def sample_hybrid_llama_context(
    model: LlamaTorchCausalLM,
    memory: SparseHebbianContextMemory,
    cfg: HybridBackoffConfig,
    tokenizer,
    kept_raw: np.ndarray,
    prompt: str,
    length: int,
    device: torch.device,
    decode_cfg: DecodingConfig | None = None,
    seed: int = 0,
) -> str:
    ids = compact_prompt_ids(tokenizer, kept_raw, prompt)
    generated = list(ids)
    rng = np.random.default_rng(seed)
    model.eval()

    for _ in range(length):
        context_ids = generated[-model.config.max_position_embeddings :]
        x = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        llama_scores = logits[0, -1].detach().cpu().numpy().astype(np.float32)
        mem_context = tuple(generated[-memory.cfg.max_order :])
        scores = cfg.neural_weight * llama_scores + cfg.memory_weight * memory.scores(mem_context)
        current = choose_next_token(scores, generated, decode_cfg, rng) if decode_cfg is not None else int(np.argmax(scores))
        generated.append(current)

    return decode_compact_ids(tokenizer, kept_raw, generated)


@torch.no_grad()
def sample_gated_llama_context(
    model: LlamaTorchCausalLM,
    memory: SparseHebbianContextMemory,
    hybrid_cfg: HybridBackoffConfig,
    gate_cfg: GatedBackoffConfig,
    tokenizer,
    kept_raw: np.ndarray,
    prompt: str,
    length: int,
    device: torch.device,
    decode_cfg: DecodingConfig | None = None,
    seed: int = 0,
) -> str:
    ids = compact_prompt_ids(tokenizer, kept_raw, prompt)
    generated = list(ids)
    rng = np.random.default_rng(seed)
    model.eval()

    for _ in range(length):
        mem_context = tuple(generated[-memory.cfg.max_order :])
        memory_scores = memory.scores(mem_context)
        if memory_is_confident(memory.confidence(mem_context), gate_cfg):
            scores = memory_scores
        else:
            context_ids = generated[-model.config.max_position_embeddings :]
            x = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            llama_scores = logits[0, -1].detach().cpu().numpy().astype(np.float32)
            scores = hybrid_cfg.neural_weight * llama_scores + hybrid_cfg.memory_weight * memory_scores
        current = choose_next_token(scores, generated, decode_cfg, rng) if decode_cfg is not None else int(np.argmax(scores))
        generated.append(current)

    return decode_compact_ids(tokenizer, kept_raw, generated)


def save_checkpoint(
    path: Path,
    kept_raw: np.ndarray,
    stdp: np.ndarray,
    btsp: np.ndarray,
    stdp_bio: np.ndarray,
    btsp_bio: np.ndarray,
    dendritic_error: DendriticErrorNetwork,
    recurrent_3factor: RecurrentThreeFactorNetwork,
    sparse_context: SparseHebbianContextMemory,
    args: argparse.Namespace,
) -> None:
    sparse_keys: list[list[int]] = []
    sparse_targets: list[int] = []
    sparse_values: list[float] = []
    sparse_orders: list[int] = []
    for order, table in enumerate(sparse_context.tables):
        if order == 0:
            continue
        for key, row in table.items():
            for target, value in row.items():
                sparse_orders.append(order)
                sparse_keys.append(list(key) + [-1] * (sparse_context.cfg.max_order - len(key)))
                sparse_targets.append(int(target))
                sparse_values.append(float(value))
    np.savez_compressed(
        path,
        kept_raw_ids=kept_raw,
        stdp_weights=stdp,
        btsp_weights=btsp,
        stdp_bio_weights=stdp_bio,
        btsp_bio_weights=btsp_bio,
        dendritic_w_in=dendritic_error.w_in,
        dendritic_b_h=dendritic_error.b_h,
        dendritic_w_out=dendritic_error.w_out,
        dendritic_b_out=dendritic_error.b_out,
        dendritic_feedback=dendritic_error.feedback,
        recurrent_w_in=recurrent_3factor.w_in,
        recurrent_w_rec=recurrent_3factor.w_rec,
        recurrent_b_h=recurrent_3factor.b_h,
        recurrent_w_out=recurrent_3factor.w_out,
        recurrent_b_out=recurrent_3factor.b_out,
        recurrent_feedback=recurrent_3factor.feedback,
        sparse_context_orders=np.array(sparse_orders, dtype=np.int16),
        sparse_context_keys=np.array(sparse_keys, dtype=np.int64).reshape(-1, sparse_context.cfg.max_order),
        sparse_context_targets=np.array(sparse_targets, dtype=np.int64),
        sparse_context_values=np.array(sparse_values, dtype=np.float32),
        sparse_context_unigram=sparse_context.unigram,
        stdp_trace_decay=np.array([STDPConfig().trace_decay], dtype=np.float32),
        btsp_trace_decay=np.array([BTSPConfig().trace_decay], dtype=np.float32),
        dendritic_trace_decay=np.array([DendriticErrorConfig().trace_decay], dtype=np.float32),
        dendritic_lambda_apical=np.array([DendriticErrorConfig().lambda_apical], dtype=np.float32),
        recurrent_eligibility_decay=np.array([recurrent_3factor.cfg.eligibility_decay], dtype=np.float32),
        sparse_context_max_order=np.array([sparse_context.cfg.max_order], dtype=np.int16),
        plastic_temperature=np.array([args.plastic_temperature], dtype=np.float32),
    )


def save_llama_torch_checkpoint(
    path: Path,
    model: LlamaTorchCausalLM,
    kept_raw: np.ndarray,
) -> None:
    torch.save(
        {
            "config": asdict(model.config),
            "kept_raw_ids": kept_raw,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        path,
    )


def write_config(path: Path, args: argparse.Namespace) -> None:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["stdp"] = asdict(STDPConfig())
    config["btsp"] = asdict(BTSPConfig())
    config["bio_structure"] = asdict(BioStructureConfig())
    config["dendritic_error_1810"] = asdict(DendriticErrorConfig())
    config["recurrent_3factor"] = {
        "eligibility_decay": args.recurrent_eligibility_decay,
        "lr_hidden": args.recurrent_lr_hidden,
        "lr_out": args.recurrent_lr_out,
        "rec_gain": args.recurrent_rec_gain,
        "input_scale": args.recurrent_input_scale,
        "feedback_scale": args.recurrent_feedback_scale,
        "grad_clip": args.recurrent_grad_clip,
        "weight_clip": args.recurrent_weight_clip,
        "feedback_mode": args.recurrent_feedback_mode,
    }
    config["sparse_hebbian_context"] = {
        "max_order": args.context_max_order,
        "alpha": args.context_alpha,
        "unigram_weight": args.context_unigram_weight,
        "order_weight": args.context_order_weight,
        "order_weight_growth": args.context_order_weight_growth,
        "score_mode": args.context_score_mode,
        "smoothing": args.context_smoothing,
        "backoff": args.context_backoff,
    }
    config["semantic_hebbian"] = {
        "order": args.semantic_order,
        "dim": args.semantic_dim,
        "hash_bits": args.semantic_hash_bits,
        "alpha": args.semantic_alpha,
        "bucket_weight": args.semantic_bucket_weight,
        "unigram_weight": args.semantic_unigram_weight,
        "seed": args.seed + 101,
        "combine_weight": args.semantic_combine_weight,
    }
    config["continuation_backoff"] = {
        "max_order": args.continuation_max_order,
        "discount": args.continuation_discount,
        "unigram_weight": args.continuation_unigram_weight,
        "continuation_weight": args.continuation_weight,
        "exact_weight": args.continuation_exact_weight,
        "exact_backoff": args.continuation_exact_backoff,
    }
    config["hybrid_backoff"] = {
        "memory_weight": args.hybrid_memory_weight,
        "neural_weight": args.hybrid_neural_weight,
    }
    config["gated_backoff"] = {
        "min_order": args.gate_min_order,
        "min_row_total": args.gate_min_row_total,
        "min_max_prob": args.gate_min_max_prob,
        "max_entropy": args.gate_max_entropy,
    }
    config["decoding"] = {
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram": args.no_repeat_ngram,
        "top_k": args.decode_top_k,
        "temperature": args.decode_temperature,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "tinystories_llama_token")
    parser.add_argument("--train-chars", type=int, default=1_000_000)
    parser.add_argument("--valid-chars", type=int, default=100_000)
    parser.add_argument("--max-vocab", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--llama-preset", choices=["tiny", "llama1b"], default="tiny")
    parser.add_argument("--llama-hidden-dim", type=int, default=128)
    parser.add_argument("--llama-intermediate-dim", type=int, default=512)
    parser.add_argument("--llama-layers", type=int, default=4)
    parser.add_argument("--llama-heads", type=int, default=4)
    parser.add_argument("--llama-kv-heads", type=int, default=1)
    parser.add_argument("--llama-batch-size", type=int, default=4)
    parser.add_argument("--llama-updates", type=int, default=1000)
    parser.add_argument("--llama-max-position-embeddings", type=int, default=256)
    parser.add_argument("--llama-dropout", type=float, default=0.0)
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--recurrent-eligibility-decay", type=float, default=0.92)
    parser.add_argument("--recurrent-lr-hidden", type=float, default=0.003)
    parser.add_argument("--recurrent-lr-out", type=float, default=0.020)
    parser.add_argument("--recurrent-rec-gain", type=float, default=0.55)
    parser.add_argument("--recurrent-input-scale", type=float, default=0.35)
    parser.add_argument("--recurrent-feedback-scale", type=float, default=1.0)
    parser.add_argument("--recurrent-grad-clip", type=float, default=1.5)
    parser.add_argument("--recurrent-weight-clip", type=float, default=4.0)
    parser.add_argument("--recurrent-feedback-mode", choices=["fixed", "symmetric"], default="fixed")
    parser.add_argument("--context-max-order", type=int, default=4)
    parser.add_argument("--context-alpha", type=float, default=0.05)
    parser.add_argument("--context-unigram-weight", type=float, default=0.15)
    parser.add_argument("--context-order-weight", type=float, default=1.0)
    parser.add_argument("--context-order-weight-growth", type=float, default=1.6)
    parser.add_argument("--context-score-mode", choices=["additive", "normalized"], default="additive")
    parser.add_argument("--context-smoothing", type=float, default=0.05)
    parser.add_argument("--context-backoff", type=float, default=0.35)
    parser.add_argument("--semantic-order", type=int, default=8)
    parser.add_argument("--semantic-dim", type=int, default=64)
    parser.add_argument("--semantic-hash-bits", type=int, default=12)
    parser.add_argument("--semantic-alpha", type=float, default=0.05)
    parser.add_argument("--semantic-bucket-weight", type=float, default=1.0)
    parser.add_argument("--semantic-unigram-weight", type=float, default=0.15)
    parser.add_argument("--semantic-combine-weight", type=float, default=0.5)
    parser.add_argument("--continuation-max-order", type=int, default=4)
    parser.add_argument("--continuation-discount", type=float, default=0.75)
    parser.add_argument("--continuation-unigram-weight", type=float, default=0.10)
    parser.add_argument("--continuation-weight", type=float, default=1.0)
    parser.add_argument("--continuation-exact-weight", type=float, default=1.0)
    parser.add_argument("--continuation-exact-backoff", type=float, default=0.4)
    parser.add_argument("--hybrid-memory-weight", type=float, default=1.0)
    parser.add_argument("--hybrid-neural-weight", type=float, default=1.0)
    parser.add_argument("--gate-min-order", type=int, default=4)
    parser.add_argument("--gate-min-row-total", type=float, default=0.20)
    parser.add_argument("--gate-min-max-prob", type=float, default=0.55)
    parser.add_argument("--gate-max-entropy", type=float, default=1.50)
    parser.add_argument("--repetition-penalty", type=float, default=0.45)
    parser.add_argument("--no-repeat-ngram", type=int, default=4)
    parser.add_argument("--decode-top-k", type=int, default=0)
    parser.add_argument("--decode-temperature", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--sample-len", type=int, default=120)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--plastic-temperature", type=float, default=0.8)
    parser.add_argument("--eval-token-limit", type=int, default=50_000)
    parser.add_argument("--curve-points", type=int, default=10)
    parser.add_argument("--curve-probe-tokens", type=int, default=32)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    train_text = read_prefix(args.train_file, args.train_chars)
    valid_text = read_prefix(args.valid_file, args.valid_chars)
    train_raw = encode_text(tokenizer, train_text)
    valid_raw = encode_text(tokenizer, valid_text)
    kept_raw, train_ids, valid_ids = build_compact_vocab(train_raw, valid_raw, args.max_vocab)
    args.max_vocab = int(len(kept_raw))

    if len(train_ids) <= args.seq_len + 1 or len(valid_ids) <= args.seq_len + 1:
        raise ValueError("Not enough compact-vocab tokens for this sequence length.")

    coverage_train = len(train_ids) / max(len(train_raw), 1)
    coverage_valid = len(valid_ids) / max(len(valid_raw), 1)
    print(f"tokenizer: {args.tokenizer_dir}")
    print(f"raw tokens: train={len(train_raw):,}, valid={len(valid_raw):,}")
    print(f"compact vocab: {args.max_vocab}, coverage train={coverage_train:.3f}, valid={coverage_valid:.3f}")
    print(f"compact tokens: train={len(train_ids):,}, valid={len(valid_ids):,}")

    stdp_cfg = STDPConfig()
    btsp_cfg = BTSPConfig()
    bio_cfg = BioStructureConfig()
    dendritic_cfg = DendriticErrorConfig()
    recurrent_cfg = RecurrentThreeFactorConfig(
        eligibility_decay=args.recurrent_eligibility_decay,
        lr_hidden=args.recurrent_lr_hidden,
        lr_out=args.recurrent_lr_out,
        rec_gain=args.recurrent_rec_gain,
        input_scale=args.recurrent_input_scale,
        feedback_scale=args.recurrent_feedback_scale,
        grad_clip=args.recurrent_grad_clip,
        weight_clip=args.recurrent_weight_clip,
        feedback_mode=args.recurrent_feedback_mode,
    )
    sparse_context_cfg = SparseHebbianContextConfig(
        max_order=args.context_max_order,
        alpha=args.context_alpha,
        unigram_weight=args.context_unigram_weight,
        order_weight=args.context_order_weight,
        order_weight_growth=args.context_order_weight_growth,
        score_mode=args.context_score_mode,
        smoothing=args.context_smoothing,
        backoff=args.context_backoff,
    )
    semantic_cfg = SemanticHebbianConfig(
        order=args.semantic_order,
        dim=args.semantic_dim,
        hash_bits=args.semantic_hash_bits,
        alpha=args.semantic_alpha,
        bucket_weight=args.semantic_bucket_weight,
        unigram_weight=args.semantic_unigram_weight,
        seed=args.seed + 101,
    )
    continuation_cfg = ContinuationBackoffConfig(
        max_order=args.continuation_max_order,
        discount=args.continuation_discount,
        unigram_weight=args.continuation_unigram_weight,
        continuation_weight=args.continuation_weight,
        exact_weight=args.continuation_exact_weight,
        exact_backoff=args.continuation_exact_backoff,
    )
    hybrid_cfg = HybridBackoffConfig(
        memory_weight=args.hybrid_memory_weight,
        neural_weight=args.hybrid_neural_weight,
    )
    gated_cfg = GatedBackoffConfig(
        min_order=args.gate_min_order,
        min_row_total=args.gate_min_row_total,
        min_max_prob=args.gate_min_max_prob,
        max_entropy=args.gate_max_entropy,
    )

    show_progress = not args.no_progress
    curve_rows: list[dict] = []
    curve_every = 0 if args.curve_points <= 0 else max(1, (len(train_ids) - 1) // args.curve_points)
    stdp_probe = (
        build_probe_traces(valid_ids, args.max_vocab, stdp_cfg.trace_decay, args.curve_probe_tokens)
        if curve_every > 0
        else None
    )
    btsp_probe = (
        build_probe_traces(valid_ids, args.max_vocab, btsp_cfg.trace_decay, args.curve_probe_tokens)
        if curve_every > 0
        else None
    )
    dendritic_probe = (
        build_probe_traces(valid_ids, args.max_vocab, dendritic_cfg.trace_decay, args.curve_probe_tokens)
        if curve_every > 0
        else None
    )
    eval_tokens = min(len(valid_ids) - 1, args.eval_token_limit)

    with Timer() as stdp_train_timer:
        stdp_weights = train_stdp_matrix(
            train_ids,
            args.max_vocab,
            stdp_cfg,
            show_progress,
            curve_rows,
            stdp_probe,
            curve_every,
            args.plastic_temperature,
            "stdp_trace",
        )
    with Timer() as stdp_bio_train_timer:
        stdp_bio_weights = train_stdp_bio_matrix(
            train_ids,
            args.max_vocab,
            stdp_cfg,
            bio_cfg,
            show_progress,
            curve_rows,
            stdp_probe,
            curve_every,
            args.plastic_temperature,
            "stdp_bio",
        )
    with Timer() as btsp_train_timer:
        btsp_weights = train_btsp_matrix(
            train_ids,
            args.max_vocab,
            btsp_cfg,
            show_progress,
            curve_rows,
            btsp_probe,
            curve_every,
            args.plastic_temperature,
            "btsp_trace",
        )
    with Timer() as btsp_bio_train_timer:
        btsp_bio_weights = train_btsp_bio_matrix(
            train_ids,
            args.max_vocab,
            btsp_cfg,
            bio_cfg,
            show_progress,
            curve_rows,
            btsp_probe,
            curve_every,
            args.plastic_temperature,
            "btsp_bio",
        )
    with Timer() as dendritic_train_timer:
        dendritic_model = train_dendritic_error(
            train_ids,
            args.max_vocab,
            args.hidden_dim,
            dendritic_cfg,
            args.seed + 29,
            show_progress,
            curve_rows,
            dendritic_probe,
            curve_every,
            args.plastic_temperature,
            "dendritic_error_1810_lite",
        )
    with Timer() as recurrent_train_timer:
        recurrent_model = train_recurrent_three_factor(
            train_ids,
            args.max_vocab,
            args.hidden_dim,
            recurrent_cfg,
            args.seed + 41,
            show_progress,
            curve_rows,
            valid_ids,
            args.curve_probe_tokens,
            curve_every,
            args.plastic_temperature,
            "recurrent_3factor",
        )
    with Timer() as sparse_context_train_timer:
        sparse_context_model = train_sparse_hebbian_context(
            train_ids,
            args.max_vocab,
            sparse_context_cfg,
            show_progress,
            curve_rows,
            valid_ids,
            args.curve_probe_tokens,
            curve_every,
            args.plastic_temperature,
            "sparse_hebbian_context",
        )
    with Timer() as semantic_train_timer:
        semantic_model = train_semantic_hebbian_memory(
            train_ids,
            args.max_vocab,
            semantic_cfg,
            show_progress,
        )
    with Timer() as continuation_train_timer:
        continuation_model = train_continuation_backoff_memory(
            train_ids,
            args.max_vocab,
            continuation_cfg,
            show_progress,
        )
    with Timer() as stdp_eval_timer:
        stdp_metrics = evaluate_plastic_matrix(
            stdp_weights,
            valid_ids,
            stdp_cfg.trace_decay,
            args.plastic_temperature,
            args.eval_token_limit,
            "STDP eval",
            show_progress,
        )
    with Timer() as stdp_bio_eval_timer:
        stdp_bio_metrics = evaluate_plastic_matrix(
            stdp_bio_weights,
            valid_ids,
            stdp_cfg.trace_decay,
            args.plastic_temperature,
            args.eval_token_limit,
            "STDP-Bio eval",
            show_progress,
        )
    with Timer() as btsp_eval_timer:
        btsp_metrics = evaluate_plastic_matrix(
            btsp_weights,
            valid_ids,
            btsp_cfg.trace_decay,
            args.plastic_temperature,
            args.eval_token_limit,
            "BTSP eval",
            show_progress,
        )
    with Timer() as btsp_bio_eval_timer:
        btsp_bio_metrics = evaluate_plastic_matrix(
            btsp_bio_weights,
            valid_ids,
            btsp_cfg.trace_decay,
            args.plastic_temperature,
            args.eval_token_limit,
            "BTSP-Bio eval",
            show_progress,
        )
    with Timer() as dendritic_eval_timer:
        dendritic_metrics = evaluate_dendritic_error(
            dendritic_model,
            valid_ids,
            args.plastic_temperature,
            args.eval_token_limit,
            "DendriticError-1810-lite eval",
            show_progress,
        )
    with Timer() as recurrent_eval_timer:
        recurrent_metrics = evaluate_recurrent_three_factor(
            recurrent_model,
            valid_ids,
            args.plastic_temperature,
            args.eval_token_limit,
            "RecurrentThreeFactor eval",
            show_progress,
        )
    with Timer() as sparse_context_eval_timer:
        sparse_context_metrics = evaluate_sparse_hebbian_context(
            sparse_context_model,
            valid_ids,
            args.plastic_temperature,
            args.eval_token_limit,
            "SparseHebbianContext eval",
            show_progress,
        )
    with Timer() as semantic_eval_timer:
        semantic_metrics = evaluate_semantic_hebbian_memory(
            semantic_model,
            valid_ids,
            args.plastic_temperature,
            args.eval_token_limit,
            "SemanticHebbian eval",
            show_progress,
        )
    with Timer() as combined_eval_timer:
        combined_metrics = evaluate_combined_context_memory(
            sparse_context_model,
            semantic_model,
            valid_ids,
            args.semantic_combine_weight,
            args.plastic_temperature,
            args.eval_token_limit,
            "CombinedContext eval",
            show_progress,
        )
    with Timer() as continuation_eval_timer:
        continuation_metrics = evaluate_continuation_backoff_memory(
            continuation_model,
            valid_ids,
            args.plastic_temperature,
            args.eval_token_limit,
            "ContinuationBackoff eval",
            show_progress,
        )
    with Timer() as hybrid_eval_timer:
        hybrid_metrics = evaluate_hybrid_dendritic_context(
            dendritic_model,
            sparse_context_model,
            valid_ids,
            hybrid_cfg,
            args.plastic_temperature,
            args.eval_token_limit,
            "HybridDendriticContext eval",
            show_progress,
        )
    with Timer() as gated_eval_timer:
        gated_metrics = evaluate_gated_dendritic_context(
            dendritic_model,
            sparse_context_model,
            valid_ids,
            hybrid_cfg,
            gated_cfg,
            args.plastic_temperature,
            args.eval_token_limit,
            "GatedDendriticContext eval",
            show_progress,
        )
    print(f"STDP valid_ce={stdp_metrics['loss']:.3f} ppl={stdp_metrics['ppl']:.2f} acc={stdp_metrics['accuracy']:.3f}")
    print(
        "STDP-Bio valid_ce="
        f"{stdp_bio_metrics['loss']:.3f}"
        f" ppl={stdp_bio_metrics['ppl']:.2f}"
        f" acc={stdp_bio_metrics['accuracy']:.3f}"
    )
    print(f"BTSP valid_ce={btsp_metrics['loss']:.3f} ppl={btsp_metrics['ppl']:.2f} acc={btsp_metrics['accuracy']:.3f}")
    print(
        "BTSP-Bio valid_ce="
        f"{btsp_bio_metrics['loss']:.3f}"
        f" ppl={btsp_bio_metrics['ppl']:.2f}"
        f" acc={btsp_bio_metrics['accuracy']:.3f}"
    )
    print(
        "DendriticError-1810-lite valid_ce="
        f"{dendritic_metrics['loss']:.3f}"
        f" ppl={dendritic_metrics['ppl']:.2f}"
        f" acc={dendritic_metrics['accuracy']:.3f}"
    )
    print(
        "RecurrentThreeFactor valid_ce="
        f"{recurrent_metrics['loss']:.3f}"
        f" ppl={recurrent_metrics['ppl']:.2f}"
        f" acc={recurrent_metrics['accuracy']:.3f}"
    )
    print(
        "SparseHebbianContext valid_ce="
        f"{sparse_context_metrics['loss']:.3f}"
        f" ppl={sparse_context_metrics['ppl']:.2f}"
        f" acc={sparse_context_metrics['accuracy']:.3f}"
        f" contexts={int(sparse_context_metrics['active_contexts'])}"
    )
    print(
        "SemanticHebbian valid_ce="
        f"{semantic_metrics['loss']:.3f}"
        f" ppl={semantic_metrics['ppl']:.2f}"
        f" acc={semantic_metrics['accuracy']:.3f}"
        f" buckets={int(semantic_metrics['active_contexts'])}"
    )
    print(
        "CombinedContext valid_ce="
        f"{combined_metrics['loss']:.3f}"
        f" ppl={combined_metrics['ppl']:.2f}"
        f" acc={combined_metrics['accuracy']:.3f}"
        f" low_conf_ce={combined_metrics['low_conf_loss']:.3f}"
    )
    print(
        "ContinuationBackoff valid_ce="
        f"{continuation_metrics['loss']:.3f}"
        f" ppl={continuation_metrics['ppl']:.2f}"
        f" acc={continuation_metrics['accuracy']:.3f}"
        f" contexts={int(continuation_metrics['active_contexts'])}"
    )
    print(
        "HybridDendriticContext valid_ce="
        f"{hybrid_metrics['loss']:.3f}"
        f" ppl={hybrid_metrics['ppl']:.2f}"
        f" acc={hybrid_metrics['accuracy']:.3f}"
    )
    print(
        "GatedDendriticContext valid_ce="
        f"{gated_metrics['loss']:.3f}"
        f" ppl={gated_metrics['ppl']:.2f}"
        f" acc={gated_metrics['accuracy']:.3f}"
        f" memory_rate={gated_metrics['gate_memory_rate']:.3f}"
        f" backoff_rate={gated_metrics['gate_backoff_rate']:.3f}"
    )

    history: list[dict] = []
    llama_model: LlamaTorchCausalLM | None = None
    llama_device = resolve_torch_device(args.device)
    llama_timing: MethodTiming | None = None
    llama_metrics: dict | None = None
    hybrid_llama_metrics: dict | None = None
    gated_llama_metrics: dict | None = None
    hybrid_llama_eval_seconds = 0.0
    gated_llama_eval_seconds = 0.0
    if not args.skip_llama:
        llama_curve_every = 0 if args.curve_points <= 0 else max(1, args.llama_updates // args.curve_points)
        llama_model, history, llama_metrics, llama_device, llama_timing = train_llama_torch(
            train_ids,
            valid_ids,
            args,
            curve_rows,
            valid_ids,
            args.curve_probe_tokens,
            llama_curve_every,
        )
        print(
            "LlamaTorch final valid_ce="
            f"{llama_metrics['loss']:.3f}"
            f" ppl={llama_metrics['ppl']:.2f}"
            f" acc={llama_metrics['accuracy']:.3f}"
        )
        with Timer(llama_device) as hybrid_llama_eval_timer:
            hybrid_llama_metrics = evaluate_hybrid_llama_context(
                llama_model,
                sparse_context_model,
                valid_ids,
                hybrid_cfg,
                args.plastic_temperature,
                args.eval_token_limit,
                llama_device,
                "HybridLlamaContext eval",
                show_progress,
            )
        hybrid_llama_eval_seconds = hybrid_llama_eval_timer.elapsed
        print(
            "HybridLlamaContext valid_ce="
            f"{hybrid_llama_metrics['loss']:.3f}"
            f" ppl={hybrid_llama_metrics['ppl']:.2f}"
            f" acc={hybrid_llama_metrics['accuracy']:.3f}"
        )
        with Timer(llama_device) as gated_llama_eval_timer:
            gated_llama_metrics = evaluate_gated_llama_context(
                llama_model,
                sparse_context_model,
                valid_ids,
                hybrid_cfg,
                gated_cfg,
                args.plastic_temperature,
                args.eval_token_limit,
                llama_device,
                "GatedLlamaContext eval",
                show_progress,
            )
        gated_llama_eval_seconds = gated_llama_eval_timer.elapsed
        print(
            "GatedLlamaContext valid_ce="
            f"{gated_llama_metrics['loss']:.3f}"
            f" ppl={gated_llama_metrics['ppl']:.2f}"
            f" acc={gated_llama_metrics['accuracy']:.3f}"
            f" memory_rate={gated_llama_metrics['gate_memory_rate']:.3f}"
            f" backoff_rate={gated_llama_metrics['gate_backoff_rate']:.3f}"
        )

    timings = {
        "stdp_trace": MethodTiming(
            stdp_train_timer.elapsed,
            stdp_eval_timer.elapsed,
            len(train_ids) * stdp_cfg.epochs,
            eval_tokens,
            "cpu_numpy",
        ),
        "stdp_bio": MethodTiming(
            stdp_bio_train_timer.elapsed,
            stdp_bio_eval_timer.elapsed,
            len(train_ids) * stdp_cfg.epochs,
            eval_tokens,
            "cpu_numpy",
        ),
        "btsp_trace": MethodTiming(
            btsp_train_timer.elapsed,
            btsp_eval_timer.elapsed,
            len(train_ids) * btsp_cfg.epochs,
            eval_tokens,
            "cpu_numpy",
        ),
        "btsp_bio": MethodTiming(
            btsp_bio_train_timer.elapsed,
            btsp_bio_eval_timer.elapsed,
            len(train_ids) * btsp_cfg.epochs,
            eval_tokens,
            "cpu_numpy",
        ),
        "dendritic_error_1810_lite": MethodTiming(
            dendritic_train_timer.elapsed,
            dendritic_eval_timer.elapsed,
            len(train_ids) - 1,
            eval_tokens,
            "cpu_numpy",
        ),
        "recurrent_3factor": MethodTiming(
            recurrent_train_timer.elapsed,
            recurrent_eval_timer.elapsed,
            len(train_ids) - 1,
            eval_tokens,
            "cpu_numpy",
        ),
        "sparse_hebbian_context": MethodTiming(
            sparse_context_train_timer.elapsed,
            sparse_context_eval_timer.elapsed,
            len(train_ids) - 1,
            eval_tokens,
            "cpu_numpy",
        ),
        "semantic_hebbian": MethodTiming(
            semantic_train_timer.elapsed,
            semantic_eval_timer.elapsed,
            len(train_ids) - 1,
            eval_tokens,
            "cpu_numpy",
        ),
        "combined_context": MethodTiming(
            sparse_context_train_timer.elapsed + semantic_train_timer.elapsed,
            combined_eval_timer.elapsed,
            (len(train_ids) - 1) * 2,
            eval_tokens,
            "cpu_numpy",
        ),
        "continuation_backoff": MethodTiming(
            continuation_train_timer.elapsed,
            continuation_eval_timer.elapsed,
            len(train_ids) - 1,
            eval_tokens,
            "cpu_numpy",
        ),
        "hybrid_dendritic_context": MethodTiming(
            dendritic_train_timer.elapsed + sparse_context_train_timer.elapsed,
            hybrid_eval_timer.elapsed,
            (len(train_ids) - 1) * 2,
            eval_tokens,
            "cpu_numpy",
        ),
        "gated_dendritic_context": MethodTiming(
            dendritic_train_timer.elapsed + sparse_context_train_timer.elapsed,
            gated_eval_timer.elapsed,
            (len(train_ids) - 1) * 2,
            eval_tokens,
            "cpu_numpy",
        ),
    }
    if llama_timing is not None:
        timings["torch_llama"] = llama_timing
    if hybrid_llama_metrics is not None:
        timings["hybrid_llama_context"] = MethodTiming(
            llama_timing.train_seconds + sparse_context_train_timer.elapsed if llama_timing is not None else sparse_context_train_timer.elapsed,
            hybrid_llama_eval_seconds,
            (len(train_ids) - 1) + (llama_timing.train_tokens if llama_timing is not None else 0),
            eval_tokens,
            str(llama_device),
        )
    if gated_llama_metrics is not None:
        timings["gated_llama_context"] = MethodTiming(
            llama_timing.train_seconds + sparse_context_train_timer.elapsed if llama_timing is not None else sparse_context_train_timer.elapsed,
            gated_llama_eval_seconds,
            (len(train_ids) - 1) + (llama_timing.train_tokens if llama_timing is not None else 0),
            eval_tokens,
            str(llama_device),
        )

    rows = [
        {
            "method": "stdp_trace",
            **add_timing(stdp_metrics, **asdict(timings["stdp_trace"])),
        },
        {
            "method": "stdp_bio",
            **add_timing(stdp_bio_metrics, **asdict(timings["stdp_bio"])),
        },
        {
            "method": "btsp_trace",
            **add_timing(btsp_metrics, **asdict(timings["btsp_trace"])),
        },
        {
            "method": "btsp_bio",
            **add_timing(btsp_bio_metrics, **asdict(timings["btsp_bio"])),
        },
        {
            "method": "dendritic_error_1810_lite",
            **add_timing(dendritic_metrics, **asdict(timings["dendritic_error_1810_lite"])),
        },
        {
            "method": "recurrent_3factor",
            **add_timing(recurrent_metrics, **asdict(timings["recurrent_3factor"])),
        },
        {
            "method": "sparse_hebbian_context",
            **add_timing(sparse_context_metrics, **asdict(timings["sparse_hebbian_context"])),
        },
        {
            "method": "semantic_hebbian",
            **add_timing(semantic_metrics, **asdict(timings["semantic_hebbian"])),
        },
        {
            "method": "combined_context",
            **add_timing(combined_metrics, **asdict(timings["combined_context"])),
        },
        {
            "method": "continuation_backoff",
            **add_timing(continuation_metrics, **asdict(timings["continuation_backoff"])),
        },
        {
            "method": "hybrid_dendritic_context",
            **add_timing(hybrid_metrics, **asdict(timings["hybrid_dendritic_context"])),
        },
        {
            "method": "gated_dendritic_context",
            **add_timing(gated_metrics, **asdict(timings["gated_dendritic_context"])),
        },
    ]
    if llama_metrics is not None:
        rows.append(
            {
                "method": "torch_llama",
                **add_timing(llama_metrics, **asdict(timings["torch_llama"])),
            }
        )
    if hybrid_llama_metrics is not None:
        rows.append(
            {
                "method": "hybrid_llama_context",
                **add_timing(hybrid_llama_metrics, **asdict(timings["hybrid_llama_context"])),
            }
        )
    if gated_llama_metrics is not None:
        rows.append(
            {
                "method": "gated_llama_context",
                **add_timing(gated_llama_metrics, **asdict(timings["gated_llama_context"])),
            }
        )
    print("\nTiming summary:")
    for row in rows:
        print(
            f"  {row['method']}:"
            f" train={row['train_seconds']:.3f}s"
            f" ({row['train_tokens_per_sec']:.1f} tok/s)"
            f" eval={row['eval_seconds']:.3f}s"
            f" ({row['eval_tokens_per_sec']:.1f} tok/s)"
            f" device={row['device']}"
        )
    write_csv(args.out_dir / "metrics.csv", rows)
    write_csv(args.out_dir / "online_loss_curves.csv", curve_rows)
    write_csv(args.out_dir / "llama_history.csv", history)
    plot_online_loss_curves(curve_rows, args.out_dir / "online_loss_curves.png")
    plot_supervised_history(history, args.out_dir / "llama_training_curve.png")
    save_checkpoint(
        args.out_dir / "checkpoint.npz",
        kept_raw,
        stdp_weights,
        btsp_weights,
        stdp_bio_weights,
        btsp_bio_weights,
        dendritic_model,
        recurrent_model,
        sparse_context_model,
        args,
    )
    if llama_model is not None:
        save_llama_torch_checkpoint(args.out_dir / "llama_torch_model.pt", llama_model, kept_raw)
    write_config(args.out_dir / "config.json", args)

    prompts = (
        read_prompt_file(args.prompt_file)
        if args.prompt_file is not None
        else build_heldout_prompts(train_text, valid_text, tokenizer, kept_raw, count=3)
    )
    prompt_checks = [prompt_status(tokenizer, kept_raw, prompt, train_text) for prompt in prompts]
    if not prompts:
        raise ValueError("No sample prompts available.")
    decode_cfg = DecodingConfig(
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram=args.no_repeat_ngram,
        top_k=args.decode_top_k,
        temperature=args.decode_temperature,
    )

    print("\nSample prompt leakage check:")
    for row in prompt_checks:
        print(
            f"  in_train_prefix={row['in_train_prefix']} "
            f"compact_tokens={row['compact_tokens']}/{row['raw_tokens']} "
            f"prompt={row['prompt']!r}"
        )

    if llama_model is not None:
        sample = sample_llama_torch(
            llama_model,
            tokenizer,
            kept_raw,
            prompts[0],
            args.sample_len,
            args.sample_temperature,
            args.seed + 77,
            llama_device,
        )
        sample_path = args.out_dir / "llama_sample.txt"
    else:
        sample = sample_sparse_hebbian_context(sparse_context_model, tokenizer, kept_raw, prompts[0], args.sample_len)
        sample_path = args.out_dir / "sparse_hebbian_context_sample.txt"
    sample_path.write_text(sample, encoding="utf-8")
    comparison_path = args.out_dir / "greedy_completions.txt"
    generation_metric_rows: list[dict] = []
    with comparison_path.open("w", encoding="utf-8") as f:
        for prompt_idx, (prompt, check) in enumerate(zip(prompts, prompt_checks)):
            outputs = {}
            if llama_model is not None:
                outputs["LlamaTorch"] = sample_llama_torch(
                    llama_model,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    1.0,
                    args.seed,
                    llama_device,
                    greedy=True,
                )
                outputs["HybridLlamaContext"] = sample_hybrid_llama_context(
                    llama_model,
                    sparse_context_model,
                    hybrid_cfg,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    llama_device,
                )
                outputs["HybridLlamaContextControlled"] = sample_hybrid_llama_context(
                    llama_model,
                    sparse_context_model,
                    hybrid_cfg,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    llama_device,
                    decode_cfg,
                    args.seed + prompt_idx + 211,
                )
                outputs["GatedLlamaContext"] = sample_gated_llama_context(
                    llama_model,
                    sparse_context_model,
                    hybrid_cfg,
                    gated_cfg,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    llama_device,
                )
                outputs["GatedLlamaContextControlled"] = sample_gated_llama_context(
                    llama_model,
                    sparse_context_model,
                    hybrid_cfg,
                    gated_cfg,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    llama_device,
                    decode_cfg,
                    args.seed + prompt_idx + 223,
                )
            outputs.update({
                "STDP": sample_plastic(stdp_weights, tokenizer, kept_raw, prompt, args.sample_len, stdp_cfg.trace_decay),
                "STDP-Bio": sample_plastic(
                    stdp_bio_weights,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    stdp_cfg.trace_decay,
                ),
                "BTSP": sample_plastic(btsp_weights, tokenizer, kept_raw, prompt, args.sample_len, btsp_cfg.trace_decay),
                "BTSP-Bio": sample_plastic(
                    btsp_bio_weights,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    btsp_cfg.trace_decay,
                ),
                "DendriticError-1810-lite": sample_dendritic_error(
                    dendritic_model,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                ),
                "RecurrentThreeFactor": sample_recurrent_three_factor(
                    recurrent_model,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                ),
                "SparseHebbianContext": sample_sparse_hebbian_context(
                    sparse_context_model,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                ),
                "SparseHebbianContextControlled": sample_sparse_hebbian_context(
                    sparse_context_model,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    decode_cfg,
                    args.seed + prompt_idx + 101,
                ),
                "HybridDendriticContext": sample_hybrid_dendritic_context(
                    dendritic_model,
                    sparse_context_model,
                    hybrid_cfg,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                ),
                "HybridDendriticContextControlled": sample_hybrid_dendritic_context(
                    dendritic_model,
                    sparse_context_model,
                    hybrid_cfg,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    decode_cfg,
                    args.seed + prompt_idx + 131,
                ),
                "GatedDendriticContext": sample_gated_dendritic_context(
                    dendritic_model,
                    sparse_context_model,
                    hybrid_cfg,
                    gated_cfg,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                ),
                "GatedDendriticContextControlled": sample_gated_dendritic_context(
                    dendritic_model,
                    sparse_context_model,
                    hybrid_cfg,
                    gated_cfg,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    decode_cfg,
                    args.seed + prompt_idx + 151,
                ),
            })
            check_line = (
                f"PROMPT: {prompt!r} | in_train_prefix={check['in_train_prefix']} "
                f"| compact_tokens={check['compact_tokens']}/{check['raw_tokens']}"
            )
            f.write(check_line + "\n")
            print(f"\n{check_line}")
            for name, text in outputs.items():
                f.write(f"\n{name} greedy:\n{text}\n")
                print(f"\n{name} greedy:")
                print(text[:800])
                metric_row = generation_quality_metrics(name, prompt, text)
                metric_row["prompt_index"] = prompt_idx
                generation_metric_rows.append(metric_row)
            f.write("\n" + "=" * 80 + "\n\n")
    write_csv(args.out_dir / "generation_metrics.csv", generation_metric_rows)
    print(f"wrote metrics: {args.out_dir / 'metrics.csv'}")
    print(f"wrote online curves: {args.out_dir / 'online_loss_curves.csv'}")
    print(f"wrote online curve plot: {args.out_dir / 'online_loss_curves.png'}")
    print(f"wrote checkpoint: {args.out_dir / 'checkpoint.npz'}")
    if llama_model is not None:
        print(f"wrote curve: {args.out_dir / 'llama_training_curve.png'}")
    print(f"wrote sample: {sample_path}")
    print(f"wrote greedy completions: {comparison_path}")
    print(f"wrote generation metrics: {args.out_dir / 'generation_metrics.csv'}")
    print("\nSample preview:")
    print(sample[:800])


if __name__ == "__main__":
    main()
