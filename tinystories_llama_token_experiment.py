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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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


def save_checkpoint(
    path: Path,
    kept_raw: np.ndarray,
    stdp: np.ndarray,
    btsp: np.ndarray,
    stdp_bio: np.ndarray,
    btsp_bio: np.ndarray,
    dendritic_error: DendriticErrorNetwork,
    args: argparse.Namespace,
) -> None:
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
        stdp_trace_decay=np.array([STDPConfig().trace_decay], dtype=np.float32),
        btsp_trace_decay=np.array([BTSPConfig().trace_decay], dtype=np.float32),
        dendritic_trace_decay=np.array([DendriticErrorConfig().trace_decay], dtype=np.float32),
        dendritic_lambda_apical=np.array([DendriticErrorConfig().lambda_apical], dtype=np.float32),
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
        "torch_llama": llama_timing,
    }

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
            "method": "torch_llama",
            **add_timing(llama_metrics, **asdict(timings["torch_llama"])),
        },
    ]
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
        args,
    )
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

    print("\nSample prompt leakage check:")
    for row in prompt_checks:
        print(
            f"  in_train_prefix={row['in_train_prefix']} "
            f"compact_tokens={row['compact_tokens']}/{row['raw_tokens']} "
            f"prompt={row['prompt']!r}"
        )

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
    sample_path.write_text(sample, encoding="utf-8")
    comparison_path = args.out_dir / "greedy_completions.txt"
    with comparison_path.open("w", encoding="utf-8") as f:
        for prompt, check in zip(prompts, prompt_checks):
            outputs = {
                "LlamaTorch": sample_llama_torch(
                    llama_model,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    1.0,
                    args.seed,
                    llama_device,
                    greedy=True,
                ),
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
            }
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
            f.write("\n" + "=" * 80 + "\n\n")
    print(f"wrote metrics: {args.out_dir / 'metrics.csv'}")
    print(f"wrote online curves: {args.out_dir / 'online_loss_curves.csv'}")
    print(f"wrote online curve plot: {args.out_dir / 'online_loss_curves.png'}")
    print(f"wrote checkpoint: {args.out_dir / 'checkpoint.npz'}")
    print(f"wrote curve: {args.out_dir / 'llama_training_curve.png'}")
    print(f"wrote sample: {sample_path}")
    print(f"wrote greedy completions: {comparison_path}")
    print("\nSample preview:")
    print(sample[:800])


if __name__ == "__main__":
    main()
