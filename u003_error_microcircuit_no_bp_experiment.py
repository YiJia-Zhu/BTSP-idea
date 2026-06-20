#!/usr/bin/env python3
"""
U003 error-neuron microcircuit no-BP next-token prototype.

This is a small NumPy prototype inspired by error-neuron cortical microcircuits:
representation neurons carry the forward activity, separate error neurons carry
the output mismatch back to every representation layer, and each synapse updates
from only its presynaptic activity and its local postsynaptic error-neuron signal.

No autograd, BP/BPTT, pretrained backbone, answer slots, n-gram tables, or raw
text replay are used by the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

import phase_binding_token_experiment as phase


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = phase.DEFAULT_TRAIN
DEFAULT_VALID = phase.DEFAULT_VALID
DEFAULT_TOKENIZER = phase.DEFAULT_TOKENIZER


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def softmax_loss_and_pred(logits: np.ndarray, target: int, temperature: float) -> tuple[float, int]:
    probs = phase.softmax(logits, temperature)
    return -math.log(float(probs[int(target)]) + 1e-9), int(np.argmax(probs))


def summarize(loss_sum: float, correct: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"loss": float("nan"), "ppl": float("nan"), "accuracy": 0.0, "tokens": 0}
    loss = loss_sum / total
    return {
        "loss": float(loss),
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / total,
        "tokens": int(total),
    }


def sinusoidal_positions(context_len: int, dim: int) -> np.ndarray:
    positions = np.arange(max(int(context_len), 1), dtype=np.float32)[:, None]
    half = max(int(dim) // 2, 1)
    div = np.exp(np.arange(half, dtype=np.float32) * (-math.log(10000.0) / max(half, 1)))
    angles = positions * div[None, :]
    codes = np.zeros((context_len, dim), dtype=np.float32)
    codes[:, 0 : 2 * half : 2] = np.sin(angles[:, : codes[:, 0::2].shape[1]])
    codes[:, 1 : 2 * half : 2] = np.cos(angles[:, : codes[:, 1::2].shape[1]])
    return phase.normalize_rows(codes)


def vector_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.astype(np.float64, copy=False)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = vector_norm(a) * vector_norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)) / denom)


def scaled_like(direction: np.ndarray, target_norm: float) -> np.ndarray:
    norm = vector_norm(direction)
    if norm == 0.0 or target_norm == 0.0:
        return np.zeros_like(direction)
    return (direction * (target_norm / norm)).astype(np.float64)


def center_difference_direction(params: np.ndarray, loss_fn: Any, eps: float) -> np.ndarray:
    flat = params.astype(np.float64, copy=True).ravel()
    grad = np.zeros_like(flat)
    for idx in range(flat.size):
        original = float(flat[idx])
        flat[idx] = original + eps
        plus = loss_fn(flat)
        flat[idx] = original - eps
        minus = loss_fn(flat)
        flat[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)
    return (-grad).reshape(params.shape)


@dataclass
class U003Config:
    context_len: int = 32
    d_model: int = 64
    blocks: int = 4
    heads: int = 4
    block_lr: float = 0.01
    block_bias_lr: float = 0.001
    output_lr: float = 0.05
    feedback_lr: float = 0.0
    logit_scale: float = 8.0
    attention_scale: float = 0.50
    ff_scale: float = 0.50
    residual_scale: float = 1.0
    direct_feedback_weight: float = 1.0
    layered_feedback_weight: float = 0.25
    error_clip: float = 2.0
    row_normalize: bool = True
    temperature: float = 1.0
    feedback_topology: str = "hybrid"
    feedback_init: str = "random"
    use_attention: bool = True
    seed: int = 0


class U003ErrorMicrocircuitNoBPModel:
    def __init__(self, vocab_size: int, cfg: U003Config) -> None:
        self.vocab_size = int(vocab_size)
        self.cfg = cfg
        self.context_len = max(int(cfg.context_len), 1)
        self.d_model = max(int(cfg.d_model), 1)
        self.blocks = max(int(cfg.blocks), 1)
        self.heads = max(int(cfg.heads), 1)
        self.rng = np.random.default_rng(cfg.seed)
        self.token_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (self.vocab_size, self.d_model)).astype(np.float32)
        )
        self.position_codes = sinusoidal_positions(self.context_len, self.d_model)
        self.ff_weights = [
            self.rng.normal(0.0, 1.0 / math.sqrt(self.d_model), (self.d_model, self.d_model)).astype(np.float32)
            for _ in range(self.blocks)
        ]
        self.ff_biases = [np.zeros(self.d_model, dtype=np.float32) for _ in range(self.blocks)]
        self.attn_q = []
        self.attn_k = []
        self.attn_v = []
        self.attn_o = []
        for _ in range(self.blocks):
            self.attn_q.append(self._fixed_projection())
            self.attn_k.append(self._fixed_projection())
            self.attn_v.append(self._fixed_projection())
            self.attn_o.append(self._fixed_projection())
        self.output_weights = phase.normalize_rows(
            self.rng.normal(0.0, 0.01, (self.vocab_size, self.d_model)).astype(np.float32)
        )
        self.output_bias = np.full(self.vocab_size, -math.log(max(self.vocab_size, 1)), dtype=np.float32)
        self.output_counts = np.ones(self.vocab_size, dtype=np.float32)
        self.output_feedback = [
            phase.normalize_rows(
                self.rng.normal(0.0, 1.0 / math.sqrt(self.vocab_size), (self.d_model, self.vocab_size)).astype(
                    np.float32
                )
            )
            for _ in range(self.blocks)
        ]
        self.error_feedback = [
            phase.normalize_rows(
                self.rng.normal(0.0, 1.0 / math.sqrt(self.d_model), (self.d_model, self.d_model)).astype(np.float32)
            )
            for _ in range(max(self.blocks - 1, 0))
        ]
        if cfg.feedback_init == "output_transpose":
            self._reset_feedback_from_output_weights()
        elif cfg.feedback_init != "random":
            raise ValueError(f"unknown feedback_init: {cfg.feedback_init}")

    def _fixed_projection(self) -> np.ndarray:
        return phase.normalize_rows(
            self.rng.normal(0.0, 1.0 / math.sqrt(self.d_model), (self.d_model, self.d_model)).astype(np.float32)
        )

    def _reset_feedback_from_output_weights(self) -> None:
        tied = phase.normalize_rows(self.output_weights.T.astype(np.float32))
        self.output_feedback = [tied.copy() for _ in range(self.blocks)]

    def parameter_count(self) -> int:
        arrays = [self.token_codes, self.position_codes, self.output_weights, self.output_bias, self.output_counts]
        arrays.extend(self.ff_weights)
        arrays.extend(self.ff_biases)
        arrays.extend(self.attn_q)
        arrays.extend(self.attn_k)
        arrays.extend(self.attn_v)
        arrays.extend(self.attn_o)
        arrays.extend(self.output_feedback)
        arrays.extend(self.error_feedback)
        return int(sum(array.size for array in arrays))

    def state_bytes(self) -> int:
        arrays = [self.token_codes, self.position_codes, self.output_weights, self.output_bias, self.output_counts]
        arrays.extend(self.ff_weights)
        arrays.extend(self.ff_biases)
        arrays.extend(self.attn_q)
        arrays.extend(self.attn_k)
        arrays.extend(self.attn_v)
        arrays.extend(self.attn_o)
        arrays.extend(self.output_feedback)
        arrays.extend(self.error_feedback)
        return int(sum(array.nbytes for array in arrays))

    def context_matrix(self, context: np.ndarray) -> np.ndarray:
        clipped = np.asarray(context[-self.context_len :], dtype=np.int64)
        offset = self.context_len - len(clipped)
        x = self.token_codes[clipped] + self.position_codes[offset:]
        return phase.normalize_rows(x.astype(np.float32))

    def attention_read(self, seq: np.ndarray, block_idx: int) -> np.ndarray:
        if not self.cfg.use_attention:
            return np.mean(seq, axis=0).astype(np.float32)
        q = seq[-1] @ self.attn_q[block_idx].T
        k = seq @ self.attn_k[block_idx].T
        v = seq @ self.attn_v[block_idx].T
        logits = (k @ q) / math.sqrt(max(self.d_model, 1))
        weights = phase.softmax(logits.astype(np.float32), temperature=1.0)
        read = weights @ v
        out = read @ self.attn_o[block_idx].T
        return phase.normalize_vector(out.astype(np.float32))

    def forward(self, context: np.ndarray, collect: bool = False) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        seq = self.context_matrix(context)
        current = seq[-1]
        traces: list[dict[str, np.ndarray]] = []
        for block_idx in range(self.blocks):
            attn = self.attention_read(seq, block_idx)
            local_input = phase.normalize_vector(current + self.cfg.attention_scale * attn)
            pre = self.ff_weights[block_idx] @ local_input + self.ff_biases[block_idx]
            hidden = np.tanh(pre).astype(np.float32)
            current = phase.normalize_vector(
                self.cfg.residual_scale * local_input + self.cfg.ff_scale * hidden
            )
            seq = seq.copy()
            seq[-1] = current
            if collect:
                traces.append({"input": local_input, "pre": pre.astype(np.float32), "hidden": hidden})
        return current, traces

    def logits_from_feature(self, feature: np.ndarray) -> np.ndarray:
        return (self.cfg.logit_scale * (self.output_weights @ feature) + self.output_bias).astype(np.float32)

    def logits(self, context: np.ndarray) -> np.ndarray:
        feature, _ = self.forward(context, collect=False)
        return self.logits_from_feature(feature)

    def output_error(self, logits: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
        probs = phase.softmax(logits.astype(np.float32), self.cfg.temperature)
        err = -probs.astype(np.float32)
        err[int(target)] += 1.0
        return err.astype(np.float32), probs.astype(np.float32)

    def hidden_error_states(
        self,
        traces: list[dict[str, np.ndarray]],
        output_error: np.ndarray,
    ) -> list[np.ndarray]:
        direct = [self.output_feedback[idx] @ output_error for idx in range(self.blocks)]
        direct = [np.asarray(err, dtype=np.float32) for err in direct]
        layered = [np.zeros(self.d_model, dtype=np.float32) for _ in range(self.blocks)]
        layered[-1] = direct[-1]
        for idx in range(self.blocks - 2, -1, -1):
            layered[idx] = (self.error_feedback[idx] @ layered[idx + 1]).astype(np.float32)
        topology = self.cfg.feedback_topology
        if topology == "direct":
            raw = direct
        elif topology == "layered":
            raw = layered
        elif topology == "hybrid":
            raw = [
                self.cfg.direct_feedback_weight * direct[idx] + self.cfg.layered_feedback_weight * layered[idx]
                for idx in range(self.blocks)
            ]
        else:
            raise ValueError(f"unknown feedback_topology: {topology}")
        states: list[np.ndarray] = []
        for trace, err in zip(traces, raw):
            local = err.astype(np.float32) * (1.0 - np.square(trace["hidden"]))
            norm = float(np.linalg.norm(local))
            if norm > self.cfg.error_clip:
                local = local * (self.cfg.error_clip / (norm + 1e-8))
            states.append(local.astype(np.float32))
        return states

    def update(self, context: np.ndarray, target: int) -> None:
        target = int(target)
        feature, traces = self.forward(context, collect=True)
        logits = self.logits_from_feature(feature)
        out_err, probs = self.output_error(logits, target)
        self.update_output(feature, out_err, target)
        error_states = self.hidden_error_states(traces, out_err)
        self.update_blocks(traces, error_states, out_err)
        self.output_counts[target] += 1.0
        freqs = self.output_counts / float(np.sum(self.output_counts))
        self.output_bias = np.log(np.maximum(freqs, 1e-9)).astype(np.float32)

    def update_output(self, feature: np.ndarray, out_err: np.ndarray, target: int) -> None:
        del target
        self.output_weights = self.output_weights + self.cfg.output_lr * np.outer(out_err, feature).astype(np.float32)
        if self.cfg.row_normalize:
            self.output_weights = phase.normalize_rows(self.output_weights)

    def update_blocks(
        self,
        traces: list[dict[str, np.ndarray]],
        error_states: list[np.ndarray],
        output_error: np.ndarray,
    ) -> None:
        for idx, (trace, local_error) in enumerate(zip(traces, error_states)):
            local_input = trace["input"]
            self.ff_weights[idx] = (
                self.ff_weights[idx] + self.cfg.block_lr * np.outer(local_error, local_input).astype(np.float32)
            )
            if self.cfg.block_bias_lr > 0.0:
                self.ff_biases[idx] = self.ff_biases[idx] + self.cfg.block_bias_lr * local_error.astype(np.float32)
            if self.cfg.row_normalize:
                self.ff_weights[idx] = phase.normalize_rows(self.ff_weights[idx])
        if self.cfg.feedback_lr > 0.0:
            self.learn_feedback(error_states, output_error)

    def learn_feedback(self, error_states: list[np.ndarray], output_error: np.ndarray) -> None:
        for idx, local_error in enumerate(error_states):
            self.output_feedback[idx] = self.output_feedback[idx] + self.cfg.feedback_lr * np.outer(
                local_error, output_error
            ).astype(np.float32)
            self.output_feedback[idx] = phase.normalize_rows(self.output_feedback[idx])
        for idx in range(len(self.error_feedback)):
            self.error_feedback[idx] = self.error_feedback[idx] + self.cfg.feedback_lr * np.outer(
                error_states[idx], error_states[idx + 1]
            ).astype(np.float32)
            self.error_feedback[idx] = phase.normalize_rows(self.error_feedback[idx])


def estimate_deep_params(vocab_size: int, d_model: int, blocks: int) -> int:
    vocab = max(int(vocab_size), 1)
    d = max(int(d_model), 1)
    b = max(int(blocks), 1)
    embeddings_head_feedback = (2 + b) * vocab * d
    per_block_forward_attention = 5 * d * d
    layered_error = max(b - 1, 0) * d * d
    return int(embeddings_head_feedback + b * per_block_forward_attention + layered_error)


def run_pass(
    model: U003ErrorMicrocircuitNoBPModel,
    ids: np.ndarray,
    update: bool,
    limit: int,
) -> dict[str, float | int]:
    order = model.context_len
    if len(ids) <= order:
        return summarize(0.0, 0, 0)
    max_idx = len(ids)
    if limit > 0:
        max_idx = min(max_idx, order + limit)
    loss_sum = 0.0
    correct = 0
    total = 0
    for idx in range(order, max_idx):
        context = ids[idx - order : idx]
        target = int(ids[idx])
        logits = model.logits(context)
        loss, pred = softmax_loss_and_pred(logits, target, model.cfg.temperature)
        loss_sum += loss
        correct += int(pred == target)
        total += 1
        if update:
            model.update(context, target)
    return summarize(loss_sum, correct, total)


def collect_context_diag_batch(
    model: U003ErrorMicrocircuitNoBPModel,
    ids: np.ndarray,
    batch_size: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    contexts: list[np.ndarray] = []
    targets: list[int] = []
    order = model.context_len
    for idx in range(order, min(len(ids), order + max(int(batch_size), 0))):
        contexts.append(ids[idx - order : idx].copy())
        targets.append(int(ids[idx]))
    return contexts, np.asarray(targets, dtype=np.int64)


def output_batch_loss(
    weights: np.ndarray,
    bias: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    scale: float,
    temperature: float,
) -> float:
    if features.shape[0] == 0:
        return 0.0
    losses = []
    for feature, target in zip(features, targets):
        logits = scale * (weights @ feature) + bias
        loss, _ = softmax_loss_and_pred(logits.astype(np.float32), int(target), temperature)
        losses.append(loss)
    return float(np.mean(losses))


def collect_output_diag_batch(
    model: U003ErrorMicrocircuitNoBPModel,
    ids: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    contexts, targets = collect_context_diag_batch(model, ids, batch_size)
    features = []
    for context in contexts:
        feature, _ = model.forward(context, collect=False)
        features.append(feature)
    if not features:
        return np.zeros((0, model.d_model), dtype=np.float32), targets
    return np.stack(features, axis=0).astype(np.float32), targets


def local_output_step(
    model: U003ErrorMicrocircuitNoBPModel,
    features: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    weights = model.output_weights.copy()
    for feature, target in zip(features, targets):
        logits = model.cfg.logit_scale * (weights @ feature) + model.output_bias
        out_err, _ = model.output_error(logits, int(target))
        weights = weights + model.cfg.output_lr * np.outer(out_err, feature).astype(np.float32)
        if model.cfg.row_normalize:
            weights = phase.normalize_rows(weights)
    return weights


def run_output_center_diff_diag(
    model: U003ErrorMicrocircuitNoBPModel,
    ids: np.ndarray,
    batch_size: int,
    eps: float,
    max_params: int,
) -> dict[str, float | int | str]:
    features, targets = collect_output_diag_batch(model, ids, batch_size)
    if features.shape[0] == 0:
        return {"diag_examples": 0, "status": "empty"}
    weights0 = model.output_weights.copy().astype(np.float64)
    if weights0.size > max_params:
        return {
            "diag_examples": int(features.shape[0]),
            "diag_params": int(weights0.size),
            "status": "skipped_param_limit",
        }
    bias = model.output_bias.copy()
    before = output_batch_loss(weights0.astype(np.float32), bias, features, targets, model.cfg.logit_scale, model.cfg.temperature)
    local_weights = local_output_step(model, features, targets).astype(np.float64)
    local_delta = local_weights - weights0
    local_norm = vector_norm(local_delta)

    def loss_fn(flat: np.ndarray) -> float:
        w = flat.reshape(weights0.shape).astype(np.float32)
        return output_batch_loss(w, bias, features, targets, model.cfg.logit_scale, model.cfg.temperature)

    cd_direction = center_difference_direction(weights0, loss_fn, eps)
    cd_step = scaled_like(cd_direction, local_norm)
    rng = np.random.default_rng(model.cfg.seed + 9001)
    random_step = scaled_like(rng.normal(0.0, 1.0, weights0.shape), local_norm)
    local_loss = loss_fn((weights0 + local_delta).ravel())
    cd_loss = loss_fn((weights0 + cd_step).ravel())
    random_loss = loss_fn((weights0 + random_step).ravel())
    return {
        "status": "ok",
        "diag_examples": int(features.shape[0]),
        "diag_params": int(weights0.size),
        "loss_before": float(before),
        "loss_after_local": float(local_loss),
        "loss_after_center_diff_scaled": float(cd_loss),
        "loss_after_random_scaled": float(random_loss),
        "loss_change_local": float(local_loss - before),
        "loss_change_center_diff_scaled": float(cd_loss - before),
        "loss_change_random_scaled": float(random_loss - before),
        "cosine_local_vs_center_diff": cosine_similarity(local_delta, cd_direction),
        "local_delta_norm": float(local_norm),
    }


def block_batch_loss(
    model: U003ErrorMicrocircuitNoBPModel,
    contexts: list[np.ndarray],
    targets: np.ndarray,
    block_idx: int,
    weights: np.ndarray,
) -> float:
    if not contexts:
        return 0.0
    original = model.ff_weights[block_idx]
    model.ff_weights[block_idx] = weights.astype(np.float32, copy=False)
    try:
        losses = []
        for context, target in zip(contexts, targets):
            logits = model.logits(context)
            loss, _ = softmax_loss_and_pred(logits, int(target), model.cfg.temperature)
            losses.append(loss)
        return float(np.mean(losses))
    finally:
        model.ff_weights[block_idx] = original


def local_block_step(
    model: U003ErrorMicrocircuitNoBPModel,
    contexts: list[np.ndarray],
    targets: np.ndarray,
    block_idx: int,
) -> np.ndarray:
    weights = model.ff_weights[block_idx].copy()
    for context, target in zip(contexts, targets):
        feature, traces = model.forward(context, collect=True)
        logits = model.logits_from_feature(feature)
        out_err, _ = model.output_error(logits, int(target))
        error_states = model.hidden_error_states(traces, out_err)
        local_error = error_states[block_idx]
        local_input = traces[block_idx]["input"]
        weights = weights + model.cfg.block_lr * np.outer(local_error, local_input).astype(np.float32)
        if model.cfg.row_normalize:
            weights = phase.normalize_rows(weights)
    return weights


def run_block_center_diff_diag(
    model: U003ErrorMicrocircuitNoBPModel,
    ids: np.ndarray,
    batch_size: int,
    eps: float,
    max_params: int,
) -> list[dict[str, float | int | str]]:
    contexts, targets = collect_context_diag_batch(model, ids, batch_size)
    rows: list[dict[str, float | int | str]] = []
    if not contexts:
        return rows
    for block_idx in range(model.blocks):
        weights0 = model.ff_weights[block_idx].copy().astype(np.float64)
        if weights0.size > max_params:
            rows.append(
                {
                    "block": int(block_idx),
                    "diag_examples": int(len(contexts)),
                    "diag_params": int(weights0.size),
                    "status": "skipped_param_limit",
                }
            )
            continue
        before = block_batch_loss(model, contexts, targets, block_idx, weights0.astype(np.float32))
        local_weights = local_block_step(model, contexts, targets, block_idx).astype(np.float64)
        local_delta = local_weights - weights0
        local_norm = vector_norm(local_delta)

        def loss_fn(flat: np.ndarray) -> float:
            w = flat.reshape(weights0.shape).astype(np.float32)
            return block_batch_loss(model, contexts, targets, block_idx, w)

        cd_direction = center_difference_direction(weights0, loss_fn, eps)
        cd_step = scaled_like(cd_direction, local_norm)
        rng = np.random.default_rng(model.cfg.seed + 10_003 + block_idx)
        random_step = scaled_like(rng.normal(0.0, 1.0, weights0.shape), local_norm)
        local_loss = loss_fn((weights0 + local_delta).ravel())
        cd_loss = loss_fn((weights0 + cd_step).ravel())
        random_loss = loss_fn((weights0 + random_step).ravel())
        rows.append(
            {
                "block": int(block_idx),
                "diag_examples": int(len(contexts)),
                "diag_params": int(weights0.size),
                "status": "ok",
                "loss_before": float(before),
                "loss_after_local": float(local_loss),
                "loss_after_center_diff_scaled": float(cd_loss),
                "loss_after_random_scaled": float(random_loss),
                "loss_change_local": float(local_loss - before),
                "loss_change_center_diff_scaled": float(cd_loss - before),
                "loss_change_random_scaled": float(random_loss - before),
                "cosine_local_vs_center_diff": cosine_similarity(local_delta, cd_direction),
                "local_delta_norm": float(local_norm),
            }
        )
    return rows


def summarize_block_diag(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return {
            "hidden_diag_layers": 0,
            "hidden_diag_mean_cosine": float("nan"),
            "hidden_diag_mean_loss_change_local": float("nan"),
            "hidden_diag_mean_loss_change_center_diff": float("nan"),
        }
    cosines = [float(row["cosine_local_vs_center_diff"]) for row in ok_rows]
    local_changes = [float(row["loss_change_local"]) for row in ok_rows]
    cd_changes = [float(row["loss_change_center_diff_scaled"]) for row in ok_rows]
    return {
        "hidden_diag_layers": int(len(ok_rows)),
        "hidden_diag_mean_cosine": float(np.mean(cosines)),
        "hidden_diag_min_cosine": float(np.min(cosines)),
        "hidden_diag_max_cosine": float(np.max(cosines)),
        "hidden_diag_mean_loss_change_local": float(np.mean(local_changes)),
        "hidden_diag_mean_loss_change_center_diff": float(np.mean(cd_changes)),
    }


def load_tinystories(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int]:
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    train_raw = phase.encode_text(tokenizer, phase.read_prefix(args.train_file, args.train_chars))
    valid_raw = phase.encode_text(tokenizer, phase.read_prefix(args.valid_file, args.valid_chars))
    _, train_ids, valid_ids = phase.build_compact_vocab(train_raw, valid_raw, args.max_vocab)
    return train_ids, valid_ids, int(min(args.max_vocab, len(set(train_ids.tolist()) | set(valid_ids.tolist()))))


def load_temporal(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int]:
    pattern = np.array([0, 1, 2, 3], dtype=np.int64)
    train_ids = np.tile(pattern, max(int(args.temporal_train_cycles), 1))
    valid_ids = np.tile(pattern, max(int(args.temporal_valid_cycles), 1))
    return train_ids, valid_ids, 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tinystories", "temporal"], default="tinystories")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "u003_error_microcircuit_no_bp")
    parser.add_argument("--train-chars", type=int, default=20_000)
    parser.add_argument("--valid-chars", type=int, default=5_000)
    parser.add_argument("--max-vocab", type=int, default=128)
    parser.add_argument("--temporal-train-cycles", type=int, default=128)
    parser.add_argument("--temporal-valid-cycles", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--block-lr", type=float, default=0.01)
    parser.add_argument("--block-bias-lr", type=float, default=0.001)
    parser.add_argument("--output-lr", type=float, default=0.05)
    parser.add_argument("--feedback-lr", type=float, default=0.0)
    parser.add_argument("--logit-scale", type=float, default=8.0)
    parser.add_argument("--attention-scale", type=float, default=0.50)
    parser.add_argument("--ff-scale", type=float, default=0.50)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--direct-feedback-weight", type=float, default=1.0)
    parser.add_argument("--layered-feedback-weight", type=float, default=0.25)
    parser.add_argument("--error-clip", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--feedback-topology", choices=["direct", "layered", "hybrid"], default="hybrid")
    parser.add_argument("--feedback-init", choices=["random", "output_transpose"], default="random")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--eval-token-limit", type=int, default=0)
    parser.add_argument("--diag-batch", type=int, default=4)
    parser.add_argument("--hidden-diag-batch", type=int, default=4)
    parser.add_argument("--center-eps", type=float, default=1e-4)
    parser.add_argument("--diag-max-params", type=int, default=16_384)
    parser.add_argument("--large-vocab-size", type=int, default=50_000)
    parser.add_argument("--large-d-model", type=int, default=1536)
    parser.add_argument("--large-blocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "tinystories":
        train_ids, valid_ids, vocab_size = load_tinystories(args)
    else:
        train_ids, valid_ids, vocab_size = load_temporal(args)

    cfg = U003Config(
        context_len=args.context_len,
        d_model=args.d_model,
        blocks=args.blocks,
        heads=args.heads,
        block_lr=args.block_lr,
        block_bias_lr=args.block_bias_lr,
        output_lr=args.output_lr,
        feedback_lr=args.feedback_lr,
        logit_scale=args.logit_scale,
        attention_scale=args.attention_scale,
        ff_scale=args.ff_scale,
        residual_scale=args.residual_scale,
        direct_feedback_weight=args.direct_feedback_weight,
        layered_feedback_weight=args.layered_feedback_weight,
        error_clip=args.error_clip,
        temperature=args.temperature,
        feedback_topology=args.feedback_topology,
        feedback_init=args.feedback_init,
        use_attention=not args.no_attention,
        seed=args.seed,
    )
    model = U003ErrorMicrocircuitNoBPModel(vocab_size, cfg)
    large_estimate = estimate_deep_params(args.large_vocab_size, args.large_d_model, args.large_blocks)

    start = time.perf_counter()
    train_summary = run_pass(model, train_ids, update=True, limit=args.eval_token_limit)
    train_seconds = time.perf_counter() - start
    valid_pre = run_pass(model, valid_ids, update=False, limit=args.eval_token_limit)
    valid_online = run_pass(model, valid_ids, update=True, limit=args.eval_token_limit)
    valid_post = run_pass(model, valid_ids, update=False, limit=args.eval_token_limit)
    output_diag = run_output_center_diff_diag(model, valid_ids, args.diag_batch, args.center_eps, args.diag_max_params)
    hidden_diag_rows = run_block_center_diff_diag(
        model,
        valid_ids,
        args.hidden_diag_batch,
        args.center_eps,
        args.diag_max_params,
    )
    hidden_diag_summary = summarize_block_diag(hidden_diag_rows)

    summary = {
        "task": args.task,
        "seed": int(args.seed),
        "vocab_size": int(vocab_size),
        "context_len": int(cfg.context_len),
        "d_model": int(cfg.d_model),
        "blocks": int(cfg.blocks),
        "feedback_topology": cfg.feedback_topology,
        "feedback_init": cfg.feedback_init,
        "actual_params": int(model.parameter_count()),
        "actual_state_bytes": int(model.state_bytes()),
        "large_param_estimate": int(large_estimate),
        "train_loss": train_summary["loss"],
        "train_acc": train_summary["accuracy"],
        "valid_pre_loss": valid_pre["loss"],
        "valid_pre_acc": valid_pre["accuracy"],
        "valid_online_loss": valid_online["loss"],
        "valid_online_acc": valid_online["accuracy"],
        "valid_post_loss": valid_post["loss"],
        "valid_post_acc": valid_post["accuracy"],
        "train_seconds": float(train_seconds),
    }
    summary.update({f"diag_{key}": value for key, value in output_diag.items()})
    summary.update(hidden_diag_summary)
    write_csv(args.out_dir / "summary.csv", [summary])
    write_csv(args.out_dir / "output_center_diff_diagnostic.csv", [output_diag])
    write_csv(args.out_dir / "hidden_center_diff_diagnostic.csv", hidden_diag_rows)
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "config": asdict(cfg),
                "summary": summary,
            },
            f,
            indent=2,
        )
    print("Summary:")
    print(
        f"  task={args.task} params={model.parameter_count():,} "
        f"large_estimate={large_estimate:,} "
        f"valid_post={valid_post['loss']:.3f}/{valid_post['accuracy']:.3f} "
        f"out_cos={output_diag.get('cosine_local_vs_center_diff', float('nan')):.3f} "
        f"hidden_mean_cos={hidden_diag_summary.get('hidden_diag_mean_cosine', float('nan')):.3f}"
    )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
