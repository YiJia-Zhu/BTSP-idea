#!/usr/bin/env python3
"""
Strict prequential online stream for the pure no-BP phase-binding token learner.

The main method is the competitive branch phase model from
phase_binding_token_experiment.py.  It predicts each next token before updating
from the ground-truth token, stores only learned state, and never stores raw
training text in the model state.  Sparse context counts are included only as an
auxiliary statistical baseline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

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
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def softmax_loss_and_pred(scores: np.ndarray, target: int, temperature: float) -> tuple[float, int]:
    probs = phase.softmax(scores, temperature)
    return -math.log(float(probs[target]) + 1e-9), int(np.argmax(probs))


def summarize(loss_sum: float, correct: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"loss": float("nan"), "ppl": float("nan"), "accuracy": 0.0}
    loss = loss_sum / total
    return {"loss": float(loss), "ppl": float(math.exp(min(loss, 20.0))), "accuracy": correct / total}


def summarize_candidate_ranks(
    rank_sum: float,
    margin_sum: float,
    wrong_margin_sum: float,
    top_hits: dict[int, int],
    error_top_hits: dict[int, int],
    error_count: int,
    total: int,
) -> dict[str, float | int]:
    """Summarize whether errors are top-k winner-selection mistakes."""
    summary: dict[str, float | int] = {
        "target_rank_mean": rank_sum / max(total, 1),
        "target_margin_mean": margin_sum / max(total, 1),
        "error_count": int(error_count),
        "error_wrong_margin_mean": wrong_margin_sum / max(error_count, 1),
    }
    for k in sorted(top_hits):
        summary[f"target_top{k}_acc"] = top_hits[k] / max(total, 1)
        summary[f"error_target_top{k}_rate"] = error_top_hits[k] / max(error_count, 1)
        summary[f"oracle_top{k}_acc"] = (total - error_count + error_top_hits[k]) / max(total, 1)
    return summary


def truncate_history(history: Sequence[int], keep: int) -> list[int]:
    if keep <= 0:
        return []
    return list(history[-keep:])


def segment_windows(ids: np.ndarray, segment_tokens: int) -> list[tuple[int, int]]:
    stride = max(segment_tokens - 1, 1)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < len(ids) - 1:
        end = min(len(ids), start + segment_tokens)
        if end - start < 2:
            break
        windows.append((start, end))
        if end >= len(ids):
            break
        start += stride
    return windows


class OnlineCompetitivePhaseMemory:
    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        self.branch_model = phase.BranchPhaseTokenLearner(vocab_size, cfg, branch_orders, branch_weights)
        self.vocab_size = vocab_size
        self.readout = phase.CompetitiveBranchReadout(
            self.branch_model,
            lr=competitive_lr,
            neg_k=competitive_neg_k,
            epochs=competitive_epochs,
            score_scale=competitive_score_scale,
            init=competitive_init,
            margin=competitive_margin,
            seed=seed,
        )
        self.max_order = self.branch_model.max_order
        self.bias_weight = cfg.bias_weight
        self.temperature = cfg.temperature

    def scores(self, context: np.ndarray) -> np.ndarray:
        return self.readout.scores(context, bias_weight=self.bias_weight)

    def update(self, context: np.ndarray, target: int) -> None:
        self.branch_model.update_context(context, target)
        self.readout.output_bias = self.branch_model.output_bias.copy()
        self.readout.update_context(context, target)

    def state_bytes(self) -> int:
        return int(self.branch_model.state_bytes() + self.readout.state_bytes())

    def active_contexts(self) -> int:
        return int(sum(np.count_nonzero(branch.prototype_counts) for branch in self.branch_model.branches))


class OnlineSparseAuxMemory:
    def __init__(self, vocab_size: int, context_order: int, alpha: float, temperature: float) -> None:
        self.memory = phase.SparseContextAux(vocab_size, phase.SparseAuxConfig(context_order=context_order, alpha=alpha, temperature=temperature))
        self.max_order = max(context_order, 1)
        self.temperature = temperature

    def scores(self, context: np.ndarray) -> np.ndarray:
        probs = self.memory.distribution(context[-self.max_order :])
        return np.log(np.maximum(probs, 1e-9)).astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        self.memory.rows[tuple(int(token) for token in context[-self.max_order :])][int(target)] += 1
        self.memory.unigram[int(target)] += 1

    def state_bytes(self) -> int:
        return int(self.memory.state_bytes_estimate())

    def active_contexts(self) -> int:
        return int(len(self.memory.rows))


class OnlineTraceCompetitivePhaseMemory:
    """
    Phase branches plus a fixed leaky token-trace branch.

    The trace branch is a short SSM/reservoir-like state over recent token codes:
    h_t = decay * h_{t-1} + code[token_t].  Only the local WTA readout and phase
    branch state are plastic; no BP or replay is used.
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        self.branch_model = phase.BranchPhaseTokenLearner(vocab_size, cfg, branch_orders, branch_weights)
        self.vocab_size = vocab_size
        self.max_order = max(self.branch_model.max_order, int(trace_order))
        self.trace_order = max(int(trace_order), 1)
        self.trace_decay = float(np.clip(trace_decay, 0.0, 0.999))
        self.trace_weight = float(trace_weight)
        self.bias_weight = cfg.bias_weight
        self.temperature = cfg.temperature
        self.lr = competitive_lr
        self.neg_k = max(int(competitive_neg_k), 0)
        self.epochs = max(int(competitive_epochs), 1)
        self.score_scale = competitive_score_scale
        self.margin = competitive_margin
        self.init = competitive_init
        self.rng = np.random.default_rng(seed + 7919)
        self.trace_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (vocab_size, max(int(trace_dim), 1))).astype(np.float32)
        )
        phase_dim = sum(2 * branch.cfg.complex_dim for branch in self.branch_model.branches)
        self.feature_dim = phase_dim + self.trace_codes.shape[1]
        if competitive_init == "random":
            self.weights = phase.normalize_rows(
                self.rng.normal(0.0, 0.01, (vocab_size, self.feature_dim)).astype(np.float32)
            )
        else:
            branch_weights_init = np.concatenate(
                [branch.prototypes for branch in self.branch_model.branches],
                axis=1,
            ).astype(np.float32)
            trace_zeros = np.zeros((vocab_size, self.trace_codes.shape[1]), dtype=np.float32)
            self.weights = phase.normalize_rows(np.concatenate([branch_weights_init, trace_zeros], axis=1))
        self.output_bias = self.branch_model.output_bias.copy()

    def trace_feature(self, context: np.ndarray) -> np.ndarray:
        state = np.zeros(self.trace_codes.shape[1], dtype=np.float32)
        for token in context[-self.trace_order :]:
            state = self.trace_decay * state + self.trace_codes[int(token)]
        return phase.normalize_vector(state)

    def feature(self, context: np.ndarray) -> np.ndarray:
        branch_features = [
            branch.feature(context[-order:])
            for order, branch in zip(self.branch_model.branch_orders, self.branch_model.branches)
        ]
        trace = self.trace_weight * self.trace_feature(context)
        return phase.normalize_vector(np.concatenate(branch_features + [trace]).astype(np.float32))

    def scores(self, context: np.ndarray) -> np.ndarray:
        feature = self.feature(context)
        return (self.score_scale * (self.weights @ feature) + self.bias_weight * self.output_bias).astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        self.branch_model.update_context(context, target)
        self.output_bias = self.branch_model.output_bias.copy()
        feature = self.feature(context)
        target = int(target)
        scores = self.weights @ feature
        self.weights[target] = phase.normalize_vector(self.weights[target] + self.lr * feature)
        if self.neg_k <= 0:
            return
        scores[target] = -1e9
        k = min(self.neg_k, self.vocab_size - 1)
        wrongs = np.argpartition(scores, -k)[-k:]
        target_score = float(self.weights[target] @ feature)
        for wrong in wrongs:
            if float(scores[wrong]) + self.margin > target_score:
                self.weights[wrong] = phase.normalize_vector(self.weights[wrong] - (self.lr / k) * feature)

    def state_bytes(self) -> int:
        return int(self.branch_model.state_bytes() + self.trace_codes.nbytes + self.weights.nbytes + self.output_bias.nbytes)

    def active_contexts(self) -> int:
        return int(sum(np.count_nonzero(branch.prototype_counts) for branch in self.branch_model.branches))


class OnlineTraceApicalGatedCompetitivePhaseMemory(OnlineTraceCompetitivePhaseMemory):
    """
    Trace/phase WTA readout with branch-wise apical prediction-error gates.

    Each dendritic segment computes a local target-vs-wrong margin.  A decaying
    apical error trace boosts updates on segments that locally failed to support
    the target.  This is a branch-local three-factor modulation of the WTA
    readout update; it does not use BP/BPTT or raw-text replay.
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        apical_decay: float,
        apical_strength: float,
        apical_margin: float,
        apical_min_gate: float,
        apical_max_gate: float,
        apical_error_clip: float,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
        apical_error_mode: str = "segment_margin",
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            trace_order,
            trace_dim,
            trace_decay,
            trace_weight,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
        )
        self.apical_decay = float(np.clip(apical_decay, 0.0, 0.999))
        self.apical_strength = max(float(apical_strength), 0.0)
        self.apical_margin = float(apical_margin)
        self.apical_min_gate = max(float(apical_min_gate), 0.0)
        self.apical_max_gate = max(float(apical_max_gate), self.apical_min_gate + 1e-6)
        self.apical_error_clip = max(float(apical_error_clip), 1e-6)
        self.apical_error_mode = str(apical_error_mode)
        if self.apical_error_mode not in {"segment_margin", "global_margin", "random_feedback", "fixed_random"}:
            raise ValueError(f"unknown apical error mode: {self.apical_error_mode}")
        self.segment_slices: list[slice] = []
        offset = 0
        for branch in self.branch_model.branches:
            width = 2 * branch.cfg.complex_dim
            self.segment_slices.append(slice(offset, offset + width))
            offset += width
        self.segment_slices.append(slice(offset, self.feature_dim))
        self.apical_error_trace = np.zeros(len(self.segment_slices), dtype=np.float32)
        apical_rng = np.random.default_rng(seed + 67867967)
        feedback = np.abs(apical_rng.normal(0.0, 1.0, len(self.segment_slices)).astype(np.float32))
        self.apical_feedback = feedback / max(float(np.mean(feedback)), 1e-6)
        fixed_gate = apical_rng.normal(0.0, 1.0, len(self.segment_slices)).astype(np.float32)
        self.apical_fixed_gate = fixed_gate / max(float(np.max(np.abs(fixed_gate))), 1e-6)

    def reset_dynamic_state(self) -> None:
        self.apical_error_trace.fill(0.0)

    def branch_error_gates(self, feature: np.ndarray, scores: np.ndarray, target: int) -> np.ndarray:
        target = int(target)
        adjusted = scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        k = min(max(self.neg_k, 1), self.vocab_size - 1)
        if k <= 0:
            return np.ones(len(self.segment_slices), dtype=np.float32)
        wrongs = np.argpartition(adjusted, -k)[-k:]
        errors = np.zeros(len(self.segment_slices), dtype=np.float32)
        full_margin = float(scores[target] - np.max(scores[wrongs]))
        if self.apical_error_mode == "fixed_random":
            gates = 1.0 + self.apical_strength * self.apical_fixed_gate
            return np.clip(gates, self.apical_min_gate, self.apical_max_gate).astype(np.float32)
        if self.apical_error_mode == "global_margin":
            errors.fill(max(0.0, self.apical_margin - full_margin))
        elif self.apical_error_mode == "random_feedback":
            errors = max(0.0, self.apical_margin - full_margin) * self.apical_feedback
        else:
            for idx, segment_slice in enumerate(self.segment_slices):
                segment = feature[segment_slice]
                target_score = float(self.weights[target, segment_slice] @ segment)
                wrong_score = float(np.max(self.weights[wrongs, segment_slice] @ segment))
                local_margin = target_score - wrong_score
                errors[idx] = max(0.0, self.apical_margin - local_margin)
        self.apical_error_trace = self.apical_decay * self.apical_error_trace + errors
        np.clip(self.apical_error_trace, 0.0, self.apical_error_clip, out=self.apical_error_trace)
        gates = 1.0 + self.apical_strength * self.apical_error_trace
        return np.clip(gates, self.apical_min_gate, self.apical_max_gate).astype(np.float32)

    def apply_segment_gates(self, feature: np.ndarray, gates: np.ndarray) -> np.ndarray:
        gated = feature.astype(np.float32, copy=True)
        for gate, segment_slice in zip(gates, self.segment_slices):
            gated[segment_slice] *= float(gate)
        return phase.normalize_vector(gated)

    def update(self, context: np.ndarray, target: int) -> None:
        target = int(target)
        pre_scores = self.scores(context)
        self.branch_model.update_context(context, target)
        self.output_bias = self.branch_model.output_bias.copy()
        feature = self.feature(context)
        gates = self.branch_error_gates(feature, pre_scores, target)
        gated_feature = self.apply_segment_gates(feature, gates)
        self.weights[target] = phase.normalize_vector(self.weights[target] + self.lr * gated_feature)
        if self.neg_k <= 0:
            return
        adjusted = pre_scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        k = min(self.neg_k, self.vocab_size - 1)
        wrongs = np.argpartition(adjusted, -k)[-k:]
        target_score = float(pre_scores[target])
        for wrong in wrongs:
            wrong = int(wrong)
            if float(pre_scores[wrong]) + self.margin > target_score:
                self.weights[wrong] = phase.normalize_vector(self.weights[wrong] - (self.lr / k) * gated_feature)

    def state_bytes(self) -> int:
        bytes_used = self.apical_error_trace.nbytes
        if self.apical_error_mode == "random_feedback":
            bytes_used += self.apical_feedback.nbytes
        if self.apical_error_mode == "fixed_random":
            bytes_used += self.apical_fixed_gate.nbytes
        return int(super().state_bytes() + bytes_used)


class EPropTraceFeatureMixin:
    """Finite-window eligibility trace over local phase/trace features."""

    def init_eprop_trace(self, eprop_order: int, eprop_decay: float, eprop_weight: float) -> None:
        self.eprop_order = max(int(eprop_order), 1)
        self.eprop_decay = float(np.clip(eprop_decay, 0.0, 0.999))
        self.eprop_weight = float(np.clip(eprop_weight, 0.0, 1.0))
        self.max_order = max(int(self.max_order), self.eprop_order)

    def instantaneous_feature(self, context: np.ndarray) -> np.ndarray:
        return OnlineTraceCompetitivePhaseMemory.feature(self, context)

    def eligibility_feature(self, context: np.ndarray) -> np.ndarray:
        context = np.asarray(context, dtype=np.int64)
        current = self.instantaneous_feature(context)
        state = np.zeros_like(current, dtype=np.float32)
        start = max(1, len(context) - self.eprop_order + 1)
        for end in range(start, len(context) + 1):
            feature = self.instantaneous_feature(context[:end])
            state = self.eprop_decay * state + feature
        eligibility = phase.normalize_vector(state)
        if self.eprop_weight >= 1.0:
            return eligibility
        return phase.normalize_vector((1.0 - self.eprop_weight) * current + self.eprop_weight * eligibility)

    def feature(self, context: np.ndarray) -> np.ndarray:
        return self.eligibility_feature(context)


class OnlineEPropTraceCompetitivePhaseMemory(EPropTraceFeatureMixin, OnlineTraceCompetitivePhaseMemory):
    """WTA readout over a local e-prop-style eligibility trace."""

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        eprop_order: int,
        eprop_decay: float,
        eprop_weight: float,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            trace_order,
            trace_dim,
            trace_decay,
            trace_weight,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
        )
        self.init_eprop_trace(eprop_order, eprop_decay, eprop_weight)


class OnlineEPropTraceApicalGatedCompetitivePhaseMemory(
    EPropTraceFeatureMixin,
    OnlineTraceApicalGatedCompetitivePhaseMemory,
):
    """Apical-gated WTA readout over a local e-prop-style eligibility trace."""

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        eprop_order: int,
        eprop_decay: float,
        eprop_weight: float,
        apical_decay: float,
        apical_strength: float,
        apical_margin: float,
        apical_min_gate: float,
        apical_max_gate: float,
        apical_error_clip: float,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
        apical_error_mode: str = "segment_margin",
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            trace_order,
            trace_dim,
            trace_decay,
            trace_weight,
            apical_decay,
            apical_strength,
            apical_margin,
            apical_min_gate,
            apical_max_gate,
            apical_error_clip,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
            apical_error_mode,
        )
        self.init_eprop_trace(eprop_order, eprop_decay, eprop_weight)


class OnlineDLLDeepLocalMemory(OnlineTraceCompetitivePhaseMemory):
    """
    Deep local-learning stack inspired by Dendritic Localized Learning.

    The phase/trace encoder supplies the basal input.  Each hidden layer has an
    independent fixed random target projection for the next-token label and is
    updated by its own local squared-error signal only:

        delta_l ~= (target_l[token] - h_l) * activation'(z_l)

    No layer receives a backpropagated downstream gradient, and no raw text is
    stored in state.
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        dll_hidden_dims: list[int],
        dll_label_dim: int,
        dll_lr: float,
        dll_bias_lr: float,
        dll_delta_clip: float,
        dll_activation: str,
        dll_row_normalize: bool,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            trace_order,
            trace_dim,
            trace_decay,
            trace_weight,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
        )
        hidden_dims = [max(int(dim), 1) for dim in dll_hidden_dims]
        if not hidden_dims:
            raise ValueError("--dll-hidden-dims must contain at least one layer width")
        self.dll_lr = max(float(dll_lr), 0.0)
        self.dll_bias_lr = max(float(dll_bias_lr), 0.0)
        self.dll_delta_clip = max(float(dll_delta_clip), 1e-6)
        self.dll_activation = str(dll_activation)
        if self.dll_activation not in {"tanh", "linear"}:
            raise ValueError(f"unknown DLL activation: {self.dll_activation}")
        self.dll_row_normalize = bool(dll_row_normalize)
        self.base_feature_dim = int(self.feature_dim)
        self.dll_hidden_dims = hidden_dims

        dll_rng = np.random.default_rng(seed + 19349663)
        label_dim = max(int(dll_label_dim), 1)
        self.dll_label_codes = phase.normalize_rows(
            dll_rng.normal(0.0, 1.0, (vocab_size, label_dim)).astype(np.float32)
        )
        dims = [self.base_feature_dim] + hidden_dims
        self.dll_weights: list[np.ndarray] = []
        self.dll_biases: list[np.ndarray] = []
        self.dll_feedback: list[np.ndarray] = []
        self.dll_targets: list[np.ndarray] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            scale = 1.0 / math.sqrt(max(in_dim, 1))
            weights = dll_rng.normal(0.0, scale, (out_dim, in_dim)).astype(np.float32)
            self.dll_weights.append(phase.normalize_rows(weights) if self.dll_row_normalize else weights)
            self.dll_biases.append(np.zeros(out_dim, dtype=np.float32))
            feedback = dll_rng.normal(0.0, 1.0 / math.sqrt(label_dim), (out_dim, label_dim)).astype(np.float32)
            feedback = phase.normalize_rows(feedback)
            self.dll_feedback.append(feedback)
            self.dll_targets.append(phase.normalize_rows(self.dll_label_codes @ feedback.T))

        self.feature_dim = hidden_dims[-1]
        if competitive_init == "random":
            self.weights = phase.normalize_rows(
                dll_rng.normal(0.0, 0.01, (vocab_size, self.feature_dim)).astype(np.float32)
            )
        else:
            self.weights = self.dll_targets[-1].copy()

    def base_feature(self, context: np.ndarray) -> np.ndarray:
        return OnlineTraceCompetitivePhaseMemory.feature(self, context)

    def activate(self, z: np.ndarray) -> np.ndarray:
        if self.dll_activation == "linear":
            return z.astype(np.float32, copy=False)
        return np.tanh(z).astype(np.float32)

    def activation_derivative(self, h: np.ndarray) -> np.ndarray:
        if self.dll_activation == "linear":
            return np.ones_like(h, dtype=np.float32)
        return (1.0 - np.square(h)).astype(np.float32)

    def forward_from_base(self, x: np.ndarray) -> np.ndarray:
        state = x.astype(np.float32, copy=False)
        for weights, bias in zip(self.dll_weights, self.dll_biases):
            hidden = self.activate(weights @ state + bias)
            state = phase.normalize_vector(hidden)
        return state

    def feature(self, context: np.ndarray) -> np.ndarray:
        return self.forward_from_base(self.base_feature(context))

    def update_dll_layers(self, base_feature: np.ndarray, target: int) -> None:
        state = base_feature.astype(np.float32, copy=False)
        for idx, (weights, bias, targets) in enumerate(
            zip(self.dll_weights, self.dll_biases, self.dll_targets)
        ):
            hidden = self.activate(weights @ state + bias)
            local_target = targets[int(target)]
            delta = (local_target - hidden) * self.activation_derivative(hidden)
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > self.dll_delta_clip:
                delta = delta * (self.dll_delta_clip / (delta_norm + 1e-8))
            self.dll_weights[idx] = weights + self.dll_lr * np.outer(delta, state).astype(np.float32)
            if self.dll_bias_lr > 0.0:
                self.dll_biases[idx] = bias + self.dll_bias_lr * delta.astype(np.float32)
            if self.dll_row_normalize:
                self.dll_weights[idx] = phase.normalize_rows(self.dll_weights[idx])
            state = phase.normalize_vector(hidden)

    def scores(self, context: np.ndarray) -> np.ndarray:
        feature = self.feature(context)
        return (self.score_scale * (self.weights @ feature) + self.bias_weight * self.output_bias).astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        target = int(target)
        self.branch_model.update_context(context, target)
        self.output_bias = self.branch_model.output_bias.copy()
        base = self.base_feature(context)
        self.update_dll_layers(base, target)
        feature = self.forward_from_base(base)
        scores = self.weights @ feature
        self.weights[target] = phase.normalize_vector(self.weights[target] + self.lr * feature)
        if self.neg_k <= 0:
            return
        scores[target] = -1e9
        k = min(self.neg_k, self.vocab_size - 1)
        wrongs = np.argpartition(scores, -k)[-k:]
        target_score = float(self.weights[target] @ feature)
        for wrong in wrongs:
            if float(scores[wrong]) + self.margin > target_score:
                self.weights[int(wrong)] = phase.normalize_vector(self.weights[int(wrong)] - (self.lr / k) * feature)

    def state_bytes(self) -> int:
        arrays = [self.dll_label_codes]
        arrays.extend(self.dll_weights)
        arrays.extend(self.dll_biases)
        arrays.extend(self.dll_feedback)
        arrays.extend(self.dll_targets)
        return int(super().state_bytes() + sum(array.nbytes for array in arrays))


class OnlineNoPropLocalDenoisingMemory(OnlineTraceCompetitivePhaseMemory):
    """
    NoProp-style decoupled local denoising stack for online token learning.

    During an update, each layer gets its own noisy target code
    z_l = sqrt(alpha_l) * target_l + sqrt(1-alpha_l) * noise_l(local_input).
    The input map learns local_input -> z_l, and the denoiser learns z_l ->
    target_l.  Deeper layers use the previous layer's clean target code during
    training, so their local updates do not require a forward chain or any
    cross-layer gradient.  Inference uses the standard feed-forward chain.
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        noprop_hidden_dims: list[int],
        noprop_label_dim: int,
        noprop_alpha_start: float,
        noprop_alpha_end: float,
        noprop_lr: float,
        noprop_denoise_lr: float,
        noprop_bias_lr: float,
        noprop_delta_clip: float,
        noprop_activation: str,
        noprop_row_normalize: bool,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            trace_order,
            trace_dim,
            trace_decay,
            trace_weight,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
        )
        hidden_dims = [max(int(dim), 1) for dim in noprop_hidden_dims]
        if not hidden_dims:
            raise ValueError("--noprop-hidden-dims must contain at least one layer width")
        self.noprop_lr = max(float(noprop_lr), 0.0)
        self.noprop_denoise_lr = max(float(noprop_denoise_lr), 0.0)
        self.noprop_bias_lr = max(float(noprop_bias_lr), 0.0)
        self.noprop_delta_clip = max(float(noprop_delta_clip), 1e-6)
        self.noprop_activation = str(noprop_activation)
        if self.noprop_activation not in {"tanh", "linear"}:
            raise ValueError(f"unknown NoProp activation: {self.noprop_activation}")
        self.noprop_row_normalize = bool(noprop_row_normalize)
        self.base_feature_dim = int(self.feature_dim)
        self.noprop_hidden_dims = hidden_dims

        noprop_rng = np.random.default_rng(seed + 23874491)
        label_dim = max(int(noprop_label_dim), 1)
        self.noprop_label_codes = phase.normalize_rows(
            noprop_rng.normal(0.0, 1.0, (vocab_size, label_dim)).astype(np.float32)
        )
        dims = [self.base_feature_dim] + hidden_dims
        layer_count = len(hidden_dims)
        if layer_count == 1:
            alphas = [float(noprop_alpha_start)]
        else:
            alphas = np.linspace(float(noprop_alpha_start), float(noprop_alpha_end), layer_count).tolist()
        self.noprop_alphas = np.clip(np.asarray(alphas, dtype=np.float32), 0.0, 1.0)

        self.noprop_targets: list[np.ndarray] = []
        self.noprop_feedback: list[np.ndarray] = []
        self.noprop_input_weights: list[np.ndarray] = []
        self.noprop_input_biases: list[np.ndarray] = []
        self.noprop_denoise_weights: list[np.ndarray] = []
        self.noprop_denoise_biases: list[np.ndarray] = []
        self.noprop_noise_projections: list[np.ndarray] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            feedback = noprop_rng.normal(0.0, 1.0 / math.sqrt(label_dim), (out_dim, label_dim)).astype(np.float32)
            feedback = phase.normalize_rows(feedback)
            self.noprop_feedback.append(feedback)
            self.noprop_targets.append(phase.normalize_rows(self.noprop_label_codes @ feedback.T))

            input_scale = 1.0 / math.sqrt(max(in_dim, 1))
            input_weights = noprop_rng.normal(0.0, input_scale, (out_dim, in_dim)).astype(np.float32)
            self.noprop_input_weights.append(
                phase.normalize_rows(input_weights) if self.noprop_row_normalize else input_weights
            )
            self.noprop_input_biases.append(np.zeros(out_dim, dtype=np.float32))

            denoise_weights = np.eye(out_dim, dtype=np.float32)
            denoise_weights += noprop_rng.normal(0.0, 0.01, (out_dim, out_dim)).astype(np.float32)
            self.noprop_denoise_weights.append(
                phase.normalize_rows(denoise_weights) if self.noprop_row_normalize else denoise_weights
            )
            self.noprop_denoise_biases.append(np.zeros(out_dim, dtype=np.float32))

            noise_projection = noprop_rng.normal(0.0, input_scale, (out_dim, in_dim)).astype(np.float32)
            self.noprop_noise_projections.append(phase.normalize_rows(noise_projection))

        self.feature_dim = hidden_dims[-1]
        if competitive_init == "random":
            self.weights = phase.normalize_rows(
                noprop_rng.normal(0.0, 0.01, (vocab_size, self.feature_dim)).astype(np.float32)
            )
        else:
            self.weights = self.noprop_targets[-1].copy()

    def base_feature(self, context: np.ndarray) -> np.ndarray:
        return OnlineTraceCompetitivePhaseMemory.feature(self, context)

    def activate(self, z: np.ndarray) -> np.ndarray:
        if self.noprop_activation == "linear":
            return z.astype(np.float32, copy=False)
        return np.tanh(z).astype(np.float32)

    def activation_derivative(self, h: np.ndarray) -> np.ndarray:
        if self.noprop_activation == "linear":
            return np.ones_like(h, dtype=np.float32)
        return (1.0 - np.square(h)).astype(np.float32)

    def local_update(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        local_input: np.ndarray,
        local_target: np.ndarray,
        lr: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden = self.activate(weights @ local_input + bias)
        delta = (local_target - hidden) * self.activation_derivative(hidden)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > self.noprop_delta_clip:
            delta = delta * (self.noprop_delta_clip / (delta_norm + 1e-8))
        new_weights = weights + lr * np.outer(delta, local_input).astype(np.float32)
        if self.noprop_row_normalize:
            new_weights = phase.normalize_rows(new_weights)
        new_bias = bias
        if self.noprop_bias_lr > 0.0:
            new_bias = bias + self.noprop_bias_lr * delta.astype(np.float32)
        return new_weights, new_bias, hidden

    def noisy_target(self, layer_idx: int, clean: np.ndarray, local_input: np.ndarray) -> np.ndarray:
        alpha = float(self.noprop_alphas[layer_idx])
        noise = phase.normalize_vector(self.noprop_noise_projections[layer_idx] @ local_input)
        noisy = math.sqrt(alpha) * clean + math.sqrt(max(1.0 - alpha, 0.0)) * noise
        return phase.normalize_vector(noisy.astype(np.float32))

    def forward_from_base(self, x: np.ndarray) -> np.ndarray:
        state = x.astype(np.float32, copy=False)
        for input_weights, input_bias, denoise_weights, denoise_bias in zip(
            self.noprop_input_weights,
            self.noprop_input_biases,
            self.noprop_denoise_weights,
            self.noprop_denoise_biases,
        ):
            noisy_state = phase.normalize_vector(self.activate(input_weights @ state + input_bias))
            clean_state = self.activate(denoise_weights @ noisy_state + denoise_bias)
            state = phase.normalize_vector(clean_state)
        return state

    def feature(self, context: np.ndarray) -> np.ndarray:
        return self.forward_from_base(self.base_feature(context))

    def update_noprop_layers(self, base_feature: np.ndarray, target: int) -> None:
        for idx in range(len(self.noprop_targets)):
            local_input = base_feature if idx == 0 else self.noprop_targets[idx - 1][int(target)]
            clean = self.noprop_targets[idx][int(target)]
            noisy = self.noisy_target(idx, clean, local_input)
            input_weights, input_bias, _ = self.local_update(
                self.noprop_input_weights[idx],
                self.noprop_input_biases[idx],
                local_input,
                noisy,
                self.noprop_lr,
            )
            denoise_weights, denoise_bias, _ = self.local_update(
                self.noprop_denoise_weights[idx],
                self.noprop_denoise_biases[idx],
                noisy,
                clean,
                self.noprop_denoise_lr,
            )
            self.noprop_input_weights[idx] = input_weights
            self.noprop_input_biases[idx] = input_bias
            self.noprop_denoise_weights[idx] = denoise_weights
            self.noprop_denoise_biases[idx] = denoise_bias

    def scores(self, context: np.ndarray) -> np.ndarray:
        feature = self.feature(context)
        return (self.score_scale * (self.weights @ feature) + self.bias_weight * self.output_bias).astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        target = int(target)
        self.branch_model.update_context(context, target)
        self.output_bias = self.branch_model.output_bias.copy()
        base = self.base_feature(context)
        self.update_noprop_layers(base, target)
        feature = self.forward_from_base(base)
        scores = self.weights @ feature
        self.weights[target] = phase.normalize_vector(self.weights[target] + self.lr * feature)
        if self.neg_k <= 0:
            return
        scores[target] = -1e9
        k = min(self.neg_k, self.vocab_size - 1)
        wrongs = np.argpartition(scores, -k)[-k:]
        target_score = float(self.weights[target] @ feature)
        for wrong in wrongs:
            wrong = int(wrong)
            if float(scores[wrong]) + self.margin > target_score:
                self.weights[wrong] = phase.normalize_vector(self.weights[wrong] - (self.lr / k) * feature)

    def state_bytes(self) -> int:
        arrays = [self.noprop_label_codes, self.noprop_alphas]
        arrays.extend(self.noprop_targets)
        arrays.extend(self.noprop_feedback)
        arrays.extend(self.noprop_input_weights)
        arrays.extend(self.noprop_input_biases)
        arrays.extend(self.noprop_denoise_weights)
        arrays.extend(self.noprop_denoise_biases)
        arrays.extend(self.noprop_noise_projections)
        return int(super().state_bytes() + sum(array.nbytes for array in arrays))


class OnlineUnifiedNoPropCalibrationMemory(OnlineNoPropLocalDenoisingMemory):
    """
    U001 unified NoProp core with local calibration and optional eligibility.

    This folds the R096-style calibration and readout gain into the NoProp
    learner itself, so score computation and local target/wrong updates live in
    one model core instead of an external wrapper stack.  The optional
    eligibility pressure is a same-readout, finite-window correction; it is not
    a separate answer table or task-specific branch.
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        noprop_hidden_dims: list[int],
        noprop_label_dim: int,
        noprop_alpha_start: float,
        noprop_alpha_end: float,
        noprop_lr: float,
        noprop_denoise_lr: float,
        noprop_bias_lr: float,
        noprop_delta_clip: float,
        noprop_activation: str,
        noprop_row_normalize: bool,
        inhibition_strength: float,
        inhibition_decay: float,
        inhibition_lr: float,
        inhibition_disinhibit_lr: float,
        inhibition_top_k: int,
        inhibition_margin: float,
        inhibition_max_weight: float,
        calibration_strength: float,
        calibration_lr: float,
        calibration_clip: float,
        calibration_dim: int,
        calibration_gate_decay: float,
        calibration_threshold: float,
        readout_gain: float,
        eligibility_order: int,
        eligibility_decay: float,
        eligibility_score_weight: float,
        eligibility_top_k: int,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            trace_order,
            trace_dim,
            trace_decay,
            trace_weight,
            noprop_hidden_dims,
            noprop_label_dim,
            noprop_alpha_start,
            noprop_alpha_end,
            noprop_lr,
            noprop_denoise_lr,
            noprop_bias_lr,
            noprop_delta_clip,
            noprop_activation,
            noprop_row_normalize,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
        )
        self.unified_inhibition_strength = float(inhibition_strength)
        self.unified_inhibition_decay = float(np.clip(inhibition_decay, 0.0, 0.999))
        self.unified_inhibition_lr = max(float(inhibition_lr), 0.0)
        self.unified_inhibition_disinhibit_lr = max(float(inhibition_disinhibit_lr), 0.0)
        self.unified_inhibition_top_k = max(int(inhibition_top_k), 0)
        self.unified_inhibition_margin = float(inhibition_margin)
        self.unified_inhibition_max_weight = max(float(inhibition_max_weight), 0.0)
        self.unified_calibration_strength = max(float(calibration_strength), 0.0)
        self.unified_calibration_lr = max(float(calibration_lr), 0.0)
        self.unified_calibration_clip = max(float(calibration_clip), 1e-6)
        self.unified_calibration_gate_decay = float(np.clip(calibration_gate_decay, 0.0, 0.999))
        self.unified_calibration_threshold = float(calibration_threshold)
        self.unified_readout_gain = max(float(readout_gain), 1e-6)
        self.unified_eligibility_order = max(int(eligibility_order), 1)
        self.unified_eligibility_decay = float(np.clip(eligibility_decay, 0.0, 0.999))
        self.unified_eligibility_score_weight = float(eligibility_score_weight)
        self.unified_eligibility_top_k = max(int(eligibility_top_k), 0)
        self.max_order = max(int(self.max_order), self.unified_eligibility_order)
        self.unified_calibration_dim = max(int(calibration_dim), 1)
        unified_rng = np.random.default_rng(seed + 32452843)
        self.unified_calibration_codes = phase.normalize_rows(
            unified_rng.normal(
                0.0,
                1.0,
                (self.max_order, self.vocab_size, self.unified_calibration_dim),
            )
            .astype(np.float32)
            .reshape(self.max_order * self.vocab_size, self.unified_calibration_dim)
        ).reshape(self.max_order, self.vocab_size, self.unified_calibration_dim)
        self.unified_calibration = np.zeros(
            (self.vocab_size, self.unified_calibration_dim),
            dtype=np.float32,
        )
        self.unified_calibration_gate = np.zeros(self.unified_calibration_dim, dtype=np.float32)
        self.unified_activity = np.zeros(self.vocab_size, dtype=np.float32)
        self.unified_inhibition = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)

    def reset_dynamic_state(self) -> None:
        self.unified_calibration_gate.fill(0.0)
        self.unified_activity.fill(0.0)

    def observe_output(self, token: int) -> None:
        self.unified_activity *= self.unified_inhibition_decay
        self.unified_activity[int(token)] += 1.0

    def observe_prompt(self, token: int) -> None:
        self.observe_output(token)

    def observe_prediction(self, token: int) -> None:
        self.observe_output(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        self.observe_gate(context)
        self.observe_output(target)

    def context_gate(self, context: np.ndarray) -> np.ndarray:
        state = np.zeros(self.unified_calibration_dim, dtype=np.float32)
        clipped = context[-self.max_order :]
        offset = self.max_order - len(clipped)
        for pos, token in enumerate(clipped):
            state += self.unified_calibration_codes[offset + pos, int(token)]
        gate = phase.normalize_vector(state)
        if self.unified_calibration_threshold > 0.0:
            active = np.abs(gate) >= self.unified_calibration_threshold
            if np.any(active):
                gate = np.where(active, gate, 0.0).astype(np.float32)
                gate = phase.normalize_vector(gate)
        return gate

    def observe_gate(self, context: np.ndarray) -> np.ndarray:
        gate = self.context_gate(context)
        self.unified_calibration_gate = phase.normalize_vector(
            self.unified_calibration_gate_decay * self.unified_calibration_gate + gate
        )
        return self.unified_calibration_gate

    def effective_gate(self, context: np.ndarray) -> np.ndarray:
        gate = self.context_gate(context)
        if self.unified_calibration_gate_decay <= 0.0 or not np.any(self.unified_calibration_gate):
            return gate
        return phase.normalize_vector(gate + self.unified_calibration_gate_decay * self.unified_calibration_gate)

    def top_k_mask(self, scores: np.ndarray, k: int) -> np.ndarray:
        mask = np.zeros(scores.size, dtype=bool)
        if k <= 0:
            mask.fill(True)
            return mask
        count = min(int(k), scores.size)
        if count >= scores.size:
            mask.fill(True)
            return mask
        candidates = np.argpartition(scores.astype(np.float32, copy=False), -count)[-count:]
        mask[candidates] = True
        return mask

    def base_scores_from_feature(self, feature: np.ndarray) -> np.ndarray:
        return (self.score_scale * (self.weights @ feature) + self.bias_weight * self.output_bias).astype(np.float32)

    def inhibited_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        if self.unified_inhibition_strength == 0.0 or not np.any(self.unified_activity):
            return raw_scores.astype(np.float32, copy=False)
        penalty = self.unified_inhibition @ self.unified_activity
        return (raw_scores.astype(np.float32, copy=False) - self.unified_inhibition_strength * penalty).astype(np.float32)

    def calibrated_scores(self, base_scores: np.ndarray, gate: np.ndarray) -> np.ndarray:
        signal = self.unified_calibration @ gate
        return (
            base_scores.astype(np.float32, copy=False)
            + self.unified_calibration_strength * signal
        ).astype(np.float32)

    def eligibility_feature(self, context: np.ndarray, current_feature: np.ndarray) -> np.ndarray:
        if self.unified_eligibility_score_weight == 0.0:
            return current_feature
        state = np.zeros_like(current_feature, dtype=np.float32)
        start = max(1, len(context) - self.unified_eligibility_order + 1)
        for end in range(start, len(context) + 1):
            feature = current_feature if end == len(context) else self.feature(context[:end])
            state = self.unified_eligibility_decay * state + feature
        return phase.normalize_vector(state)

    def eligibility_signal(
        self,
        context: np.ndarray,
        current_feature: np.ndarray,
        base_scores: np.ndarray,
    ) -> np.ndarray:
        if self.unified_eligibility_score_weight == 0.0:
            return np.zeros_like(base_scores, dtype=np.float32)
        eligibility = self.eligibility_feature(context, current_feature)
        signal = self.score_scale * (self.weights @ eligibility - self.weights @ current_feature)
        if self.unified_eligibility_top_k > 0:
            mask = self.top_k_mask(base_scores, self.unified_eligibility_top_k)
            limited = np.zeros_like(signal, dtype=np.float32)
            limited[mask] = signal[mask]
            signal = limited
        return signal.astype(np.float32)

    def combined_scores_from_feature(
        self,
        context: np.ndarray,
        feature: np.ndarray,
        gate: np.ndarray,
        apply_gain: bool,
    ) -> np.ndarray:
        raw_scores = self.base_scores_from_feature(feature)
        base_scores = self.inhibited_scores(raw_scores)
        scores = self.calibrated_scores(base_scores, gate)
        if self.unified_eligibility_score_weight != 0.0:
            scores = (
                scores
                + self.unified_eligibility_score_weight * self.eligibility_signal(context, feature, base_scores)
            ).astype(np.float32)
        if apply_gain:
            scores = (self.unified_readout_gain * scores).astype(np.float32)
        return scores

    def scores(self, context: np.ndarray) -> np.ndarray:
        feature = self.feature(context)
        gate = self.effective_gate(context)
        return self.combined_scores_from_feature(context, feature, gate, apply_gain=True)

    def learn_unified_inhibition(self, scores: np.ndarray, target: int) -> None:
        if (
            self.unified_inhibition_lr <= 0.0
            or self.unified_inhibition_top_k <= 0
            or not np.any(self.unified_activity)
        ):
            return
        target = int(target)
        adjusted = scores.astype(np.float32, copy=True)
        target_score = float(adjusted[target])
        adjusted[target] = -np.inf
        k = min(self.unified_inhibition_top_k, self.vocab_size - 1)
        if k <= 0:
            return
        wrongs = np.argpartition(adjusted, -k)[-k:]
        active_delta = self.unified_inhibition_lr * self.unified_activity
        for wrong in wrongs:
            wrong = int(wrong)
            if float(adjusted[wrong]) + self.unified_inhibition_margin <= target_score:
                continue
            self.unified_inhibition[wrong] = np.minimum(
                self.unified_inhibition_max_weight,
                self.unified_inhibition[wrong] + active_delta,
            )
        if self.unified_inhibition_disinhibit_lr > 0.0:
            self.unified_inhibition[target] = np.maximum(
                0.0,
                self.unified_inhibition[target] - self.unified_inhibition_disinhibit_lr * self.unified_activity,
            )

    def learn_unified_calibration(self, scores: np.ndarray, target: int, gate: np.ndarray) -> None:
        if self.unified_calibration_lr <= 0.0:
            return
        target = int(target)
        pred = int(np.argmax(scores))
        if pred == target:
            return
        delta = self.unified_calibration_lr * gate
        self.unified_calibration[target] = np.clip(
            self.unified_calibration[target] + delta,
            -self.unified_calibration_clip,
            self.unified_calibration_clip,
        )
        self.unified_calibration[pred] = np.clip(
            self.unified_calibration[pred] - delta,
            -self.unified_calibration_clip,
            self.unified_calibration_clip,
        )

    def update(self, context: np.ndarray, target: int) -> None:
        target = int(target)
        feature = self.feature(context)
        gate = self.effective_gate(context)
        raw_scores = self.base_scores_from_feature(feature)
        inhibited_scores = self.inhibited_scores(raw_scores)
        pre_scores = self.calibrated_scores(inhibited_scores, gate)
        if self.unified_eligibility_score_weight != 0.0:
            pre_scores = (
                pre_scores
                + self.unified_eligibility_score_weight * self.eligibility_signal(context, feature, inhibited_scores)
            ).astype(np.float32)
        self.learn_unified_calibration(pre_scores, target, gate)
        self.learn_unified_inhibition(inhibited_scores, target)
        super().update(context, target)
        self.observe_gate(context)
        self.observe_output(target)

    def state_bytes(self) -> int:
        return int(
            super().state_bytes()
            + self.unified_calibration_codes.nbytes
            + self.unified_calibration.nbytes
            + self.unified_calibration_gate.nbytes
            + self.unified_activity.nbytes
            + self.unified_inhibition.nbytes
        )


class OnlineTraceHebbianKVCompetitivePhaseMemory(OnlineTraceCompetitivePhaseMemory):
    """
    Trace/phase learner with a pure Hebbian key-value associative branch.

    The KV branch stores a low-rank compressed history, not raw tokens:
    M <- (1 - decay) M + lr * value[target] outer key(context).
    At prediction time M @ key(context) is concatenated as another dendritic
    segment for the local WTA readout.  All writes are local rank-1 plasticity.
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        kv_order: int,
        kv_dim: int,
        kv_trace_decay: float,
        kv_weight: float,
        kv_score_weight: float,
        kv_gate_mode: str,
        kv_gate_base_margin: float,
        kv_gate_kv_margin: float,
        kv_gate_min_norm: float,
        kv_lr: float,
        kv_decay: float,
        kv_clip: float,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            trace_order,
            trace_dim,
            trace_decay,
            trace_weight,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
        )
        self.kv_order = max(int(kv_order), 1)
        self.max_order = max(self.max_order, self.kv_order)
        self.kv_dim = max(int(kv_dim), 1)
        self.kv_trace_decay = float(np.clip(kv_trace_decay, 0.0, 0.999))
        self.kv_weight = float(kv_weight)
        self.kv_score_weight = float(kv_score_weight)
        self.kv_gate_mode = str(kv_gate_mode)
        if self.kv_gate_mode not in {"none", "norm", "base_low_margin", "kv_margin", "base_and_kv", "base_or_kv"}:
            raise ValueError(f"unknown KV gate mode: {self.kv_gate_mode}")
        self.kv_gate_base_margin = max(float(kv_gate_base_margin), 0.0)
        self.kv_gate_kv_margin = max(float(kv_gate_kv_margin), 0.0)
        self.kv_gate_min_norm = max(float(kv_gate_min_norm), 0.0)
        self.kv_lr = max(float(kv_lr), 0.0)
        self.kv_decay = float(np.clip(kv_decay, 0.0, 0.999))
        self.kv_clip = max(float(kv_clip), 1e-6)
        kv_rng = np.random.default_rng(seed + 86028121)
        self.kv_key_codes = phase.normalize_rows(
            kv_rng.normal(0.0, 1.0, (vocab_size, self.kv_dim)).astype(np.float32)
        )
        self.kv_value_codes = phase.normalize_rows(
            kv_rng.normal(0.0, 1.0, (vocab_size, self.kv_dim)).astype(np.float32)
        )
        self.kv_matrix = np.zeros((self.kv_dim, self.kv_dim), dtype=np.float32)
        self.kv_writes = 0
        old_weights = self.weights
        self.feature_dim += self.kv_dim
        kv_zeros = np.zeros((vocab_size, self.kv_dim), dtype=np.float32)
        self.weights = phase.normalize_rows(np.concatenate([old_weights, kv_zeros], axis=1))

    def kv_key_feature(self, context: np.ndarray) -> np.ndarray:
        state = np.zeros(self.kv_dim, dtype=np.float32)
        for token in context[-self.kv_order :]:
            state = self.kv_trace_decay * state + self.kv_key_codes[int(token)]
        return phase.normalize_vector(state)

    def kv_raw_recall(self, context: np.ndarray) -> np.ndarray:
        query = self.kv_key_feature(context)
        return (self.kv_matrix @ query).astype(np.float32)

    def kv_recall_with_norm(self, context: np.ndarray) -> tuple[np.ndarray, float]:
        raw = self.kv_raw_recall(context)
        norm = float(np.linalg.norm(raw))
        return phase.normalize_vector(raw), norm

    def kv_recall(self, context: np.ndarray) -> np.ndarray:
        recall, _norm = self.kv_recall_with_norm(context)
        return recall

    def kv_feature(self, context: np.ndarray) -> np.ndarray:
        return (self.kv_weight * self.kv_recall(context)).astype(np.float32)

    @staticmethod
    def score_margin(scores: np.ndarray) -> float:
        if scores.size < 2:
            return 0.0
        top2 = np.partition(scores.astype(np.float32, copy=False), -2)[-2:]
        return float(np.max(top2) - np.min(top2))

    def kv_gate(self, base_scores: np.ndarray, kv_anchor_scores: np.ndarray, recall_norm: float) -> float:
        if self.kv_score_weight == 0.0:
            return 0.0
        norm_ok = recall_norm >= self.kv_gate_min_norm
        if self.kv_gate_mode == "none":
            return 1.0 if norm_ok else 0.0
        if self.kv_gate_mode == "norm":
            return 1.0 if norm_ok else 0.0
        base_ok = self.score_margin(base_scores) <= self.kv_gate_base_margin
        kv_ok = self.score_margin(kv_anchor_scores) >= self.kv_gate_kv_margin
        if self.kv_gate_mode == "base_low_margin":
            ok = base_ok
        elif self.kv_gate_mode == "kv_margin":
            ok = kv_ok
        elif self.kv_gate_mode == "base_and_kv":
            ok = base_ok and kv_ok
        elif self.kv_gate_mode == "base_or_kv":
            ok = base_ok or kv_ok
        else:
            ok = True
        return 1.0 if (norm_ok and ok) else 0.0

    def kv_scores(self, context: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        if self.kv_score_weight == 0.0:
            return np.zeros(self.vocab_size, dtype=np.float32)
        recall, recall_norm = self.kv_recall_with_norm(context)
        anchor_scores = (self.kv_value_codes @ recall).astype(np.float32)
        gate = self.kv_gate(base_scores, anchor_scores, recall_norm)
        return (gate * self.kv_score_weight * anchor_scores).astype(np.float32)

    def feature(self, context: np.ndarray) -> np.ndarray:
        branch_features = [
            branch.feature(context[-order:])
            for order, branch in zip(self.branch_model.branch_orders, self.branch_model.branches)
        ]
        trace = self.trace_weight * self.trace_feature(context)
        kv = self.kv_feature(context)
        return phase.normalize_vector(np.concatenate(branch_features + [trace, kv]).astype(np.float32))

    def update_kv(self, context: np.ndarray, target: int) -> None:
        if self.kv_lr <= 0.0:
            return
        key = self.kv_key_feature(context)
        value = self.kv_value_codes[int(target)]
        self.kv_matrix *= 1.0 - self.kv_decay
        self.kv_matrix += self.kv_lr * np.outer(value, key).astype(np.float32)
        np.clip(self.kv_matrix, -self.kv_clip, self.kv_clip, out=self.kv_matrix)
        self.kv_writes += 1

    def scores(self, context: np.ndarray) -> np.ndarray:
        feature = self.feature(context)
        scores = self.score_scale * (self.weights @ feature) + self.bias_weight * self.output_bias
        scores = scores + self.kv_scores(context, scores)
        return scores.astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        self.branch_model.update_context(context, target)
        self.output_bias = self.branch_model.output_bias.copy()
        self.update_kv(context, int(target))
        feature = self.feature(context)
        target = int(target)
        scores = self.weights @ feature
        self.weights[target] = phase.normalize_vector(self.weights[target] + self.lr * feature)
        if self.neg_k <= 0:
            return
        scores[target] = -1e9
        k = min(self.neg_k, self.vocab_size - 1)
        wrongs = np.argpartition(scores, -k)[-k:]
        target_score = float(self.weights[target] @ feature)
        for wrong in wrongs:
            if float(scores[wrong]) + self.margin > target_score:
                self.weights[wrong] = phase.normalize_vector(self.weights[wrong] - (self.lr / k) * feature)

    def state_bytes(self) -> int:
        return int(
            super().state_bytes()
            + self.kv_key_codes.nbytes
            + self.kv_value_codes.nbytes
            + self.kv_matrix.nbytes
        )


class OnlineTraceHebbianKVApicalGatedCompetitivePhaseMemory(OnlineTraceHebbianKVCompetitivePhaseMemory):
    """Hebbian KV branch plus branch-local apical prediction-error gates."""

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        kv_order: int,
        kv_dim: int,
        kv_trace_decay: float,
        kv_weight: float,
        kv_score_weight: float,
        kv_gate_mode: str,
        kv_gate_base_margin: float,
        kv_gate_kv_margin: float,
        kv_gate_min_norm: float,
        kv_lr: float,
        kv_decay: float,
        kv_clip: float,
        apical_decay: float,
        apical_strength: float,
        apical_margin: float,
        apical_min_gate: float,
        apical_max_gate: float,
        apical_error_clip: float,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
        apical_error_mode: str = "segment_margin",
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            trace_order,
            trace_dim,
            trace_decay,
            trace_weight,
            kv_order,
            kv_dim,
            kv_trace_decay,
            kv_weight,
            kv_score_weight,
            kv_gate_mode,
            kv_gate_base_margin,
            kv_gate_kv_margin,
            kv_gate_min_norm,
            kv_lr,
            kv_decay,
            kv_clip,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
        )
        self.apical_decay = float(np.clip(apical_decay, 0.0, 0.999))
        self.apical_strength = max(float(apical_strength), 0.0)
        self.apical_margin = float(apical_margin)
        self.apical_min_gate = max(float(apical_min_gate), 0.0)
        self.apical_max_gate = max(float(apical_max_gate), self.apical_min_gate + 1e-6)
        self.apical_error_clip = max(float(apical_error_clip), 1e-6)
        self.apical_error_mode = str(apical_error_mode)
        if self.apical_error_mode not in {"segment_margin", "global_margin", "random_feedback", "fixed_random"}:
            raise ValueError(f"unknown apical error mode: {self.apical_error_mode}")
        self.segment_slices: list[slice] = []
        offset = 0
        for branch in self.branch_model.branches:
            width = 2 * branch.cfg.complex_dim
            self.segment_slices.append(slice(offset, offset + width))
            offset += width
        self.segment_slices.append(slice(offset, offset + self.trace_codes.shape[1]))
        offset += self.trace_codes.shape[1]
        self.segment_slices.append(slice(offset, offset + self.kv_dim))
        self.apical_error_trace = np.zeros(len(self.segment_slices), dtype=np.float32)
        apical_rng = np.random.default_rng(seed + 67867967)
        feedback = np.abs(apical_rng.normal(0.0, 1.0, len(self.segment_slices)).astype(np.float32))
        self.apical_feedback = feedback / max(float(np.mean(feedback)), 1e-6)
        fixed_gate = apical_rng.normal(0.0, 1.0, len(self.segment_slices)).astype(np.float32)
        self.apical_fixed_gate = fixed_gate / max(float(np.max(np.abs(fixed_gate))), 1e-6)

    def reset_dynamic_state(self) -> None:
        self.apical_error_trace.fill(0.0)

    def branch_error_gates(self, feature: np.ndarray, scores: np.ndarray, target: int) -> np.ndarray:
        target = int(target)
        adjusted = scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        k = min(max(self.neg_k, 1), self.vocab_size - 1)
        if k <= 0:
            return np.ones(len(self.segment_slices), dtype=np.float32)
        wrongs = np.argpartition(adjusted, -k)[-k:]
        errors = np.zeros(len(self.segment_slices), dtype=np.float32)
        full_margin = float(scores[target] - np.max(scores[wrongs]))
        if self.apical_error_mode == "fixed_random":
            gates = 1.0 + self.apical_strength * self.apical_fixed_gate
            return np.clip(gates, self.apical_min_gate, self.apical_max_gate).astype(np.float32)
        if self.apical_error_mode == "global_margin":
            errors.fill(max(0.0, self.apical_margin - full_margin))
        elif self.apical_error_mode == "random_feedback":
            errors = max(0.0, self.apical_margin - full_margin) * self.apical_feedback
        else:
            for idx, segment_slice in enumerate(self.segment_slices):
                segment = feature[segment_slice]
                target_score = float(self.weights[target, segment_slice] @ segment)
                wrong_score = float(np.max(self.weights[wrongs, segment_slice] @ segment))
                local_margin = target_score - wrong_score
                errors[idx] = max(0.0, self.apical_margin - local_margin)
        self.apical_error_trace = self.apical_decay * self.apical_error_trace + errors
        np.clip(self.apical_error_trace, 0.0, self.apical_error_clip, out=self.apical_error_trace)
        gates = 1.0 + self.apical_strength * self.apical_error_trace
        return np.clip(gates, self.apical_min_gate, self.apical_max_gate).astype(np.float32)

    def apply_segment_gates(self, feature: np.ndarray, gates: np.ndarray) -> np.ndarray:
        gated = feature.astype(np.float32, copy=True)
        for gate, segment_slice in zip(gates, self.segment_slices):
            gated[segment_slice] *= float(gate)
        return phase.normalize_vector(gated)

    def update(self, context: np.ndarray, target: int) -> None:
        target = int(target)
        pre_scores = self.scores(context)
        self.branch_model.update_context(context, target)
        self.output_bias = self.branch_model.output_bias.copy()
        self.update_kv(context, target)
        feature = self.feature(context)
        gates = self.branch_error_gates(feature, pre_scores, target)
        gated_feature = self.apply_segment_gates(feature, gates)
        self.weights[target] = phase.normalize_vector(self.weights[target] + self.lr * gated_feature)
        if self.neg_k <= 0:
            return
        adjusted = pre_scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        k = min(self.neg_k, self.vocab_size - 1)
        wrongs = np.argpartition(adjusted, -k)[-k:]
        target_score = float(pre_scores[target])
        for wrong in wrongs:
            wrong = int(wrong)
            if float(pre_scores[wrong]) + self.margin > target_score:
                self.weights[wrong] = phase.normalize_vector(self.weights[wrong] - (self.lr / k) * gated_feature)

    def state_bytes(self) -> int:
        bytes_used = self.apical_error_trace.nbytes
        if self.apical_error_mode == "random_feedback":
            bytes_used += self.apical_feedback.nbytes
        if self.apical_error_mode == "fixed_random":
            bytes_used += self.apical_fixed_gate.nbytes
        return int(super().state_bytes() + bytes_used)


class OnlinePlasticSSMCompetitivePhaseMemory:
    """
    Phase branches plus a locally plastic recurrent/SSM branch.

    The recurrent branch unfolds over the current context window with a learned
    transition matrix.  Its transition update is target-modulated Hebbian/Oja
    plasticity on local pre/post state activity, not BPTT.  The final recurrent
    state is concatenated into the same local WTA readout used by phase/trace.
    """

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        ssm_order: int,
        ssm_dim: int,
        ssm_decay: float,
        ssm_recurrent_scale: float,
        ssm_weight: float,
        ssm_lr: float,
        ssm_target_mix: float,
        ssm_clip: float,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        self.branch_model = phase.BranchPhaseTokenLearner(vocab_size, cfg, branch_orders, branch_weights)
        self.vocab_size = vocab_size
        self.max_order = max(self.branch_model.max_order, int(ssm_order))
        self.ssm_order = max(int(ssm_order), 1)
        self.ssm_dim = max(int(ssm_dim), 1)
        self.ssm_decay = float(np.clip(ssm_decay, 0.0, 0.999))
        self.ssm_recurrent_scale = float(ssm_recurrent_scale)
        self.ssm_weight = float(ssm_weight)
        self.ssm_lr = max(float(ssm_lr), 0.0)
        self.ssm_target_mix = max(float(ssm_target_mix), 0.0)
        self.ssm_clip = max(float(ssm_clip), 1e-6)
        self.bias_weight = cfg.bias_weight
        self.temperature = cfg.temperature
        self.lr = competitive_lr
        self.neg_k = max(int(competitive_neg_k), 0)
        self.epochs = max(int(competitive_epochs), 1)
        self.score_scale = competitive_score_scale
        self.margin = competitive_margin
        self.init = competitive_init
        self.rng = np.random.default_rng(seed + 32452843)
        self.input_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (vocab_size, self.ssm_dim)).astype(np.float32)
        )
        self.target_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (vocab_size, self.ssm_dim)).astype(np.float32)
        )
        recurrent_init = self.rng.normal(0.0, 1.0 / math.sqrt(self.ssm_dim), (self.ssm_dim, self.ssm_dim)).astype(np.float32)
        self.transition = (0.10 * recurrent_init).astype(np.float32)
        phase_dim = sum(2 * branch.cfg.complex_dim for branch in self.branch_model.branches)
        self.feature_dim = phase_dim + self.ssm_dim
        if competitive_init == "random":
            self.weights = phase.normalize_rows(
                self.rng.normal(0.0, 0.01, (vocab_size, self.feature_dim)).astype(np.float32)
            )
        else:
            branch_weights_init = np.concatenate(
                [branch.prototypes for branch in self.branch_model.branches],
                axis=1,
            ).astype(np.float32)
            ssm_zeros = np.zeros((vocab_size, self.ssm_dim), dtype=np.float32)
            self.weights = phase.normalize_rows(np.concatenate([branch_weights_init, ssm_zeros], axis=1))
        self.output_bias = self.branch_model.output_bias.copy()

    def ssm_states(self, context: np.ndarray) -> list[np.ndarray]:
        state = np.zeros(self.ssm_dim, dtype=np.float32)
        states = [state.copy()]
        for token in context[-self.ssm_order :]:
            recurrent = self.transition @ state
            raw = self.ssm_decay * state + self.ssm_recurrent_scale * recurrent + self.input_codes[int(token)]
            state = phase.normalize_vector(np.tanh(raw).astype(np.float32))
            states.append(state.copy())
        return states

    def ssm_feature(self, context: np.ndarray) -> np.ndarray:
        return self.ssm_states(context)[-1]

    def feature(self, context: np.ndarray) -> np.ndarray:
        branch_features = [
            branch.feature(context[-order:])
            for order, branch in zip(self.branch_model.branch_orders, self.branch_model.branches)
        ]
        ssm = self.ssm_weight * self.ssm_feature(context)
        return phase.normalize_vector(np.concatenate(branch_features + [ssm]).astype(np.float32))

    def scores(self, context: np.ndarray) -> np.ndarray:
        feature = self.feature(context)
        return (self.score_scale * (self.weights @ feature) + self.bias_weight * self.output_bias).astype(np.float32)

    def update_transition(self, context: np.ndarray, target: int) -> None:
        if self.ssm_lr <= 0.0:
            return
        states = self.ssm_states(context)
        target_code = self.target_codes[int(target)]
        delta = np.zeros_like(self.transition)
        for prev, post in zip(states[:-1], states[1:]):
            if not np.any(prev):
                continue
            desired_post = phase.normalize_vector(post + self.ssm_target_mix * target_code)
            pred_post = self.transition @ prev
            local_error = desired_post - pred_post
            delta += np.outer(local_error, prev).astype(np.float32)
        denom = max(len(states) - 1, 1)
        self.transition += (self.ssm_lr / denom) * delta
        np.clip(self.transition, -self.ssm_clip, self.ssm_clip, out=self.transition)

    def update(self, context: np.ndarray, target: int) -> None:
        self.branch_model.update_context(context, target)
        self.output_bias = self.branch_model.output_bias.copy()
        self.update_transition(context, int(target))
        feature = self.feature(context)
        target = int(target)
        scores = self.weights @ feature
        self.weights[target] = phase.normalize_vector(self.weights[target] + self.lr * feature)
        if self.neg_k <= 0:
            return
        scores[target] = -1e9
        k = min(self.neg_k, self.vocab_size - 1)
        wrongs = np.argpartition(scores, -k)[-k:]
        target_score = float(self.weights[target] @ feature)
        for wrong in wrongs:
            if float(scores[wrong]) + self.margin > target_score:
                self.weights[wrong] = phase.normalize_vector(self.weights[wrong] - (self.lr / k) * feature)

    def state_bytes(self) -> int:
        return int(
            self.branch_model.state_bytes()
            + self.input_codes.nbytes
            + self.target_codes.nbytes
            + self.transition.nbytes
            + self.weights.nbytes
            + self.output_bias.nbytes
        )

    def active_contexts(self) -> int:
        return int(sum(np.count_nonzero(branch.prototype_counts) for branch in self.branch_model.branches))


class OnlineTracePlasticSSMCompetitivePhaseMemory(OnlinePlasticSSMCompetitivePhaseMemory):
    """Fixed leaky trace plus locally plastic recurrent/SSM branch."""

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        ssm_order: int,
        ssm_dim: int,
        ssm_decay: float,
        ssm_recurrent_scale: float,
        ssm_weight: float,
        ssm_lr: float,
        ssm_target_mix: float,
        ssm_clip: float,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            ssm_order,
            ssm_dim,
            ssm_decay,
            ssm_recurrent_scale,
            ssm_weight,
            ssm_lr,
            ssm_target_mix,
            ssm_clip,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
        )
        self.max_order = max(self.max_order, int(trace_order))
        self.trace_order = max(int(trace_order), 1)
        self.trace_decay = float(np.clip(trace_decay, 0.0, 0.999))
        self.trace_weight = float(trace_weight)
        self.trace_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (vocab_size, max(int(trace_dim), 1))).astype(np.float32)
        )
        phase_dim = sum(2 * branch.cfg.complex_dim for branch in self.branch_model.branches)
        self.feature_dim = phase_dim + self.trace_codes.shape[1] + self.ssm_dim
        if competitive_init == "random":
            self.weights = phase.normalize_rows(
                self.rng.normal(0.0, 0.01, (vocab_size, self.feature_dim)).astype(np.float32)
            )
        else:
            branch_weights_init = np.concatenate(
                [branch.prototypes for branch in self.branch_model.branches],
                axis=1,
            ).astype(np.float32)
            trace_zeros = np.zeros((vocab_size, self.trace_codes.shape[1]), dtype=np.float32)
            ssm_zeros = np.zeros((vocab_size, self.ssm_dim), dtype=np.float32)
            self.weights = phase.normalize_rows(np.concatenate([branch_weights_init, trace_zeros, ssm_zeros], axis=1))

    def trace_feature(self, context: np.ndarray) -> np.ndarray:
        state = np.zeros(self.trace_codes.shape[1], dtype=np.float32)
        for token in context[-self.trace_order :]:
            state = self.trace_decay * state + self.trace_codes[int(token)]
        return phase.normalize_vector(state)

    def feature(self, context: np.ndarray) -> np.ndarray:
        branch_features = [
            branch.feature(context[-order:])
            for order, branch in zip(self.branch_model.branch_orders, self.branch_model.branches)
        ]
        trace = self.trace_weight * self.trace_feature(context)
        ssm = self.ssm_weight * self.ssm_feature(context)
        return phase.normalize_vector(np.concatenate(branch_features + [trace, ssm]).astype(np.float32))

    def state_bytes(self) -> int:
        return int(
            self.branch_model.state_bytes()
            + self.trace_codes.nbytes
            + self.input_codes.nbytes
            + self.target_codes.nbytes
            + self.transition.nbytes
            + self.weights.nbytes
            + self.output_bias.nbytes
        )


class EligibilitySSMTransitionMixin:
    """Decaying synaptic eligibility trace for target-gated recurrent writes."""

    def init_eligibility(self, decay: float, clip: float) -> None:
        self.ssm_eligibility_decay = float(np.clip(decay, 0.0, 0.999))
        self.ssm_eligibility_clip = max(float(clip), 1e-6)
        self.eligibility = np.zeros_like(self.transition, dtype=np.float32)

    def update_transition(self, context: np.ndarray, target: int) -> None:
        if self.ssm_lr <= 0.0:
            return
        states = self.ssm_states(context)
        target_code = self.target_codes[int(target)]
        delta = np.zeros_like(self.transition)
        for prev, post in zip(states[:-1], states[1:]):
            if not np.any(prev):
                continue
            step_eligibility = np.outer(post, prev).astype(np.float32)
            self.eligibility = self.ssm_eligibility_decay * self.eligibility + step_eligibility
            np.clip(self.eligibility, -self.ssm_eligibility_clip, self.ssm_eligibility_clip, out=self.eligibility)
            desired_post = phase.normalize_vector(post + self.ssm_target_mix * target_code)
            pred_post = self.transition @ prev
            local_error = desired_post - pred_post
            gated_delta = np.outer(local_error, prev).astype(np.float32) * np.abs(self.eligibility)
            delta += gated_delta
        denom = max(len(states) - 1, 1)
        self.transition += (self.ssm_lr / denom) * delta
        np.clip(self.transition, -self.ssm_clip, self.ssm_clip, out=self.transition)

    def state_bytes(self) -> int:
        return int(super().state_bytes() + self.eligibility.nbytes)


class OnlineEligibilitySSMCompetitivePhaseMemory(EligibilitySSMTransitionMixin, OnlinePlasticSSMCompetitivePhaseMemory):
    """Plastic SSM branch with decaying eligibility-gated transition writes."""

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        ssm_order: int,
        ssm_dim: int,
        ssm_decay: float,
        ssm_recurrent_scale: float,
        ssm_weight: float,
        ssm_lr: float,
        ssm_target_mix: float,
        ssm_clip: float,
        ssm_eligibility_decay: float,
        ssm_eligibility_clip: float,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            ssm_order,
            ssm_dim,
            ssm_decay,
            ssm_recurrent_scale,
            ssm_weight,
            ssm_lr,
            ssm_target_mix,
            ssm_clip,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
        )
        self.init_eligibility(ssm_eligibility_decay, ssm_eligibility_clip)


class OnlineTraceEligibilitySSMCompetitivePhaseMemory(
    EligibilitySSMTransitionMixin,
    OnlineTracePlasticSSMCompetitivePhaseMemory,
):
    """Fixed trace plus eligibility-gated plastic SSM branch."""

    def __init__(
        self,
        vocab_size: int,
        cfg: phase.PhaseTokenConfig,
        branch_orders: list[int],
        branch_weights: list[float],
        trace_order: int,
        trace_dim: int,
        trace_decay: float,
        trace_weight: float,
        ssm_order: int,
        ssm_dim: int,
        ssm_decay: float,
        ssm_recurrent_scale: float,
        ssm_weight: float,
        ssm_lr: float,
        ssm_target_mix: float,
        ssm_clip: float,
        ssm_eligibility_decay: float,
        ssm_eligibility_clip: float,
        competitive_lr: float,
        competitive_neg_k: int,
        competitive_epochs: int,
        competitive_score_scale: float,
        competitive_init: str,
        competitive_margin: float,
        seed: int,
    ) -> None:
        super().__init__(
            vocab_size,
            cfg,
            branch_orders,
            branch_weights,
            trace_order,
            trace_dim,
            trace_decay,
            trace_weight,
            ssm_order,
            ssm_dim,
            ssm_decay,
            ssm_recurrent_scale,
            ssm_weight,
            ssm_lr,
            ssm_target_mix,
            ssm_clip,
            competitive_lr,
            competitive_neg_k,
            competitive_epochs,
            competitive_score_scale,
            competitive_init,
            competitive_margin,
            seed,
        )
        self.init_eligibility(ssm_eligibility_decay, ssm_eligibility_clip)


class OutputFatigueMemory:
    """
    Local output-neuron fatigue wrapper.

    Recently active output neurons receive a decaying inhibitory offset.  The
    state is updated from observed targets in stream scoring and from predicted
    tokens during generation.  This is dynamic local inhibition, not a learned
    statistical cache.
    """

    def __init__(self, base: Any, strength: float, decay: float) -> None:
        self.base = base
        self.strength = float(strength)
        self.decay = float(np.clip(decay, 0.0, 0.999))
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        self.fatigue = np.zeros(self.vocab_size, dtype=np.float32)

    def reset_dynamic_state(self) -> None:
        self.fatigue.fill(0.0)

    def observe_output(self, token: int) -> None:
        self.fatigue *= self.decay
        self.fatigue[int(token)] += 1.0

    def observe_prompt(self, token: int) -> None:
        self.observe_output(token)

    def observe_prediction(self, token: int) -> None:
        self.observe_output(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)
        self.observe_output(target)

    def scores(self, context: np.ndarray) -> np.ndarray:
        return (self.base.scores(context) - self.strength * self.fatigue).astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        self.base.update(context, target)
        self.observe_output(target)

    def state_bytes(self) -> int:
        return int(self.base.state_bytes() + self.fatigue.nbytes)

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())


class LoopPressureInhibitionMemory:
    """
    Dynamic local inhibition for repeated-output attractors.

    A decaying output activity trace estimates whether the current output stream
    is revisiting the same winner.  The loop pressure is a transient neural
    state used directly in scoring; it is updated from observed targets during
    stream evaluation and from predicted tokens during generation.
    """

    def __init__(
        self,
        base: Any,
        strength: float,
        activity_decay: float,
        pressure_decay: float,
        threshold: float,
        clip: float,
        repeat_gain: float,
        transition_strength: float,
        transition_decay: float,
        transition_threshold: float,
        transition_clip: float,
        transition_gain: float,
    ) -> None:
        self.base = base
        self.strength = max(float(strength), 0.0)
        self.activity_decay = float(np.clip(activity_decay, 0.0, 0.999))
        self.pressure_decay = float(np.clip(pressure_decay, 0.0, 0.999))
        self.threshold = max(float(threshold), 0.0)
        self.clip = max(float(clip), 1e-6)
        self.repeat_gain = max(float(repeat_gain), 0.0)
        self.transition_strength = max(float(transition_strength), 0.0)
        self.transition_decay = float(np.clip(transition_decay, 0.0, 0.999))
        self.transition_threshold = max(float(transition_threshold), 0.0)
        self.transition_clip = max(float(transition_clip), 1e-6)
        self.transition_gain = max(float(transition_gain), 0.0)
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        self.loop_activity = np.zeros(self.vocab_size, dtype=np.float32)
        self.loop_pressure = np.zeros(self.vocab_size, dtype=np.float32)
        self.loop_prev_output = np.zeros(self.vocab_size, dtype=np.float32)
        self.loop_transition_pressure = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()
        self.loop_activity.fill(0.0)
        self.loop_pressure.fill(0.0)
        self.loop_prev_output.fill(0.0)
        self.loop_transition_pressure.fill(0.0)

    def observe_output(self, token: int) -> None:
        token = int(token)
        repeat_signal = float(self.loop_activity[token])
        has_previous = bool(np.any(self.loop_prev_output))
        previous = int(np.argmax(self.loop_prev_output)) if has_previous else -1
        self.loop_activity *= self.activity_decay
        self.loop_pressure *= self.pressure_decay
        self.loop_transition_pressure *= self.transition_decay
        self.loop_activity[token] += 1.0
        self.loop_pressure[token] += self.repeat_gain * (1.0 + repeat_signal)
        if previous >= 0:
            transition_repeat = float(self.loop_transition_pressure[previous, token])
            self.loop_transition_pressure[previous, token] += self.transition_gain * (1.0 + transition_repeat)
        self.loop_prev_output.fill(0.0)
        self.loop_prev_output[token] = 1.0
        np.clip(self.loop_pressure, 0.0, self.clip, out=self.loop_pressure)
        np.clip(self.loop_transition_pressure, 0.0, self.transition_clip, out=self.loop_transition_pressure)

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)
        self.observe_output(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)
        self.observe_output(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)
        self.observe_output(target)

    def scores(self, context: np.ndarray) -> np.ndarray:
        pressure = np.maximum(0.0, self.loop_pressure - self.threshold)
        penalty = self.strength * pressure
        if self.transition_strength > 0.0 and len(context) > 0:
            last = int(context[-1])
            transition_pressure = np.maximum(0.0, self.loop_transition_pressure[last] - self.transition_threshold)
            penalty = penalty + self.transition_strength * transition_pressure
        return (self.base.scores(context) - penalty).astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        self.base.update(context, target)
        self.observe_output(target)

    def state_bytes(self) -> int:
        return int(
            self.base.state_bytes()
            + self.loop_activity.nbytes
            + self.loop_pressure.nbytes
            + self.loop_prev_output.nbytes
            + self.loop_transition_pressure.nbytes
        )

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())

    def state_config_metadata(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "strength": float(self.strength),
            "activity_decay": float(self.activity_decay),
            "pressure_decay": float(self.pressure_decay),
            "threshold": float(self.threshold),
            "clip": float(self.clip),
            "repeat_gain": float(self.repeat_gain),
            "transition_strength": float(self.transition_strength),
            "transition_decay": float(self.transition_decay),
            "transition_threshold": float(self.transition_threshold),
            "transition_clip": float(self.transition_clip),
            "transition_gain": float(self.transition_gain),
        }


class SegmentAttractorInhibitionMemory:
    """
    Segment-level attractor detector over generated/observed output codes.

    Unlike token-level loop pressure, this keeps a compact decaying state over
    random output codes and compares it to older segment states.  When the
    current trajectory revisits an older state, a local inhibitory pressure is
    applied to the recent output trace.  Codes are derived from seed and model
    shape, so checkpoints store only dynamic neural state.
    """

    def __init__(
        self,
        base: Any,
        strength: float,
        state_dim: int,
        state_decay: float,
        trace_decay: float,
        pressure_decay: float,
        threshold: float,
        gain: float,
        clip: float,
        slots: int,
        lag: int,
        stride: int,
        gate_mode: str,
        gate_margin_threshold: float,
        gate_inhibition_threshold: float,
        gate_branch_threshold: float,
        gate_gain: float,
        seed: int,
    ) -> None:
        self.base = base
        self.strength = max(float(strength), 0.0)
        self.state_decay = float(np.clip(state_decay, 0.0, 0.999))
        self.trace_decay = float(np.clip(trace_decay, 0.0, 0.999))
        self.pressure_decay = float(np.clip(pressure_decay, 0.0, 0.999))
        self.threshold = float(np.clip(threshold, -1.0, 1.0))
        self.gain = max(float(gain), 0.0)
        self.clip = max(float(clip), 1e-6)
        self.slots = max(int(slots), 1)
        self.lag = max(int(lag), 0)
        self.stride = max(int(stride), 1)
        self.gate_mode = str(gate_mode)
        if self.gate_mode not in {
            "none",
            "margin",
            "inhibition",
            "branch",
            "margin_or_inhibition",
            "margin_and_inhibition",
            "either",
            "both",
        }:
            raise ValueError(f"unknown segment attractor gate mode: {self.gate_mode}")
        self.gate_margin_threshold = max(float(gate_margin_threshold), 0.0)
        self.gate_inhibition_threshold = max(float(gate_inhibition_threshold), 0.0)
        self.gate_branch_threshold = float(gate_branch_threshold)
        self.gate_gain = max(float(gate_gain), 0.0)
        self.seed_offset = int(seed) + 130092997
        self.derived_state_names = {"segment_attractor_codes"}
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        self.state_dim = max(int(state_dim), 1)
        rng = np.random.default_rng(self.seed_offset)
        self.segment_attractor_codes = phase.normalize_rows(
            rng.normal(0.0, 1.0, (self.vocab_size, self.state_dim)).astype(np.float32)
        )
        self.segment_state = np.zeros(self.state_dim, dtype=np.float32)
        self.segment_slots = np.zeros((self.slots, self.state_dim), dtype=np.float32)
        self.segment_slot_age = np.zeros(self.slots, dtype=np.int32)
        self.segment_slot_filled = np.zeros(self.slots, dtype=np.bool_)
        self.segment_output_trace = np.zeros(self.vocab_size, dtype=np.float32)
        self.segment_attractor_pressure = np.zeros(1, dtype=np.float32)
        self.segment_step = np.zeros(1, dtype=np.int32)
        self.segment_slot_index = np.zeros(1, dtype=np.int32)

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()
        self.segment_state.fill(0.0)
        self.segment_slots.fill(0.0)
        self.segment_slot_age.fill(0)
        self.segment_slot_filled.fill(False)
        self.segment_output_trace.fill(0.0)
        self.segment_attractor_pressure.fill(0.0)
        self.segment_step.fill(0)
        self.segment_slot_index.fill(0)

    def observe_output(self, token: int) -> None:
        token = int(token)
        self.segment_slot_age[self.segment_slot_filled] += 1
        code = self.segment_attractor_codes[token]
        self.segment_state = phase.normalize_vector(self.state_decay * self.segment_state + code)
        self.segment_output_trace *= self.trace_decay
        self.segment_output_trace[token] += 1.0

        valid = self.segment_slot_filled & (self.segment_slot_age >= self.lag)
        pressure_input = 0.0
        if np.any(valid):
            sims = self.segment_slots[valid] @ self.segment_state
            pressure_input = max(0.0, float(np.max(sims)) - self.threshold)
        self.segment_attractor_pressure[0] = min(
            self.clip,
            self.pressure_decay * float(self.segment_attractor_pressure[0]) + self.gain * pressure_input,
        )

        self.segment_step[0] += 1
        if int(self.segment_step[0]) % self.stride == 0:
            idx = int(self.segment_slot_index[0])
            self.segment_slots[idx] = self.segment_state
            self.segment_slot_age[idx] = 0
            self.segment_slot_filled[idx] = True
            self.segment_slot_index[0] = (idx + 1) % self.slots

    def find_inhibition_memory(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "inhibition") and hasattr(obj, "activity"):
                return obj
            obj = getattr(obj, "base", None)
        return None

    def inhibition_pressure(self) -> np.ndarray:
        inhibition_memory = self.find_inhibition_memory()
        if inhibition_memory is None or not np.any(inhibition_memory.activity):
            return np.zeros(self.vocab_size, dtype=np.float32)
        pressure = inhibition_memory.inhibition @ inhibition_memory.activity
        return pressure.astype(np.float32, copy=False)

    def branch_confidence(self, context: np.ndarray, winner: int) -> float:
        if not hasattr(self.base, "branch_supports"):
            return 0.0
        supports = self.base.branch_supports(context)
        if supports.ndim != 2 or supports.shape[0] <= 0:
            return 0.0
        winner_supports = supports[:, int(winner)].astype(np.float32, copy=False)
        return float(np.mean(winner_supports) - np.var(winner_supports))

    def event_gate(self, context: np.ndarray, base_scores: np.ndarray) -> float:
        if self.gate_mode == "none":
            return 1.0
        if base_scores.size <= 1:
            return 0.0
        top2 = np.partition(base_scores.astype(np.float32), -2)[-2:]
        margin = float(top2[-1] - top2[-2])
        winner = int(np.argmax(base_scores))
        signals: list[float] = []
        if self.gate_mode in {"margin", "margin_or_inhibition", "margin_and_inhibition", "either", "both"}:
            denom = max(self.gate_margin_threshold, 1e-6)
            signals.append(max(0.0, self.gate_margin_threshold - margin) / denom)
        if self.gate_mode in {"inhibition", "margin_or_inhibition", "margin_and_inhibition", "either", "both"}:
            pressure = self.inhibition_pressure()
            signals.append(max(0.0, float(pressure[winner]) - self.gate_inhibition_threshold))
        if self.gate_mode in {"branch", "either", "both"}:
            confidence = self.branch_confidence(context, winner)
            signals.append(max(0.0, self.gate_branch_threshold - confidence))
        if not signals:
            return 0.0
        if self.gate_mode in {"both", "margin_and_inhibition"} and any(value <= 0.0 for value in signals):
            return 0.0
        active = max(signals)
        if active <= 0.0:
            return 0.0
        return 1.0 + self.gate_gain * math.tanh(active)

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)
        self.observe_output(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)
        self.observe_output(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)
        self.observe_output(target)

    def scores(self, context: np.ndarray) -> np.ndarray:
        base_scores = self.base.scores(context).astype(np.float32, copy=False)
        trace_max = max(float(np.max(self.segment_output_trace)), 1e-6)
        trace = self.segment_output_trace / trace_max
        pressure = float(self.segment_attractor_pressure[0])
        gate = self.event_gate(context, base_scores)
        penalty = self.strength * pressure * gate * trace
        return (base_scores - penalty).astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        self.base.update(context, target)
        self.observe_output(target)

    def state_bytes(self) -> int:
        return int(
            self.base.state_bytes()
            + self.segment_state.nbytes
            + self.segment_slots.nbytes
            + self.segment_slot_age.nbytes
            + self.segment_slot_filled.nbytes
            + self.segment_output_trace.nbytes
            + self.segment_attractor_pressure.nbytes
            + self.segment_step.nbytes
            + self.segment_slot_index.nbytes
        )

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())

    def state_config_metadata(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "strength": float(self.strength),
            "state_dim": int(self.state_dim),
            "state_decay": float(self.state_decay),
            "trace_decay": float(self.trace_decay),
            "pressure_decay": float(self.pressure_decay),
            "threshold": float(self.threshold),
            "gain": float(self.gain),
            "clip": float(self.clip),
            "slots": int(self.slots),
            "lag": int(self.lag),
            "stride": int(self.stride),
            "gate_mode": self.gate_mode,
            "gate_margin_threshold": float(self.gate_margin_threshold),
            "gate_inhibition_threshold": float(self.gate_inhibition_threshold),
            "gate_branch_threshold": float(self.gate_branch_threshold),
            "gate_gain": float(self.gate_gain),
            "seed_offset": int(self.seed_offset),
            "derived_codes": True,
        }


class LoopEscapeCompetitorMemory:
    """
    Learned loop-escape competitor.

    Segment pressure is used only as a trigger.  When the current trajectory is
    loop-like, a small token-by-branch synaptic matrix adds an escape competitor
    over dendritic branch supports.  Online updates are local target-vs-wrong
    WTA corrections; no raw text, replay, token-probability table, or BP signal
    is stored.
    """

    def __init__(
        self,
        base: Any,
        strength: float,
        lr: float,
        decay: float,
        clip: float,
        support_clip: float,
        top_k: int,
        margin: float,
        gate_mode: str,
        pressure_threshold: float,
        pressure_gain: float,
        margin_threshold: float,
        score_mode: str,
        score_top_k: int,
        update_mode: str,
        learn_candidate_k: int,
    ) -> None:
        self.base = base
        self.strength = max(float(strength), 0.0)
        self.lr = max(float(lr), 0.0)
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.clip = max(float(clip), 1e-6)
        self.support_clip = max(float(support_clip), 1e-6)
        self.top_k = max(int(top_k), 0)
        self.margin = float(margin)
        self.gate_mode = str(gate_mode)
        if self.gate_mode not in {"pressure", "pressure_and_margin", "pressure_or_margin"}:
            raise ValueError(f"unknown loop escape gate mode: {self.gate_mode}")
        self.pressure_threshold = max(float(pressure_threshold), 0.0)
        self.pressure_gain = max(float(pressure_gain), 0.0)
        self.margin_threshold = max(float(margin_threshold), 0.0)
        self.score_mode = str(score_mode)
        if self.score_mode not in {"all", "base_topk", "winner_suppress"}:
            raise ValueError(f"unknown loop escape score mode: {self.score_mode}")
        self.score_top_k = max(int(score_top_k), 1)
        self.update_mode = str(update_mode)
        if self.update_mode not in {"target_wrong", "wrong_only"}:
            raise ValueError(f"unknown loop escape update mode: {self.update_mode}")
        self.learn_candidate_k = max(int(learn_candidate_k), 0)
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        branch_model = self.find_branch_model()
        if branch_model is None:
            raise ValueError("loop escape competitor requires a wrapped learner with branch_model")
        self.branch_count = len(branch_model.branches)
        if self.branch_count <= 0:
            raise ValueError("loop escape competitor found no branches")
        self.loop_escape_weights = np.zeros((self.vocab_size, self.branch_count), dtype=np.float32)

    def find_branch_model(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "branch_model"):
                return obj.branch_model
            obj = getattr(obj, "base", None)
        return None

    def find_segment_memory(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "segment_attractor_pressure") and hasattr(obj, "segment_state"):
                return obj
            obj = getattr(obj, "base", None)
        return None

    def branch_supports(self, context: np.ndarray) -> np.ndarray:
        branch_model = self.find_branch_model()
        if branch_model is None:
            return np.zeros((self.branch_count, self.vocab_size), dtype=np.float32)
        rows: list[np.ndarray] = []
        for order, branch in zip(branch_model.branch_orders, branch_model.branches):
            feature = branch.feature(context[-order:])
            scores = (branch.prototypes @ feature).astype(np.float32)
            scores = scores - float(np.mean(scores))
            scale = max(float(np.std(scores)), 1e-6)
            rows.append(np.clip(scores / scale, -self.support_clip, self.support_clip).astype(np.float32))
        return np.stack(rows, axis=0).astype(np.float32)

    def escape_signal(self, supports: np.ndarray) -> np.ndarray:
        return np.einsum("tb,bt->t", self.loop_escape_weights, supports, optimize=True).astype(np.float32)

    def candidate_limited_signal(self, base_scores: np.ndarray, signal: np.ndarray) -> np.ndarray:
        if self.score_mode == "all":
            return signal.astype(np.float32, copy=False)
        limited = np.zeros_like(signal, dtype=np.float32)
        if self.score_mode == "winner_suppress":
            winner = int(np.argmax(base_scores))
            limited[winner] = min(float(signal[winner]), 0.0)
            return limited
        k = min(self.score_top_k, signal.size)
        if k <= 0:
            return limited
        candidates = np.argpartition(base_scores.astype(np.float32), -k)[-k:]
        limited[candidates] = signal[candidates]
        return limited

    def segment_pressure(self) -> float:
        segment = self.find_segment_memory()
        if segment is None:
            return 0.0
        return float(segment.segment_attractor_pressure[0])

    def gate_value(self, base_scores: np.ndarray) -> float:
        pressure_signal = max(0.0, self.segment_pressure() - self.pressure_threshold)
        margin_signal = 0.0
        if base_scores.size > 1 and self.margin_threshold > 0.0:
            top2 = np.partition(base_scores.astype(np.float32), -2)[-2:]
            margin = float(top2[-1] - top2[-2])
            margin_signal = max(0.0, self.margin_threshold - margin) / self.margin_threshold
        if self.gate_mode == "pressure":
            active = pressure_signal
        elif self.gate_mode == "pressure_and_margin":
            if pressure_signal <= 0.0 or margin_signal <= 0.0:
                return 0.0
            active = pressure_signal + margin_signal
        else:
            active = max(pressure_signal, margin_signal)
        if active <= 0.0:
            return 0.0
        return 1.0 + self.pressure_gain * math.tanh(active)

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)

    def scores(self, context: np.ndarray) -> np.ndarray:
        base_scores = self.base.scores(context).astype(np.float32, copy=False)
        gate = self.gate_value(base_scores)
        if gate <= 0.0 or self.strength <= 0.0:
            return base_scores.astype(np.float32)
        supports = self.branch_supports(context)
        signal = self.candidate_limited_signal(base_scores, self.escape_signal(supports))
        return (base_scores + self.strength * gate * signal).astype(np.float32)

    def learn_escape(
        self,
        base_scores: np.ndarray,
        scores: np.ndarray,
        supports: np.ndarray,
        target: int,
        gate: float,
    ) -> None:
        if self.decay < 1.0:
            self.loop_escape_weights *= self.decay
        if self.lr <= 0.0 or self.top_k <= 0 or gate <= 0.0:
            return
        target = int(target)
        adjusted = scores.astype(np.float32, copy=True)
        target_score = float(adjusted[target])
        adjusted[target] = -np.inf
        if self.learn_candidate_k > 0:
            candidate_count = min(self.learn_candidate_k, adjusted.size)
            candidate_mask = np.zeros(adjusted.size, dtype=bool)
            candidate_idx = np.argpartition(base_scores.astype(np.float32), -candidate_count)[-candidate_count:]
            candidate_mask[candidate_idx] = True
            adjusted[~candidate_mask] = -np.inf
        k = min(self.top_k, self.vocab_size - 1)
        if k <= 0:
            return
        wrongs = np.argpartition(adjusted, -k)[-k:]
        for wrong in wrongs:
            wrong = int(wrong)
            if not np.isfinite(float(adjusted[wrong])):
                continue
            if float(adjusted[wrong]) + self.margin <= target_score:
                continue
            step = self.lr * gate
            if self.update_mode == "target_wrong":
                self.loop_escape_weights[target] = np.clip(
                    self.loop_escape_weights[target] + step * supports[:, target],
                    -self.clip,
                    self.clip,
                )
            self.loop_escape_weights[wrong] = np.clip(
                self.loop_escape_weights[wrong] - (step / k) * supports[:, wrong],
                -self.clip,
                self.clip,
            )

    def update(self, context: np.ndarray, target: int) -> None:
        base_scores = self.base.scores(context).astype(np.float32, copy=False)
        gate = self.gate_value(base_scores)
        supports = self.branch_supports(context)
        signal = self.candidate_limited_signal(base_scores, self.escape_signal(supports))
        pre_scores = (base_scores + self.strength * gate * signal).astype(np.float32)
        self.learn_escape(base_scores, pre_scores, supports, target, gate)
        self.base.update(context, target)

    def state_bytes(self) -> int:
        return int(self.base.state_bytes() + self.loop_escape_weights.nbytes)

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())

    def state_config_metadata(self) -> dict[str, Any]:
        branch_model = self.find_branch_model()
        branch_orders = list(getattr(branch_model, "branch_orders", [])) if branch_model is not None else []
        return {
            "class": self.__class__.__name__,
            "strength": float(self.strength),
            "lr": float(self.lr),
            "decay": float(self.decay),
            "clip": float(self.clip),
            "support_clip": float(self.support_clip),
            "top_k": int(self.top_k),
            "margin": float(self.margin),
            "gate_mode": self.gate_mode,
            "pressure_threshold": float(self.pressure_threshold),
            "pressure_gain": float(self.pressure_gain),
            "margin_threshold": float(self.margin_threshold),
            "score_mode": self.score_mode,
            "score_top_k": int(self.score_top_k),
            "update_mode": self.update_mode,
            "learn_candidate_k": int(self.learn_candidate_k),
            "branch_count": int(self.branch_count),
            "branch_orders": [int(order) for order in branch_orders],
        }


class BranchStateStabilizerMemory:
    """
    Representation-level recurrent branch-state stabilizer.

    The wrapper keeps a compact dynamic state in the same feature space as the
    phase/trace WTA readout.  A local projection maps that state to a feature
    residual, and scores receive the residual only through the existing readout
    weights: W @ (P @ state).  This is a pre-readout feature correction rather
    than direct token suppression or a token-probability table.
    """

    def __init__(
        self,
        base: Any,
        strength: float,
        lr: float,
        state_decay: float,
        projection_decay: float,
        clip: float,
        target_mix: float,
        gate_mode: str,
        margin_threshold: float,
        branch_threshold: float,
        inhibition_threshold: float,
        apical_threshold: float,
        gate_gain: float,
        top_k: int,
        update_target_top_k: int,
        support_clip: float,
        input_mode: str,
        projection_rank: int,
        novelty_slots: int,
        novelty_threshold: float,
        novelty_strength: float,
        anti_attractor_strength: float,
        anti_attractor_threshold: float,
        anti_attractor_orthogonal: float,
        anti_score_strength: float,
        anti_candidate_top_k: int,
        anti_candidate_center: bool,
        anti_candidate_agreement_weight: float,
        anti_prediction_only: bool,
        seed: int,
        derived_codes: bool,
    ) -> None:
        self.base = base
        self.strength = max(float(strength), 0.0)
        self.lr = max(float(lr), 0.0)
        self.state_decay = float(np.clip(state_decay, 0.0, 0.999))
        self.projection_decay = float(np.clip(projection_decay, 0.0, 1.0))
        self.clip = max(float(clip), 1e-6)
        self.target_mix = max(float(target_mix), 0.0)
        self.gate_mode = str(gate_mode)
        if self.gate_mode not in {
            "none",
            "margin",
            "branch",
            "inhibition",
            "apical",
            "any",
            "all",
        }:
            raise ValueError(f"unknown branch-state gate mode: {self.gate_mode}")
        self.margin_threshold = max(float(margin_threshold), 0.0)
        self.branch_threshold = max(float(branch_threshold), 0.0)
        self.inhibition_threshold = max(float(inhibition_threshold), 0.0)
        self.apical_threshold = max(float(apical_threshold), 0.0)
        self.gate_gain = max(float(gate_gain), 0.0)
        self.top_k = max(int(top_k), 0)
        self.update_target_top_k = max(int(update_target_top_k), 0)
        self.support_clip = max(float(support_clip), 1e-6)
        self.input_mode = str(input_mode)
        if self.input_mode not in {"feature", "target", "mixed"}:
            raise ValueError(f"unknown branch-state input mode: {self.input_mode}")
        self.projection_rank = max(int(projection_rank), 0)
        self.novelty_slots = max(int(novelty_slots), 0)
        self.novelty_threshold = float(np.clip(novelty_threshold, -1.0, 1.0))
        self.novelty_strength = max(float(novelty_strength), 0.0)
        self.anti_attractor_strength = max(float(anti_attractor_strength), 0.0)
        self.anti_attractor_threshold = float(np.clip(anti_attractor_threshold, -1.0, 1.0))
        self.anti_attractor_orthogonal = max(float(anti_attractor_orthogonal), 0.0)
        self.anti_score_strength = max(float(anti_score_strength), 0.0)
        self.anti_candidate_top_k = max(int(anti_candidate_top_k), 0)
        self.anti_candidate_center = bool(anti_candidate_center)
        self.anti_candidate_agreement_weight = float(anti_candidate_agreement_weight)
        self.anti_prediction_only = bool(anti_prediction_only)
        self.seed_offset = int(seed) + 433494437
        self.derived_codes = bool(derived_codes)
        self.derived_state_names: set[str] = set()
        if self.derived_codes:
            self.derived_state_names.add("branch_state_codes")
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        feature_model = self.find_feature_model()
        if feature_model is None:
            raise ValueError("branch-state stabilizer requires a wrapped learner with feature() and weights")
        self.feature_dim = int(feature_model.weights.shape[1])
        if self.projection_rank > self.feature_dim:
            self.projection_rank = self.feature_dim
        rng = np.random.default_rng(self.seed_offset)
        self.branch_state_codes = phase.normalize_rows(
            rng.normal(0.0, 1.0, (self.vocab_size, self.feature_dim)).astype(np.float32)
        )
        self.branch_state = np.zeros(self.feature_dim, dtype=np.float32)
        if self.projection_rank > 0:
            self.branch_state_encoder = phase.normalize_rows(
                rng.normal(0.0, 1.0, (self.projection_rank, self.feature_dim)).astype(np.float32)
            )
            if self.derived_codes:
                self.derived_state_names.add("branch_state_encoder")
            self.branch_state_projection = np.zeros((self.feature_dim, self.projection_rank), dtype=np.float32)
        else:
            self.branch_state_encoder = np.eye(self.feature_dim, dtype=np.float32)
            self.derived_state_names.add("branch_state_encoder")
            self.branch_state_projection = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float32)
        self.branch_state_novelty_slots = np.zeros((self.novelty_slots, self.feature_dim), dtype=np.float32)
        self.branch_state_novelty_filled = np.zeros(self.novelty_slots, dtype=np.bool_)
        self.branch_state_novelty_index = np.zeros(1, dtype=np.int32)
        self.branch_state_anti_pressure = np.zeros(1, dtype=np.float32)
        self.branch_state_anti_vector = np.zeros(self.feature_dim, dtype=np.float32)

    def find_feature_model(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "feature") and hasattr(obj, "weights"):
                return obj
            obj = getattr(obj, "base", None)
        return None

    def find_branch_model(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "branch_model"):
                return obj.branch_model
            obj = getattr(obj, "base", None)
        return None

    def find_branch_agreement_memory(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "agreement_signal") and hasattr(obj, "branch_supports"):
                return obj
            obj = getattr(obj, "base", None)
        return None

    def find_inhibition_memory(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "inhibition") and hasattr(obj, "activity"):
                return obj
            obj = getattr(obj, "base", None)
        return None

    def find_apical_model(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "apical_error_trace"):
                return obj
            obj = getattr(obj, "base", None)
        return None

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()
        self.branch_state.fill(0.0)
        self.branch_state_novelty_slots.fill(0.0)
        self.branch_state_novelty_filled.fill(False)
        self.branch_state_novelty_index.fill(0)
        self.branch_state_anti_pressure.fill(0.0)
        self.branch_state_anti_vector.fill(0.0)

    def remember_previous_state(self) -> None:
        if self.novelty_slots <= 0 or not np.any(self.branch_state):
            return
        idx = int(self.branch_state_novelty_index[0])
        self.branch_state_novelty_slots[idx] = self.branch_state
        self.branch_state_novelty_filled[idx] = True
        self.branch_state_novelty_index[0] = (idx + 1) % self.novelty_slots

    def anti_attractor_adjustment(self, state: np.ndarray, value: np.ndarray) -> np.ndarray:
        self.branch_state_anti_pressure[0] = 0.0
        if self.novelty_slots <= 0 or self.anti_attractor_strength <= 0.0 or not np.any(state):
            self.branch_state_anti_vector.fill(0.0)
            return state
        valid = self.branch_state_novelty_filled
        if not np.any(valid):
            self.branch_state_anti_vector.fill(0.0)
            return state
        slots = self.branch_state_novelty_slots[valid]
        sims = slots @ state
        if sims.size == 0:
            self.branch_state_anti_vector.fill(0.0)
            return state
        excess = np.maximum(0.0, sims - self.anti_attractor_threshold).astype(np.float32)
        if not np.any(excess):
            self.branch_state_anti_vector.fill(0.0)
            return state
        pressure = min(1.0, float(np.max(excess)) / max(1.0 - self.anti_attractor_threshold, 1e-6))
        weights = excess / max(float(np.sum(excess)), 1e-6)
        attractor = phase.normalize_vector((weights @ slots).astype(np.float32))
        self.branch_state_anti_vector = attractor
        adjusted = state - self.anti_attractor_strength * pressure * attractor
        if self.anti_attractor_orthogonal > 0.0 and np.any(value):
            orthogonal = value - float(value @ attractor) * attractor
            if np.any(orthogonal):
                adjusted = adjusted + self.anti_attractor_orthogonal * pressure * phase.normalize_vector(orthogonal)
        self.branch_state_anti_pressure[0] = pressure
        if not np.any(adjusted):
            return state
        return phase.normalize_vector(adjusted.astype(np.float32))

    def observe_state_vector(self, value: np.ndarray, apply_anti: bool = True) -> None:
        self.remember_previous_state()
        raw_state = phase.normalize_vector(
            self.state_decay * self.branch_state + value.astype(np.float32, copy=False)
        )
        if apply_anti:
            self.branch_state = self.anti_attractor_adjustment(raw_state, value.astype(np.float32, copy=False))
        else:
            self.branch_state = raw_state
            self.branch_state_anti_pressure[0] = 0.0
            self.branch_state_anti_vector.fill(0.0)

    def state_input(self, context: np.ndarray | None, token: int | None) -> np.ndarray:
        feature_model = self.find_feature_model()
        feature = (
            feature_model.feature(context).astype(np.float32, copy=False)
            if context is not None and feature_model is not None
            else np.zeros(self.feature_dim, dtype=np.float32)
        )
        token_code = (
            self.branch_state_codes[int(token)].astype(np.float32, copy=False)
            if token is not None and 0 <= int(token) < self.vocab_size
            else np.zeros(self.feature_dim, dtype=np.float32)
        )
        if self.input_mode == "feature":
            value = feature
        elif self.input_mode == "target":
            value = token_code
        else:
            value = feature + self.target_mix * token_code
        return phase.normalize_vector(value)

    def observe_output(self, token: int, apply_anti: bool = True) -> None:
        self.observe_state_vector(self.state_input(None, int(token)), apply_anti=apply_anti)

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)
        self.observe_output(int(token), apply_anti=not self.anti_prediction_only)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)
        self.observe_output(int(token), apply_anti=True)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)
        self.observe_state_vector(self.state_input(context, int(target)), apply_anti=not self.anti_prediction_only)

    def branch_supports(self, context: np.ndarray) -> np.ndarray:
        branch_model = self.find_branch_model()
        if branch_model is None:
            return np.zeros((1, self.vocab_size), dtype=np.float32)
        rows: list[np.ndarray] = []
        for order, branch in zip(branch_model.branch_orders, branch_model.branches):
            feature = branch.feature(context[-order:])
            scores = (branch.prototypes @ feature).astype(np.float32)
            scores = scores - float(np.mean(scores))
            scale = max(float(np.std(scores)), 1e-6)
            rows.append(np.clip(scores / scale, -self.support_clip, self.support_clip).astype(np.float32))
        return np.stack(rows, axis=0).astype(np.float32)

    def branch_disagreement(self, context: np.ndarray, token: int) -> float:
        supports = self.branch_supports(context)
        if supports.shape[0] <= 1:
            return 0.0
        return float(np.var(supports[:, int(token)]))

    def inhibition_pressure(self, token: int) -> float:
        inhibition_memory = self.find_inhibition_memory()
        if inhibition_memory is None or not np.any(inhibition_memory.activity):
            return 0.0
        pressure = inhibition_memory.inhibition @ inhibition_memory.activity
        return float(pressure[int(token)])

    def apical_pressure(self) -> float:
        apical_model = self.find_apical_model()
        if apical_model is None:
            return 0.0
        trace = apical_model.apical_error_trace
        return float(np.max(trace)) if trace.size else 0.0

    def gate_signals(self, context: np.ndarray, base_scores: np.ndarray) -> dict[str, float]:
        pred = int(np.argmax(base_scores))
        margin_signal = 0.0
        if base_scores.size > 1 and self.margin_threshold > 0.0:
            top2 = np.partition(base_scores.astype(np.float32), -2)[-2:]
            margin = float(top2[-1] - top2[-2])
            margin_signal = max(0.0, self.margin_threshold - margin) / self.margin_threshold
        branch_signal = max(0.0, self.branch_disagreement(context, pred) - self.branch_threshold)
        inhibition_signal = max(0.0, self.inhibition_pressure(pred) - self.inhibition_threshold)
        apical_signal = max(0.0, self.apical_pressure() - self.apical_threshold)
        return {
            "margin": margin_signal,
            "branch": branch_signal,
            "inhibition": inhibition_signal,
            "apical": apical_signal,
        }

    def gate_value(self, context: np.ndarray, base_scores: np.ndarray) -> float:
        if self.gate_mode == "none":
            return 1.0
        signals = self.gate_signals(context, base_scores)
        if self.gate_mode in signals:
            active = signals[self.gate_mode]
        elif self.gate_mode == "any":
            active = max(signals.values())
        else:
            values = list(signals.values())
            if any(value <= 0.0 for value in values):
                return 0.0
            active = min(values)
        if active <= 0.0:
            return 0.0
        return 1.0 + self.gate_gain * math.tanh(active)

    def novelty_multiplier(self) -> float:
        if self.novelty_slots <= 0 or self.novelty_strength <= 0.0 or not np.any(self.branch_state):
            return 1.0
        valid = self.branch_state_novelty_filled
        if not np.any(valid):
            return 1.0
        sims = self.branch_state_novelty_slots[valid] @ self.branch_state
        max_sim = float(np.max(sims)) if sims.size else 0.0
        if max_sim <= self.novelty_threshold:
            return 1.0
        denom = max(1.0 - self.novelty_threshold, 1e-6)
        pressure = min(1.0, (max_sim - self.novelty_threshold) / denom)
        return max(0.0, 1.0 - self.novelty_strength * pressure)

    def projection_input(self, state: np.ndarray) -> np.ndarray:
        if self.projection_rank > 0:
            return (self.branch_state_encoder @ state).astype(np.float32)
        return state.astype(np.float32, copy=False)

    def residual_scores(self) -> np.ndarray:
        feature_model = self.find_feature_model()
        if feature_model is None or not np.any(self.branch_state):
            return np.zeros(self.vocab_size, dtype=np.float32)
        residual = self.branch_state_projection @ self.projection_input(self.branch_state)
        if not np.any(residual):
            return np.zeros(self.vocab_size, dtype=np.float32)
        residual = phase.normalize_vector(residual)
        score_scale = float(getattr(feature_model, "score_scale", 1.0))
        return (score_scale * (feature_model.weights @ residual)).astype(np.float32)

    def anti_attractor_scores(self, base_scores: np.ndarray | None = None) -> np.ndarray:
        feature_model = self.find_feature_model()
        if (
            feature_model is None
            or self.anti_score_strength <= 0.0
            or float(self.branch_state_anti_pressure[0]) <= 0.0
            or not np.any(self.branch_state_anti_vector)
        ):
            return np.zeros(self.vocab_size, dtype=np.float32)
        anti_feature = phase.normalize_vector(self.branch_state_anti_vector)
        score_scale = float(getattr(feature_model, "score_scale", 1.0))
        anti_scores = (score_scale * (feature_model.weights @ anti_feature)).astype(np.float32)
        if base_scores is None or self.anti_candidate_top_k <= 0 or anti_scores.size <= 1:
            return anti_scores
        candidate_count = min(self.anti_candidate_top_k, anti_scores.size - 1)
        if candidate_count <= 0:
            return anti_scores
        candidate_scores = base_scores.astype(np.float32, copy=False)
        candidates = np.argpartition(candidate_scores, -candidate_count)[-candidate_count:]
        limited = np.zeros_like(anti_scores)
        limited[candidates] = anti_scores[candidates]
        if self.anti_candidate_center and candidates.size > 1:
            centered = limited[candidates] - float(np.mean(limited[candidates]))
            limited[candidates] = centered.astype(np.float32, copy=False)
        return limited.astype(np.float32)

    def candidate_competition_scores(self, context: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
        if self.anti_candidate_top_k <= 0 or (
            self.anti_candidate_agreement_weight <= 0.0
            and float(self.branch_state_anti_pressure[0]) <= 0.0
        ):
            return np.zeros(self.vocab_size, dtype=np.float32)
        candidate_count = min(self.anti_candidate_top_k, base_scores.size - 1)
        if candidate_count <= 0:
            return np.zeros(self.vocab_size, dtype=np.float32)
        candidates = np.argpartition(base_scores.astype(np.float32), -candidate_count)[-candidate_count:]
        signal = np.zeros(self.vocab_size, dtype=np.float32)
        supports = self.branch_supports(context)
        agreement_memory = self.find_branch_agreement_memory()
        if agreement_memory is not None:
            agreement_scores = agreement_memory.agreement_signal(supports).astype(np.float32, copy=False)
        else:
            agreement_scores = np.mean(supports, axis=0).astype(np.float32)
        anti_scores = self.anti_attractor_scores(base_scores)
        local = self.anti_candidate_agreement_weight * agreement_scores[candidates] - anti_scores[candidates]
        if self.anti_candidate_center and candidates.size > 1:
            local = local - float(np.mean(local))
        signal[candidates] = local.astype(np.float32, copy=False)
        return signal.astype(np.float32)

    def scores(self, context: np.ndarray) -> np.ndarray:
        base_scores = self.base.scores(context).astype(np.float32, copy=False)
        if self.strength <= 0.0 or not np.any(self.branch_state_projection):
            return base_scores.astype(np.float32)
        gate = self.gate_value(context, base_scores)
        if gate <= 0.0:
            return base_scores.astype(np.float32)
        novelty = self.novelty_multiplier()
        if novelty <= 0.0:
            return base_scores.astype(np.float32)
        adjusted = base_scores + self.strength * gate * novelty * self.residual_scores()
        if self.anti_score_strength > 0.0:
            if self.anti_candidate_top_k > 0:
                adjusted = adjusted - self.anti_score_strength * float(self.branch_state_anti_pressure[0]) * self.candidate_competition_scores(context, base_scores)
            else:
                adjusted = adjusted - self.anti_score_strength * float(self.branch_state_anti_pressure[0]) * self.anti_attractor_scores(base_scores)
        return adjusted.astype(np.float32)

    def learn_projection(self, context: np.ndarray, target: int, scores: np.ndarray, gate: float) -> None:
        if self.projection_decay < 1.0:
            self.branch_state_projection *= self.projection_decay
        if self.lr <= 0.0 or self.top_k <= 0 or gate <= 0.0:
            return
        feature_model = self.find_feature_model()
        if feature_model is None:
            return
        state = self.branch_state
        if not np.any(state):
            state = self.state_input(context, int(target))
        if not np.any(state):
            return
        latent = self.projection_input(state)
        if not np.any(latent):
            return
        target = int(target)
        adjusted = scores.astype(np.float32, copy=True)
        target_score = float(adjusted[target])
        if self.update_target_top_k > 0:
            target_rank = 1 + int(np.sum(adjusted.astype(np.float32, copy=False) > target_score))
            if target_rank > min(self.update_target_top_k, self.vocab_size):
                return
        adjusted[target] = -np.inf
        k = min(self.top_k, self.vocab_size - 1)
        if k <= 0:
            return
        wrongs = np.argpartition(adjusted, -k)[-k:]
        for wrong in wrongs:
            wrong = int(wrong)
            if not np.isfinite(float(adjusted[wrong])):
                continue
            if float(adjusted[wrong]) <= target_score:
                continue
            direction = phase.normalize_vector(feature_model.weights[target] - feature_model.weights[wrong])
            step = (self.lr * gate) / k
            self.branch_state_projection += step * np.outer(direction, latent).astype(np.float32)
        np.clip(self.branch_state_projection, -self.clip, self.clip, out=self.branch_state_projection)

    def update(self, context: np.ndarray, target: int) -> None:
        base_scores = self.base.scores(context).astype(np.float32, copy=False)
        gate = self.gate_value(context, base_scores) * self.novelty_multiplier()
        pre_scores = (
            base_scores + self.strength * gate * self.residual_scores()
            if self.strength > 0.0 and gate > 0.0
            else base_scores
        ).astype(np.float32)
        if self.anti_score_strength > 0.0 and float(self.branch_state_anti_pressure[0]) > 0.0:
            if self.anti_candidate_top_k > 0:
                pre_scores = (
                    pre_scores
                    - self.anti_score_strength
                    * float(self.branch_state_anti_pressure[0])
                    * self.candidate_competition_scores(context, base_scores)
                ).astype(np.float32)
            else:
                pre_scores = (
                    pre_scores
                    - self.anti_score_strength
                    * float(self.branch_state_anti_pressure[0])
                    * self.anti_attractor_scores(base_scores)
                ).astype(np.float32)
        self.learn_projection(context, int(target), pre_scores, gate)
        self.base.update(context, target)
        self.observe_state_vector(self.state_input(context, int(target)), apply_anti=not self.anti_prediction_only)

    def state_bytes(self) -> int:
        code_bytes = 0 if self.derived_codes else self.branch_state_codes.nbytes
        encoder_bytes = 0 if self.derived_codes or self.projection_rank <= 0 else self.branch_state_encoder.nbytes
        return int(
            self.base.state_bytes()
            + code_bytes
            + encoder_bytes
            + self.branch_state.nbytes
            + self.branch_state_projection.nbytes
            + self.branch_state_novelty_slots.nbytes
            + self.branch_state_novelty_filled.nbytes
            + self.branch_state_novelty_index.nbytes
            + self.branch_state_anti_pressure.nbytes
            + self.branch_state_anti_vector.nbytes
        )

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())

    def state_config_metadata(self) -> dict[str, Any]:
        feature_model = self.find_feature_model()
        return {
            "class": self.__class__.__name__,
            "strength": float(self.strength),
            "lr": float(self.lr),
            "state_decay": float(self.state_decay),
            "projection_decay": float(self.projection_decay),
            "clip": float(self.clip),
            "target_mix": float(self.target_mix),
            "gate_mode": self.gate_mode,
            "margin_threshold": float(self.margin_threshold),
            "branch_threshold": float(self.branch_threshold),
            "inhibition_threshold": float(self.inhibition_threshold),
            "apical_threshold": float(self.apical_threshold),
            "gate_gain": float(self.gate_gain),
            "top_k": int(self.top_k),
            "update_target_top_k": int(self.update_target_top_k),
            "support_clip": float(self.support_clip),
            "input_mode": self.input_mode,
            "projection_rank": int(self.projection_rank),
            "novelty_slots": int(self.novelty_slots),
            "novelty_threshold": float(self.novelty_threshold),
            "novelty_strength": float(self.novelty_strength),
            "anti_attractor_strength": float(self.anti_attractor_strength),
            "anti_attractor_threshold": float(self.anti_attractor_threshold),
            "anti_attractor_orthogonal": float(self.anti_attractor_orthogonal),
            "anti_score_strength": float(self.anti_score_strength),
            "anti_candidate_top_k": int(self.anti_candidate_top_k),
            "anti_candidate_center": bool(self.anti_candidate_center),
            "anti_candidate_agreement_weight": float(self.anti_candidate_agreement_weight),
            "anti_prediction_only": bool(self.anti_prediction_only),
            "seed_offset": int(self.seed_offset),
            "derived_codes": bool(self.derived_codes),
            "feature_dim": int(self.feature_dim),
            "projection_shape": list(self.branch_state_projection.shape),
            "feature_model_class": feature_model.__class__.__name__ if feature_model is not None else "",
        }


class AdaptiveOutputInhibitionMemory:
    """
    Locally plastic inhibition over a recent output trace.

    On each online mistake, recently observed output neurons strengthen
    inhibitory edges into the wrong winner.  The target row is locally
    disinhibited.  This is a no-BP WTA/interneuron-style rule: the only teaching
    signal is the current target token, and the state stores learned inhibitory
    weights rather than raw text or context counts.
    """

    def __init__(
        self,
        base: Any,
        strength: float,
        decay: float,
        lr: float,
        disinhibit_lr: float,
        top_k: int,
        margin: float,
        max_weight: float,
    ) -> None:
        self.base = base
        self.strength = float(strength)
        self.decay = float(np.clip(decay, 0.0, 0.999))
        self.lr = max(float(lr), 0.0)
        self.disinhibit_lr = max(float(disinhibit_lr), 0.0)
        self.top_k = max(int(top_k), 0)
        self.margin = float(margin)
        self.max_weight = max(float(max_weight), 0.0)
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        self.activity = np.zeros(self.vocab_size, dtype=np.float32)
        self.inhibition = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()
        self.activity.fill(0.0)

    def observe_output(self, token: int) -> None:
        self.activity *= self.decay
        self.activity[int(token)] += 1.0

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)
        self.observe_output(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)
        self.observe_output(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)
        self.observe_output(target)

    def scores(self, context: np.ndarray) -> np.ndarray:
        penalty = self.inhibition @ self.activity
        return (self.base.scores(context) - self.strength * penalty).astype(np.float32)

    def learn_inhibition(self, scores: np.ndarray, target: int) -> None:
        if self.lr <= 0.0 or self.top_k <= 0 or not np.any(self.activity):
            return
        target = int(target)
        adjusted = scores.astype(np.float32, copy=True)
        target_score = float(adjusted[target])
        adjusted[target] = -np.inf
        k = min(self.top_k, self.vocab_size - 1)
        if k <= 0:
            return
        wrongs = np.argpartition(adjusted, -k)[-k:]
        active_delta = self.lr * self.activity
        for wrong in wrongs:
            wrong = int(wrong)
            if float(adjusted[wrong]) + self.margin <= target_score:
                continue
            self.inhibition[wrong] = np.minimum(self.max_weight, self.inhibition[wrong] + active_delta)
        if self.disinhibit_lr > 0.0:
            self.inhibition[target] = np.maximum(0.0, self.inhibition[target] - self.disinhibit_lr * self.activity)

    def update(self, context: np.ndarray, target: int) -> None:
        self.learn_inhibition(self.scores(context), target)
        self.base.update(context, target)
        self.observe_output(target)

    def state_bytes(self) -> int:
        return int(self.base.state_bytes() + self.activity.nbytes + self.inhibition.nbytes)

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())


class ContextGatedOutputInhibitionMemory:
    """
    Context-gated local output inhibition.

    A fixed random context encoder produces a compact gate vector.  On a local
    mistake, the wrong output row strengthens inhibitory edges from the current
    gate.  At inference, inhibition is applied only when a similar gate is
    active.  This keeps the anti-winner memory local to a context region instead
    of globally suppressing an output token everywhere.
    """

    def __init__(
        self,
        base: Any,
        strength: float,
        lr: float,
        disinhibit_lr: float,
        top_k: int,
        margin: float,
        max_weight: float,
        gate_dim: int,
        gate_decay: float,
        gate_threshold: float,
        seed: int,
    ) -> None:
        self.base = base
        self.strength = float(strength)
        self.lr = max(float(lr), 0.0)
        self.disinhibit_lr = max(float(disinhibit_lr), 0.0)
        self.top_k = max(int(top_k), 0)
        self.margin = float(margin)
        self.max_weight = max(float(max_weight), 0.0)
        self.gate_decay = float(np.clip(gate_decay, 0.0, 0.999))
        self.gate_threshold = float(gate_threshold)
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        self.gate_dim = max(int(gate_dim), 1)
        self.rng = np.random.default_rng(seed + 15485863)
        self.context_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (self.max_order, self.vocab_size, self.gate_dim)).astype(np.float32).reshape(
                self.max_order * self.vocab_size, self.gate_dim
            )
        ).reshape(self.max_order, self.vocab_size, self.gate_dim)
        self.inhibition = np.zeros((self.vocab_size, self.gate_dim), dtype=np.float32)
        self.dynamic_gate = np.zeros(self.gate_dim, dtype=np.float32)

    def context_gate(self, context: np.ndarray) -> np.ndarray:
        state = np.zeros(self.gate_dim, dtype=np.float32)
        clipped = context[-self.max_order :]
        offset = self.max_order - len(clipped)
        for pos, token in enumerate(clipped):
            state += self.context_codes[offset + pos, int(token)]
        gate = phase.normalize_vector(state)
        if self.gate_threshold > 0.0:
            active = np.abs(gate) >= self.gate_threshold
            if np.any(active):
                gate = np.where(active, gate, 0.0).astype(np.float32)
                gate = phase.normalize_vector(gate)
        return gate

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()
        self.dynamic_gate.fill(0.0)

    def observe_gate(self, context: np.ndarray) -> np.ndarray:
        gate = self.context_gate(context)
        self.dynamic_gate = phase.normalize_vector(self.gate_decay * self.dynamic_gate + gate)
        return self.dynamic_gate

    def effective_gate(self, context: np.ndarray) -> np.ndarray:
        gate = self.context_gate(context)
        if self.gate_decay <= 0.0 or not np.any(self.dynamic_gate):
            return gate
        return phase.normalize_vector(gate + self.gate_decay * self.dynamic_gate)

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)
        self.observe_gate(context)

    def scores(self, context: np.ndarray) -> np.ndarray:
        gate = self.effective_gate(context)
        penalty = self.inhibition @ gate
        return (self.base.scores(context) - self.strength * penalty).astype(np.float32)

    def learn_inhibition(self, scores: np.ndarray, target: int, gate: np.ndarray) -> None:
        if self.lr <= 0.0 or self.top_k <= 0:
            return
        target = int(target)
        adjusted = scores.astype(np.float32, copy=True)
        target_score = float(adjusted[target])
        adjusted[target] = -np.inf
        k = min(self.top_k, self.vocab_size - 1)
        if k <= 0:
            return
        wrongs = np.argpartition(adjusted, -k)[-k:]
        active_delta = self.lr * gate
        for wrong in wrongs:
            wrong = int(wrong)
            if float(adjusted[wrong]) + self.margin <= target_score:
                continue
            self.inhibition[wrong] = np.clip(self.inhibition[wrong] + active_delta, -self.max_weight, self.max_weight)
        if self.disinhibit_lr > 0.0:
            self.inhibition[target] = np.clip(
                self.inhibition[target] - self.disinhibit_lr * gate,
                -self.max_weight,
                self.max_weight,
            )

    def update(self, context: np.ndarray, target: int) -> None:
        gate = self.effective_gate(context)
        self.learn_inhibition(self.scores(context), target, gate)
        self.base.update(context, target)
        self.observe_gate(context)

    def state_bytes(self) -> int:
        return int(self.base.state_bytes() + self.context_codes.nbytes + self.inhibition.nbytes + self.dynamic_gate.nbytes)

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())


class OutputHomeostasisMemory:
    """
    Bounded output-neuron excitability for bias-free calibration.

    This is a local WTA correction, not a token-probability table: on a current
    output error it raises the target neuron's excitability and lowers the wrong
    winner's excitability, then decays the state.  It uses only the current
    prediction/target pair and stores no raw text or context counts.
    """

    def __init__(
        self,
        base: Any,
        strength: float,
        lr: float,
        decay: float,
        clip: float,
    ) -> None:
        self.base = base
        self.strength = max(float(strength), 0.0)
        self.lr = max(float(lr), 0.0)
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.clip = max(float(clip), 1e-6)
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        self.excitability = np.zeros(self.vocab_size, dtype=np.float32)

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)

    def scores(self, context: np.ndarray) -> np.ndarray:
        return (self.base.scores(context) + self.strength * self.excitability).astype(np.float32)

    def learn_homeostasis(self, scores: np.ndarray, target: int) -> None:
        self.excitability *= self.decay
        if self.lr <= 0.0:
            return
        target = int(target)
        pred = int(np.argmax(scores))
        if pred == target:
            return
        self.excitability[target] = np.clip(self.excitability[target] + self.lr, -self.clip, self.clip)
        self.excitability[pred] = np.clip(self.excitability[pred] - self.lr, -self.clip, self.clip)

    def update(self, context: np.ndarray, target: int) -> None:
        pre_scores = self.scores(context)
        self.learn_homeostasis(pre_scores, target)
        self.base.update(context, target)

    def state_bytes(self) -> int:
        return int(self.base.state_bytes() + self.excitability.nbytes)

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())


class FeatureConditionedCalibrationMemory:
    """
    Context-conditioned local output calibration.

    A fixed random context encoder produces a compact gate.  On a local WTA
    mistake, only the target and wrong-winner rows receive a Hebbian-style
    calibration update from the gate.  The state is synaptic and local; it is
    not a token-probability table or context-count cache.
    """

    def __init__(
        self,
        base: Any,
        strength: float,
        lr: float,
        decay: float,
        clip: float,
        gate_dim: int,
        gate_decay: float,
        gate_threshold: float,
        memory_scope: str,
        score_top_k: int,
        update_top_k: int,
        update_margin: float,
        update_rank_tau: float,
        update_margin_tau: float,
        update_mode: str,
        seed: int,
        derived_codes: bool,
    ) -> None:
        self.base = base
        self.strength = max(float(strength), 0.0)
        self.lr = max(float(lr), 0.0)
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.clip = max(float(clip), 1e-6)
        self.gate_decay = float(np.clip(gate_decay, 0.0, 0.999))
        self.gate_threshold = float(gate_threshold)
        self.memory_scope = str(memory_scope)
        if self.memory_scope not in {"persistent", "dynamic"}:
            raise ValueError(f"unknown feature calibration memory scope: {self.memory_scope}")
        self.score_top_k = max(int(score_top_k), 0)
        self.update_top_k = max(int(update_top_k), 0)
        self.update_margin = max(float(update_margin), 0.0)
        self.update_rank_tau = max(float(update_rank_tau), 0.0)
        self.update_margin_tau = max(float(update_margin_tau), 0.0)
        self.update_mode = str(update_mode)
        if self.update_mode not in {"target_wrong", "wrong_only"}:
            raise ValueError(f"unknown feature calibration update mode: {self.update_mode}")
        self.seed_offset = int(seed) + 32452843
        self.derived_codes = bool(derived_codes)
        self.derived_state_names = {"calibration_codes"} if derived_codes else set()
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        self.gate_dim = max(int(gate_dim), 1)
        self.rng = np.random.default_rng(self.seed_offset)
        self.calibration_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (self.max_order, self.vocab_size, self.gate_dim)).astype(np.float32).reshape(
                self.max_order * self.vocab_size, self.gate_dim
            )
        ).reshape(self.max_order, self.vocab_size, self.gate_dim)
        self.calibration = np.zeros((self.vocab_size, self.gate_dim), dtype=np.float32)
        self.calibration_gate = np.zeros(self.gate_dim, dtype=np.float32)

    def context_gate(self, context: np.ndarray) -> np.ndarray:
        state = np.zeros(self.gate_dim, dtype=np.float32)
        clipped = context[-self.max_order :]
        offset = self.max_order - len(clipped)
        for pos, token in enumerate(clipped):
            state += self.calibration_codes[offset + pos, int(token)]
        gate = phase.normalize_vector(state)
        if self.gate_threshold > 0.0:
            active = np.abs(gate) >= self.gate_threshold
            if np.any(active):
                gate = np.where(active, gate, 0.0).astype(np.float32)
                gate = phase.normalize_vector(gate)
        return gate

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()
        self.calibration_gate.fill(0.0)
        if self.memory_scope == "dynamic":
            self.calibration.fill(0.0)

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)
        self.observe_gate(context)

    def observe_gate(self, context: np.ndarray) -> np.ndarray:
        gate = self.context_gate(context)
        self.calibration_gate = phase.normalize_vector(self.gate_decay * self.calibration_gate + gate)
        return self.calibration_gate

    def effective_gate(self, context: np.ndarray) -> np.ndarray:
        gate = self.context_gate(context)
        if self.gate_decay <= 0.0 or not np.any(self.calibration_gate):
            return gate
        return phase.normalize_vector(gate + self.gate_decay * self.calibration_gate)

    def top_k_mask(self, scores: np.ndarray, k: int) -> np.ndarray:
        mask = np.zeros(scores.size, dtype=bool)
        if k <= 0:
            mask.fill(True)
            return mask
        count = min(int(k), scores.size)
        if count >= scores.size:
            mask.fill(True)
            return mask
        candidates = np.argpartition(scores.astype(np.float32, copy=False), -count)[-count:]
        mask[candidates] = True
        return mask

    def calibrated_scores(self, base_scores: np.ndarray, gate: np.ndarray) -> np.ndarray:
        signal = self.calibration @ gate
        if self.score_top_k > 0:
            mask = self.top_k_mask(base_scores, self.score_top_k)
            limited = np.zeros_like(signal, dtype=np.float32)
            limited[mask] = signal[mask]
            signal = limited
        return (base_scores.astype(np.float32, copy=False) + self.strength * signal).astype(np.float32)

    def scores(self, context: np.ndarray) -> np.ndarray:
        gate = self.effective_gate(context)
        base_scores = self.base.scores(context)
        return self.calibrated_scores(base_scores, gate)

    def learn_calibration(
        self,
        scores: np.ndarray,
        target: int,
        gate: np.ndarray,
        base_scores: np.ndarray,
    ) -> None:
        if self.decay < 1.0:
            self.calibration *= self.decay
        if self.lr <= 0.0:
            return
        target = int(target)
        pred = int(np.argmax(scores))
        if pred == target:
            return
        if self.update_top_k > 0:
            update_mask = self.top_k_mask(base_scores, self.update_top_k)
            if not bool(update_mask[target]):
                return
        target_base_score = float(base_scores[target])
        if self.update_margin > 0.0 and base_scores.size > 1:
            adjusted = base_scores.astype(np.float32, copy=True)
            adjusted[target] = -np.inf
            wrong_score = float(np.max(adjusted))
            if wrong_score - target_base_score > self.update_margin:
                return
        update_weight = 1.0
        if self.update_rank_tau > 0.0:
            target_rank = 1 + int(np.sum(base_scores.astype(np.float32, copy=False) > target_base_score))
            update_weight *= math.exp(-float(max(target_rank - 1, 0)) / self.update_rank_tau)
        if self.update_margin_tau > 0.0 and base_scores.size > 1:
            adjusted = base_scores.astype(np.float32, copy=True)
            adjusted[target] = -np.inf
            wrong_score = float(np.max(adjusted))
            update_weight *= math.exp(-max(wrong_score - target_base_score, 0.0) / self.update_margin_tau)
        if update_weight <= 1e-8:
            return
        delta = (self.lr * update_weight) * gate
        if self.update_mode == "target_wrong":
            self.calibration[target] = np.clip(self.calibration[target] + delta, -self.clip, self.clip)
        self.calibration[pred] = np.clip(self.calibration[pred] - delta, -self.clip, self.clip)

    def update(self, context: np.ndarray, target: int) -> None:
        gate = self.effective_gate(context)
        base_scores = self.base.scores(context)
        self.learn_calibration(self.calibrated_scores(base_scores, gate), target, gate, base_scores)
        self.base.update(context, target)
        self.observe_gate(context)

    def state_bytes(self) -> int:
        return int(
            self.base.state_bytes()
            + self.calibration_codes.nbytes
            + self.calibration.nbytes
            + self.calibration_gate.nbytes
        )

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())

    def state_config_metadata(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "strength": float(self.strength),
            "lr": float(self.lr),
            "decay": float(self.decay),
            "clip": float(self.clip),
            "gate_decay": float(self.gate_decay),
            "gate_threshold": float(self.gate_threshold),
            "memory_scope": self.memory_scope,
            "score_top_k": int(self.score_top_k),
            "update_top_k": int(self.update_top_k),
            "update_margin": float(self.update_margin),
            "update_rank_tau": float(self.update_rank_tau),
            "update_margin_tau": float(self.update_margin_tau),
            "update_mode": self.update_mode,
            "seed_offset": int(self.seed_offset),
            "derived_codes": bool(self.derived_codes),
            "max_order": int(self.max_order),
            "vocab_size": int(self.vocab_size),
            "gate_dim": int(self.gate_dim),
            "calibration_codes_shape": list(self.calibration_codes.shape),
        }


class ReadoutGainMemory:
    """
    Non-statistical readout energy gain.

    The fixed mode turns a global temperature audit into an explicit model-side
    gain.  The margin mode is a local dynamic gain from current WTA separation:
    sharper energy when the winner is clear, softer energy when competition is
    ambiguous.  It stores no token counts or raw text.
    """

    def __init__(
        self,
        base: Any,
        gain: float,
        mode: str,
        margin_center: float,
        margin_scale: float,
        min_gain: float,
        max_gain: float,
    ) -> None:
        self.base = base
        self.gain = max(float(gain), 1e-6)
        self.mode = str(mode)
        if self.mode not in {"fixed", "margin"}:
            raise ValueError(f"unknown readout gain mode: {self.mode}")
        self.margin_center = float(margin_center)
        self.margin_scale = max(float(margin_scale), 1e-6)
        self.min_gain = max(float(min_gain), 1e-6)
        self.max_gain = max(float(max_gain), self.min_gain)
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)

    def local_gain(self, scores: np.ndarray) -> float:
        if self.mode == "fixed" or scores.size < 2:
            return self.gain
        top2 = np.partition(scores.astype(np.float32), -2)[-2:]
        margin = float(top2[-1] - top2[-2])
        centered = (margin - self.margin_center) / self.margin_scale
        dynamic = self.gain * (1.0 + math.tanh(centered))
        return float(np.clip(dynamic, self.min_gain, self.max_gain))

    def scores(self, context: np.ndarray) -> np.ndarray:
        scores = self.base.scores(context).astype(np.float32, copy=False)
        return (self.local_gain(scores) * scores).astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        self.base.update(context, target)

    def state_bytes(self) -> int:
        return int(self.base.state_bytes())

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())

    def state_config_metadata(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "gain": float(self.gain),
            "mode": self.mode,
            "margin_center": float(self.margin_center),
            "margin_scale": float(self.margin_scale),
            "min_gain": float(self.min_gain),
            "max_gain": float(self.max_gain),
        }


class LocalAdaptiveReadoutGainMemory:
    """
    Context-local readout energy gain.

    A fixed random context encoder produces a compact gate.  A scalar synaptic
    gain trace is learned online from local target-vs-winner feedback: correct
    WTA decisions increase the local readout energy, mistakes decrease it.
    This stores no token counts or raw text; it is a checkpointable neural
    modulation state layered over the existing no-BP learner.
    """

    def __init__(
        self,
        base: Any,
        base_gain: float,
        strength: float,
        lr: float,
        decay: float,
        clip: float,
        min_gain: float,
        max_gain: float,
        gate_dim: int,
        gate_decay: float,
        gate_threshold: float,
        correct_margin: float,
        mistake_margin: float,
        update_mode: str,
        error_clip: float,
        memory_scope: str,
        seed: int,
        derived_codes: bool,
    ) -> None:
        self.base = base
        self.base_gain = max(float(base_gain), 1e-6)
        self.strength = max(float(strength), 0.0)
        self.lr = max(float(lr), 0.0)
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.clip = max(float(clip), 1e-6)
        self.min_gain = max(float(min_gain), 1e-6)
        self.max_gain = max(float(max_gain), self.min_gain)
        self.gate_decay = float(np.clip(gate_decay, 0.0, 0.999))
        self.gate_threshold = float(gate_threshold)
        self.correct_margin = float(correct_margin)
        self.mistake_margin = float(mistake_margin)
        self.update_mode = str(update_mode)
        if self.update_mode not in {"wta", "ce"}:
            raise ValueError(f"unknown local readout gain update mode: {self.update_mode}")
        self.error_clip = max(float(error_clip), 1e-6)
        self.memory_scope = str(memory_scope)
        if self.memory_scope not in {"persistent", "dynamic"}:
            raise ValueError(f"unknown local readout gain memory scope: {self.memory_scope}")
        self.seed_offset = int(seed) + 49979687
        self.derived_codes = bool(derived_codes)
        self.derived_state_names = {"gain_codes"} if derived_codes else set()
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        self.gate_dim = max(int(gate_dim), 1)
        self.rng = np.random.default_rng(self.seed_offset)
        self.gain_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (self.max_order, self.vocab_size, self.gate_dim)).astype(np.float32).reshape(
                self.max_order * self.vocab_size, self.gate_dim
            )
        ).reshape(self.max_order, self.vocab_size, self.gate_dim)
        self.gain_weights = np.zeros(self.gate_dim, dtype=np.float32)
        self.gain_gate = np.zeros(self.gate_dim, dtype=np.float32)

    def context_gate(self, context: np.ndarray) -> np.ndarray:
        state = np.zeros(self.gate_dim, dtype=np.float32)
        clipped = context[-self.max_order :]
        offset = self.max_order - len(clipped)
        for pos, token in enumerate(clipped):
            state += self.gain_codes[offset + pos, int(token)]
        gate = phase.normalize_vector(state)
        if self.gate_threshold > 0.0:
            active = np.abs(gate) >= self.gate_threshold
            if np.any(active):
                gate = np.where(active, gate, 0.0).astype(np.float32)
                gate = phase.normalize_vector(gate)
        return gate

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()
        self.gain_gate.fill(0.0)
        if self.memory_scope == "dynamic":
            self.gain_weights.fill(0.0)

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)
        self.observe_gate(context)

    def observe_gate(self, context: np.ndarray) -> np.ndarray:
        gate = self.context_gate(context)
        self.gain_gate = phase.normalize_vector(self.gate_decay * self.gain_gate + gate)
        return self.gain_gate

    def effective_gate(self, context: np.ndarray) -> np.ndarray:
        gate = self.context_gate(context)
        if self.gate_decay <= 0.0 or not np.any(self.gain_gate):
            return gate
        return phase.normalize_vector(gate + self.gate_decay * self.gain_gate)

    def local_gain(self, gate: np.ndarray) -> float:
        modulation = float(self.gain_weights @ gate)
        gain = self.base_gain * (1.0 + self.strength * math.tanh(modulation))
        return float(np.clip(gain, self.min_gain, self.max_gain))

    def scores(self, context: np.ndarray) -> np.ndarray:
        gate = self.effective_gate(context)
        scores = self.base.scores(context).astype(np.float32, copy=False)
        return (self.local_gain(gate) * scores).astype(np.float32)

    def learn_gain(self, base_scores: np.ndarray, target: int, gate: np.ndarray) -> None:
        if self.decay < 1.0:
            self.gain_weights *= self.decay
        if self.lr <= 0.0:
            return
        target = int(target)
        if self.update_mode == "ce":
            probs = phase.softmax(base_scores.astype(np.float32, copy=False), temperature=1.0)
            expected_score = float(probs @ base_scores)
            local_error = float(base_scores[target]) - expected_score
            delta = self.lr * float(np.clip(local_error, -self.error_clip, self.error_clip))
            if abs(delta) <= 1e-12:
                return
            self.gain_weights = np.clip(self.gain_weights + delta * gate, -self.clip, self.clip).astype(np.float32)
            return
        adjusted = base_scores.astype(np.float32, copy=True)
        target_score = float(adjusted[target])
        adjusted[target] = -np.inf
        if adjusted.size <= 1:
            return
        wrong_score = float(np.max(adjusted))
        pred = int(np.argmax(base_scores))
        delta = 0.0
        if pred == target and target_score - wrong_score >= self.correct_margin:
            delta = self.lr
        elif pred != target and wrong_score - target_score >= self.mistake_margin:
            delta = -self.lr
        if delta == 0.0:
            return
        self.gain_weights = np.clip(self.gain_weights + delta * gate, -self.clip, self.clip).astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        gate = self.effective_gate(context)
        base_scores = self.base.scores(context)
        self.learn_gain(base_scores, target, gate)
        self.base.update(context, target)
        self.observe_gate(context)

    def state_bytes(self) -> int:
        return int(self.base.state_bytes() + self.gain_codes.nbytes + self.gain_weights.nbytes + self.gain_gate.nbytes)

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())

    def state_config_metadata(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "base_gain": float(self.base_gain),
            "strength": float(self.strength),
            "lr": float(self.lr),
            "decay": float(self.decay),
            "clip": float(self.clip),
            "min_gain": float(self.min_gain),
            "max_gain": float(self.max_gain),
            "gate_decay": float(self.gate_decay),
            "gate_threshold": float(self.gate_threshold),
            "correct_margin": float(self.correct_margin),
            "mistake_margin": float(self.mistake_margin),
            "update_mode": self.update_mode,
            "error_clip": float(self.error_clip),
            "memory_scope": self.memory_scope,
            "seed_offset": int(self.seed_offset),
            "derived_codes": bool(self.derived_codes),
            "max_order": int(self.max_order),
            "vocab_size": int(self.vocab_size),
            "gate_dim": int(self.gate_dim),
            "gain_codes_shape": list(self.gain_codes.shape),
        }


class BranchAgreementReadoutMemory:
    """
    Dendritic branch-agreement readout.

    The wrapper reuses the learned phase branches inside the no-BP learner and
    adds a token-wise boost only when independent branches support the same
    output.  Unlike scalar gain, this can change winner ordering.  It stores no
    token-count table or raw text.
    """

    def __init__(
        self,
        base: Any,
        strength: float,
        mode: str,
        clip: float,
        threshold: float,
        variance_penalty: float,
    ) -> None:
        self.base = base
        self.strength = float(strength)
        self.mode = str(mode)
        if self.mode not in {"mean_min", "positive_fraction", "low_variance", "min"}:
            raise ValueError(f"unknown branch agreement mode: {self.mode}")
        self.clip = max(float(clip), 1e-6)
        self.threshold = float(threshold)
        self.variance_penalty = max(float(variance_penalty), 0.0)
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        branch_model = self.find_branch_model()
        if branch_model is None:
            raise ValueError("branch agreement readout requires a wrapped learner with branch_model")
        self.branch_count = len(branch_model.branches)
        if self.branch_count <= 0:
            raise ValueError("branch agreement readout found no branches")

    def find_branch_model(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "branch_model"):
                return obj.branch_model
            obj = getattr(obj, "base", None)
        return None

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)

    def branch_supports(self, context: np.ndarray) -> np.ndarray:
        branch_model = self.find_branch_model()
        if branch_model is None:
            return np.zeros((1, self.vocab_size), dtype=np.float32)
        rows: list[np.ndarray] = []
        for order, branch in zip(branch_model.branch_orders, branch_model.branches):
            feature = branch.feature(context[-order:])
            scores = (branch.prototypes @ feature).astype(np.float32)
            scores = scores - float(np.mean(scores))
            scale = max(float(np.std(scores)), 1e-6)
            rows.append(np.clip(scores / scale, -self.clip, self.clip).astype(np.float32))
        return np.stack(rows, axis=0).astype(np.float32)

    def agreement_signal(self, supports: np.ndarray) -> np.ndarray:
        if supports.shape[0] <= 1:
            signal = supports[0]
        elif self.mode == "min":
            signal = np.min(supports, axis=0)
        elif self.mode == "mean_min":
            signal = 0.5 * (np.mean(supports, axis=0) + np.min(supports, axis=0))
        elif self.mode == "positive_fraction":
            signal = np.mean(supports, axis=0) * np.mean(supports > self.threshold, axis=0)
        else:
            signal = np.mean(supports, axis=0) - self.variance_penalty * np.var(supports, axis=0)
        return np.clip(signal, -self.clip, self.clip).astype(np.float32)

    def scores(self, context: np.ndarray) -> np.ndarray:
        base_scores = self.base.scores(context).astype(np.float32, copy=False)
        signal = self.agreement_signal(self.branch_supports(context))
        return (base_scores + self.strength * signal).astype(np.float32)

    def update(self, context: np.ndarray, target: int) -> None:
        self.base.update(context, target)

    def state_bytes(self) -> int:
        return int(self.base.state_bytes())

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())

    def state_config_metadata(self) -> dict[str, Any]:
        branch_model = self.find_branch_model()
        branch_orders = list(getattr(branch_model, "branch_orders", [])) if branch_model is not None else []
        return {
            "class": self.__class__.__name__,
            "strength": float(self.strength),
            "mode": self.mode,
            "clip": float(self.clip),
            "threshold": float(self.threshold),
            "variance_penalty": float(self.variance_penalty),
            "branch_count": int(self.branch_count),
            "branch_orders": [int(order) for order in branch_orders],
        }


class PlasticBranchAgreementReadoutMemory:
    """
    Locally plastic branch-agreement correction.

    Each output token owns a small vector over dendritic branches.  On a local
    WTA error, the target row is moved toward its current branch-support vector
    and the wrong-winner row is moved away from its support vector.  The score
    correction is token-wise, so it can alter winner ordering while storing only
    synaptic state.
    """

    def __init__(
        self,
        base: Any,
        strength: float,
        lr: float,
        decay: float,
        clip: float,
        support_clip: float,
        top_k: int,
        margin: float,
        pressure_mode: str,
        pressure_threshold: float,
        pressure_gain: float,
        loop_window: int,
    ) -> None:
        self.base = base
        self.strength = float(strength)
        self.lr = max(float(lr), 0.0)
        self.decay = float(np.clip(decay, 0.0, 1.0))
        self.clip = max(float(clip), 1e-6)
        self.support_clip = max(float(support_clip), 1e-6)
        self.top_k = max(int(top_k), 0)
        self.margin = float(margin)
        self.pressure_mode = str(pressure_mode)
        if self.pressure_mode not in {"none", "inhibition", "context_loop", "either", "both"}:
            raise ValueError(f"unknown plastic branch pressure mode: {self.pressure_mode}")
        self.pressure_threshold = max(float(pressure_threshold), 0.0)
        self.pressure_gain = max(float(pressure_gain), 0.0)
        self.loop_window = max(int(loop_window), 1)
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        branch_model = self.find_branch_model()
        if branch_model is None:
            raise ValueError("plastic branch agreement requires a wrapped learner with branch_model")
        self.branch_count = len(branch_model.branches)
        if self.branch_count <= 0:
            raise ValueError("plastic branch agreement found no branches")
        self.plastic_branch_weights = np.zeros((self.vocab_size, self.branch_count), dtype=np.float32)

    def find_branch_model(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "branch_model"):
                return obj.branch_model
            obj = getattr(obj, "base", None)
        return None

    def find_inhibition_memory(self) -> Any | None:
        obj = self.base
        seen: set[int] = set()
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            if hasattr(obj, "inhibition") and hasattr(obj, "activity"):
                return obj
            obj = getattr(obj, "base", None)
        return None

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)

    def branch_supports(self, context: np.ndarray) -> np.ndarray:
        branch_model = self.find_branch_model()
        if branch_model is None:
            return np.zeros((self.branch_count, self.vocab_size), dtype=np.float32)
        rows: list[np.ndarray] = []
        for order, branch in zip(branch_model.branch_orders, branch_model.branches):
            feature = branch.feature(context[-order:])
            scores = (branch.prototypes @ feature).astype(np.float32)
            scores = scores - float(np.mean(scores))
            scale = max(float(np.std(scores)), 1e-6)
            rows.append(np.clip(scores / scale, -self.support_clip, self.support_clip).astype(np.float32))
        return np.stack(rows, axis=0).astype(np.float32)

    def plastic_signal(self, supports: np.ndarray) -> np.ndarray:
        return np.einsum("tb,bt->t", self.plastic_branch_weights, supports, optimize=True).astype(np.float32)

    def inhibition_pressure(self) -> np.ndarray:
        inhibition_memory = self.find_inhibition_memory()
        if inhibition_memory is None or not np.any(inhibition_memory.activity):
            return np.zeros(self.vocab_size, dtype=np.float32)
        pressure = inhibition_memory.inhibition @ inhibition_memory.activity
        return pressure.astype(np.float32, copy=False)

    def context_loop_pressure(self, context: np.ndarray) -> float:
        window = [int(token) for token in context[-self.loop_window :]]
        if len(window) <= 1:
            return 0.0
        counts: dict[int, int] = {}
        for token in window:
            counts[token] = counts.get(token, 0) + 1
        max_fraction = max(counts.values()) / len(window)
        repeated_adjacent = sum(int(window[idx] == window[idx - 1]) for idx in range(1, len(window)))
        adjacent_fraction = repeated_adjacent / max(len(window) - 1, 1)
        return float(max(max_fraction, adjacent_fraction))

    def pressure_gate(self, wrong: int, context: np.ndarray, pressure: np.ndarray) -> float:
        if self.pressure_mode == "none":
            return 1.0
        active: list[float] = []
        if self.pressure_mode in {"inhibition", "either", "both"}:
            inhibition_value = float(pressure[int(wrong)])
            if inhibition_value >= self.pressure_threshold:
                active.append(inhibition_value)
        if self.pressure_mode in {"context_loop", "either", "both"}:
            loop_value = self.context_loop_pressure(context)
            if loop_value >= self.pressure_threshold:
                active.append(loop_value)
        if self.pressure_mode == "both" and len(active) < 2:
            return 0.0
        if self.pressure_mode in {"inhibition", "context_loop"} and not active:
            return 0.0
        if self.pressure_mode == "either" and not active:
            return 0.0
        return 1.0 + self.pressure_gain * math.tanh(max(active) - self.pressure_threshold)

    def scores(self, context: np.ndarray) -> np.ndarray:
        base_scores = self.base.scores(context).astype(np.float32, copy=False)
        supports = self.branch_supports(context)
        return (base_scores + self.strength * self.plastic_signal(supports)).astype(np.float32)

    def learn_plastic(self, scores: np.ndarray, supports: np.ndarray, target: int, context: np.ndarray) -> None:
        if self.decay < 1.0:
            self.plastic_branch_weights *= self.decay
        if self.lr <= 0.0 or self.top_k <= 0:
            return
        target = int(target)
        adjusted = scores.astype(np.float32, copy=True)
        target_score = float(adjusted[target])
        adjusted[target] = -np.inf
        k = min(self.top_k, self.vocab_size - 1)
        if k <= 0:
            return
        wrongs = np.argpartition(adjusted, -k)[-k:]
        pressure = self.inhibition_pressure()
        for wrong in wrongs:
            wrong = int(wrong)
            if float(adjusted[wrong]) + self.margin <= target_score:
                continue
            gate = self.pressure_gate(wrong, context, pressure)
            if gate <= 0.0:
                continue
            step = self.lr * gate
            self.plastic_branch_weights[target] = np.clip(
                self.plastic_branch_weights[target] + step * supports[:, target],
                -self.clip,
                self.clip,
            )
            self.plastic_branch_weights[wrong] = np.clip(
                self.plastic_branch_weights[wrong] - (step / k) * supports[:, wrong],
                -self.clip,
                self.clip,
            )

    def update(self, context: np.ndarray, target: int) -> None:
        supports = self.branch_supports(context)
        pre_scores = (self.base.scores(context) + self.strength * self.plastic_signal(supports)).astype(np.float32)
        self.learn_plastic(pre_scores, supports, target, context)
        self.base.update(context, target)

    def state_bytes(self) -> int:
        return int(self.base.state_bytes() + self.plastic_branch_weights.nbytes)

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())

    def state_config_metadata(self) -> dict[str, Any]:
        branch_model = self.find_branch_model()
        branch_orders = list(getattr(branch_model, "branch_orders", [])) if branch_model is not None else []
        return {
            "class": self.__class__.__name__,
            "strength": float(self.strength),
            "lr": float(self.lr),
            "decay": float(self.decay),
            "clip": float(self.clip),
            "support_clip": float(self.support_clip),
            "top_k": int(self.top_k),
            "margin": float(self.margin),
            "pressure_mode": self.pressure_mode,
            "pressure_threshold": float(self.pressure_threshold),
            "pressure_gain": float(self.pressure_gain),
            "loop_window": int(self.loop_window),
            "branch_count": int(self.branch_count),
            "branch_orders": [int(order) for order in branch_orders],
        }


class LowPrecisionStateWrapper:
    """
    Post-update low-precision projection for hardware-oriented audits.

    The wrapper quantizes learned NumPy state in-place after each local update.
    It is only an audit/simulation layer; it does not introduce replay, BP, or
    statistical context storage.
    """

    TARGET_GROUPS = {
        "plastic": {
            "code_banks",
            "prototypes",
            "prototype_counts",
            "unigram_counts",
            "output_bias",
            "weights",
            "transition",
            "inhibition",
        },
        "readout": {"weights", "output_bias", "inhibition"},
        "phase": {"code_banks", "prototypes", "prototype_counts", "unigram_counts", "output_bias"},
        "bias": {"output_bias"},
        "counts": {"prototype_counts", "unigram_counts"},
        "phase_codes": {"code_banks"},
        "phase_prototypes": {"prototypes"},
        "readout_weights": {"weights"},
        "inhibition": {"inhibition"},
        "transition": {"transition", "eligibility"},
        "phase_banks": {"code_banks", "prototypes", "prototype_counts", "unigram_counts"},
        "fixed": {
            "trace_codes",
            "context_codes",
            "input_codes",
            "target_codes",
            "target_anchors",
            "apical_feedback",
            "apical_fixed_gate",
            "calibration_codes",
            "gain_codes",
            "branch_state_codes",
            "branch_state_encoder",
        },
        "dynamic": {
            "activity",
            "fatigue",
            "dynamic_gate",
            "apical_error_trace",
            "eligibility",
            "excitability",
            "calibration",
            "calibration_gate",
            "gain_weights",
            "gain_gate",
            "plastic_branch_weights",
            "loop_activity",
            "loop_pressure",
            "loop_prev_output",
            "loop_transition_pressure",
            "segment_state",
            "segment_slots",
            "segment_slot_age",
            "segment_slot_filled",
            "segment_output_trace",
            "segment_attractor_pressure",
            "segment_step",
            "segment_slot_index",
            "loop_escape_weights",
            "branch_state",
            "branch_state_projection",
            "branch_state_novelty_slots",
            "branch_state_novelty_filled",
            "branch_state_novelty_index",
            "branch_state_anti_pressure",
            "branch_state_anti_vector",
        },
    }

    def __init__(self, base: Any, bits: int, clip: float, scale_mode: str, targets: Any, bias_clip: float) -> None:
        self.base = base
        self.bits = min(max(int(bits), 2), 32)
        self.clip = max(float(clip), 1e-6)
        self.bias_clip = max(float(bias_clip), 0.0)
        self.scale_mode = str(scale_mode)
        if self.scale_mode not in {"fixed", "tensor", "row"}:
            raise ValueError(f"unknown low-precision scale mode: {self.scale_mode}")
        if isinstance(targets, (list, tuple)):
            target_text = " ".join(str(target) for target in targets)
        else:
            target_text = str(targets)
        self.targets = {part.strip() for part in target_text.replace(",", " ").split() if part.strip()}
        if not self.targets:
            self.targets = {"all"}
        allowed_targets = {"all", "none", *self.TARGET_GROUPS}
        unknown = self.targets - allowed_targets
        if unknown:
            raise ValueError(f"unknown low-precision target groups: {sorted(unknown)}")
        if "all" in self.targets and len(self.targets) > 1:
            raise ValueError("--low-precision-targets=all cannot be combined with other groups")
        if "none" in self.targets and len(self.targets) > 1:
            raise ValueError("--low-precision-targets=none cannot be combined with other groups")
        self.max_order = int(base.max_order)
        self.vocab_size = int(base.vocab_size)
        self._array_refs = self.collect_array_refs(self.base, set())
        self.project_state()

    def derived_state_names_for_owner(self, owner: Any) -> set[str]:
        names = getattr(owner, "derived_state_names", set())
        return {str(name) for name in names}

    def is_derived_ref(self, ref: tuple[Any, str, Any, str, bool]) -> bool:
        owner, _kind, _key, name, _selected = ref
        return name in self.derived_state_names_for_owner(owner)

    def collect_state_config_metadata(self, obj: Any, seen: set[int]) -> list[dict[str, Any]]:
        obj_id = id(obj)
        if obj_id in seen:
            return []
        seen.add(obj_id)
        rows: list[dict[str, Any]] = []
        if hasattr(obj, "state_config_metadata"):
            rows.append(obj.state_config_metadata())
        if isinstance(obj, dict):
            return rows
        if isinstance(obj, (str, bytes, int, float, bool, type(None), Path, np.ndarray)):
            return rows
        if isinstance(obj, (list, tuple)):
            for item in obj:
                rows.extend(self.collect_state_config_metadata(item, seen))
            return rows
        if hasattr(obj, "__dict__"):
            for name, value in vars(obj).items():
                if name in {"rng"}:
                    continue
                if isinstance(value, np.ndarray):
                    continue
                if hasattr(value, "__dict__") or isinstance(value, (list, tuple)):
                    rows.extend(self.collect_state_config_metadata(value, seen))
        return rows

    def state_config_metadata(self) -> list[dict[str, Any]]:
        return self.collect_state_config_metadata(self.base, set())

    @staticmethod
    def canonical_json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def config_signature(self, entries: list[dict[str, Any]]) -> str:
        signature_payload = {
            "format": "phase_binding_low_precision_state_npz_v1",
            "bits": int(self.bits),
            "clip": float(self.clip),
            "bias_clip": float(self.bias_clip),
            "scale_mode": self.scale_mode,
            "targets": sorted(self.targets),
            "derived_state_names": sorted(
                {
                    name
                    for ref in self.unique_array_refs()
                    for name in self.derived_state_names_for_owner(ref[0])
                }
            ),
            "state_config": self.state_config_metadata(),
            "entries": [
                {
                    "name": entry["name"],
                    "shape": entry["shape"],
                    "dtype": entry["dtype"],
                    "selected": bool(entry["selected"]),
                    "encoding": entry["encoding"],
                    "bits": entry.get("bits"),
                    "code_dtype": entry.get("code_dtype"),
                    "scale_shape": entry.get("scale_shape"),
                }
                for entry in entries
            ],
        }
        return hashlib.sha256(self.canonical_json(signature_payload).encode("utf-8")).hexdigest()

    def should_quantize_name(self, name: str) -> bool:
        if "all" in self.targets:
            return True
        if "none" in self.targets:
            return False
        return any(name in self.TARGET_GROUPS[group] for group in self.targets)

    def collect_array_refs(
        self,
        obj: Any,
        seen: set[int],
        container_name: str = "",
        owner: Any | None = None,
        owner_kind: str = "",
        owner_key: Any = None,
    ) -> list[tuple[Any, str, Any, str, bool]]:
        obj_id = id(obj)
        if obj_id in seen:
            return []
        seen.add(obj_id)
        refs: list[tuple[Any, str, Any, str, bool]] = []
        if isinstance(obj, np.ndarray):
            if owner is not None and container_name:
                refs.append((owner, owner_kind, owner_key, container_name, self.should_quantize_name(container_name)))
            return refs
        if isinstance(obj, (str, bytes, int, float, bool, type(None), Path, dict)):
            return refs
        if isinstance(obj, list):
            for idx, item in enumerate(obj):
                refs.extend(
                    self.collect_array_refs(
                        item,
                        seen,
                        container_name=container_name,
                        owner=obj,
                        owner_kind="index",
                        owner_key=idx,
                    )
                )
            return refs
        if isinstance(obj, tuple):
            for idx, item in enumerate(obj):
                refs.extend(
                    self.collect_array_refs(
                        item,
                        seen,
                        container_name=container_name,
                        owner=obj,
                        owner_kind="index",
                        owner_key=idx,
                    )
                )
            return refs
        if hasattr(obj, "__dict__"):
            for name, value in vars(obj).items():
                if name in {"memory", "rows", "unigram"}:
                    continue
                if isinstance(value, np.ndarray):
                    refs.append((obj, "attr", name, name, self.should_quantize_name(name)))
                elif isinstance(value, (list, tuple)):
                    refs.extend(
                        self.collect_array_refs(
                            value,
                            seen,
                            container_name=name,
                            owner=obj,
                            owner_kind="attr",
                            owner_key=name,
                        )
                    )
                elif hasattr(value, "__dict__"):
                    refs.extend(self.collect_array_refs(value, seen))
        return refs

    def resolve_array_ref(self, ref: tuple[Any, str, Any, str, bool]) -> np.ndarray:
        owner, kind, key, _name, _selected = ref
        if kind == "attr":
            return getattr(owner, key)
        if kind == "index":
            return owner[key]
        raise ValueError(f"unknown array ref kind: {kind}")

    def unique_array_refs(self) -> list[tuple[Any, str, Any, str, bool]]:
        refs: list[tuple[Any, str, Any, str, bool]] = []
        seen: set[int] = set()
        for ref in self._array_refs:
            value = self.resolve_array_ref(ref)
            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)
            refs.append(ref)
        return refs

    def quantization_scale(self, value: np.ndarray, name: str = "") -> np.ndarray:
        clip = self.bias_clip if name == "output_bias" and self.bias_clip > 0.0 else self.clip
        if self.scale_mode == "row" and value.ndim == 2:
            scale = np.maximum(np.max(np.abs(value), axis=1, keepdims=True), 1e-6)
            return np.minimum(scale, clip).astype(np.float32, copy=False)
        if self.scale_mode in {"tensor", "row"}:
            scale = min(max(float(np.max(np.abs(value))), 1e-6), clip)
        else:
            scale = clip
        return np.array(scale, dtype=np.float32)

    def code_dtype(self) -> np.dtype[Any]:
        if self.bits <= 8:
            return np.dtype(np.uint8)
        if self.bits <= 16:
            return np.dtype(np.uint16)
        return np.dtype(np.uint32)

    def quantize_codes(self, value: np.ndarray, name: str = "") -> tuple[np.ndarray, np.ndarray]:
        scale = self.quantization_scale(value, name)
        levels = (1 << self.bits) - 1
        clipped = np.clip(value, -scale, scale)
        codes = np.round((clipped + scale) * levels / (2.0 * scale)).astype(self.code_dtype(), copy=False)
        return codes, scale.astype(np.float32, copy=False)

    @staticmethod
    def dequantize_codes(codes: np.ndarray, scale: np.ndarray, bits: int, dtype: np.dtype[Any]) -> np.ndarray:
        levels = (1 << int(bits)) - 1
        values = codes.astype(np.float32) * (2.0 * scale.astype(np.float32) / levels) - scale.astype(np.float32)
        return values.astype(dtype, copy=False)

    def quantize_array(self, value: np.ndarray, name: str = "") -> np.ndarray:
        if not np.issubdtype(value.dtype, np.floating):
            return value
        codes, scale = self.quantize_codes(value, name)
        return self.dequantize_codes(codes, scale, self.bits, value.dtype)

    def project_state(self) -> None:
        for owner, kind, key, name, selected in self._array_refs:
            if not selected:
                continue
            value = self.resolve_array_ref((owner, kind, key, name, selected))
            value[...] = self.quantize_array(value, name)

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()
            self.project_state()

    def observe_prompt(self, token: int) -> None:
        if hasattr(self.base, "observe_prompt"):
            self.base.observe_prompt(token)
            self.project_state()

    def observe_prediction(self, token: int) -> None:
        if hasattr(self.base, "observe_prediction"):
            self.base.observe_prediction(token)
            self.project_state()

    def observe(self, context: np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)
            self.project_state()

    def scores(self, context: np.ndarray) -> np.ndarray:
        return self.base.scores(context)

    def update(self, context: np.ndarray, target: int) -> None:
        self.base.update(context, target)
        self.project_state()

    def state_bytes(self) -> int:
        total = 0
        for ref in self._array_refs:
            if self.is_derived_ref(ref):
                continue
            owner, kind, key, name, selected = ref
            value = self.resolve_array_ref(ref)
            if np.issubdtype(value.dtype, np.floating) and selected:
                total += int(math.ceil(value.nbytes * self.bits / 32.0))
            else:
                total += int(value.nbytes)
        return int(total)

    def serialized_array_bytes(self, value: np.ndarray, name: str, selected: bool) -> int:
        if not np.issubdtype(value.dtype, np.floating) or not selected:
            return int(value.nbytes)
        data_bytes = int(value.size * math.ceil(self.bits / 8.0))
        scale_count = int(value.shape[0]) if self.scale_mode == "row" and value.ndim == 2 else 1
        return int(data_bytes + scale_count * np.dtype(np.float32).itemsize)

    def serialized_state_bytes(self) -> int:
        total = 0
        for ref in self.unique_array_refs():
            if self.is_derived_ref(ref):
                continue
            owner, kind, key, name, selected = ref
            value = self.resolve_array_ref(ref)
            total += self.serialized_array_bytes(value, name, selected)
        return int(total)

    def serialized_state_manifest(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for ref in self.unique_array_refs():
            if self.is_derived_ref(ref):
                continue
            owner, kind, key, name, selected = ref
            value = self.resolve_array_ref(ref)
            rows.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "selected": bool(selected),
                    "serialized_bytes": self.serialized_array_bytes(value, name, selected),
                    "raw_bytes": int(value.nbytes),
                }
            )
        return rows

    def save_serialized_state(self, path: Path) -> dict[str, Any]:
        arrays: dict[str, np.ndarray] = {}
        entries: list[dict[str, Any]] = []
        quantized_arrays = 0
        raw_arrays = 0
        for idx, ref in enumerate(self.unique_array_refs()):
            if self.is_derived_ref(ref):
                continue
            value = self.resolve_array_ref(ref)
            _owner, _kind, _key, name, selected = ref
            data_key = f"arr_{idx:04d}"
            entry: dict[str, Any] = {
                "index": idx,
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "selected": bool(selected),
                "data_key": data_key,
                "raw_bytes": int(value.nbytes),
            }
            if np.issubdtype(value.dtype, np.floating) and selected:
                codes, scale = self.quantize_codes(value, name)
                scale_key = f"{data_key}_scale"
                arrays[data_key] = codes
                arrays[scale_key] = scale.astype(np.float32, copy=False)
                entry.update(
                    {
                        "encoding": "uniform_symmetric",
                        "bits": int(self.bits),
                        "code_dtype": str(codes.dtype),
                        "scale_key": scale_key,
                        "scale_shape": list(scale.shape),
                        "serialized_bytes": self.serialized_array_bytes(value, name, selected),
                    }
                )
                quantized_arrays += 1
            else:
                arrays[data_key] = value.copy()
                entry.update(
                    {
                        "encoding": "raw",
                        "serialized_bytes": int(value.nbytes),
                    }
                )
                raw_arrays += 1
            entries.append(entry)
        metadata = {
            "format": "phase_binding_low_precision_state_npz_v1",
            "bits": int(self.bits),
            "clip": float(self.clip),
            "bias_clip": float(self.bias_clip),
            "scale_mode": self.scale_mode,
            "targets": sorted(self.targets),
            "derived_state_names": sorted(
                {
                    name
                    for ref in self.unique_array_refs()
                    for name in self.derived_state_names_for_owner(ref[0])
                }
            ),
            "state_config": self.state_config_metadata(),
            "entry_count": len(entries),
            "quantized_arrays": quantized_arrays,
            "raw_arrays": raw_arrays,
            "serialized_state_bytes": self.serialized_state_bytes(),
            "entries": entries,
        }
        metadata["config_signature"] = self.config_signature(entries)
        arrays["metadata_json"] = np.array(json.dumps(metadata), dtype=np.str_)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        metadata["checkpoint_file_bytes"] = int(path.stat().st_size)
        return metadata

    def load_serialized_state(self, path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata_json"].item()))
            if metadata.get("format") != "phase_binding_low_precision_state_npz_v1":
                raise ValueError(f"unknown serialized state format: {metadata.get('format')}")
            entries = list(metadata.get("entries", []))
            refs = self.unique_array_refs()
            refs = [ref for ref in refs if not self.is_derived_ref(ref)]
            if len(entries) != len(refs):
                raise ValueError(f"checkpoint entry count {len(entries)} != model array count {len(refs)}")
            expected_signature = metadata.get("config_signature")
            actual_signature = self.config_signature(entries)
            if expected_signature and expected_signature != actual_signature:
                raise ValueError(
                    "checkpoint config signature mismatch: "
                    f"expected {expected_signature}, got {actual_signature}"
                )
            for entry, ref in zip(entries, refs):
                value = self.resolve_array_ref(ref)
                _owner, _kind, _key, name, _selected = ref
                expected_shape = tuple(int(dim) for dim in entry["shape"])
                if name != entry["name"] or value.shape != expected_shape or str(value.dtype) != entry["dtype"]:
                    raise ValueError(
                        "checkpoint array mismatch: "
                        f"expected {entry['name']} {expected_shape} {entry['dtype']}, "
                        f"got {name} {value.shape} {value.dtype}"
                    )
                if entry["encoding"] == "uniform_symmetric":
                    codes = arrays[entry["data_key"]]
                    scale = arrays[entry["scale_key"]]
                    value[...] = self.dequantize_codes(codes, scale, int(entry["bits"]), value.dtype)
                elif entry["encoding"] == "raw":
                    value[...] = arrays[entry["data_key"]].astype(value.dtype, copy=False)
                else:
                    raise ValueError(f"unknown checkpoint encoding: {entry['encoding']}")
        return metadata

    def active_contexts(self) -> int:
        return int(self.base.active_contexts())


def serialized_state_bytes(obj: Any) -> int:
    if hasattr(obj, "serialized_state_bytes"):
        return int(obj.serialized_state_bytes())
    return int(obj.state_bytes())


def serialized_state_manifest(obj: Any) -> list[dict[str, Any]]:
    if hasattr(obj, "serialized_state_manifest"):
        return list(obj.serialized_state_manifest())
    return []


def save_serialized_state(obj: Any, path: Path) -> dict[str, Any] | None:
    if hasattr(obj, "save_serialized_state"):
        return obj.save_serialized_state(path)
    return None


def load_serialized_state(obj: Any, path: Path) -> dict[str, Any] | None:
    if hasattr(obj, "load_serialized_state"):
        return obj.load_serialized_state(path)
    return None


def checkpoint_prediction_parity(
    reference: Any,
    loaded: Any,
    ids: np.ndarray,
    segment_tokens: int,
    start_history: Sequence[int],
    temperature: float,
    limit: int,
) -> dict[str, Any]:
    order = int(reference.max_order)
    if int(loaded.max_order) != order:
        raise ValueError(f"loaded max_order {loaded.max_order} != reference max_order {order}")
    history = truncate_history(start_history, order - 1)
    max_abs_score_diff = 0.0
    score_abs_sum = 0.0
    score_value_count = 0
    pred_matches = 0
    contexts = 0
    ref_loss_sum = 0.0
    loaded_loss_sum = 0.0
    loss_abs_sum = 0.0
    for start, end in segment_windows(ids, segment_tokens):
        segment = ids[start:end]
        segment_history = list(history)
        for idx in range(len(segment) - 1):
            current = int(segment[idx])
            target = int(segment[idx + 1])
            context_list = truncate_history(segment_history + [current], order)
            if len(context_list) < order:
                segment_history = truncate_history(segment_history + [current], order - 1)
                continue
            context = np.array(context_list, dtype=np.int64)
            ref_scores = reference.scores(context)
            loaded_scores = loaded.scores(context)
            abs_diff = np.abs(ref_scores.astype(np.float32) - loaded_scores.astype(np.float32))
            max_abs_score_diff = max(max_abs_score_diff, float(np.max(abs_diff)))
            score_abs_sum += float(np.sum(abs_diff))
            score_value_count += int(abs_diff.size)
            ref_loss, ref_pred = softmax_loss_and_pred(ref_scores, target, temperature)
            loaded_loss, loaded_pred = softmax_loss_and_pred(loaded_scores, target, temperature)
            pred_matches += int(ref_pred == loaded_pred)
            ref_loss_sum += ref_loss
            loaded_loss_sum += loaded_loss
            loss_abs_sum += abs(ref_loss - loaded_loss)
            contexts += 1
            if hasattr(reference, "observe"):
                reference.observe(context, target)
            if hasattr(loaded, "observe"):
                loaded.observe(context, target)
            segment_history = truncate_history(segment_history + [current], order - 1)
            if limit > 0 and contexts >= limit:
                break
        history = segment_history
        if limit > 0 and contexts >= limit:
            break
    return {
        "checkpoint_contexts": contexts,
        "checkpoint_pred_match": pred_matches / max(contexts, 1),
        "checkpoint_max_abs_score_diff": max_abs_score_diff,
        "checkpoint_mean_abs_score_diff": score_abs_sum / max(score_value_count, 1),
        "checkpoint_ref_loss": ref_loss_sum / max(contexts, 1),
        "checkpoint_loaded_loss": loaded_loss_sum / max(contexts, 1),
        "checkpoint_loss_diff": (loaded_loss_sum - ref_loss_sum) / max(contexts, 1),
        "checkpoint_mean_abs_loss_diff": loss_abs_sum / max(contexts, 1),
    }


def run_stream_pass(
    method: str,
    memory: Any,
    ids: np.ndarray,
    segment_tokens: int,
    start_history: Sequence[int],
    update: bool,
    pass_name: str,
    temperature: float,
) -> tuple[dict[str, Any], list[int], list[dict[str, Any]]]:
    order = int(memory.max_order)
    history = truncate_history(start_history, order - 1)
    rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    correct = 0
    total = 0
    processed = 0
    candidate_ks = (2, 4, 8)
    rank_sum = 0.0
    margin_sum = 0.0
    wrong_margin_sum = 0.0
    top_hits = {k: 0 for k in candidate_ks}
    error_top_hits = {k: 0 for k in candidate_ks}
    error_count = 0
    for segment_idx, (start, end) in enumerate(segment_windows(ids, segment_tokens)):
        segment = ids[start:end]
        segment_history = list(history)
        seg_loss = 0.0
        seg_correct = 0
        seg_total = 0
        seg_rank_sum = 0.0
        seg_margin_sum = 0.0
        seg_wrong_margin_sum = 0.0
        seg_top_hits = {k: 0 for k in candidate_ks}
        seg_error_top_hits = {k: 0 for k in candidate_ks}
        seg_error_count = 0
        for idx in range(len(segment) - 1):
            current = int(segment[idx])
            target = int(segment[idx + 1])
            context_list = truncate_history(segment_history + [current], order)
            if len(context_list) < order:
                segment_history = truncate_history(segment_history + [current], order - 1)
                continue
            context = np.array(context_list, dtype=np.int64)
            scores = memory.scores(context)
            loss, pred = softmax_loss_and_pred(scores, target, temperature)
            target_score = float(scores[target])
            adjusted = scores.astype(np.float32, copy=True)
            adjusted[target] = -np.inf
            best_wrong_score = float(np.max(adjusted)) if adjusted.size > 1 else -np.inf
            rank = 1 + int(np.sum(scores.astype(np.float32, copy=False) > target_score))
            target_margin = target_score - best_wrong_score
            seg_loss += loss
            is_correct = int(pred == target)
            seg_correct += is_correct
            seg_total += 1
            seg_rank_sum += rank
            seg_margin_sum += target_margin
            for k in candidate_ks:
                hit = int(rank <= min(k, scores.size))
                seg_top_hits[k] += hit
            if not is_correct:
                seg_error_count += 1
                seg_wrong_margin_sum += best_wrong_score - target_score
                for k in candidate_ks:
                    seg_error_top_hits[k] += int(rank <= min(k, scores.size))
            if update:
                memory.update(context, target)
            elif hasattr(memory, "observe"):
                memory.observe(context, target)
            segment_history = truncate_history(segment_history + [current], order - 1)
        history = segment_history
        loss_sum += seg_loss
        correct += seg_correct
        total += seg_total
        rank_sum += seg_rank_sum
        margin_sum += seg_margin_sum
        wrong_margin_sum += seg_wrong_margin_sum
        error_count += seg_error_count
        for k in candidate_ks:
            top_hits[k] += seg_top_hits[k]
            error_top_hits[k] += seg_error_top_hits[k]
        processed += seg_total
        seg_summary = summarize(seg_loss, seg_correct, seg_total)
        seg_summary.update(
            summarize_candidate_ranks(
                seg_rank_sum,
                seg_margin_sum,
                seg_wrong_margin_sum,
                seg_top_hits,
                seg_error_top_hits,
                seg_error_count,
                seg_total,
            )
        )
        state_bytes = memory.state_bytes()
        rows.append(
            {
                "method": method,
                "pass": pass_name,
                "segment_idx": segment_idx,
                "segment_start": start,
                "segment_end": end,
                "target_tokens": seg_total,
                "loss": seg_summary["loss"],
                "ppl": seg_summary["ppl"],
                "accuracy": seg_summary["accuracy"],
                "target_rank_mean": seg_summary["target_rank_mean"],
                "target_margin_mean": seg_summary["target_margin_mean"],
                "error_count": seg_summary["error_count"],
                "error_wrong_margin_mean": seg_summary["error_wrong_margin_mean"],
                "target_top2_acc": seg_summary["target_top2_acc"],
                "target_top4_acc": seg_summary["target_top4_acc"],
                "target_top8_acc": seg_summary["target_top8_acc"],
                "error_target_top2_rate": seg_summary["error_target_top2_rate"],
                "error_target_top4_rate": seg_summary["error_target_top4_rate"],
                "error_target_top8_rate": seg_summary["error_target_top8_rate"],
                "oracle_top2_acc": seg_summary["oracle_top2_acc"],
                "oracle_top4_acc": seg_summary["oracle_top4_acc"],
                "oracle_top8_acc": seg_summary["oracle_top8_acc"],
                "state_bytes": state_bytes,
                "bytes_per_target": state_bytes / max(processed, 1),
                "active_contexts": memory.active_contexts(),
                "updated": update,
                "stores_raw_text": False,
            }
        )
    summary = summarize(loss_sum, correct, total)
    summary.update(
        {
            "method": method,
            "pass": pass_name,
            "target_tokens": total,
            "segment_count": len(rows),
            "state_bytes": memory.state_bytes(),
            "bytes_per_target": memory.state_bytes() / max(total, 1),
            "active_contexts": memory.active_contexts(),
            "history_tokens": len(history),
            "stores_raw_text": False,
        }
    )
    summary.update(
        summarize_candidate_ranks(
            rank_sum,
            margin_sum,
            wrong_margin_sum,
            top_hits,
            error_top_hits,
            error_count,
            total,
        )
    )
    return summary, history, rows


def evaluate_sequence(memory: Any, ids: np.ndarray, start_history: Sequence[int], temperature: float) -> dict[str, Any]:
    summary, _, _ = run_stream_pass(
        method="retention_eval",
        memory=memory,
        ids=ids,
        segment_tokens=max(len(ids), 2),
        start_history=start_history,
        update=False,
        pass_name="retention",
        temperature=temperature,
    )
    return summary


def state_pickle_bytes(obj: Any) -> int:
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def clone_memory(memory: Any) -> Any:
    return copy.deepcopy(memory)


def clone_for_eval(memory: Any, reset_dynamic: bool) -> Any:
    cloned = clone_memory(memory)
    if reset_dynamic and hasattr(cloned, "reset_dynamic_state"):
        cloned.reset_dynamic_state()
    return cloned


def decode_compact_ids(tokenizer: Any, kept_raw: np.ndarray, ids: Sequence[int]) -> str:
    raw_ids = [int(kept_raw[int(idx)]) for idx in ids if 0 <= int(idx) < len(kept_raw)]
    return tokenizer.decode(raw_ids, skip_special_tokens=True)


def make_generation_prompts(
    ids: np.ndarray,
    prompt_count: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> list[dict[str, Any]]:
    if prompt_count <= 0 or len(ids) < 2:
        return []
    prompt_tokens = max(int(prompt_tokens), 1)
    completion_tokens = max(int(completion_tokens), 1)
    max_start = max(len(ids) - prompt_tokens - completion_tokens, 0)
    if prompt_count == 1:
        starts = [0]
    else:
        starts = sorted({int(round(x)) for x in np.linspace(0, max_start, prompt_count)})
    prompts: list[dict[str, Any]] = []
    for prompt_idx, start in enumerate(starts):
        prompt_end = min(start + prompt_tokens, len(ids) - 1)
        ref_end = min(prompt_end + completion_tokens, len(ids))
        if prompt_end <= start or ref_end <= prompt_end:
            continue
        prompts.append(
            {
                "prompt_idx": prompt_idx,
                "start": int(start),
                "prompt_ids": [int(token) for token in ids[start:prompt_end]],
                "reference_ids": [int(token) for token in ids[prompt_end:ref_end]],
            }
        )
    return prompts


def apply_generation_controls(
    scores: np.ndarray,
    generated: Sequence[int],
    repetition_penalty: float,
    no_repeat_ngram: int,
) -> np.ndarray:
    adjusted = scores.astype(np.float32, copy=True)
    if repetition_penalty > 0.0:
        counts: dict[int, int] = {}
        for token in generated[-32:]:
            counts[int(token)] = counts.get(int(token), 0) + 1
        for token, count in counts.items():
            adjusted[token] -= repetition_penalty * float(count)
    if no_repeat_ngram > 1 and len(generated) >= no_repeat_ngram - 1:
        prefix = tuple(int(token) for token in generated[-(no_repeat_ngram - 1) :])
        banned: set[int] = set()
        for idx in range(len(generated) - no_repeat_ngram + 1):
            ngram = tuple(int(token) for token in generated[idx : idx + no_repeat_ngram])
            if ngram[:-1] == prefix:
                banned.add(ngram[-1])
        for token in banned:
            adjusted[token] = -1e9
    return adjusted


def generate_compact_ids(
    memory: Any,
    prompt_ids: Sequence[int],
    completion_tokens: int,
    repetition_penalty: float = 0.0,
    no_repeat_ngram: int = 0,
) -> list[int]:
    generated = [int(token) for token in prompt_ids]
    order = int(memory.max_order)
    if len(generated) < order:
        return []
    if hasattr(memory, "reset_dynamic_state"):
        memory.reset_dynamic_state()
    if hasattr(memory, "observe_prompt"):
        for token in generated:
            memory.observe_prompt(int(token))
    new_tokens: list[int] = []
    for _ in range(max(int(completion_tokens), 0)):
        context = np.array(generated[-order:], dtype=np.int64)
        scores = memory.scores(context)
        if repetition_penalty > 0.0 or no_repeat_ngram > 1:
            scores = apply_generation_controls(scores, generated, repetition_penalty, no_repeat_ngram)
        next_token = int(np.argmax(scores))
        generated.append(next_token)
        new_tokens.append(next_token)
        if hasattr(memory, "observe_prediction"):
            memory.observe_prediction(next_token)
    return new_tokens


def ngram_stats(tokens: Sequence[int], n: int) -> tuple[float, float]:
    if len(tokens) < n or n <= 0:
        return 0.0, 0.0
    ngrams = [tuple(int(token) for token in tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return unique / max(total, 1), (total - unique) / max(total, 1)


def max_token_fraction(tokens: Sequence[int]) -> float:
    if not tokens:
        return 0.0
    counts: dict[int, int] = {}
    for token in tokens:
        counts[int(token)] = counts.get(int(token), 0) + 1
    return max(counts.values()) / len(tokens)


def max_run_fraction(tokens: Sequence[int]) -> float:
    if not tokens:
        return 0.0
    best = 1
    current = 1
    previous = int(tokens[0])
    for token in tokens[1:]:
        token = int(token)
        if token == previous:
            current += 1
        else:
            best = max(best, current)
            current = 1
            previous = token
    best = max(best, current)
    return best / len(tokens)


def loop_pressure_stats(tokens: Sequence[int], window: int = 8) -> tuple[float, float]:
    if len(tokens) < 2:
        return 0.0, 0.0
    window = max(int(window), 1)
    pressures: list[float] = []
    values = [int(token) for token in tokens]
    for end in range(2, len(values) + 1):
        recent = values[max(0, end - window) : end]
        counts: dict[int, int] = {}
        for token in recent:
            counts[token] = counts.get(token, 0) + 1
        max_fraction = max(counts.values()) / len(recent)
        adjacent = sum(int(recent[idx] == recent[idx - 1]) for idx in range(1, len(recent)))
        adjacent_fraction = adjacent / max(len(recent) - 1, 1)
        pressures.append(max(max_fraction, adjacent_fraction))
    return float(np.mean(pressures)), float(np.max(pressures))


def prefix_match_tokens(generated: Sequence[int], reference: Sequence[int]) -> int:
    matched = 0
    for pred, target in zip(generated, reference):
        if int(pred) != int(target):
            break
        matched += 1
    return matched


def build_generation_rows(
    method: str,
    stage: str,
    decode_mode: str,
    memory: Any,
    tokenizer: Any,
    kept_raw: np.ndarray,
    prompts: list[dict[str, Any]],
    completion_tokens: int,
    repetition_penalty: float = 0.0,
    no_repeat_ngram: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_ids = prompt["prompt_ids"]
        reference_ids = prompt["reference_ids"]
        prompt_memory = clone_memory(memory)
        generated_ids = generate_compact_ids(
            prompt_memory,
            prompt_ids,
            completion_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram=no_repeat_ngram,
        )
        distinct_1, repeat_1 = ngram_stats(generated_ids, 1)
        distinct_2, repeat_2 = ngram_stats(generated_ids, 2)
        loop_pressure_mean, loop_pressure_max = loop_pressure_stats(generated_ids)
        prefix_match = prefix_match_tokens(generated_ids, reference_ids)
        rows.append(
            {
                "method": method,
                "stage": stage,
                "decode_mode": decode_mode,
                "prompt_idx": prompt["prompt_idx"],
                "prompt_start": prompt["start"],
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(generated_ids),
                "reference_tokens": len(reference_ids),
                "first_token_match": int(bool(generated_ids and reference_ids and generated_ids[0] == reference_ids[0])),
                "prefix_match_tokens": prefix_match,
                "prefix_match_fraction": prefix_match / max(min(len(generated_ids), len(reference_ids)), 1),
                "distinct_1": distinct_1,
                "repeat_1": repeat_1,
                "distinct_2": distinct_2,
                "repeat_2": repeat_2,
                "max_token_fraction": max_token_fraction(generated_ids),
                "max_run_fraction": max_run_fraction(generated_ids),
                "loop_pressure_mean": loop_pressure_mean,
                "loop_pressure_max": loop_pressure_max,
                "repetition_penalty": repetition_penalty,
                "no_repeat_ngram": no_repeat_ngram,
                "model_stores_raw_text": False,
                "artifact_contains_decoded_text": True,
                "prompt_text": decode_compact_ids(tokenizer, kept_raw, prompt_ids),
                "generated_text": decode_compact_ids(tokenizer, kept_raw, prompt_ids + generated_ids),
                "reference_text": decode_compact_ids(tokenizer, kept_raw, prompt_ids + reference_ids),
            }
        )
    return rows


def summarize_generation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["stage"]), str(row["decode_mode"])), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    numeric_fields = [
        "first_token_match",
        "prefix_match_tokens",
        "prefix_match_fraction",
        "distinct_1",
        "distinct_2",
        "repeat_1",
        "repeat_2",
        "max_token_fraction",
        "max_run_fraction",
        "loop_pressure_mean",
        "loop_pressure_max",
    ]
    for (method, stage, decode_mode), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "method": method,
            "stage": stage,
            "decode_mode": decode_mode,
            "prompt_count": len(group),
        }
        for field in numeric_fields:
            summary[field] = float(np.mean([float(row[field]) for row in group])) if group else float("nan")
        summary["model_stores_raw_text"] = False
        summary["artifact_contains_decoded_text"] = False
        summary_rows.append(summary)
    return summary_rows


def write_generation_text(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                f"\n## {row['method']} {row['stage']} {row['decode_mode']} prompt={row['prompt_idx']} "
                f"start={row['prompt_start']}\n"
            )
            f.write("\n[PROMPT]\n")
            f.write(str(row["prompt_text"]).strip() + "\n")
            f.write("\n[GENERATED]\n")
            f.write(str(row["generated_text"]).strip() + "\n")
            f.write("\n[REFERENCE]\n")
            f.write(str(row["reference_text"]).strip() + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "phase_binding_online_stream")
    parser.add_argument("--train-chars", type=int, default=50_000)
    parser.add_argument("--valid-chars", type=int, default=10_000)
    parser.add_argument("--max-vocab", type=int, default=256)
    parser.add_argument("--warmup-token-limit", type=int, default=0)
    parser.add_argument("--stream-token-limit", type=int, default=0)
    parser.add_argument("--retention-token-limit", type=int, default=1024)
    parser.add_argument("--retention-reset-dynamic", action="store_true")
    parser.add_argument("--segment-tokens", type=int, default=256)
    parser.add_argument("--method-filter", nargs="+", default=[])
    parser.add_argument("--low-precision-bits", type=int, default=0)
    parser.add_argument("--low-precision-clip", type=float, default=1.0)
    parser.add_argument("--low-precision-bias-clip", type=float, default=0.0)
    parser.add_argument("--low-precision-scale-mode", choices=["fixed", "tensor", "row"], default="fixed")
    parser.add_argument(
        "--low-precision-targets",
        nargs="+",
        default=["all"],
        help=(
            "Space/comma-separated groups to quantize: all, plastic, readout, phase, fixed, dynamic, "
            "bias, counts, phase_codes, phase_prototypes, phase_banks, readout_weights, inhibition, transition, none."
        ),
    )
    parser.add_argument("--save-serialized-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-parity-limit", type=int, default=0)
    parser.add_argument("--phase-dim", type=int, default=128)
    parser.add_argument("--phase-lr", type=float, default=0.10)
    parser.add_argument("--phase-logit-scale", type=float, default=10.0)
    parser.add_argument("--phase-bias-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--branch-orders", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--branch-weights", type=float, nargs="+", default=[0.5, 0.5])
    parser.add_argument("--competitive-lr", type=float, default=0.16)
    parser.add_argument("--competitive-neg-k", type=int, default=8)
    parser.add_argument("--competitive-epochs", type=int, default=1)
    parser.add_argument("--competitive-score-scale", type=float, default=10.0)
    parser.add_argument("--competitive-init", choices=["average", "random"], default="average")
    parser.add_argument("--competitive-margin", type=float, default=0.0)
    parser.add_argument("--trace-branch", action="store_true")
    parser.add_argument("--trace-order", type=int, default=16)
    parser.add_argument("--trace-dim", type=int, default=128)
    parser.add_argument("--trace-decay", type=float, default=0.85)
    parser.add_argument("--trace-weight", type=float, default=1.0)
    parser.add_argument("--dll-depth-branch", action="store_true")
    parser.add_argument("--dll-hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--dll-label-dim", type=int, default=128)
    parser.add_argument("--dll-lr", type=float, default=0.02)
    parser.add_argument("--dll-bias-lr", type=float, default=0.002)
    parser.add_argument("--dll-delta-clip", type=float, default=1.0)
    parser.add_argument("--dll-activation", choices=["tanh", "linear"], default="tanh")
    parser.add_argument("--dll-disable-row-normalize", action="store_true")
    parser.add_argument("--noprop-depth-branch", action="store_true")
    parser.add_argument("--noprop-hidden-dims", type=int, nargs="+", default=[512])
    parser.add_argument("--noprop-label-dim", type=int, default=128)
    parser.add_argument("--noprop-alpha-start", type=float, default=0.85)
    parser.add_argument("--noprop-alpha-end", type=float, default=0.35)
    parser.add_argument("--noprop-lr", type=float, default=0.02)
    parser.add_argument("--noprop-denoise-lr", type=float, default=0.02)
    parser.add_argument("--noprop-bias-lr", type=float, default=0.002)
    parser.add_argument("--noprop-delta-clip", type=float, default=1.0)
    parser.add_argument("--noprop-activation", choices=["tanh", "linear"], default="tanh")
    parser.add_argument("--noprop-disable-row-normalize", action="store_true")
    parser.add_argument("--unified-noprop-core", action="store_true")
    parser.add_argument("--unified-calibration-strength", type=float, default=1.0)
    parser.add_argument("--unified-calibration-lr", type=float, default=0.02)
    parser.add_argument("--unified-calibration-clip", type=float, default=2.0)
    parser.add_argument("--unified-calibration-dim", type=int, default=64)
    parser.add_argument("--unified-calibration-gate-decay", type=float, default=0.50)
    parser.add_argument("--unified-calibration-threshold", type=float, default=0.0)
    parser.add_argument("--unified-readout-gain", type=float, default=1.15)
    parser.add_argument("--unified-eligibility-order", type=int, default=4)
    parser.add_argument("--unified-eligibility-decay", type=float, default=0.90)
    parser.add_argument("--unified-eligibility-score-weight", type=float, default=0.0)
    parser.add_argument("--unified-eligibility-top-k", type=int, default=8)
    parser.add_argument("--eprop-trace-readout", action="store_true")
    parser.add_argument("--eprop-order", type=int, default=20)
    parser.add_argument("--eprop-decay", type=float, default=0.95)
    parser.add_argument("--eprop-weight", type=float, default=1.0)
    parser.add_argument("--hebbian-kv-branch", action="store_true")
    parser.add_argument("--kv-order", type=int, default=96)
    parser.add_argument("--kv-dim", type=int, default=64)
    parser.add_argument("--kv-trace-decay", type=float, default=0.95)
    parser.add_argument("--kv-weight", type=float, default=0.50)
    parser.add_argument("--kv-score-weight", type=float, default=0.0)
    parser.add_argument(
        "--kv-gate-mode",
        choices=["none", "norm", "base_low_margin", "kv_margin", "base_and_kv", "base_or_kv"],
        default="none",
    )
    parser.add_argument("--kv-gate-base-margin", type=float, default=0.75)
    parser.add_argument("--kv-gate-kv-margin", type=float, default=0.05)
    parser.add_argument("--kv-gate-min-norm", type=float, default=0.0)
    parser.add_argument("--kv-lr", type=float, default=0.04)
    parser.add_argument("--kv-decay", type=float, default=0.002)
    parser.add_argument("--kv-clip", type=float, default=2.0)
    parser.add_argument("--apical-gating-branch", action="store_true")
    parser.add_argument("--apical-decay", type=float, default=0.90)
    parser.add_argument("--apical-strength", type=float, default=0.75)
    parser.add_argument("--apical-margin", type=float, default=0.0)
    parser.add_argument("--apical-min-gate", type=float, default=0.5)
    parser.add_argument("--apical-max-gate", type=float, default=2.0)
    parser.add_argument("--apical-error-clip", type=float, default=2.0)
    parser.add_argument(
        "--apical-error-mode",
        choices=["segment_margin", "global_margin", "random_feedback", "fixed_random"],
        default="segment_margin",
    )
    parser.add_argument("--plastic-ssm-branch", action="store_true")
    parser.add_argument("--ssm-order", type=int, default=16)
    parser.add_argument("--ssm-dim", type=int, default=64)
    parser.add_argument("--ssm-decay", type=float, default=0.80)
    parser.add_argument("--ssm-recurrent-scale", type=float, default=0.40)
    parser.add_argument("--ssm-weight", type=float, default=0.5)
    parser.add_argument("--ssm-lr", type=float, default=0.01)
    parser.add_argument("--ssm-target-mix", type=float, default=0.25)
    parser.add_argument("--ssm-clip", type=float, default=1.0)
    parser.add_argument("--eligibility-ssm-branch", action="store_true")
    parser.add_argument("--ssm-eligibility-decay", type=float, default=0.90)
    parser.add_argument("--ssm-eligibility-clip", type=float, default=2.0)
    parser.add_argument("--output-fatigue", action="store_true")
    parser.add_argument("--fatigue-strength", type=float, default=0.75)
    parser.add_argument("--fatigue-decay", type=float, default=0.80)
    parser.add_argument("--adaptive-inhibition", action="store_true")
    parser.add_argument("--inhibit-strength", type=float, default=0.35)
    parser.add_argument("--inhibit-decay", type=float, default=0.85)
    parser.add_argument("--inhibit-lr", type=float, default=0.02)
    parser.add_argument("--inhibit-disinhibit-lr", type=float, default=0.004)
    parser.add_argument("--inhibit-top-k", type=int, default=4)
    parser.add_argument("--inhibit-margin", type=float, default=0.0)
    parser.add_argument("--inhibit-max-weight", type=float, default=2.5)
    parser.add_argument("--context-gated-inhibition", action="store_true")
    parser.add_argument("--gate-inhibit-strength", type=float, default=0.35)
    parser.add_argument("--gate-inhibit-lr", type=float, default=0.01)
    parser.add_argument("--gate-inhibit-disinhibit-lr", type=float, default=0.002)
    parser.add_argument("--gate-inhibit-top-k", type=int, default=2)
    parser.add_argument("--gate-inhibit-margin", type=float, default=0.0)
    parser.add_argument("--gate-inhibit-max-weight", type=float, default=2.5)
    parser.add_argument("--gate-dim", type=int, default=64)
    parser.add_argument("--gate-decay", type=float, default=0.0)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--output-homeostasis", action="store_true")
    parser.add_argument("--homeostasis-strength", type=float, default=1.0)
    parser.add_argument("--homeostasis-lr", type=float, default=0.02)
    parser.add_argument("--homeostasis-decay", type=float, default=0.995)
    parser.add_argument("--homeostasis-clip", type=float, default=2.0)
    parser.add_argument("--feature-calibration", action="store_true")
    parser.add_argument("--feature-calibration-strength", type=float, default=1.0)
    parser.add_argument("--feature-calibration-lr", type=float, default=0.02)
    parser.add_argument("--feature-calibration-decay", type=float, default=1.0)
    parser.add_argument("--feature-calibration-clip", type=float, default=2.0)
    parser.add_argument("--feature-calibration-dim", type=int, default=64)
    parser.add_argument("--feature-calibration-gate-decay", type=float, default=0.0)
    parser.add_argument("--feature-calibration-threshold", type=float, default=0.0)
    parser.add_argument("--feature-calibration-derived-codes", action="store_true")
    parser.add_argument("--transient-feature-calibration", action="store_true")
    parser.add_argument("--transient-feature-calibration-strength", type=float, default=0.25)
    parser.add_argument("--transient-feature-calibration-lr", type=float, default=0.005)
    parser.add_argument("--transient-feature-calibration-decay", type=float, default=1.0)
    parser.add_argument("--transient-feature-calibration-clip", type=float, default=1.0)
    parser.add_argument("--transient-feature-calibration-dim", type=int, default=32)
    parser.add_argument("--transient-feature-calibration-gate-decay", type=float, default=0.50)
    parser.add_argument("--transient-feature-calibration-threshold", type=float, default=0.0)
    parser.add_argument("--transient-feature-calibration-score-top-k", type=int, default=0)
    parser.add_argument("--transient-feature-calibration-update-top-k", type=int, default=0)
    parser.add_argument("--transient-feature-calibration-update-margin", type=float, default=0.0)
    parser.add_argument("--transient-feature-calibration-update-rank-tau", type=float, default=0.0)
    parser.add_argument("--transient-feature-calibration-update-margin-tau", type=float, default=0.0)
    parser.add_argument("--transient-feature-calibration-derived-codes", action="store_true")
    parser.add_argument("--transient-winner-inhibition", action="store_true")
    parser.add_argument("--transient-winner-inhibit-strength", type=float, default=0.25)
    parser.add_argument("--transient-winner-inhibit-lr", type=float, default=0.005)
    parser.add_argument("--transient-winner-inhibit-decay", type=float, default=1.0)
    parser.add_argument("--transient-winner-inhibit-clip", type=float, default=1.0)
    parser.add_argument("--transient-winner-inhibit-dim", type=int, default=32)
    parser.add_argument("--transient-winner-inhibit-gate-decay", type=float, default=0.50)
    parser.add_argument("--transient-winner-inhibit-threshold", type=float, default=0.0)
    parser.add_argument("--transient-winner-inhibit-derived-codes", action="store_true")
    parser.add_argument("--readout-gain", type=float, default=1.0)
    parser.add_argument("--readout-gain-mode", choices=["fixed", "margin"], default="fixed")
    parser.add_argument("--readout-gain-margin-center", type=float, default=1.0)
    parser.add_argument("--readout-gain-margin-scale", type=float, default=1.0)
    parser.add_argument("--readout-gain-min", type=float, default=0.5)
    parser.add_argument("--readout-gain-max", type=float, default=3.0)
    parser.add_argument("--local-readout-gain", action="store_true")
    parser.add_argument("--local-readout-base-gain", type=float, default=1.4285714286)
    parser.add_argument("--local-readout-gain-strength", type=float, default=0.35)
    parser.add_argument("--local-readout-gain-lr", type=float, default=0.01)
    parser.add_argument("--local-readout-gain-decay", type=float, default=1.0)
    parser.add_argument("--local-readout-gain-clip", type=float, default=2.0)
    parser.add_argument("--local-readout-gain-min", type=float, default=0.7)
    parser.add_argument("--local-readout-gain-max", type=float, default=2.0)
    parser.add_argument("--local-readout-gain-dim", type=int, default=32)
    parser.add_argument("--local-readout-gain-gate-decay", type=float, default=0.50)
    parser.add_argument("--local-readout-gain-threshold", type=float, default=0.0)
    parser.add_argument("--local-readout-gain-correct-margin", type=float, default=0.0)
    parser.add_argument("--local-readout-gain-mistake-margin", type=float, default=0.0)
    parser.add_argument("--local-readout-gain-update-mode", choices=["wta", "ce"], default="wta")
    parser.add_argument("--local-readout-gain-error-clip", type=float, default=1.0)
    parser.add_argument("--local-readout-gain-memory-scope", choices=["persistent", "dynamic"], default="persistent")
    parser.add_argument("--local-readout-gain-derived-codes", action="store_true")
    parser.add_argument("--branch-state-stabilizer", action="store_true")
    parser.add_argument("--branch-state-strength", type=float, default=0.10)
    parser.add_argument("--branch-state-lr", type=float, default=0.001)
    parser.add_argument("--branch-state-decay", type=float, default=0.85)
    parser.add_argument("--branch-state-projection-decay", type=float, default=1.0)
    parser.add_argument("--branch-state-clip", type=float, default=0.75)
    parser.add_argument("--branch-state-target-mix", type=float, default=0.25)
    parser.add_argument(
        "--branch-state-gate-mode",
        choices=["none", "margin", "branch", "inhibition", "apical", "any", "all"],
        default="any",
    )
    parser.add_argument("--branch-state-margin-threshold", type=float, default=0.35)
    parser.add_argument("--branch-state-branch-threshold", type=float, default=0.05)
    parser.add_argument("--branch-state-inhibition-threshold", type=float, default=0.02)
    parser.add_argument("--branch-state-apical-threshold", type=float, default=0.0)
    parser.add_argument("--branch-state-gate-gain", type=float, default=1.0)
    parser.add_argument("--branch-state-top-k", type=int, default=1)
    parser.add_argument("--branch-state-update-target-top-k", type=int, default=0)
    parser.add_argument("--branch-state-support-clip", type=float, default=3.0)
    parser.add_argument("--branch-state-input-mode", choices=["feature", "target", "mixed"], default="mixed")
    parser.add_argument("--branch-state-projection-rank", type=int, default=0)
    parser.add_argument("--branch-state-novelty-slots", type=int, default=0)
    parser.add_argument("--branch-state-novelty-threshold", type=float, default=0.92)
    parser.add_argument("--branch-state-novelty-strength", type=float, default=0.75)
    parser.add_argument("--branch-state-anti-strength", type=float, default=0.0)
    parser.add_argument("--branch-state-anti-threshold", type=float, default=0.80)
    parser.add_argument("--branch-state-anti-orthogonal", type=float, default=0.0)
    parser.add_argument("--branch-state-anti-score-strength", type=float, default=0.0)
    parser.add_argument("--branch-state-anti-candidate-top-k", type=int, default=0)
    parser.add_argument("--branch-state-anti-candidate-center", action="store_true")
    parser.add_argument("--branch-state-anti-candidate-agreement-weight", type=float, default=0.0)
    parser.add_argument("--branch-state-anti-prediction-only", action="store_true")
    parser.add_argument("--branch-state-derived-codes", action="store_true")
    parser.add_argument("--branch-agreement-readout", action="store_true")
    parser.add_argument("--branch-agreement-strength", type=float, default=0.1)
    parser.add_argument("--branch-agreement-mode", choices=["mean_min", "positive_fraction", "low_variance", "min"], default="mean_min")
    parser.add_argument("--branch-agreement-clip", type=float, default=3.0)
    parser.add_argument("--branch-agreement-threshold", type=float, default=0.0)
    parser.add_argument("--branch-agreement-variance-penalty", type=float, default=0.25)
    parser.add_argument("--plastic-branch-agreement", action="store_true")
    parser.add_argument("--plastic-branch-agreement-strength", type=float, default=0.05)
    parser.add_argument("--plastic-branch-agreement-lr", type=float, default=0.005)
    parser.add_argument("--plastic-branch-agreement-decay", type=float, default=1.0)
    parser.add_argument("--plastic-branch-agreement-clip", type=float, default=1.5)
    parser.add_argument("--plastic-branch-agreement-support-clip", type=float, default=3.0)
    parser.add_argument("--plastic-branch-agreement-top-k", type=int, default=1)
    parser.add_argument("--plastic-branch-agreement-margin", type=float, default=0.0)
    parser.add_argument(
        "--plastic-branch-agreement-pressure-mode",
        choices=["none", "inhibition", "context_loop", "either", "both"],
        default="none",
    )
    parser.add_argument("--plastic-branch-agreement-pressure-threshold", type=float, default=0.0)
    parser.add_argument("--plastic-branch-agreement-pressure-gain", type=float, default=0.0)
    parser.add_argument("--plastic-branch-agreement-loop-window", type=int, default=8)
    parser.add_argument("--loop-inhibition", action="store_true")
    parser.add_argument("--loop-inhibit-strength", type=float, default=0.15)
    parser.add_argument("--loop-inhibit-activity-decay", type=float, default=0.85)
    parser.add_argument("--loop-inhibit-pressure-decay", type=float, default=0.80)
    parser.add_argument("--loop-inhibit-threshold", type=float, default=0.5)
    parser.add_argument("--loop-inhibit-clip", type=float, default=4.0)
    parser.add_argument("--loop-inhibit-repeat-gain", type=float, default=1.0)
    parser.add_argument("--loop-inhibit-transition-strength", type=float, default=0.0)
    parser.add_argument("--loop-inhibit-transition-decay", type=float, default=0.80)
    parser.add_argument("--loop-inhibit-transition-threshold", type=float, default=0.5)
    parser.add_argument("--loop-inhibit-transition-clip", type=float, default=4.0)
    parser.add_argument("--loop-inhibit-transition-gain", type=float, default=1.0)
    parser.add_argument("--segment-attractor-inhibition", action="store_true")
    parser.add_argument("--segment-attractor-strength", type=float, default=0.75)
    parser.add_argument("--segment-attractor-dim", type=int, default=32)
    parser.add_argument("--segment-attractor-state-decay", type=float, default=0.90)
    parser.add_argument("--segment-attractor-trace-decay", type=float, default=0.85)
    parser.add_argument("--segment-attractor-pressure-decay", type=float, default=0.85)
    parser.add_argument("--segment-attractor-threshold", type=float, default=0.80)
    parser.add_argument("--segment-attractor-gain", type=float, default=1.0)
    parser.add_argument("--segment-attractor-clip", type=float, default=4.0)
    parser.add_argument("--segment-attractor-slots", type=int, default=16)
    parser.add_argument("--segment-attractor-lag", type=int, default=4)
    parser.add_argument("--segment-attractor-stride", type=int, default=2)
    parser.add_argument(
        "--segment-attractor-gate-mode",
        choices=[
            "none",
            "margin",
            "inhibition",
            "branch",
            "margin_or_inhibition",
            "margin_and_inhibition",
            "either",
            "both",
        ],
        default="none",
    )
    parser.add_argument("--segment-attractor-gate-margin-threshold", type=float, default=0.35)
    parser.add_argument("--segment-attractor-gate-inhibition-threshold", type=float, default=0.05)
    parser.add_argument("--segment-attractor-gate-branch-threshold", type=float, default=0.0)
    parser.add_argument("--segment-attractor-gate-gain", type=float, default=1.0)
    parser.add_argument("--loop-escape", action="store_true")
    parser.add_argument("--loop-escape-strength", type=float, default=0.75)
    parser.add_argument("--loop-escape-lr", type=float, default=0.005)
    parser.add_argument("--loop-escape-decay", type=float, default=1.0)
    parser.add_argument("--loop-escape-clip", type=float, default=1.5)
    parser.add_argument("--loop-escape-support-clip", type=float, default=3.0)
    parser.add_argument("--loop-escape-top-k", type=int, default=1)
    parser.add_argument("--loop-escape-margin", type=float, default=0.0)
    parser.add_argument(
        "--loop-escape-gate-mode",
        choices=["pressure", "pressure_and_margin", "pressure_or_margin"],
        default="pressure_and_margin",
    )
    parser.add_argument("--loop-escape-pressure-threshold", type=float, default=0.20)
    parser.add_argument("--loop-escape-pressure-gain", type=float, default=1.0)
    parser.add_argument("--loop-escape-margin-threshold", type=float, default=0.35)
    parser.add_argument(
        "--loop-escape-score-mode",
        choices=["all", "base_topk", "winner_suppress"],
        default="all",
    )
    parser.add_argument("--loop-escape-score-top-k", type=int, default=8)
    parser.add_argument(
        "--loop-escape-update-mode",
        choices=["target_wrong", "wrong_only"],
        default="target_wrong",
    )
    parser.add_argument("--loop-escape-learn-candidate-k", type=int, default=0)
    parser.add_argument("--sparse-alpha", type=float, default=0.10)
    parser.add_argument("--completion-count", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--completion-tokens", type=int, default=48)
    parser.add_argument("--repetition-penalty", type=float, default=0.45)
    parser.add_argument("--no-repeat-ngram", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.branch_orders) != len(args.branch_weights):
        raise ValueError("--branch-orders and --branch-weights must have the same length")
    if args.output_homeostasis and args.feature_calibration:
        raise ValueError("--output-homeostasis and --feature-calibration are alternative calibration wrappers")
    if args.loop_escape and not args.segment_attractor_inhibition:
        raise ValueError("--loop-escape requires --segment-attractor-inhibition")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    train_raw = phase.encode_text(tokenizer, phase.read_prefix(args.train_file, args.train_chars))
    valid_raw = phase.encode_text(tokenizer, phase.read_prefix(args.valid_file, args.valid_chars))
    kept_raw, train_ids, valid_ids = phase.build_compact_vocab(train_raw, valid_raw, args.max_vocab)
    if args.warmup_token_limit > 0:
        train_ids = train_ids[: args.warmup_token_limit]
    if args.stream_token_limit > 0:
        valid_ids = valid_ids[: args.stream_token_limit]
    vocab_size = int(len(kept_raw))

    phase_cfg = phase.PhaseTokenConfig(
        context_order=max(args.branch_orders),
        complex_dim=args.phase_dim,
        lr=args.phase_lr,
        epochs=1,
        logit_scale=args.phase_logit_scale,
        bias_weight=args.phase_bias_weight,
        temperature=args.temperature,
        seed=args.seed,
    )
    def adaptive_wrap(base: Any) -> AdaptiveOutputInhibitionMemory:
        return AdaptiveOutputInhibitionMemory(
            base,
            strength=args.inhibit_strength,
            decay=args.inhibit_decay,
            lr=args.inhibit_lr,
            disinhibit_lr=args.inhibit_disinhibit_lr,
            top_k=args.inhibit_top_k,
            margin=args.inhibit_margin,
            max_weight=args.inhibit_max_weight,
        )

    def gated_wrap(base: Any) -> ContextGatedOutputInhibitionMemory:
        return ContextGatedOutputInhibitionMemory(
            base,
            strength=args.gate_inhibit_strength,
            lr=args.gate_inhibit_lr,
            disinhibit_lr=args.gate_inhibit_disinhibit_lr,
            top_k=args.gate_inhibit_top_k,
            margin=args.gate_inhibit_margin,
            max_weight=args.gate_inhibit_max_weight,
            gate_dim=args.gate_dim,
            gate_decay=args.gate_decay,
            gate_threshold=args.gate_threshold,
            seed=args.seed,
        )

    def homeostasis_wrap(base: Any) -> OutputHomeostasisMemory:
        return OutputHomeostasisMemory(
            base,
            strength=args.homeostasis_strength,
            lr=args.homeostasis_lr,
            decay=args.homeostasis_decay,
            clip=args.homeostasis_clip,
        )

    def feature_calibration_wrap(base: Any) -> FeatureConditionedCalibrationMemory:
        return FeatureConditionedCalibrationMemory(
            base,
            strength=args.feature_calibration_strength,
            lr=args.feature_calibration_lr,
            decay=args.feature_calibration_decay,
            clip=args.feature_calibration_clip,
            gate_dim=args.feature_calibration_dim,
            gate_decay=args.feature_calibration_gate_decay,
            gate_threshold=args.feature_calibration_threshold,
            memory_scope="persistent",
            score_top_k=0,
            update_top_k=0,
            update_margin=0.0,
            update_rank_tau=0.0,
            update_margin_tau=0.0,
            update_mode="target_wrong",
            seed=args.seed,
            derived_codes=args.feature_calibration_derived_codes,
        )

    def transient_feature_calibration_wrap(base: Any) -> FeatureConditionedCalibrationMemory:
        return FeatureConditionedCalibrationMemory(
            base,
            strength=args.transient_feature_calibration_strength,
            lr=args.transient_feature_calibration_lr,
            decay=args.transient_feature_calibration_decay,
            clip=args.transient_feature_calibration_clip,
            gate_dim=args.transient_feature_calibration_dim,
            gate_decay=args.transient_feature_calibration_gate_decay,
            gate_threshold=args.transient_feature_calibration_threshold,
            memory_scope="dynamic",
            score_top_k=args.transient_feature_calibration_score_top_k,
            update_top_k=args.transient_feature_calibration_update_top_k,
            update_margin=args.transient_feature_calibration_update_margin,
            update_rank_tau=args.transient_feature_calibration_update_rank_tau,
            update_margin_tau=args.transient_feature_calibration_update_margin_tau,
            update_mode="target_wrong",
            seed=args.seed + 104729,
            derived_codes=args.transient_feature_calibration_derived_codes,
        )

    def transient_winner_inhibition_wrap(base: Any) -> FeatureConditionedCalibrationMemory:
        return FeatureConditionedCalibrationMemory(
            base,
            strength=args.transient_winner_inhibit_strength,
            lr=args.transient_winner_inhibit_lr,
            decay=args.transient_winner_inhibit_decay,
            clip=args.transient_winner_inhibit_clip,
            gate_dim=args.transient_winner_inhibit_dim,
            gate_decay=args.transient_winner_inhibit_gate_decay,
            gate_threshold=args.transient_winner_inhibit_threshold,
            memory_scope="dynamic",
            score_top_k=0,
            update_top_k=0,
            update_margin=0.0,
            update_rank_tau=0.0,
            update_margin_tau=0.0,
            update_mode="wrong_only",
            seed=args.seed + 209759,
            derived_codes=args.transient_winner_inhibit_derived_codes,
        )

    def readout_gain_wrap(base: Any) -> ReadoutGainMemory:
        return ReadoutGainMemory(
            base,
            gain=args.readout_gain,
            mode=args.readout_gain_mode,
            margin_center=args.readout_gain_margin_center,
            margin_scale=args.readout_gain_margin_scale,
            min_gain=args.readout_gain_min,
            max_gain=args.readout_gain_max,
        )

    def local_readout_gain_wrap(base: Any) -> LocalAdaptiveReadoutGainMemory:
        return LocalAdaptiveReadoutGainMemory(
            base,
            base_gain=args.local_readout_base_gain,
            strength=args.local_readout_gain_strength,
            lr=args.local_readout_gain_lr,
            decay=args.local_readout_gain_decay,
            clip=args.local_readout_gain_clip,
            min_gain=args.local_readout_gain_min,
            max_gain=args.local_readout_gain_max,
            gate_dim=args.local_readout_gain_dim,
            gate_decay=args.local_readout_gain_gate_decay,
            gate_threshold=args.local_readout_gain_threshold,
            correct_margin=args.local_readout_gain_correct_margin,
            mistake_margin=args.local_readout_gain_mistake_margin,
            update_mode=args.local_readout_gain_update_mode,
            error_clip=args.local_readout_gain_error_clip,
            memory_scope=args.local_readout_gain_memory_scope,
            seed=args.seed,
            derived_codes=args.local_readout_gain_derived_codes,
        )

    def branch_agreement_wrap(base: Any) -> BranchAgreementReadoutMemory:
        return BranchAgreementReadoutMemory(
            base,
            strength=args.branch_agreement_strength,
            mode=args.branch_agreement_mode,
            clip=args.branch_agreement_clip,
            threshold=args.branch_agreement_threshold,
            variance_penalty=args.branch_agreement_variance_penalty,
        )

    def plastic_branch_agreement_wrap(base: Any) -> PlasticBranchAgreementReadoutMemory:
        return PlasticBranchAgreementReadoutMemory(
            base,
            strength=args.plastic_branch_agreement_strength,
            lr=args.plastic_branch_agreement_lr,
            decay=args.plastic_branch_agreement_decay,
            clip=args.plastic_branch_agreement_clip,
            support_clip=args.plastic_branch_agreement_support_clip,
            top_k=args.plastic_branch_agreement_top_k,
            margin=args.plastic_branch_agreement_margin,
            pressure_mode=args.plastic_branch_agreement_pressure_mode,
            pressure_threshold=args.plastic_branch_agreement_pressure_threshold,
            pressure_gain=args.plastic_branch_agreement_pressure_gain,
            loop_window=args.plastic_branch_agreement_loop_window,
        )

    def loop_inhibition_wrap(base: Any) -> LoopPressureInhibitionMemory:
        return LoopPressureInhibitionMemory(
            base,
            strength=args.loop_inhibit_strength,
            activity_decay=args.loop_inhibit_activity_decay,
            pressure_decay=args.loop_inhibit_pressure_decay,
            threshold=args.loop_inhibit_threshold,
            clip=args.loop_inhibit_clip,
            repeat_gain=args.loop_inhibit_repeat_gain,
            transition_strength=args.loop_inhibit_transition_strength,
            transition_decay=args.loop_inhibit_transition_decay,
            transition_threshold=args.loop_inhibit_transition_threshold,
            transition_clip=args.loop_inhibit_transition_clip,
            transition_gain=args.loop_inhibit_transition_gain,
        )

    def segment_attractor_wrap(base: Any) -> SegmentAttractorInhibitionMemory:
        return SegmentAttractorInhibitionMemory(
            base,
            strength=args.segment_attractor_strength,
            state_dim=args.segment_attractor_dim,
            state_decay=args.segment_attractor_state_decay,
            trace_decay=args.segment_attractor_trace_decay,
            pressure_decay=args.segment_attractor_pressure_decay,
            threshold=args.segment_attractor_threshold,
            gain=args.segment_attractor_gain,
            clip=args.segment_attractor_clip,
            slots=args.segment_attractor_slots,
            lag=args.segment_attractor_lag,
            stride=args.segment_attractor_stride,
            gate_mode=args.segment_attractor_gate_mode,
            gate_margin_threshold=args.segment_attractor_gate_margin_threshold,
            gate_inhibition_threshold=args.segment_attractor_gate_inhibition_threshold,
            gate_branch_threshold=args.segment_attractor_gate_branch_threshold,
            gate_gain=args.segment_attractor_gate_gain,
            seed=args.seed,
        )

    def loop_escape_wrap(base: Any) -> LoopEscapeCompetitorMemory:
        return LoopEscapeCompetitorMemory(
            base,
            strength=args.loop_escape_strength,
            lr=args.loop_escape_lr,
            decay=args.loop_escape_decay,
            clip=args.loop_escape_clip,
            support_clip=args.loop_escape_support_clip,
            top_k=args.loop_escape_top_k,
            margin=args.loop_escape_margin,
            gate_mode=args.loop_escape_gate_mode,
            pressure_threshold=args.loop_escape_pressure_threshold,
            pressure_gain=args.loop_escape_pressure_gain,
            margin_threshold=args.loop_escape_margin_threshold,
            score_mode=args.loop_escape_score_mode,
            score_top_k=args.loop_escape_score_top_k,
            update_mode=args.loop_escape_update_mode,
            learn_candidate_k=args.loop_escape_learn_candidate_k,
        )

    def branch_state_wrap(base: Any) -> BranchStateStabilizerMemory:
        return BranchStateStabilizerMemory(
            base,
            strength=args.branch_state_strength,
            lr=args.branch_state_lr,
            state_decay=args.branch_state_decay,
            projection_decay=args.branch_state_projection_decay,
            clip=args.branch_state_clip,
            target_mix=args.branch_state_target_mix,
            gate_mode=args.branch_state_gate_mode,
            margin_threshold=args.branch_state_margin_threshold,
            branch_threshold=args.branch_state_branch_threshold,
            inhibition_threshold=args.branch_state_inhibition_threshold,
            apical_threshold=args.branch_state_apical_threshold,
            gate_gain=args.branch_state_gate_gain,
            top_k=args.branch_state_top_k,
            update_target_top_k=args.branch_state_update_target_top_k,
            support_clip=args.branch_state_support_clip,
            input_mode=args.branch_state_input_mode,
            projection_rank=args.branch_state_projection_rank,
            novelty_slots=args.branch_state_novelty_slots,
            novelty_threshold=args.branch_state_novelty_threshold,
            novelty_strength=args.branch_state_novelty_strength,
            anti_attractor_strength=args.branch_state_anti_strength,
            anti_attractor_threshold=args.branch_state_anti_threshold,
            anti_attractor_orthogonal=args.branch_state_anti_orthogonal,
            anti_score_strength=args.branch_state_anti_score_strength,
            anti_candidate_top_k=args.branch_state_anti_candidate_top_k,
            anti_candidate_center=args.branch_state_anti_candidate_center,
            anti_candidate_agreement_weight=args.branch_state_anti_candidate_agreement_weight,
            anti_prediction_only=args.branch_state_anti_prediction_only,
            seed=args.seed,
            derived_codes=args.branch_state_derived_codes,
        )

    def eprop_trace_memory(apical: bool = False) -> Any:
        if apical:
            return OnlineEPropTraceApicalGatedCompetitivePhaseMemory(
                vocab_size,
                phase_cfg,
                args.branch_orders,
                args.branch_weights,
                args.trace_order,
                args.trace_dim,
                args.trace_decay,
                args.trace_weight,
                args.eprop_order,
                args.eprop_decay,
                args.eprop_weight,
                args.apical_decay,
                args.apical_strength,
                args.apical_margin,
                args.apical_min_gate,
                args.apical_max_gate,
                args.apical_error_clip,
                args.competitive_lr,
                args.competitive_neg_k,
                args.competitive_epochs,
                args.competitive_score_scale,
                args.competitive_init,
                args.competitive_margin,
                args.seed,
                args.apical_error_mode,
            )
        return OnlineEPropTraceCompetitivePhaseMemory(
            vocab_size,
            phase_cfg,
            args.branch_orders,
            args.branch_weights,
            args.trace_order,
            args.trace_dim,
            args.trace_decay,
            args.trace_weight,
            args.eprop_order,
            args.eprop_decay,
            args.eprop_weight,
            args.competitive_lr,
            args.competitive_neg_k,
            args.competitive_epochs,
            args.competitive_score_scale,
            args.competitive_init,
            args.competitive_margin,
            args.seed,
        )

    def trace_kv_memory(apical: bool = False) -> Any:
        if apical:
            return OnlineTraceHebbianKVApicalGatedCompetitivePhaseMemory(
                vocab_size,
                phase_cfg,
                args.branch_orders,
                args.branch_weights,
                args.trace_order,
                args.trace_dim,
                args.trace_decay,
                args.trace_weight,
                args.kv_order,
                args.kv_dim,
                args.kv_trace_decay,
                args.kv_weight,
                args.kv_score_weight,
                args.kv_gate_mode,
                args.kv_gate_base_margin,
                args.kv_gate_kv_margin,
                args.kv_gate_min_norm,
                args.kv_lr,
                args.kv_decay,
                args.kv_clip,
                args.apical_decay,
                args.apical_strength,
                args.apical_margin,
                args.apical_min_gate,
                args.apical_max_gate,
                args.apical_error_clip,
                args.competitive_lr,
                args.competitive_neg_k,
                args.competitive_epochs,
                args.competitive_score_scale,
                args.competitive_init,
                args.competitive_margin,
                args.seed,
                args.apical_error_mode,
            )
        return OnlineTraceHebbianKVCompetitivePhaseMemory(
            vocab_size,
            phase_cfg,
            args.branch_orders,
            args.branch_weights,
            args.trace_order,
            args.trace_dim,
            args.trace_decay,
            args.trace_weight,
            args.kv_order,
            args.kv_dim,
            args.kv_trace_decay,
            args.kv_weight,
            args.kv_score_weight,
            args.kv_gate_mode,
            args.kv_gate_base_margin,
            args.kv_gate_kv_margin,
            args.kv_gate_min_norm,
            args.kv_lr,
            args.kv_decay,
            args.kv_clip,
            args.competitive_lr,
            args.competitive_neg_k,
            args.competitive_epochs,
            args.competitive_score_scale,
            args.competitive_init,
            args.competitive_margin,
            args.seed,
        )

    def dll_depth_memory() -> Any:
        return OnlineDLLDeepLocalMemory(
            vocab_size,
            phase_cfg,
            args.branch_orders,
            args.branch_weights,
            args.trace_order,
            args.trace_dim,
            args.trace_decay,
            args.trace_weight,
            args.dll_hidden_dims,
            args.dll_label_dim,
            args.dll_lr,
            args.dll_bias_lr,
            args.dll_delta_clip,
            args.dll_activation,
            not args.dll_disable_row_normalize,
            args.competitive_lr,
            args.competitive_neg_k,
            args.competitive_epochs,
            args.competitive_score_scale,
            args.competitive_init,
            args.competitive_margin,
            args.seed,
        )

    def noprop_depth_memory() -> Any:
        return OnlineNoPropLocalDenoisingMemory(
            vocab_size,
            phase_cfg,
            args.branch_orders,
            args.branch_weights,
            args.trace_order,
            args.trace_dim,
            args.trace_decay,
            args.trace_weight,
            args.noprop_hidden_dims,
            args.noprop_label_dim,
            args.noprop_alpha_start,
            args.noprop_alpha_end,
            args.noprop_lr,
            args.noprop_denoise_lr,
            args.noprop_bias_lr,
            args.noprop_delta_clip,
            args.noprop_activation,
            not args.noprop_disable_row_normalize,
            args.competitive_lr,
            args.competitive_neg_k,
            args.competitive_epochs,
            args.competitive_score_scale,
            args.competitive_init,
            args.competitive_margin,
            args.seed,
        )

    def unified_noprop_memory() -> Any:
        return OnlineUnifiedNoPropCalibrationMemory(
            vocab_size,
            phase_cfg,
            args.branch_orders,
            args.branch_weights,
            args.trace_order,
            args.trace_dim,
            args.trace_decay,
            args.trace_weight,
            args.noprop_hidden_dims,
            args.noprop_label_dim,
            args.noprop_alpha_start,
            args.noprop_alpha_end,
            args.noprop_lr,
            args.noprop_denoise_lr,
            args.noprop_bias_lr,
            args.noprop_delta_clip,
            args.noprop_activation,
            not args.noprop_disable_row_normalize,
            args.inhibit_strength,
            args.inhibit_decay,
            args.inhibit_lr,
            args.inhibit_disinhibit_lr,
            args.inhibit_top_k,
            args.inhibit_margin,
            args.inhibit_max_weight,
            args.unified_calibration_strength,
            args.unified_calibration_lr,
            args.unified_calibration_clip,
            args.unified_calibration_dim,
            args.unified_calibration_gate_decay,
            args.unified_calibration_threshold,
            args.unified_readout_gain,
            args.unified_eligibility_order,
            args.unified_eligibility_decay,
            args.unified_eligibility_score_weight,
            args.unified_eligibility_top_k,
            args.competitive_lr,
            args.competitive_neg_k,
            args.competitive_epochs,
            args.competitive_score_scale,
            args.competitive_init,
            args.competitive_margin,
            args.seed,
        )

    builders = {
        "phase_competitive_online": lambda: OnlineCompetitivePhaseMemory(
            vocab_size,
            phase_cfg,
            args.branch_orders,
            args.branch_weights,
            args.competitive_lr,
            args.competitive_neg_k,
            args.competitive_epochs,
            args.competitive_score_scale,
            args.competitive_init,
            args.competitive_margin,
            args.seed,
        ),
    }
    if args.dll_depth_branch:
        builders["phase_trace_dll_local_competitive_online"] = lambda: dll_depth_memory()
        if args.output_fatigue:
            builders["phase_trace_dll_local_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                dll_depth_memory(),
                strength=args.fatigue_strength,
                decay=args.fatigue_decay,
            )
            if args.adaptive_inhibition:
                builders["phase_trace_dll_local_fatigue_inhib_competitive_online"] = lambda: adaptive_wrap(
                    OutputFatigueMemory(
                        dll_depth_memory(),
                        strength=args.fatigue_strength,
                        decay=args.fatigue_decay,
                    )
                )
        if args.adaptive_inhibition:
            builders["phase_trace_dll_local_inhib_competitive_online"] = lambda: adaptive_wrap(dll_depth_memory())
        if args.context_gated_inhibition:
            builders["phase_trace_dll_local_gate_inhib_competitive_online"] = lambda: gated_wrap(dll_depth_memory())
    if args.noprop_depth_branch:
        builders["phase_trace_noprop_local_competitive_online"] = lambda: noprop_depth_memory()
        if args.output_fatigue:
            builders["phase_trace_noprop_local_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                noprop_depth_memory(),
                strength=args.fatigue_strength,
                decay=args.fatigue_decay,
            )
            if args.adaptive_inhibition:
                builders["phase_trace_noprop_local_fatigue_inhib_competitive_online"] = lambda: adaptive_wrap(
                    OutputFatigueMemory(
                        noprop_depth_memory(),
                        strength=args.fatigue_strength,
                        decay=args.fatigue_decay,
                    )
                )
        if args.adaptive_inhibition:
            builders["phase_trace_noprop_local_inhib_competitive_online"] = lambda: adaptive_wrap(noprop_depth_memory())
        if args.context_gated_inhibition:
            builders["phase_trace_noprop_local_gate_inhib_competitive_online"] = lambda: gated_wrap(
                noprop_depth_memory()
            )
    if args.unified_noprop_core:
        builders["u001_unified_noprop_calib_online"] = lambda: unified_noprop_memory()
    if args.output_fatigue:
        builders["phase_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
            OnlineCompetitivePhaseMemory(
                vocab_size,
                phase_cfg,
                args.branch_orders,
                args.branch_weights,
                args.competitive_lr,
                args.competitive_neg_k,
                args.competitive_epochs,
                args.competitive_score_scale,
                args.competitive_init,
                args.competitive_margin,
                args.seed,
            ),
            strength=args.fatigue_strength,
            decay=args.fatigue_decay,
        )
        if args.adaptive_inhibition:
            builders["phase_fatigue_inhib_competitive_online"] = lambda: adaptive_wrap(
                OutputFatigueMemory(
                    OnlineCompetitivePhaseMemory(
                        vocab_size,
                        phase_cfg,
                        args.branch_orders,
                        args.branch_weights,
                        args.competitive_lr,
                        args.competitive_neg_k,
                        args.competitive_epochs,
                        args.competitive_score_scale,
                        args.competitive_init,
                        args.competitive_margin,
                        args.seed,
                    ),
                    strength=args.fatigue_strength,
                    decay=args.fatigue_decay,
                )
            )
    if args.adaptive_inhibition:
        builders["phase_inhib_competitive_online"] = lambda: adaptive_wrap(
            OnlineCompetitivePhaseMemory(
                vocab_size,
                phase_cfg,
                args.branch_orders,
                args.branch_weights,
                args.competitive_lr,
                args.competitive_neg_k,
                args.competitive_epochs,
                args.competitive_score_scale,
                args.competitive_init,
                args.competitive_margin,
                args.seed,
            )
        )
    if args.context_gated_inhibition:
        builders["phase_gate_inhib_competitive_online"] = lambda: gated_wrap(
            OnlineCompetitivePhaseMemory(
                vocab_size,
                phase_cfg,
                args.branch_orders,
                args.branch_weights,
                args.competitive_lr,
                args.competitive_neg_k,
                args.competitive_epochs,
                args.competitive_score_scale,
                args.competitive_init,
                args.competitive_margin,
                args.seed,
            )
        )
    if args.trace_branch:
        builders["phase_trace_competitive_online"] = lambda: OnlineTraceCompetitivePhaseMemory(
            vocab_size,
            phase_cfg,
            args.branch_orders,
            args.branch_weights,
            args.trace_order,
            args.trace_dim,
            args.trace_decay,
            args.trace_weight,
            args.competitive_lr,
            args.competitive_neg_k,
            args.competitive_epochs,
            args.competitive_score_scale,
            args.competitive_init,
            args.competitive_margin,
            args.seed,
        )
        if args.output_fatigue:
            builders["phase_trace_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                OnlineTraceCompetitivePhaseMemory(
                    vocab_size,
                    phase_cfg,
                    args.branch_orders,
                    args.branch_weights,
                    args.trace_order,
                    args.trace_dim,
                    args.trace_decay,
                    args.trace_weight,
                    args.competitive_lr,
                    args.competitive_neg_k,
                    args.competitive_epochs,
                    args.competitive_score_scale,
                    args.competitive_init,
                    args.competitive_margin,
                    args.seed,
                ),
                strength=args.fatigue_strength,
                decay=args.fatigue_decay,
            )
            if args.adaptive_inhibition:
                builders["phase_trace_fatigue_inhib_competitive_online"] = lambda: adaptive_wrap(
                    OutputFatigueMemory(
                        OnlineTraceCompetitivePhaseMemory(
                            vocab_size,
                            phase_cfg,
                            args.branch_orders,
                            args.branch_weights,
                            args.trace_order,
                            args.trace_dim,
                            args.trace_decay,
                            args.trace_weight,
                            args.competitive_lr,
                            args.competitive_neg_k,
                            args.competitive_epochs,
                            args.competitive_score_scale,
                            args.competitive_init,
                            args.competitive_margin,
                            args.seed,
                        ),
                        strength=args.fatigue_strength,
                        decay=args.fatigue_decay,
                    )
                )
            if args.context_gated_inhibition:
                builders["phase_trace_fatigue_gate_inhib_competitive_online"] = lambda: gated_wrap(
                    OutputFatigueMemory(
                        OnlineTraceCompetitivePhaseMemory(
                            vocab_size,
                            phase_cfg,
                            args.branch_orders,
                            args.branch_weights,
                            args.trace_order,
                            args.trace_dim,
                            args.trace_decay,
                            args.trace_weight,
                            args.competitive_lr,
                            args.competitive_neg_k,
                            args.competitive_epochs,
                            args.competitive_score_scale,
                            args.competitive_init,
                            args.competitive_margin,
                            args.seed,
                        ),
                        strength=args.fatigue_strength,
                        decay=args.fatigue_decay,
                    )
                )
        if args.adaptive_inhibition:
            builders["phase_trace_inhib_competitive_online"] = lambda: adaptive_wrap(
                OnlineTraceCompetitivePhaseMemory(
                    vocab_size,
                    phase_cfg,
                    args.branch_orders,
                    args.branch_weights,
                    args.trace_order,
                    args.trace_dim,
                    args.trace_decay,
                    args.trace_weight,
                    args.competitive_lr,
                    args.competitive_neg_k,
                    args.competitive_epochs,
                    args.competitive_score_scale,
                    args.competitive_init,
                    args.competitive_margin,
                    args.seed,
                )
            )
        if args.context_gated_inhibition:
            builders["phase_trace_gate_inhib_competitive_online"] = lambda: gated_wrap(
                OnlineTraceCompetitivePhaseMemory(
                    vocab_size,
                    phase_cfg,
                    args.branch_orders,
                    args.branch_weights,
                    args.trace_order,
                    args.trace_dim,
                    args.trace_decay,
                    args.trace_weight,
                    args.competitive_lr,
                    args.competitive_neg_k,
                    args.competitive_epochs,
                    args.competitive_score_scale,
                    args.competitive_init,
                    args.competitive_margin,
                    args.seed,
                )
            )
        if args.apical_gating_branch:
            builders["phase_trace_apical_competitive_online"] = lambda: OnlineTraceApicalGatedCompetitivePhaseMemory(
                vocab_size,
                phase_cfg,
                args.branch_orders,
                args.branch_weights,
                args.trace_order,
                args.trace_dim,
                args.trace_decay,
                args.trace_weight,
                args.apical_decay,
                args.apical_strength,
                args.apical_margin,
                args.apical_min_gate,
                args.apical_max_gate,
                args.apical_error_clip,
                args.competitive_lr,
                args.competitive_neg_k,
                args.competitive_epochs,
                args.competitive_score_scale,
                args.competitive_init,
                args.competitive_margin,
                args.seed,
            )
            if args.output_fatigue:
                builders["phase_trace_apical_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                    OnlineTraceApicalGatedCompetitivePhaseMemory(
                        vocab_size,
                        phase_cfg,
                        args.branch_orders,
                        args.branch_weights,
                        args.trace_order,
                        args.trace_dim,
                        args.trace_decay,
                        args.trace_weight,
                        args.apical_decay,
                        args.apical_strength,
                        args.apical_margin,
                        args.apical_min_gate,
                        args.apical_max_gate,
                        args.apical_error_clip,
                        args.competitive_lr,
                        args.competitive_neg_k,
                        args.competitive_epochs,
                        args.competitive_score_scale,
                        args.competitive_init,
                        args.competitive_margin,
                        args.seed,
                        args.apical_error_mode,
                    ),
                    strength=args.fatigue_strength,
                    decay=args.fatigue_decay,
                )
                if args.adaptive_inhibition:
                    builders["phase_trace_apical_fatigue_inhib_competitive_online"] = lambda: adaptive_wrap(
                        OutputFatigueMemory(
                            OnlineTraceApicalGatedCompetitivePhaseMemory(
                                vocab_size,
                                phase_cfg,
                                args.branch_orders,
                                args.branch_weights,
                                args.trace_order,
                                args.trace_dim,
                                args.trace_decay,
                                args.trace_weight,
                                args.apical_decay,
                                args.apical_strength,
                                args.apical_margin,
                                args.apical_min_gate,
                                args.apical_max_gate,
                                args.apical_error_clip,
                                args.competitive_lr,
                                args.competitive_neg_k,
                                args.competitive_epochs,
                                args.competitive_score_scale,
                                args.competitive_init,
                                args.competitive_margin,
                                args.seed,
                                args.apical_error_mode,
                            ),
                            strength=args.fatigue_strength,
                            decay=args.fatigue_decay,
                        )
                    )
            if args.adaptive_inhibition:
                builders["phase_trace_apical_inhib_competitive_online"] = lambda: adaptive_wrap(
                    OnlineTraceApicalGatedCompetitivePhaseMemory(
                        vocab_size,
                        phase_cfg,
                        args.branch_orders,
                        args.branch_weights,
                        args.trace_order,
                        args.trace_dim,
                        args.trace_decay,
                        args.trace_weight,
                        args.apical_decay,
                        args.apical_strength,
                        args.apical_margin,
                        args.apical_min_gate,
                        args.apical_max_gate,
                        args.apical_error_clip,
                        args.competitive_lr,
                        args.competitive_neg_k,
                        args.competitive_epochs,
                        args.competitive_score_scale,
                        args.competitive_init,
                        args.competitive_margin,
                        args.seed,
                        args.apical_error_mode,
                    )
                )
    if args.eprop_trace_readout:
        builders["phase_eprop_trace_competitive_online"] = lambda: eprop_trace_memory(apical=False)
        if args.output_fatigue:
            builders["phase_eprop_trace_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                eprop_trace_memory(apical=False),
                strength=args.fatigue_strength,
                decay=args.fatigue_decay,
            )
            if args.adaptive_inhibition:
                builders["phase_eprop_trace_fatigue_inhib_competitive_online"] = lambda: adaptive_wrap(
                    OutputFatigueMemory(
                        eprop_trace_memory(apical=False),
                        strength=args.fatigue_strength,
                        decay=args.fatigue_decay,
                    )
                )
        if args.adaptive_inhibition:
            builders["phase_eprop_trace_inhib_competitive_online"] = lambda: adaptive_wrap(
                eprop_trace_memory(apical=False)
            )
        if args.context_gated_inhibition:
            builders["phase_eprop_trace_gate_inhib_competitive_online"] = lambda: gated_wrap(
                eprop_trace_memory(apical=False)
            )
        if args.apical_gating_branch:
            builders["phase_eprop_trace_apical_competitive_online"] = lambda: eprop_trace_memory(apical=True)
            if args.output_fatigue:
                builders["phase_eprop_trace_apical_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                    eprop_trace_memory(apical=True),
                    strength=args.fatigue_strength,
                    decay=args.fatigue_decay,
                )
                if args.adaptive_inhibition:
                    builders["phase_eprop_trace_apical_fatigue_inhib_competitive_online"] = lambda: adaptive_wrap(
                        OutputFatigueMemory(
                            eprop_trace_memory(apical=True),
                            strength=args.fatigue_strength,
                            decay=args.fatigue_decay,
                        )
                    )
            if args.adaptive_inhibition:
                builders["phase_eprop_trace_apical_inhib_competitive_online"] = lambda: adaptive_wrap(
                    eprop_trace_memory(apical=True)
                )
    if args.hebbian_kv_branch:
        builders["phase_trace_kv_competitive_online"] = lambda: trace_kv_memory(apical=False)
        if args.output_fatigue:
            builders["phase_trace_kv_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                trace_kv_memory(apical=False),
                strength=args.fatigue_strength,
                decay=args.fatigue_decay,
            )
            if args.adaptive_inhibition:
                builders["phase_trace_kv_fatigue_inhib_competitive_online"] = lambda: adaptive_wrap(
                    OutputFatigueMemory(
                        trace_kv_memory(apical=False),
                        strength=args.fatigue_strength,
                        decay=args.fatigue_decay,
                    )
                )
        if args.adaptive_inhibition:
            builders["phase_trace_kv_inhib_competitive_online"] = lambda: adaptive_wrap(trace_kv_memory(apical=False))
        if args.context_gated_inhibition:
            builders["phase_trace_kv_gate_inhib_competitive_online"] = lambda: gated_wrap(trace_kv_memory(apical=False))
        if args.apical_gating_branch:
            builders["phase_trace_kv_apical_competitive_online"] = lambda: trace_kv_memory(apical=True)
            if args.output_fatigue:
                builders["phase_trace_kv_apical_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                    trace_kv_memory(apical=True),
                    strength=args.fatigue_strength,
                    decay=args.fatigue_decay,
                )
                if args.adaptive_inhibition:
                    builders["phase_trace_kv_apical_fatigue_inhib_competitive_online"] = lambda: adaptive_wrap(
                        OutputFatigueMemory(
                            trace_kv_memory(apical=True),
                            strength=args.fatigue_strength,
                            decay=args.fatigue_decay,
                        )
                    )
            if args.adaptive_inhibition:
                builders["phase_trace_kv_apical_inhib_competitive_online"] = lambda: adaptive_wrap(
                    trace_kv_memory(apical=True)
                )
    if args.plastic_ssm_branch:
        builders["phase_plastic_ssm_competitive_online"] = lambda: OnlinePlasticSSMCompetitivePhaseMemory(
            vocab_size,
            phase_cfg,
            args.branch_orders,
            args.branch_weights,
            args.ssm_order,
            args.ssm_dim,
            args.ssm_decay,
            args.ssm_recurrent_scale,
            args.ssm_weight,
            args.ssm_lr,
            args.ssm_target_mix,
            args.ssm_clip,
            args.competitive_lr,
            args.competitive_neg_k,
            args.competitive_epochs,
            args.competitive_score_scale,
            args.competitive_init,
            args.competitive_margin,
            args.seed,
        )
        if args.output_fatigue:
            builders["phase_plastic_ssm_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                OnlinePlasticSSMCompetitivePhaseMemory(
                    vocab_size,
                    phase_cfg,
                    args.branch_orders,
                    args.branch_weights,
                    args.ssm_order,
                    args.ssm_dim,
                    args.ssm_decay,
                    args.ssm_recurrent_scale,
                    args.ssm_weight,
                    args.ssm_lr,
                    args.ssm_target_mix,
                    args.ssm_clip,
                    args.competitive_lr,
                    args.competitive_neg_k,
                    args.competitive_epochs,
                    args.competitive_score_scale,
                    args.competitive_init,
                    args.competitive_margin,
                    args.seed,
                ),
                strength=args.fatigue_strength,
                decay=args.fatigue_decay,
            )
        if args.adaptive_inhibition:
            builders["phase_plastic_ssm_inhib_competitive_online"] = lambda: adaptive_wrap(
                OnlinePlasticSSMCompetitivePhaseMemory(
                    vocab_size,
                    phase_cfg,
                    args.branch_orders,
                    args.branch_weights,
                    args.ssm_order,
                    args.ssm_dim,
                    args.ssm_decay,
                    args.ssm_recurrent_scale,
                    args.ssm_weight,
                    args.ssm_lr,
                    args.ssm_target_mix,
                    args.ssm_clip,
                    args.competitive_lr,
                    args.competitive_neg_k,
                    args.competitive_epochs,
                    args.competitive_score_scale,
                    args.competitive_init,
                    args.competitive_margin,
                    args.seed,
                )
            )
        if args.trace_branch:
            builders["phase_trace_plastic_ssm_competitive_online"] = lambda: OnlineTracePlasticSSMCompetitivePhaseMemory(
                vocab_size,
                phase_cfg,
                args.branch_orders,
                args.branch_weights,
                args.trace_order,
                args.trace_dim,
                args.trace_decay,
                args.trace_weight,
                args.ssm_order,
                args.ssm_dim,
                args.ssm_decay,
                args.ssm_recurrent_scale,
                args.ssm_weight,
                args.ssm_lr,
                args.ssm_target_mix,
                args.ssm_clip,
                args.competitive_lr,
                args.competitive_neg_k,
                args.competitive_epochs,
                args.competitive_score_scale,
                args.competitive_init,
                args.competitive_margin,
                args.seed,
            )
            if args.output_fatigue:
                builders["phase_trace_plastic_ssm_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                    OnlineTracePlasticSSMCompetitivePhaseMemory(
                        vocab_size,
                        phase_cfg,
                        args.branch_orders,
                        args.branch_weights,
                        args.trace_order,
                        args.trace_dim,
                        args.trace_decay,
                        args.trace_weight,
                        args.ssm_order,
                        args.ssm_dim,
                        args.ssm_decay,
                        args.ssm_recurrent_scale,
                        args.ssm_weight,
                        args.ssm_lr,
                        args.ssm_target_mix,
                        args.ssm_clip,
                        args.competitive_lr,
                        args.competitive_neg_k,
                        args.competitive_epochs,
                        args.competitive_score_scale,
                        args.competitive_init,
                        args.competitive_margin,
                        args.seed,
                    ),
                    strength=args.fatigue_strength,
                    decay=args.fatigue_decay,
                )
            if args.adaptive_inhibition:
                builders["phase_trace_plastic_ssm_inhib_competitive_online"] = lambda: adaptive_wrap(
                    OnlineTracePlasticSSMCompetitivePhaseMemory(
                        vocab_size,
                        phase_cfg,
                        args.branch_orders,
                        args.branch_weights,
                        args.trace_order,
                        args.trace_dim,
                        args.trace_decay,
                        args.trace_weight,
                        args.ssm_order,
                        args.ssm_dim,
                        args.ssm_decay,
                        args.ssm_recurrent_scale,
                        args.ssm_weight,
                        args.ssm_lr,
                        args.ssm_target_mix,
                        args.ssm_clip,
                        args.competitive_lr,
                        args.competitive_neg_k,
                        args.competitive_epochs,
                        args.competitive_score_scale,
                        args.competitive_init,
                        args.competitive_margin,
                        args.seed,
                    )
                )
    if args.eligibility_ssm_branch:
        builders["phase_elig_ssm_competitive_online"] = lambda: OnlineEligibilitySSMCompetitivePhaseMemory(
            vocab_size,
            phase_cfg,
            args.branch_orders,
            args.branch_weights,
            args.ssm_order,
            args.ssm_dim,
            args.ssm_decay,
            args.ssm_recurrent_scale,
            args.ssm_weight,
            args.ssm_lr,
            args.ssm_target_mix,
            args.ssm_clip,
            args.ssm_eligibility_decay,
            args.ssm_eligibility_clip,
            args.competitive_lr,
            args.competitive_neg_k,
            args.competitive_epochs,
            args.competitive_score_scale,
            args.competitive_init,
            args.competitive_margin,
            args.seed,
        )
        if args.output_fatigue:
            builders["phase_elig_ssm_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                OnlineEligibilitySSMCompetitivePhaseMemory(
                    vocab_size,
                    phase_cfg,
                    args.branch_orders,
                    args.branch_weights,
                    args.ssm_order,
                    args.ssm_dim,
                    args.ssm_decay,
                    args.ssm_recurrent_scale,
                    args.ssm_weight,
                    args.ssm_lr,
                    args.ssm_target_mix,
                    args.ssm_clip,
                    args.ssm_eligibility_decay,
                    args.ssm_eligibility_clip,
                    args.competitive_lr,
                    args.competitive_neg_k,
                    args.competitive_epochs,
                    args.competitive_score_scale,
                    args.competitive_init,
                    args.competitive_margin,
                    args.seed,
                ),
                strength=args.fatigue_strength,
                decay=args.fatigue_decay,
            )
        if args.adaptive_inhibition:
            builders["phase_elig_ssm_inhib_competitive_online"] = lambda: adaptive_wrap(
                OnlineEligibilitySSMCompetitivePhaseMemory(
                    vocab_size,
                    phase_cfg,
                    args.branch_orders,
                    args.branch_weights,
                    args.ssm_order,
                    args.ssm_dim,
                    args.ssm_decay,
                    args.ssm_recurrent_scale,
                    args.ssm_weight,
                    args.ssm_lr,
                    args.ssm_target_mix,
                    args.ssm_clip,
                    args.ssm_eligibility_decay,
                    args.ssm_eligibility_clip,
                    args.competitive_lr,
                    args.competitive_neg_k,
                    args.competitive_epochs,
                    args.competitive_score_scale,
                    args.competitive_init,
                    args.competitive_margin,
                    args.seed,
                )
            )
        if args.trace_branch:
            builders["phase_trace_elig_ssm_competitive_online"] = lambda: OnlineTraceEligibilitySSMCompetitivePhaseMemory(
                vocab_size,
                phase_cfg,
                args.branch_orders,
                args.branch_weights,
                args.trace_order,
                args.trace_dim,
                args.trace_decay,
                args.trace_weight,
                args.ssm_order,
                args.ssm_dim,
                args.ssm_decay,
                args.ssm_recurrent_scale,
                args.ssm_weight,
                args.ssm_lr,
                args.ssm_target_mix,
                args.ssm_clip,
                args.ssm_eligibility_decay,
                args.ssm_eligibility_clip,
                args.competitive_lr,
                args.competitive_neg_k,
                args.competitive_epochs,
                args.competitive_score_scale,
                args.competitive_init,
                args.competitive_margin,
                args.seed,
            )
            if args.output_fatigue:
                builders["phase_trace_elig_ssm_fatigue_competitive_online"] = lambda: OutputFatigueMemory(
                    OnlineTraceEligibilitySSMCompetitivePhaseMemory(
                        vocab_size,
                        phase_cfg,
                        args.branch_orders,
                        args.branch_weights,
                        args.trace_order,
                        args.trace_dim,
                        args.trace_decay,
                        args.trace_weight,
                        args.ssm_order,
                        args.ssm_dim,
                        args.ssm_decay,
                        args.ssm_recurrent_scale,
                        args.ssm_weight,
                        args.ssm_lr,
                        args.ssm_target_mix,
                        args.ssm_clip,
                        args.ssm_eligibility_decay,
                        args.ssm_eligibility_clip,
                        args.competitive_lr,
                        args.competitive_neg_k,
                        args.competitive_epochs,
                        args.competitive_score_scale,
                        args.competitive_init,
                        args.competitive_margin,
                        args.seed,
                    ),
                    strength=args.fatigue_strength,
                    decay=args.fatigue_decay,
                )
            if args.adaptive_inhibition:
                builders["phase_trace_elig_ssm_inhib_competitive_online"] = lambda: adaptive_wrap(
                    OnlineTraceEligibilitySSMCompetitivePhaseMemory(
                        vocab_size,
                        phase_cfg,
                        args.branch_orders,
                        args.branch_weights,
                        args.trace_order,
                        args.trace_dim,
                        args.trace_decay,
                        args.trace_weight,
                        args.ssm_order,
                        args.ssm_dim,
                        args.ssm_decay,
                        args.ssm_recurrent_scale,
                        args.ssm_weight,
                        args.ssm_lr,
                        args.ssm_target_mix,
                        args.ssm_clip,
                        args.ssm_eligibility_decay,
                        args.ssm_eligibility_clip,
                        args.competitive_lr,
                        args.competitive_neg_k,
                        args.competitive_epochs,
                        args.competitive_score_scale,
                        args.competitive_init,
                        args.competitive_margin,
                        args.seed,
                    )
                )
    builders["sparse_context_aux"] = lambda: OnlineSparseAuxMemory(
        vocab_size,
        context_order=max(args.branch_orders),
        alpha=args.sparse_alpha,
        temperature=args.temperature,
    )
    if args.method_filter:
        filters = tuple(str(item) for item in args.method_filter)
        builders = {
            name: builder
            for name, builder in builders.items()
            if any(pattern in name for pattern in filters)
        }
        if not builders:
            raise ValueError(f"method filter matched no builders: {args.method_filter}")
    if args.output_homeostasis:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_homeostasis"): (
                builder if name == "sparse_context_aux" else (lambda builder=builder: homeostasis_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.feature_calibration:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_feature_calib"): (
                builder if name == "sparse_context_aux" else (lambda builder=builder: feature_calibration_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.readout_gain != 1.0 or args.readout_gain_mode != "fixed":
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_gain"): (
                builder if name == "sparse_context_aux" else (lambda builder=builder: readout_gain_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.transient_feature_calibration:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_transient_feature_calib"): (
                builder
                if name == "sparse_context_aux"
                else (lambda builder=builder: transient_feature_calibration_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.transient_winner_inhibition:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_winner_inhib"): (
                builder
                if name == "sparse_context_aux"
                else (lambda builder=builder: transient_winner_inhibition_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.local_readout_gain:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_local_gain"): (
                builder if name == "sparse_context_aux" else (lambda builder=builder: local_readout_gain_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.branch_agreement_readout:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_branch_agree"): (
                builder if name == "sparse_context_aux" else (lambda builder=builder: branch_agreement_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.plastic_branch_agreement:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_plastic_branch_agree"): (
                builder if name == "sparse_context_aux" else (lambda builder=builder: plastic_branch_agreement_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.branch_state_stabilizer:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_branch_state"): (
                builder if name == "sparse_context_aux" else (lambda builder=builder: branch_state_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.loop_inhibition:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_loop_inhib"): (
                builder if name == "sparse_context_aux" else (lambda builder=builder: loop_inhibition_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.segment_attractor_inhibition:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_segment_attractor"): (
                builder if name == "sparse_context_aux" else (lambda builder=builder: segment_attractor_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.loop_escape:
        builders = {
            (name if name == "sparse_context_aux" else f"{name}_loop_escape"): (
                builder if name == "sparse_context_aux" else (lambda builder=builder: loop_escape_wrap(builder()))
            )
            for name, builder in builders.items()
        }
    if args.low_precision_bits > 0:
        builders = {
            name: (builder if name == "sparse_context_aux" else (lambda builder=builder: LowPrecisionStateWrapper(
                builder(),
                args.low_precision_bits,
                args.low_precision_clip,
                args.low_precision_scale_mode,
                args.low_precision_targets,
                args.low_precision_bias_clip,
            )))
            for name, builder in builders.items()
        }

    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    generation_prompts = make_generation_prompts(
        valid_ids,
        args.completion_count,
        args.prompt_tokens,
        args.completion_tokens,
    )
    result_json: dict[str, Any] = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "token_counts": {
            "train_raw": int(len(train_raw)),
            "valid_raw": int(len(valid_raw)),
            "train_compact": int(len(train_ids)),
            "valid_compact": int(len(valid_ids)),
            "vocab_size": vocab_size,
        },
        "methods": {},
    }

    for method, builder in builders.items():
        memory = builder()
        start = time.perf_counter()
        warmup, warmup_history, warmup_rows = run_stream_pass(
            method,
            memory,
            train_ids,
            args.segment_tokens,
            start_history=[],
            update=True,
            pass_name="warmup_online",
            temperature=args.temperature,
        )
        warmup_seconds = time.perf_counter() - start
        retention_ids = train_ids[: min(len(train_ids), args.retention_token_limit)]
        retention_before = evaluate_sequence(
            clone_for_eval(memory, args.retention_reset_dynamic),
            retention_ids,
            [],
            args.temperature,
        )
        generation_rows.extend(
            build_generation_rows(
                method,
                "pre_online",
                "greedy",
                memory,
                tokenizer,
                kept_raw,
                generation_prompts,
                args.completion_tokens,
            )
        )
        if args.repetition_penalty > 0.0 or args.no_repeat_ngram > 1:
            generation_rows.extend(
                build_generation_rows(
                    method,
                    "pre_online",
                    "controlled",
                    memory,
                    tokenizer,
                    kept_raw,
                    generation_prompts,
                    args.completion_tokens,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram=args.no_repeat_ngram,
                )
            )

        start = time.perf_counter()
        stream_pre, _, pre_rows = run_stream_pass(
            method,
            clone_memory(memory),
            valid_ids,
            args.segment_tokens,
            start_history=warmup_history,
            update=False,
            pass_name="stream_pre",
            temperature=args.temperature,
        )
        pre_seconds = time.perf_counter() - start

        start = time.perf_counter()
        stream_online, final_history, online_rows = run_stream_pass(
            method,
            memory,
            valid_ids,
            args.segment_tokens,
            start_history=warmup_history,
            update=True,
            pass_name="stream_online",
            temperature=args.temperature,
        )
        online_seconds = time.perf_counter() - start

        start = time.perf_counter()
        stream_post, _, post_rows = run_stream_pass(
            method,
            clone_memory(memory),
            valid_ids,
            args.segment_tokens,
            start_history=warmup_history,
            update=False,
            pass_name="stream_post",
            temperature=args.temperature,
        )
        post_seconds = time.perf_counter() - start
        retention_after = evaluate_sequence(
            clone_for_eval(memory, args.retention_reset_dynamic),
            retention_ids,
            [],
            args.temperature,
        )
        generation_rows.extend(
            build_generation_rows(
                method,
                "post_online",
                "greedy",
                memory,
                tokenizer,
                kept_raw,
                generation_prompts,
                args.completion_tokens,
            )
        )
        if args.repetition_penalty > 0.0 or args.no_repeat_ngram > 1:
            generation_rows.extend(
                build_generation_rows(
                    method,
                    "post_online",
                    "controlled",
                    memory,
                    tokenizer,
                    kept_raw,
                    generation_prompts,
                    args.completion_tokens,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram=args.no_repeat_ngram,
                )
            )

        segment_rows.extend(warmup_rows + pre_rows + online_rows + post_rows)
        state_bytes = memory.state_bytes()
        serialized_bytes = serialized_state_bytes(memory)
        row = {
            "method": method,
            "warmup_loss": warmup["loss"],
            "warmup_acc": warmup["accuracy"],
            "stream_pre_loss": stream_pre["loss"],
            "stream_pre_acc": stream_pre["accuracy"],
            "stream_pre_target_rank": stream_pre["target_rank_mean"],
            "stream_pre_top4_acc": stream_pre["target_top4_acc"],
            "stream_pre_error_top4": stream_pre["error_target_top4_rate"],
            "stream_pre_oracle_top4_acc": stream_pre["oracle_top4_acc"],
            "stream_pre_error_margin": stream_pre["error_wrong_margin_mean"],
            "stream_online_loss": stream_online["loss"],
            "stream_online_acc": stream_online["accuracy"],
            "stream_online_target_rank": stream_online["target_rank_mean"],
            "stream_online_top4_acc": stream_online["target_top4_acc"],
            "stream_online_error_top4": stream_online["error_target_top4_rate"],
            "stream_online_oracle_top4_acc": stream_online["oracle_top4_acc"],
            "stream_online_error_margin": stream_online["error_wrong_margin_mean"],
            "stream_post_loss": stream_post["loss"],
            "stream_post_acc": stream_post["accuracy"],
            "stream_post_target_rank": stream_post["target_rank_mean"],
            "stream_post_top4_acc": stream_post["target_top4_acc"],
            "stream_post_error_top4": stream_post["error_target_top4_rate"],
            "stream_post_oracle_top4_acc": stream_post["oracle_top4_acc"],
            "stream_post_error_margin": stream_post["error_wrong_margin_mean"],
            "stream_delta_loss": stream_pre["loss"] - stream_post["loss"],
            "online_to_post_delta": stream_online["loss"] - stream_post["loss"],
            "retention_before_loss": retention_before["loss"],
            "retention_before_acc": retention_before["accuracy"],
            "retention_before_top4_acc": retention_before["target_top4_acc"],
            "retention_before_error_top4": retention_before["error_target_top4_rate"],
            "retention_before_oracle_top4_acc": retention_before["oracle_top4_acc"],
            "retention_after_loss": retention_after["loss"],
            "retention_after_acc": retention_after["accuracy"],
            "retention_after_top4_acc": retention_after["target_top4_acc"],
            "retention_after_error_top4": retention_after["error_target_top4_rate"],
            "retention_after_oracle_top4_acc": retention_after["oracle_top4_acc"],
            "warmup_targets": warmup["target_tokens"],
            "stream_targets": stream_post["target_tokens"],
            "warmup_seconds": warmup_seconds,
            "stream_pre_seconds": pre_seconds,
            "stream_online_seconds": online_seconds,
            "stream_post_seconds": post_seconds,
            "active_contexts": memory.active_contexts(),
            "state_bytes": state_bytes,
            "serialized_state_bytes": serialized_bytes,
            "pickle_state_bytes": state_pickle_bytes(memory),
            "bytes_per_target": state_bytes / max(warmup["target_tokens"] + stream_online["target_tokens"], 1),
            "serialized_bytes_per_target": serialized_bytes / max(warmup["target_tokens"] + stream_online["target_tokens"], 1),
            "stores_raw_text": False,
        }
        manifest = serialized_state_manifest(memory)
        if manifest:
            manifest_path = args.out_dir / f"{method}_serialized_state_manifest.json"
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            row["serialized_manifest"] = str(manifest_path)
        checkpoint_metadata = None
        if args.save_serialized_checkpoint:
            checkpoint_path = args.out_dir / f"{method}_serialized_state.npz"
            checkpoint_metadata = save_serialized_state(memory, checkpoint_path)
            if checkpoint_metadata is not None:
                checkpoint_metadata["path"] = str(checkpoint_path)
                row["serialized_checkpoint"] = str(checkpoint_path)
                row["checkpoint_file_bytes"] = int(checkpoint_path.stat().st_size)
                row["checkpoint_quantized_arrays"] = int(checkpoint_metadata["quantized_arrays"])
                row["checkpoint_raw_arrays"] = int(checkpoint_metadata["raw_arrays"])
        if args.checkpoint_parity_limit > 0:
            if checkpoint_metadata is None:
                raise ValueError("--checkpoint-parity-limit requires --save-serialized-checkpoint")
            loaded_memory = builder()
            loaded_metadata = load_serialized_state(loaded_memory, Path(str(checkpoint_metadata["path"])))
            if loaded_metadata is None:
                raise ValueError(f"method does not support serialized checkpoint loading: {method}")
            parity = checkpoint_prediction_parity(
                clone_memory(memory),
                loaded_memory,
                valid_ids,
                args.segment_tokens,
                warmup_history,
                args.temperature,
                args.checkpoint_parity_limit,
            )
            row.update(parity)
            checkpoint_metadata["parity"] = parity
        summary_rows.append(row)
        result_row = dict(row)
        if checkpoint_metadata is not None:
            result_row["checkpoint_metadata"] = checkpoint_metadata
        result_json["methods"][method] = result_row

    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "segment_metrics.csv", segment_rows)
    generation_summary_rows = summarize_generation_rows(generation_rows)
    write_csv(args.out_dir / "generation_metrics.csv", generation_rows)
    write_csv(args.out_dir / "generation_summary.csv", generation_summary_rows)
    write_generation_text(args.out_dir / "greedy_completions.txt", generation_rows)
    result_json["generation_summary"] = generation_summary_rows
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    print("Summary:")
    for row in summary_rows:
        print(
            f"  {row['method']}: pre={row['stream_pre_loss']:.3f}/{row['stream_pre_acc']:.3f} "
            f"online={row['stream_online_loss']:.3f}/{row['stream_online_acc']:.3f} "
            f"post={row['stream_post_loss']:.3f}/{row['stream_post_acc']:.3f} "
            f"bytes={int(row['state_bytes']):,}"
        )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
