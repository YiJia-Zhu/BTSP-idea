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

    def scores(self, context: np.ndarray) -> np.ndarray:
        gate = self.effective_gate(context)
        return (self.base.scores(context) + self.strength * (self.calibration @ gate)).astype(np.float32)

    def learn_calibration(self, scores: np.ndarray, target: int, gate: np.ndarray) -> None:
        if self.decay < 1.0:
            self.calibration *= self.decay
        if self.lr <= 0.0:
            return
        target = int(target)
        pred = int(np.argmax(scores))
        if pred == target:
            return
        delta = self.lr * gate
        self.calibration[target] = np.clip(self.calibration[target] + delta, -self.clip, self.clip)
        self.calibration[pred] = np.clip(self.calibration[pred] - delta, -self.clip, self.clip)

    def update(self, context: np.ndarray, target: int) -> None:
        gate = self.effective_gate(context)
        self.learn_calibration(self.scores(context), target, gate)
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
    for segment_idx, (start, end) in enumerate(segment_windows(ids, segment_tokens)):
        segment = ids[start:end]
        segment_history = list(history)
        seg_loss = 0.0
        seg_correct = 0
        seg_total = 0
        for idx in range(len(segment) - 1):
            current = int(segment[idx])
            target = int(segment[idx + 1])
            context_list = truncate_history(segment_history + [current], order)
            if len(context_list) < order:
                segment_history = truncate_history(segment_history + [current], order - 1)
                continue
            context = np.array(context_list, dtype=np.int64)
            loss, pred = softmax_loss_and_pred(memory.scores(context), target, temperature)
            seg_loss += loss
            seg_correct += int(pred == target)
            seg_total += 1
            if update:
                memory.update(context, target)
            elif hasattr(memory, "observe"):
                memory.observe(context, target)
            segment_history = truncate_history(segment_history + [current], order - 1)
        history = segment_history
        loss_sum += seg_loss
        correct += seg_correct
        total += seg_total
        processed += seg_total
        seg_summary = summarize(seg_loss, seg_correct, seg_total)
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
    parser.add_argument("--readout-gain", type=float, default=1.0)
    parser.add_argument("--readout-gain-mode", choices=["fixed", "margin"], default="fixed")
    parser.add_argument("--readout-gain-margin-center", type=float, default=1.0)
    parser.add_argument("--readout-gain-margin-scale", type=float, default=1.0)
    parser.add_argument("--readout-gain-min", type=float, default=0.5)
    parser.add_argument("--readout-gain-max", type=float, default=3.0)
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
            seed=args.seed,
            derived_codes=args.feature_calibration_derived_codes,
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
        retention_before = evaluate_sequence(clone_memory(memory), retention_ids, [], args.temperature)
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
        retention_after = evaluate_sequence(clone_memory(memory), retention_ids, [], args.temperature)
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
            "stream_online_loss": stream_online["loss"],
            "stream_online_acc": stream_online["accuracy"],
            "stream_post_loss": stream_post["loss"],
            "stream_post_acc": stream_post["accuracy"],
            "stream_delta_loss": stream_pre["loss"] - stream_post["loss"],
            "online_to_post_delta": stream_online["loss"] - stream_post["loss"],
            "retention_before_loss": retention_before["loss"],
            "retention_before_acc": retention_before["accuracy"],
            "retention_after_loss": retention_after["loss"],
            "retention_after_acc": retention_after["accuracy"],
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
