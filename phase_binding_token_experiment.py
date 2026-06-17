#!/usr/bin/env python3
"""
Pure no-BP phase-binding next-token prototype.

The main model is a target-only local phase-binding learner:

    code_left[token_{t-2}] * code_right[token_{t-1}] -> target_token_t

The only teaching signal is the next-token class.  Updates use local complex
binding and conjugate attraction; there is no BP, BPTT, pretrained model, or
API in the learning rule.  Sparse context counts are reported only as an
auxiliary statistical baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = SCRIPT_DIR / "data" / "TinyStories-train.txt"
DEFAULT_VALID = SCRIPT_DIR / "data" / "TinyStories-valid.txt"
DEFAULT_TOKENIZER = Path("/private/zhenningshi/model_weights/Llama-3.2-1B-Instruct")


@dataclass
class PhaseTokenConfig:
    context_order: int = 2
    complex_dim: int = 96
    lr: float = 0.16
    prototype_lr: float = 0.0
    anti_hebbian_lr: float = 0.0
    anti_hebbian_warmup: int = 500
    epochs: int = 1
    logit_scale: float = 6.0
    bias_weight: float = 1.0
    temperature: float = 1.0
    seed: int = 0


@dataclass
class SparseAuxConfig:
    context_order: int = 2
    alpha: float = 0.10
    temperature: float = 1.0


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


def read_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


def encode_text(tokenizer: Any, text: str) -> np.ndarray:
    return np.array(tokenizer.encode(text, add_special_tokens=False), dtype=np.int64)


def build_compact_vocab(train_raw: np.ndarray, valid_raw: np.ndarray, max_vocab: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(train_raw)
    max_vocab = min(max_vocab, counts.shape[0])
    kept_raw = np.argsort(-counts)[:max_vocab]
    kept_raw = kept_raw[np.argsort(kept_raw)]
    raw_to_compact = np.full(max(int(max(train_raw.max(), valid_raw.max())) + 1, int(kept_raw.max()) + 1), -1, dtype=np.int64)
    raw_to_compact[kept_raw] = np.arange(len(kept_raw), dtype=np.int64)
    train_compact = raw_to_compact[train_raw]
    valid_compact = raw_to_compact[valid_raw]
    return kept_raw.astype(np.int64), train_compact[train_compact >= 0].astype(np.int64), valid_compact[valid_compact >= 0].astype(np.int64)


def normalize_vector(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x) + 1e-8)).astype(np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def random_complex_codes(rng: np.random.Generator, rows: int, complex_dim: int) -> np.ndarray:
    phases = rng.uniform(-math.pi, math.pi, (rows, complex_dim)).astype(np.float32)
    codes = np.empty((rows, 2 * complex_dim), dtype=np.float32)
    codes[:, 0::2] = np.cos(phases)
    codes[:, 1::2] = np.sin(phases)
    return normalize_rows(codes)


def conjugate_phase_vector(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    out[1::2] *= -1.0
    return out


def complex_bind_phase_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    bound = np.empty_like(a)
    bound[0::2] = a[0::2] * b[0::2] - a[1::2] * b[1::2]
    bound[1::2] = a[1::2] * b[0::2] + a[0::2] * b[1::2]
    return normalize_vector(bound)


def phase_identity(complex_dim: int) -> np.ndarray:
    identity = np.zeros(2 * complex_dim, dtype=np.float32)
    identity[0::2] = 1.0
    return normalize_vector(identity)


def bind_code_sequence(codes: list[np.ndarray], complex_dim: int) -> np.ndarray:
    if not codes:
        return phase_identity(complex_dim)
    bound = codes[0]
    for code in codes[1:]:
        bound = complex_bind_phase_vectors(bound, code)
    return normalize_vector(bound)


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / max(temperature, 1e-6)
    z = z - float(np.max(z))
    exp_z = np.exp(z)
    return (exp_z / np.sum(exp_z)).astype(np.float32)


def batch_metric_from_scores(scores: np.ndarray, target: int, temperature: float) -> tuple[float, int]:
    probs = softmax(scores, temperature)
    return -math.log(float(probs[target]) + 1e-9), int(np.argmax(probs) == target)


class PhaseBindingTokenLearner:
    def __init__(self, vocab_size: int, cfg: PhaseTokenConfig) -> None:
        self.vocab_size = vocab_size
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.code_banks = [
            random_complex_codes(self.rng, vocab_size, cfg.complex_dim)
            for _ in range(max(cfg.context_order, 1))
        ]
        self.target_anchors = random_complex_codes(self.rng, vocab_size, cfg.complex_dim)
        self.prototypes = np.zeros((vocab_size, 2 * cfg.complex_dim), dtype=np.float32)
        self.prototype_counts = np.zeros(vocab_size, dtype=np.float32)
        self.unigram_counts = np.ones(vocab_size, dtype=np.float32)
        self.output_bias = np.full(vocab_size, -math.log(max(vocab_size, 1)), dtype=np.float32)

    def feature(self, context: list[int] | np.ndarray) -> np.ndarray:
        codes = [self.code_banks[pos][int(token)] for pos, token in enumerate(context)]
        return bind_code_sequence(codes, self.cfg.complex_dim)

    def update_context(self, context: list[int] | np.ndarray, target: int) -> np.ndarray:
        context = [int(token) for token in context]
        target = int(target)
        target_anchor = self.target_anchors[target]
        codes = [self.code_banks[pos][token] for pos, token in enumerate(context)]
        for pos, token in enumerate(context):
            other_codes = [code for other_pos, code in enumerate(codes) if other_pos != pos]
            other_bound = bind_code_sequence(other_codes, self.cfg.complex_dim)
            desired_code = complex_bind_phase_vectors(target_anchor, conjugate_phase_vector(other_bound))
            new_code = normalize_vector((1.0 - self.cfg.lr) * codes[pos] + self.cfg.lr * desired_code)
            self.code_banks[pos][token] = new_code
            codes[pos] = new_code

        feature = bind_code_sequence(codes, self.cfg.complex_dim)
        count = self.prototype_counts[target]
        if self.cfg.prototype_lr > 0.0 and count > 0.0:
            self.prototypes[target] = normalize_vector(
                (1.0 - self.cfg.prototype_lr) * self.prototypes[target] + self.cfg.prototype_lr * feature
            )
        else:
            self.prototypes[target] = normalize_vector((self.prototypes[target] * count + feature) / (count + 1.0))
        self.prototype_counts[target] = count + 1.0
        self.unigram_counts[target] += 1.0
        probs = self.unigram_counts / float(np.sum(self.unigram_counts))
        self.output_bias = np.log(np.maximum(probs, 1e-9)).astype(np.float32)
        return feature

    def train(self, ids: np.ndarray) -> None:
        order = max(self.cfg.context_order, 1)
        if len(ids) <= order:
            return
        step = 0
        for _ in range(self.cfg.epochs):
            for idx in range(order, len(ids)):
                context = [int(token) for token in ids[idx - order : idx]]
                target = int(ids[idx])
                feature = self.update_context(context, target)
                if self.cfg.anti_hebbian_lr > 0.0 and step >= self.cfg.anti_hebbian_warmup:
                    scores = self.prototypes @ feature
                    scores[target] = -1e9
                    wrong = int(np.argmax(scores))
                    if self.prototype_counts[wrong] > 0:
                        self.prototypes[wrong] = normalize_vector(
                            self.prototypes[wrong] - self.cfg.anti_hebbian_lr * feature
                        )
                step += 1

    def scores(self, context: list[int] | np.ndarray, use_bias: bool) -> np.ndarray:
        feature = self.feature(context)
        scores = self.cfg.logit_scale * (self.prototypes @ feature)
        if use_bias:
            scores = scores + self.cfg.bias_weight * self.output_bias
        return scores.astype(np.float32)

    def state_bytes(self) -> int:
        return int(
            sum(bank.nbytes for bank in self.code_banks)
            + self.target_anchors.nbytes
            + self.prototypes.nbytes
            + self.prototype_counts.nbytes
            + self.unigram_counts.nbytes
            + self.output_bias.nbytes
        )


class SparseContextAux:
    def __init__(self, vocab_size: int, cfg: SparseAuxConfig) -> None:
        self.vocab_size = vocab_size
        self.cfg = cfg
        self.rows: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.unigram = Counter()

    def train(self, ids: np.ndarray) -> None:
        order = max(self.cfg.context_order, 1)
        for idx in range(order, len(ids)):
            context = tuple(int(token) for token in ids[idx - order : idx])
            target = int(ids[idx])
            self.rows[context][target] += 1
            self.unigram[target] += 1

    def distribution(self, context: list[int] | np.ndarray) -> np.ndarray:
        base = np.ones(self.vocab_size, dtype=np.float32)
        for token, count in self.unigram.items():
            base[int(token)] += float(count)
        base /= float(np.sum(base))
        probs = self.cfg.alpha * base
        row = self.rows.get(tuple(int(token) for token in context))
        if row:
            row_total = float(sum(row.values()))
            for token, count in row.items():
                probs[int(token)] += (1.0 - self.cfg.alpha) * float(count) / max(row_total, 1.0)
        probs /= float(np.sum(probs))
        return probs

    def state_bytes_estimate(self) -> int:
        active_entries = sum(len(row) for row in self.rows.values())
        return int(len(self.rows) * max(self.cfg.context_order, 1) * 8 + active_entries * 2 * 8 + len(self.unigram) * 2 * 8)


class BranchPhaseTokenLearner:
    """Dendritic-style fixed branch sum over several local phase-binding learners."""

    def __init__(self, vocab_size: int, base_cfg: PhaseTokenConfig, branch_orders: list[int], branch_weights: list[float]) -> None:
        if len(branch_orders) != len(branch_weights):
            raise ValueError("branch_orders and branch_weights must have the same length")
        if not branch_orders:
            raise ValueError("at least one branch order is required")
        self.vocab_size = vocab_size
        self.branch_orders = [max(int(order), 1) for order in branch_orders]
        self.branch_weights = [float(weight) for weight in branch_weights]
        self.max_order = max(self.branch_orders)
        self.cfg = base_cfg
        self.branches: list[PhaseBindingTokenLearner] = []
        for branch_idx, order in enumerate(self.branch_orders):
            branch_cfg = PhaseTokenConfig(
                context_order=order,
                complex_dim=base_cfg.complex_dim,
                lr=base_cfg.lr,
                prototype_lr=base_cfg.prototype_lr,
                anti_hebbian_lr=base_cfg.anti_hebbian_lr,
                anti_hebbian_warmup=base_cfg.anti_hebbian_warmup,
                epochs=base_cfg.epochs,
                logit_scale=base_cfg.logit_scale,
                bias_weight=base_cfg.bias_weight,
                temperature=base_cfg.temperature,
                seed=base_cfg.seed + 1009 * branch_idx,
            )
            self.branches.append(PhaseBindingTokenLearner(vocab_size, branch_cfg))
        self.output_bias = np.full(vocab_size, -math.log(max(vocab_size, 1)), dtype=np.float32)

    def train(self, ids: np.ndarray) -> None:
        for branch in self.branches:
            branch.train(ids)
        largest_branch_idx = int(np.argmax(self.branch_orders))
        self.output_bias = self.branches[largest_branch_idx].output_bias.copy()

    def update_context(self, context: list[int] | np.ndarray, target: int) -> None:
        for order, branch in zip(self.branch_orders, self.branches):
            branch.update_context(context[-order:], target)
        largest_branch_idx = int(np.argmax(self.branch_orders))
        self.output_bias = self.branches[largest_branch_idx].output_bias.copy()

    def scores(self, context: np.ndarray, use_bias: bool) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        for order, weight, branch in zip(self.branch_orders, self.branch_weights, self.branches):
            scores += weight * branch.scores(context[-order:], use_bias=False)
        if use_bias:
            scores += self.cfg.bias_weight * self.output_bias
        return scores.astype(np.float32)

    def state_bytes(self) -> int:
        return int(sum(branch.state_bytes() for branch in self.branches) + self.output_bias.nbytes)


class CompetitiveBranchReadout:
    """
    Local winner-take-all readout over fixed phase branches.

    The branch encoders are trained with target-only phase binding.  This
    readout then applies a local perceptron-like rule: pull the target row
    toward the current feature and push the top wrong winner rows away.
    """

    def __init__(
        self,
        branch_model: BranchPhaseTokenLearner,
        lr: float,
        neg_k: int,
        epochs: int,
        score_scale: float,
        init: str = "average",
        margin: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.branch_model = branch_model
        self.vocab_size = branch_model.vocab_size
        self.lr = lr
        self.neg_k = max(int(neg_k), 0)
        self.epochs = max(int(epochs), 1)
        self.score_scale = score_scale
        self.init = init
        self.margin = margin
        self.rng = np.random.default_rng(seed)
        self.max_order = branch_model.max_order
        self.feature_dim = sum(2 * branch.cfg.complex_dim for branch in branch_model.branches)
        if init == "random":
            self.weights = normalize_rows(
                self.rng.normal(0.0, 0.01, (self.vocab_size, self.feature_dim)).astype(np.float32)
            )
        else:
            self.weights = normalize_rows(
                np.concatenate([branch.prototypes for branch in branch_model.branches], axis=1).astype(np.float32)
            )
        self.output_bias = branch_model.output_bias.copy()

    def feature(self, context: np.ndarray) -> np.ndarray:
        features = []
        for order, branch in zip(self.branch_model.branch_orders, self.branch_model.branches):
            features.append(branch.feature(context[-order:]))
        return normalize_vector(np.concatenate(features).astype(np.float32))

    def train(self, ids: np.ndarray) -> None:
        if len(ids) <= self.max_order:
            return
        for _ in range(self.epochs):
            for idx in range(self.max_order, len(ids)):
                self.update_context(ids[idx - self.max_order : idx], int(ids[idx]))

    def update_context(self, context: list[int] | np.ndarray, target: int) -> None:
        feature = self.feature(np.asarray(context, dtype=np.int64))
        target = int(target)
        scores = self.weights @ feature
        self.weights[target] = normalize_vector(self.weights[target] + self.lr * feature)
        if self.neg_k > 0:
            scores[target] = -1e9
            k = min(self.neg_k, self.vocab_size - 1)
            wrongs = np.argpartition(scores, -k)[-k:]
            target_score = float(self.weights[target] @ feature)
            for wrong in wrongs:
                if float(scores[wrong]) + self.margin > target_score:
                    self.weights[wrong] = normalize_vector(self.weights[wrong] - (self.lr / k) * feature)

    def scores(self, context: np.ndarray, bias_weight: float) -> np.ndarray:
        return (self.score_scale * (self.weights @ self.feature(context)) + bias_weight * self.output_bias).astype(np.float32)

    def state_bytes(self) -> int:
        return int(self.weights.nbytes + self.output_bias.nbytes)


def evaluate_phase(model: PhaseBindingTokenLearner, ids: np.ndarray, limit: int, use_bias: bool) -> dict[str, Any]:
    order = max(model.cfg.context_order, 1)
    eval_steps = min(max(len(ids) - order, 0), limit)
    losses: list[float] = []
    correct = 0
    for offset in range(eval_steps):
        idx = offset + order
        loss, is_correct = batch_metric_from_scores(
            model.scores(ids[idx - order : idx], use_bias=use_bias),
            int(ids[idx]),
            model.cfg.temperature,
        )
        losses.append(loss)
        correct += is_correct
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": correct / max(eval_steps, 1),
        "eval_tokens": eval_steps,
        "state_bytes": model.state_bytes(),
        "stores_raw_text": False,
    }


def evaluate_branch(model: BranchPhaseTokenLearner, ids: np.ndarray, limit: int, use_bias: bool) -> dict[str, Any]:
    eval_steps = min(max(len(ids) - model.max_order, 0), limit)
    losses: list[float] = []
    correct = 0
    for offset in range(eval_steps):
        idx = offset + model.max_order
        loss, is_correct = batch_metric_from_scores(
            model.scores(ids[idx - model.max_order : idx], use_bias=use_bias),
            int(ids[idx]),
            model.cfg.temperature,
        )
        losses.append(loss)
        correct += is_correct
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": correct / max(eval_steps, 1),
        "eval_tokens": eval_steps,
        "state_bytes": model.state_bytes(),
        "stores_raw_text": False,
        "branch_orders": ",".join(str(order) for order in model.branch_orders),
        "branch_weights": ",".join(f"{weight:g}" for weight in model.branch_weights),
    }


def evaluate_competitive_readout(
    model: CompetitiveBranchReadout,
    ids: np.ndarray,
    limit: int,
    bias_weight: float,
    temperature: float,
) -> dict[str, Any]:
    eval_steps = min(max(len(ids) - model.max_order, 0), limit)
    losses: list[float] = []
    correct = 0
    for offset in range(eval_steps):
        idx = offset + model.max_order
        loss, is_correct = batch_metric_from_scores(
            model.scores(ids[idx - model.max_order : idx], bias_weight=bias_weight),
            int(ids[idx]),
            temperature,
        )
        losses.append(loss)
        correct += is_correct
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": correct / max(eval_steps, 1),
        "eval_tokens": eval_steps,
        "state_bytes": model.state_bytes(),
        "stores_raw_text": False,
        "competitive_lr": model.lr,
        "competitive_neg_k": model.neg_k,
        "competitive_epochs": model.epochs,
        "competitive_init": model.init,
        "competitive_score_scale": model.score_scale,
    }


def evaluate_unigram(model: PhaseBindingTokenLearner, ids: np.ndarray, limit: int) -> dict[str, Any]:
    order = max(model.cfg.context_order, 1)
    eval_steps = min(max(len(ids) - order, 0), limit)
    losses: list[float] = []
    correct = 0
    pred = int(np.argmax(model.output_bias))
    probs = np.exp(model.output_bias)
    probs /= float(np.sum(probs))
    for offset in range(eval_steps):
        target = int(ids[offset + order])
        losses.append(-math.log(float(probs[target]) + 1e-9))
        correct += int(pred == target)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": correct / max(eval_steps, 1),
        "eval_tokens": eval_steps,
        "state_bytes": int(model.unigram_counts.nbytes + model.output_bias.nbytes),
        "stores_raw_text": False,
    }


def evaluate_sparse(model: SparseContextAux, ids: np.ndarray, limit: int) -> dict[str, Any]:
    order = max(model.cfg.context_order, 1)
    eval_steps = min(max(len(ids) - order, 0), limit)
    losses: list[float] = []
    correct = 0
    for offset in range(eval_steps):
        idx = offset + order
        probs = model.distribution(ids[idx - order : idx])
        target = int(ids[idx])
        losses.append(-math.log(float(probs[target]) + 1e-9))
        correct += int(np.argmax(probs) == target)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": correct / max(eval_steps, 1),
        "eval_tokens": eval_steps,
        "active_contexts": len(model.rows),
        "active_entries": sum(len(row) for row in model.rows.values()),
        "state_bytes_estimate": model.state_bytes_estimate(),
        "stores_raw_text": False,
        "auxiliary_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "phase_binding_token")
    parser.add_argument("--train-chars", type=int, default=10_000)
    parser.add_argument("--valid-chars", type=int, default=3_000)
    parser.add_argument("--max-vocab", type=int, default=128)
    parser.add_argument("--eval-token-limit", type=int, default=1_000)
    parser.add_argument("--context-order", type=int, default=2)
    parser.add_argument("--branch-orders", type=int, nargs="*", default=[])
    parser.add_argument("--branch-weights", type=float, nargs="*", default=[])
    parser.add_argument("--competitive-readout", action="store_true")
    parser.add_argument("--competitive-lr", type=float, default=0.02)
    parser.add_argument("--competitive-neg-k", type=int, default=8)
    parser.add_argument("--competitive-epochs", type=int, default=2)
    parser.add_argument("--competitive-score-scale", type=float, default=8.0)
    parser.add_argument("--competitive-init", choices=["average", "random"], default="average")
    parser.add_argument("--competitive-margin", type=float, default=0.0)
    parser.add_argument("--phase-dim", type=int, default=96)
    parser.add_argument("--phase-lr", type=float, default=0.16)
    parser.add_argument("--prototype-lr", type=float, default=0.0)
    parser.add_argument("--anti-hebbian-lr", type=float, default=0.0)
    parser.add_argument("--anti-hebbian-warmup", type=int, default=500)
    parser.add_argument("--phase-epochs", type=int, default=1)
    parser.add_argument("--phase-logit-scale", type=float, default=6.0)
    parser.add_argument("--phase-bias-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sparse-alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    train_text = read_prefix(args.train_file, args.train_chars)
    valid_text = read_prefix(args.valid_file, args.valid_chars)
    train_raw = encode_text(tokenizer, train_text)
    valid_raw = encode_text(tokenizer, valid_text)
    kept_raw, train_ids, valid_ids = build_compact_vocab(train_raw, valid_raw, args.max_vocab)
    vocab_size = int(len(kept_raw))

    rows: list[dict[str, Any]] = []

    phase_cfg = PhaseTokenConfig(
        context_order=args.context_order,
        complex_dim=args.phase_dim,
        lr=args.phase_lr,
        prototype_lr=args.prototype_lr,
        anti_hebbian_lr=args.anti_hebbian_lr,
        anti_hebbian_warmup=args.anti_hebbian_warmup,
        epochs=args.phase_epochs,
        logit_scale=args.phase_logit_scale,
        bias_weight=args.phase_bias_weight,
        temperature=args.temperature,
        seed=args.seed,
    )
    start = time.perf_counter()
    phase_model = PhaseBindingTokenLearner(vocab_size, phase_cfg)
    phase_model.train(train_ids)
    phase_train_seconds = time.perf_counter() - start

    for method, use_bias in [
        ("phase_binding_token_no_bias", False),
        ("phase_binding_token", True),
    ]:
        start = time.perf_counter()
        metrics = evaluate_phase(phase_model, valid_ids, args.eval_token_limit, use_bias=use_bias)
        eval_seconds = time.perf_counter() - start
        rows.append(
            {
                "method": method,
                **metrics,
                "train_seconds": phase_train_seconds,
                "eval_seconds": eval_seconds,
                "train_tokens": max(len(train_ids) - args.context_order, 0) * args.phase_epochs,
                "train_tokens_per_sec": (max(len(train_ids) - args.context_order, 0) * args.phase_epochs) / max(phase_train_seconds, 1e-9),
                "context_order": args.context_order,
                "vocab_size": vocab_size,
            }
        )

    rows.append(
        {
            "method": "unigram_aux",
            **evaluate_unigram(phase_model, valid_ids, args.eval_token_limit),
            "train_seconds": phase_train_seconds,
            "eval_seconds": 0.0,
            "train_tokens": max(len(train_ids) - args.context_order, 0) * args.phase_epochs,
            "context_order": args.context_order,
            "vocab_size": vocab_size,
            "auxiliary_only": True,
        }
    )

    if args.branch_orders:
        branch_weights = args.branch_weights if args.branch_weights else [1.0 / len(args.branch_orders)] * len(args.branch_orders)
        start = time.perf_counter()
        branch_model = BranchPhaseTokenLearner(vocab_size, phase_cfg, args.branch_orders, branch_weights)
        branch_model.train(train_ids)
        branch_train_seconds = time.perf_counter() - start
        for method, use_bias in [
            ("phase_branch_token_no_bias", False),
            ("phase_branch_token", True),
        ]:
            start = time.perf_counter()
            metrics = evaluate_branch(branch_model, valid_ids, args.eval_token_limit, use_bias=use_bias)
            eval_seconds = time.perf_counter() - start
            rows.append(
                {
                    "method": method,
                    **metrics,
                    "train_seconds": branch_train_seconds,
                    "eval_seconds": eval_seconds,
                    "train_tokens": sum(max(len(train_ids) - max(order, 1), 0) for order in args.branch_orders) * args.phase_epochs,
                    "train_tokens_per_sec": (
                        sum(max(len(train_ids) - max(order, 1), 0) for order in args.branch_orders) * args.phase_epochs
                    )
                    / max(branch_train_seconds, 1e-9),
                    "context_order": max(args.branch_orders),
                    "vocab_size": vocab_size,
                }
            )
        if args.competitive_readout:
            start = time.perf_counter()
            competitive = CompetitiveBranchReadout(
                branch_model,
                lr=args.competitive_lr,
                neg_k=args.competitive_neg_k,
                epochs=args.competitive_epochs,
                score_scale=args.competitive_score_scale,
                init=args.competitive_init,
                margin=args.competitive_margin,
                seed=args.seed,
            )
            competitive.train(train_ids)
            competitive_train_seconds = time.perf_counter() - start
            start = time.perf_counter()
            metrics = evaluate_competitive_readout(
                competitive,
                valid_ids,
                args.eval_token_limit,
                bias_weight=args.phase_bias_weight,
                temperature=args.temperature,
            )
            eval_seconds = time.perf_counter() - start
            rows.append(
                {
                    "method": "phase_competitive_branch_readout",
                    **metrics,
                    "train_seconds": branch_train_seconds + competitive_train_seconds,
                    "readout_train_seconds": competitive_train_seconds,
                    "eval_seconds": eval_seconds,
                    "train_tokens": (
                        sum(max(len(train_ids) - max(order, 1), 0) for order in args.branch_orders)
                        + max(len(train_ids) - max(args.branch_orders), 0) * args.competitive_epochs
                    ),
                    "train_tokens_per_sec": (
                        sum(max(len(train_ids) - max(order, 1), 0) for order in args.branch_orders)
                        + max(len(train_ids) - max(args.branch_orders), 0) * args.competitive_epochs
                    )
                    / max(branch_train_seconds + competitive_train_seconds, 1e-9),
                    "context_order": max(args.branch_orders),
                    "branch_orders": ",".join(str(order) for order in args.branch_orders),
                    "branch_weights": ",".join(f"{weight:g}" for weight in branch_weights),
                    "vocab_size": vocab_size,
                }
            )

    sparse_cfg = SparseAuxConfig(context_order=args.context_order, alpha=args.sparse_alpha, temperature=args.temperature)
    start = time.perf_counter()
    sparse = SparseContextAux(vocab_size, sparse_cfg)
    sparse.train(train_ids)
    sparse_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    sparse_metrics = evaluate_sparse(sparse, valid_ids, args.eval_token_limit)
    sparse_eval_seconds = time.perf_counter() - start
    rows.append(
        {
            "method": "sparse_context_aux",
            **sparse_metrics,
            "train_seconds": sparse_train_seconds,
            "eval_seconds": sparse_eval_seconds,
            "train_tokens": max(len(train_ids) - args.context_order, 0),
            "train_tokens_per_sec": max(len(train_ids) - args.context_order, 0) / max(sparse_train_seconds, 1e-9),
            "context_order": args.context_order,
            "vocab_size": vocab_size,
        }
    )

    write_csv(args.out_dir / "metrics.csv", rows)
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "phase_config": asdict(phase_cfg),
                "branch_orders": args.branch_orders,
                "branch_weights": branch_weights if args.branch_orders else [],
                "competitive_readout": args.competitive_readout,
                "sparse_aux_config": asdict(sparse_cfg),
                "token_counts": {
                    "train_raw": int(len(train_raw)),
                    "valid_raw": int(len(valid_raw)),
                    "train_compact": int(len(train_ids)),
                    "valid_compact": int(len(valid_ids)),
                    "vocab_size": vocab_size,
                },
                "rows": rows,
            },
            f,
            indent=2,
            default=str,
        )

    print("Summary:")
    for row in rows:
        print(f"  {row['method']}: loss={row['loss']:.3f} acc={row['accuracy']:.3f}")
    print(f"wrote metrics: {args.out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
