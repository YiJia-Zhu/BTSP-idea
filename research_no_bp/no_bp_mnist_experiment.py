#!/usr/bin/env python3
"""
MNIST pilot for backprop alternatives.

Methods:
  - bp: ordinary backprop MLP baseline.
  - dfa_3factor: direct feedback alignment written as a three-factor local rule.
  - dfa_resampled: same local rule, but feedback is resampled every batch.
  - output_only: fixed random hidden layer with local output delta rule only.
  - zo_spsa: zeroth-order simultaneous perturbation stochastic approximation.
  - stdp: existing unsupervised competitive STDP feature learner + label voting.

The experiment is intentionally small and pure NumPy so it can be inspected and
modified quickly. It is not meant to be a state-of-the-art MNIST benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mnist_backprop_vs_stdp import (
    STDPConfig,
    SimpleSTDPClassifier,
    accuracy,
    one_hot,
    prepare_data,
)


NUM_CLASSES = 10


@dataclass
class MethodResult:
    method: str
    seed: int
    train_size: int
    test_size: int
    input_dim: int
    hidden_dim: int
    epochs: int
    test_acc: float
    train_time_sec: float
    final_train_loss: float | None
    final_train_acc: float | None
    mean_feedback_alignment: float | None


class TinyMLP:
    def __init__(self, input_dim: int, hidden_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), (hidden_dim, input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (NUM_CLASSES, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(NUM_CLASSES, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.w1.T + self.b1
        h = np.maximum(z1, 0.0)
        logits = h @ self.w2.T + self.b2
        probs = softmax(logits)
        return z1, h, probs

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, _, probs = self.forward(x)
        return probs.argmax(axis=1)

    def loss_acc(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        _, _, probs = self.forward(x)
        return cross_entropy(probs, y), accuracy(probs.argmax(axis=1), y)

    def pack(self) -> np.ndarray:
        return np.concatenate(
            [
                self.w1.ravel(),
                self.b1.ravel(),
                self.w2.ravel(),
                self.b2.ravel(),
            ]
        ).astype(np.float32)

    def unpack(self, theta: np.ndarray) -> None:
        p = 0
        n = self.w1.size
        self.w1 = theta[p : p + n].reshape(self.w1.shape).astype(np.float32)
        p += n
        n = self.b1.size
        self.b1 = theta[p : p + n].reshape(self.b1.shape).astype(np.float32)
        p += n
        n = self.w2.size
        self.w2 = theta[p : p + n].reshape(self.w2.shape).astype(np.float32)
        p += n
        n = self.b2.size
        self.b2 = theta[p : p + n].reshape(self.b2.shape).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(probs[np.arange(y.shape[0]), y] + 1e-8).mean())


def minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        idx = perm[start : start + batch_size]
        yield x[idx], y[idx]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def train_bp(
    model: TinyMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[MethodResult, list[dict]]:
    rng = np.random.default_rng(seed + 101)
    history = []
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        for xb, yb in minibatches(x_train, y_train, args.batch_size, rng):
            z1, h, probs = model.forward(xb)
            yoh = one_hot(yb, NUM_CLASSES)
            grad_logits = (probs - yoh) / xb.shape[0]

            grad_w2 = grad_logits.T @ h
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ model.w2
            grad_z1 = grad_h * (z1 > 0.0)
            grad_w1 = grad_z1.T @ xb
            grad_b1 = grad_z1.sum(axis=0)

            model.w2 -= args.lr * grad_w2.astype(np.float32)
            model.b2 -= args.lr * grad_b2.astype(np.float32)
            model.w1 -= args.lr * grad_w1.astype(np.float32)
            model.b1 -= args.lr * grad_b1.astype(np.float32)

        train_loss, train_acc = model.loss_acc(x_train, y_train)
        test_acc = accuracy(model.predict(x_test), y_test)
        history.append(
            {
                "method": "bp",
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "feedback_alignment": None,
            }
        )

    elapsed = time.perf_counter() - start_time
    final = history[-1]
    result = MethodResult(
        method="bp",
        seed=seed,
        train_size=x_train.shape[0],
        test_size=x_test.shape[0],
        input_dim=x_train.shape[1],
        hidden_dim=model.w1.shape[0],
        epochs=args.epochs,
        test_acc=float(final["test_acc"]),
        train_time_sec=elapsed,
        final_train_loss=float(final["train_loss"]),
        final_train_acc=float(final["train_acc"]),
        mean_feedback_alignment=None,
    )
    return result, history


def train_dfa_3factor(
    model: TinyMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[MethodResult, list[dict]]:
    rng = np.random.default_rng(seed + 202)
    feedback = rng.normal(0.0, 1.0 / math.sqrt(NUM_CLASSES), (model.w1.shape[0], NUM_CLASSES)).astype(np.float32)
    history = []
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_alignments = []
        for xb, yb in minibatches(x_train, y_train, args.batch_size, rng):
            z1, h, probs = model.forward(xb)
            yoh = one_hot(yb, NUM_CLASSES)
            output_error = (probs - yoh) / xb.shape[0]

            grad_w2 = output_error.T @ h
            grad_b2 = output_error.sum(axis=0)

            bp_hidden_signal = output_error @ model.w2
            random_hidden_signal = output_error @ feedback.T
            epoch_alignments.append(cosine(bp_hidden_signal, random_hidden_signal))

            # Three-factor local update: pre-synaptic x, post-synaptic ReLU
            # eligibility, and a neuron-local modulatory signal from random feedback.
            local_factor = random_hidden_signal * (z1 > 0.0)
            grad_w1 = local_factor.T @ xb
            grad_b1 = local_factor.sum(axis=0)

            model.w2 -= args.lr * grad_w2.astype(np.float32)
            model.b2 -= args.lr * grad_b2.astype(np.float32)
            model.w1 -= args.dfa_lr * grad_w1.astype(np.float32)
            model.b1 -= args.dfa_lr * grad_b1.astype(np.float32)

        train_loss, train_acc = model.loss_acc(x_train, y_train)
        test_acc = accuracy(model.predict(x_test), y_test)
        history.append(
            {
                "method": "dfa_3factor",
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "feedback_alignment": float(np.mean(epoch_alignments)),
            }
        )

    elapsed = time.perf_counter() - start_time
    final = history[-1]
    result = MethodResult(
        method="dfa_3factor",
        seed=seed,
        train_size=x_train.shape[0],
        test_size=x_test.shape[0],
        input_dim=x_train.shape[1],
        hidden_dim=model.w1.shape[0],
        epochs=args.epochs,
        test_acc=float(final["test_acc"]),
        train_time_sec=elapsed,
        final_train_loss=float(final["train_loss"]),
        final_train_acc=float(final["train_acc"]),
        mean_feedback_alignment=float(np.mean([row["feedback_alignment"] for row in history])),
    )
    return result, history


def train_dfa_resampled(
    model: TinyMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[MethodResult, list[dict]]:
    rng = np.random.default_rng(seed + 232)
    history = []
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_alignments = []
        for xb, yb in minibatches(x_train, y_train, args.batch_size, rng):
            feedback = rng.normal(0.0, 1.0 / math.sqrt(NUM_CLASSES), (model.w1.shape[0], NUM_CLASSES)).astype(np.float32)
            z1, h, probs = model.forward(xb)
            yoh = one_hot(yb, NUM_CLASSES)
            output_error = (probs - yoh) / xb.shape[0]

            grad_w2 = output_error.T @ h
            grad_b2 = output_error.sum(axis=0)

            bp_hidden_signal = output_error @ model.w2
            random_hidden_signal = output_error @ feedback.T
            epoch_alignments.append(cosine(bp_hidden_signal, random_hidden_signal))

            local_factor = random_hidden_signal * (z1 > 0.0)
            grad_w1 = local_factor.T @ xb
            grad_b1 = local_factor.sum(axis=0)

            model.w2 -= args.lr * grad_w2.astype(np.float32)
            model.b2 -= args.lr * grad_b2.astype(np.float32)
            model.w1 -= args.dfa_lr * grad_w1.astype(np.float32)
            model.b1 -= args.dfa_lr * grad_b1.astype(np.float32)

        train_loss, train_acc = model.loss_acc(x_train, y_train)
        test_acc = accuracy(model.predict(x_test), y_test)
        history.append(
            {
                "method": "dfa_resampled",
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "feedback_alignment": float(np.mean(epoch_alignments)),
            }
        )

    elapsed = time.perf_counter() - start_time
    final = history[-1]
    result = MethodResult(
        method="dfa_resampled",
        seed=seed,
        train_size=x_train.shape[0],
        test_size=x_test.shape[0],
        input_dim=x_train.shape[1],
        hidden_dim=model.w1.shape[0],
        epochs=args.epochs,
        test_acc=float(final["test_acc"]),
        train_time_sec=elapsed,
        final_train_loss=float(final["train_loss"]),
        final_train_acc=float(final["train_acc"]),
        mean_feedback_alignment=float(np.mean([row["feedback_alignment"] for row in history])),
    )
    return result, history


def train_output_only(
    model: TinyMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[MethodResult, list[dict]]:
    rng = np.random.default_rng(seed + 252)
    history = []
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        for xb, yb in minibatches(x_train, y_train, args.batch_size, rng):
            _, h, probs = model.forward(xb)
            yoh = one_hot(yb, NUM_CLASSES)
            output_error = (probs - yoh) / xb.shape[0]
            grad_w2 = output_error.T @ h
            grad_b2 = output_error.sum(axis=0)
            model.w2 -= args.lr * grad_w2.astype(np.float32)
            model.b2 -= args.lr * grad_b2.astype(np.float32)

        train_loss, train_acc = model.loss_acc(x_train, y_train)
        test_acc = accuracy(model.predict(x_test), y_test)
        history.append(
            {
                "method": "output_only",
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "feedback_alignment": None,
            }
        )

    elapsed = time.perf_counter() - start_time
    final = history[-1]
    result = MethodResult(
        method="output_only",
        seed=seed,
        train_size=x_train.shape[0],
        test_size=x_test.shape[0],
        input_dim=x_train.shape[1],
        hidden_dim=model.w1.shape[0],
        epochs=args.epochs,
        test_acc=float(final["test_acc"]),
        train_time_sec=elapsed,
        final_train_loss=float(final["train_loss"]),
        final_train_acc=float(final["train_acc"]),
        mean_feedback_alignment=None,
    )
    return result, history


def train_zo_spsa(
    model: TinyMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[MethodResult, list[dict]]:
    rng = np.random.default_rng(seed + 303)
    theta = model.pack()
    history = []
    start_time = time.perf_counter()

    def batch_loss(theta_value: np.ndarray, xb: np.ndarray, yb: np.ndarray) -> float:
        model.unpack(theta_value)
        _, _, probs = model.forward(xb)
        return cross_entropy(probs, yb)

    for epoch in range(args.zo_epochs):
        for xb, yb in minibatches(x_train, y_train, args.zo_batch_size, rng):
            direction = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=theta.shape)
            loss_pos = batch_loss(theta + args.zo_eps * direction, xb, yb)
            loss_neg = batch_loss(theta - args.zo_eps * direction, xb, yb)
            grad_estimate = ((loss_pos - loss_neg) / (2.0 * args.zo_eps)) * direction
            theta = theta - args.zo_lr * grad_estimate.astype(np.float32)

        model.unpack(theta)
        train_loss, train_acc = model.loss_acc(x_train, y_train)
        test_acc = accuracy(model.predict(x_test), y_test)
        history.append(
            {
                "method": "zo_spsa",
                "seed": seed,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "feedback_alignment": None,
            }
        )

    elapsed = time.perf_counter() - start_time
    final = history[-1]
    result = MethodResult(
        method="zo_spsa",
        seed=seed,
        train_size=x_train.shape[0],
        test_size=x_test.shape[0],
        input_dim=x_train.shape[1],
        hidden_dim=model.w1.shape[0],
        epochs=args.zo_epochs,
        test_acc=float(final["test_acc"]),
        train_time_sec=elapsed,
        final_train_loss=float(final["train_loss"]),
        final_train_acc=float(final["train_acc"]),
        mean_feedback_alignment=None,
    )
    return result, history


def train_stdp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[MethodResult, list[dict], SimpleSTDPClassifier]:
    cfg = STDPConfig(
        input_dim=x_train.shape[1],
        neurons=args.stdp_neurons,
        steps=args.stdp_steps,
        seed=seed,
    )
    model = SimpleSTDPClassifier(cfg)
    start_time = time.perf_counter()
    model.fit(x_train, epochs=args.stdp_epochs)
    model.assign_labels(x_train, y_train)
    elapsed = time.perf_counter() - start_time
    preds_train = model.predict(x_train)
    preds_test = model.predict(x_test)
    train_acc = accuracy(preds_train, y_train)
    test_acc = accuracy(preds_test, y_test)
    history = [
        {
            "method": "stdp",
            "seed": seed,
            "epoch": args.stdp_epochs,
            "train_loss": None,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "feedback_alignment": None,
        }
    ]
    result = MethodResult(
        method="stdp",
        seed=seed,
        train_size=x_train.shape[0],
        test_size=x_test.shape[0],
        input_dim=x_train.shape[1],
        hidden_dim=args.stdp_neurons,
        epochs=args.stdp_epochs,
        test_acc=test_acc,
        train_time_sec=elapsed,
        final_train_loss=None,
        final_train_acc=train_acc,
        mean_feedback_alignment=None,
    )
    return result, history, model


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def method_summary(results: list[MethodResult]) -> list[dict]:
    rows = []
    for method in sorted({r.method for r in results}):
        values = np.array([r.test_acc for r in results if r.method == method], dtype=np.float64)
        times = np.array([r.train_time_sec for r in results if r.method == method], dtype=np.float64)
        rows.append(
            {
                "method": method,
                "n": int(values.size),
                "mean_test_acc": float(values.mean()),
                "std_test_acc": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "mean_train_time_sec": float(times.mean()),
            }
        )
    return rows


def plot_summary(results: list[MethodResult], histories: list[dict], out_dir: str) -> None:
    summary = method_summary(results)
    methods = [row["method"] for row in summary]
    means = np.array([row["mean_test_acc"] for row in summary])
    stds = np.array([row["std_test_acc"] for row in summary])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#7a8b5f", "#d05f3f", "#5877a6", "#b79a43", "#6f6f6f", "#9b6b6b"]
    axes[0].bar(methods, means, yerr=stds, color=colors[: len(methods)])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("MNIST no-BP pilot")
    axes[0].tick_params(axis="x", rotation=20)

    for method in ["bp", "dfa_3factor", "dfa_resampled", "output_only", "zo_spsa"]:
        rows = [row for row in histories if row["method"] == method]
        by_epoch: dict[int, list[float]] = {}
        for row in rows:
            by_epoch.setdefault(int(row["epoch"]), []).append(float(row["test_acc"]))
        if not by_epoch:
            continue
        epochs = sorted(by_epoch)
        values = [np.mean(by_epoch[e]) for e in epochs]
        axes[1].plot(epochs, values, marker="o", label=method)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].set_title("learning curves")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=170)
    plt.close(fig)


def plot_filters(model: TinyMLP, input_dim: int, out_path: str, title: str) -> None:
    side = int(math.sqrt(input_dim))
    if side * side != input_dim:
        return
    n = min(16, model.w1.shape[0])
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for ax, filt in zip(axes.ravel(), model.w1[:n]):
        image = filt.reshape(side, side)
        ax.imshow(image, cmap="coolwarm")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    all_results: list[MethodResult] = []
    all_histories: list[dict] = []
    saved_filter = {method: False for method in methods}

    for seed in seeds:
        x_train, y_train, x_test, y_test = prepare_data(
            data_dir=args.data_dir,
            train_size=args.train_size,
            test_size=args.test_size,
            downsample=args.downsample,
            seed=seed,
        )
        for method in methods:
            print(f"\n=== method={method} seed={seed} ===")
            if method == "bp":
                model = TinyMLP(x_train.shape[1], args.hidden_dim, seed)
                result, history = train_bp(model, x_train, y_train, x_test, y_test, args, seed)
                if not saved_filter[method]:
                    plot_filters(model, x_train.shape[1], os.path.join(args.out_dir, "filters_bp.png"), "BP hidden filters")
                    saved_filter[method] = True
            elif method == "dfa_3factor":
                model = TinyMLP(x_train.shape[1], args.hidden_dim, seed)
                result, history = train_dfa_3factor(model, x_train, y_train, x_test, y_test, args, seed)
                if not saved_filter[method]:
                    plot_filters(
                        model,
                        x_train.shape[1],
                        os.path.join(args.out_dir, "filters_dfa_3factor.png"),
                        "Three-factor DFA hidden filters",
                    )
                    saved_filter[method] = True
            elif method == "dfa_resampled":
                model = TinyMLP(x_train.shape[1], args.hidden_dim, seed)
                result, history = train_dfa_resampled(model, x_train, y_train, x_test, y_test, args, seed)
            elif method == "output_only":
                model = TinyMLP(x_train.shape[1], args.hidden_dim, seed)
                result, history = train_output_only(model, x_train, y_train, x_test, y_test, args, seed)
            elif method == "zo_spsa":
                model = TinyMLP(x_train.shape[1], args.zo_hidden_dim, seed)
                result, history = train_zo_spsa(model, x_train, y_train, x_test, y_test, args, seed)
            elif method == "stdp":
                result, history, _ = train_stdp(x_train, y_train, x_test, y_test, args, seed)
            else:
                raise ValueError(f"Unknown method: {method}")

            all_results.append(result)
            all_histories.extend(history)
            print(
                f"{method} seed={seed}: test_acc={result.test_acc:.4f}, "
                f"train_time={result.train_time_sec:.2f}s"
            )

    result_rows = [asdict(r) for r in all_results]
    summary_rows = method_summary(all_results)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "results": result_rows,
                "summary": summary_rows,
                "history": all_histories,
            },
            f,
            indent=2,
        )
    write_csv(os.path.join(args.out_dir, "results.csv"), result_rows)
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary_rows)
    write_csv(os.path.join(args.out_dir, "history.csv"), all_histories)
    plot_summary(all_results, all_histories, args.out_dir)

    print("\n=== summary ===")
    for row in summary_rows:
        print(
            f"{row['method']:>12s}: "
            f"acc={row['mean_test_acc']:.4f} +/- {row['std_test_acc']:.4f}, "
            f"time={row['mean_train_time_sec']:.2f}s"
        )
    print(f"Saved results to {args.out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/mnist_numpy")
    parser.add_argument("--out-dir", default="research_no_bp/results/pilot")
    parser.add_argument("--methods", default="bp,dfa_3factor,output_only,zo_spsa,stdp")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--dfa-lr", type=float, default=0.08)
    parser.add_argument("--zo-hidden-dim", type=int, default=32)
    parser.add_argument("--zo-epochs", type=int, default=5)
    parser.add_argument("--zo-batch-size", type=int, default=128)
    parser.add_argument("--zo-lr", type=float, default=0.002)
    parser.add_argument("--zo-eps", type=float, default=0.05)
    parser.add_argument("--stdp-neurons", type=int, default=96)
    parser.add_argument("--stdp-epochs", type=int, default=1)
    parser.add_argument("--stdp-steps", type=int, default=10)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
