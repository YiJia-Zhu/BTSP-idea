#!/usr/bin/env python3
"""
Temporal next-token experiment for no-BP learning rules.

Task: delayed cue-to-target prediction.

Episode:
  C_k, F, F, ..., F, T_k, SEP

At the final filler step, the next token is T_k. Since all filler inputs are
identical, a model must preserve the cue in hidden state to predict the target.

Compared methods:
  - bptt_rnn: recurrent neural net trained with full BPTT over each episode.
  - eprop_3factor: online recurrent three-factor rule with eligibility traces.
  - eprop_resampled: same, but feedback is resampled every step.
  - reservoir: fixed recurrent hidden state, train output readout only.
  - bigram: count-based next-token baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np


TOKENS = ["C0", "C1", "C2", "C3", "F", "T0", "T1", "T2", "T3", "SEP"]
NUM_CUES = 4
FILLER = 4
TARGET_OFFSET = 5
SEP = 9
VOCAB_SIZE = len(TOKENS)


@dataclass
class RunResult:
    method: str
    seed: int
    delay: int
    train_episodes: int
    test_episodes: int
    hidden_dim: int
    epochs: int
    overall_acc: float
    target_acc: float
    mean_loss: float | None
    train_time_sec: float


class TinyRNN:
    def __init__(self, hidden_dim: int, seed: int, spectral_radius: float) -> None:
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.w_in = rng.normal(0.0, 1.0 / math.sqrt(VOCAB_SIZE), (hidden_dim, VOCAB_SIZE)).astype(np.float32)
        w_rec = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), (hidden_dim, hidden_dim)).astype(np.float32)
        eigvals = np.linalg.eigvals(w_rec.astype(np.float64))
        radius = max(abs(eigvals)) + 1e-8
        self.w_rec = (w_rec * (spectral_radius / radius)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)
        self.w_out = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), (VOCAB_SIZE, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(VOCAB_SIZE, dtype=np.float32)

    def step(self, token: int, h_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = one_hot_token(token)
        z = self.w_in @ x + self.w_rec @ h_prev + self.b_h
        h = np.tanh(z)
        probs = softmax(self.w_out @ h + self.b_out)
        return z, h, probs


def one_hot_token(token: int) -> np.ndarray:
    x = np.zeros(VOCAB_SIZE, dtype=np.float32)
    x[token] = 1.0
    return x


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - float(np.max(logits))
    exp_z = np.exp(z)
    return (exp_z / np.sum(exp_z)).astype(np.float32)


def cross_entropy(probs: np.ndarray, target: int) -> float:
    return float(-np.log(float(probs[target]) + 1e-8))


def generate_episodes(n: int, delay: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    episodes = np.zeros((n, delay + 3), dtype=np.int64)
    for i in range(n):
        cue = int(rng.integers(NUM_CUES))
        episodes[i, 0] = cue
        episodes[i, 1 : delay + 1] = FILLER
        episodes[i, delay + 1] = TARGET_OFFSET + cue
        episodes[i, delay + 2] = SEP
    return episodes


def target_position(delay: int) -> int:
    return delay


def evaluate_rnn(model: TinyRNN, episodes: np.ndarray, delay: int) -> tuple[float, float, float]:
    correct = 0
    total = 0
    target_correct = 0
    target_total = 0
    losses = []
    target_t = target_position(delay)
    for episode in episodes:
        h = np.zeros(model.hidden_dim, dtype=np.float32)
        for t in range(len(episode) - 1):
            _, h, probs = model.step(int(episode[t]), h)
            y = int(episode[t + 1])
            pred = int(np.argmax(probs))
            correct += int(pred == y)
            total += 1
            losses.append(cross_entropy(probs, y))
            if t == target_t:
                target_correct += int(pred == y)
                target_total += 1
    return correct / total, target_correct / target_total, float(np.mean(losses))


def train_bptt(
    episodes: np.ndarray,
    test_episodes: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[RunResult, list[dict]]:
    rng = np.random.default_rng(seed + 11)
    model = TinyRNN(args.hidden_dim, seed, args.spectral_radius)
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        for idx in rng.permutation(len(episodes)):
            episode = episodes[idx]
            hs = [np.zeros(model.hidden_dim, dtype=np.float32)]
            zs = []
            probs_list = []
            for t in range(len(episode) - 1):
                z, h, probs = model.step(int(episode[t]), hs[-1])
                zs.append(z)
                hs.append(h)
                probs_list.append(probs)

            grad_w_in = np.zeros_like(model.w_in)
            grad_w_rec = np.zeros_like(model.w_rec)
            grad_b_h = np.zeros_like(model.b_h)
            grad_w_out = np.zeros_like(model.w_out)
            grad_b_out = np.zeros_like(model.b_out)
            dh_next = np.zeros(model.hidden_dim, dtype=np.float32)

            for t in reversed(range(len(episode) - 1)):
                y = int(episode[t + 1])
                err = probs_list[t].copy()
                err[y] -= 1.0
                grad_w_out += np.outer(err, hs[t + 1])
                grad_b_out += err

                dh = model.w_out.T @ err + dh_next
                dz = dh * (1.0 - np.tanh(zs[t]) ** 2)
                grad_w_in += np.outer(dz, one_hot_token(int(episode[t])))
                grad_w_rec += np.outer(dz, hs[t])
                grad_b_h += dz
                dh_next = model.w_rec.T @ dz

            clip_grads([grad_w_in, grad_w_rec, grad_b_h, grad_w_out, grad_b_out], args.grad_clip)
            model.w_in -= args.lr * grad_w_in
            model.w_rec -= args.lr * grad_w_rec
            model.b_h -= args.lr * grad_b_h
            model.w_out -= args.lr * grad_w_out
            model.b_out -= args.lr * grad_b_out

        overall, target_acc, loss = evaluate_rnn(model, test_episodes, args.delay)
        history.append({"method": "bptt_rnn", "seed": seed, "epoch": epoch + 1, "overall_acc": overall, "target_acc": target_acc, "loss": loss})

    elapsed = time.perf_counter() - start
    overall, target_acc, loss = evaluate_rnn(model, test_episodes, args.delay)
    result = RunResult("bptt_rnn", seed, args.delay, len(episodes), len(test_episodes), args.hidden_dim, args.epochs, overall, target_acc, loss, elapsed)
    return result, history


def clip_grads(grads: list[np.ndarray], max_norm: float) -> None:
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
    if norm <= max_norm or norm <= 1e-12:
        return
    scale = max_norm / norm
    for g in grads:
        g *= scale


def train_eprop(
    episodes: np.ndarray,
    test_episodes: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    resample_feedback: bool,
) -> tuple[RunResult, list[dict]]:
    rng = np.random.default_rng(seed + (31 if not resample_feedback else 37))
    model = TinyRNN(args.hidden_dim, seed, args.spectral_radius)
    feedback = rng.normal(0.0, 1.0 / math.sqrt(VOCAB_SIZE), (args.hidden_dim, VOCAB_SIZE)).astype(np.float32)
    method = "eprop_resampled" if resample_feedback else "eprop_3factor"
    history = []
    start = time.perf_counter()

    for epoch in range(args.epochs):
        for idx in rng.permutation(len(episodes)):
            episode = episodes[idx]
            h_prev = np.zeros(args.hidden_dim, dtype=np.float32)
            e_in = np.zeros_like(model.w_in)
            e_rec = np.zeros_like(model.w_rec)
            e_b = np.zeros_like(model.b_h)
            for t in range(len(episode) - 1):
                token = int(episode[t])
                y = int(episode[t + 1])
                x = one_hot_token(token)
                z, h, probs = model.step(token, h_prev)

                err = probs.copy()
                err[y] -= 1.0

                model.w_out -= args.lr_out * np.outer(err, h)
                model.b_out -= args.lr_out * err

                if resample_feedback:
                    feedback = rng.normal(0.0, 1.0 / math.sqrt(VOCAB_SIZE), (args.hidden_dim, VOCAB_SIZE)).astype(np.float32)
                modulator = feedback @ err
                local_deriv = 1.0 - np.tanh(z) ** 2
                e_in = args.trace_decay * e_in + np.outer(local_deriv, x)
                e_rec = args.trace_decay * e_rec + np.outer(local_deriv, h_prev)
                e_b = args.trace_decay * e_b + local_deriv

                model.w_in -= args.lr_local * modulator[:, None] * e_in
                model.w_rec -= args.lr_local * modulator[:, None] * e_rec
                model.b_h -= args.lr_local * modulator * e_b
                h_prev = h

        overall, target_acc, loss = evaluate_rnn(model, test_episodes, args.delay)
        history.append({"method": method, "seed": seed, "epoch": epoch + 1, "overall_acc": overall, "target_acc": target_acc, "loss": loss})

    elapsed = time.perf_counter() - start
    overall, target_acc, loss = evaluate_rnn(model, test_episodes, args.delay)
    result = RunResult(method, seed, args.delay, len(episodes), len(test_episodes), args.hidden_dim, args.epochs, overall, target_acc, loss, elapsed)
    return result, history


def train_reservoir(
    episodes: np.ndarray,
    test_episodes: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[RunResult, list[dict]]:
    rng = np.random.default_rng(seed + 51)
    model = TinyRNN(args.hidden_dim, seed, args.spectral_radius)
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        for idx in rng.permutation(len(episodes)):
            episode = episodes[idx]
            h = np.zeros(args.hidden_dim, dtype=np.float32)
            for t in range(len(episode) - 1):
                _, h, probs = model.step(int(episode[t]), h)
                y = int(episode[t + 1])
                err = probs.copy()
                err[y] -= 1.0
                model.w_out -= args.lr_out * np.outer(err, h)
                model.b_out -= args.lr_out * err

        overall, target_acc, loss = evaluate_rnn(model, test_episodes, args.delay)
        history.append({"method": "reservoir", "seed": seed, "epoch": epoch + 1, "overall_acc": overall, "target_acc": target_acc, "loss": loss})

    elapsed = time.perf_counter() - start
    overall, target_acc, loss = evaluate_rnn(model, test_episodes, args.delay)
    result = RunResult("reservoir", seed, args.delay, len(episodes), len(test_episodes), args.hidden_dim, args.epochs, overall, target_acc, loss, elapsed)
    return result, history


def eval_bigram(train_episodes: np.ndarray, test_episodes: np.ndarray, delay: int, seed: int) -> RunResult:
    counts = np.ones((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)
    for episode in train_episodes:
        for t in range(len(episode) - 1):
            counts[int(episode[t]), int(episode[t + 1])] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    correct = 0
    total = 0
    target_correct = 0
    target_total = 0
    losses = []
    target_t = target_position(delay)
    for episode in test_episodes:
        for t in range(len(episode) - 1):
            x = int(episode[t])
            y = int(episode[t + 1])
            pred = int(np.argmax(probs[x]))
            correct += int(pred == y)
            total += 1
            losses.append(cross_entropy(probs[x].astype(np.float32), y))
            if t == target_t:
                target_correct += int(pred == y)
                target_total += 1
    return RunResult("bigram", seed, delay, len(train_episodes), len(test_episodes), 0, 0, correct / total, target_correct / target_total, float(np.mean(losses)), 0.0)


def summarize(results: list[RunResult]) -> list[dict]:
    rows = []
    for method in sorted({r.method for r in results}):
        subset = [r for r in results if r.method == method]
        overall = np.array([r.overall_acc for r in subset])
        target = np.array([r.target_acc for r in subset])
        loss = np.array([r.mean_loss for r in subset if r.mean_loss is not None])
        rows.append(
            {
                "method": method,
                "n": len(subset),
                "overall_acc_mean": float(overall.mean()),
                "overall_acc_std": float(overall.std(ddof=1)) if len(subset) > 1 else 0.0,
                "target_acc_mean": float(target.mean()),
                "target_acc_std": float(target.std(ddof=1)) if len(subset) > 1 else 0.0,
                "loss_mean": float(loss.mean()) if len(loss) else None,
                "train_time_sec_mean": float(np.mean([r.train_time_sec for r in subset])),
            }
        )
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary: list[dict], histories: list[dict], out_dir: str) -> None:
    methods = [row["method"] for row in summary]
    target_means = [row["target_acc_mean"] for row in summary]
    target_stds = [row["target_acc_std"] for row in summary]
    overall_means = [row["overall_acc_mean"] for row in summary]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(methods, target_means, yerr=target_stds, color="#496d63")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("target-position accuracy")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Delayed cue target prediction")

    axes[0].plot(methods, overall_means, marker="o", color="#b45b3e", label="overall acc")
    axes[0].legend()

    for method in ["bptt_rnn", "eprop_3factor", "eprop_resampled", "reservoir"]:
        rows = [row for row in histories if row["method"] == method]
        if not rows:
            continue
        by_epoch: dict[int, list[float]] = {}
        for row in rows:
            by_epoch.setdefault(int(row["epoch"]), []).append(float(row["target_acc"]))
        epochs = sorted(by_epoch)
        values = [float(np.mean(by_epoch[e])) for e in epochs]
        axes[1].plot(epochs, values, marker="o", label=method)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("target-position accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Learning curves")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "temporal_summary.png"), dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    all_results: list[RunResult] = []
    histories: list[dict] = []

    for seed in seeds:
        train_episodes = generate_episodes(args.train_episodes, args.delay, seed)
        test_episodes = generate_episodes(args.test_episodes, args.delay, seed + 10000)
        for method in methods:
            print(f"\n=== method={method} seed={seed} ===")
            if method == "bigram":
                result = eval_bigram(train_episodes, test_episodes, args.delay, seed)
                history = []
            elif method == "reservoir":
                result, history = train_reservoir(train_episodes, test_episodes, args, seed)
            elif method == "eprop_3factor":
                result, history = train_eprop(train_episodes, test_episodes, args, seed, resample_feedback=False)
            elif method == "eprop_resampled":
                result, history = train_eprop(train_episodes, test_episodes, args, seed, resample_feedback=True)
            elif method == "bptt_rnn":
                result, history = train_bptt(train_episodes, test_episodes, args, seed)
            else:
                raise ValueError(f"Unknown method: {method}")
            all_results.append(result)
            histories.extend(history)
            print(
                f"{method} seed={seed}: overall={result.overall_acc:.4f}, "
                f"target={result.target_acc:.4f}, loss={result.mean_loss}"
            )

    rows = [asdict(r) for r in all_results]
    summary = summarize(all_results)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": rows, "summary": summary, "history": histories, "tokens": TOKENS}, f, indent=2)
    write_csv(os.path.join(args.out_dir, "results.csv"), rows)
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    write_csv(os.path.join(args.out_dir, "history.csv"), histories)
    plot_results(summary, histories, args.out_dir)

    print("\n=== summary ===")
    for row in summary:
        print(
            f"{row['method']:>16s}: overall={row['overall_acc_mean']:.4f}, "
            f"target={row['target_acc_mean']:.4f} +/- {row['target_acc_std']:.4f}, "
            f"loss={row['loss_mean']}"
        )
    print(f"Saved to {args.out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="research_no_bp/temporal/results/delayed_v1")
    parser.add_argument("--methods", default="bigram,reservoir,eprop_3factor,eprop_resampled,bptt_rnn")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--test-episodes", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--lr-out", type=float, default=0.03)
    parser.add_argument("--lr-local", type=float, default=0.004)
    parser.add_argument("--trace-decay", type=float, default=0.85)
    parser.add_argument("--spectral-radius", type=float, default=0.90)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
