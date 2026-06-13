#!/usr/bin/env python3
"""
TinyStories character-level next-token experiment with NumPy LSTM, STDP and BTSP.

The goal is a small NLP sanity check:
  - LSTM learns next-character prediction by BPTT.
  - STDP/BTSP learn local recurrent character associations from eligibility traces.
  - All methods are pure NumPy; no PyTorch dependency.

Default settings intentionally use a small prefix of TinyStories so the experiment
can run quickly before scaling up.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = SCRIPT_DIR / "data" / "TinyStories-train.txt"
DEFAULT_VALID = SCRIPT_DIR / "data" / "TinyStories-valid.txt"


@dataclass
class STDPConfig:
    a_plus: float = 0.010
    a_minus: float = 0.004
    trace_decay: float = 0.72
    row_decay: float = 0.9995
    epochs: int = 1


@dataclass
class BTSPConfig:
    potentiation: float = 0.010
    heterosynaptic_depression: float = 0.001
    trace_decay: float = 0.88
    row_decay: float = 0.9995
    max_weight: float = 8.0
    epochs: int = 1


class CharVocab:
    def __init__(self, text: str) -> None:
        self.itos = sorted(set(text))
        self.stoi = {ch: idx for idx, ch in enumerate(self.itos)}

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[ch] for ch in text if ch in self.stoi], dtype=np.int64)

    def decode(self, ids: np.ndarray | list[int]) -> str:
        return "".join(self.itos[int(idx)] for idx in ids)

    def __len__(self) -> int:
        return len(self.itos)


class CharLSTM:
    def __init__(self, vocab_size: int, hidden_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        scale_x = 1.0 / math.sqrt(vocab_size)
        scale_h = 1.0 / math.sqrt(hidden_dim)
        self.wx = rng.normal(0.0, scale_x, (4 * hidden_dim, vocab_size)).astype(np.float32)
        self.wh = rng.normal(0.0, scale_h, (4 * hidden_dim, hidden_dim)).astype(np.float32)
        self.b = np.zeros(4 * hidden_dim, dtype=np.float32)
        self.b[hidden_dim : 2 * hidden_dim] = 1.0
        self.w_out = rng.normal(0.0, scale_h, (vocab_size, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)

    def params(self) -> dict[str, np.ndarray]:
        return {
            "wx": self.wx,
            "wh": self.wh,
            "b": self.b,
            "w_out": self.w_out,
            "b_out": self.b_out,
        }

    def zero_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(self.hidden_dim, dtype=np.float32),
            np.zeros(self.hidden_dim, dtype=np.float32),
        )

    def step(
        self, x_id: int, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        hdim = self.hidden_dim
        z = self.wx[:, x_id] + self.wh @ h_prev + self.b
        i = sigmoid(z[:hdim])
        f = sigmoid(z[hdim : 2 * hdim])
        o = sigmoid(z[2 * hdim : 3 * hdim])
        g = np.tanh(z[3 * hdim :])
        c = f * c_prev + i * g
        tanh_c = np.tanh(c)
        h = o * tanh_c
        logits = self.w_out @ h + self.b_out
        cache = {
            "x_id": x_id,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "i": i,
            "f": f,
            "o": o,
            "g": g,
            "c": c,
            "tanh_c": tanh_c,
            "h": h,
        }
        return logits, h.astype(np.float32), c.astype(np.float32), cache


def read_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))).astype(np.float32)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(temperature, 1e-6)
    z = z - float(np.max(z))
    exp_z = np.exp(z)
    return (exp_z / np.sum(exp_z)).astype(np.float32)


def cross_entropy_from_scores(scores: np.ndarray, target: int, temperature: float = 1.0) -> tuple[float, int]:
    probs = softmax(scores, temperature)
    return -math.log(float(probs[target]) + 1e-9), int(np.argmax(probs))


def clip_grads(grads: dict[str, np.ndarray], max_norm: float) -> None:
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if norm == 0.0 or norm <= max_norm:
        return
    scale = max_norm / (norm + 1e-8)
    for grad in grads.values():
        grad *= scale


def adam_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: dict[str, dict[str, np.ndarray] | int],
    lr: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    state["t"] = int(state.get("t", 0)) + 1
    t = int(state["t"])
    m = state.setdefault("m", {})
    v = state.setdefault("v", {})

    for name, param in params.items():
        if name not in m:
            m[name] = np.zeros_like(param)
            v[name] = np.zeros_like(param)
        m[name] = beta1 * m[name] + (1.0 - beta1) * grads[name]
        v[name] = beta2 * v[name] + (1.0 - beta2) * (grads[name] * grads[name])
        m_hat = m[name] / (1.0 - beta1**t)
        v_hat = v[name] / (1.0 - beta2**t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_lstm(
    train_ids: np.ndarray,
    valid_ids: np.ndarray,
    vocab: CharVocab,
    args: argparse.Namespace,
) -> tuple[CharLSTM, list[dict], dict]:
    rng = np.random.default_rng(args.seed + 11)
    model = CharLSTM(len(vocab), args.hidden_dim, args.seed)
    opt_state: dict[str, dict[str, np.ndarray] | int] = {}
    history: list[dict] = []

    max_start = len(train_ids) - args.seq_len - 1
    for update in range(1, args.lstm_updates + 1):
        start = int(rng.integers(0, max_start))
        chunk = train_ids[start : start + args.seq_len + 1]
        h, c = model.zero_state()
        caches: list[dict] = []
        probs: list[np.ndarray] = []
        train_loss = 0.0

        for t in range(args.seq_len):
            logits, h, c, cache = model.step(int(chunk[t]), h, c)
            p = softmax(logits)
            target = int(chunk[t + 1])
            train_loss += -math.log(float(p[target]) + 1e-9)
            caches.append(cache)
            probs.append(p)

        grads = {name: np.zeros_like(param) for name, param in model.params().items()}
        dh_next = np.zeros(model.hidden_dim, dtype=np.float32)
        dc_next = np.zeros(model.hidden_dim, dtype=np.float32)

        for t in reversed(range(args.seq_len)):
            target = int(chunk[t + 1])
            cache = caches[t]
            dy = probs[t].copy()
            dy[target] -= 1.0

            grads["w_out"] += np.outer(dy, cache["h"])
            grads["b_out"] += dy

            dh = model.w_out.T @ dy + dh_next
            dc = dh * cache["o"] * (1.0 - cache["tanh_c"] ** 2) + dc_next

            di = dc * cache["g"]
            df = dc * cache["c_prev"]
            do = dh * cache["tanh_c"]
            dg = dc * cache["i"]

            dz = np.concatenate(
                [
                    di * cache["i"] * (1.0 - cache["i"]),
                    df * cache["f"] * (1.0 - cache["f"]),
                    do * cache["o"] * (1.0 - cache["o"]),
                    dg * (1.0 - cache["g"] ** 2),
                ]
            ).astype(np.float32)

            grads["wx"][:, int(cache["x_id"])] += dz
            grads["wh"] += np.outer(dz, cache["h_prev"])
            grads["b"] += dz

            dh_next = (model.wh.T @ dz).astype(np.float32)
            dc_next = (dc * cache["f"]).astype(np.float32)

        scale = 1.0 / args.seq_len
        for grad in grads.values():
            grad *= scale
        clip_grads(grads, args.grad_clip)
        adam_update(model.params(), grads, opt_state, args.lr)

        if update == 1 or update % args.eval_every == 0 or update == args.lstm_updates:
            valid_metrics = evaluate_lstm(
                model,
                valid_ids,
                args.seq_len,
                args.eval_batches,
                args.seed + update,
            )
            row = {
                "update": update,
                "train_loss": train_loss / args.seq_len,
                "valid_loss": valid_metrics["loss"],
                "valid_ppl": valid_metrics["ppl"],
                "valid_acc": valid_metrics["accuracy"],
            }
            history.append(row)
            print(
                "LSTM"
                f" update={update:4d}"
                f" train_ce={row['train_loss']:.3f}"
                f" valid_ce={row['valid_loss']:.3f}"
                f" ppl={row['valid_ppl']:.2f}"
                f" acc={row['valid_acc']:.3f}"
            )

    final_metrics = evaluate_lstm(
        model,
        valid_ids,
        args.seq_len,
        max(args.eval_batches * 2, 16),
        args.seed + 1000,
    )
    return model, history, final_metrics


def evaluate_lstm(
    model: CharLSTM,
    ids: np.ndarray,
    seq_len: int,
    batches: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    max_start = len(ids) - seq_len - 1
    losses: list[float] = []
    correct = 0
    total = 0

    for _ in range(batches):
        start = int(rng.integers(0, max_start))
        chunk = ids[start : start + seq_len + 1]
        h, c = model.zero_state()
        for t in range(seq_len):
            logits, h, c, _ = model.step(int(chunk[t]), h, c)
            target = int(chunk[t + 1])
            loss, pred = cross_entropy_from_scores(logits, target)
            losses.append(loss)
            correct += int(pred == target)
            total += 1

    loss = float(np.mean(losses))
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / max(total, 1),
    }


def train_stdp_matrix(ids: np.ndarray, vocab_size: int, cfg: STDPConfig) -> np.ndarray:
    weights = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    trace = np.zeros(vocab_size, dtype=np.float32)

    for _ in range(cfg.epochs):
        trace.fill(0.0)
        for token in ids:
            token = int(token)
            weights[token] *= cfg.row_decay
            weights[token] += cfg.a_plus * trace
            weights[:, token] -= cfg.a_minus * trace
            trace *= cfg.trace_decay
            trace[token] += 1.0

    return weights


def train_btsp_matrix(ids: np.ndarray, vocab_size: int, cfg: BTSPConfig) -> np.ndarray:
    weights = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    trace = np.zeros(vocab_size, dtype=np.float32)

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

    return weights


def evaluate_plastic_matrix(
    weights: np.ndarray,
    ids: np.ndarray,
    trace_decay: float,
    temperature: float,
) -> dict:
    vocab_size = weights.shape[0]
    trace = np.zeros(vocab_size, dtype=np.float32)
    losses: list[float] = []
    correct = 0
    total = 0

    for idx in range(len(ids) - 1):
        current = int(ids[idx])
        target = int(ids[idx + 1])
        trace *= trace_decay
        trace[current] += 1.0
        scores = weights @ trace
        loss, pred = cross_entropy_from_scores(scores, target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1

    loss = float(np.mean(losses))
    return {
        "loss": loss,
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / max(total, 1),
    }


def sample_lstm(
    model: CharLSTM,
    vocab: CharVocab,
    prompt: str,
    length: int,
    temperature: float,
    seed: int,
) -> str:
    rng = np.random.default_rng(seed)
    h, c = model.zero_state()
    output = list(prompt)
    prompt_ids = vocab.encode(prompt)

    if len(prompt_ids) == 0:
        prompt_ids = np.array([0], dtype=np.int64)

    for token in prompt_ids[:-1]:
        _, h, c, _ = model.step(int(token), h, c)

    current = int(prompt_ids[-1])
    for _ in range(length):
        logits, h, c, _ = model.step(current, h, c)
        probs = softmax(logits, temperature)
        current = int(rng.choice(len(vocab), p=probs))
        output.append(vocab.itos[current])

    return "".join(output)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    vocab: CharVocab,
    lstm_model: CharLSTM,
    stdp_weights: np.ndarray,
    btsp_weights: np.ndarray,
    args: argparse.Namespace,
) -> None:
    np.savez_compressed(
        path,
        itos=np.array(vocab.itos),
        lstm_wx=lstm_model.wx,
        lstm_wh=lstm_model.wh,
        lstm_b=lstm_model.b,
        lstm_w_out=lstm_model.w_out,
        lstm_b_out=lstm_model.b_out,
        stdp_weights=stdp_weights,
        btsp_weights=btsp_weights,
        hidden_dim=np.array([lstm_model.hidden_dim], dtype=np.int64),
        plastic_temperature=np.array([args.plastic_temperature], dtype=np.float32),
        stdp_trace_decay=np.array([STDPConfig().trace_decay], dtype=np.float32),
        btsp_trace_decay=np.array([BTSPConfig().trace_decay], dtype=np.float32),
    )


def write_config(path: Path, args: argparse.Namespace) -> None:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["stdp"] = asdict(STDPConfig())
    config["btsp"] = asdict(BTSPConfig())
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def plot_lstm_history(history: list[dict], path: Path) -> None:
    if not history:
        return
    updates = [int(row["update"]) for row in history]
    valid_loss = [float(row["valid_loss"]) for row in history]
    valid_acc = [float(row["valid_acc"]) for row in history]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(updates, valid_loss, marker="o", color="firebrick", label="valid CE")
    ax1.set_xlabel("LSTM update")
    ax1.set_ylabel("cross entropy")
    ax2 = ax1.twinx()
    ax2.plot(updates, valid_acc, marker="s", color="royalblue", label="valid accuracy")
    ax2.set_ylabel("accuracy")
    ax1.set_title("TinyStories char-level LSTM")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "tinystories_nlp")
    parser.add_argument("--train-chars", type=int, default=250_000)
    parser.add_argument("--valid-chars", type=int, default=50_000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=80)
    parser.add_argument("--lstm-updates", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--sample-len", type=int, default=500)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--plastic-temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_text = read_prefix(args.train_file, args.train_chars)
    valid_text = read_prefix(args.valid_file, args.valid_chars)
    vocab = CharVocab(train_text + valid_text)
    train_ids = vocab.encode(train_text)
    valid_ids = vocab.encode(valid_text)

    if len(train_ids) <= args.seq_len + 1 or len(valid_ids) <= args.seq_len + 1:
        raise ValueError("Not enough data for the requested sequence length.")

    print(f"train file: {args.train_file}")
    print(f"valid file: {args.valid_file}")
    print(f"train chars: {len(train_ids):,}, valid chars: {len(valid_ids):,}")
    print(f"vocab size: {len(vocab)}")

    stdp_weights = train_stdp_matrix(train_ids, len(vocab), STDPConfig())
    btsp_weights = train_btsp_matrix(train_ids, len(vocab), BTSPConfig())
    stdp_metrics = evaluate_plastic_matrix(
        stdp_weights,
        valid_ids,
        STDPConfig().trace_decay,
        args.plastic_temperature,
    )
    btsp_metrics = evaluate_plastic_matrix(
        btsp_weights,
        valid_ids,
        BTSPConfig().trace_decay,
        args.plastic_temperature,
    )

    print(
        "STDP"
        f" valid_ce={stdp_metrics['loss']:.3f}"
        f" ppl={stdp_metrics['ppl']:.2f}"
        f" acc={stdp_metrics['accuracy']:.3f}"
    )
    print(
        "BTSP"
        f" valid_ce={btsp_metrics['loss']:.3f}"
        f" ppl={btsp_metrics['ppl']:.2f}"
        f" acc={btsp_metrics['accuracy']:.3f}"
    )

    lstm_model, lstm_history, lstm_metrics = train_lstm(train_ids, valid_ids, vocab, args)
    print(
        "LSTM final"
        f" valid_ce={lstm_metrics['loss']:.3f}"
        f" ppl={lstm_metrics['ppl']:.2f}"
        f" acc={lstm_metrics['accuracy']:.3f}"
    )

    metrics_rows = [
        {"method": "stdp_trace", **stdp_metrics},
        {"method": "btsp_trace", **btsp_metrics},
        {"method": "numpy_lstm", **lstm_metrics},
    ]
    write_csv(args.out_dir / "metrics.csv", metrics_rows)
    write_csv(args.out_dir / "lstm_history.csv", lstm_history)
    save_checkpoint(
        args.out_dir / "checkpoint.npz",
        vocab,
        lstm_model,
        stdp_weights,
        btsp_weights,
        args,
    )
    write_config(args.out_dir / "config.json", args)
    plot_lstm_history(lstm_history, args.out_dir / "lstm_training_curve.png")

    prompt = "Once upon a time"
    sample = sample_lstm(
        lstm_model,
        vocab,
        prompt,
        args.sample_len,
        args.sample_temperature,
        args.seed + 123,
    )
    sample_path = args.out_dir / "lstm_sample.txt"
    with sample_path.open("w", encoding="utf-8") as f:
        f.write(sample)

    print(f"wrote metrics: {args.out_dir / 'metrics.csv'}")
    print(f"wrote LSTM history: {args.out_dir / 'lstm_history.csv'}")
    print(f"wrote checkpoint: {args.out_dir / 'checkpoint.npz'}")
    print(f"wrote config: {args.out_dir / 'config.json'}")
    print(f"wrote curve: {args.out_dir / 'lstm_training_curve.png'}")
    print(f"wrote sample: {sample_path}")
    print("\nSample preview:")
    print(sample[:700])


if __name__ == "__main__":
    main()
