#!/usr/bin/env python3
"""
TinyStories token-level next-token experiment with the local Llama tokenizer.

This is closer to LLM training than the character-level sanity check:
  - text is encoded with /private/zhenningshi/model_weights/Llama-3.2-1B-Instruct
  - the experiment uses a compact vocab of frequent tokenizer IDs to keep NumPy
    LSTM and plasticity matrices small
  - target is next tokenizer token, not next character
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = SCRIPT_DIR / "data" / "TinyStories-train.txt"
DEFAULT_VALID = SCRIPT_DIR / "data" / "TinyStories-valid.txt"
DEFAULT_TOKENIZER = Path("/private/zhenningshi/model_weights/Llama-3.2-1B-Instruct")


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


class TokenLSTM:
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.emb = rng.normal(0.0, 1.0 / math.sqrt(embed_dim), (vocab_size, embed_dim)).astype(np.float32)
        scale = 1.0 / math.sqrt(embed_dim + hidden_dim)
        self.wx = rng.normal(0.0, scale, (4 * hidden_dim, embed_dim)).astype(np.float32)
        self.wh = rng.normal(0.0, scale, (4 * hidden_dim, hidden_dim)).astype(np.float32)
        self.b = np.zeros(4 * hidden_dim, dtype=np.float32)
        self.b[hidden_dim : 2 * hidden_dim] = 1.0
        self.w_out = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), (vocab_size, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)

    def params(self) -> dict[str, np.ndarray]:
        return {
            "emb": self.emb,
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
        self, token: int, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        hdim = self.hidden_dim
        x = self.emb[token]
        z = self.wx @ x + self.wh @ h_prev + self.b
        i = sigmoid(z[:hdim])
        f = sigmoid(z[hdim : 2 * hdim])
        o = sigmoid(z[2 * hdim : 3 * hdim])
        g = np.tanh(z[3 * hdim :])
        c = f * c_prev + i * g
        tanh_c = np.tanh(c)
        h = o * tanh_c
        logits = self.w_out @ h + self.b_out
        cache = {
            "token": token,
            "x": x,
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


def evaluate_plastic_matrix(weights: np.ndarray, ids: np.ndarray, trace_decay: float, temperature: float) -> dict:
    trace = np.zeros(weights.shape[0], dtype=np.float32)
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


def evaluate_lstm(model: TokenLSTM, ids: np.ndarray, seq_len: int, batches: int, seed: int) -> dict:
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


def train_lstm(train_ids: np.ndarray, valid_ids: np.ndarray, args: argparse.Namespace) -> tuple[TokenLSTM, list[dict], dict]:
    rng = np.random.default_rng(args.seed + 17)
    model = TokenLSTM(args.max_vocab, args.embed_dim, args.hidden_dim, args.seed)
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
            probs.append(p)
            caches.append(cache)

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

            grads["wx"] += np.outer(dz, cache["x"])
            grads["wh"] += np.outer(dz, cache["h_prev"])
            grads["b"] += dz
            grads["emb"][int(cache["token"])] += model.wx.T @ dz
            dh_next = (model.wh.T @ dz).astype(np.float32)
            dc_next = (dc * cache["f"]).astype(np.float32)

        scale = 1.0 / args.seq_len
        for grad in grads.values():
            grad *= scale
        clip_grads(grads, args.grad_clip)
        adam_update(model.params(), grads, opt_state, args.lr)

        if update == 1 or update % args.eval_every == 0 or update == args.lstm_updates:
            valid_metrics = evaluate_lstm(model, valid_ids, args.seq_len, args.eval_batches, args.seed + update)
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

    final_metrics = evaluate_lstm(model, valid_ids, args.seq_len, max(args.eval_batches * 2, 16), args.seed + 1000)
    return model, history, final_metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_lstm_history(history: list[dict], path: Path) -> None:
    updates = [int(row["update"]) for row in history]
    valid_loss = [float(row["valid_loss"]) for row in history]
    valid_acc = [float(row["valid_acc"]) for row in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(updates, valid_loss, marker="o", color="firebrick")
    ax1.set_xlabel("LSTM update")
    ax1.set_ylabel("token CE")
    ax2 = ax1.twinx()
    ax2.plot(updates, valid_acc, marker="s", color="royalblue")
    ax2.set_ylabel("top-1 token accuracy")
    ax1.set_title("TinyStories Llama-token LSTM")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sample_lstm(model: TokenLSTM, tokenizer, kept_raw: np.ndarray, prompt: str, length: int, temperature: float, seed: int) -> str:
    raw_to_compact = {int(raw): idx for idx, raw in enumerate(kept_raw)}
    compact_to_raw = {idx: int(raw) for idx, raw in enumerate(kept_raw)}
    raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = [raw_to_compact[token] for token in raw_ids if token in raw_to_compact]
    if not ids:
        ids = [0]

    rng = np.random.default_rng(seed)
    h, c = model.zero_state()
    generated = list(ids)
    for token in ids[:-1]:
        _, h, c, _ = model.step(int(token), h, c)
    current = int(ids[-1])
    for _ in range(length):
        logits, h, c, _ = model.step(current, h, c)
        probs = softmax(logits, temperature)
        current = int(rng.choice(model.vocab_size, p=probs))
        generated.append(current)

    raw_generated = [compact_to_raw[idx] for idx in generated]
    return tokenizer.decode(raw_generated, skip_special_tokens=True)


def save_checkpoint(path: Path, model: TokenLSTM, kept_raw: np.ndarray, stdp: np.ndarray, btsp: np.ndarray, args: argparse.Namespace) -> None:
    np.savez_compressed(
        path,
        kept_raw_ids=kept_raw,
        emb=model.emb,
        wx=model.wx,
        wh=model.wh,
        b=model.b,
        w_out=model.w_out,
        b_out=model.b_out,
        stdp_weights=stdp,
        btsp_weights=btsp,
        stdp_trace_decay=np.array([STDPConfig().trace_decay], dtype=np.float32),
        btsp_trace_decay=np.array([BTSPConfig().trace_decay], dtype=np.float32),
        plastic_temperature=np.array([args.plastic_temperature], dtype=np.float32),
    )


def write_config(path: Path, args: argparse.Namespace) -> None:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["stdp"] = asdict(STDPConfig())
    config["btsp"] = asdict(BTSPConfig())
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
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lstm-updates", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--sample-len", type=int, default=120)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--plastic-temperature", type=float, default=0.8)
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

    stdp_weights = train_stdp_matrix(train_ids, args.max_vocab, STDPConfig())
    btsp_weights = train_btsp_matrix(train_ids, args.max_vocab, BTSPConfig())
    stdp_metrics = evaluate_plastic_matrix(stdp_weights, valid_ids, STDPConfig().trace_decay, args.plastic_temperature)
    btsp_metrics = evaluate_plastic_matrix(btsp_weights, valid_ids, BTSPConfig().trace_decay, args.plastic_temperature)
    print(f"STDP valid_ce={stdp_metrics['loss']:.3f} ppl={stdp_metrics['ppl']:.2f} acc={stdp_metrics['accuracy']:.3f}")
    print(f"BTSP valid_ce={btsp_metrics['loss']:.3f} ppl={btsp_metrics['ppl']:.2f} acc={btsp_metrics['accuracy']:.3f}")

    model, history, lstm_metrics = train_lstm(train_ids, valid_ids, args)
    print(f"LSTM final valid_ce={lstm_metrics['loss']:.3f} ppl={lstm_metrics['ppl']:.2f} acc={lstm_metrics['accuracy']:.3f}")

    rows = [
        {"method": "stdp_trace", **stdp_metrics},
        {"method": "btsp_trace", **btsp_metrics},
        {"method": "numpy_lstm", **lstm_metrics},
    ]
    write_csv(args.out_dir / "metrics.csv", rows)
    write_csv(args.out_dir / "lstm_history.csv", history)
    plot_lstm_history(history, args.out_dir / "lstm_training_curve.png")
    save_checkpoint(args.out_dir / "checkpoint.npz", model, kept_raw, stdp_weights, btsp_weights, args)
    write_config(args.out_dir / "config.json", args)

    sample = sample_lstm(model, tokenizer, kept_raw, "Once upon a time", args.sample_len, args.sample_temperature, args.seed + 77)
    sample_path = args.out_dir / "lstm_sample.txt"
    sample_path.write_text(sample, encoding="utf-8")
    print(f"wrote metrics: {args.out_dir / 'metrics.csv'}")
    print(f"wrote checkpoint: {args.out_dir / 'checkpoint.npz'}")
    print(f"wrote curve: {args.out_dir / 'lstm_training_curve.png'}")
    print(f"wrote sample: {sample_path}")
    print("\nSample preview:")
    print(sample[:800])


if __name__ == "__main__":
    main()
