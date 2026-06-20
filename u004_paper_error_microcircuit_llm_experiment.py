#!/usr/bin/env python3
"""
U004 paper-faithful error-neuron microcircuit LLM adapter.

This prototype keeps close to the error-neuron microcircuit variables used in
Max et al.'s reference implementation:

  WPP: representation-to-representation forward weights
  BII: error-neuron top-down weights
  WIP: representation-to-error lateral weights
  BPI: error-to-representation lateral weights

The adaptation to next-token language modeling is deliberately narrow:
tokens are encoded as embeddings plus positions, representation layers are a
deep residual stack, the output layer is a token-neuron layer, and training uses
the paper-style local post-difference outer presynaptic-rate update.

No autograd, BP/BPTT, pretrained backbone, answer slots, n-gram tables, or raw
text replay are used by the model.  `model_type=BP` is diagnostic only.
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
DEFAULT_GSM_TRAIN = SCRIPT_DIR / "data" / "GSM8k-Aug" / "gsm8k_aug_train.json"
DEFAULT_GSM_VALID = SCRIPT_DIR / "data" / "GSM8k-Aug" / "gsm8k_test.json"


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


def softmax_loss_and_pred(logits: np.ndarray, target: int, temperature: float) -> tuple[float, int]:
    probs = phase.softmax(logits.astype(np.float32), temperature)
    return -math.log(float(probs[int(target)]) + 1e-9), int(np.argmax(probs))


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
class U004Config:
    context_len: int = 32
    d_model: int = 64
    blocks: int = 4
    eta_fw: float = 0.02
    eta_output: float = 0.05
    logit_scale: float = 8.0
    gl: float = 0.03
    gbas: float = 0.10
    gapi: float = 0.06
    gden: float = 0.10
    gntgt: float = 0.06
    residual_scale: float = 1.0
    ff_scale: float = 1.0
    attention_scale: float = 0.0
    dWPP_use_activation: bool = False
    varphi_transfer: bool = True
    fw_connection_mode: str = "layered"
    bw_connection_mode: str = "skip"
    model_type: str = "FA"
    temperature: float = 1.0
    seed: int = 0


class U004PaperErrorMicrocircuitLLM:
    def __init__(self, vocab_size: int, cfg: U004Config) -> None:
        self.vocab_size = int(vocab_size)
        self.cfg = cfg
        self.context_len = max(int(cfg.context_len), 1)
        self.d_model = max(int(cfg.d_model), 1)
        self.blocks = max(int(cfg.blocks), 1)
        self.rng = np.random.default_rng(cfg.seed)
        self.layer_dims = [self.d_model for _ in range(self.blocks)] + [self.vocab_size]
        self.error_dims = self.layer_dims.copy()
        self.token_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (self.vocab_size, self.d_model)).astype(np.float32)
        )
        self.position_codes = sinusoidal_positions(self.context_len, self.d_model)
        self.WPP = self.init_wpp()
        self.BII_layered: list[np.ndarray] = []
        self.BII_skip: list[list[np.ndarray | None]] = []
        if cfg.bw_connection_mode == "layered":
            self.BII_layered = self.init_bii_layered()
        elif cfg.bw_connection_mode == "skip":
            self.BII_skip = self.init_bii_skip()
        else:
            raise ValueError(f"unknown bw_connection_mode: {cfg.bw_connection_mode}")
        if cfg.model_type == "BP":
            self.set_bp_feedback()
        elif cfg.model_type != "FA":
            raise ValueError(f"unknown model_type: {cfg.model_type}")
        self.output_bias = np.full(self.vocab_size, -math.log(max(self.vocab_size, 1)), dtype=np.float32)
        self.output_counts = np.ones(self.vocab_size, dtype=np.float32)
        self.attn_q = [self.fixed_projection() for _ in range(self.blocks)]
        self.attn_k = [self.fixed_projection() for _ in range(self.blocks)]
        self.attn_v = [self.fixed_projection() for _ in range(self.blocks)]
        self.attn_o = [self.fixed_projection() for _ in range(self.blocks)]

    def init_wpp(self) -> list[np.ndarray]:
        weights: list[np.ndarray] = []
        prev_dim = self.d_model
        for dim in self.layer_dims:
            scale = 1.0 / math.sqrt(max(prev_dim, 1))
            weights.append(self.rng.normal(0.0, scale, (dim, prev_dim)).astype(np.float32))
            prev_dim = dim
        return weights

    def init_bii_layered(self) -> list[np.ndarray]:
        mats: list[np.ndarray] = []
        for lower, upper in zip(self.error_dims[:-1], self.error_dims[1:]):
            scale = 1.0 / math.sqrt(max(upper, 1))
            mats.append(self.rng.normal(0.0, scale, (lower, upper)).astype(np.float32))
        return mats

    def init_bii_skip(self) -> list[list[np.ndarray | None]]:
        blocks: list[list[np.ndarray | None]] = []
        for i, lower in enumerate(self.error_dims):
            row: list[np.ndarray | None] = []
            for j, upper in enumerate(self.error_dims):
                if j <= i:
                    row.append(None)
                    continue
                scale = 1.0 / math.sqrt(max(upper, 1))
                row.append(self.rng.normal(0.0, scale, (lower, upper)).astype(np.float32))
            blocks.append(row)
        return blocks

    def set_bp_feedback(self) -> None:
        if self.cfg.bw_connection_mode == "layered":
            self.BII_layered = [w.T.copy().astype(np.float32) for w in self.WPP[1:]]
        elif self.cfg.bw_connection_mode == "skip":
            blocks: list[list[np.ndarray | None]] = []
            for i, _lower in enumerate(self.error_dims):
                row: list[np.ndarray | None] = []
                for j, _upper in enumerate(self.error_dims):
                    if j == i + 1:
                        row.append(self.WPP[j].T.copy().astype(np.float32))
                    else:
                        row.append(None)
                blocks.append(row)
            self.BII_skip = blocks

    def fixed_projection(self) -> np.ndarray:
        return phase.normalize_rows(
            self.rng.normal(0.0, 1.0 / math.sqrt(self.d_model), (self.d_model, self.d_model)).astype(np.float32)
        )

    @property
    def basal_gain(self) -> float:
        return self.cfg.gbas / (self.cfg.gl + self.cfg.gbas + self.cfg.gapi)

    @property
    def apical_gain(self) -> float:
        return self.cfg.gapi / (self.cfg.gl + self.cfg.gbas + self.cfg.gapi)

    def parameter_count(self) -> int:
        arrays: list[np.ndarray] = [self.token_codes, self.position_codes, self.output_bias, self.output_counts]
        arrays.extend(self.WPP)
        arrays.extend(self.BII_layered)
        arrays.extend(block for row in self.BII_skip for block in row if block is not None)
        arrays.extend(self.attn_q)
        arrays.extend(self.attn_k)
        arrays.extend(self.attn_v)
        arrays.extend(self.attn_o)
        return int(sum(array.size for array in arrays))

    def state_bytes(self) -> int:
        arrays: list[np.ndarray] = [self.token_codes, self.position_codes, self.output_bias, self.output_counts]
        arrays.extend(self.WPP)
        arrays.extend(self.BII_layered)
        arrays.extend(block for row in self.BII_skip for block in row if block is not None)
        arrays.extend(self.attn_q)
        arrays.extend(self.attn_k)
        arrays.extend(self.attn_v)
        arrays.extend(self.attn_o)
        return int(sum(array.nbytes for array in arrays))

    def context_matrix(self, context: np.ndarray) -> np.ndarray:
        clipped = np.asarray(context[-self.context_len :], dtype=np.int64)
        offset = self.context_len - len(clipped)
        x = self.token_codes[clipped] + self.position_codes[offset:]
        return phase.normalize_rows(x.astype(np.float32))

    def attention_read(self, seq: np.ndarray, layer_idx: int) -> np.ndarray:
        if self.cfg.attention_scale == 0.0:
            return np.zeros(self.d_model, dtype=np.float32)
        q = seq[-1] @ self.attn_q[layer_idx].T
        k = seq @ self.attn_k[layer_idx].T
        v = seq @ self.attn_v[layer_idx].T
        logits = (k @ q) / math.sqrt(max(self.d_model, 1))
        weights = phase.softmax(logits.astype(np.float32), temperature=1.0)
        read = weights @ v
        out = read @ self.attn_o[layer_idx].T
        return phase.normalize_vector(out.astype(np.float32))

    def forward_free(self, context: np.ndarray, collect: bool = False) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        seq = self.context_matrix(context)
        state = seq[-1]
        traces: list[dict[str, np.ndarray]] = []
        for layer_idx in range(self.blocks):
            attn = self.attention_read(seq, layer_idx)
            pre_rate = phase.normalize_vector(state + self.cfg.attention_scale * attn)
            vbas = self.WPP[layer_idx] @ pre_rate
            u_basal = self.basal_gain * vbas
            rate = np.tanh(u_basal).astype(np.float32)
            state = phase.normalize_vector(self.cfg.residual_scale * pre_rate + self.cfg.ff_scale * rate)
            seq = seq.copy()
            seq[-1] = state
            if collect:
                traces.append(
                    {
                        "pre_rate": pre_rate,
                        "vbas": vbas.astype(np.float32),
                        "u_basal": u_basal.astype(np.float32),
                        "rate": rate,
                        "state": state,
                    }
                )
        vbas_out = self.WPP[-1] @ state
        logits = (self.cfg.logit_scale * (self.basal_gain * vbas_out) + self.output_bias).astype(np.float32)
        if collect:
            traces.append(
                {
                    "pre_rate": state,
                    "vbas": vbas_out.astype(np.float32),
                    "u_basal": (self.basal_gain * vbas_out).astype(np.float32),
                    "rate": logits,
                    "state": logits,
                }
            )
        return logits, traces

    def logits(self, context: np.ndarray) -> np.ndarray:
        logits, _ = self.forward_free(context, collect=False)
        return logits

    def output_error(self, logits: np.ndarray, target: int) -> np.ndarray:
        probs = phase.softmax(logits.astype(np.float32), self.cfg.temperature)
        err = -probs.astype(np.float32)
        err[int(target)] += 1.0
        return err.astype(np.float32)

    def hidden_error_rates(
        self,
        traces: list[dict[str, np.ndarray]],
        output_error: np.ndarray,
    ) -> list[np.ndarray]:
        errors = [np.zeros(dim, dtype=np.float32) for dim in self.error_dims]
        errors[-1] = output_error.astype(np.float32)
        if self.cfg.bw_connection_mode == "layered":
            for idx in range(len(errors) - 2, -1, -1):
                derivative = 1.0 - np.square(np.tanh(traces[idx]["u_basal"]))
                topdown = self.BII_layered[idx] @ errors[idx + 1]
                errors[idx] = (derivative * topdown).astype(np.float32) if self.cfg.varphi_transfer else topdown.astype(np.float32)
        elif self.cfg.bw_connection_mode == "skip":
            for idx, dim in enumerate(self.error_dims[:-1]):
                topdown = np.zeros(dim, dtype=np.float32)
                for upper_idx in range(idx + 1, len(self.error_dims)):
                    block = self.BII_skip[idx][upper_idx]
                    if block is not None:
                        topdown = topdown + block @ errors[upper_idx]
                derivative = 1.0 - np.square(np.tanh(traces[idx]["u_basal"]))
                errors[idx] = (derivative * topdown).astype(np.float32) if self.cfg.varphi_transfer else topdown.astype(np.float32)
        else:
            raise ValueError(f"unknown bw_connection_mode: {self.cfg.bw_connection_mode}")
        return errors

    def local_post_diffs(
        self,
        traces: list[dict[str, np.ndarray]],
        error_rates: list[np.ndarray],
    ) -> list[np.ndarray]:
        diffs: list[np.ndarray] = []
        for idx in range(self.blocks):
            vapi = error_rates[idx]
            u_nudged = traces[idx]["u_basal"] + self.apical_gain * vapi
            if self.cfg.dWPP_use_activation:
                post_diff = np.tanh(u_nudged) - np.tanh(traces[idx]["u_basal"])
            else:
                post_diff = u_nudged - traces[idx]["u_basal"]
            diffs.append(post_diff.astype(np.float32))
        out_diff = self.apical_gain * error_rates[-1]
        diffs.append(out_diff.astype(np.float32))
        return diffs

    def update(self, context: np.ndarray, target: int) -> None:
        target = int(target)
        logits, traces = self.forward_free(context, collect=True)
        out_error = self.output_error(logits, target)
        error_rates = self.hidden_error_rates(traces, out_error)
        post_diffs = self.local_post_diffs(traces, error_rates)
        for idx, post_diff in enumerate(post_diffs):
            pre_rate = traces[idx]["pre_rate"]
            eta = self.cfg.eta_output if idx == len(post_diffs) - 1 else self.cfg.eta_fw
            self.WPP[idx] = self.WPP[idx] + eta * np.outer(post_diff, pre_rate).astype(np.float32)
        self.output_counts[target] += 1.0
        freqs = self.output_counts / float(np.sum(self.output_counts))
        self.output_bias = np.log(np.maximum(freqs, 1e-9)).astype(np.float32)
        if self.cfg.model_type == "BP":
            self.set_bp_feedback()


def gsm_records_to_text(path: Path, max_items: int) -> str:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("question", [])
    cots = data.get("cot", [])
    answers = data.get("answer", [])
    count = min(max(int(max_items), 0), len(questions), len(cots), len(answers))
    docs = []
    for idx in range(count):
        docs.append(f"Question: {questions[idx]}\nReasoning: {cots[idx]}\nAnswer: {answers[idx]}\n")
    return "\n".join(docs)


def load_ids(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int]:
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    train_text = ""
    valid_text = ""
    if args.task in {"tinystories", "mix"}:
        train_text += phase.read_prefix(args.train_file, args.train_chars)
        valid_text += phase.read_prefix(args.valid_file, args.valid_chars)
    if args.task in {"gsm8k", "mix"}:
        train_text += "\n" + gsm_records_to_text(args.gsm_train_file, args.gsm_train_items)
        valid_text += "\n" + gsm_records_to_text(args.gsm_valid_file, args.gsm_valid_items)
    train_raw = phase.encode_text(tokenizer, train_text)
    valid_raw = phase.encode_text(tokenizer, valid_text)
    _, train_ids, valid_ids = phase.build_compact_vocab(train_raw, valid_raw, args.max_vocab)
    vocab_size = int(min(args.max_vocab, len(set(train_ids.tolist()) | set(valid_ids.tolist()))))
    return train_ids, valid_ids, vocab_size


def run_pass(
    model: U004PaperErrorMicrocircuitLLM,
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


def collect_context_batch(
    model: U004PaperErrorMicrocircuitLLM,
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


def block_batch_loss(
    model: U004PaperErrorMicrocircuitLLM,
    contexts: list[np.ndarray],
    targets: np.ndarray,
    block_idx: int,
    weights: np.ndarray,
) -> float:
    original = model.WPP[block_idx]
    model.WPP[block_idx] = weights.astype(np.float32, copy=False)
    try:
        losses = []
        for context, target in zip(contexts, targets):
            logits = model.logits(context)
            loss, _ = softmax_loss_and_pred(logits, int(target), model.cfg.temperature)
            losses.append(loss)
        return float(np.mean(losses)) if losses else 0.0
    finally:
        model.WPP[block_idx] = original


def local_block_step(
    model: U004PaperErrorMicrocircuitLLM,
    contexts: list[np.ndarray],
    targets: np.ndarray,
    block_idx: int,
) -> np.ndarray:
    weights = model.WPP[block_idx].copy()
    for context, target in zip(contexts, targets):
        logits, traces = model.forward_free(context, collect=True)
        out_error = model.output_error(logits, int(target))
        error_rates = model.hidden_error_rates(traces, out_error)
        post_diffs = model.local_post_diffs(traces, error_rates)
        eta = model.cfg.eta_output if block_idx == len(model.WPP) - 1 else model.cfg.eta_fw
        weights = weights + eta * np.outer(post_diffs[block_idx], traces[block_idx]["pre_rate"]).astype(np.float32)
    return weights


def run_block_center_diff_diag(
    model: U004PaperErrorMicrocircuitLLM,
    ids: np.ndarray,
    batch_size: int,
    eps: float,
    max_params: int,
) -> list[dict[str, float | int | str]]:
    contexts, targets = collect_context_batch(model, ids, batch_size)
    rows: list[dict[str, float | int | str]] = []
    if not contexts:
        return rows
    for block_idx in range(len(model.WPP)):
        weights0 = model.WPP[block_idx].copy().astype(np.float64)
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
        rng = np.random.default_rng(model.cfg.seed + 20_003 + block_idx)
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


def summarize_diag(rows: list[dict[str, float | int | str]], hidden_blocks: int) -> dict[str, float | int]:
    ok = [row for row in rows if row.get("status") == "ok"]
    hidden = [row for row in ok if int(row["block"]) < hidden_blocks]
    output = [row for row in ok if int(row["block"]) == hidden_blocks]
    out: dict[str, float | int] = {
        "diag_layers": int(len(ok)),
        "hidden_diag_layers": int(len(hidden)),
        "hidden_diag_mean_cosine": float(np.mean([float(row["cosine_local_vs_center_diff"]) for row in hidden]))
        if hidden
        else float("nan"),
        "hidden_diag_mean_loss_change_local": float(np.mean([float(row["loss_change_local"]) for row in hidden]))
        if hidden
        else float("nan"),
        "hidden_diag_mean_loss_change_center_diff": float(
            np.mean([float(row["loss_change_center_diff_scaled"]) for row in hidden])
        )
        if hidden
        else float("nan"),
    }
    if output:
        out["output_diag_cosine"] = float(output[0]["cosine_local_vs_center_diff"])
        out["output_diag_loss_change_local"] = float(output[0]["loss_change_local"])
    else:
        out["output_diag_cosine"] = float("nan")
        out["output_diag_loss_change_local"] = float("nan")
    return out


def estimate_large_params(vocab_size: int, d_model: int, blocks: int, bw_connection_mode: str) -> int:
    vocab = max(int(vocab_size), 1)
    d = max(int(d_model), 1)
    b = max(int(blocks), 1)
    embeddings = vocab * d
    hidden_wpp = b * d * d
    output = vocab * d
    hidden_dims = [d for _ in range(b)] + [vocab]
    if bw_connection_mode == "layered":
        bii = sum(lower * upper for lower, upper in zip(hidden_dims[:-1], hidden_dims[1:]))
    else:
        bii = 0
        for idx, lower in enumerate(hidden_dims):
            for upper in hidden_dims[idx + 1 :]:
                bii += lower * upper
    attention = 4 * b * d * d
    return int(embeddings + hidden_wpp + output + bii + attention)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tinystories", "gsm8k", "mix"], default="tinystories")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--gsm-train-file", type=Path, default=DEFAULT_GSM_TRAIN)
    parser.add_argument("--gsm-valid-file", type=Path, default=DEFAULT_GSM_VALID)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "u004_paper_error_mc_llm")
    parser.add_argument("--train-chars", type=int, default=20_000)
    parser.add_argument("--valid-chars", type=int, default=5_000)
    parser.add_argument("--gsm-train-items", type=int, default=256)
    parser.add_argument("--gsm-valid-items", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--eta-fw", type=float, default=0.02)
    parser.add_argument("--eta-output", type=float, default=0.05)
    parser.add_argument("--logit-scale", type=float, default=8.0)
    parser.add_argument("--attention-scale", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--ff-scale", type=float, default=1.0)
    parser.add_argument("--dWPP-use-activation", action="store_true")
    parser.add_argument("--no-varphi-transfer", action="store_true")
    parser.add_argument("--fw-connection-mode", choices=["layered"], default="layered")
    parser.add_argument("--bw-connection-mode", choices=["layered", "skip"], default="skip")
    parser.add_argument("--model-type", choices=["FA", "BP"], default="FA")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-token-limit", type=int, default=0)
    parser.add_argument("--diag-batch", type=int, default=4)
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
    train_ids, valid_ids, vocab_size = load_ids(args)
    cfg = U004Config(
        context_len=args.context_len,
        d_model=args.d_model,
        blocks=args.blocks,
        eta_fw=args.eta_fw,
        eta_output=args.eta_output,
        logit_scale=args.logit_scale,
        attention_scale=args.attention_scale,
        residual_scale=args.residual_scale,
        ff_scale=args.ff_scale,
        dWPP_use_activation=args.dWPP_use_activation,
        varphi_transfer=not args.no_varphi_transfer,
        fw_connection_mode=args.fw_connection_mode,
        bw_connection_mode=args.bw_connection_mode,
        model_type=args.model_type,
        temperature=args.temperature,
        seed=args.seed,
    )
    model = U004PaperErrorMicrocircuitLLM(vocab_size, cfg)
    start = time.perf_counter()
    train_summary = run_pass(model, train_ids, update=True, limit=args.eval_token_limit)
    train_seconds = time.perf_counter() - start
    valid_pre = run_pass(model, valid_ids, update=False, limit=args.eval_token_limit)
    valid_online = run_pass(model, valid_ids, update=True, limit=args.eval_token_limit)
    valid_post = run_pass(model, valid_ids, update=False, limit=args.eval_token_limit)
    diag_rows = run_block_center_diff_diag(model, valid_ids, args.diag_batch, args.center_eps, args.diag_max_params)
    diag_summary = summarize_diag(diag_rows, model.blocks)
    large_estimate = estimate_large_params(
        args.large_vocab_size,
        args.large_d_model,
        args.large_blocks,
        args.bw_connection_mode,
    )
    summary = {
        "task": args.task,
        "seed": int(args.seed),
        "vocab_size": int(vocab_size),
        "context_len": int(cfg.context_len),
        "d_model": int(cfg.d_model),
        "blocks": int(cfg.blocks),
        "model_type": cfg.model_type,
        "bw_connection_mode": cfg.bw_connection_mode,
        "attention_scale": float(cfg.attention_scale),
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
    summary.update(diag_summary)
    write_csv(args.out_dir / "summary.csv", [summary])
    write_csv(args.out_dir / "center_diff_diagnostic.csv", diag_rows)
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
        f"  task={args.task} model={cfg.model_type}/{cfg.bw_connection_mode} "
        f"params={model.parameter_count():,} large_estimate={large_estimate:,} "
        f"valid_post={valid_post['loss']:.3f}/{valid_post['accuracy']:.3f} "
        f"out_cos={diag_summary.get('output_diag_cosine', float('nan')):.3f} "
        f"hidden_mean_cos={diag_summary.get('hidden_diag_mean_cosine', float('nan')):.3f}"
    )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
