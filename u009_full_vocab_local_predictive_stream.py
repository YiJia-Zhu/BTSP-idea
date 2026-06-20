#!/usr/bin/env python3
"""
U009 full-tokenizer local predictive stream learner.

This experiment keeps the Llama tokenizer ids intact: every tokenizer id has an
output neuron/logit.  The learning rule is still pure no-BP:

  - no autograd, no pretrained weights, no BPTT;
  - the output layer updates from its own softmax error;
  - hidden layers receive a token-population error represented in a fixed code
    space, then update only from their local input and local activation.

The architecture is transformer-like only in the forward path: causal attention
over a recent context window, sinusoidal positions, residual blocks, and a
full-vocabulary next-token readout.
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


@dataclass
class U009Config:
    context_len: int = 64
    d_model: int = 256
    blocks: int = 3
    output_lr: float = 0.015
    bias_lr: float = 0.010
    hidden_lr: float = 0.002
    hidden_bias_lr: float = 0.0005
    embedding_lr: float = 0.0005
    logit_scale: float = 1.0
    attention_scale: float = 0.70
    ff_scale: float = 0.35
    position_scale: float = 0.25
    temperature: float = 1.0
    row_norm_interval: int = 128
    readout_init: str = "target_code"
    bias_mode: str = "count"
    bias_alpha: float = 0.01
    seed: int = 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(loss_sum: float, correct: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"loss": float("nan"), "ppl": float("nan"), "accuracy": 0.0, "tokens": 0}
    loss = loss_sum / float(total)
    return {
        "loss": float(loss),
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": float(correct / float(total)),
        "tokens": int(total),
    }


def softmax_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits.astype(np.float32, copy=False) / max(float(temperature), 1e-6)
    z = z - float(np.max(z))
    exp_z = np.exp(z, dtype=np.float32)
    denom = float(np.sum(exp_z))
    return (exp_z / max(denom, 1e-12)).astype(np.float32)


def loss_pred_probs(logits: np.ndarray, target: int, temperature: float) -> tuple[float, int, np.ndarray]:
    probs = softmax_probs(logits, temperature)
    target = int(target)
    return -math.log(float(probs[target]) + 1e-12), int(np.argmax(probs)), probs


def normalize_vector(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x) + 1e-8)).astype(np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return (x / norms).astype(np.float32)


def sinusoidal_positions(context_len: int, dim: int) -> np.ndarray:
    context_len = max(int(context_len), 1)
    dim = max(int(dim), 1)
    positions = np.arange(context_len, dtype=np.float32)[:, None]
    half = max(dim // 2, 1)
    div = np.exp(np.arange(half, dtype=np.float32) * (-math.log(10000.0) / max(half, 1)))
    angles = positions * div[None, :]
    codes = np.zeros((context_len, dim), dtype=np.float32)
    codes[:, 0 : 2 * half : 2] = np.sin(angles[:, : codes[:, 0::2].shape[1]])
    codes[:, 1 : 2 * half : 2] = np.cos(angles[:, : codes[:, 1::2].shape[1]])
    return normalize_rows(codes)


def split_text_documents(text: str, fallback_chars: int) -> list[str]:
    docs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(docs) > 1 or fallback_chars <= 0:
        return docs if docs else ([text] if text.strip() else [])
    stripped = text.strip()
    return [stripped[idx : idx + fallback_chars] for idx in range(0, len(stripped), fallback_chars)]


def gsm_records_to_docs(path: Path, max_items: int) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("question", [])
    cots = data.get("cot", [])
    answers = data.get("answer", [])
    count = min(max(int(max_items), 0), len(questions), len(cots), len(answers))
    docs: list[str] = []
    for idx in range(count):
        docs.append(f"Question: {questions[idx]}\nReasoning: {cots[idx]}\nAnswer: {answers[idx]}\n")
    return docs


def interleave_doc_groups(groups: list[list[str]]) -> list[str]:
    out: list[str] = []
    max_len = max((len(group) for group in groups), default=0)
    for idx in range(max_len):
        for group in groups:
            if idx < len(group):
                out.append(group[idx])
    return out


def load_text_documents(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    train_groups: list[list[str]] = []
    valid_groups: list[list[str]] = []
    if args.task in {"tinystories", "mix"}:
        train_groups.append(split_text_documents(phase.read_prefix(args.train_file, args.train_chars), args.doc_chars))
        valid_groups.append(split_text_documents(phase.read_prefix(args.valid_file, args.valid_chars), args.doc_chars))
    if args.task in {"gsm8k", "mix"}:
        train_groups.append(gsm_records_to_docs(args.gsm_train_file, args.gsm_train_items))
        valid_groups.append(gsm_records_to_docs(args.gsm_valid_file, args.gsm_valid_items))
    return interleave_doc_groups(train_groups), interleave_doc_groups(valid_groups)


def tokenize_documents(tokenizer: Any, docs: list[str]) -> list[np.ndarray]:
    arrays = [phase.encode_text(tokenizer, doc) for doc in docs if doc.strip()]
    return [arr.astype(np.int64, copy=False) for arr in arrays if arr.size >= 2]


def count_document_pairs(docs: list[np.ndarray]) -> int:
    return int(sum(max(int(doc.size) - 1, 0) for doc in docs))


class FullVocabLocalPredictiveModel:
    def __init__(self, vocab_size: int, cfg: U009Config) -> None:
        self.vocab_size = int(vocab_size)
        self.cfg = cfg
        self.context_len = max(int(cfg.context_len), 1)
        self.d_model = max(int(cfg.d_model), 1)
        self.blocks = max(int(cfg.blocks), 1)
        self.rng = np.random.default_rng(cfg.seed)
        self.step = 0

        self.input_codes = normalize_rows(
            self.rng.normal(0.0, 1.0, (self.vocab_size, self.d_model)).astype(np.float32)
        )
        self.target_codes = normalize_rows(
            self.rng.normal(0.0, 1.0, (self.vocab_size, self.d_model)).astype(np.float32)
        )
        self.position_codes = sinusoidal_positions(self.context_len, self.d_model)

        self.ff_weights = [
            self.rng.normal(0.0, 1.0 / math.sqrt(self.d_model), (self.d_model, self.d_model)).astype(np.float32)
            for _ in range(self.blocks)
        ]
        self.ff_biases = [np.zeros(self.d_model, dtype=np.float32) for _ in range(self.blocks)]
        self.attn_q = [self._fixed_projection() for _ in range(self.blocks)]
        self.attn_k = [self._fixed_projection() for _ in range(self.blocks)]
        self.attn_v = [self._fixed_projection() for _ in range(self.blocks)]
        self.attn_o = [self._fixed_projection() for _ in range(self.blocks)]

        if cfg.readout_init == "target_code":
            self.output_weights = self.target_codes.copy()
        elif cfg.readout_init == "random":
            self.output_weights = self.rng.normal(
                0.0, 0.02 / math.sqrt(self.d_model), (self.vocab_size, self.d_model)
            ).astype(np.float32)
        else:
            raise ValueError(f"unknown readout_init: {cfg.readout_init}")
        self.output_counts = np.full(self.vocab_size, max(float(cfg.bias_alpha), 1e-9), dtype=np.float32)
        self.output_bias = np.full(self.vocab_size, -math.log(max(self.vocab_size, 1)), dtype=np.float32)
        self.refresh_count_bias()

    def _fixed_projection(self) -> np.ndarray:
        return normalize_rows(
            self.rng.normal(0.0, 1.0 / math.sqrt(self.d_model), (self.d_model, self.d_model)).astype(np.float32)
        )

    def parameter_count(self) -> int:
        arrays: list[np.ndarray] = [
            self.input_codes,
            self.target_codes,
            self.position_codes,
            self.output_weights,
            self.output_bias,
            self.output_counts,
        ]
        arrays.extend(self.ff_weights)
        arrays.extend(self.ff_biases)
        arrays.extend(self.attn_q)
        arrays.extend(self.attn_k)
        arrays.extend(self.attn_v)
        arrays.extend(self.attn_o)
        return int(sum(array.size for array in arrays))

    def state_bytes(self) -> int:
        arrays: list[np.ndarray] = [
            self.input_codes,
            self.target_codes,
            self.position_codes,
            self.output_weights,
            self.output_bias,
            self.output_counts,
        ]
        arrays.extend(self.ff_weights)
        arrays.extend(self.ff_biases)
        arrays.extend(self.attn_q)
        arrays.extend(self.attn_k)
        arrays.extend(self.attn_v)
        arrays.extend(self.attn_o)
        return int(sum(array.nbytes for array in arrays))

    def context_matrix(self, context: np.ndarray) -> np.ndarray:
        clipped = np.asarray(context[-self.context_len :], dtype=np.int64)
        offset = self.context_len - int(clipped.size)
        x = self.input_codes[clipped] + self.cfg.position_scale * self.position_codes[offset:]
        return normalize_rows(x.astype(np.float32))

    def attention_read(self, seq: np.ndarray, block_idx: int) -> np.ndarray:
        q = seq[-1] @ self.attn_q[block_idx].T
        k = seq @ self.attn_k[block_idx].T
        v = seq @ self.attn_v[block_idx].T
        logits = (k @ q).astype(np.float32) / math.sqrt(max(self.d_model, 1))
        weights = softmax_probs(logits, temperature=1.0)
        read = weights @ v
        out = read @ self.attn_o[block_idx].T
        return normalize_vector(out.astype(np.float32))

    def forward(self, context: np.ndarray, collect: bool = False) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        seq = self.context_matrix(context)
        current = seq[-1]
        traces: list[dict[str, np.ndarray]] = []
        for block_idx in range(self.blocks):
            attn = self.attention_read(seq, block_idx)
            local_input = normalize_vector(current + self.cfg.attention_scale * attn)
            pre = self.ff_weights[block_idx] @ local_input + self.ff_biases[block_idx]
            hidden = np.tanh(pre).astype(np.float32)
            current = normalize_vector(local_input + self.cfg.ff_scale * hidden)
            seq = seq.copy()
            seq[-1] = current
            if collect:
                traces.append({"input": local_input, "hidden": hidden})
        return current.astype(np.float32), traces

    def logits_from_feature(self, feature: np.ndarray) -> np.ndarray:
        return (self.cfg.logit_scale * (self.output_weights @ feature) + self.output_bias).astype(np.float32)

    def logits(self, context: np.ndarray) -> np.ndarray:
        feature, _ = self.forward(context, collect=False)
        return self.logits_from_feature(feature)

    def update_from_forward(
        self,
        context: np.ndarray,
        current_token: int,
        target: int,
        feature: np.ndarray,
        traces: list[dict[str, np.ndarray]],
        probs: np.ndarray,
    ) -> None:
        target = int(target)
        error = -probs.astype(np.float32, copy=True)
        error[target] += 1.0

        if self.cfg.output_lr > 0.0:
            self.output_weights += (self.cfg.output_lr * error)[:, None] * feature[None, :]
        if self.cfg.bias_mode == "sgd" and self.cfg.bias_lr > 0.0:
            self.output_bias += self.cfg.bias_lr * error
        elif self.cfg.bias_mode == "count":
            self.output_counts[target] += 1.0
            self.refresh_count_bias()
        elif self.cfg.bias_mode != "none":
            raise ValueError(f"unknown bias_mode: {self.cfg.bias_mode}")

        if self.cfg.hidden_lr > 0.0 or self.cfg.embedding_lr > 0.0:
            expected_code = probs @ self.target_codes
            code_error = (self.target_codes[target] - expected_code).astype(np.float32)
            if self.cfg.hidden_lr > 0.0:
                for block_idx, trace in enumerate(traces):
                    local_input = trace["input"]
                    hidden = trace["hidden"]
                    delta = code_error * (1.0 - np.square(hidden))
                    self.ff_weights[block_idx] += self.cfg.hidden_lr * np.outer(delta, local_input).astype(np.float32)
                    if self.cfg.hidden_bias_lr > 0.0:
                        self.ff_biases[block_idx] += self.cfg.hidden_bias_lr * delta.astype(np.float32)
            if self.cfg.embedding_lr > 0.0:
                token = int(current_token)
                self.input_codes[token] = normalize_vector(self.input_codes[token] + self.cfg.embedding_lr * code_error)

        self.step += 1
        if self.cfg.row_norm_interval > 0 and self.step % int(self.cfg.row_norm_interval) == 0:
            self.ff_weights = [normalize_rows(weight) for weight in self.ff_weights]

    def refresh_count_bias(self) -> None:
        probs = self.output_counts / max(float(np.sum(self.output_counts)), 1e-12)
        self.output_bias = np.log(np.maximum(probs, 1e-12)).astype(np.float32)

    def observe(self, context: np.ndarray, target: int) -> tuple[float, int]:
        feature, traces = self.forward(context, collect=True)
        logits = self.logits_from_feature(feature)
        loss, pred, probs = loss_pred_probs(logits, int(target), self.cfg.temperature)
        self.update_from_forward(context, int(context[-1]), int(target), feature, traces, probs)
        return loss, pred


def run_documents(
    model: FullVocabLocalPredictiveModel,
    docs: list[np.ndarray],
    update: bool,
    limit: int,
    chunk_tokens: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    loss_sum = 0.0
    correct = 0
    total = 0
    chunk_loss = 0.0
    chunk_correct = 0
    chunk_total = 0
    chunks: list[dict[str, float | int | str]] = []
    context_len = model.context_len
    for doc_idx, doc in enumerate(docs):
        if doc.size < 2:
            continue
        for pos in range(1, int(doc.size)):
            left = max(0, pos - context_len)
            context = doc[left:pos]
            target = int(doc[pos])
            if update:
                loss, pred = model.observe(context, target)
            else:
                logits = model.logits(context)
                loss, pred, _ = loss_pred_probs(logits, target, model.cfg.temperature)
            hit = int(pred == target)
            loss_sum += loss
            correct += hit
            total += 1
            chunk_loss += loss
            chunk_correct += hit
            chunk_total += 1
            if chunk_tokens > 0 and chunk_total >= chunk_tokens:
                chunks.append(
                    {
                        "phase": "train" if update else "eval",
                        "chunk": int(len(chunks)),
                        "tokens_seen": int(total),
                        "doc_idx": int(doc_idx),
                        "loss": float(chunk_loss / float(chunk_total)),
                        "accuracy": float(chunk_correct / float(chunk_total)),
                        "tokens": int(chunk_total),
                    }
                )
                chunk_loss = 0.0
                chunk_correct = 0
                chunk_total = 0
            if limit > 0 and total >= limit:
                if chunk_total > 0:
                    chunks.append(
                        {
                            "phase": "train" if update else "eval",
                            "chunk": int(len(chunks)),
                            "tokens_seen": int(total),
                            "doc_idx": int(doc_idx),
                            "loss": float(chunk_loss / float(chunk_total)),
                            "accuracy": float(chunk_correct / float(chunk_total)),
                            "tokens": int(chunk_total),
                        }
                    )
                return summarize(loss_sum, correct, total), chunks
    if chunk_total > 0:
        chunks.append(
            {
                "phase": "train" if update else "eval",
                "chunk": int(len(chunks)),
                "tokens_seen": int(total),
                "doc_idx": int(max(len(docs) - 1, 0)),
                "loss": float(chunk_loss / float(chunk_total)),
                "accuracy": float(chunk_correct / float(chunk_total)),
                "tokens": int(chunk_total),
            }
        )
    return summarize(loss_sum, correct, total), chunks


def unigram_from_docs(docs: list[np.ndarray], vocab_size: int) -> np.ndarray:
    counts = np.ones(int(vocab_size), dtype=np.float32)
    for doc in docs:
        if doc.size > 1:
            np.add.at(counts, doc[1:].astype(np.int64, copy=False), 1.0)
    return counts / float(np.sum(counts))


def evaluate_unigram(docs: list[np.ndarray], probs: np.ndarray, limit: int) -> dict[str, float | int]:
    loss_sum = 0.0
    correct = 0
    total = 0
    pred = int(np.argmax(probs))
    for doc in docs:
        for pos in range(1, int(doc.size)):
            target = int(doc[pos])
            loss_sum += -math.log(float(probs[target]) + 1e-12)
            correct += int(pred == target)
            total += 1
            if limit > 0 and total >= limit:
                return summarize(loss_sum, correct, total)
    return summarize(loss_sum, correct, total)


def generate_tokens(
    model: FullVocabLocalPredictiveModel,
    prompt: np.ndarray,
    max_new_tokens: int,
    sample: bool,
    rng: np.random.Generator,
) -> list[int]:
    out = [int(x) for x in prompt.tolist()]
    if not out:
        return []
    for _ in range(max(int(max_new_tokens), 0)):
        context = np.asarray(out[-model.context_len :], dtype=np.int64)
        logits = model.logits(context)
        probs = softmax_probs(logits, model.cfg.temperature)
        if sample:
            sample_probs = probs.astype(np.float64)
            sample_probs /= max(float(np.sum(sample_probs)), 1e-12)
            next_id = int(rng.choice(model.vocab_size, p=sample_probs))
        else:
            next_id = int(np.argmax(probs))
        out.append(next_id)
    return out


def generate_samples(
    model: FullVocabLocalPredictiveModel,
    docs: list[np.ndarray],
    tokenizer: Any,
    sample_count: int,
    prompt_tokens: int,
    new_tokens: int,
    seed: int,
) -> list[dict[str, str | int]]:
    rng = np.random.default_rng(seed + 1709)
    rows: list[dict[str, str | int]] = []
    usable_docs = [doc for doc in docs if int(doc.size) > max(int(prompt_tokens), 1)]
    for idx, doc in enumerate(usable_docs[: max(int(sample_count), 0)]):
        prompt = doc[: max(int(prompt_tokens), 1)]
        greedy = generate_tokens(model, prompt, new_tokens, sample=False, rng=rng)
        sampled = generate_tokens(model, prompt, new_tokens, sample=True, rng=rng)
        prompt_text = tokenizer.decode([int(x) for x in prompt.tolist()], skip_special_tokens=False)
        rows.append(
            {
                "sample": int(idx),
                "prompt_tokens": int(prompt.size),
                "generated_tokens": int(max(int(new_tokens), 0)),
                "prompt": prompt_text,
                "greedy_completion": tokenizer.decode(greedy[int(prompt.size) :], skip_special_tokens=False),
                "sampled_completion": tokenizer.decode(sampled[int(prompt.size) :], skip_special_tokens=False),
                "greedy_full_text": tokenizer.decode(greedy, skip_special_tokens=False),
                "sampled_full_text": tokenizer.decode(sampled, skip_special_tokens=False),
            }
        )
    return rows


def write_samples(path: Path, samples: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(f"Sample {row['sample']}\n")
            f.write(f"Prompt tokens: {row['prompt_tokens']}  Generated tokens: {row['generated_tokens']}\n")
            f.write("Prompt:\n")
            f.write(str(row["prompt"]).strip() + "\n")
            f.write("Greedy completion:\n")
            f.write(str(row["greedy_completion"]).strip() + "\n")
            f.write("Sampled completion:\n")
            f.write(str(row["sampled_completion"]).strip() + "\n")
            f.write("Greedy full:\n")
            f.write(str(row["greedy_full_text"]).strip() + "\n")
            f.write("Sampled full:\n")
            f.write(str(row["sampled_full_text"]).strip() + "\n\n")


def estimate_large_params(vocab_size: int, d_model: int, blocks: int) -> int:
    vocab = max(int(vocab_size), 1)
    d = max(int(d_model), 1)
    b = max(int(blocks), 1)
    token_tables = 3 * vocab * d + vocab
    per_block = 5 * d * d + d
    return int(token_tables + b * per_block + max(int(context_position_count(d)), 1))


def context_position_count(d_model: int) -> int:
    return 4096 * max(int(d_model), 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tinystories", "gsm8k", "mix"], default="mix")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--gsm-train-file", type=Path, default=DEFAULT_GSM_TRAIN)
    parser.add_argument("--gsm-valid-file", type=Path, default=DEFAULT_GSM_VALID)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "u009_full_vocab_local_predictive")
    parser.add_argument("--train-chars", type=int, default=60_000)
    parser.add_argument("--valid-chars", type=int, default=12_000)
    parser.add_argument("--doc-chars", type=int, default=1_200)
    parser.add_argument("--gsm-train-items", type=int, default=96)
    parser.add_argument("--gsm-valid-items", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--output-lr", type=float, default=0.015)
    parser.add_argument("--bias-lr", type=float, default=0.010)
    parser.add_argument("--hidden-lr", type=float, default=0.002)
    parser.add_argument("--hidden-bias-lr", type=float, default=0.0005)
    parser.add_argument("--embedding-lr", type=float, default=0.0005)
    parser.add_argument("--logit-scale", type=float, default=1.0)
    parser.add_argument("--attention-scale", type=float, default=0.70)
    parser.add_argument("--ff-scale", type=float, default=0.35)
    parser.add_argument("--position-scale", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--row-norm-interval", type=int, default=128)
    parser.add_argument("--readout-init", choices=["target_code", "random"], default="target_code")
    parser.add_argument("--bias-mode", choices=["count", "sgd", "none"], default="count")
    parser.add_argument("--bias-alpha", type=float, default=0.01)
    parser.add_argument("--train-token-limit", type=int, default=0)
    parser.add_argument("--eval-token-limit", type=int, default=0)
    parser.add_argument("--chunk-tokens", type=int, default=200)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--sample-prompt-tokens", type=int, default=24)
    parser.add_argument("--sample-new-tokens", type=int, default=64)
    parser.add_argument("--large-d-model", type=int, default=2048)
    parser.add_argument("--large-blocks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    train_text_docs, valid_text_docs = load_text_documents(args)
    train_docs = tokenize_documents(tokenizer, train_text_docs)
    valid_docs = tokenize_documents(tokenizer, valid_text_docs)
    vocab_size = int(len(tokenizer))
    if not train_docs:
        raise ValueError("no training documents after tokenization")
    if not valid_docs:
        raise ValueError("no validation documents after tokenization")

    cfg = U009Config(
        context_len=args.context_len,
        d_model=args.d_model,
        blocks=args.blocks,
        output_lr=args.output_lr,
        bias_lr=args.bias_lr,
        hidden_lr=args.hidden_lr,
        hidden_bias_lr=args.hidden_bias_lr,
        embedding_lr=args.embedding_lr,
        logit_scale=args.logit_scale,
        attention_scale=args.attention_scale,
        ff_scale=args.ff_scale,
        position_scale=args.position_scale,
        temperature=args.temperature,
        row_norm_interval=args.row_norm_interval,
        readout_init=args.readout_init,
        bias_mode=args.bias_mode,
        bias_alpha=args.bias_alpha,
        seed=args.seed,
    )
    model = FullVocabLocalPredictiveModel(vocab_size, cfg)

    pre_train_probe, _ = run_documents(model, train_docs, update=False, limit=args.eval_token_limit, chunk_tokens=0)
    valid_pre, _ = run_documents(model, valid_docs, update=False, limit=args.eval_token_limit, chunk_tokens=0)
    start = time.perf_counter()
    train_online, train_chunks = run_documents(
        model,
        train_docs,
        update=True,
        limit=args.train_token_limit,
        chunk_tokens=args.chunk_tokens,
    )
    train_seconds = time.perf_counter() - start
    post_train_probe, _ = run_documents(model, train_docs, update=False, limit=args.eval_token_limit, chunk_tokens=0)
    valid_post, _ = run_documents(model, valid_docs, update=False, limit=args.eval_token_limit, chunk_tokens=0)
    train_unigram = unigram_from_docs(train_docs, vocab_size)
    valid_unigram = evaluate_unigram(valid_docs, train_unigram, args.eval_token_limit)
    samples = generate_samples(
        model,
        valid_docs,
        tokenizer,
        args.sample_count,
        args.sample_prompt_tokens,
        args.sample_new_tokens,
        args.seed,
    )
    write_samples(args.out_dir / "samples.txt", samples)
    write_csv(args.out_dir / "train_chunks.csv", train_chunks)

    large_estimate = estimate_large_params(vocab_size, args.large_d_model, args.large_blocks)
    summary = {
        "task": args.task,
        "seed": int(args.seed),
        "tokenizer": str(args.tokenizer),
        "tokenizer_len": int(vocab_size),
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", vocab_size)),
        "train_docs": int(len(train_docs)),
        "valid_docs": int(len(valid_docs)),
        "train_available_pairs": int(count_document_pairs(train_docs)),
        "valid_available_pairs": int(count_document_pairs(valid_docs)),
        "context_len": int(args.context_len),
        "d_model": int(args.d_model),
        "blocks": int(args.blocks),
        "readout_init": args.readout_init,
        "bias_mode": args.bias_mode,
        "bias_alpha": float(args.bias_alpha),
        "parameters": int(model.parameter_count()),
        "state_bytes": int(model.state_bytes()),
        "large_param_estimate": int(large_estimate),
        "large_d_model": int(args.large_d_model),
        "large_blocks": int(args.large_blocks),
        "pre_train_probe_loss": pre_train_probe["loss"],
        "pre_train_probe_acc": pre_train_probe["accuracy"],
        "valid_pre_loss": valid_pre["loss"],
        "valid_pre_acc": valid_pre["accuracy"],
        "train_online_loss": train_online["loss"],
        "train_online_acc": train_online["accuracy"],
        "train_online_tokens": train_online["tokens"],
        "post_train_probe_loss": post_train_probe["loss"],
        "post_train_probe_acc": post_train_probe["accuracy"],
        "valid_post_loss": valid_post["loss"],
        "valid_post_acc": valid_post["accuracy"],
        "valid_unigram_loss": valid_unigram["loss"],
        "valid_unigram_acc": valid_unigram["accuracy"],
        "train_seconds": float(train_seconds),
        "tokens_per_second": float(train_online["tokens"] / max(train_seconds, 1e-9)),
    }
    write_csv(args.out_dir / "summary.csv", [summary])
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "config": asdict(cfg),
                "summary": summary,
                "train_chunks": train_chunks,
                "samples": samples,
            },
            f,
            indent=2,
        )

    first_chunk = train_chunks[0]["loss"] if train_chunks else float("nan")
    last_chunk = train_chunks[-1]["loss"] if train_chunks else float("nan")
    print("Summary:")
    print(
        f"  task={args.task} tokenizer_len={vocab_size} d={args.d_model} blocks={args.blocks} "
        f"params={model.parameter_count():,} state={model.state_bytes() / (1024 ** 2):.1f} MiB"
    )
    print(
        f"  train_online={train_online['loss']:.4f}/{train_online['accuracy']:.4f} "
        f"chunks={first_chunk:.4f}->{last_chunk:.4f} tokens={train_online['tokens']} "
        f"speed={summary['tokens_per_second']:.2f} tok/s"
    )
    print(
        f"  probe_train={pre_train_probe['loss']:.4f}->{post_train_probe['loss']:.4f} "
        f"valid={valid_pre['loss']:.4f}->{valid_post['loss']:.4f} "
        f"valid_unigram={valid_unigram['loss']:.4f}"
    )
    print(f"  samples: {args.out_dir / 'samples.txt'}")
    print(f"  summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
