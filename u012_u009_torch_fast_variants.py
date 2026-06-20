#!/usr/bin/env python3
"""
U012 torch.no_grad acceleration variants for the U009 local learner.

Modes:

  exact:  same per-token update order as U009, but tensors live on GPU.
  chunk:  same local update formulas, but updates are accumulated over a chunk
          and applied with GEMM.  This changes online timing, not the local
          target/error rule.
  samplebatch:
          same local update formulas, but each batch takes at most one
          next-token pair from each document/sequence before applying a GEMM
          update.  This avoids batching multiple positions from the same seq.
  packed:
          GPT/DeepSeek-style packed LM sequences. Documents are concatenated
          with EOS boundaries, split into context_len-token blocks, and all
          next-token positions in a block batch update together.

No autograd or BP is used in either mode.
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
import torch
from transformers import AutoTokenizer

import packed_lm_data
import phase_binding_token_experiment as phase
import u009_full_vocab_local_predictive_stream as u009


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = phase.DEFAULT_TRAIN
DEFAULT_VALID = phase.DEFAULT_VALID
DEFAULT_TOKENIZER = phase.DEFAULT_TOKENIZER
DEFAULT_GSM_TRAIN = u009.DEFAULT_GSM_TRAIN
DEFAULT_GSM_VALID = u009.DEFAULT_GSM_VALID


@dataclass
class U012Config:
    context_len: int = 48
    d_model: int = 64
    blocks: int = 3
    attn_rank: int = 0
    output_lr: float = 0.030
    hidden_lr: float = 0.004
    hidden_bias_lr: float = 0.0005
    embedding_lr: float = 0.001
    logit_scale: float = 2.0
    attention_scale: float = 0.70
    ff_scale: float = 0.35
    position_scale: float = 0.25
    temperature: float = 1.0
    row_norm_interval: int = 128
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


def count_document_pairs(docs: list[np.ndarray]) -> int:
    return int(sum(max(int(doc.size) - 1, 0) for doc in docs))


def torch_normalize_vector(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.vector_norm(x), min=1e-8)


def torch_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.vector_norm(x, dim=1, keepdim=True), min=1e-8)


def np_normalize_rows(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def np_sinusoidal_positions(context_len: int, dim: int) -> np.ndarray:
    positions = np.arange(max(int(context_len), 1), dtype=np.float32)[:, None]
    half = max(int(dim) // 2, 1)
    div = np.exp(np.arange(half, dtype=np.float32) * (-math.log(10000.0) / max(half, 1)))
    angles = positions * div[None, :]
    codes = np.zeros((context_len, dim), dtype=np.float32)
    codes[:, 0 : 2 * half : 2] = np.sin(angles[:, : codes[:, 0::2].shape[1]])
    codes[:, 1 : 2 * half : 2] = np.cos(angles[:, : codes[:, 1::2].shape[1]])
    return np_normalize_rows(codes)


class TorchU009LocalModel:
    def __init__(self, vocab_size: int, cfg: U012Config, device: torch.device) -> None:
        self.vocab_size = int(vocab_size)
        self.cfg = cfg
        self.device = device
        self.context_len = max(int(cfg.context_len), 1)
        self.d_model = max(int(cfg.d_model), 1)
        self.blocks = max(int(cfg.blocks), 1)
        self.step = 0
        rng = np.random.default_rng(cfg.seed)

        input_codes = np_normalize_rows(rng.normal(0.0, 1.0, (self.vocab_size, self.d_model)).astype(np.float32))
        target_codes = np_normalize_rows(rng.normal(0.0, 1.0, (self.vocab_size, self.d_model)).astype(np.float32))
        self.input_codes = torch.tensor(input_codes, device=device)
        self.target_codes = torch.tensor(target_codes, device=device)
        self.position_codes = torch.tensor(np_sinusoidal_positions(self.context_len, self.d_model), device=device)

        self.ff_weights = [
            torch.tensor(
                rng.normal(0.0, 1.0 / math.sqrt(self.d_model), (self.d_model, self.d_model)).astype(np.float32),
                device=device,
            )
            for _ in range(self.blocks)
        ]
        self.ff_biases = [torch.zeros(self.d_model, device=device) for _ in range(self.blocks)]
        self.attn_rank = int(cfg.attn_rank) if int(cfg.attn_rank) > 0 else self.d_model
        self.lowrank_attention = self.attn_rank != self.d_model
        self.attn_q = [self._fixed_projection(rng, self.attn_rank, self.d_model) for _ in range(self.blocks)]
        self.attn_k = [self._fixed_projection(rng, self.attn_rank, self.d_model) for _ in range(self.blocks)]
        self.attn_v = [self._fixed_projection(rng, self.attn_rank, self.d_model) for _ in range(self.blocks)]
        self.attn_o = [self._fixed_projection(rng, self.d_model, self.attn_rank) for _ in range(self.blocks)]

        self.output_weights = self.target_codes.clone()
        self.output_counts = torch.full(
            (self.vocab_size,), max(float(cfg.bias_alpha), 1e-9), dtype=torch.float32, device=device
        )
        self.output_total = torch.tensor(float(self.output_counts.sum().item()), device=device)
        self.output_bias = torch.log(torch.clamp(self.output_counts / self.output_total, min=1e-12))

    def _fixed_projection(self, rng: np.random.Generator, rows: int, cols: int) -> torch.Tensor:
        values = rng.normal(0.0, 1.0 / math.sqrt(max(int(cols), 1)), (int(rows), int(cols))).astype(np.float32)
        return torch.tensor(np_normalize_rows(values), device=self.device)

    def parameter_count(self) -> int:
        tensors = [self.input_codes, self.target_codes, self.position_codes, self.output_weights, self.output_bias]
        tensors.extend(self.ff_weights)
        tensors.extend(self.ff_biases)
        tensors.extend(self.attn_q)
        tensors.extend(self.attn_k)
        tensors.extend(self.attn_v)
        tensors.extend(self.attn_o)
        tensors.append(self.output_counts)
        return int(sum(t.numel() for t in tensors))

    def state_bytes(self) -> int:
        tensors = [self.input_codes, self.target_codes, self.position_codes, self.output_weights, self.output_bias]
        tensors.extend(self.ff_weights)
        tensors.extend(self.ff_biases)
        tensors.extend(self.attn_q)
        tensors.extend(self.attn_k)
        tensors.extend(self.attn_v)
        tensors.extend(self.attn_o)
        tensors.append(self.output_counts)
        return int(sum(t.numel() * t.element_size() for t in tensors))

    def context_matrix(self, context: np.ndarray) -> torch.Tensor:
        clipped = np.asarray(context[-self.context_len :], dtype=np.int64)
        ids = torch.as_tensor(clipped, dtype=torch.long, device=self.device)
        offset = self.context_len - int(ids.numel())
        x = self.input_codes.index_select(0, ids) + self.cfg.position_scale * self.position_codes[offset:]
        return torch_normalize_rows(x)

    def attention_read(self, seq: torch.Tensor, block_idx: int) -> torch.Tensor:
        q = seq[-1] @ self.attn_q[block_idx].T
        k = seq @ self.attn_k[block_idx].T
        v = seq @ self.attn_v[block_idx].T
        logits = (k @ q) / math.sqrt(max(self.attn_rank, 1))
        weights = torch.softmax(logits, dim=0)
        read = weights @ v
        out = read @ self.attn_o[block_idx].T
        return torch_normalize_vector(out)

    def batch_context_matrix(self, contexts: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = len(contexts)
        ids = torch.zeros((batch, self.context_len), dtype=torch.long, device=self.device)
        mask = torch.zeros((batch, self.context_len), dtype=torch.bool, device=self.device)
        for row, context in enumerate(contexts):
            clipped = np.asarray(context[-self.context_len :], dtype=np.int64)
            n = int(clipped.size)
            ids[row, self.context_len - n :] = torch.as_tensor(clipped, dtype=torch.long, device=self.device)
            mask[row, self.context_len - n :] = True
        x = self.input_codes[ids] + self.cfg.position_scale * self.position_codes[None, :, :]
        return torch_normalize_rows(x.reshape(-1, self.d_model)).reshape(batch, self.context_len, self.d_model), mask

    def forward_batch(
        self,
        contexts: list[np.ndarray],
        collect: bool,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], list[int]]:
        seq, mask = self.batch_context_matrix(contexts)
        current = seq[:, -1, :]
        traces: list[dict[str, torch.Tensor]] = []
        for block_idx in range(self.blocks):
            q = current @ self.attn_q[block_idx].T
            k = seq @ self.attn_k[block_idx].T
            v = seq @ self.attn_v[block_idx].T
            logits = torch.sum(k * q[:, None, :], dim=-1) / math.sqrt(max(self.attn_rank, 1))
            logits = logits.masked_fill(~mask, -1.0e9)
            weights = torch.softmax(logits, dim=1)
            read = torch.sum(weights[:, :, None] * v, dim=1)
            attn = torch_normalize_rows(read @ self.attn_o[block_idx].T)
            local_input = torch_normalize_rows(current + self.cfg.attention_scale * attn)
            hidden = torch.tanh(local_input @ self.ff_weights[block_idx].T + self.ff_biases[block_idx][None, :])
            current = torch_normalize_rows(local_input + self.cfg.ff_scale * hidden)
            seq = seq.clone()
            seq[:, -1, :] = current
            if collect:
                traces.append({"input": local_input, "hidden": hidden})
        current_tokens = [int(context[-1]) for context in contexts]
        return current, traces, current_tokens

    def forward_one(self, context: np.ndarray, collect: bool) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        seq = self.context_matrix(context)
        current = seq[-1]
        traces: list[dict[str, torch.Tensor]] = []
        for block_idx in range(self.blocks):
            attn = self.attention_read(seq, block_idx)
            local_input = torch_normalize_vector(current + self.cfg.attention_scale * attn)
            pre = self.ff_weights[block_idx] @ local_input + self.ff_biases[block_idx]
            hidden = torch.tanh(pre)
            current = torch_normalize_vector(local_input + self.cfg.ff_scale * hidden)
            seq = seq.clone()
            seq[-1] = current
            if collect:
                traces.append({"input": local_input, "hidden": hidden})
        return current, traces

    def logits_from_feature(self, feature: torch.Tensor) -> torch.Tensor:
        return self.cfg.logit_scale * (self.output_weights @ feature) + self.output_bias

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.cfg.logit_scale * (features @ self.output_weights.T) + self.output_bias[None, :]

    def update_count_bias_one(self, target: int) -> None:
        old_total = self.output_total.clone()
        self.output_counts[target] += 1.0
        self.output_total += 1.0
        self.output_bias -= torch.log(self.output_total / old_total)
        self.output_bias[target] = torch.log(torch.clamp(self.output_counts[target] / self.output_total, min=1e-12))

    def update_count_bias_chunk(self, targets: torch.Tensor) -> None:
        old_total = self.output_total.clone()
        bincount = torch.bincount(targets, minlength=self.vocab_size).to(dtype=torch.float32)
        self.output_counts += bincount
        self.output_total += float(targets.numel())
        self.output_bias = torch.log(torch.clamp(self.output_counts / self.output_total, min=1e-12))

    def observe_exact(self, context: np.ndarray, target: int) -> tuple[float, int]:
        feature, traces = self.forward_one(context, collect=True)
        logits = self.logits_from_feature(feature)
        probs = torch.softmax(logits / max(float(self.cfg.temperature), 1e-6), dim=0)
        target_t = int(target)
        loss = -torch.log(torch.clamp(probs[target_t], min=1e-12))
        pred = int(torch.argmax(probs).item())
        error = -probs
        error[target_t] += 1.0
        torch.addr(self.output_weights, error, feature, beta=1.0, alpha=float(self.cfg.output_lr), out=self.output_weights)
        self.update_count_bias_one(target_t)

        expected_code = probs @ self.target_codes
        code_error = self.target_codes[target_t] - expected_code
        for block_idx, trace in enumerate(traces):
            local_input = trace["input"]
            hidden = trace["hidden"]
            delta = code_error * (1.0 - hidden.square())
            torch.addr(
                self.ff_weights[block_idx],
                delta,
                local_input,
                beta=1.0,
                alpha=float(self.cfg.hidden_lr),
                out=self.ff_weights[block_idx],
            )
            if self.cfg.hidden_bias_lr > 0.0:
                self.ff_biases[block_idx].add_(delta, alpha=float(self.cfg.hidden_bias_lr))
        if self.cfg.embedding_lr > 0.0:
            token = int(context[-1])
            self.input_codes[token] = torch_normalize_vector(
                self.input_codes[token] + float(self.cfg.embedding_lr) * code_error
            )
        self.step += 1
        if self.cfg.row_norm_interval > 0 and self.step % int(self.cfg.row_norm_interval) == 0:
            self.ff_weights = [torch_normalize_rows(weight) for weight in self.ff_weights]
        return float(loss.item()), pred

    def observe_chunk(self, contexts: list[np.ndarray], targets_np: np.ndarray) -> tuple[float, int, int]:
        h, traces_by_block, current_tokens = self.forward_batch(contexts, collect=True)
        targets = torch.as_tensor(targets_np, dtype=torch.long, device=self.device)
        logits = self.logits_from_features(h)
        probs = torch.softmax(logits / max(float(self.cfg.temperature), 1e-6), dim=1)
        target_probs = probs.gather(1, targets[:, None]).squeeze(1)
        losses = -torch.log(torch.clamp(target_probs, min=1e-12))
        preds = torch.argmax(probs, dim=1)
        correct = int((preds == targets).sum().item())

        error = -probs
        error[torch.arange(targets.numel(), device=self.device), targets] += 1.0
        self.output_weights.addmm_(error.T, h, beta=1.0, alpha=float(self.cfg.output_lr))
        self.update_count_bias_chunk(targets)

        expected_codes = probs @ self.target_codes
        code_errors = self.target_codes.index_select(0, targets) - expected_codes
        for block_idx in range(self.blocks):
            local_inputs = traces_by_block[block_idx]["input"]
            hidden = traces_by_block[block_idx]["hidden"]
            delta = code_errors * (1.0 - hidden.square())
            self.ff_weights[block_idx].addmm_(delta.T, local_inputs, beta=1.0, alpha=float(self.cfg.hidden_lr))
            if self.cfg.hidden_bias_lr > 0.0:
                self.ff_biases[block_idx].add_(delta.sum(dim=0), alpha=float(self.cfg.hidden_bias_lr))

        if self.cfg.embedding_lr > 0.0:
            for token, code_error in zip(current_tokens, code_errors):
                self.input_codes[token] = torch_normalize_vector(
                    self.input_codes[token] + float(self.cfg.embedding_lr) * code_error
                )
        self.step += int(targets.numel())
        if self.cfg.row_norm_interval > 0 and self.step // int(self.cfg.row_norm_interval) != (
            (self.step - int(targets.numel())) // int(self.cfg.row_norm_interval)
        ):
            self.ff_weights = [torch_normalize_rows(weight) for weight in self.ff_weights]
        return float(losses.sum().item()), correct, int(targets.numel())

    def predict_loss(self, context: np.ndarray, target: int) -> tuple[float, int]:
        feature, _ = self.forward_one(context, collect=False)
        logits = self.logits_from_feature(feature)
        probs = torch.softmax(logits / max(float(self.cfg.temperature), 1e-6), dim=0)
        target_t = int(target)
        loss = -torch.log(torch.clamp(probs[target_t], min=1e-12))
        return float(loss.item()), int(torch.argmax(probs).item())

    def generate(self, prompt: np.ndarray, max_new_tokens: int, sample: bool, temperature: float) -> list[int]:
        out = [int(x) for x in prompt.tolist()]
        for _ in range(max(int(max_new_tokens), 0)):
            context = np.asarray(out[-self.context_len :], dtype=np.int64)
            feature, _ = self.forward_one(context, collect=False)
            logits = self.logits_from_feature(feature)
            probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=0)
            if sample:
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_id = int(torch.argmax(probs).item())
            out.append(next_id)
        return out


def iter_doc_pairs(doc: np.ndarray, context_len: int) -> list[tuple[np.ndarray, int]]:
    pairs: list[tuple[np.ndarray, int]] = []
    for pos in range(1, int(doc.size)):
        left = max(0, pos - context_len)
        pairs.append((doc[left:pos], int(doc[pos])))
    return pairs


def iter_sample_batches(
    docs: list[np.ndarray],
    context_len: int,
    batch_size: int,
) -> list[tuple[int, list[np.ndarray], np.ndarray]]:
    doc_pairs = [iter_doc_pairs(doc, context_len) for doc in docs]
    positions = [0 for _ in doc_pairs]
    batches: list[tuple[int, list[np.ndarray], np.ndarray]] = []
    batch_size = max(int(batch_size), 1)
    while True:
        contexts: list[np.ndarray] = []
        targets: list[int] = []
        last_doc_idx = 0
        for doc_idx, pairs in enumerate(doc_pairs):
            if positions[doc_idx] >= len(pairs):
                continue
            context, target = pairs[positions[doc_idx]]
            positions[doc_idx] += 1
            contexts.append(context)
            targets.append(int(target))
            last_doc_idx = doc_idx
            if len(contexts) >= batch_size:
                break
        if not contexts:
            break
        batches.append((last_doc_idx, contexts, np.asarray(targets, dtype=np.int64)))
    return batches


@torch.no_grad()
def predict_chunk(
    model: TorchU009LocalModel,
    contexts: list[np.ndarray],
    targets_np: np.ndarray,
) -> tuple[float, int, int]:
    h, _, _ = model.forward_batch(contexts, collect=False)
    targets = torch.as_tensor(targets_np, dtype=torch.long, device=model.device)
    logits = model.logits_from_features(h)
    probs = torch.softmax(logits / max(float(model.cfg.temperature), 1e-6), dim=1)
    target_probs = probs.gather(1, targets[:, None]).squeeze(1)
    losses = -torch.log(torch.clamp(target_probs, min=1e-12))
    preds = torch.argmax(probs, dim=1)
    correct = int((preds == targets).sum().item())
    return float(losses.sum().item()), correct, int(targets.numel())


@torch.no_grad()
def run_documents(
    model: TorchU009LocalModel,
    docs: list[np.ndarray],
    update: bool,
    mode: str,
    chunk_size: int,
    chunk_tokens: int,
    eos_id: int | None = None,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    loss_sum = 0.0
    correct = 0
    total = 0
    chunk_loss = 0.0
    chunk_correct = 0
    chunk_total = 0
    rows: list[dict[str, float | int | str]] = []

    def record_chunk(doc_idx: int) -> None:
        nonlocal chunk_loss, chunk_correct, chunk_total
        if chunk_total <= 0:
            return
        rows.append(
            {
                "phase": "train" if update else "eval",
                "chunk": int(len(rows)),
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

    if mode == "packed":
        pack_eos = int(model.vocab_size - 1 if eos_id is None else eos_id)
        sequences = packed_lm_data.pack_documents(docs, model.context_len, pack_eos)
        seq_batch = max(int(chunk_size), 1)
        for seq_start in range(0, len(sequences), seq_batch):
            group = sequences[seq_start : seq_start + seq_batch]
            contexts, targets = packed_lm_data.flatten_sequence_batch(group, model.context_len)
            if len(contexts) == 0:
                continue
            if update:
                batch_loss, batch_correct, batch_total = model.observe_chunk(contexts, targets)
            else:
                batch_loss, batch_correct, batch_total = predict_chunk(model, contexts, targets)
            loss_sum += batch_loss
            correct += batch_correct
            total += batch_total
            chunk_loss += batch_loss
            chunk_correct += batch_correct
            chunk_total += batch_total
            if chunk_tokens > 0 and chunk_total >= chunk_tokens:
                record_chunk(seq_start)
        record_chunk(max(len(sequences) - 1, 0))
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        return summarize(loss_sum, correct, total), rows

    for doc_idx, doc in enumerate(docs):
        if doc.size < 2:
            continue
        if update and mode == "samplebatch":
            break
        pairs = iter_doc_pairs(doc, model.context_len)
        if update and mode == "chunk":
            for start in range(0, len(pairs), max(int(chunk_size), 1)):
                group = pairs[start : start + max(int(chunk_size), 1)]
                contexts = [item[0] for item in group]
                targets = np.asarray([item[1] for item in group], dtype=np.int64)
                batch_loss, batch_correct, batch_total = model.observe_chunk(contexts, targets)
                loss_sum += batch_loss
                correct += batch_correct
                total += batch_total
                chunk_loss += batch_loss
                chunk_correct += batch_correct
                chunk_total += batch_total
                if chunk_tokens > 0 and chunk_total >= chunk_tokens:
                    record_chunk(doc_idx)
        else:
            for context, target in pairs:
                if update:
                    loss, pred = model.observe_exact(context, target)
                else:
                    loss, pred = model.predict_loss(context, target)
                hit = int(pred == int(target))
                loss_sum += loss
                correct += hit
                total += 1
                chunk_loss += loss
                chunk_correct += hit
                chunk_total += 1
                if chunk_tokens > 0 and chunk_total >= chunk_tokens:
                    record_chunk(doc_idx)
    if update and mode == "samplebatch":
        for doc_idx, contexts, targets in iter_sample_batches(docs, model.context_len, chunk_size):
            batch_loss, batch_correct, batch_total = model.observe_chunk(contexts, targets)
            loss_sum += batch_loss
            correct += batch_correct
            total += batch_total
            chunk_loss += batch_loss
            chunk_correct += batch_correct
            chunk_total += batch_total
            if chunk_tokens > 0 and chunk_total >= chunk_tokens:
                record_chunk(doc_idx)
    record_chunk(max(len(docs) - 1, 0))
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    return summarize(loss_sum, correct, total), rows


def unigram_from_docs(docs: list[np.ndarray], vocab_size: int) -> np.ndarray:
    counts = np.ones(int(vocab_size), dtype=np.float32)
    for doc in docs:
        if doc.size > 1:
            np.add.at(counts, doc[1:].astype(np.int64, copy=False), 1.0)
    return counts / float(np.sum(counts))


def evaluate_unigram(docs: list[np.ndarray], probs: np.ndarray) -> dict[str, float | int]:
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
    return summarize(loss_sum, correct, total)


@torch.no_grad()
def generate_samples(
    model: TorchU009LocalModel,
    docs: list[np.ndarray],
    tokenizer: Any,
    sample_count: int,
    prompt_tokens: int,
    new_tokens: int,
    temperature: float,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    usable = [doc for doc in docs if int(doc.size) > max(int(prompt_tokens), 1)]
    for idx, doc in enumerate(usable[: max(int(sample_count), 0)]):
        prompt = doc[: max(int(prompt_tokens), 1)]
        greedy = model.generate(prompt, new_tokens, sample=False, temperature=temperature)
        sampled = model.generate(prompt, new_tokens, sample=True, temperature=temperature)
        rows.append(
            {
                "sample": int(idx),
                "prompt_tokens": int(prompt.size),
                "generated_tokens": int(new_tokens),
                "prompt": tokenizer.decode([int(x) for x in prompt.tolist()], skip_special_tokens=False),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tinystories", "gsm8k", "mix"], default="tinystories")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--gsm-train-file", type=Path, default=DEFAULT_GSM_TRAIN)
    parser.add_argument("--gsm-valid-file", type=Path, default=DEFAULT_GSM_VALID)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "u012_u009_torch_fast")
    parser.add_argument("--mode", choices=["exact", "chunk", "samplebatch", "packed"], default="exact")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train-chars", type=int, default=100_000)
    parser.add_argument("--valid-chars", type=int, default=12_000)
    parser.add_argument("--doc-chars", type=int, default=1_200)
    parser.add_argument("--gsm-train-items", type=int, default=0)
    parser.add_argument("--gsm-valid-items", type=int, default=0)
    parser.add_argument("--context-len", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--attn-rank", type=int, default=0)
    parser.add_argument("--output-lr", type=float, default=0.030)
    parser.add_argument("--hidden-lr", type=float, default=0.004)
    parser.add_argument("--hidden-bias-lr", type=float, default=0.0005)
    parser.add_argument("--embedding-lr", type=float, default=0.001)
    parser.add_argument("--logit-scale", type=float, default=2.0)
    parser.add_argument("--attention-scale", type=float, default=0.70)
    parser.add_argument("--ff-scale", type=float, default=0.35)
    parser.add_argument("--position-scale", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--row-norm-interval", type=int, default=128)
    parser.add_argument("--bias-alpha", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--chunk-tokens", type=int, default=1000)
    parser.add_argument("--pack-eos-id", type=int, default=-1)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--sample-prompt-tokens", type=int, default=20)
    parser.add_argument("--sample-new-tokens", type=int, default=64)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--skip-train-probes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    if int(args.pack_eos_id) < 0:
        args.pack_eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else len(tokenizer) - 1)
    train_text_docs, valid_text_docs = u009.load_text_documents(args)
    train_docs = u009.tokenize_documents(tokenizer, train_text_docs)
    valid_docs = u009.tokenize_documents(tokenizer, valid_text_docs)
    cfg = U012Config(
        context_len=args.context_len,
        d_model=args.d_model,
        blocks=args.blocks,
        attn_rank=args.attn_rank,
        output_lr=args.output_lr,
        hidden_lr=args.hidden_lr,
        hidden_bias_lr=args.hidden_bias_lr,
        embedding_lr=args.embedding_lr,
        logit_scale=args.logit_scale,
        attention_scale=args.attention_scale,
        ff_scale=args.ff_scale,
        position_scale=args.position_scale,
        temperature=args.temperature,
        row_norm_interval=args.row_norm_interval,
        bias_alpha=args.bias_alpha,
        seed=args.seed,
    )
    model = TorchU009LocalModel(int(len(tokenizer)), cfg, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.skip_train_probes:
        pre_train_probe = {"loss": float("nan"), "accuracy": float("nan"), "tokens": 0, "ppl": float("nan")}
    else:
        eval_mode = "packed" if args.mode == "packed" else "exact"
        pre_train_probe, _ = run_documents(
            model,
            train_docs,
            update=False,
            mode=eval_mode,
            chunk_size=args.chunk_size,
            chunk_tokens=0,
            eos_id=args.pack_eos_id,
        )
    eval_mode = "packed" if args.mode == "packed" else "exact"
    valid_pre, _ = run_documents(
        model,
        valid_docs,
        update=False,
        mode=eval_mode,
        chunk_size=args.chunk_size,
        chunk_tokens=0,
        eos_id=args.pack_eos_id,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    train_summary, train_chunks = run_documents(
        model,
        train_docs,
        update=True,
        mode=args.mode,
        chunk_size=args.chunk_size,
        chunk_tokens=args.chunk_tokens,
        eos_id=args.pack_eos_id,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    if args.skip_train_probes:
        post_train_probe = {"loss": float("nan"), "accuracy": float("nan"), "tokens": 0, "ppl": float("nan")}
    else:
        post_train_probe, _ = run_documents(
            model,
            train_docs,
            update=False,
            mode=eval_mode,
            chunk_size=args.chunk_size,
            chunk_tokens=0,
            eos_id=args.pack_eos_id,
        )
    valid_post, _ = run_documents(
        model,
        valid_docs,
        update=False,
        mode=eval_mode,
        chunk_size=args.chunk_size,
        chunk_tokens=0,
        eos_id=args.pack_eos_id,
    )
    valid_unigram = evaluate_unigram(valid_docs, unigram_from_docs(train_docs, int(len(tokenizer))))
    samples = generate_samples(
        model,
        valid_docs,
        tokenizer,
        args.sample_count,
        args.sample_prompt_tokens,
        args.sample_new_tokens,
        args.sample_temperature,
    )
    write_samples(args.out_dir / "samples.txt", samples)
    write_csv(args.out_dir / "train_chunks.csv", train_chunks)

    max_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    train_packed_sequences = packed_lm_data.pack_documents(train_docs, args.context_len, args.pack_eos_id)
    valid_packed_sequences = packed_lm_data.pack_documents(valid_docs, args.context_len, args.pack_eos_id)
    summary = {
        "task": args.task,
        "mode": args.mode,
        "seed": int(args.seed),
        "device": str(device),
        "tokenizer_len": int(len(tokenizer)),
        "train_docs": int(len(train_docs)),
        "valid_docs": int(len(valid_docs)),
        "train_pairs": int(count_document_pairs(train_docs)),
        "valid_pairs": int(count_document_pairs(valid_docs)),
        "packed_train_sequences": int(len(train_packed_sequences)),
        "packed_valid_sequences": int(len(valid_packed_sequences)),
        "packed_train_targets": int(packed_lm_data.count_sequence_targets(train_packed_sequences)),
        "packed_valid_targets": int(packed_lm_data.count_sequence_targets(valid_packed_sequences)),
        "context_len": int(args.context_len),
        "pack_eos_id": int(args.pack_eos_id),
        "d_model": int(args.d_model),
        "blocks": int(args.blocks),
        "attn_rank": int(args.attn_rank),
        "chunk_size": int(args.chunk_size),
        "parameters": int(model.parameter_count()),
        "state_bytes": int(model.state_bytes()),
        "pre_train_probe_loss": pre_train_probe["loss"],
        "valid_pre_loss": valid_pre["loss"],
        "train_loss": train_summary["loss"],
        "train_acc": train_summary["accuracy"],
        "train_tokens": train_summary["tokens"],
        "post_train_probe_loss": post_train_probe["loss"],
        "valid_post_loss": valid_post["loss"],
        "valid_post_acc": valid_post["accuracy"],
        "valid_unigram_loss": valid_unigram["loss"],
        "train_seconds": float(train_seconds),
        "tokens_per_second": float(train_summary["tokens"] / max(train_seconds, 1e-9)),
        "max_cuda_mem_bytes": int(max_mem),
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
        f"  mode={args.mode} task={args.task} tokenizer_len={len(tokenizer)} d={args.d_model} blocks={args.blocks} "
        f"params={model.parameter_count():,} state={model.state_bytes() / (1024 ** 2):.1f} MiB"
    )
    print(
        f"  train={train_summary['loss']:.4f}/{train_summary['accuracy']:.4f} "
        f"chunks={first_chunk:.4f}->{last_chunk:.4f} tokens={train_summary['tokens']} "
        f"speed={summary['tokens_per_second']:.2f} tok/s"
    )
    print(
        f"  valid={valid_pre['loss']:.4f}->{valid_post['loss']:.4f} "
        f"valid_unigram={valid_unigram['loss']:.4f} max_mem={max_mem / (1024 ** 3):.2f} GiB"
    )
    print(f"  samples: {args.out_dir / 'samples.txt'}")
    print(f"  summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
