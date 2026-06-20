#!/usr/bin/env python3
"""
Unified token-level bAbI QA evaluation for the no-BP phase learner.

This script is deliberately different from the archived bAbI answer selectors:
it does not build a task-specific QA head, parser, entity state, or answer
classifier.  Each bAbI example is serialized as ordinary text:

    Context:
    ...
    Question: ...
    Answer:

The model predicts the next token after the prompt with the same compact
vocabulary next-token distribution used for TinyStories.  Training, when
enabled, is online next-token learning over the same serialized examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from transformers import AutoTokenizer

import phase_binding_online_stream_experiment as stream
import phase_binding_token_experiment as phase


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "babi_qa_processed"
DEFAULT_TOKENIZER = phase.DEFAULT_TOKENIZER
DEFAULT_PRETRAIN_FILE = phase.DEFAULT_TRAIN
RANK_DIAGNOSTIC_KS = (2, 4, 8)


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


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def row_prompt(row: dict[str, Any]) -> str:
    context_lines = [str(item["text"]).strip() for item in row.get("context", [])]
    context = "\n".join(line for line in context_lines if line)
    question = str(row["question"]).strip()
    return f"Context:\n{context}\nQuestion: {question}\nAnswer:"


def row_answer_text(row: dict[str, Any]) -> str:
    return " " + str(row["answer"]).strip()


def row_train_text(row: dict[str, Any]) -> str:
    return row_prompt(row) + row_answer_text(row) + "\n\n"


def encode(tokenizer: Any, text: str) -> np.ndarray:
    return np.array(tokenizer.encode(text, add_special_tokens=False), dtype=np.int64)


def answer_token_ids(tokenizer: Any, answer: str) -> list[int]:
    ids = tokenizer.encode(" " + answer.strip(), add_special_tokens=False)
    if ids:
        return [int(x) for x in ids]
    return [int(x) for x in tokenizer.encode(answer.strip(), add_special_tokens=False)]


def read_prefix(path: Path, chars: int) -> str:
    if chars <= 0:
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read(chars)


def build_compact_vocab_with_forced(
    train_raw: np.ndarray,
    forced_raw: Sequence[int],
    max_vocab: int,
) -> tuple[np.ndarray, np.ndarray]:
    forced = {int(x) for x in forced_raw if int(x) >= 0}
    if train_raw.size:
        counts = np.bincount(train_raw)
        ranked = [int(x) for x in np.argsort(-counts)]
    else:
        counts = np.zeros(0, dtype=np.int64)
        ranked = []
    keep: list[int] = sorted(forced)
    remaining = max(int(max_vocab), len(keep)) - len(keep)
    for raw_id in ranked:
        if remaining <= 0:
            break
        if raw_id in forced:
            continue
        keep.append(raw_id)
        remaining -= 1
    kept_raw = np.array(sorted(set(keep)), dtype=np.int64)
    max_raw = int(kept_raw.max()) if kept_raw.size else 0
    if train_raw.size:
        max_raw = max(max_raw, int(train_raw.max()))
    raw_to_compact = np.full(max_raw + 1, -1, dtype=np.int64)
    raw_to_compact[kept_raw] = np.arange(len(kept_raw), dtype=np.int64)
    return kept_raw, raw_to_compact


def to_compact(raw_ids: Sequence[int] | np.ndarray, raw_to_compact: np.ndarray) -> list[int]:
    compact: list[int] = []
    for raw_id in raw_ids:
        raw = int(raw_id)
        if 0 <= raw < len(raw_to_compact):
            mapped = int(raw_to_compact[raw])
            if mapped >= 0:
                compact.append(mapped)
    return compact


def softmax_loss_and_pred(scores: np.ndarray, target: int, temperature: float) -> tuple[float, int, float]:
    probs = phase.softmax(scores, temperature)
    pred = int(np.argmax(probs))
    return -math.log(float(probs[int(target)]) + 1e-9), pred, float(probs[int(target)])


def candidate_rank_metrics(
    scores: np.ndarray,
    target: int,
    pred: int,
    ks: Sequence[int] = RANK_DIAGNOSTIC_KS,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float32)
    target_id = int(target)
    pred_id = int(pred)
    if target_id < 0 or target_id >= values.shape[0]:
        return {
            "target_rank": values.shape[0] + 1,
            "target_margin": float("nan"),
            "correct": 0,
            "error": 1,
            **{f"top{k}": 0 for k in ks},
            **{f"error_top{k}": 0 for k in ks},
        }
    target_score = float(values[target_id])
    target_rank = 1 + int(np.sum(values > target_score))
    wrong_scores = values.copy()
    wrong_scores[target_id] = -np.inf
    best_wrong = float(np.max(wrong_scores)) if wrong_scores.size > 1 else -np.inf
    target_margin = target_score - best_wrong if math.isfinite(best_wrong) else 0.0
    correct = int(pred_id == target_id)
    error = 1 - correct
    row: dict[str, Any] = {
        "target_rank": int(target_rank),
        "target_margin": float(target_margin),
        "correct": correct,
        "error": error,
    }
    for k in ks:
        top_hit = int(target_rank <= min(int(k), values.shape[0]))
        row[f"top{k}"] = top_hit
        row[f"error_top{k}"] = int(error and top_hit)
    return row


def summarize_rank_diagnostics(
    prefix: str,
    rank_sum: float,
    margin_sum: float,
    top_hits: dict[int, int],
    error_top_hits: dict[int, int],
    error_count: int,
    total: int,
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        f"{prefix}_target_rank_mean": rank_sum / max(total, 1),
        f"{prefix}_target_margin_mean": margin_sum / max(total, 1),
        f"{prefix}_error_count": int(error_count),
    }
    for k in sorted(top_hits):
        summary[f"{prefix}_top{k}_acc"] = top_hits[k] / max(total, 1)
        summary[f"{prefix}_error_top{k}_rate"] = error_top_hits[k] / max(error_count, 1)
        summary[f"{prefix}_oracle_top{k}_acc"] = (
            total - error_count + error_top_hits[k]
        ) / max(total, 1)
    return summary


def safe_state_bytes(memory: Any) -> int:
    return int(memory.state_bytes()) if hasattr(memory, "state_bytes") else 0


def safe_active_contexts(memory: Any) -> int:
    return int(memory.active_contexts()) if hasattr(memory, "active_contexts") else 0


def safe_event_cache_stats(memory: Any) -> dict[str, Any]:
    return memory.event_cache_stats() if hasattr(memory, "event_cache_stats") else {}


def safe_role_score_gate_stats(memory: Any) -> dict[str, Any]:
    return memory.role_score_gate_stats() if hasattr(memory, "role_score_gate_stats") else {}


def safe_role_branch_arbiter_stats(memory: Any) -> dict[str, Any]:
    return memory.role_branch_arbiter_stats() if hasattr(memory, "role_branch_arbiter_stats") else {}


def safe_role_joint_suppress_stats(memory: Any) -> dict[str, Any]:
    return memory.role_joint_suppress_stats() if hasattr(memory, "role_joint_suppress_stats") else {}


def safe_edge_path_cleanup_stats(memory: Any) -> dict[str, Any]:
    return memory.edge_path_cleanup_stats() if hasattr(memory, "edge_path_cleanup_stats") else {}


def safe_edge_path_direct_stats(memory: Any) -> dict[str, Any]:
    return memory.edge_path_direct_stats() if hasattr(memory, "edge_path_direct_stats") else {}


def safe_answer_slot_stats(memory: Any) -> dict[str, Any]:
    return memory.answer_slot_stats() if hasattr(memory, "answer_slot_stats") else {}


def reset_dynamic(memory: Any) -> None:
    if hasattr(memory, "reset_dynamic_state"):
        memory.reset_dynamic_state()


def observe_prompt(memory: Any, prompt_ids: Sequence[int]) -> None:
    if hasattr(memory, "observe_prompt"):
        for token in prompt_ids:
            memory.observe_prompt(int(token))


class OnlineStateMicroPrototypeMemory:
    """
    Local recurrent-state memory with a full-vocabulary micro-prototype readout.

    The state is a fixed random token/position recurrent trace over the current
    context.  Each output token owns a bounded set of prototype slots.  Updates
    touch only the target slot and, optionally, the current wrong winner slot.
    There is no parser, BP, raw replay, or task-specific QA head.
    """

    def __init__(
        self,
        vocab_size: int,
        state_dim: int,
        state_order: int,
        state_decay: float,
        slots: int,
        lr: float,
        wrong_lr: float,
        score_scale: float,
        bias_weight: float,
        margin: float,
        binding_hops: int,
        binding_window: int,
        binding_query_order: int,
        binding_query_mode: str,
        binding_focus_k: int,
        binding_decay: float,
        binding_bidirectional: bool,
        binding_mode: str,
        binding_span_window: int,
        binding_span_top_k: int,
        binding_span_decay: float,
        binding_span_learned_gate: bool,
        binding_span_gate_lr: float,
        binding_span_gate_neg_lr: float,
        binding_span_gate_strength: float,
        binding_span_gate_clip: float,
        latent_transition_branch: bool,
        transition_window: int,
        transition_passes: int,
        transition_decay: float,
        transition_threshold: float,
        transition_strength: float,
        event_cell_branch: bool,
        event_cell_count: int,
        event_cell_window: int,
        event_cell_top_k: int,
        event_cell_lr: float,
        event_cell_credit_lr: float,
        event_cell_neg_lr: float,
        event_cell_query_weight: float,
        event_cell_recency_decay: float,
        seed: int,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.state_dim = max(int(state_dim), 1)
        self.max_order = max(int(state_order), 1)
        self.state_decay = float(np.clip(state_decay, 0.0, 0.999))
        self.slots = max(int(slots), 1)
        self.lr = float(np.clip(lr, 0.0, 1.0))
        self.wrong_lr = float(np.clip(wrong_lr, 0.0, 1.0))
        self.score_scale = float(score_scale)
        self.bias_weight = float(bias_weight)
        self.margin = float(margin)
        self.binding_hops = max(int(binding_hops), 0)
        self.binding_window = max(int(binding_window), 1)
        self.binding_query_order = max(int(binding_query_order), 1)
        self.binding_query_mode = str(binding_query_mode)
        if self.binding_query_mode not in {"recent_trace", "prefix_overlap"}:
            raise ValueError(f"unknown binding query mode: {self.binding_query_mode}")
        self.binding_focus_k = max(int(binding_focus_k), 1)
        self.binding_decay = float(np.clip(binding_decay, 0.0, 1.0))
        self.binding_bidirectional = bool(binding_bidirectional)
        self.binding_mode = str(binding_mode)
        if self.binding_mode not in {"pair_apply", "span_sparse"}:
            raise ValueError(f"unknown binding mode: {self.binding_mode}")
        self.binding_span_window = max(int(binding_span_window), 1)
        self.binding_span_top_k = max(int(binding_span_top_k), 1)
        self.binding_span_decay = float(np.clip(binding_span_decay, 0.0, 1.0))
        self.binding_span_learned_gate = bool(binding_span_learned_gate)
        self.binding_span_gate_lr = float(max(binding_span_gate_lr, 0.0))
        self.binding_span_gate_neg_lr = float(max(binding_span_gate_neg_lr, 0.0))
        self.binding_span_gate_strength = float(max(binding_span_gate_strength, 0.0))
        self.binding_span_gate_clip = float(max(binding_span_gate_clip, 0.0))
        self.latent_transition_branch = bool(latent_transition_branch)
        self.transition_window = max(int(transition_window), 1)
        self.transition_passes = max(int(transition_passes), 1)
        self.transition_decay = float(np.clip(transition_decay, 0.0, 1.0))
        self.transition_threshold = float(transition_threshold)
        self.transition_strength = float(np.clip(transition_strength, 0.0, 1.0))
        self.event_cell_branch = bool(event_cell_branch)
        self.event_cell_count = max(int(event_cell_count), 1)
        self.event_cell_window = max(int(event_cell_window), 1)
        self.event_cell_top_k = max(int(event_cell_top_k), 1)
        self.event_cell_lr = float(np.clip(event_cell_lr, 0.0, 1.0))
        self.event_cell_credit_lr = float(max(event_cell_credit_lr, 0.0))
        self.event_cell_neg_lr = float(max(event_cell_neg_lr, 0.0))
        self.event_cell_query_weight = float(max(event_cell_query_weight, 0.0))
        self.event_cell_recency_decay = float(np.clip(event_cell_recency_decay, 0.0, 1.0))
        extra_branches = (
            self.binding_hops
            + int(self.latent_transition_branch)
            + int(self.event_cell_branch)
        )
        self.feature_dim = self.state_dim * (1 + extra_branches)
        rng = np.random.default_rng(seed)
        self.token_codes = phase.normalize_rows(
            rng.normal(0.0, 1.0, (self.vocab_size, self.state_dim)).astype(np.float32)
        )
        self.position_codes = phase.normalize_rows(
            rng.normal(0.0, 1.0, (self.max_order, self.state_dim)).astype(np.float32)
        )
        if self.binding_span_learned_gate:
            self.span_distance_codes = phase.normalize_rows(
                rng.normal(0.0, 1.0, (self.binding_span_window, self.state_dim)).astype(np.float32)
            )
            self.span_gate_weights = np.zeros(self.state_dim, dtype=np.float32)
        else:
            self.span_distance_codes = np.zeros((0, self.state_dim), dtype=np.float32)
            self.span_gate_weights = np.zeros(0, dtype=np.float32)
        if self.event_cell_branch:
            self.event_cell_keys = phase.normalize_rows(
                rng.normal(0.0, 1.0, (self.event_cell_count, self.state_dim)).astype(np.float32)
            )
            self.event_position_codes = phase.normalize_rows(
                rng.normal(0.0, 1.0, (2 * self.event_cell_window + 1, self.state_dim)).astype(np.float32)
            )
            self.event_cell_values = np.zeros((self.event_cell_count, self.state_dim), dtype=np.float32)
            self.event_cell_counts = np.zeros(self.event_cell_count, dtype=np.float32)
        else:
            self.event_cell_keys = np.zeros((0, self.state_dim), dtype=np.float32)
            self.event_position_codes = np.zeros((0, self.state_dim), dtype=np.float32)
            self.event_cell_values = np.zeros((0, self.state_dim), dtype=np.float32)
            self.event_cell_counts = np.zeros(0, dtype=np.float32)
        self.prototypes = np.zeros((self.vocab_size, self.slots, self.feature_dim), dtype=np.float32)
        self.prototype_counts = np.zeros((self.vocab_size, self.slots), dtype=np.float32)
        self.unigram_counts = np.ones(self.vocab_size, dtype=np.float32)
        self.output_bias = np.full(self.vocab_size, -math.log(max(self.vocab_size, 1)), dtype=np.float32)

    def recurrent_state(self, tokens: Sequence[int]) -> np.ndarray:
        if not tokens:
            return np.zeros(self.state_dim, dtype=np.float32)
        state = np.zeros(self.state_dim, dtype=np.float32)
        pos_start = self.max_order - len(tokens)
        for offset, token in enumerate(tokens):
            pos = pos_start + offset
            bound = phase.normalize_vector(self.token_codes[token] * self.position_codes[pos])
            state = self.state_decay * state + bound
        return phase.normalize_vector(state)

    def token_trace(self, tokens: Sequence[int]) -> np.ndarray:
        if not tokens:
            return np.zeros(self.state_dim, dtype=np.float32)
        state = np.zeros(self.state_dim, dtype=np.float32)
        for token in tokens:
            state = self.state_decay * state + self.token_codes[int(token)]
        return phase.normalize_vector(state)

    def binding_query_state(self, tokens: Sequence[int]) -> np.ndarray:
        recent = [int(token) for token in tokens[-self.binding_query_order :]]
        if self.binding_query_mode == "recent_trace" or not recent:
            return self.token_trace(recent)
        prefix = [int(token) for token in tokens[: max(len(tokens) - len(recent), 0)]]
        counts: dict[int, int] = {}
        for token in prefix:
            counts[token] = counts.get(token, 0) + 1
        candidates: list[tuple[int, int, int]] = []
        for pos, token in enumerate(recent):
            count = counts.get(token, 0)
            if count > 0:
                candidates.append((count, -pos, token))
        if not candidates:
            return self.token_trace(recent)
        candidates.sort()
        state = np.zeros(self.state_dim, dtype=np.float32)
        for count, _, token in candidates[: self.binding_focus_k]:
            state += self.token_codes[token] / math.sqrt(float(count))
        return phase.normalize_vector(state)

    def binding_query_tokens(self, tokens: Sequence[int]) -> list[int]:
        recent = [int(token) for token in tokens[-self.binding_query_order :]]
        if not recent:
            return []
        if self.binding_query_mode == "recent_trace":
            seen: set[int] = set()
            chosen: list[int] = []
            for token in reversed(recent):
                if token in seen:
                    continue
                seen.add(token)
                chosen.append(token)
                if len(chosen) >= self.binding_focus_k:
                    break
            return list(reversed(chosen))
        prefix = [int(token) for token in tokens[: max(len(tokens) - len(recent), 0)]]
        counts: dict[int, int] = {}
        for token in prefix:
            counts[token] = counts.get(token, 0) + 1
        candidates: list[tuple[int, int, int]] = []
        for pos, token in enumerate(recent):
            count = counts.get(token, 0)
            if count > 0:
                candidates.append((count, -pos, token))
        if not candidates:
            return self.binding_query_tokens_recent_fallback(recent)
        candidates.sort()
        return [int(token) for _, _, token in candidates[: self.binding_focus_k]]

    def binding_query_tokens_recent_fallback(self, recent: Sequence[int]) -> list[int]:
        seen: set[int] = set()
        chosen: list[int] = []
        for token in reversed([int(x) for x in recent]):
            if token in seen:
                continue
            seen.add(token)
            chosen.append(token)
            if len(chosen) >= self.binding_focus_k:
                break
        return list(reversed(chosen))

    def apply_binding(self, tokens: Sequence[int], state: np.ndarray) -> np.ndarray:
        out = np.zeros(self.state_dim, dtype=np.float32)
        writes = 0
        total = len(tokens)
        for right, token_right in enumerate(tokens):
            start = max(0, right - self.binding_window)
            if start >= right:
                continue
            recency = self.binding_decay ** max(total - 1 - right, 0)
            value = self.token_codes[int(token_right)]
            for left in range(start, right):
                key = self.token_codes[int(tokens[left])]
                out += recency * value * float(key @ state)
                writes += 1
                if self.binding_bidirectional:
                    out += recency * key * float(value @ state)
                    writes += 1
        if writes > 0:
            out /= math.sqrt(float(writes))
        return phase.normalize_vector(out)

    def span_sparse_events(
        self,
        tokens: Sequence[int],
        seeds: Sequence[int],
    ) -> list[tuple[int, int, float, int]]:
        query_len = min(self.binding_query_order, len(tokens))
        prefix_end = max(len(tokens) - query_len, 0)
        prefix = [int(token) for token in tokens[:prefix_end]]
        if not prefix:
            prefix = [int(token) for token in tokens]
            prefix_end = len(prefix)
        seed_set = {int(token) for token in seeds}
        events: list[tuple[int, int, float, int]] = []
        if not seed_set:
            return events
        for pos, seed_token in enumerate(prefix):
            if seed_token not in seed_set:
                continue
            left = max(0, pos - self.binding_span_window)
            right = min(prefix_end, pos + self.binding_span_window + 1)
            recency = self.binding_decay ** max(prefix_end - 1 - pos, 0)
            for neighbor_pos in range(left, right):
                if neighbor_pos == pos:
                    continue
                distance = abs(neighbor_pos - pos)
                locality = self.binding_span_decay ** max(distance - 1, 0)
                weight = float(recency * locality / math.sqrt(float(distance)))
                events.append((int(seed_token), int(prefix[neighbor_pos]), weight, int(distance)))
        return events

    def span_event_feature(self, seed_token: int, neighbor_token: int, distance: int) -> np.ndarray:
        if not self.binding_span_learned_gate:
            return np.zeros(self.state_dim, dtype=np.float32)
        distance_idx = min(max(int(distance) - 1, 0), self.binding_span_window - 1)
        event = (
            self.token_codes[int(seed_token)]
            * self.token_codes[int(neighbor_token)]
            * self.span_distance_codes[distance_idx]
        )
        return phase.normalize_vector(event)

    def span_event_gate(self, seed_token: int, neighbor_token: int, distance: int) -> float:
        if (
            not self.binding_span_learned_gate
            or self.binding_span_gate_strength <= 0.0
            or self.span_gate_weights.size == 0
        ):
            return 1.0
        event = self.span_event_feature(seed_token, neighbor_token, distance)
        raw = float(self.span_gate_weights @ event)
        gate = 1.0 + self.binding_span_gate_strength * math.tanh(raw)
        if self.binding_span_gate_clip > 0.0:
            gate = float(np.clip(gate, 0.0, self.binding_span_gate_clip))
        return gate

    def span_sparse_binding_apply(self, tokens: Sequence[int], seeds: Sequence[int]) -> np.ndarray:
        if not seeds:
            return np.zeros(self.state_dim, dtype=np.float32)
        out = np.zeros(self.state_dim, dtype=np.float32)
        weight_sq_sum = 0.0
        for seed_token, neighbor_token, base_weight, distance in self.span_sparse_events(tokens, seeds):
            gate = self.span_event_gate(seed_token, neighbor_token, distance)
            weight = float(base_weight * gate)
            out += weight * self.token_codes[int(neighbor_token)]
            weight_sq_sum += weight * weight
        if weight_sq_sum <= 0.0:
            return self.token_trace(seeds)
        out /= math.sqrt(weight_sq_sum)
        return phase.normalize_vector(out)

    def update_span_gate(self, tokens: Sequence[int], target: int, wrong: int, apply_update: bool) -> None:
        if (
            not apply_update
            or not self.binding_span_learned_gate
            or self.binding_mode != "span_sparse"
            or self.binding_hops <= 0
        ):
            return
        target = int(target)
        wrong = int(wrong)
        seeds = self.binding_query_tokens(tokens)
        if not seeds:
            return
        delta = np.zeros(self.state_dim, dtype=np.float32)
        for _ in range(self.binding_hops):
            events = self.span_sparse_events(tokens, seeds)
            if not events:
                break
            for seed_token, neighbor_token, weight, distance in events:
                if neighbor_token == target and self.binding_span_gate_lr > 0.0:
                    delta += self.binding_span_gate_lr * weight * self.span_event_feature(
                        seed_token, neighbor_token, distance
                    )
                elif neighbor_token == wrong and self.binding_span_gate_neg_lr > 0.0:
                    delta -= self.binding_span_gate_neg_lr * weight * self.span_event_feature(
                        seed_token, neighbor_token, distance
                    )
            state = self.span_sparse_binding_apply(tokens, seeds)
            seeds = self.span_sparse_top_tokens(tokens, state)
            if not seeds:
                break
        if np.any(delta):
            self.span_gate_weights = phase.normalize_vector(self.span_gate_weights + delta).astype(np.float32)

    def span_sparse_top_tokens(self, tokens: Sequence[int], state: np.ndarray) -> list[int]:
        query_len = min(self.binding_query_order, len(tokens))
        prefix_end = max(len(tokens) - query_len, 0)
        prefix = [int(token) for token in tokens[:prefix_end]]
        if not prefix:
            prefix = [int(token) for token in tokens]
        counts: dict[int, int] = {}
        for token in prefix:
            counts[token] = counts.get(token, 0) + 1
        scored: list[tuple[float, int, int]] = []
        for token, count in counts.items():
            score = float(self.token_codes[int(token)] @ state)
            scored.append((-score, count, int(token)))
        scored.sort()
        return [token for _, _, token in scored[: self.binding_span_top_k]]

    def span_sparse_binding_features(self, tokens: Sequence[int]) -> list[np.ndarray]:
        seeds = self.binding_query_tokens(tokens)
        if not seeds:
            return []
        hops: list[np.ndarray] = []
        for _ in range(self.binding_hops):
            state = self.span_sparse_binding_apply(tokens, seeds)
            hops.append(state)
            seeds = self.span_sparse_top_tokens(tokens, state)
            if not seeds:
                break
        return hops

    def binding_features(self, tokens: Sequence[int]) -> list[np.ndarray]:
        if self.binding_hops <= 0 or not tokens:
            return []
        if self.binding_mode == "span_sparse":
            hops = self.span_sparse_binding_features(tokens)
            while len(hops) < self.binding_hops:
                hops.append(np.zeros(self.state_dim, dtype=np.float32))
            return hops
        state = self.binding_query_state(tokens)
        hops: list[np.ndarray] = []
        for _ in range(self.binding_hops):
            state = self.apply_binding(tokens, state)
            hops.append(state)
        return hops

    def latent_transition_state(self, tokens: Sequence[int]) -> np.ndarray:
        if not self.latent_transition_branch or not tokens:
            return np.zeros(self.state_dim, dtype=np.float32)
        query_len = min(self.binding_query_order, len(tokens))
        prefix_end = max(len(tokens) - query_len, 0)
        prefix = [int(token) for token in tokens[:prefix_end]]
        if not prefix:
            prefix = [int(token) for token in tokens]
            prefix_end = len(prefix)
        state = self.binding_query_state(tokens)
        if not np.any(state):
            state = self.token_trace(tokens[-query_len:])
        for _ in range(self.transition_passes):
            out = np.zeros(self.state_dim, dtype=np.float32)
            weight_sq_sum = 0.0
            for pos, token in enumerate(prefix):
                match = float(self.token_codes[int(token)] @ state)
                activation = max(match - self.transition_threshold, 0.0)
                if activation <= 0.0:
                    continue
                left = max(0, pos - self.transition_window)
                right = min(prefix_end, pos + self.transition_window + 1)
                recency = self.transition_decay ** max(prefix_end - 1 - pos, 0)
                for neighbor_pos in range(left, right):
                    if neighbor_pos == pos:
                        continue
                    distance = abs(neighbor_pos - pos)
                    weight = float(recency * activation / math.sqrt(float(distance)))
                    out += weight * self.token_codes[int(prefix[neighbor_pos])]
                    weight_sq_sum += weight * weight
            if weight_sq_sum <= 0.0:
                break
            out = phase.normalize_vector(out / math.sqrt(weight_sq_sum))
            state = phase.normalize_vector(
                (1.0 - self.transition_strength) * state + self.transition_strength * out
            )
        return state.astype(np.float32)

    def event_cell_prefix(self, tokens: Sequence[int]) -> list[int]:
        query_len = min(self.binding_query_order, len(tokens))
        prefix_end = max(len(tokens) - query_len, 0)
        prefix = [int(token) for token in tokens[:prefix_end]]
        if not prefix:
            prefix = [int(token) for token in tokens]
        return prefix

    def event_cell_local_feature(self, prefix: Sequence[int], pos: int) -> tuple[np.ndarray, list[int]]:
        out = np.zeros(self.state_dim, dtype=np.float32)
        weight_sq_sum = 0.0
        neighbors: list[int] = []
        left = max(0, int(pos) - self.event_cell_window)
        right = min(len(prefix), int(pos) + self.event_cell_window + 1)
        for neighbor_pos in range(left, right):
            rel = neighbor_pos - int(pos)
            code_idx = rel + self.event_cell_window
            distance = abs(rel)
            weight = 1.0 / math.sqrt(float(distance + 1))
            token = int(prefix[neighbor_pos])
            out += weight * phase.normalize_vector(
                self.token_codes[token] * self.event_position_codes[code_idx]
            )
            weight_sq_sum += weight * weight
            neighbors.append(token)
        if weight_sq_sum <= 0.0:
            return np.zeros(self.state_dim, dtype=np.float32), neighbors
        out /= math.sqrt(weight_sq_sum)
        return phase.normalize_vector(out), neighbors

    def event_cell_candidates(
        self,
        tokens: Sequence[int],
    ) -> list[tuple[float, int, int, np.ndarray, list[int]]]:
        if not self.event_cell_branch or not tokens:
            return []
        prefix = self.event_cell_prefix(tokens)
        if not prefix:
            return []
        query_state = self.binding_query_state(tokens)
        candidates: list[tuple[float, int, int, np.ndarray, list[int]]] = []
        total = len(prefix)
        for pos in range(total):
            feature, neighbors = self.event_cell_local_feature(prefix, pos)
            if not np.any(feature):
                continue
            cell_scores = self.event_cell_keys @ feature
            cell = int(np.argmax(cell_scores))
            cell_score = float(cell_scores[cell])
            query_score = max(float(query_state @ feature), 0.0)
            recency = self.event_cell_recency_decay ** max(total - 1 - pos, 0)
            score = recency * (max(cell_score, 0.0) + self.event_cell_query_weight * query_score)
            candidates.append((float(score), int(pos), cell, feature, neighbors))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return candidates[: self.event_cell_top_k]

    def event_cell_state(self, tokens: Sequence[int]) -> np.ndarray:
        if not self.event_cell_branch:
            return np.zeros(self.state_dim, dtype=np.float32)
        out = np.zeros(self.state_dim, dtype=np.float32)
        weight_sq_sum = 0.0
        for score, _, cell, feature, _ in self.event_cell_candidates(tokens):
            if self.event_cell_counts[cell] > 0.0:
                value = self.event_cell_values[cell]
            else:
                value = feature
            weight = max(float(score), 1e-3)
            out += weight * value
            weight_sq_sum += weight * weight
        if weight_sq_sum <= 0.0:
            return np.zeros(self.state_dim, dtype=np.float32)
        out /= math.sqrt(weight_sq_sum)
        return phase.normalize_vector(out)

    def update_event_cells(self, tokens: Sequence[int], target: int, wrong: int, apply_credit: bool) -> None:
        if not self.event_cell_branch:
            return
        target = int(target)
        wrong = int(wrong)
        for _, _, cell, feature, neighbors in self.event_cell_candidates(tokens):
            if self.event_cell_counts[cell] > 0.0:
                base = self.event_cell_values[cell]
            else:
                base = feature
            delta = self.event_cell_lr * feature
            if apply_credit and target in neighbors and self.event_cell_credit_lr > 0.0:
                delta = delta + self.event_cell_credit_lr * self.token_codes[target]
            if apply_credit and wrong in neighbors and self.event_cell_neg_lr > 0.0:
                delta = delta - self.event_cell_neg_lr * self.token_codes[wrong]
            self.event_cell_values[cell] = phase.normalize_vector(
                (1.0 - self.event_cell_lr) * base + delta
            ).astype(np.float32)
            self.event_cell_counts[cell] += 1.0

    def feature(self, context: Sequence[int] | np.ndarray) -> np.ndarray:
        tokens = [int(token) for token in list(context)[-self.max_order :]]
        pieces = [self.recurrent_state(tokens)]
        pieces.extend(self.binding_features(tokens))
        if self.latent_transition_branch:
            pieces.append(self.latent_transition_state(tokens))
        if self.event_cell_branch:
            pieces.append(self.event_cell_state(tokens))
        if len(pieces) == 1:
            return pieces[0]
        return phase.normalize_vector(np.concatenate(pieces).astype(np.float32))

    def scores_from_feature(self, feature: np.ndarray) -> np.ndarray:
        scores = (self.bias_weight * self.output_bias).astype(np.float32, copy=True)
        active = np.flatnonzero(np.any(self.prototype_counts > 0.0, axis=1))
        if active.size == 0:
            return scores
        proto = self.prototypes[active]
        dots = np.einsum("asd,d->as", proto, feature, optimize=True).astype(np.float32)
        dots = np.where(self.prototype_counts[active] > 0.0, dots, -np.inf)
        scores[active] += self.score_scale * np.max(dots, axis=1)
        return scores.astype(np.float32)

    def scores(self, context: Sequence[int] | np.ndarray) -> np.ndarray:
        return self.scores_from_feature(self.feature(context))

    def update_target_slot(self, target: int, feature: np.ndarray) -> None:
        target = int(target)
        counts = self.prototype_counts[target]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            slot = int(empty[0])
            self.prototypes[target, slot] = feature
            self.prototype_counts[target, slot] = 1.0
            return
        dots = self.prototypes[target] @ feature
        slot = int(np.argmax(dots))
        self.prototypes[target, slot] = phase.normalize_vector(
            (1.0 - self.lr) * self.prototypes[target, slot] + self.lr * feature
        )
        self.prototype_counts[target, slot] += 1.0

    def update_wrong_slot(self, wrong: int, feature: np.ndarray) -> None:
        wrong = int(wrong)
        active = self.prototype_counts[wrong] > 0.0
        if not np.any(active):
            return
        dots = self.prototypes[wrong] @ feature
        dots = np.where(active, dots, -np.inf)
        slot = int(np.argmax(dots))
        self.prototypes[wrong, slot] = phase.normalize_vector(
            self.prototypes[wrong, slot] - self.wrong_lr * feature
        )

    def update(self, context: Sequence[int] | np.ndarray, target: int) -> None:
        target = int(target)
        tokens = [int(token) for token in list(context)[-self.max_order :]]
        feature = self.feature(context)
        scores = self.scores_from_feature(feature)
        target_score = float(scores[target])
        adjusted = scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        wrong = int(np.argmax(adjusted))
        should_apply_credit = float(adjusted[wrong]) + self.margin > target_score
        self.update_span_gate(tokens, target, wrong, should_apply_credit)
        self.update_event_cells(tokens, target, wrong, should_apply_credit)
        self.update_target_slot(target, feature)
        if self.wrong_lr > 0.0 and should_apply_credit:
            self.update_wrong_slot(wrong, feature)
        self.unigram_counts[target] += 1.0
        probs = self.unigram_counts / float(np.sum(self.unigram_counts))
        self.output_bias = np.log(np.maximum(probs, 1e-9)).astype(np.float32)

    def state_bytes(self) -> int:
        return int(
            self.token_codes.nbytes
            + self.position_codes.nbytes
            + self.span_distance_codes.nbytes
            + self.span_gate_weights.nbytes
            + self.event_cell_keys.nbytes
            + self.event_position_codes.nbytes
            + self.event_cell_values.nbytes
            + self.event_cell_counts.nbytes
            + self.prototypes.nbytes
            + self.prototype_counts.nbytes
            + self.unigram_counts.nbytes
            + self.output_bias.nbytes
        )

    def active_contexts(self) -> int:
        return int(np.count_nonzero(self.prototype_counts))


class OnlineQueryEventCleanupMemory:
    """
    Query-seeded local event assembly plus event-only cleanup readout.

    This mirrors the R158 synthetic object-carry mechanism, but operates only on
    compact token IDs from the unified bAbI prompt.  It has no bAbI parser,
    symbolic state, answer head, BP, or raw replay.
    """

    def __init__(
        self,
        vocab_size: int,
        state_dim: int,
        state_order: int,
        state_decay: float,
        slots: int,
        lr: float,
        wrong_lr: float,
        score_scale: float,
        bias_weight: float,
        margin: float,
        query_order: int,
        assembly_hops: int,
        assembly_event_window: int,
        assembly_seed_top_k: int,
        assembly_recency_decay: float,
        assembly_locality_decay: float,
        cleanup_slots: int,
        cleanup_lr: float,
        cleanup_wrong_lr: float,
        cleanup_score_scale: float,
        cleanup_top_k: int,
        cleanup_inhibit: float,
        seed: int,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.state_dim = max(int(state_dim), 1)
        self.max_order = max(int(state_order), 1)
        self.state_decay = float(np.clip(state_decay, 0.0, 0.999))
        self.slots = max(int(slots), 1)
        self.lr = float(np.clip(lr, 0.0, 1.0))
        self.wrong_lr = float(np.clip(wrong_lr, 0.0, 1.0))
        self.score_scale = float(score_scale)
        self.bias_weight = float(bias_weight)
        self.margin = float(margin)
        self.query_order = max(int(query_order), 1)
        self.hops = max(int(assembly_hops), 1)
        self.event_window = max(int(assembly_event_window), 1)
        self.seed_top_k = max(int(assembly_seed_top_k), 1)
        self.recency_decay = float(np.clip(assembly_recency_decay, 0.0, 1.0))
        self.locality_decay = float(np.clip(assembly_locality_decay, 0.0, 1.0))
        self.feature_dim = self.state_dim * (1 + self.hops)
        self.cleanup_dim = self.state_dim * self.hops
        self.cleanup_slots = int(cleanup_slots) if int(cleanup_slots) > 0 else self.slots
        self.cleanup_lr = float(np.clip(cleanup_lr, 0.0, 1.0))
        self.cleanup_wrong_lr = float(np.clip(cleanup_wrong_lr, 0.0, 1.0))
        self.cleanup_score_scale = float(cleanup_score_scale)
        self.cleanup_top_k = max(int(cleanup_top_k), 0)
        self.cleanup_inhibit = float(max(cleanup_inhibit, 0.0))
        rng = np.random.default_rng(seed + 7919)
        self.token_codes = phase.normalize_rows(
            rng.normal(0.0, 1.0, (self.vocab_size, self.state_dim)).astype(np.float32)
        )
        self.position_codes = phase.normalize_rows(
            rng.normal(0.0, 1.0, (self.max_order, self.state_dim)).astype(np.float32)
        )
        self.relative_codes = phase.normalize_rows(
            rng.normal(0.0, 1.0, (2 * self.event_window + 1, self.state_dim)).astype(np.float32)
        )
        self.prototypes = np.zeros((self.vocab_size, self.slots, self.feature_dim), dtype=np.float32)
        self.prototype_counts = np.zeros((self.vocab_size, self.slots), dtype=np.float32)
        self.cleanup_prototypes = np.zeros(
            (self.vocab_size, self.cleanup_slots, self.cleanup_dim),
            dtype=np.float32,
        )
        self.cleanup_counts = np.zeros((self.vocab_size, self.cleanup_slots), dtype=np.float32)
        self.unigram_counts = np.ones(self.vocab_size, dtype=np.float32)
        self.output_bias = np.full(self.vocab_size, -math.log(max(self.vocab_size, 1)), dtype=np.float32)
        self.last_event_feature = np.zeros(self.cleanup_dim, dtype=np.float32)

    def recurrent_state(self, tokens: Sequence[int]) -> np.ndarray:
        if not tokens:
            return np.zeros(self.state_dim, dtype=np.float32)
        state = np.zeros(self.state_dim, dtype=np.float32)
        pos_start = self.max_order - len(tokens)
        for offset, token in enumerate(tokens):
            pos = pos_start + offset
            bound = phase.normalize_vector(self.token_codes[int(token)] * self.position_codes[pos])
            state = self.state_decay * state + bound
        return phase.normalize_vector(state)

    def prefix_and_query(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        query_len = min(self.query_order, len(tokens))
        prefix_end = max(len(tokens) - query_len, 0)
        prefix = [int(token) for token in tokens[:prefix_end]]
        query = [int(token) for token in tokens[prefix_end:]]
        if not prefix:
            prefix = [int(token) for token in tokens]
        return prefix, query

    def initial_query_seeds(self, prefix: list[int], query: list[int]) -> list[int]:
        if not prefix or not query:
            return []
        prefix_counts: dict[int, int] = {}
        for token in prefix:
            token = int(token)
            prefix_counts[token] = prefix_counts.get(token, 0) + 1
        candidates: list[tuple[int, int, int]] = []
        seen: set[int] = set()
        for pos, token in enumerate(query):
            token = int(token)
            count = prefix_counts.get(token, 0)
            if count <= 0 or token in seen:
                continue
            seen.add(token)
            candidates.append((count, -pos, token))
        candidates.sort()
        return [token for _, _, token in candidates[: self.seed_top_k]]

    def local_event_state(self, prefix: list[int], seeds: list[int]) -> np.ndarray:
        if not prefix or not seeds:
            return np.zeros(self.state_dim, dtype=np.float32)
        seed_set = {int(token) for token in seeds}
        out = np.zeros(self.state_dim, dtype=np.float32)
        weight_sq_sum = 0.0
        total = len(prefix)
        for pos, token in enumerate(prefix):
            if int(token) not in seed_set:
                continue
            left = max(0, pos - self.event_window)
            right = min(total, pos + self.event_window + 1)
            recency = self.recency_decay ** max(total - 1 - pos, 0)
            for neighbor_pos in range(left, right):
                if neighbor_pos == pos:
                    continue
                rel = neighbor_pos - pos
                distance = abs(rel)
                rel_idx = rel + self.event_window
                locality = self.locality_decay ** max(distance - 1, 0)
                weight = float(recency * locality / math.sqrt(float(distance)))
                neighbor = int(prefix[neighbor_pos])
                event = phase.normalize_vector(self.token_codes[neighbor] * self.relative_codes[rel_idx])
                out += weight * event
                weight_sq_sum += weight * weight
        if weight_sq_sum <= 0.0:
            return np.zeros(self.state_dim, dtype=np.float32)
        return phase.normalize_vector(out / math.sqrt(weight_sq_sum))

    def select_seeds(self, prefix: list[int], state: np.ndarray, excluded: set[int]) -> list[int]:
        if not prefix or not np.any(state):
            return []
        counts: dict[int, int] = {}
        for token in prefix:
            token = int(token)
            counts[token] = counts.get(token, 0) + 1
        scored: list[tuple[float, int, int]] = []
        for token, count in counts.items():
            if token in excluded:
                continue
            score = float(self.token_codes[token] @ state)
            scored.append((-score, -count, token))
        scored.sort()
        return [token for _, _, token in scored[: self.seed_top_k]]

    def feature(self, context: Sequence[int] | np.ndarray) -> np.ndarray:
        tokens = [int(token) for token in list(context)[-self.max_order :]]
        prefix, query = self.prefix_and_query(tokens)
        seeds = self.initial_query_seeds(prefix, query)
        excluded = set(seeds)
        pieces = [self.recurrent_state(tokens)]
        for _ in range(self.hops):
            state = self.local_event_state(prefix, seeds)
            pieces.append(state)
            new_seeds = self.select_seeds(prefix, state, excluded)
            excluded.update(new_seeds)
            seeds = new_seeds
        self.last_event_feature = phase.normalize_vector(np.concatenate(pieces[1:]).astype(np.float32))
        return phase.normalize_vector(np.concatenate(pieces).astype(np.float32))

    def scores_from_feature(self, feature: np.ndarray) -> np.ndarray:
        scores = (self.bias_weight * self.output_bias).astype(np.float32, copy=True)
        active = np.flatnonzero(np.any(self.prototype_counts > 0.0, axis=1))
        if active.size == 0:
            return scores
        proto = self.prototypes[active]
        dots = np.einsum("asd,d->as", proto, feature, optimize=True).astype(np.float32)
        dots = np.where(self.prototype_counts[active] > 0.0, dots, -np.inf)
        scores[active] += self.score_scale * np.max(dots, axis=1)
        return scores.astype(np.float32)

    def cleanup_scores_from_event(self, event_feature: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        active = np.flatnonzero(np.any(self.cleanup_counts > 0.0, axis=1))
        if active.size == 0:
            return scores
        proto = self.cleanup_prototypes[active]
        dots = np.einsum("asd,d->as", proto, event_feature, optimize=True).astype(np.float32)
        dots = np.where(self.cleanup_counts[active] > 0.0, dots, -np.inf)
        scores[active] = self.cleanup_score_scale * np.max(dots, axis=1)
        if self.cleanup_top_k > 0 and self.cleanup_inhibit > 0.0 and active.size > self.cleanup_top_k:
            active_scores = scores[active]
            winner_local = np.argpartition(active_scores, -self.cleanup_top_k)[-self.cleanup_top_k :]
            winner_tokens = set(int(active[idx]) for idx in winner_local)
            for token in active:
                if int(token) not in winner_tokens:
                    scores[int(token)] -= self.cleanup_inhibit
        return scores

    def combined_scores(self, feature: np.ndarray, event_feature: np.ndarray) -> np.ndarray:
        return (self.scores_from_feature(feature) + self.cleanup_scores_from_event(event_feature)).astype(np.float32)

    def scores(self, context: Sequence[int] | np.ndarray) -> np.ndarray:
        feature = self.feature(context)
        return self.combined_scores(feature, self.last_event_feature)

    def update_target_slot(self, target: int, feature: np.ndarray) -> None:
        target = int(target)
        counts = self.prototype_counts[target]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            slot = int(empty[0])
            self.prototypes[target, slot] = feature
            self.prototype_counts[target, slot] = 1.0
            return
        dots = self.prototypes[target] @ feature
        slot = int(np.argmax(dots))
        self.prototypes[target, slot] = phase.normalize_vector(
            (1.0 - self.lr) * self.prototypes[target, slot] + self.lr * feature
        )
        self.prototype_counts[target, slot] += 1.0

    def update_wrong_slot(self, wrong: int, feature: np.ndarray) -> None:
        wrong = int(wrong)
        active = self.prototype_counts[wrong] > 0.0
        if not np.any(active):
            return
        dots = self.prototypes[wrong] @ feature
        dots = np.where(active, dots, -np.inf)
        slot = int(np.argmax(dots))
        self.prototypes[wrong, slot] = phase.normalize_vector(
            self.prototypes[wrong, slot] - self.wrong_lr * feature
        )

    def update_cleanup_target_slot(self, target: int, event_feature: np.ndarray) -> None:
        target = int(target)
        counts = self.cleanup_counts[target]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            slot = int(empty[0])
            self.cleanup_prototypes[target, slot] = event_feature
            self.cleanup_counts[target, slot] = 1.0
            return
        dots = self.cleanup_prototypes[target] @ event_feature
        slot = int(np.argmax(dots))
        self.cleanup_prototypes[target, slot] = phase.normalize_vector(
            (1.0 - self.cleanup_lr) * self.cleanup_prototypes[target, slot]
            + self.cleanup_lr * event_feature
        )
        self.cleanup_counts[target, slot] += 1.0

    def update_cleanup_wrong_slot(self, wrong: int, event_feature: np.ndarray) -> None:
        wrong = int(wrong)
        active = self.cleanup_counts[wrong] > 0.0
        if not np.any(active):
            return
        dots = self.cleanup_prototypes[wrong] @ event_feature
        dots = np.where(active, dots, -np.inf)
        slot = int(np.argmax(dots))
        self.cleanup_prototypes[wrong, slot] = phase.normalize_vector(
            self.cleanup_prototypes[wrong, slot] - self.cleanup_wrong_lr * event_feature
        )

    def update(self, context: Sequence[int] | np.ndarray, target: int) -> None:
        target = int(target)
        feature = self.feature(context)
        event_feature = self.last_event_feature
        scores = self.combined_scores(feature, event_feature)
        target_score = float(scores[target])
        adjusted = scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        wrong = int(np.argmax(adjusted))
        should_credit = float(adjusted[wrong]) + self.margin > target_score
        self.update_target_slot(target, feature)
        self.update_cleanup_target_slot(target, event_feature)
        if should_credit:
            if self.wrong_lr > 0.0:
                self.update_wrong_slot(wrong, feature)
            if self.cleanup_wrong_lr > 0.0:
                self.update_cleanup_wrong_slot(wrong, event_feature)
        self.unigram_counts[target] += 1.0
        probs = self.unigram_counts / float(np.sum(self.unigram_counts))
        self.output_bias = np.log(np.maximum(probs, 1e-9)).astype(np.float32)

    def state_bytes(self) -> int:
        return int(
            self.token_codes.nbytes
            + self.position_codes.nbytes
            + self.relative_codes.nbytes
            + self.prototypes.nbytes
            + self.prototype_counts.nbytes
            + self.cleanup_prototypes.nbytes
            + self.cleanup_counts.nbytes
            + self.unigram_counts.nbytes
            + self.output_bias.nbytes
        )

    def active_contexts(self) -> int:
        return int(np.count_nonzero(self.prototype_counts) + np.count_nonzero(self.cleanup_counts))


class OnlineLocalRoleTransitionMemory:
    """
    Parser-free local role/transition circuit for unified token QA.

    The circuit starts from query-overlap seed tokens, walks prompt-local token
    neighborhoods for a small number of hops, and learns a local event gate from
    answer-token target/wrong credit.  It never receives bAbI event labels,
    symbolic roles, answer classes, BP gradients, raw replay, or a separate QA
    head.
    """

    BASE_BRANCH_ARBITER_VARIANTS = ("base_only", "role_only", "base_plus_role", "base_plus_direct")
    JOINT_BRANCH_ARBITER_VARIANTS = ("base_plus_role_joint", "base_plus_direct_joint")
    BRANCH_ARBITER_VARIANTS = BASE_BRANCH_ARBITER_VARIANTS
    JOINT_SUPPRESS_MODES = ("all_wrong", "protect_direct", "joint_only")
    EDGE_PATH_DIRECT_MODES = ("soft_feature", "candidate_scores", "structured_scores")
    EDGE_PATH_CLEANUP_CREDIT_MODES = (
        "selected_target",
        "reward_punish",
        "soft_eligibility",
        "margin_gated_soft_eligibility",
        "learned_margin_escape",
        "transient_inhibit_escape",
    )
    EDGE_PATH_RUNNER_ARBITER_CREDIT_MODES = ("answer_error", "counterfactual_positive")
    EDGE_PATH_RUNNER_ARBITER_NEGATIVE_MODES = ("subtract", "separate")
    EDGE_PATH_RUNNER_ARBITER_FEATURE_MODES = ("pair", "rich_gaps")
    EDGE_PATH_AFFINITY_FEATURE_DIM = 12

    def __init__(
        self,
        vocab_size: int,
        state_dim: int,
        state_order: int,
        state_decay: float,
        slots: int,
        lr: float,
        wrong_lr: float,
        score_scale: float,
        bias_weight: float,
        margin: float,
        query_order: int,
        role_hops: int,
        role_window: int,
        role_top_k: int,
        role_recency_decay: float,
        role_locality_decay: float,
        role_gate_lr: float,
        role_gate_wrong_lr: float,
        role_gate_strength: float,
        role_score_scale: float,
        role_downstream_bonus: float,
        role_channel_gates: bool,
        role_final_score_only: bool,
        role_score_top_k: int,
        role_score_inhibit: float,
        role_score_gate_mode: str,
        role_score_gate_base_margin: float,
        role_score_gate_role_margin: float,
        role_branch_readout: bool,
        role_branch_base_score_scale: float,
        role_branch_role_score_scale: float,
        role_joint_rescue_readout: bool,
        role_joint_rescue_score_scale: float,
        role_joint_rescue_top_k: int,
        role_joint_rescue_inhibit: float,
        role_joint_suppress_slots: int,
        role_joint_suppress_lr: float,
        role_joint_suppress_score_scale: float,
        role_joint_suppress_margin: float,
        role_joint_suppress_mode: str,
        role_joint_suppress_direct_threshold: float,
        role_joint_suppress_joint_threshold: float,
        role_branch_arbiter: str,
        role_branch_arbiter_default: str,
        role_branch_arbiter_slots: int,
        role_branch_arbiter_lr: float,
        role_branch_arbiter_wrong_lr: float,
        role_branch_arbiter_score_scale: float,
        role_branch_arbiter_margin: float,
        role_branch_arbiter_min_count: float,
        role_branch_arbiter_base_margin: float,
        role_branch_arbiter_threshold_lr: float,
        role_branch_arbiter_rescue_role_threshold: float,
        role_branch_arbiter_rescue_joint_threshold: float,
        role_branch_arbiter_joint_variants: bool,
        role_branch_arbiter_rich_conflict_features: bool,
        role_event_cache_size: int,
        edge_path_cleanup_answer_slots: int,
        edge_path_cleanup_slots: int,
        edge_path_cleanup_lr: float,
        edge_path_cleanup_wrong_lr: float,
        edge_path_cleanup_score_scale: float,
        edge_path_cleanup_top_k: int,
        edge_path_cleanup_inhibit: float,
        edge_path_cleanup_credit_mode: str,
        edge_path_margin_gate: float,
        edge_path_margin_min_scale: float,
        edge_path_margin_alt_scale: float,
        edge_path_margin_learned_dominance: float,
        edge_path_margin_escape_scale: float,
        edge_path_transient_inhibit_scale: float,
        edge_path_transient_inhibit_lr: float,
        edge_path_transient_inhibit_decay: float,
        edge_path_transient_inhibit_key: str,
        edge_path_transient_inhibit_hash_size: int,
        edge_path_transient_boost_scale: float,
        edge_path_transient_boost_lr: float,
        edge_path_transient_boost_support_margin: float,
        edge_path_transient_boost_consistency_margin: float,
        edge_path_transient_boost_runner_learned_max: float,
        edge_path_transient_boost_counterfactual_min_gain: float,
        edge_path_homeostasis_scale: float,
        edge_path_homeostasis_lr: float,
        edge_path_homeostasis_decay: float,
        edge_path_homeostasis_min_slot: int,
        edge_path_homeostasis_learned_dominance: float,
        edge_path_homeostasis_structure_margin: float,
        edge_path_homeostasis_soft_mod_scale: float,
        edge_path_homeostasis_soft_mod_floor: float,
        edge_path_homeostasis_trace_threshold: float,
        edge_path_homeostasis_trace_gain: float,
        edge_path_runner_arbiter_slots: int,
        edge_path_runner_arbiter_lr: float,
        edge_path_runner_arbiter_wrong_lr: float,
        edge_path_runner_arbiter_score_scale: float,
        edge_path_runner_arbiter_margin: float,
        edge_path_runner_arbiter_min_count: float,
        edge_path_runner_arbiter_negative_mode: str,
        edge_path_runner_arbiter_feature_mode: str,
        edge_path_runner_arbiter_gap_scale: float,
        edge_path_runner_arbiter_credit_mode: str,
        edge_path_soft_top_k: int,
        edge_path_soft_temperature: float,
        edge_path_soft_consistency_scale: float,
        edge_path_soft_learned_scale: float,
        edge_path_closure_score_scale: float,
        edge_path_closure_proto_slots: int,
        edge_path_closure_proto_lr: float,
        edge_path_closure_proto_wrong_lr: float,
        edge_path_closure_proto_score_scale: float,
        edge_path_closure_proto_min_count: float,
        edge_path_affinity_slots: int,
        edge_path_affinity_lr: float,
        edge_path_affinity_wrong_lr: float,
        edge_path_affinity_score_scale: float,
        edge_path_affinity_min_count: float,
        edge_path_affinity_margin_gate: float,
        edge_path_affinity_learned_dominance: float,
        edge_path_affinity_consistency_protect: float,
        edge_path_direct_answer_slots: int,
        edge_path_direct_slots: int,
        edge_path_direct_lr: float,
        edge_path_direct_wrong_lr: float,
        edge_path_direct_score_scale: float,
        edge_path_direct_mode: str,
        edge_path_structured_side_weight: float,
        edge_path_structured_path_weight: float,
        edge_path_structured_other_weight: float,
        seed: int,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.state_dim = max(int(state_dim), 1)
        self.max_order = max(int(state_order), 1)
        self.state_decay = float(np.clip(state_decay, 0.0, 0.999))
        self.slots = max(int(slots), 1)
        self.lr = float(np.clip(lr, 0.0, 1.0))
        self.wrong_lr = float(np.clip(wrong_lr, 0.0, 1.0))
        self.score_scale = float(score_scale)
        self.bias_weight = float(bias_weight)
        self.margin = float(margin)
        self.query_order = max(int(query_order), 1)
        self.role_hops = max(int(role_hops), 1)
        self.role_window = max(int(role_window), 1)
        self.role_top_k = max(int(role_top_k), 1)
        self.role_recency_decay = float(np.clip(role_recency_decay, 0.0, 1.0))
        self.role_locality_decay = float(np.clip(role_locality_decay, 0.0, 1.0))
        self.role_gate_lr = float(max(role_gate_lr, 0.0))
        self.role_gate_wrong_lr = float(max(role_gate_wrong_lr, 0.0))
        self.role_gate_strength = float(max(role_gate_strength, 0.0))
        self.role_score_scale = float(max(role_score_scale, 0.0))
        self.role_downstream_bonus = float(max(role_downstream_bonus, 0.0))
        self.role_channel_gates = bool(role_channel_gates)
        self.role_final_score_only = bool(role_final_score_only)
        self.role_score_top_k = max(int(role_score_top_k), 0)
        self.role_score_inhibit = float(max(role_score_inhibit, 0.0))
        self.role_score_gate_mode = str(role_score_gate_mode)
        self.role_score_gate_base_margin = float(max(role_score_gate_base_margin, 0.0))
        self.role_score_gate_role_margin = float(max(role_score_gate_role_margin, 0.0))
        self.role_branch_readout = bool(role_branch_readout)
        self.role_branch_base_score_scale = float(max(role_branch_base_score_scale, 0.0))
        self.role_branch_role_score_scale = float(max(role_branch_role_score_scale, 0.0))
        self.role_joint_rescue_readout = bool(role_joint_rescue_readout)
        self.role_joint_rescue_score_scale = float(max(role_joint_rescue_score_scale, 0.0))
        self.role_joint_rescue_top_k = max(int(role_joint_rescue_top_k), 0)
        self.role_joint_rescue_inhibit = float(max(role_joint_rescue_inhibit, 0.0))
        self.role_joint_suppress_slots = max(int(role_joint_suppress_slots), 1)
        self.role_joint_suppress_lr = float(np.clip(role_joint_suppress_lr, 0.0, 1.0))
        self.role_joint_suppress_score_scale = float(max(role_joint_suppress_score_scale, 0.0))
        self.role_joint_suppress_margin = float(max(role_joint_suppress_margin, 0.0))
        self.role_joint_suppress_mode = str(role_joint_suppress_mode)
        self.role_joint_suppress_direct_threshold = float(max(role_joint_suppress_direct_threshold, 0.0))
        self.role_joint_suppress_joint_threshold = float(max(role_joint_suppress_joint_threshold, 0.0))
        if self.role_joint_rescue_readout and not self.role_branch_readout:
            raise ValueError("--role-joint-rescue-readout requires --role-branch-readout")
        if self.role_joint_suppress_score_scale > 0.0 and not self.role_joint_rescue_readout:
            raise ValueError("--role-joint-suppress-score-scale requires --role-joint-rescue-readout")
        if self.role_joint_suppress_mode not in self.JOINT_SUPPRESS_MODES:
            raise ValueError(f"unknown role_joint_suppress_mode: {self.role_joint_suppress_mode}")
        self.role_branch_arbiter = str(role_branch_arbiter)
        self.role_branch_arbiter_default = str(role_branch_arbiter_default)
        self.role_branch_arbiter_slots = max(int(role_branch_arbiter_slots), 1)
        self.role_branch_arbiter_lr = float(np.clip(role_branch_arbiter_lr, 0.0, 1.0))
        self.role_branch_arbiter_wrong_lr = float(np.clip(role_branch_arbiter_wrong_lr, 0.0, 1.0))
        self.role_branch_arbiter_score_scale = float(max(role_branch_arbiter_score_scale, 0.0))
        self.role_branch_arbiter_margin = float(max(role_branch_arbiter_margin, 0.0))
        self.role_branch_arbiter_min_count = float(max(role_branch_arbiter_min_count, 0.0))
        self.role_branch_arbiter_base_margin = float(max(role_branch_arbiter_base_margin, 0.0))
        self.role_branch_arbiter_threshold_lr = float(np.clip(role_branch_arbiter_threshold_lr, 0.0, 1.0))
        self.role_branch_arbiter_rescue_role_threshold = float(
            max(role_branch_arbiter_rescue_role_threshold, 0.0)
        )
        self.role_branch_arbiter_rescue_joint_threshold = float(
            max(role_branch_arbiter_rescue_joint_threshold, 0.0)
        )
        self.role_branch_arbiter_joint_variants = bool(role_branch_arbiter_joint_variants)
        self.role_branch_arbiter_rich_conflict_features = bool(
            role_branch_arbiter_rich_conflict_features
        )
        if self.role_branch_arbiter_joint_variants and not self.role_joint_rescue_readout:
            raise ValueError("--role-branch-arbiter-joint-variants requires --role-joint-rescue-readout")
        self.branch_arbiter_variants = self.BASE_BRANCH_ARBITER_VARIANTS + (
            self.JOINT_BRANCH_ARBITER_VARIANTS if self.role_branch_arbiter_joint_variants else ()
        )
        if self.role_branch_arbiter != "none" and not self.role_branch_readout:
            raise ValueError("--role-branch-arbiter requires --role-branch-readout")
        if self.role_branch_arbiter_default not in self.branch_arbiter_variants:
            raise ValueError(f"unknown role_branch_arbiter_default: {self.role_branch_arbiter_default}")
        self.role_event_cache_size = max(int(role_event_cache_size), 0)
        self.edge_path_cleanup_answer_slots = max(int(edge_path_cleanup_answer_slots), 1)
        self.edge_path_cleanup_slots = max(int(edge_path_cleanup_slots), 1)
        self.edge_path_cleanup_lr = float(np.clip(edge_path_cleanup_lr, 0.0, 1.0))
        self.edge_path_cleanup_wrong_lr = float(np.clip(edge_path_cleanup_wrong_lr, 0.0, 1.0))
        self.edge_path_cleanup_score_scale = float(max(edge_path_cleanup_score_scale, 0.0))
        self.edge_path_cleanup_top_k = max(int(edge_path_cleanup_top_k), 1)
        self.edge_path_cleanup_inhibit = float(max(edge_path_cleanup_inhibit, 0.0))
        self.edge_path_cleanup_credit_mode = str(edge_path_cleanup_credit_mode)
        if self.edge_path_cleanup_credit_mode not in self.EDGE_PATH_CLEANUP_CREDIT_MODES:
            raise ValueError(f"unknown edge_path_cleanup_credit_mode: {self.edge_path_cleanup_credit_mode}")
        self.edge_path_margin_gate = float(max(edge_path_margin_gate, 0.0))
        self.edge_path_margin_min_scale = float(np.clip(edge_path_margin_min_scale, 0.0, 1.0))
        self.edge_path_margin_alt_scale = float(np.clip(edge_path_margin_alt_scale, 0.0, 1.0))
        self.edge_path_margin_learned_dominance = float(max(edge_path_margin_learned_dominance, 0.0))
        self.edge_path_margin_escape_scale = float(np.clip(edge_path_margin_escape_scale, 0.0, 1.0))
        self.edge_path_transient_inhibit_scale = float(max(edge_path_transient_inhibit_scale, 0.0))
        self.edge_path_transient_inhibit_lr = float(np.clip(edge_path_transient_inhibit_lr, 0.0, 1.0))
        self.edge_path_transient_inhibit_decay = float(
            np.clip(edge_path_transient_inhibit_decay, 0.0, 1.0)
        )
        self.edge_path_transient_inhibit_key = str(edge_path_transient_inhibit_key)
        if self.edge_path_transient_inhibit_key not in {"mid", "path_hash", "anchor_path"}:
            raise ValueError(
                f"unknown edge_path_transient_inhibit_key: {self.edge_path_transient_inhibit_key}"
            )
        self.edge_path_transient_inhibit_hash_size = max(int(edge_path_transient_inhibit_hash_size), 1)
        self.edge_path_transient_boost_scale = float(max(edge_path_transient_boost_scale, 0.0))
        self.edge_path_transient_boost_lr = float(np.clip(edge_path_transient_boost_lr, 0.0, 1.0))
        self.edge_path_transient_boost_support_margin = float(max(edge_path_transient_boost_support_margin, 0.0))
        self.edge_path_transient_boost_consistency_margin = float(
            edge_path_transient_boost_consistency_margin
        )
        self.edge_path_transient_boost_runner_learned_max = float(
            edge_path_transient_boost_runner_learned_max
        )
        self.edge_path_transient_boost_counterfactual_min_gain = float(
            edge_path_transient_boost_counterfactual_min_gain
        )
        self.edge_path_homeostasis_scale = float(max(edge_path_homeostasis_scale, 0.0))
        self.edge_path_homeostasis_lr = float(np.clip(edge_path_homeostasis_lr, 0.0, 1.0))
        self.edge_path_homeostasis_decay = float(np.clip(edge_path_homeostasis_decay, 0.0, 1.0))
        self.edge_path_homeostasis_min_slot = max(int(edge_path_homeostasis_min_slot), 0)
        self.edge_path_homeostasis_learned_dominance = float(
            max(edge_path_homeostasis_learned_dominance, 0.0)
        )
        self.edge_path_homeostasis_structure_margin = float(
            max(edge_path_homeostasis_structure_margin, 0.0)
        )
        self.edge_path_homeostasis_soft_mod_scale = float(
            max(edge_path_homeostasis_soft_mod_scale, 0.0)
        )
        self.edge_path_homeostasis_soft_mod_floor = float(
            np.clip(edge_path_homeostasis_soft_mod_floor, 0.0, 1.0)
        )
        self.edge_path_homeostasis_trace_threshold = float(
            np.clip(edge_path_homeostasis_trace_threshold, 0.0, 1.0)
        )
        self.edge_path_homeostasis_trace_gain = float(max(edge_path_homeostasis_trace_gain, 0.0))
        self.edge_path_runner_arbiter_slots = max(int(edge_path_runner_arbiter_slots), 1)
        self.edge_path_runner_arbiter_lr = float(np.clip(edge_path_runner_arbiter_lr, 0.0, 1.0))
        self.edge_path_runner_arbiter_wrong_lr = float(
            np.clip(edge_path_runner_arbiter_wrong_lr, 0.0, 1.0)
        )
        self.edge_path_runner_arbiter_score_scale = float(max(edge_path_runner_arbiter_score_scale, 0.0))
        self.edge_path_runner_arbiter_margin = float(max(edge_path_runner_arbiter_margin, 0.0))
        self.edge_path_runner_arbiter_min_count = float(max(edge_path_runner_arbiter_min_count, 0.0))
        self.edge_path_runner_arbiter_negative_mode = str(edge_path_runner_arbiter_negative_mode)
        if self.edge_path_runner_arbiter_negative_mode not in self.EDGE_PATH_RUNNER_ARBITER_NEGATIVE_MODES:
            raise ValueError(
                f"unknown edge_path_runner_arbiter_negative_mode: {self.edge_path_runner_arbiter_negative_mode}"
            )
        self.edge_path_runner_arbiter_feature_mode = str(edge_path_runner_arbiter_feature_mode)
        if self.edge_path_runner_arbiter_feature_mode not in self.EDGE_PATH_RUNNER_ARBITER_FEATURE_MODES:
            raise ValueError(
                f"unknown edge_path_runner_arbiter_feature_mode: {self.edge_path_runner_arbiter_feature_mode}"
            )
        self.edge_path_runner_arbiter_gap_scale = float(max(edge_path_runner_arbiter_gap_scale, 0.0))
        self.edge_path_runner_arbiter_credit_mode = str(edge_path_runner_arbiter_credit_mode)
        if self.edge_path_runner_arbiter_credit_mode not in self.EDGE_PATH_RUNNER_ARBITER_CREDIT_MODES:
            raise ValueError(
                f"unknown edge_path_runner_arbiter_credit_mode: {self.edge_path_runner_arbiter_credit_mode}"
            )
        self.edge_path_soft_top_k = max(int(edge_path_soft_top_k), 1)
        self.edge_path_soft_temperature = float(max(edge_path_soft_temperature, 1e-4))
        self.edge_path_soft_consistency_scale = float(edge_path_soft_consistency_scale)
        self.edge_path_soft_learned_scale = float(edge_path_soft_learned_scale)
        self.edge_path_closure_score_scale = float(max(edge_path_closure_score_scale, 0.0))
        self.edge_path_closure_proto_slots = max(int(edge_path_closure_proto_slots), 1)
        self.edge_path_closure_proto_lr = float(np.clip(edge_path_closure_proto_lr, 0.0, 1.0))
        self.edge_path_closure_proto_wrong_lr = float(
            np.clip(edge_path_closure_proto_wrong_lr, 0.0, 1.0)
        )
        self.edge_path_closure_proto_score_scale = float(max(edge_path_closure_proto_score_scale, 0.0))
        self.edge_path_closure_proto_min_count = float(max(edge_path_closure_proto_min_count, 0.0))
        self.edge_path_affinity_slots = max(int(edge_path_affinity_slots), 1)
        self.edge_path_affinity_lr = float(np.clip(edge_path_affinity_lr, 0.0, 1.0))
        self.edge_path_affinity_wrong_lr = float(np.clip(edge_path_affinity_wrong_lr, 0.0, 1.0))
        self.edge_path_affinity_score_scale = float(max(edge_path_affinity_score_scale, 0.0))
        self.edge_path_affinity_min_count = float(max(edge_path_affinity_min_count, 0.0))
        self.edge_path_affinity_margin_gate = float(max(edge_path_affinity_margin_gate, 0.0))
        self.edge_path_affinity_learned_dominance = float(max(edge_path_affinity_learned_dominance, 0.0))
        self.edge_path_affinity_consistency_protect = float(max(edge_path_affinity_consistency_protect, 0.0))
        self.edge_path_direct_answer_slots = max(int(edge_path_direct_answer_slots), 1)
        self.edge_path_direct_slots = max(int(edge_path_direct_slots), 1)
        self.edge_path_direct_lr = float(np.clip(edge_path_direct_lr, 0.0, 1.0))
        self.edge_path_direct_wrong_lr = float(np.clip(edge_path_direct_wrong_lr, 0.0, 1.0))
        self.edge_path_direct_score_scale = float(max(edge_path_direct_score_scale, 0.0))
        self.edge_path_direct_mode = str(edge_path_direct_mode)
        if self.edge_path_direct_mode not in self.EDGE_PATH_DIRECT_MODES:
            raise ValueError(f"unknown edge_path_direct_mode: {self.edge_path_direct_mode}")
        self.edge_path_structured_side_weight = float(max(edge_path_structured_side_weight, 0.0))
        self.edge_path_structured_path_weight = float(max(edge_path_structured_path_weight, 0.0))
        self.edge_path_structured_other_weight = float(max(edge_path_structured_other_weight, 0.0))
        self.feature_dim = self.state_dim * (1 + self.role_hops)
        rng = np.random.default_rng(seed + 15485863)
        self.token_codes = phase.normalize_rows(
            rng.normal(0.0, 1.0, (self.vocab_size, self.state_dim)).astype(np.float32)
        )
        self.position_codes = phase.normalize_rows(
            rng.normal(0.0, 1.0, (self.max_order, self.state_dim)).astype(np.float32)
        )
        self.relative_codes = phase.normalize_rows(
            rng.normal(0.0, 1.0, (2 * self.role_window + 1, self.state_dim)).astype(np.float32)
        )
        if self.edge_path_runner_arbiter_feature_mode == "rich_gaps":
            self.edge_path_runner_arbiter_gap_codes = phase.normalize_rows(
                rng.normal(0.0, 1.0, (4, self.state_dim)).astype(np.float32)
            )
        else:
            self.edge_path_runner_arbiter_gap_codes = None
        gate_rows = self.role_hops if self.role_channel_gates else 1
        self.role_gate_weights = np.zeros((gate_rows, self.state_dim), dtype=np.float32)
        if self.role_branch_readout:
            self.prototypes = None
            self.prototype_counts = None
            self.base_branch_prototypes = np.zeros(
                (self.vocab_size, self.slots, self.state_dim), dtype=np.float32
            )
            self.base_branch_counts = np.zeros((self.vocab_size, self.slots), dtype=np.float32)
            self.role_branch_prototypes = np.zeros(
                (self.vocab_size, self.slots, self.state_dim * self.role_hops), dtype=np.float32
            )
            self.role_branch_counts = np.zeros((self.vocab_size, self.slots), dtype=np.float32)
            if self.role_joint_rescue_readout:
                self.joint_rescue_prototypes = np.zeros(
                    (self.vocab_size, self.slots, self.feature_dim), dtype=np.float32
                )
                self.joint_rescue_counts = np.zeros((self.vocab_size, self.slots), dtype=np.float32)
                if self.role_joint_suppress_score_scale > 0.0:
                    self.joint_suppress_prototypes = np.zeros(
                        (self.vocab_size, self.role_joint_suppress_slots, self.feature_dim),
                        dtype=np.float32,
                    )
                    self.joint_suppress_counts = np.zeros(
                        (self.vocab_size, self.role_joint_suppress_slots),
                        dtype=np.float32,
                    )
                else:
                    self.joint_suppress_prototypes = None
                    self.joint_suppress_counts = None
            else:
                self.joint_rescue_prototypes = None
                self.joint_rescue_counts = None
                self.joint_suppress_prototypes = None
                self.joint_suppress_counts = None
        else:
            self.prototypes = np.zeros((self.vocab_size, self.slots, self.feature_dim), dtype=np.float32)
            self.prototype_counts = np.zeros((self.vocab_size, self.slots), dtype=np.float32)
            self.base_branch_prototypes = None
            self.base_branch_counts = None
            self.role_branch_prototypes = None
            self.role_branch_counts = None
            self.joint_rescue_prototypes = None
            self.joint_rescue_counts = None
            self.joint_suppress_prototypes = None
            self.joint_suppress_counts = None
        self.unigram_counts = np.ones(self.vocab_size, dtype=np.float32)
        self.output_bias = np.full(self.vocab_size, -math.log(max(self.vocab_size, 1)), dtype=np.float32)
        self.edge_path_cleanup_prototypes = np.zeros(
            (
                self.edge_path_cleanup_answer_slots,
                self.vocab_size,
                self.edge_path_cleanup_slots,
                self.state_dim,
            ),
            dtype=np.float32,
        )
        self.edge_path_cleanup_counts = np.zeros(
            (self.edge_path_cleanup_answer_slots, self.vocab_size, self.edge_path_cleanup_slots),
            dtype=np.float32,
        )
        self.edge_path_cleanup_updates = np.zeros(self.edge_path_cleanup_answer_slots, dtype=np.int64)
        self.edge_path_cleanup_wrong_updates = np.zeros(self.edge_path_cleanup_answer_slots, dtype=np.int64)
        self.edge_path_cleanup_checks = 0
        self.edge_path_cleanup_candidates = 0
        self.edge_path_cleanup_wins = np.zeros(self.edge_path_cleanup_answer_slots, dtype=np.int64)
        transient_trace_size = (
            self.vocab_size
            if self.edge_path_transient_inhibit_key == "mid"
            else self.edge_path_transient_inhibit_hash_size
        )
        self.edge_path_transient_inhibit_trace = np.zeros(
            (self.edge_path_cleanup_answer_slots, transient_trace_size),
            dtype=np.float32,
        )
        self.edge_path_transient_inhibit_updates = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_transient_boost_trace = np.zeros(
            (self.edge_path_cleanup_answer_slots, transient_trace_size),
            dtype=np.float32,
        )
        self.edge_path_transient_boost_updates = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_transient_boost_consistency_skips = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_transient_boost_learned_skips = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_transient_boost_counterfactual_skips = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        if self.edge_path_homeostasis_scale > 0.0 or self.edge_path_homeostasis_lr > 0.0:
            self.edge_path_homeostasis_trace = np.zeros(
                (self.edge_path_cleanup_answer_slots, transient_trace_size),
                dtype=np.float32,
            )
        else:
            self.edge_path_homeostasis_trace = None
        self.edge_path_homeostasis_updates = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_homeostasis_gate_checks = 0
        self.edge_path_homeostasis_gate_passes = 0
        self.edge_path_homeostasis_gate_skips = 0
        self.edge_path_homeostasis_soft_mod_checks = 0
        self.edge_path_homeostasis_soft_mod_sum = 0.0
        self.edge_path_homeostasis_soft_mod_min = 1.0
        self.edge_path_homeostasis_soft_mod_max = 0.0
        self.edge_path_homeostasis_trace_mod_checks = 0
        self.edge_path_homeostasis_trace_mod_raw_sum = 0.0
        self.edge_path_homeostasis_trace_mod_effective_sum = 0.0
        self.edge_path_homeostasis_trace_mod_active = 0
        self.edge_path_runner_arbiter_prototypes = np.zeros(
            (
                self.edge_path_cleanup_answer_slots,
                self.edge_path_runner_arbiter_slots,
                self.state_dim,
            ),
            dtype=np.float32,
        )
        self.edge_path_runner_arbiter_counts = np.zeros(
            (self.edge_path_cleanup_answer_slots, self.edge_path_runner_arbiter_slots),
            dtype=np.float32,
        )
        if self.edge_path_runner_arbiter_negative_mode == "separate":
            self.edge_path_runner_arbiter_negative_prototypes = np.zeros(
                (
                    self.edge_path_cleanup_answer_slots,
                    self.edge_path_runner_arbiter_slots,
                    self.state_dim,
                ),
                dtype=np.float32,
            )
            self.edge_path_runner_arbiter_negative_counts = np.zeros(
                (self.edge_path_cleanup_answer_slots, self.edge_path_runner_arbiter_slots),
                dtype=np.float32,
            )
        else:
            self.edge_path_runner_arbiter_negative_prototypes = None
            self.edge_path_runner_arbiter_negative_counts = None
        self.edge_path_runner_arbiter_updates = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_runner_arbiter_wrong_updates = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_runner_arbiter_score_checks = 0
        self.edge_path_runner_arbiter_score_applied = 0
        self.edge_path_runner_arbiter_score_count_skips = 0
        if self.edge_path_closure_proto_score_scale > 0.0:
            self.edge_path_closure_proto_positive = np.zeros(
                (
                    self.edge_path_cleanup_answer_slots,
                    self.edge_path_closure_proto_slots,
                    self.state_dim,
                ),
                dtype=np.float32,
            )
            self.edge_path_closure_proto_positive_counts = np.zeros(
                (self.edge_path_cleanup_answer_slots, self.edge_path_closure_proto_slots),
                dtype=np.float32,
            )
            self.edge_path_closure_proto_negative = np.zeros(
                (
                    self.edge_path_cleanup_answer_slots,
                    self.edge_path_closure_proto_slots,
                    self.state_dim,
                ),
                dtype=np.float32,
            )
            self.edge_path_closure_proto_negative_counts = np.zeros(
                (self.edge_path_cleanup_answer_slots, self.edge_path_closure_proto_slots),
                dtype=np.float32,
            )
        else:
            self.edge_path_closure_proto_positive = None
            self.edge_path_closure_proto_positive_counts = None
            self.edge_path_closure_proto_negative = None
            self.edge_path_closure_proto_negative_counts = None
        self.edge_path_closure_proto_updates = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_closure_proto_wrong_updates = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_closure_proto_score_checks = 0
        self.edge_path_closure_proto_score_applied = 0
        self.edge_path_closure_proto_score_count_skips = 0
        if self.edge_path_affinity_score_scale > 0.0:
            self.edge_path_affinity_positive = np.zeros(
                (
                    self.edge_path_cleanup_answer_slots,
                    self.edge_path_affinity_slots,
                    self.EDGE_PATH_AFFINITY_FEATURE_DIM,
                ),
                dtype=np.float32,
            )
            self.edge_path_affinity_positive_counts = np.zeros(
                (self.edge_path_cleanup_answer_slots, self.edge_path_affinity_slots),
                dtype=np.float32,
            )
            self.edge_path_affinity_negative = np.zeros(
                (
                    self.edge_path_cleanup_answer_slots,
                    self.edge_path_affinity_slots,
                    self.EDGE_PATH_AFFINITY_FEATURE_DIM,
                ),
                dtype=np.float32,
            )
            self.edge_path_affinity_negative_counts = np.zeros(
                (self.edge_path_cleanup_answer_slots, self.edge_path_affinity_slots),
                dtype=np.float32,
            )
        else:
            self.edge_path_affinity_positive = None
            self.edge_path_affinity_positive_counts = None
            self.edge_path_affinity_negative = None
            self.edge_path_affinity_negative_counts = None
        self.edge_path_affinity_updates = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_affinity_wrong_updates = np.zeros(
            self.edge_path_cleanup_answer_slots,
            dtype=np.int64,
        )
        self.edge_path_affinity_score_checks = 0
        self.edge_path_affinity_score_applied = 0
        self.edge_path_affinity_score_count_skips = 0
        self.edge_path_affinity_gate_checks = 0
        self.edge_path_affinity_gate_passes = 0
        self.edge_path_affinity_gate_near_passes = 0
        self.edge_path_affinity_gate_conflict_passes = 0
        self.edge_path_affinity_gate_skips = 0
        self.edge_path_direct_prototypes = np.zeros(
            (
                self.edge_path_direct_answer_slots,
                self.vocab_size,
                self.edge_path_direct_slots,
                self.state_dim,
            ),
            dtype=np.float32,
        )
        self.edge_path_direct_counts = np.zeros(
            (self.edge_path_direct_answer_slots, self.vocab_size, self.edge_path_direct_slots),
            dtype=np.float32,
        )
        self.edge_path_direct_updates = np.zeros(self.edge_path_direct_answer_slots, dtype=np.int64)
        self.edge_path_direct_wrong_updates = np.zeros(self.edge_path_direct_answer_slots, dtype=np.int64)
        self.edge_path_direct_score_checks = np.zeros(self.edge_path_direct_answer_slots, dtype=np.int64)
        self.last_edge_path_direct_feature: dict[int, np.ndarray] = {}
        self.last_edge_path_direct_feature_bundle: dict[int, list[tuple[float, np.ndarray]]] = {}
        self.last_edge_path_selection: dict[int, tuple[int, np.ndarray, float, float, float, float, int, int]] = {}
        self.last_edge_path_runner_up: dict[int, tuple[int, np.ndarray, float, float, float, float, int, int]] = {}
        self.last_edge_path_selection_states: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.last_edge_path_runner_states: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.last_edge_path_candidate_count: dict[int, int] = {}
        self.last_edge_path_trace_index: dict[int, int] = {}
        self.last_edge_path_runner_trace_index: dict[int, int] = {}
        self.last_edge_path_top_candidates: dict[
            int,
            list[tuple[int, np.ndarray, float, float, float, float]],
        ] = {}
        self.last_role_scores = np.zeros(self.vocab_size, dtype=np.float32)
        self.last_joint_rescue_delta = np.zeros(self.vocab_size, dtype=np.float32)
        self.last_base_feature = np.zeros(self.state_dim, dtype=np.float32)
        self.last_role_feature = np.zeros(self.state_dim * self.role_hops, dtype=np.float32)
        self.last_branch_components: dict[str, np.ndarray] = {}
        self.last_branch_arbiter_feature = np.zeros(1, dtype=np.float32)
        self.branch_arbiter_feature_dim = (
            1
            + len(self.branch_arbiter_variants)
            + (len(self.branch_arbiter_variants) * (len(self.branch_arbiter_variants) - 1)) // 2
            + self.state_dim * len(self.branch_arbiter_variants)
        )
        if self.role_branch_arbiter_rich_conflict_features:
            self.branch_arbiter_feature_dim += self.state_dim + 8
        if self.role_branch_arbiter in {"local_proto", "conflict_proto"}:
            self.branch_arbiter_prototypes = np.zeros(
                (
                    len(self.branch_arbiter_variants),
                    self.role_branch_arbiter_slots,
                    self.branch_arbiter_feature_dim,
                ),
                dtype=np.float32,
            )
            self.branch_arbiter_counts = np.zeros(
                (len(self.branch_arbiter_variants), self.role_branch_arbiter_slots),
                dtype=np.float32,
            )
        else:
            self.branch_arbiter_prototypes = None
            self.branch_arbiter_counts = None
        self.branch_arbiter_checks = 0
        self.branch_arbiter_updates = 0
        self.branch_arbiter_threshold = self.role_branch_arbiter_base_margin
        self.branch_arbiter_chosen = np.zeros(len(self.branch_arbiter_variants), dtype=np.int64)
        self.branch_arbiter_target_updates = np.zeros(len(self.branch_arbiter_variants), dtype=np.int64)
        self.joint_suppress_candidates = 0
        self.joint_suppress_updates = 0
        self.joint_suppress_skipped_direct = 0
        self.joint_suppress_skipped_joint = 0
        self.role_event_cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self.role_event_cache_hits = 0
        self.role_event_cache_misses = 0
        self.role_score_gate_checks = 0
        self.role_score_gate_opens = 0

    def recurrent_state(self, tokens: Sequence[int]) -> np.ndarray:
        if not tokens:
            return np.zeros(self.state_dim, dtype=np.float32)
        state = np.zeros(self.state_dim, dtype=np.float32)
        pos_start = self.max_order - len(tokens)
        for offset, token in enumerate(tokens):
            pos = pos_start + offset
            bound = phase.normalize_vector(self.token_codes[int(token)] * self.position_codes[pos])
            state = self.state_decay * state + bound
        return phase.normalize_vector(state)

    def prefix_and_query(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        query_len = min(self.query_order, len(tokens))
        prefix_end = max(len(tokens) - query_len, 0)
        prefix = [int(token) for token in tokens[:prefix_end]]
        query = [int(token) for token in tokens[prefix_end:]]
        if not prefix:
            prefix = [int(token) for token in tokens]
        return prefix, query

    def initial_query_seeds(self, prefix: list[int], query: list[int]) -> list[int]:
        if not prefix or not query:
            return []
        counts: dict[int, int] = {}
        for token in prefix:
            token = int(token)
            counts[token] = counts.get(token, 0) + 1
        candidates: list[tuple[int, int, int]] = []
        seen: set[int] = set()
        for pos, token in enumerate(query):
            token = int(token)
            count = counts.get(token, 0)
            if count <= 0 or token in seen:
                continue
            seen.add(token)
            candidates.append((count, -pos, token))
        candidates.sort()
        return [token for _, _, token in candidates[: self.role_top_k]]

    def event_feature(self, seed_token: int, neighbor_token: int, rel_idx: int) -> np.ndarray:
        key = (int(seed_token), int(neighbor_token), int(rel_idx))
        if self.role_event_cache_size > 0:
            cached = self.role_event_cache.get(key)
            if cached is not None:
                self.role_event_cache_hits += 1
                self.role_event_cache.move_to_end(key)
                return cached
            self.role_event_cache_misses += 1
        event = (
            self.token_codes[key[0]]
            * self.token_codes[key[1]]
            * self.relative_codes[key[2]]
        )
        feature = phase.normalize_vector(event)
        if self.role_event_cache_size > 0:
            if len(self.role_event_cache) >= self.role_event_cache_size:
                self.role_event_cache.popitem(last=False)
            self.role_event_cache[key] = feature
        return feature

    def event_gate(self, feature: np.ndarray, hop: int) -> float:
        if self.role_gate_strength <= 0.0:
            return 1.0
        gate_idx = min(max(int(hop), 0), self.role_gate_weights.shape[0] - 1)
        raw = float(self.role_gate_weights[gate_idx] @ feature)
        return float(max(0.0, 1.0 + self.role_gate_strength * math.tanh(raw)))

    def transition_events(
        self,
        prefix: list[int],
        seeds: Sequence[int],
        hop: int,
    ) -> list[tuple[float, int, int, float, np.ndarray]]:
        if not prefix or not seeds:
            return []
        seed_set = {int(token) for token in seeds}
        events: list[tuple[float, int, int, float, np.ndarray]] = []
        total = len(prefix)
        for pos, token in enumerate(prefix):
            seed_token = int(token)
            if seed_token not in seed_set:
                continue
            left = max(0, pos - self.role_window)
            right = min(total, pos + self.role_window + 1)
            recency = self.role_recency_decay ** max(total - 1 - pos, 0)
            for neighbor_pos in range(left, right):
                if neighbor_pos == pos:
                    continue
                rel = neighbor_pos - pos
                distance = abs(rel)
                rel_idx = rel + self.role_window
                locality = self.role_locality_decay ** max(distance - 1, 0)
                base_weight = float(recency * locality / math.sqrt(float(distance)))
                neighbor = int(prefix[neighbor_pos])
                feature = self.event_feature(seed_token, neighbor, rel_idx)
                score = float(base_weight * self.event_gate(feature, hop))
                if score <= 0.0:
                    continue
                events.append((score, seed_token, neighbor, base_weight, feature))
        return events

    def local_neighbor_strength(self, prefix: list[int], seed_token: int, target_token: int) -> float:
        if not prefix:
            return 0.0
        seed_token = int(seed_token)
        target_token = int(target_token)
        total = len(prefix)
        best = 0.0
        for pos, token in enumerate(prefix):
            if int(token) != seed_token:
                continue
            left = max(0, pos - self.role_window)
            right = min(total, pos + self.role_window + 1)
            recency = self.role_recency_decay ** max(total - 1 - pos, 0)
            for neighbor_pos in range(left, right):
                if neighbor_pos == pos or int(prefix[neighbor_pos]) != target_token:
                    continue
                distance = abs(neighbor_pos - pos)
                locality = self.role_locality_decay ** max(distance - 1, 0)
                best = max(best, float(recency * locality / math.sqrt(float(distance))))
        return best

    def top_tokens_from_scores(self, token_scores: dict[int, float]) -> list[int]:
        if not token_scores:
            return []
        ranked = sorted(token_scores.items(), key=lambda item: (-float(item[1]), int(item[0])))
        return [int(token) for token, _ in ranked[: self.role_top_k]]

    def transition_rollout(self, tokens: list[int]) -> tuple[list[np.ndarray], np.ndarray]:
        prefix, query = self.prefix_and_query(tokens)
        seeds = self.initial_query_seeds(prefix, query)
        role_scores = np.zeros(self.vocab_size, dtype=np.float32)
        pieces: list[np.ndarray] = []
        for hop in range(self.role_hops):
            events = self.transition_events(prefix, seeds, hop)
            token_scores: dict[int, float] = {}
            state = np.zeros(self.state_dim, dtype=np.float32)
            weight_sq_sum = 0.0
            for score, _, neighbor, _, _ in events:
                neighbor = int(neighbor)
                token_scores[neighbor] = token_scores.get(neighbor, 0.0) + float(score)
                state += float(score) * self.token_codes[neighbor]
                weight_sq_sum += float(score) * float(score)
            if weight_sq_sum > 0.0:
                state = phase.normalize_vector(state / math.sqrt(weight_sq_sum))
            pieces.append(state.astype(np.float32))
            if not self.role_final_score_only or hop + 1 == self.role_hops:
                for token, score in token_scores.items():
                    if 0 <= int(token) < self.vocab_size:
                        role_scores[int(token)] += float(score)
            seeds = self.top_tokens_from_scores(token_scores)
        if self.role_score_top_k > 0:
            active = np.flatnonzero(role_scores > 0.0)
            if active.size > self.role_score_top_k:
                active_scores = role_scores[active]
                winner_local = np.argpartition(active_scores, -self.role_score_top_k)[-self.role_score_top_k :]
                winner_tokens = {int(active[idx]) for idx in winner_local}
                max_score = float(np.max(active_scores)) if active_scores.size else 0.0
                for token in active:
                    token = int(token)
                    if token in winner_tokens:
                        continue
                    if self.role_score_inhibit > 0.0 and max_score > 0.0:
                        role_scores[token] = -self.role_score_inhibit * max_score
                    else:
                        role_scores[token] = 0.0
        while len(pieces) < self.role_hops:
            pieces.append(np.zeros(self.state_dim, dtype=np.float32))
        norm = float(np.linalg.norm(role_scores))
        if norm > 0.0:
            role_scores = (role_scores / norm).astype(np.float32)
        return pieces, role_scores

    def feature(self, context: Sequence[int] | np.ndarray) -> np.ndarray:
        tokens = [int(token) for token in list(context)[-self.max_order :]]
        role_pieces, role_scores = self.transition_rollout(tokens)
        self.last_role_scores = role_scores
        base_feature = self.recurrent_state(tokens)
        role_feature = np.concatenate(role_pieces).astype(np.float32)
        self.last_base_feature = base_feature.astype(np.float32, copy=True)
        self.last_role_feature = phase.normalize_vector(role_feature)
        pieces = [base_feature]
        pieces.extend(role_pieces)
        return phase.normalize_vector(np.concatenate(pieces).astype(np.float32))

    def ordered_query_anchors(self, prefix: list[int], query: list[int], limit: int = 2) -> list[int]:
        if not prefix or not query:
            return []
        counts: dict[int, int] = {}
        for token in prefix:
            token = int(token)
            counts[token] = counts.get(token, 0) + 1
        seen: set[int] = set()
        candidates: list[tuple[int, int, int]] = []
        for pos, token in enumerate(query):
            token = int(token)
            count = counts.get(token, 0)
            if count <= 0 or token in seen:
                continue
            seen.add(token)
            candidates.append((count, pos, token))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        chosen = sorted(candidates[: max(int(limit), 1)], key=lambda item: item[1])
        return [int(token) for _, _, token in chosen]

    def anchor_edge_map(self, prefix: list[int], anchor: int) -> dict[int, np.ndarray]:
        edge_map: dict[int, np.ndarray] = {}
        if not prefix:
            return edge_map
        anchor = int(anchor)
        total = len(prefix)
        for pos, token in enumerate(prefix):
            if int(token) != anchor:
                continue
            left = max(0, pos - self.role_window)
            right = min(total, pos + self.role_window + 1)
            recency = self.role_recency_decay ** max(total - 1 - pos, 0)
            for neighbor_pos in range(left, right):
                if neighbor_pos == pos:
                    continue
                neighbor = int(prefix[neighbor_pos])
                rel = neighbor_pos - pos
                rel_idx = rel + self.role_window
                distance = abs(rel)
                locality = self.role_locality_decay ** max(distance - 1, 0)
                weight = float(recency * locality / math.sqrt(float(distance)))
                feature = self.event_feature(anchor, neighbor, rel_idx)
                edge_map[neighbor] = edge_map.get(neighbor, np.zeros(self.state_dim, dtype=np.float32))
                edge_map[neighbor] = edge_map[neighbor] + weight * feature
        for neighbor, feature in list(edge_map.items()):
            edge_map[neighbor] = phase.normalize_vector(feature.astype(np.float32))
        return edge_map

    def edge_path_state(self, tokens: list[int], slot: int) -> np.ndarray:
        prefix, query = self.prefix_and_query(tokens)
        anchors = self.ordered_query_anchors(prefix, query, limit=2)
        if len(anchors) < 2:
            return np.zeros(self.state_dim, dtype=np.float32)
        source, destination = int(anchors[0]), int(anchors[-1])
        source_edges = self.anchor_edge_map(prefix, source)
        destination_edges = self.anchor_edge_map(prefix, destination)
        shared = [
            token
            for token in source_edges
            if token in destination_edges and token not in {source, destination}
        ]
        if not shared:
            return np.zeros(self.state_dim, dtype=np.float32)
        state = np.zeros(self.state_dim, dtype=np.float32)
        weight_sum = 0.0
        for mid in shared:
            source_feature = source_edges[mid]
            destination_feature = destination_edges[mid]
            source_strength = float(np.linalg.norm(source_feature))
            destination_strength = float(np.linalg.norm(destination_feature))
            weight = max(source_strength * destination_strength, 1e-6)
            if int(slot) <= 0:
                state += weight * phase.normalize_vector(
                    source_feature * self.token_codes[int(mid)]
                )
            else:
                state += weight * phase.normalize_vector(
                    destination_feature * self.token_codes[int(mid)]
                )
            weight_sum += weight
        if weight_sum <= 0.0:
            return np.zeros(self.state_dim, dtype=np.float32)
        return phase.normalize_vector((state / weight_sum).astype(np.float32))

    def edge_path_slot_index(self, slot: int) -> int:
        return min(max(int(slot), 0), self.edge_path_cleanup_answer_slots - 1)

    def edge_path_direct_slot_index(self, slot: int) -> int:
        return min(max(int(slot), 0), self.edge_path_direct_answer_slots - 1)

    def edge_path_direct_feature_bundle(
        self,
        slot: int,
        source_state: np.ndarray,
        destination_state: np.ndarray,
        path_feature: np.ndarray,
    ) -> list[tuple[float, np.ndarray]]:
        if self.edge_path_direct_mode != "structured_scores":
            return [(1.0, path_feature.astype(np.float32, copy=False))]
        side_feature = source_state if int(slot) <= 0 else destination_state
        other_feature = destination_state if int(slot) <= 0 else source_state
        weighted_features = [
            (self.edge_path_structured_side_weight, side_feature),
            (self.edge_path_structured_path_weight, path_feature),
            (self.edge_path_structured_other_weight, other_feature),
        ]
        weight_sum = sum(float(weight) for weight, _ in weighted_features if float(weight) > 0.0)
        if weight_sum <= 0.0:
            return [(1.0, path_feature.astype(np.float32, copy=False))]
        return [
            (float(weight) / weight_sum, feature.astype(np.float32, copy=False))
            for weight, feature in weighted_features
            if float(weight) > 0.0
        ]

    def edge_path_closure_score(
        self,
        source_state: np.ndarray,
        destination_state: np.ndarray,
        path_feature: np.ndarray,
    ) -> float:
        if self.edge_path_closure_score_scale <= 0.0:
            return 0.0
        return self.edge_path_raw_closure_score(source_state, destination_state, path_feature)

    @staticmethod
    def edge_path_raw_closure_score(
        source_state: np.ndarray,
        destination_state: np.ndarray,
        path_feature: np.ndarray,
    ) -> float:
        closure_feature = phase.normalize_vector((source_state * destination_state).astype(np.float32))
        if not np.any(closure_feature):
            return 0.0
        return float(max(path_feature @ closure_feature, 0.0))

    def edge_path_closure_proto_feature(
        self,
        source_state: np.ndarray,
        destination_state: np.ndarray,
        path_feature: np.ndarray,
    ) -> np.ndarray:
        closure_feature = phase.normalize_vector((source_state * destination_state).astype(np.float32))
        feature = (
            path_feature.astype(np.float32, copy=False)
            + closure_feature
            + 0.50 * (path_feature.astype(np.float32, copy=False) * closure_feature)
        )
        return phase.normalize_vector(feature.astype(np.float32))

    def edge_path_closure_proto_bank_score(
        self,
        prototypes: np.ndarray | None,
        counts: np.ndarray | None,
        slot_idx: int,
        feature: np.ndarray,
    ) -> tuple[float, float]:
        if prototypes is None or counts is None:
            return 0.0, 0.0
        slot_idx = self.edge_path_slot_index(slot_idx)
        slot_counts = counts[slot_idx]
        active = slot_counts > 0.0
        if not np.any(active):
            return 0.0, 0.0
        dots = prototypes[slot_idx] @ feature
        dots = np.where(active, dots, -np.inf)
        bank_slot = int(np.argmax(dots))
        return float(max(dots[bank_slot], 0.0)), float(slot_counts[bank_slot])

    def edge_path_closure_proto_score(
        self,
        slot_idx: int,
        source_state: np.ndarray,
        destination_state: np.ndarray,
        path_feature: np.ndarray,
    ) -> float:
        if self.edge_path_closure_proto_score_scale <= 0.0:
            return 0.0
        feature = self.edge_path_closure_proto_feature(source_state, destination_state, path_feature)
        positive, positive_count = self.edge_path_closure_proto_bank_score(
            self.edge_path_closure_proto_positive,
            self.edge_path_closure_proto_positive_counts,
            slot_idx,
            feature,
        )
        negative, _ = self.edge_path_closure_proto_bank_score(
            self.edge_path_closure_proto_negative,
            self.edge_path_closure_proto_negative_counts,
            slot_idx,
            feature,
        )
        self.edge_path_closure_proto_score_checks += 1
        if positive_count < self.edge_path_closure_proto_min_count:
            self.edge_path_closure_proto_score_count_skips += 1
            return 0.0
        score = positive - negative
        if score > 0.0:
            self.edge_path_closure_proto_score_applied += 1
        return score

    def update_edge_path_closure_proto_bank(
        self,
        prototypes: np.ndarray | None,
        counts: np.ndarray | None,
        slot_idx: int,
        feature: np.ndarray,
        lr: float,
    ) -> bool:
        if prototypes is None or counts is None or lr <= 0.0:
            return False
        slot_idx = self.edge_path_slot_index(slot_idx)
        slot_counts = counts[slot_idx]
        empty = np.flatnonzero(slot_counts <= 0.0)
        if empty.size:
            bank_slot = int(empty[0])
            prototypes[slot_idx, bank_slot] = feature
            counts[slot_idx, bank_slot] = 1.0
            return True
        dots = prototypes[slot_idx] @ feature
        bank_slot = int(np.argmax(dots))
        prototypes[slot_idx, bank_slot] = phase.normalize_vector(
            (1.0 - lr) * prototypes[slot_idx, bank_slot] + lr * feature
        )
        counts[slot_idx, bank_slot] += 1.0
        return True

    def update_edge_path_closure_proto_positive(self, slot_idx: int, feature: np.ndarray) -> None:
        if self.update_edge_path_closure_proto_bank(
            self.edge_path_closure_proto_positive,
            self.edge_path_closure_proto_positive_counts,
            slot_idx,
            feature,
            self.edge_path_closure_proto_lr,
        ):
            self.edge_path_closure_proto_updates[self.edge_path_slot_index(slot_idx)] += 1

    def update_edge_path_closure_proto_negative(self, slot_idx: int, feature: np.ndarray) -> None:
        if self.update_edge_path_closure_proto_bank(
            self.edge_path_closure_proto_negative,
            self.edge_path_closure_proto_negative_counts,
            slot_idx,
            feature,
            self.edge_path_closure_proto_wrong_lr,
        ):
            self.edge_path_closure_proto_wrong_updates[self.edge_path_slot_index(slot_idx)] += 1

    @staticmethod
    def bounded_edge_gap(value: float, other_value: float, scale: float = 0.25) -> float:
        return math.tanh((float(value) - float(other_value)) / max(float(scale), 1e-6))

    def edge_path_affinity_feature(
        self,
        support: float,
        closure: float,
        consistency: float,
        learned: float,
        max_support_other: float,
        max_closure_other: float,
        max_consistency_other: float,
        max_learned_other: float,
    ) -> np.ndarray:
        support_b = math.tanh(float(support))
        learned_b = math.tanh(float(learned))
        closure_b = float(np.clip(closure, 0.0, 1.0))
        consistency_b = float(np.clip(consistency, 0.0, 1.0))
        support_gap = self.bounded_edge_gap(support, max_support_other)
        closure_gap = self.bounded_edge_gap(closure, max_closure_other)
        consistency_gap = self.bounded_edge_gap(consistency, max_consistency_other)
        learned_gap = self.bounded_edge_gap(learned, max_learned_other)
        feature = np.array(
            [
                1.0,
                support_b,
                closure_b,
                consistency_b,
                learned_b,
                support_gap,
                closure_gap,
                consistency_gap,
                learned_gap,
                closure_gap - support_gap,
                closure_b * consistency_b,
                closure_b - support_b,
            ],
            dtype=np.float32,
        )
        return phase.normalize_vector(feature)

    def edge_path_affinity_gate(
        self,
        support: float,
        closure: float,
        consistency: float,
        learned: float,
        max_support_other: float,
        max_closure_other: float,
        max_consistency_other: float,
        max_learned_other: float,
    ) -> bool:
        if (
            self.edge_path_affinity_margin_gate <= 0.0
            and self.edge_path_affinity_learned_dominance <= 0.0
        ):
            return True
        self.edge_path_affinity_gate_checks += 1
        support_margin = float(support) - float(max_support_other)
        closure_margin = float(closure) - float(max_closure_other)
        consistency_margin = float(consistency) - float(max_consistency_other)
        learned_margin = float(learned) - float(max_learned_other)
        near = False
        if self.edge_path_affinity_margin_gate > 0.0:
            near = abs(support_margin) <= self.edge_path_affinity_margin_gate
        protect = self.edge_path_affinity_consistency_protect
        weak_structure = (
            float(closure) + protect < float(max_closure_other)
            or float(consistency) + protect < float(max_consistency_other)
            or (
                self.edge_path_affinity_margin_gate > 0.0
                and float(support) + self.edge_path_affinity_margin_gate < float(max_support_other)
            )
        )
        learned_dominant = (
            self.edge_path_affinity_learned_dominance > 0.0
            and learned_margin > 0.0
            and learned_margin
            >= self.edge_path_affinity_learned_dominance * max(support_margin, 0.0)
        )
        conflict = learned_dominant and weak_structure
        if near or conflict:
            self.edge_path_affinity_gate_passes += 1
            if near:
                self.edge_path_affinity_gate_near_passes += 1
            if conflict:
                self.edge_path_affinity_gate_conflict_passes += 1
            return True
        self.edge_path_affinity_gate_skips += 1
        return False

    def edge_path_affinity_bank_score(
        self,
        prototypes: np.ndarray | None,
        counts: np.ndarray | None,
        slot_idx: int,
        feature: np.ndarray,
    ) -> tuple[float, float]:
        if prototypes is None or counts is None:
            return 0.0, 0.0
        slot_idx = self.edge_path_slot_index(slot_idx)
        slot_counts = counts[slot_idx]
        active = slot_counts > 0.0
        if not np.any(active):
            return 0.0, 0.0
        dots = prototypes[slot_idx] @ feature
        dots = np.where(active, dots, -np.inf)
        bank_slot = int(np.argmax(dots))
        return float(max(dots[bank_slot], 0.0)), float(slot_counts[bank_slot])

    def edge_path_affinity_score(self, slot_idx: int, feature: np.ndarray) -> float:
        if self.edge_path_affinity_score_scale <= 0.0:
            return 0.0
        positive, positive_count = self.edge_path_affinity_bank_score(
            self.edge_path_affinity_positive,
            self.edge_path_affinity_positive_counts,
            slot_idx,
            feature,
        )
        negative, _ = self.edge_path_affinity_bank_score(
            self.edge_path_affinity_negative,
            self.edge_path_affinity_negative_counts,
            slot_idx,
            feature,
        )
        self.edge_path_affinity_score_checks += 1
        if positive_count < self.edge_path_affinity_min_count:
            self.edge_path_affinity_score_count_skips += 1
            return 0.0
        score = positive - negative
        if score > 0.0:
            self.edge_path_affinity_score_applied += 1
        return score

    def update_edge_path_affinity_bank(
        self,
        prototypes: np.ndarray | None,
        counts: np.ndarray | None,
        slot_idx: int,
        feature: np.ndarray,
        lr: float,
    ) -> bool:
        if prototypes is None or counts is None or lr <= 0.0:
            return False
        slot_idx = self.edge_path_slot_index(slot_idx)
        slot_counts = counts[slot_idx]
        empty = np.flatnonzero(slot_counts <= 0.0)
        if empty.size:
            bank_slot = int(empty[0])
            prototypes[slot_idx, bank_slot] = feature
            counts[slot_idx, bank_slot] = 1.0
            return True
        dots = prototypes[slot_idx] @ feature
        bank_slot = int(np.argmax(dots))
        prototypes[slot_idx, bank_slot] = phase.normalize_vector(
            (1.0 - lr) * prototypes[slot_idx, bank_slot] + lr * feature
        )
        counts[slot_idx, bank_slot] += 1.0
        return True

    def update_edge_path_affinity_positive(self, slot_idx: int, feature: np.ndarray) -> None:
        if self.update_edge_path_affinity_bank(
            self.edge_path_affinity_positive,
            self.edge_path_affinity_positive_counts,
            slot_idx,
            feature,
            self.edge_path_affinity_lr,
        ):
            self.edge_path_affinity_updates[self.edge_path_slot_index(slot_idx)] += 1

    def update_edge_path_affinity_negative(self, slot_idx: int, feature: np.ndarray) -> None:
        if self.update_edge_path_affinity_bank(
            self.edge_path_affinity_negative,
            self.edge_path_affinity_negative_counts,
            slot_idx,
            feature,
            self.edge_path_affinity_wrong_lr,
        ):
            self.edge_path_affinity_wrong_updates[self.edge_path_slot_index(slot_idx)] += 1

    def anchor_edge_bundle_map(self, prefix: list[int], anchor: int) -> dict[int, tuple[np.ndarray, float]]:
        accum: dict[int, np.ndarray] = {}
        support: dict[int, float] = {}
        if not prefix:
            return {}
        anchor = int(anchor)
        total = len(prefix)
        for pos, token in enumerate(prefix):
            if int(token) != anchor:
                continue
            left = max(0, pos - self.role_window)
            right = min(total, pos + self.role_window + 1)
            recency = self.role_recency_decay ** max(total - 1 - pos, 0)
            for neighbor_pos in range(left, right):
                if neighbor_pos == pos:
                    continue
                neighbor = int(prefix[neighbor_pos])
                rel = neighbor_pos - pos
                rel_idx = rel + self.role_window
                distance = abs(rel)
                locality = self.role_locality_decay ** max(distance - 1, 0)
                weight = float(recency * locality / math.sqrt(float(distance)))
                feature = self.event_feature(anchor, neighbor, rel_idx)
                if neighbor not in accum:
                    accum[neighbor] = np.zeros(self.state_dim, dtype=np.float32)
                    support[neighbor] = 0.0
                accum[neighbor] = accum[neighbor] + weight * feature
                support[neighbor] += weight * weight
        bundles: dict[int, tuple[np.ndarray, float]] = {}
        for neighbor, feature in accum.items():
            bundles[int(neighbor)] = (
                phase.normalize_vector(feature.astype(np.float32)),
                math.sqrt(max(float(support.get(int(neighbor), 0.0)), 0.0)),
            )
        return bundles

    def edge_path_candidates(
        self,
        tokens: list[int],
    ) -> list[tuple[float, int, int, int, np.ndarray, np.ndarray, np.ndarray]]:
        prefix, query = self.prefix_and_query(tokens)
        anchors = self.ordered_query_anchors(prefix, query, limit=2)
        if len(anchors) < 2:
            return []
        source, destination = int(anchors[0]), int(anchors[-1])
        source_edges = self.anchor_edge_bundle_map(prefix, source)
        destination_edges = self.anchor_edge_bundle_map(prefix, destination)
        shared = [
            token
            for token in source_edges
            if token in destination_edges and token not in {source, destination}
        ]
        candidates: list[tuple[float, int, int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for mid in shared:
            source_feature, source_strength = source_edges[int(mid)]
            destination_feature, destination_strength = destination_edges[int(mid)]
            support = math.sqrt(max(float(source_strength * destination_strength), 0.0))
            source_state = phase.normalize_vector(source_feature * self.token_codes[int(mid)])
            destination_state = phase.normalize_vector(destination_feature * self.token_codes[int(mid)])
            path_feature = phase.normalize_vector(
                source_feature * destination_feature * self.token_codes[int(mid)]
            )
            candidates.append(
                (
                    support,
                    int(source),
                    int(mid),
                    int(destination),
                    source_state,
                    destination_state,
                    path_feature,
                )
            )
        candidates.sort(key=lambda item: (-float(item[0]), int(item[2])))
        return candidates

    def edge_path_cleanup_score(self, slot_idx: int, mid: int, path_feature: np.ndarray) -> float:
        counts = self.edge_path_cleanup_counts[int(slot_idx), int(mid)]
        active = counts > 0.0
        if not np.any(active):
            return 0.0
        dots = self.edge_path_cleanup_prototypes[int(slot_idx), int(mid)] @ path_feature
        dots = np.where(active, dots, -np.inf)
        return float(np.max(dots))

    def edge_path_runner_arbiter_feature(
        self,
        selected_mid: int,
        selected_feature: np.ndarray,
        runner_mid: int,
        runner_feature: np.ndarray,
        selected_score: float | None = None,
        selected_support: float | None = None,
        selected_learned: float | None = None,
        selected_consistency: float | None = None,
        runner_score: float | None = None,
        runner_support: float | None = None,
        runner_learned: float | None = None,
        runner_consistency: float | None = None,
    ) -> np.ndarray:
        selected_mid = int(selected_mid)
        runner_mid = int(runner_mid)
        feature = (
            runner_feature
            + 0.50 * (runner_feature - selected_feature)
            + 0.25 * (runner_feature * selected_feature)
            + 0.25 * (self.token_codes[runner_mid] - self.token_codes[selected_mid])
        )
        if (
            self.edge_path_runner_arbiter_feature_mode == "rich_gaps"
            and self.edge_path_runner_arbiter_gap_codes is not None
            and self.edge_path_runner_arbiter_gap_scale > 0.0
        ):
            gap_pairs = (
                (runner_score, selected_score),
                (runner_support, selected_support),
                (runner_learned, selected_learned),
                (runner_consistency, selected_consistency),
            )
            gap_feature = np.zeros(self.state_dim, dtype=np.float32)
            for idx, (runner_value, selected_value) in enumerate(gap_pairs):
                if runner_value is None or selected_value is None:
                    continue
                gap_value = math.tanh((float(runner_value) - float(selected_value)) / 0.25)
                gap_feature = gap_feature + gap_value * self.edge_path_runner_arbiter_gap_codes[idx]
            feature = feature + self.edge_path_runner_arbiter_gap_scale * gap_feature
        return phase.normalize_vector(feature.astype(np.float32))

    def edge_path_runner_arbiter_positive_score(
        self,
        slot_idx: int,
        pair_feature: np.ndarray,
    ) -> tuple[float, float]:
        slot_idx = self.edge_path_slot_index(slot_idx)
        counts = self.edge_path_runner_arbiter_counts[slot_idx]
        active = counts > 0.0
        if not np.any(active):
            return 0.0, 0.0
        dots = self.edge_path_runner_arbiter_prototypes[slot_idx] @ pair_feature
        dots = np.where(active, dots, -np.inf)
        bank_slot = int(np.argmax(dots))
        return float(dots[bank_slot]), float(counts[bank_slot])

    def edge_path_runner_arbiter_negative_score(self, slot_idx: int, pair_feature: np.ndarray) -> float:
        if (
            self.edge_path_runner_arbiter_negative_prototypes is None
            or self.edge_path_runner_arbiter_negative_counts is None
        ):
            return 0.0
        slot_idx = self.edge_path_slot_index(slot_idx)
        counts = self.edge_path_runner_arbiter_negative_counts[slot_idx]
        active = counts > 0.0
        if not np.any(active):
            return 0.0
        dots = self.edge_path_runner_arbiter_negative_prototypes[slot_idx] @ pair_feature
        dots = np.where(active, dots, -np.inf)
        return float(max(np.max(dots), 0.0))

    def edge_path_runner_arbiter_score(
        self,
        slot_idx: int,
        pair_feature: np.ndarray,
    ) -> tuple[float, float, float]:
        positive_score, positive_count = self.edge_path_runner_arbiter_positive_score(
            slot_idx,
            pair_feature,
        )
        negative_score = self.edge_path_runner_arbiter_negative_score(slot_idx, pair_feature)
        return positive_score - negative_score, positive_count, negative_score

    def update_edge_path_runner_arbiter_target(self, slot_idx: int, pair_feature: np.ndarray) -> None:
        if self.edge_path_runner_arbiter_lr <= 0.0:
            return
        slot_idx = self.edge_path_slot_index(slot_idx)
        counts = self.edge_path_runner_arbiter_counts[slot_idx]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            bank_slot = int(empty[0])
            self.edge_path_runner_arbiter_prototypes[slot_idx, bank_slot] = pair_feature
            self.edge_path_runner_arbiter_counts[slot_idx, bank_slot] = 1.0
            self.edge_path_runner_arbiter_updates[slot_idx] += 1
            return
        dots = self.edge_path_runner_arbiter_prototypes[slot_idx] @ pair_feature
        bank_slot = int(np.argmax(dots))
        self.edge_path_runner_arbiter_prototypes[slot_idx, bank_slot] = phase.normalize_vector(
            (1.0 - self.edge_path_runner_arbiter_lr)
            * self.edge_path_runner_arbiter_prototypes[slot_idx, bank_slot]
            + self.edge_path_runner_arbiter_lr * pair_feature
        )
        self.edge_path_runner_arbiter_counts[slot_idx, bank_slot] += 1.0
        self.edge_path_runner_arbiter_updates[slot_idx] += 1

    def update_edge_path_runner_arbiter_negative(self, slot_idx: int, pair_feature: np.ndarray) -> None:
        if (
            self.edge_path_runner_arbiter_wrong_lr <= 0.0
            or self.edge_path_runner_arbiter_negative_prototypes is None
            or self.edge_path_runner_arbiter_negative_counts is None
        ):
            return
        slot_idx = self.edge_path_slot_index(slot_idx)
        counts = self.edge_path_runner_arbiter_negative_counts[slot_idx]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            bank_slot = int(empty[0])
            self.edge_path_runner_arbiter_negative_prototypes[slot_idx, bank_slot] = pair_feature
            self.edge_path_runner_arbiter_negative_counts[slot_idx, bank_slot] = 1.0
            self.edge_path_runner_arbiter_wrong_updates[slot_idx] += 1
            return
        dots = self.edge_path_runner_arbiter_negative_prototypes[slot_idx] @ pair_feature
        bank_slot = int(np.argmax(dots))
        self.edge_path_runner_arbiter_negative_prototypes[slot_idx, bank_slot] = phase.normalize_vector(
            (1.0 - self.edge_path_runner_arbiter_wrong_lr)
            * self.edge_path_runner_arbiter_negative_prototypes[slot_idx, bank_slot]
            + self.edge_path_runner_arbiter_wrong_lr * pair_feature
        )
        self.edge_path_runner_arbiter_negative_counts[slot_idx, bank_slot] += 1.0
        self.edge_path_runner_arbiter_wrong_updates[slot_idx] += 1

    def suppress_edge_path_runner_arbiter(self, slot_idx: int, pair_feature: np.ndarray) -> None:
        if self.edge_path_runner_arbiter_wrong_lr <= 0.0:
            return
        if self.edge_path_runner_arbiter_negative_mode == "separate":
            self.update_edge_path_runner_arbiter_negative(slot_idx, pair_feature)
            return
        slot_idx = self.edge_path_slot_index(slot_idx)
        counts = self.edge_path_runner_arbiter_counts[slot_idx]
        active = counts > 0.0
        if not np.any(active):
            return
        dots = self.edge_path_runner_arbiter_prototypes[slot_idx] @ pair_feature
        dots = np.where(active, dots, -np.inf)
        bank_slot = int(np.argmax(dots))
        if float(dots[bank_slot]) <= 0.0:
            return
        self.edge_path_runner_arbiter_prototypes[slot_idx, bank_slot] = phase.normalize_vector(
            self.edge_path_runner_arbiter_prototypes[slot_idx, bank_slot]
            - self.edge_path_runner_arbiter_wrong_lr * pair_feature
        )
        self.edge_path_runner_arbiter_wrong_updates[slot_idx] += 1

    def apply_edge_path_runner_arbiter(
        self,
        slot_idx: int,
        ranked: list[tuple[float, float, float, float, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]],
    ) -> list[tuple[float, float, float, float, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]]:
        if self.edge_path_runner_arbiter_score_scale <= 0.0 or len(ranked) < 2:
            return ranked
        if not np.any(self.edge_path_runner_arbiter_counts[self.edge_path_slot_index(slot_idx)] > 0.0):
            return ranked
        selected = ranked[0]
        runner = ranked[1]
        pair_feature = self.edge_path_runner_arbiter_feature(
            int(selected[4]),
            selected[10],
            int(runner[4]),
            runner[10],
            selected_score=float(selected[0]),
            selected_support=float(selected[1]),
            selected_learned=float(selected[3]),
            selected_consistency=float(selected[2]),
            runner_score=float(runner[0]),
            runner_support=float(runner[1]),
            runner_learned=float(runner[3]),
            runner_consistency=float(runner[2]),
        )
        arbiter_score, arbiter_count, _ = self.edge_path_runner_arbiter_score(slot_idx, pair_feature)
        self.edge_path_runner_arbiter_score_checks += 1
        if arbiter_count < self.edge_path_runner_arbiter_min_count:
            self.edge_path_runner_arbiter_score_count_skips += 1
            return ranked
        if arbiter_score <= self.edge_path_runner_arbiter_margin:
            return ranked
        adjusted = list(ranked)
        runner_item = list(adjusted[1])
        runner_item[0] = float(runner_item[0]) + self.edge_path_runner_arbiter_score_scale * (
            arbiter_score - self.edge_path_runner_arbiter_margin
        )
        adjusted[1] = tuple(runner_item)  # type: ignore[list-item]
        adjusted.sort(key=lambda item: (-float(item[0]), -float(item[1]), int(item[4])))
        self.edge_path_runner_arbiter_score_applied += 1
        return adjusted

    def edge_path_transient_trace_index(
        self,
        mid: int,
        path_feature: np.ndarray,
        source_anchor: int | None = None,
        destination_anchor: int | None = None,
    ) -> int:
        if self.edge_path_transient_inhibit_key == "mid":
            return int(mid)
        if self.edge_path_transient_inhibit_key == "anchor_path":
            source = int(source_anchor) if source_anchor is not None else 0
            destination = int(destination_anchor) if destination_anchor is not None else 0
            h = 2166136261
            for value in (source, int(mid), destination):
                h ^= int(value) & 0xFFFFFFFF
                h = (h * 16777619) & 0xFFFFFFFF
            return int(h % self.edge_path_transient_inhibit_hash_size)
        values = np.asarray(path_feature, dtype=np.float32)
        if values.size == 0:
            return int(mid) % self.edge_path_transient_inhibit_hash_size
        top_k = min(4, values.size)
        indices = np.argpartition(np.abs(values), -top_k)[-top_k:]
        indices = indices[np.argsort(-np.abs(values[indices]))]
        h = (2166136261 ^ int(mid)) & 0xFFFFFFFF
        for idx in indices:
            sign_bit = 1 if float(values[int(idx)]) >= 0.0 else 0
            h ^= ((int(idx) + 1) * 16777619) ^ sign_bit
            h = (h * 16777619) & 0xFFFFFFFF
        return int(h % self.edge_path_transient_inhibit_hash_size)

    def edge_path_homeostasis_gate(
        self,
        support: float,
        closure: float,
        consistency: float,
        learned: float,
        max_support_other: float,
        max_closure_other: float,
        max_consistency_other: float,
        max_learned_other: float,
    ) -> bool:
        if self.edge_path_homeostasis_learned_dominance <= 0.0:
            return True
        self.edge_path_homeostasis_gate_checks += 1
        support_margin = float(support) - float(max_support_other)
        closure_margin = float(closure) - float(max_closure_other)
        consistency_margin = float(consistency) - float(max_consistency_other)
        learned_margin = float(learned) - float(max_learned_other)
        learned_dominant = (
            learned_margin > 0.0
            and learned_margin
            >= self.edge_path_homeostasis_learned_dominance * max(support_margin, 0.0)
        )
        weak_structure = (
            support_margin <= self.edge_path_homeostasis_structure_margin
            or closure_margin <= self.edge_path_homeostasis_structure_margin
            or consistency_margin <= self.edge_path_homeostasis_structure_margin
        )
        if learned_dominant and weak_structure:
            self.edge_path_homeostasis_gate_passes += 1
            return True
        self.edge_path_homeostasis_gate_skips += 1
        return False

    def edge_path_homeostasis_hard_gate_enabled(self) -> bool:
        return (
            self.edge_path_homeostasis_learned_dominance > 0.0
            and self.edge_path_homeostasis_soft_mod_scale <= 0.0
        )

    def edge_path_homeostasis_soft_multiplier(
        self,
        support: float,
        closure: float,
        consistency: float,
        learned: float,
        max_support_other: float,
        max_closure_other: float,
        max_consistency_other: float,
        max_learned_other: float,
    ) -> float:
        if (
            self.edge_path_homeostasis_soft_mod_scale <= 0.0
            or self.edge_path_homeostasis_soft_mod_floor >= 1.0
        ):
            return 1.0
        support_margin = float(support) - float(max_support_other)
        closure_margin = float(closure) - float(max_closure_other)
        consistency_margin = float(consistency) - float(max_consistency_other)
        learned_margin = float(learned) - float(max_learned_other)
        structure_deficit = max(
            0.0,
            self.edge_path_homeostasis_structure_margin - support_margin,
            self.edge_path_homeostasis_structure_margin - closure_margin,
            self.edge_path_homeostasis_structure_margin - consistency_margin,
        )
        if self.edge_path_homeostasis_learned_dominance > 0.0:
            learned_excess = max(
                0.0,
                learned_margin
                - self.edge_path_homeostasis_learned_dominance * max(support_margin, 0.0),
            )
        else:
            learned_excess = max(0.0, learned_margin)
        raw_pressure = self.edge_path_homeostasis_soft_mod_scale * (
            structure_deficit + learned_excess
        )
        pressure = raw_pressure / (1.0 + raw_pressure) if raw_pressure > 0.0 else 0.0
        multiplier = self.edge_path_homeostasis_soft_mod_floor + (
            1.0 - self.edge_path_homeostasis_soft_mod_floor
        ) * pressure
        multiplier = float(np.clip(multiplier, 0.0, 1.0))
        self.edge_path_homeostasis_soft_mod_checks += 1
        self.edge_path_homeostasis_soft_mod_sum += multiplier
        self.edge_path_homeostasis_soft_mod_min = min(
            self.edge_path_homeostasis_soft_mod_min,
            multiplier,
        )
        self.edge_path_homeostasis_soft_mod_max = max(
            self.edge_path_homeostasis_soft_mod_max,
            multiplier,
        )
        return multiplier

    def edge_path_homeostasis_effective_trace(self, trace_value: float) -> float:
        value = float(np.clip(trace_value, 0.0, 1.0))
        if (
            self.edge_path_homeostasis_trace_threshold <= 0.0
            and abs(self.edge_path_homeostasis_trace_gain - 1.0) <= 1e-12
        ):
            return value
        if self.edge_path_homeostasis_trace_threshold > 0.0:
            denom = max(1.0 - self.edge_path_homeostasis_trace_threshold, 1e-6)
            effective = max(0.0, value - self.edge_path_homeostasis_trace_threshold) / denom
        else:
            effective = value
        effective = float(np.clip(effective * self.edge_path_homeostasis_trace_gain, 0.0, 1.0))
        self.edge_path_homeostasis_trace_mod_checks += 1
        self.edge_path_homeostasis_trace_mod_raw_sum += value
        self.edge_path_homeostasis_trace_mod_effective_sum += effective
        if effective > 0.0:
            self.edge_path_homeostasis_trace_mod_active += 1
        return effective

    def edge_path_homeostasis_penalty(
        self,
        slot_idx: int,
        trace_idx: int,
        support: float,
        closure: float,
        consistency: float,
        learned: float,
        max_support_other: float,
        max_closure_other: float,
        max_consistency_other: float,
        max_learned_other: float,
    ) -> float:
        if (
            self.edge_path_homeostasis_trace is None
            or self.edge_path_homeostasis_scale <= 0.0
            or int(slot_idx) < self.edge_path_homeostasis_min_slot
        ):
            return 0.0
        if self.edge_path_homeostasis_hard_gate_enabled() and not self.edge_path_homeostasis_gate(
            support,
            closure,
            consistency,
            learned,
            max_support_other,
            max_closure_other,
            max_consistency_other,
            max_learned_other,
        ):
            return 0.0
        multiplier = self.edge_path_homeostasis_soft_multiplier(
            support,
            closure,
            consistency,
            learned,
            max_support_other,
            max_closure_other,
            max_consistency_other,
            max_learned_other,
        )
        trace_value = float(
            self.edge_path_homeostasis_trace[self.edge_path_slot_index(slot_idx), int(trace_idx)]
        )
        return (
            self.edge_path_homeostasis_scale
            * self.edge_path_homeostasis_effective_trace(trace_value)
            * multiplier
        )

    def update_edge_path_homeostasis(self, slot_idx: int, trace_idx: int, weight: float = 1.0) -> None:
        if (
            self.edge_path_homeostasis_trace is None
            or self.edge_path_homeostasis_lr <= 0.0
            or int(slot_idx) < self.edge_path_homeostasis_min_slot
        ):
            return
        slot_idx = self.edge_path_slot_index(slot_idx)
        if self.edge_path_homeostasis_decay < 1.0:
            self.edge_path_homeostasis_trace[slot_idx] *= self.edge_path_homeostasis_decay
        delta = self.edge_path_homeostasis_lr * float(max(weight, 0.0))
        if delta <= 0.0:
            return
        trace_idx = int(trace_idx)
        current = float(self.edge_path_homeostasis_trace[slot_idx, trace_idx])
        self.edge_path_homeostasis_trace[slot_idx, trace_idx] = min(current + delta, 1.0)
        self.edge_path_homeostasis_updates[slot_idx] += 1

    def edge_path_ranked_candidates(
        self,
        tokens: list[int],
        slot: int,
    ) -> list[tuple[float, float, float, float, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]]:
        slot_idx = self.edge_path_slot_index(slot)
        raw_candidates: list[
            tuple[float, float, float, float, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for (
            support,
            source_anchor,
            mid,
            destination_anchor,
            source_state,
            destination_state,
            path_feature,
        ) in self.edge_path_candidates(tokens):
            learned = self.edge_path_cleanup_score(slot_idx, mid, path_feature)
            consistency = max(float(source_state @ destination_state), 0.0)
            trace_idx = self.edge_path_transient_trace_index(
                int(mid),
                path_feature,
                int(source_anchor),
                int(destination_anchor),
            )
            closure = self.edge_path_raw_closure_score(source_state, destination_state, path_feature)
            raw_candidates.append(
                (
                    float(support),
                    float(consistency),
                    float(learned),
                    float(closure),
                    int(trace_idx),
                    int(source_anchor),
                    int(mid),
                    int(destination_anchor),
                    source_state,
                    destination_state,
                    path_feature,
                )
            )
        if not raw_candidates:
            return []
        support_values = [float(item[0]) for item in raw_candidates]
        consistency_values = [float(item[1]) for item in raw_candidates]
        learned_values = [float(item[2]) for item in raw_candidates]
        closure_values = [float(item[3]) for item in raw_candidates]

        def max_other(values: list[float], idx: int) -> float:
            if len(values) <= 1:
                return 0.0
            return max(float(value) for j, value in enumerate(values) if j != idx)

        ranked: list[
            tuple[float, float, float, float, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for idx, (
            support,
            consistency,
            learned,
            closure,
            trace_idx,
            source_anchor,
            mid,
            destination_anchor,
            source_state,
            destination_state,
            path_feature,
        ) in enumerate(raw_candidates):
            closure_proto = self.edge_path_closure_proto_score(
                slot_idx,
                source_state,
                destination_state,
                path_feature,
            )
            max_support_other = max_other(support_values, idx)
            max_closure_other = max_other(closure_values, idx)
            max_consistency_other = max_other(consistency_values, idx)
            max_learned_other = max_other(learned_values, idx)
            affinity = 0.0
            if self.edge_path_affinity_gate(
                float(support),
                float(closure),
                float(consistency),
                float(learned),
                max_support_other,
                max_closure_other,
                max_consistency_other,
                max_learned_other,
            ):
                affinity_feature = self.edge_path_affinity_feature(
                    float(support),
                    float(closure),
                    float(consistency),
                    float(learned),
                    max_support_other,
                    max_closure_other,
                    max_consistency_other,
                    max_learned_other,
                )
                affinity = self.edge_path_affinity_score(slot_idx, affinity_feature)
            inhibit = 0.0
            if self.edge_path_transient_inhibit_scale > 0.0:
                inhibit = self.edge_path_transient_inhibit_scale * float(
                    self.edge_path_transient_inhibit_trace[slot_idx, int(trace_idx)]
                )
            boost = 0.0
            if self.edge_path_transient_boost_scale > 0.0:
                boost = self.edge_path_transient_boost_scale * float(
                    self.edge_path_transient_boost_trace[slot_idx, int(trace_idx)]
                )
            homeostasis = self.edge_path_homeostasis_penalty(
                slot_idx,
                int(trace_idx),
                float(support),
                float(closure),
                float(consistency),
                float(learned),
                max_support_other,
                max_closure_other,
                max_consistency_other,
                max_learned_other,
            )
            score = float(
                support
                + self.edge_path_cleanup_score_scale * learned
                + self.edge_path_closure_score_scale * closure
                + self.edge_path_closure_proto_score_scale * closure_proto
                + self.edge_path_affinity_score_scale * affinity
                - inhibit
                - homeostasis
                + boost
            )
            ranked.append(
                (
                    score,
                    float(support),
                    float(consistency),
                    float(learned),
                    int(mid),
                    int(trace_idx),
                    int(source_anchor),
                    int(destination_anchor),
                    source_state,
                    destination_state,
                    path_feature,
                )
            )
        ranked.sort(key=lambda item: (-float(item[0]), -float(item[1]), int(item[4])))
        ranked = self.apply_edge_path_runner_arbiter(slot_idx, ranked)
        return ranked

    def edge_path_soft_ranked_candidates(
        self,
        tokens: list[int],
        slot: int,
    ) -> list[tuple[float, float, float, float, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]]:
        slot_idx = self.edge_path_slot_index(slot)
        raw_candidates: list[
            tuple[float, float, float, float, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for (
            support,
            source_anchor,
            mid,
            destination_anchor,
            source_state,
            destination_state,
            path_feature,
        ) in self.edge_path_candidates(tokens):
            learned = self.edge_path_cleanup_score(slot_idx, mid, path_feature)
            consistency = max(float(source_state @ destination_state), 0.0)
            trace_idx = self.edge_path_transient_trace_index(
                int(mid),
                path_feature,
                int(source_anchor),
                int(destination_anchor),
            )
            closure = self.edge_path_raw_closure_score(source_state, destination_state, path_feature)
            raw_candidates.append(
                (
                    float(support),
                    float(consistency),
                    float(learned),
                    float(closure),
                    int(trace_idx),
                    int(source_anchor),
                    int(mid),
                    int(destination_anchor),
                    source_state,
                    destination_state,
                    path_feature,
                )
            )
        if not raw_candidates:
            return []
        support_values = [float(item[0]) for item in raw_candidates]
        consistency_values = [float(item[1]) for item in raw_candidates]
        learned_values = [float(item[2]) for item in raw_candidates]
        closure_values = [float(item[3]) for item in raw_candidates]

        def max_other(values: list[float], idx: int) -> float:
            if len(values) <= 1:
                return 0.0
            return max(float(value) for j, value in enumerate(values) if j != idx)

        ranked: list[
            tuple[float, float, float, float, int, int, int, int, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for idx, (
            support,
            consistency,
            learned,
            closure,
            trace_idx,
            source_anchor,
            mid,
            destination_anchor,
            source_state,
            destination_state,
            path_feature,
        ) in enumerate(raw_candidates):
            closure_proto = self.edge_path_closure_proto_score(
                slot_idx,
                source_state,
                destination_state,
                path_feature,
            )
            max_support_other = max_other(support_values, idx)
            max_closure_other = max_other(closure_values, idx)
            max_consistency_other = max_other(consistency_values, idx)
            max_learned_other = max_other(learned_values, idx)
            affinity = 0.0
            if self.edge_path_affinity_gate(
                float(support),
                float(closure),
                float(consistency),
                float(learned),
                max_support_other,
                max_closure_other,
                max_consistency_other,
                max_learned_other,
            ):
                affinity_feature = self.edge_path_affinity_feature(
                    float(support),
                    float(closure),
                    float(consistency),
                    float(learned),
                    max_support_other,
                    max_closure_other,
                    max_consistency_other,
                    max_learned_other,
                )
                affinity = self.edge_path_affinity_score(slot_idx, affinity_feature)
            inhibit = 0.0
            if self.edge_path_transient_inhibit_scale > 0.0:
                inhibit = self.edge_path_transient_inhibit_scale * float(
                    self.edge_path_transient_inhibit_trace[slot_idx, int(trace_idx)]
                )
            boost = 0.0
            if self.edge_path_transient_boost_scale > 0.0:
                boost = self.edge_path_transient_boost_scale * float(
                    self.edge_path_transient_boost_trace[slot_idx, int(trace_idx)]
                )
            homeostasis = self.edge_path_homeostasis_penalty(
                slot_idx,
                int(trace_idx),
                float(support),
                float(closure),
                float(consistency),
                float(learned),
                max_support_other,
                max_closure_other,
                max_consistency_other,
                max_learned_other,
            )
            score = (
                float(support)
                + self.edge_path_soft_consistency_scale * consistency
                + self.edge_path_soft_learned_scale * learned
                + self.edge_path_closure_score_scale * closure
                + self.edge_path_closure_proto_score_scale * closure_proto
                + self.edge_path_affinity_score_scale * affinity
                - inhibit
                - homeostasis
                + boost
            )
            ranked.append(
                (
                    float(score),
                    float(support),
                    float(consistency),
                    float(learned),
                    int(mid),
                    int(trace_idx),
                    int(source_anchor),
                    int(destination_anchor),
                    source_state,
                    destination_state,
                    path_feature,
                )
            )
        ranked.sort(key=lambda item: (-float(item[0]), -float(item[1]), int(item[4])))
        ranked = self.apply_edge_path_runner_arbiter(slot_idx, ranked)
        return ranked

    def edge_path_wta_state(self, tokens: list[int], slot: int) -> np.ndarray:
        slot_idx = self.edge_path_slot_index(slot)
        ranked = self.edge_path_ranked_candidates(tokens, slot_idx)
        self.edge_path_cleanup_checks += 1
        self.edge_path_cleanup_candidates += len(ranked)
        self.last_edge_path_candidate_count[slot_idx] = len(ranked)
        self.last_edge_path_selection.pop(slot_idx, None)
        self.last_edge_path_runner_up.pop(slot_idx, None)
        self.last_edge_path_trace_index.pop(slot_idx, None)
        self.last_edge_path_runner_trace_index.pop(slot_idx, None)
        self.last_edge_path_top_candidates.pop(slot_idx, None)
        self.last_edge_path_selection_states.pop(slot_idx, None)
        self.last_edge_path_runner_states.pop(slot_idx, None)
        direct_slot = self.edge_path_direct_slot_index(slot)
        self.last_edge_path_direct_feature.pop(direct_slot, None)
        self.last_edge_path_direct_feature_bundle.pop(direct_slot, None)
        if not ranked:
            return np.zeros(self.state_dim, dtype=np.float32)
        (
            score,
            support,
            consistency,
            learned,
            mid,
            trace_idx,
            source_anchor,
            destination_anchor,
            source_state,
            destination_state,
            path_feature,
        ) = ranked[0]
        self.last_edge_path_trace_index[slot_idx] = int(trace_idx)
        self.last_edge_path_selection[slot_idx] = (
            int(mid),
            path_feature.astype(np.float32, copy=True),
            float(score),
            float(support),
            float(learned),
            float(consistency),
            int(source_anchor),
            int(destination_anchor),
        )
        self.last_edge_path_selection_states[slot_idx] = (
            source_state.astype(np.float32, copy=True),
            destination_state.astype(np.float32, copy=True),
        )
        self.last_edge_path_direct_feature_bundle[direct_slot] = self.edge_path_direct_feature_bundle(
            int(slot),
            source_state,
            destination_state,
            path_feature,
        )
        self.edge_path_cleanup_wins[slot_idx] += 1
        if len(ranked) > 1:
            (
                r_score,
                r_support,
                r_consistency,
                r_learned,
                r_mid,
                r_trace_idx,
                r_source_anchor,
                r_destination_anchor,
                _,
                _,
                r_path_feature,
            ) = ranked[1]
            self.last_edge_path_runner_trace_index[slot_idx] = int(r_trace_idx)
            self.last_edge_path_runner_up[slot_idx] = (
                int(r_mid),
                r_path_feature.astype(np.float32, copy=True),
                float(r_score),
                float(r_support),
                float(r_learned),
                float(r_consistency),
                int(r_source_anchor),
                int(r_destination_anchor),
            )
            self.last_edge_path_runner_states[slot_idx] = (
                ranked[1][8].astype(np.float32, copy=True),
                ranked[1][9].astype(np.float32, copy=True),
            )
        self.last_edge_path_top_candidates[slot_idx] = [
            (
                int(mid),
                path_feature.astype(np.float32, copy=True),
                float(score),
                float(support),
                float(learned),
                1.0,
            )
        ]
        top_k = min(max(self.edge_path_cleanup_top_k, 1), len(ranked))
        max_score = float(ranked[0][0])
        state = np.zeros(self.state_dim, dtype=np.float32)
        weight_sum = 0.0
        for cand_score, cand_support, _, _, _, _, _, _, cand_source, cand_destination, _ in ranked[:top_k]:
            inhibited = max_score - float(cand_score)
            weight = float(cand_support + max(float(cand_score), 0.0))
            if self.edge_path_cleanup_inhibit > 0.0:
                weight = max(weight - self.edge_path_cleanup_inhibit * inhibited, 0.0)
            if weight <= 0.0:
                continue
            state += weight * (cand_source if int(slot) <= 0 else cand_destination)
            weight_sum += weight
        if weight_sum <= 0.0:
            return source_state if int(slot) <= 0 else destination_state
        return phase.normalize_vector((state / weight_sum).astype(np.float32))

    def edge_path_soft_state(self, tokens: list[int], slot: int) -> np.ndarray:
        slot_idx = self.edge_path_slot_index(slot)
        ranked = self.edge_path_soft_ranked_candidates(tokens, slot_idx)
        self.edge_path_cleanup_checks += 1
        self.edge_path_cleanup_candidates += len(ranked)
        self.last_edge_path_candidate_count[slot_idx] = len(ranked)
        self.last_edge_path_selection.pop(slot_idx, None)
        self.last_edge_path_runner_up.pop(slot_idx, None)
        self.last_edge_path_trace_index.pop(slot_idx, None)
        self.last_edge_path_runner_trace_index.pop(slot_idx, None)
        self.last_edge_path_top_candidates.pop(slot_idx, None)
        self.last_edge_path_selection_states.pop(slot_idx, None)
        self.last_edge_path_runner_states.pop(slot_idx, None)
        direct_slot = self.edge_path_direct_slot_index(slot)
        self.last_edge_path_direct_feature_bundle.pop(direct_slot, None)
        if not ranked:
            return np.zeros(self.state_dim, dtype=np.float32)
        best = ranked[0]
        self.last_edge_path_trace_index[slot_idx] = int(best[5])
        self.last_edge_path_selection[slot_idx] = (
            int(best[4]),
            best[10].astype(np.float32, copy=True),
            float(best[0]),
            float(best[1]),
            float(best[3]),
            float(best[2]),
            int(best[6]),
            int(best[7]),
        )
        self.last_edge_path_selection_states[slot_idx] = (
            best[8].astype(np.float32, copy=True),
            best[9].astype(np.float32, copy=True),
        )
        self.last_edge_path_direct_feature_bundle[direct_slot] = self.edge_path_direct_feature_bundle(
            int(slot),
            best[8],
            best[9],
            best[10],
        )
        self.edge_path_cleanup_wins[slot_idx] += 1
        if len(ranked) > 1:
            runner = ranked[1]
            self.last_edge_path_runner_trace_index[slot_idx] = int(runner[5])
            self.last_edge_path_runner_up[slot_idx] = (
                int(runner[4]),
                runner[10].astype(np.float32, copy=True),
                float(runner[0]),
                float(runner[1]),
                float(runner[3]),
                float(runner[2]),
                int(runner[6]),
                int(runner[7]),
            )
            self.last_edge_path_runner_states[slot_idx] = (
                runner[8].astype(np.float32, copy=True),
                runner[9].astype(np.float32, copy=True),
            )
        top_k = min(max(self.edge_path_soft_top_k, 1), len(ranked))
        top = ranked[:top_k]
        max_score = max(float(item[0]) for item in top)
        state = np.zeros(self.state_dim, dtype=np.float32)
        path_state = np.zeros(self.state_dim, dtype=np.float32)
        weight_sum = 0.0
        weighted_candidates: list[tuple[int, np.ndarray, float, float, float, float]] = []
        for score, support, _, learned, mid, _, _, _, source_state, destination_state, path_feature in top:
            soft = math.exp((float(score) - max_score) / self.edge_path_soft_temperature)
            weight = max(float(support), 1e-6) * soft
            state += weight * (source_state if int(slot) <= 0 else destination_state)
            path_state += weight * path_feature
            weight_sum += weight
            weighted_candidates.append(
                (
                    int(mid),
                    path_feature.astype(np.float32, copy=True),
                    float(score),
                    float(support),
                    float(learned),
                    float(weight),
                )
            )
        if weight_sum <= 0.0:
            self.last_edge_path_direct_feature[self.edge_path_direct_slot_index(slot)] = top[0][10].astype(
                np.float32,
                copy=True,
            )
            self.last_edge_path_top_candidates[slot_idx] = [
                (
                    int(top[0][4]),
                    top[0][10].astype(np.float32, copy=True),
                    float(top[0][0]),
                    float(top[0][1]),
                    float(top[0][3]),
                    1.0,
                )
            ]
            return top[0][8] if int(slot) <= 0 else top[0][9]
        self.last_edge_path_top_candidates[slot_idx] = [
            (mid, feature, score, support, learned, weight / weight_sum)
            for mid, feature, score, support, learned, weight in weighted_candidates
        ]
        self.last_edge_path_direct_feature[self.edge_path_direct_slot_index(slot)] = phase.normalize_vector(
            (path_state / weight_sum).astype(np.float32)
        )
        return phase.normalize_vector((state / weight_sum).astype(np.float32))

    def edge_path_last_candidate_answer_feature(
        self,
        slot: int,
        candidate: str = "runner",
    ) -> np.ndarray | None:
        slot_idx = self.edge_path_slot_index(slot)
        states = (
            self.last_edge_path_runner_states.get(slot_idx)
            if str(candidate) == "runner"
            else self.last_edge_path_selection_states.get(slot_idx)
        )
        if states is None:
            return None
        side_state = states[0] if int(slot) <= 0 else states[1]
        return phase.normalize_vector(
            np.concatenate([self.last_base_feature, side_state]).astype(np.float32)
        )

    def edge_path_last_candidate_metrics(self, slot: int) -> dict[str, Any]:
        slot_idx = self.edge_path_slot_index(slot)
        row: dict[str, Any] = {
            "edge_path_candidate_count": int(self.last_edge_path_candidate_count.get(slot_idx, 0)),
            "edge_path_cleanup_credit_mode": self.edge_path_cleanup_credit_mode,
            "edge_path_margin_gate": float(self.edge_path_margin_gate),
            "edge_path_margin_min_scale": float(self.edge_path_margin_min_scale),
            "edge_path_margin_alt_scale": float(self.edge_path_margin_alt_scale),
            "edge_path_margin_learned_dominance": float(self.edge_path_margin_learned_dominance),
            "edge_path_margin_escape_scale": float(self.edge_path_margin_escape_scale),
            "edge_path_transient_inhibit_scale": float(self.edge_path_transient_inhibit_scale),
            "edge_path_transient_inhibit_key": self.edge_path_transient_inhibit_key,
            "edge_path_soft_learned_scale": float(self.edge_path_soft_learned_scale),
        }
        selection = self.last_edge_path_selection.get(slot_idx)
        runner = self.last_edge_path_runner_up.get(slot_idx)
        for prefix, item in [
            ("edge_path_selected", selection),
            ("edge_path_runner", runner),
        ]:
            if item is None:
                row[f"{prefix}_mid"] = -1
                row[f"{prefix}_score"] = float("nan")
                row[f"{prefix}_support"] = float("nan")
                row[f"{prefix}_learned"] = float("nan")
                row[f"{prefix}_consistency"] = float("nan")
                row[f"{prefix}_source_anchor"] = -1
                row[f"{prefix}_destination_anchor"] = -1
                continue
            mid, _, score, support, learned, consistency, source_anchor, destination_anchor = item
            row[f"{prefix}_mid"] = int(mid)
            row[f"{prefix}_score"] = float(score)
            row[f"{prefix}_support"] = float(support)
            row[f"{prefix}_learned"] = float(learned)
            row[f"{prefix}_consistency"] = float(consistency)
            row[f"{prefix}_source_anchor"] = int(source_anchor)
            row[f"{prefix}_destination_anchor"] = int(destination_anchor)
        if selection is not None and runner is not None:
            row["edge_path_selected_vs_runner_margin"] = float(selection[2]) - float(runner[2])
            row["edge_path_selected_support_vs_runner"] = float(selection[3]) - float(runner[3])
            row["edge_path_selected_learned_vs_runner"] = float(selection[4]) - float(runner[4])
            row["edge_path_selected_consistency_vs_runner"] = float(selection[5]) - float(runner[5])
        else:
            row["edge_path_selected_vs_runner_margin"] = float("nan")
            row["edge_path_selected_support_vs_runner"] = float("nan")
            row["edge_path_selected_learned_vs_runner"] = float("nan")
            row["edge_path_selected_consistency_vs_runner"] = float("nan")
        return row

    def update_edge_path_cleanup_slot(
        self,
        slot_idx: int,
        mid: int,
        path_feature: np.ndarray,
        lr: float,
    ) -> None:
        if lr <= 0.0:
            return
        slot_idx = self.edge_path_slot_index(slot_idx)
        mid = int(mid)
        counts = self.edge_path_cleanup_counts[slot_idx, mid]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            bank_slot = int(empty[0])
            self.edge_path_cleanup_prototypes[slot_idx, mid, bank_slot] = path_feature
            self.edge_path_cleanup_counts[slot_idx, mid, bank_slot] = 1.0
            return
        dots = self.edge_path_cleanup_prototypes[slot_idx, mid] @ path_feature
        bank_slot = int(np.argmax(dots))
        self.edge_path_cleanup_prototypes[slot_idx, mid, bank_slot] = phase.normalize_vector(
            (1.0 - lr) * self.edge_path_cleanup_prototypes[slot_idx, mid, bank_slot]
            + lr * path_feature
        )
        self.edge_path_cleanup_counts[slot_idx, mid, bank_slot] += 1.0

    def suppress_edge_path_cleanup_slot(
        self,
        slot_idx: int,
        mid: int,
        path_feature: np.ndarray,
        lr_scale: float = 1.0,
    ) -> None:
        lr = self.edge_path_cleanup_wrong_lr * float(max(lr_scale, 0.0))
        if lr <= 0.0:
            return
        slot_idx = self.edge_path_slot_index(slot_idx)
        mid = int(mid)
        counts = self.edge_path_cleanup_counts[slot_idx, mid]
        active = counts > 0.0
        if not np.any(active):
            return
        dots = self.edge_path_cleanup_prototypes[slot_idx, mid] @ path_feature
        dots = np.where(active, dots, -np.inf)
        bank_slot = int(np.argmax(dots))
        self.edge_path_cleanup_prototypes[slot_idx, mid, bank_slot] = phase.normalize_vector(
            self.edge_path_cleanup_prototypes[slot_idx, mid, bank_slot]
            - lr * path_feature
        )
        self.edge_path_cleanup_wrong_updates[slot_idx] += 1

    def edge_path_direct_scores_from_feature(self, slot_idx: int, feature: np.ndarray) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        if self.edge_path_direct_score_scale <= 0.0:
            return scores
        slot_idx = self.edge_path_direct_slot_index(slot_idx)
        counts = self.edge_path_direct_counts[slot_idx]
        active = np.flatnonzero(np.any(counts > 0.0, axis=1))
        if active.size == 0:
            return scores
        proto = self.edge_path_direct_prototypes[slot_idx, active]
        dots = np.einsum("asd,d->as", proto, feature, optimize=True).astype(np.float32)
        dots = np.where(counts[active] > 0.0, dots, -np.inf)
        scores[active] = float(self.edge_path_direct_score_scale) * np.max(dots, axis=1)
        self.edge_path_direct_score_checks[slot_idx] += 1
        return scores

    def edge_path_direct_scores_from_bundle(
        self,
        slot_idx: int,
        feature_bundle: list[tuple[float, np.ndarray]],
    ) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        weight_sum = 0.0
        for feature_weight, feature in feature_bundle:
            feature_weight = float(feature_weight)
            if feature_weight <= 0.0:
                continue
            scores += feature_weight * self.edge_path_direct_scores_from_feature(slot_idx, feature)
            weight_sum += feature_weight
        if weight_sum <= 0.0:
            return np.zeros(self.vocab_size, dtype=np.float32)
        return (scores / weight_sum).astype(np.float32)

    def edge_path_candidate_direct_scores(self, tokens: list[int], slot: int) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        if self.edge_path_direct_score_scale <= 0.0:
            return scores
        ranked = self.edge_path_soft_ranked_candidates(tokens, int(slot))
        if not ranked:
            return scores
        top_k = min(max(self.edge_path_soft_top_k, 1), len(ranked))
        top = ranked[:top_k]
        max_score = max(float(item[0]) for item in top)
        slot_idx = self.edge_path_direct_slot_index(slot)
        weight_sum = 0.0
        for score, support, _, _, _, _, _, _, source_state, destination_state, path_feature in top:
            soft = math.exp((float(score) - max_score) / self.edge_path_soft_temperature)
            weight = max(float(support), 1e-6) * soft
            if weight <= 0.0:
                continue
            feature_bundle = self.edge_path_direct_feature_bundle(
                int(slot),
                source_state,
                destination_state,
                path_feature,
            )
            scores += weight * self.edge_path_direct_scores_from_bundle(slot_idx, feature_bundle)
            weight_sum += weight
        if weight_sum <= 0.0:
            return np.zeros(self.vocab_size, dtype=np.float32)
        return (scores / weight_sum).astype(np.float32)

    def answer_slot_score_delta(
        self,
        context: Sequence[int] | np.ndarray,
        slot: int,
        mode: str,
    ) -> np.ndarray:
        if str(mode) != "edge_path_soft_direct":
            return np.zeros(self.vocab_size, dtype=np.float32)
        tokens = [int(token) for token in list(context)[-self.max_order :]]
        if self.edge_path_direct_mode in {"candidate_scores", "structured_scores"}:
            return self.edge_path_candidate_direct_scores(tokens, int(slot))
        slot_idx = self.edge_path_direct_slot_index(slot)
        feature = self.last_edge_path_direct_feature.get(slot_idx)
        if feature is None:
            self.edge_path_soft_state(tokens, int(slot))
            feature = self.last_edge_path_direct_feature.get(slot_idx)
        if feature is None:
            return np.zeros(self.vocab_size, dtype=np.float32)
        return self.edge_path_direct_scores_from_feature(slot_idx, feature)

    def update_edge_path_direct_target(
        self,
        slot_idx: int,
        target: int,
        feature: np.ndarray,
        lr_scale: float = 1.0,
    ) -> None:
        lr = self.edge_path_direct_lr * float(max(lr_scale, 0.0))
        if lr <= 0.0:
            return
        slot_idx = self.edge_path_direct_slot_index(slot_idx)
        target = int(target)
        counts = self.edge_path_direct_counts[slot_idx, target]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            bank_slot = int(empty[0])
            self.edge_path_direct_prototypes[slot_idx, target, bank_slot] = feature
            self.edge_path_direct_counts[slot_idx, target, bank_slot] = 1.0
            return
        dots = self.edge_path_direct_prototypes[slot_idx, target] @ feature
        bank_slot = int(np.argmax(dots))
        self.edge_path_direct_prototypes[slot_idx, target, bank_slot] = phase.normalize_vector(
            (1.0 - lr)
            * self.edge_path_direct_prototypes[slot_idx, target, bank_slot]
            + lr * feature
        )
        self.edge_path_direct_counts[slot_idx, target, bank_slot] += 1.0

    def update_edge_path_direct_wrong(
        self,
        slot_idx: int,
        wrong: int,
        feature: np.ndarray,
        lr_scale: float = 1.0,
    ) -> None:
        lr = self.edge_path_direct_wrong_lr * float(max(lr_scale, 0.0))
        if lr <= 0.0:
            return
        slot_idx = self.edge_path_direct_slot_index(slot_idx)
        wrong = int(wrong)
        counts = self.edge_path_direct_counts[slot_idx, wrong]
        active = counts > 0.0
        if not np.any(active):
            return
        dots = self.edge_path_direct_prototypes[slot_idx, wrong] @ feature
        dots = np.where(active, dots, -np.inf)
        bank_slot = int(np.argmax(dots))
        self.edge_path_direct_prototypes[slot_idx, wrong, bank_slot] = phase.normalize_vector(
            self.edge_path_direct_prototypes[slot_idx, wrong, bank_slot]
            - lr * feature
        )
        self.edge_path_direct_wrong_updates[slot_idx] += 1

    def update_edge_path_direct_bundle(
        self,
        slot_idx: int,
        target: int,
        wrong: int,
        should_apply_credit: bool,
        feature_bundle: list[tuple[float, np.ndarray]],
    ) -> bool:
        wrote_any = False
        for feature_weight, feature in feature_bundle:
            feature_weight = float(feature_weight)
            if feature_weight <= 0.0:
                continue
            self.update_edge_path_direct_target(slot_idx, int(target), feature, feature_weight)
            wrote_any = True
            if bool(should_apply_credit):
                self.update_edge_path_direct_wrong(slot_idx, int(wrong), feature, feature_weight)
        return wrote_any

    def update_answer_slot_feature(
        self,
        context: Sequence[int] | np.ndarray,
        target: int,
        wrong: int,
        slot: int,
        should_apply_credit: bool,
        mode: str,
        runner_counterfactual_margin_gain: float | None = None,
    ) -> None:
        mode = str(mode)
        if mode not in {"edge_path_wta", "edge_path_soft", "edge_path_soft_direct"}:
            return
        slot_idx = self.edge_path_slot_index(slot)
        selection = self.last_edge_path_selection.get(slot_idx)
        if selection is None:
            tokens = [int(token) for token in list(context)[-self.max_order :]]
            if mode in {"edge_path_soft", "edge_path_soft_direct"}:
                self.edge_path_soft_state(tokens, slot_idx)
            else:
                self.edge_path_wta_state(tokens, slot_idx)
            selection = self.last_edge_path_selection.get(slot_idx)
        direct_slot = self.edge_path_direct_slot_index(slot)
        direct_bundle: list[tuple[float, np.ndarray]] = []
        if self.edge_path_direct_mode == "candidate_scores":
            if selection is not None:
                direct_bundle = [(1.0, selection[1])]
        elif self.edge_path_direct_mode == "structured_scores":
            direct_bundle = self.last_edge_path_direct_feature_bundle.get(direct_slot, [])
        else:
            direct_feature = self.last_edge_path_direct_feature.get(direct_slot)
            if direct_feature is not None:
                direct_bundle = [(1.0, direct_feature)]
        if mode == "edge_path_soft_direct" and direct_bundle:
            wrote_any = self.update_edge_path_direct_bundle(
                direct_slot,
                int(target),
                int(wrong),
                bool(should_apply_credit),
                direct_bundle,
            )
            if wrote_any:
                self.edge_path_direct_updates[direct_slot] += 1
        if selection is None:
            return
        mid, path_feature, _, _, _, _, _, _ = selection
        runner = self.last_edge_path_runner_up.get(slot_idx)
        runner_pair_feature = None
        if runner is not None:
            runner_mid_for_pair, runner_feature_for_pair, _, _, _, _, _, _ = runner
            runner_pair_feature = self.edge_path_runner_arbiter_feature(
                int(mid),
                path_feature,
                int(runner_mid_for_pair),
                runner_feature_for_pair,
                selected_score=float(selection[2]),
                selected_support=float(selection[3]),
                selected_learned=float(selection[4]),
                selected_consistency=float(selection[5]),
                runner_score=float(runner[2]),
                runner_support=float(runner[3]),
                runner_learned=float(runner[4]),
                runner_consistency=float(runner[5]),
            )
        selected_closure_feature = None
        selected_states = self.last_edge_path_selection_states.get(slot_idx)
        if selected_states is not None:
            selected_closure_feature = self.edge_path_closure_proto_feature(
                selected_states[0],
                selected_states[1],
                path_feature,
            )
        runner_closure_feature = None
        runner_states = self.last_edge_path_runner_states.get(slot_idx)
        if runner is not None and runner_states is not None:
            runner_closure_feature = self.edge_path_closure_proto_feature(
                runner_states[0],
                runner_states[1],
                runner[1],
            )
        selected_affinity_feature = None
        runner_affinity_feature = None
        selected_closure_score = 0.0
        runner_closure_score = 0.0
        runner_support = 0.0
        runner_consistency = 0.0
        runner_learned = 0.0
        if runner is not None:
            runner_support = float(runner[3])
            runner_learned = float(runner[4])
            runner_consistency = float(runner[5])
        if selected_states is not None:
            selected_closure_score = self.edge_path_raw_closure_score(
                selected_states[0],
                selected_states[1],
                path_feature,
            )
        if runner is not None and runner_states is not None:
            runner_closure_score = self.edge_path_raw_closure_score(
                runner_states[0],
                runner_states[1],
                runner[1],
            )
        if self.edge_path_affinity_score_scale > 0.0 and selected_states is not None:
            if self.edge_path_affinity_gate(
                float(selection[3]),
                float(selected_closure_score),
                float(selection[5]),
                float(selection[4]),
                runner_support,
                runner_closure_score,
                runner_consistency,
                runner_learned,
            ):
                selected_affinity_feature = self.edge_path_affinity_feature(
                    float(selection[3]),
                    float(selected_closure_score),
                    float(selection[5]),
                    float(selection[4]),
                    runner_support,
                    runner_closure_score,
                    runner_consistency,
                    runner_learned,
                )
            if runner is not None:
                if self.edge_path_affinity_gate(
                    runner_support,
                    runner_closure_score,
                    runner_consistency,
                    runner_learned,
                    float(selection[3]),
                    float(selected_closure_score),
                    float(selection[5]),
                    float(selection[4]),
                ):
                    runner_affinity_feature = self.edge_path_affinity_feature(
                        runner_support,
                        runner_closure_score,
                        runner_consistency,
                        runner_learned,
                        float(selection[3]),
                        float(selected_closure_score),
                        float(selection[5]),
                        float(selection[4]),
                    )
        if self.edge_path_homeostasis_trace is not None:
            selected_homeostasis_weight = 1.0
            top_candidates_for_homeostasis = self.last_edge_path_top_candidates.get(slot_idx, [])
            if top_candidates_for_homeostasis:
                selected_homeostasis_weight = float(max(top_candidates_for_homeostasis[0][5], 0.0))
            selected_trace_idx = self.last_edge_path_trace_index.get(slot_idx)
            if selected_trace_idx is None:
                selected_trace_idx = self.edge_path_transient_trace_index(int(mid), path_feature)
            selected_homeostasis_update = True
            if self.edge_path_homeostasis_hard_gate_enabled():
                selected_homeostasis_update = self.edge_path_homeostasis_gate(
                    float(selection[3]),
                    float(selected_closure_score),
                    float(selection[5]),
                    float(selection[4]),
                    runner_support,
                    runner_closure_score,
                    runner_consistency,
                    runner_learned,
                )
            if selected_homeostasis_update:
                selected_homeostasis_weight *= self.edge_path_homeostasis_soft_multiplier(
                    float(selection[3]),
                    float(selected_closure_score),
                    float(selection[5]),
                    float(selection[4]),
                    runner_support,
                    runner_closure_score,
                    runner_consistency,
                    runner_learned,
                )
                self.update_edge_path_homeostasis(
                    slot_idx,
                    int(selected_trace_idx),
                    selected_homeostasis_weight,
                )
        if self.edge_path_cleanup_credit_mode == "reward_punish":
            if bool(should_apply_credit):
                self.suppress_edge_path_cleanup_slot(slot_idx, mid, path_feature)
            else:
                self.update_edge_path_cleanup_slot(slot_idx, mid, path_feature, self.edge_path_cleanup_lr)
                self.edge_path_cleanup_updates[slot_idx] += 1
                if runner is not None:
                    runner_mid, runner_feature, _, _, _, _, _, _ = runner
                    self.suppress_edge_path_cleanup_slot(slot_idx, runner_mid, runner_feature)
        elif self.edge_path_cleanup_credit_mode == "soft_eligibility":
            top_candidates = self.last_edge_path_top_candidates.get(slot_idx, [])
            if bool(should_apply_credit):
                selected_weight = 1.0
                if top_candidates:
                    selected_weight = float(max(top_candidates[0][5], 0.0))
                self.suppress_edge_path_cleanup_slot(
                    slot_idx,
                    mid,
                    path_feature,
                    lr_scale=selected_weight,
                )
            elif top_candidates:
                for cand_mid, cand_feature, _, _, _, cand_weight in top_candidates:
                    cand_weight = float(max(cand_weight, 0.0))
                    if cand_weight <= 0.0:
                        continue
                    self.update_edge_path_cleanup_slot(
                        slot_idx,
                        cand_mid,
                        cand_feature,
                        self.edge_path_cleanup_lr * cand_weight,
                    )
                    self.edge_path_cleanup_updates[slot_idx] += 1
                if runner is not None:
                    runner_mid, runner_feature, _, _, _, _, _, _ = runner
                    self.suppress_edge_path_cleanup_slot(
                        slot_idx,
                        runner_mid,
                        runner_feature,
                        lr_scale=0.5,
                    )
            else:
                self.update_edge_path_cleanup_slot(slot_idx, mid, path_feature, self.edge_path_cleanup_lr)
                self.edge_path_cleanup_updates[slot_idx] += 1
        elif self.edge_path_cleanup_credit_mode == "margin_gated_soft_eligibility":
            top_candidates = self.last_edge_path_top_candidates.get(slot_idx, [])
            margin = 0.0
            ambiguity = 0.0
            if runner is not None:
                _, _, runner_score, runner_support, _, _, _, _ = runner
                margin = max(float(selection[2]) - float(runner_score), 0.0)
                if self.edge_path_margin_gate <= 0.0:
                    ambiguity = 1.0
                else:
                    ambiguity = float(
                        np.clip((self.edge_path_margin_gate - margin) / self.edge_path_margin_gate, 0.0, 1.0)
                    )
            update_scale = self.edge_path_margin_min_scale + (
                (1.0 - self.edge_path_margin_min_scale) * ambiguity
            )
            if bool(should_apply_credit):
                selected_weight = 1.0
                selected_support = float(selection[3])
                support_protect = 1.0
                if runner is not None and selected_support > float(runner_support) + self.edge_path_margin_gate:
                    support_protect = 0.25
                if top_candidates:
                    selected_weight = float(max(top_candidates[0][5], 0.0))
                    for cand_mid, cand_feature, _, cand_support, _, cand_weight in top_candidates[1:]:
                        cand_weight = float(max(cand_weight, 0.0))
                        if cand_weight <= 0.0:
                            continue
                        if float(cand_support) + self.edge_path_margin_gate < selected_support:
                            continue
                        self.update_edge_path_cleanup_slot(
                            slot_idx,
                            cand_mid,
                            cand_feature,
                            self.edge_path_cleanup_lr
                            * cand_weight
                            * update_scale
                            * self.edge_path_margin_alt_scale,
                        )
                        self.edge_path_cleanup_updates[slot_idx] += 1
                self.suppress_edge_path_cleanup_slot(
                    slot_idx,
                    mid,
                    path_feature,
                    lr_scale=selected_weight * update_scale * support_protect,
                )
            elif top_candidates:
                for cand_mid, cand_feature, _, _, _, cand_weight in top_candidates:
                    cand_weight = float(max(cand_weight, 0.0))
                    if cand_weight <= 0.0:
                        continue
                    self.update_edge_path_cleanup_slot(
                        slot_idx,
                        cand_mid,
                        cand_feature,
                        self.edge_path_cleanup_lr * cand_weight * update_scale,
                    )
                    self.edge_path_cleanup_updates[slot_idx] += 1
                if runner is not None and ambiguity > 0.0:
                    runner_mid, runner_feature, _, _, _, _, _, _ = runner
                    self.suppress_edge_path_cleanup_slot(
                        slot_idx,
                        runner_mid,
                        runner_feature,
                        lr_scale=0.5 * ambiguity,
                    )
            else:
                self.update_edge_path_cleanup_slot(
                    slot_idx,
                    mid,
                    path_feature,
                    self.edge_path_cleanup_lr * update_scale,
                )
                self.edge_path_cleanup_updates[slot_idx] += 1
        elif self.edge_path_cleanup_credit_mode == "learned_margin_escape":
            top_candidates = self.last_edge_path_top_candidates.get(slot_idx, [])
            margin = 0.0
            support_margin = 0.0
            learned_margin = 0.0
            ambiguity = 0.0
            if runner is not None:
                _, _, runner_score, runner_support, runner_learned, _, _, _ = runner
                margin = max(float(selection[2]) - float(runner_score), 0.0)
                support_margin = float(selection[3]) - float(runner_support)
                learned_margin = float(selection[4]) - float(runner_learned)
                if self.edge_path_margin_gate <= 0.0:
                    ambiguity = 1.0
                else:
                    ambiguity = float(
                        np.clip((self.edge_path_margin_gate - margin) / self.edge_path_margin_gate, 0.0, 1.0)
                    )
            update_scale = self.edge_path_margin_min_scale + (
                (1.0 - self.edge_path_margin_min_scale) * ambiguity
            )
            selected_weight = 1.0
            if top_candidates:
                selected_weight = float(max(top_candidates[0][5], 0.0))
            if bool(should_apply_credit):
                learned_dominates = (
                    runner is not None
                    and learned_margin > 0.0
                    and learned_margin
                    >= self.edge_path_margin_learned_dominance * max(float(support_margin), 0.0)
                )
                high_margin = runner is not None and margin >= self.edge_path_margin_gate
                if high_margin and learned_dominates:
                    self.suppress_edge_path_cleanup_slot(
                        slot_idx,
                        mid,
                        path_feature,
                        lr_scale=selected_weight * self.edge_path_margin_escape_scale,
                    )
                else:
                    self.suppress_edge_path_cleanup_slot(
                        slot_idx,
                        mid,
                        path_feature,
                        lr_scale=selected_weight * update_scale,
                    )
            elif top_candidates:
                for cand_mid, cand_feature, _, _, _, cand_weight in top_candidates:
                    cand_weight = float(max(cand_weight, 0.0))
                    if cand_weight <= 0.0:
                        continue
                    self.update_edge_path_cleanup_slot(
                        slot_idx,
                        cand_mid,
                        cand_feature,
                        self.edge_path_cleanup_lr * cand_weight * update_scale,
                    )
                    self.edge_path_cleanup_updates[slot_idx] += 1
                if runner is not None and ambiguity > 0.0:
                    runner_mid, runner_feature, _, _, _, _, _, _ = runner
                    self.suppress_edge_path_cleanup_slot(
                        slot_idx,
                        runner_mid,
                        runner_feature,
                        lr_scale=0.5 * ambiguity,
                    )
            else:
                self.update_edge_path_cleanup_slot(
                    slot_idx,
                    mid,
                    path_feature,
                    self.edge_path_cleanup_lr * update_scale,
                )
                self.edge_path_cleanup_updates[slot_idx] += 1
        elif self.edge_path_cleanup_credit_mode == "transient_inhibit_escape":
            top_candidates = self.last_edge_path_top_candidates.get(slot_idx, [])
            if self.edge_path_transient_inhibit_decay < 1.0:
                self.edge_path_transient_inhibit_trace[slot_idx] *= self.edge_path_transient_inhibit_decay
                self.edge_path_transient_boost_trace[slot_idx] *= self.edge_path_transient_inhibit_decay
            margin = 0.0
            support_margin = 0.0
            learned_margin = 0.0
            ambiguity = 0.0
            if runner is not None:
                _, _, runner_score, runner_support, runner_learned, _, _, _ = runner
                margin = max(float(selection[2]) - float(runner_score), 0.0)
                support_margin = float(selection[3]) - float(runner_support)
                learned_margin = float(selection[4]) - float(runner_learned)
                if self.edge_path_margin_gate <= 0.0:
                    ambiguity = 1.0
                else:
                    ambiguity = float(
                        np.clip((self.edge_path_margin_gate - margin) / self.edge_path_margin_gate, 0.0, 1.0)
                    )
            update_scale = self.edge_path_margin_min_scale + (
                (1.0 - self.edge_path_margin_min_scale) * ambiguity
            )
            selected_weight = 1.0
            if top_candidates:
                selected_weight = float(max(top_candidates[0][5], 0.0))
            if bool(should_apply_credit):
                learned_dominates = (
                    runner is not None
                    and learned_margin > 0.0
                    and learned_margin
                    >= self.edge_path_margin_learned_dominance * max(float(support_margin), 0.0)
                )
                high_margin = runner is not None and margin >= self.edge_path_margin_gate
                if high_margin and learned_dominates and self.edge_path_transient_inhibit_lr > 0.0:
                    runner_close = (
                        runner is not None
                        and support_margin <= self.edge_path_transient_boost_support_margin
                    )
                    if runner_close and selected_closure_feature is not None:
                        self.update_edge_path_closure_proto_negative(
                            slot_idx,
                            selected_closure_feature,
                        )
                    if runner_close and runner_closure_feature is not None:
                        self.update_edge_path_closure_proto_positive(
                            slot_idx,
                            runner_closure_feature,
                        )
                    if runner_close and selected_affinity_feature is not None:
                        self.update_edge_path_affinity_negative(
                            slot_idx,
                            selected_affinity_feature,
                        )
                    if runner_close and runner_affinity_feature is not None:
                        self.update_edge_path_affinity_positive(
                            slot_idx,
                            runner_affinity_feature,
                        )
                    if runner_close and runner_pair_feature is not None:
                        arbiter_credit = self.edge_path_runner_arbiter_credit_mode == "answer_error"
                        if self.edge_path_runner_arbiter_credit_mode == "counterfactual_positive":
                            arbiter_credit = (
                                runner_counterfactual_margin_gain is not None
                                and math.isfinite(float(runner_counterfactual_margin_gain))
                                and float(runner_counterfactual_margin_gain) >= 0.0
                            )
                        if arbiter_credit:
                            self.update_edge_path_runner_arbiter_target(
                                slot_idx,
                                runner_pair_feature,
                            )
                    trace_delta = self.edge_path_transient_inhibit_lr * selected_weight
                    trace_idx = self.last_edge_path_trace_index.get(slot_idx)
                    if trace_idx is None:
                        trace_idx = self.edge_path_transient_trace_index(int(mid), path_feature)
                    current = float(self.edge_path_transient_inhibit_trace[slot_idx, trace_idx])
                    self.edge_path_transient_inhibit_trace[slot_idx, trace_idx] = min(
                        current + trace_delta,
                        1.0,
                    )
                    self.edge_path_transient_inhibit_updates[slot_idx] += 1
                    runner_consistent = True
                    if runner is not None and self.edge_path_transient_boost_consistency_margin >= 0.0:
                        runner_consistent = (
                            float(runner[5]) + self.edge_path_transient_boost_consistency_margin
                            >= float(selection[5])
                        )
                    runner_fresh = True
                    if runner is not None and self.edge_path_transient_boost_runner_learned_max >= 0.0:
                        runner_fresh = (
                            float(runner[4]) <= self.edge_path_transient_boost_runner_learned_max
                        )
                    if runner_close and not runner_consistent:
                        self.edge_path_transient_boost_consistency_skips[slot_idx] += 1
                    if runner_close and runner_consistent and not runner_fresh:
                        self.edge_path_transient_boost_learned_skips[slot_idx] += 1
                    runner_counterfactual_ok = True
                    if self.edge_path_transient_boost_counterfactual_min_gain > -1.0:
                        runner_counterfactual_ok = (
                            runner_counterfactual_margin_gain is not None
                            and math.isfinite(float(runner_counterfactual_margin_gain))
                            and float(runner_counterfactual_margin_gain)
                            >= self.edge_path_transient_boost_counterfactual_min_gain
                        )
                    if (
                        runner_close
                        and runner_consistent
                        and runner_fresh
                        and not runner_counterfactual_ok
                    ):
                        self.edge_path_transient_boost_counterfactual_skips[slot_idx] += 1
                    if (
                        runner_close
                        and runner_consistent
                        and runner_fresh
                        and runner_counterfactual_ok
                        and self.edge_path_transient_boost_lr > 0.0
                    ):
                        runner_trace_idx = self.last_edge_path_runner_trace_index.get(slot_idx)
                        if runner_trace_idx is not None:
                            runner_weight = 1.0
                            if len(top_candidates) > 1:
                                runner_weight = float(max(top_candidates[1][5], 0.0))
                            boost_delta = self.edge_path_transient_boost_lr * runner_weight
                            boost_current = float(
                                self.edge_path_transient_boost_trace[slot_idx, runner_trace_idx]
                            )
                            self.edge_path_transient_boost_trace[slot_idx, runner_trace_idx] = min(
                                boost_current + boost_delta,
                                1.0,
                            )
                            self.edge_path_transient_boost_updates[slot_idx] += 1
                else:
                    self.suppress_edge_path_cleanup_slot(
                        slot_idx,
                        mid,
                        path_feature,
                        lr_scale=selected_weight * update_scale,
                    )
            elif top_candidates:
                if selected_closure_feature is not None:
                    self.update_edge_path_closure_proto_positive(slot_idx, selected_closure_feature)
                if runner_closure_feature is not None:
                    self.update_edge_path_closure_proto_negative(slot_idx, runner_closure_feature)
                if selected_affinity_feature is not None:
                    self.update_edge_path_affinity_positive(slot_idx, selected_affinity_feature)
                if runner_affinity_feature is not None:
                    self.update_edge_path_affinity_negative(slot_idx, runner_affinity_feature)
                if runner_pair_feature is not None:
                    self.suppress_edge_path_runner_arbiter(slot_idx, runner_pair_feature)
                for cand_mid, cand_feature, _, _, _, cand_weight in top_candidates:
                    cand_weight = float(max(cand_weight, 0.0))
                    if cand_weight <= 0.0:
                        continue
                    self.update_edge_path_cleanup_slot(
                        slot_idx,
                        cand_mid,
                        cand_feature,
                        self.edge_path_cleanup_lr * cand_weight * update_scale,
                    )
                    self.edge_path_cleanup_updates[slot_idx] += 1
                if runner is not None and ambiguity > 0.0:
                    runner_mid, runner_feature, _, _, _, _, _, _ = runner
                    self.suppress_edge_path_cleanup_slot(
                        slot_idx,
                        runner_mid,
                        runner_feature,
                        lr_scale=0.5 * ambiguity,
                    )
            else:
                self.update_edge_path_cleanup_slot(
                    slot_idx,
                    mid,
                    path_feature,
                    self.edge_path_cleanup_lr * update_scale,
                )
                self.edge_path_cleanup_updates[slot_idx] += 1
        else:
            lr = self.edge_path_cleanup_lr
            if bool(should_apply_credit):
                lr *= 0.25
            self.update_edge_path_cleanup_slot(slot_idx, mid, path_feature, lr)
            self.edge_path_cleanup_updates[slot_idx] += 1
            if runner is not None and not bool(should_apply_credit):
                runner_mid, runner_feature, _, _, _, _, _, _ = runner
                self.suppress_edge_path_cleanup_slot(slot_idx, runner_mid, runner_feature)

    def answer_slot_feature(
        self,
        context: Sequence[int] | np.ndarray,
        slot: int,
        mode: str = "role_hop",
    ) -> np.ndarray:
        tokens = [int(token) for token in list(context)[-self.max_order :]]
        role_pieces, role_scores = self.transition_rollout(tokens)
        self.last_role_scores = role_scores
        base_feature = self.recurrent_state(tokens)
        mode = str(mode)
        if mode == "edge_path":
            role_feature = self.edge_path_state(tokens, int(slot))
        elif mode == "edge_path_wta":
            role_feature = self.edge_path_wta_state(tokens, int(slot))
        elif mode in {"edge_path_soft", "edge_path_soft_direct"}:
            role_feature = self.edge_path_soft_state(tokens, int(slot))
        elif role_pieces:
            slot_idx = min(max(int(slot), 0), len(role_pieces) - 1)
            role_feature = role_pieces[slot_idx]
        else:
            role_feature = np.zeros(self.state_dim, dtype=np.float32)
        self.last_base_feature = base_feature.astype(np.float32, copy=True)
        self.last_role_feature = phase.normalize_vector(np.concatenate(role_pieces).astype(np.float32))
        return phase.normalize_vector(np.concatenate([base_feature, role_feature]).astype(np.float32))

    def add_bank_scores(
        self,
        scores: np.ndarray,
        prototypes: np.ndarray | None,
        counts: np.ndarray | None,
        feature: np.ndarray,
        scale: float,
    ) -> None:
        if prototypes is None or counts is None or scale <= 0.0:
            return
        active = np.flatnonzero(np.any(counts > 0.0, axis=1))
        if active.size == 0:
            return
        proto = prototypes[active]
        dots = np.einsum("asd,d->as", proto, feature, optimize=True).astype(np.float32)
        dots = np.where(counts[active] > 0.0, dots, -np.inf)
        scores[active] += float(scale) * np.max(dots, axis=1)

    def cleanup_joint_delta(self, delta: np.ndarray) -> np.ndarray:
        if self.role_joint_rescue_top_k <= 0:
            return delta
        active = np.flatnonzero(delta > 0.0)
        if active.size <= self.role_joint_rescue_top_k:
            return delta
        active_scores = delta[active]
        winner_local = np.argpartition(active_scores, -self.role_joint_rescue_top_k)[
            -self.role_joint_rescue_top_k :
        ]
        winner_tokens = {int(active[idx]) for idx in winner_local}
        max_score = float(np.max(active_scores)) if active_scores.size else 0.0
        cleaned = delta.astype(np.float32, copy=True)
        for token in active:
            token = int(token)
            if token in winner_tokens:
                continue
            if self.role_joint_rescue_inhibit > 0.0 and max_score > 0.0:
                cleaned[token] = -self.role_joint_rescue_inhibit * max_score
            else:
                cleaned[token] = 0.0
        return cleaned

    def branch_component_scores(self, feature: np.ndarray) -> dict[str, np.ndarray]:
        base_scores = (self.bias_weight * self.output_bias).astype(np.float32, copy=True)
        self.add_bank_scores(
            base_scores,
            self.base_branch_prototypes,
            self.base_branch_counts,
            self.last_base_feature,
            self.role_branch_base_score_scale,
        )

        role_scores = (self.bias_weight * self.output_bias).astype(np.float32, copy=True)
        self.add_bank_scores(
            role_scores,
            self.role_branch_prototypes,
            self.role_branch_counts,
            self.last_role_feature,
            self.role_branch_role_score_scale,
        )

        base_role_scores = base_scores.astype(np.float32, copy=True)
        self.add_bank_scores(
            base_role_scores,
            self.role_branch_prototypes,
            self.role_branch_counts,
            self.last_role_feature,
            self.role_branch_role_score_scale,
        )

        direct_scores = base_role_scores.astype(np.float32, copy=True)
        if self.role_score_scale > 0.0:
            direct_scores += self.role_score_scale * self.last_role_scores

        out = {
            "base_only": base_scores.astype(np.float32),
            "role_only": role_scores.astype(np.float32),
            "base_plus_role": base_role_scores.astype(np.float32),
            "base_plus_direct": direct_scores.astype(np.float32),
        }

        if self.role_joint_rescue_readout:
            joint_delta = np.zeros(self.vocab_size, dtype=np.float32)
            self.add_bank_scores(
                joint_delta,
                self.joint_rescue_prototypes,
                self.joint_rescue_counts,
                feature,
                self.role_joint_rescue_score_scale,
            )
            self.last_joint_rescue_delta = joint_delta.astype(np.float32, copy=True)
            if self.joint_suppress_prototypes is not None and self.joint_suppress_counts is not None:
                suppress_delta = np.zeros(self.vocab_size, dtype=np.float32)
                self.add_bank_scores(
                    suppress_delta,
                    self.joint_suppress_prototypes,
                    self.joint_suppress_counts,
                    feature,
                    self.role_joint_suppress_score_scale,
                )
                joint_delta -= suppress_delta
            joint_delta = self.cleanup_joint_delta(joint_delta)
            joint_scores = base_role_scores.astype(np.float32, copy=True)
            joint_scores += joint_delta
            direct_joint_scores = joint_scores.astype(np.float32, copy=True)
            if self.role_score_scale > 0.0:
                direct_joint_scores += self.role_score_scale * self.last_role_scores
            out["base_plus_role_joint"] = joint_scores.astype(np.float32)
            out["base_plus_direct_joint"] = direct_joint_scores.astype(np.float32)

        return out

    def branch_arbiter_index(self, variant: str) -> int:
        return int(self.branch_arbiter_variants.index(variant))

    def branch_arbiter_margin(self, scores: np.ndarray) -> float:
        margin = self.top2_margin(scores)
        if not math.isfinite(margin):
            return 0.0
        return float(math.tanh(0.25 * margin))

    def branch_arbiter_feature(self, components: dict[str, np.ndarray]) -> np.ndarray:
        variants = self.branch_arbiter_variants
        preds = [int(np.argmax(components[name])) for name in variants]
        pieces: list[np.ndarray] = [
            np.array([1.0], dtype=np.float32),
            np.array([self.branch_arbiter_margin(components[name]) for name in variants], dtype=np.float32),
        ]
        agreements: list[float] = []
        for left in range(len(preds)):
            for right in range(left + 1, len(preds)):
                agreements.append(1.0 if preds[left] == preds[right] else -1.0)
        pieces.append(np.array(agreements, dtype=np.float32))
        for pred in preds:
            if 0 <= pred < self.vocab_size:
                pieces.append(self.token_codes[pred].astype(np.float32, copy=False))
            else:
                pieces.append(np.zeros(self.state_dim, dtype=np.float32))
        if self.role_branch_arbiter_rich_conflict_features:
            base_idx = self.branch_arbiter_index("base_only")
            default_idx = self.branch_arbiter_index(self.role_branch_arbiter_default)
            base_pred = preds[base_idx]
            default_pred = preds[default_idx]
            if 0 <= base_pred < self.vocab_size and 0 <= default_pred < self.vocab_size:
                pair_code = phase.normalize_vector(self.token_codes[base_pred] * self.token_codes[default_pred])
                base_role = float(self.last_role_scores[base_pred])
                default_role = float(self.last_role_scores[default_pred])
                base_joint = (
                    float(self.last_joint_rescue_delta[base_pred])
                    if self.last_joint_rescue_delta.size
                    else 0.0
                )
                default_joint = (
                    float(self.last_joint_rescue_delta[default_pred])
                    if self.last_joint_rescue_delta.size
                    else 0.0
                )
                base_scores = components["base_only"]
                default_scores = components[self.role_branch_arbiter_default]
                base_support_gap = float(base_scores[base_pred] - base_scores[default_pred])
                default_support_gap = float(default_scores[default_pred] - default_scores[base_pred])
                rich_scalars = np.array(
                    [
                        math.tanh(base_role),
                        math.tanh(default_role),
                        math.tanh(base_role - default_role),
                        math.tanh(base_joint),
                        math.tanh(default_joint),
                        math.tanh(base_joint - default_joint),
                        math.tanh(0.25 * base_support_gap),
                        math.tanh(0.25 * default_support_gap),
                    ],
                    dtype=np.float32,
                )
                pieces.append(rich_scalars)
                pieces.append(pair_code.astype(np.float32, copy=False))
            else:
                pieces.append(np.zeros(8, dtype=np.float32))
                pieces.append(np.zeros(self.state_dim, dtype=np.float32))
        return phase.normalize_vector(np.concatenate(pieces).astype(np.float32))

    def branch_arbiter_scores(self, feature: np.ndarray) -> np.ndarray:
        scores, _ = self.branch_arbiter_scores_and_counts(feature)
        return scores

    def branch_arbiter_scores_and_counts(self, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.branch_arbiter_prototypes is None or self.branch_arbiter_counts is None:
            return (
                np.zeros(len(self.branch_arbiter_variants), dtype=np.float32),
                np.zeros(len(self.branch_arbiter_variants), dtype=np.float32),
            )
        scores = np.zeros(len(self.branch_arbiter_variants), dtype=np.float32)
        best_counts = np.zeros(len(self.branch_arbiter_variants), dtype=np.float32)
        active_variants = np.flatnonzero(np.any(self.branch_arbiter_counts > 0.0, axis=1))
        for variant_idx in active_variants:
            counts = self.branch_arbiter_counts[variant_idx]
            dots = self.branch_arbiter_prototypes[variant_idx] @ feature
            dots = np.where(counts > 0.0, dots, -np.inf)
            best_slot = int(np.argmax(dots))
            scores[variant_idx] = float(self.role_branch_arbiter_score_scale) * float(dots[best_slot])
            best_counts[variant_idx] = float(counts[best_slot])
        return scores, best_counts

    def branch_rescue_evidence(self) -> tuple[float, float]:
        role_evidence = float(np.max(self.last_role_scores)) if self.last_role_scores.size else 0.0
        joint_evidence = (
            float(np.max(self.last_joint_rescue_delta))
            if self.role_joint_rescue_readout and self.last_joint_rescue_delta.size
            else 0.0
        )
        return role_evidence, joint_evidence

    def choose_branch_variant(self, components: dict[str, np.ndarray]) -> str:
        if self.role_branch_arbiter == "base_margin_adaptive":
            self.branch_arbiter_checks += 1
            base_margin = self.top2_margin(components["base_only"])
            if not math.isfinite(base_margin):
                base_margin = 0.0
            if float(base_margin) >= self.branch_arbiter_threshold:
                chosen_idx = self.branch_arbiter_index("base_only")
            else:
                chosen_idx = self.branch_arbiter_index(self.role_branch_arbiter_default)
            self.branch_arbiter_chosen[chosen_idx] += 1
            return self.branch_arbiter_variants[chosen_idx]
        if self.role_branch_arbiter == "base_margin_rescue":
            self.branch_arbiter_checks += 1
            base_margin = self.top2_margin(components["base_only"])
            if not math.isfinite(base_margin):
                base_margin = 0.0
            role_evidence, joint_evidence = self.branch_rescue_evidence()
            rescue_allowed = (
                role_evidence >= self.role_branch_arbiter_rescue_role_threshold
                or joint_evidence >= self.role_branch_arbiter_rescue_joint_threshold
            )
            if float(base_margin) >= self.role_branch_arbiter_base_margin and not rescue_allowed:
                chosen_idx = self.branch_arbiter_index("base_only")
            else:
                chosen_idx = self.branch_arbiter_index(self.role_branch_arbiter_default)
            self.branch_arbiter_chosen[chosen_idx] += 1
            return self.branch_arbiter_variants[chosen_idx]
        if self.role_branch_arbiter == "agreement_base_protect":
            self.branch_arbiter_checks += 1
            base_margin = self.top2_margin(components["base_only"])
            if not math.isfinite(base_margin):
                base_margin = 0.0
            default_scores = components[self.role_branch_arbiter_default]
            base_pred = int(np.argmax(components["base_only"]))
            default_pred = int(np.argmax(default_scores))
            if base_pred != default_pred and float(base_margin) >= self.role_branch_arbiter_base_margin:
                chosen_idx = self.branch_arbiter_index("base_only")
            else:
                chosen_idx = self.branch_arbiter_index(self.role_branch_arbiter_default)
            self.branch_arbiter_chosen[chosen_idx] += 1
            return self.branch_arbiter_variants[chosen_idx]
        if self.role_branch_arbiter == "conflict_proto":
            self.branch_arbiter_checks += 1
            base_pred = int(np.argmax(components["base_only"]))
            default_pred = int(np.argmax(components[self.role_branch_arbiter_default]))
            if base_pred == default_pred:
                chosen_idx = self.branch_arbiter_index(self.role_branch_arbiter_default)
            else:
                feature = self.branch_arbiter_feature(components)
                self.last_branch_arbiter_feature = feature
                scores, best_counts = self.branch_arbiter_scores_and_counts(feature)
                base_idx = self.branch_arbiter_index("base_only")
                default_idx = self.branch_arbiter_index(self.role_branch_arbiter_default)
                if self.branch_arbiter_counts is None or not np.any(self.branch_arbiter_counts > 0.0):
                    chosen_idx = default_idx
                elif (
                    float(best_counts[base_idx]) >= self.role_branch_arbiter_min_count
                    and float(scores[base_idx]) > float(scores[default_idx]) + self.role_branch_arbiter_margin
                ):
                    chosen_idx = base_idx
                else:
                    chosen_idx = default_idx
            self.branch_arbiter_chosen[chosen_idx] += 1
            return self.branch_arbiter_variants[chosen_idx]
        if self.role_branch_arbiter != "local_proto":
            return self.role_branch_arbiter_default
        feature = self.branch_arbiter_feature(components)
        self.last_branch_arbiter_feature = feature
        scores = self.branch_arbiter_scores(feature)
        self.branch_arbiter_checks += 1
        if self.branch_arbiter_counts is None or not np.any(self.branch_arbiter_counts > 0.0):
            chosen_idx = self.branch_arbiter_index(self.role_branch_arbiter_default)
        else:
            chosen_idx = int(np.argmax(scores))
        self.branch_arbiter_chosen[chosen_idx] += 1
        return self.branch_arbiter_variants[chosen_idx]

    def scores_from_feature(self, feature: np.ndarray, use_branch_arbiter: bool = True) -> np.ndarray:
        if self.role_branch_readout:
            components = self.branch_component_scores(feature)
            self.last_branch_components = components
            if use_branch_arbiter and self.role_branch_arbiter != "none":
                variant = self.choose_branch_variant(components)
                return components[variant].astype(np.float32, copy=True)
            base_variant = "base_plus_role_joint" if self.role_joint_rescue_readout else "base_plus_role"
            direct_variant = (
                "base_plus_direct_joint" if self.role_joint_rescue_readout else "base_plus_direct"
            )
            scores = components[base_variant].astype(np.float32, copy=True)
            if self.role_score_scale > 0.0 and self.role_score_gate_allows(scores):
                scores = components[direct_variant].astype(np.float32, copy=True)
            return scores.astype(np.float32)

        scores = (self.bias_weight * self.output_bias).astype(np.float32, copy=True)
        self.add_bank_scores(scores, self.prototypes, self.prototype_counts, feature, self.score_scale)
        if self.role_score_scale > 0.0 and self.role_score_gate_allows(scores):
            scores += self.role_score_scale * self.last_role_scores
        return scores.astype(np.float32)

    def scores(self, context: Sequence[int] | np.ndarray) -> np.ndarray:
        return self.scores_from_feature(self.feature(context))

    @staticmethod
    def top2_margin(scores: np.ndarray) -> float:
        if scores.size < 2:
            return float("inf")
        top2 = np.partition(scores.astype(np.float64, copy=False), -2)[-2:]
        return float(top2[1] - top2[0])

    def role_score_gate_allows(self, base_scores: np.ndarray) -> bool:
        mode = self.role_score_gate_mode
        if mode == "none":
            return True
        self.role_score_gate_checks += 1
        base_margin = self.top2_margin(base_scores)
        role_active = self.last_role_scores[self.last_role_scores > 0.0]
        role_margin = self.top2_margin(role_active) if role_active.size >= 2 else 0.0
        if mode == "base_low_margin":
            allow = base_margin < self.role_score_gate_base_margin
        elif mode == "role_high_margin":
            allow = role_margin >= self.role_score_gate_role_margin
        elif mode == "base_low_and_role_high":
            allow = (
                base_margin < self.role_score_gate_base_margin
                and role_margin >= self.role_score_gate_role_margin
            )
        else:
            raise ValueError(f"unknown role_score_gate_mode: {mode}")
        self.role_score_gate_opens += int(allow)
        return bool(allow)

    def compute_role_gate_delta(
        self,
        tokens: list[int],
        target: int,
        wrong: int,
        apply_credit: bool,
    ) -> np.ndarray:
        delta = np.zeros_like(self.role_gate_weights)
        if not apply_credit:
            return delta
        if self.role_gate_lr <= 0.0 and self.role_gate_wrong_lr <= 0.0:
            return delta
        prefix, query = self.prefix_and_query(tokens)
        seeds = self.initial_query_seeds(prefix, query)
        for hop in range(self.role_hops):
            events = self.transition_events(prefix, seeds, hop)
            if not events:
                break
            token_scores: dict[int, float] = {}
            gate_idx = min(hop, delta.shape[0] - 1)
            for score, _, neighbor, base_weight, feature in events:
                neighbor = int(neighbor)
                token_scores[neighbor] = token_scores.get(neighbor, 0.0) + float(score)
                positive = 1.0 if neighbor == int(target) else 0.0
                negative = 1.0 if neighbor == int(wrong) else 0.0
                if self.role_downstream_bonus > 0.0 and hop + 1 < self.role_hops:
                    positive += self.role_downstream_bonus * self.local_neighbor_strength(
                        prefix, neighbor, int(target)
                    )
                    negative += self.role_downstream_bonus * self.local_neighbor_strength(
                        prefix, neighbor, int(wrong)
                    )
                if positive > 0.0 and self.role_gate_lr > 0.0:
                    delta[gate_idx] += float(self.role_gate_lr * base_weight * positive) * feature
                if negative > 0.0 and self.role_gate_wrong_lr > 0.0:
                    delta[gate_idx] -= float(self.role_gate_wrong_lr * base_weight * negative) * feature
            seeds = self.top_tokens_from_scores(token_scores)
        return delta

    def apply_role_gate_delta(self, delta: np.ndarray) -> None:
        if np.any(delta):
            updated = self.role_gate_weights + delta
            for idx in range(updated.shape[0]):
                if np.any(updated[idx]):
                    updated[idx] = phase.normalize_vector(updated[idx])
            self.role_gate_weights = updated.astype(np.float32)

    def update_role_gate(self, tokens: list[int], target: int, wrong: int, apply_credit: bool) -> None:
        self.apply_role_gate_delta(self.compute_role_gate_delta(tokens, target, wrong, apply_credit))

    def update_bank_target(
        self,
        prototypes: np.ndarray,
        counts_by_slot: np.ndarray,
        target: int,
        feature: np.ndarray,
        lr: float,
    ) -> None:
        target = int(target)
        counts = counts_by_slot[target]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            slot = int(empty[0])
            prototypes[target, slot] = feature
            counts_by_slot[target, slot] = 1.0
            return
        dots = prototypes[target] @ feature
        slot = int(np.argmax(dots))
        prototypes[target, slot] = phase.normalize_vector(
            (1.0 - lr) * prototypes[target, slot] + lr * feature
        )
        counts_by_slot[target, slot] += 1.0

    def update_bank_wrong(
        self,
        prototypes: np.ndarray,
        counts_by_slot: np.ndarray,
        wrong: int,
        feature: np.ndarray,
    ) -> None:
        wrong = int(wrong)
        active = counts_by_slot[wrong] > 0.0
        if not np.any(active):
            return
        dots = prototypes[wrong] @ feature
        dots = np.where(active, dots, -np.inf)
        slot = int(np.argmax(dots))
        prototypes[wrong, slot] = phase.normalize_vector(
            prototypes[wrong, slot] - self.wrong_lr * feature
        )

    def update_target_slot(self, target: int, feature: np.ndarray) -> None:
        if self.role_branch_readout:
            assert self.base_branch_prototypes is not None
            assert self.base_branch_counts is not None
            assert self.role_branch_prototypes is not None
            assert self.role_branch_counts is not None
            self.update_bank_target(
                self.base_branch_prototypes,
                self.base_branch_counts,
                target,
                self.last_base_feature,
                self.lr,
            )
            self.update_bank_target(
                self.role_branch_prototypes,
                self.role_branch_counts,
                target,
                self.last_role_feature,
                self.lr,
            )
            if self.role_joint_rescue_readout:
                assert self.joint_rescue_prototypes is not None
                assert self.joint_rescue_counts is not None
                self.update_bank_target(
                    self.joint_rescue_prototypes,
                    self.joint_rescue_counts,
                    target,
                    feature,
                    self.lr,
                )
            return
        assert self.prototypes is not None
        assert self.prototype_counts is not None
        self.update_bank_target(self.prototypes, self.prototype_counts, target, feature, self.lr)

    def update_wrong_slot(self, wrong: int, feature: np.ndarray) -> None:
        if self.role_branch_readout:
            assert self.base_branch_prototypes is not None
            assert self.base_branch_counts is not None
            assert self.role_branch_prototypes is not None
            assert self.role_branch_counts is not None
            self.update_bank_wrong(
                self.base_branch_prototypes,
                self.base_branch_counts,
                wrong,
                self.last_base_feature,
            )
            self.update_bank_wrong(
                self.role_branch_prototypes,
                self.role_branch_counts,
                wrong,
                self.last_role_feature,
            )
            if self.role_joint_rescue_readout:
                assert self.joint_rescue_prototypes is not None
                assert self.joint_rescue_counts is not None
                self.update_bank_wrong(
                    self.joint_rescue_prototypes,
                    self.joint_rescue_counts,
                    wrong,
                    feature,
                )
            return
        assert self.prototypes is not None
        assert self.prototype_counts is not None
        self.update_bank_wrong(self.prototypes, self.prototype_counts, wrong, feature)

    @staticmethod
    def target_margin(scores: np.ndarray, target: int) -> float:
        target = int(target)
        target_score = float(scores[target])
        adjusted = scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        return target_score - float(np.max(adjusted))

    def branch_arbiter_target_variant(self, target: int) -> int:
        if not self.last_branch_components:
            return self.branch_arbiter_index(self.role_branch_arbiter_default)
        margins = [
            self.target_margin(self.last_branch_components[name], int(target))
            for name in self.branch_arbiter_variants
        ]
        return int(np.argmax(np.array(margins, dtype=np.float32)))

    def update_branch_arbiter_target(self, variant_idx: int, feature: np.ndarray) -> None:
        if self.branch_arbiter_prototypes is None or self.branch_arbiter_counts is None:
            return
        counts = self.branch_arbiter_counts[variant_idx]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            slot = int(empty[0])
            self.branch_arbiter_prototypes[variant_idx, slot] = feature
            self.branch_arbiter_counts[variant_idx, slot] = 1.0
            return
        dots = self.branch_arbiter_prototypes[variant_idx] @ feature
        slot = int(np.argmax(dots))
        self.branch_arbiter_prototypes[variant_idx, slot] = phase.normalize_vector(
            (1.0 - self.role_branch_arbiter_lr) * self.branch_arbiter_prototypes[variant_idx, slot]
            + self.role_branch_arbiter_lr * feature
        )
        self.branch_arbiter_counts[variant_idx, slot] += 1.0

    def update_branch_arbiter_wrong(self, variant_idx: int, feature: np.ndarray) -> None:
        if (
            self.branch_arbiter_prototypes is None
            or self.branch_arbiter_counts is None
            or self.role_branch_arbiter_wrong_lr <= 0.0
        ):
            return
        active = self.branch_arbiter_counts[variant_idx] > 0.0
        if not np.any(active):
            return
        dots = self.branch_arbiter_prototypes[variant_idx] @ feature
        dots = np.where(active, dots, -np.inf)
        slot = int(np.argmax(dots))
        self.branch_arbiter_prototypes[variant_idx, slot] = phase.normalize_vector(
            self.branch_arbiter_prototypes[variant_idx, slot]
            - self.role_branch_arbiter_wrong_lr * feature
        )

    def update_branch_arbiter(self, target: int) -> None:
        if self.role_branch_arbiter == "base_margin_adaptive":
            if not self.last_branch_components:
                return
            base_margin = self.top2_margin(self.last_branch_components["base_only"])
            if not math.isfinite(base_margin):
                base_margin = 0.0
            base_target_margin = self.target_margin(self.last_branch_components["base_only"], int(target))
            default_target_margin = self.target_margin(
                self.last_branch_components[self.role_branch_arbiter_default],
                int(target),
            )
            desired_base = base_target_margin >= default_target_margin
            if desired_base:
                target_threshold = max(0.0, float(base_margin) - self.role_branch_arbiter_margin)
                target_idx = self.branch_arbiter_index("base_only")
            else:
                target_threshold = float(base_margin) + self.role_branch_arbiter_margin
                target_idx = self.branch_arbiter_index(self.role_branch_arbiter_default)
            lr = self.role_branch_arbiter_threshold_lr
            self.branch_arbiter_threshold = (1.0 - lr) * self.branch_arbiter_threshold + lr * target_threshold
            self.branch_arbiter_updates += 1
            self.branch_arbiter_target_updates[target_idx] += 1
            return
        if self.role_branch_arbiter == "base_margin_rescue":
            if not self.last_branch_components:
                return
            base_target_margin = self.target_margin(self.last_branch_components["base_only"], int(target))
            default_target_margin = self.target_margin(
                self.last_branch_components[self.role_branch_arbiter_default],
                int(target),
            )
            target_variant = "base_only" if base_target_margin >= default_target_margin else self.role_branch_arbiter_default
            target_idx = self.branch_arbiter_index(target_variant)
            self.branch_arbiter_updates += 1
            self.branch_arbiter_target_updates[target_idx] += 1
            return
        if self.role_branch_arbiter == "agreement_base_protect":
            if not self.last_branch_components:
                return
            base_target_margin = self.target_margin(self.last_branch_components["base_only"], int(target))
            default_target_margin = self.target_margin(
                self.last_branch_components[self.role_branch_arbiter_default],
                int(target),
            )
            target_variant = "base_only" if base_target_margin >= default_target_margin else self.role_branch_arbiter_default
            target_idx = self.branch_arbiter_index(target_variant)
            self.branch_arbiter_updates += 1
            self.branch_arbiter_target_updates[target_idx] += 1
            return
        if self.role_branch_arbiter == "conflict_proto":
            if not self.last_branch_components:
                return
            base_pred = int(np.argmax(self.last_branch_components["base_only"]))
            default_pred = int(np.argmax(self.last_branch_components[self.role_branch_arbiter_default]))
            if base_pred == default_pred:
                return
            feature = self.branch_arbiter_feature(self.last_branch_components)
            base_idx = self.branch_arbiter_index("base_only")
            default_idx = self.branch_arbiter_index(self.role_branch_arbiter_default)
            base_target_margin = self.target_margin(self.last_branch_components["base_only"], int(target))
            default_target_margin = self.target_margin(
                self.last_branch_components[self.role_branch_arbiter_default],
                int(target),
            )
            target_idx = base_idx if base_target_margin >= default_target_margin else default_idx
            arbiter_scores = self.branch_arbiter_scores(feature)
            if self.branch_arbiter_counts is None or not np.any(self.branch_arbiter_counts > 0.0):
                wrong_idx = default_idx
            elif float(arbiter_scores[base_idx]) > float(arbiter_scores[default_idx]) + self.role_branch_arbiter_margin:
                wrong_idx = base_idx
            else:
                wrong_idx = default_idx
            self.update_branch_arbiter_target(target_idx, feature)
            if (
                wrong_idx != target_idx
                and float(arbiter_scores[wrong_idx]) + self.role_branch_arbiter_margin
                > float(arbiter_scores[target_idx])
            ):
                self.update_branch_arbiter_wrong(wrong_idx, feature)
            self.branch_arbiter_updates += 1
            self.branch_arbiter_target_updates[target_idx] += 1
            return
        if self.role_branch_arbiter != "local_proto":
            return
        if not self.last_branch_components:
            return
        feature = self.branch_arbiter_feature(self.last_branch_components)
        target_idx = self.branch_arbiter_target_variant(int(target))
        arbiter_scores = self.branch_arbiter_scores(feature)
        if self.branch_arbiter_counts is None or not np.any(self.branch_arbiter_counts > 0.0):
            wrong_idx = self.branch_arbiter_index(self.role_branch_arbiter_default)
        else:
            wrong_idx = int(np.argmax(arbiter_scores))
        self.update_branch_arbiter_target(target_idx, feature)
        if (
            wrong_idx != target_idx
            and float(arbiter_scores[wrong_idx]) + self.role_branch_arbiter_margin
            > float(arbiter_scores[target_idx])
        ):
            self.update_branch_arbiter_wrong(wrong_idx, feature)
        self.branch_arbiter_updates += 1
        self.branch_arbiter_target_updates[target_idx] += 1

    def update_joint_suppression(self, target: int, feature: np.ndarray) -> None:
        if self.joint_suppress_prototypes is None or self.joint_suppress_counts is None:
            return
        if "base_plus_direct_joint" not in self.last_branch_components:
            return
        joint_scores = self.last_branch_components["base_plus_direct_joint"]
        target = int(target)
        target_score = float(joint_scores[target])
        adjusted = joint_scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        wrong = int(np.argmax(adjusted))
        if float(adjusted[wrong]) + self.role_joint_suppress_margin <= target_score:
            return
        self.joint_suppress_candidates += 1
        direct_evidence = float(self.last_role_scores[wrong])
        joint_evidence = float(self.last_joint_rescue_delta[wrong])
        if (
            self.role_joint_suppress_mode in {"protect_direct", "joint_only"}
            and direct_evidence >= self.role_joint_suppress_direct_threshold
        ):
            self.joint_suppress_skipped_direct += 1
            return
        if (
            self.role_joint_suppress_mode == "joint_only"
            and joint_evidence <= self.role_joint_suppress_joint_threshold
        ):
            self.joint_suppress_skipped_joint += 1
            return
        self.update_bank_target(
            self.joint_suppress_prototypes,
            self.joint_suppress_counts,
            wrong,
            feature,
            self.role_joint_suppress_lr,
        )
        self.joint_suppress_updates += 1

    def update(self, context: Sequence[int] | np.ndarray, target: int) -> None:
        target = int(target)
        tokens = [int(token) for token in list(context)[-self.max_order :]]
        feature = self.feature(context)
        scores = self.scores_from_feature(feature, use_branch_arbiter=False)
        self.update_branch_arbiter(target)
        self.update_joint_suppression(target, feature)
        target_score = float(scores[target])
        adjusted = scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        wrong = int(np.argmax(adjusted))
        should_apply_credit = float(adjusted[wrong]) + self.margin > target_score
        self.update_role_gate(tokens, target, wrong, should_apply_credit)
        self.update_target_slot(target, feature)
        if self.wrong_lr > 0.0 and should_apply_credit:
            self.update_wrong_slot(wrong, feature)
        self.unigram_counts[target] += 1.0
        probs = self.unigram_counts / float(np.sum(self.unigram_counts))
        self.output_bias = np.log(np.maximum(probs, 1e-9)).astype(np.float32)

    def state_bytes(self) -> int:
        total = int(
            self.token_codes.nbytes
            + self.position_codes.nbytes
            + self.relative_codes.nbytes
            + self.role_gate_weights.nbytes
            + self.unigram_counts.nbytes
            + self.output_bias.nbytes
            + self.edge_path_cleanup_prototypes.nbytes
            + self.edge_path_cleanup_counts.nbytes
            + self.edge_path_cleanup_updates.nbytes
            + self.edge_path_cleanup_wrong_updates.nbytes
            + self.edge_path_cleanup_wins.nbytes
            + self.edge_path_transient_inhibit_trace.nbytes
            + self.edge_path_transient_inhibit_updates.nbytes
            + self.edge_path_transient_boost_trace.nbytes
            + self.edge_path_transient_boost_updates.nbytes
            + self.edge_path_transient_boost_consistency_skips.nbytes
            + self.edge_path_transient_boost_learned_skips.nbytes
            + self.edge_path_transient_boost_counterfactual_skips.nbytes
            + self.edge_path_runner_arbiter_prototypes.nbytes
            + self.edge_path_runner_arbiter_counts.nbytes
            + self.edge_path_runner_arbiter_updates.nbytes
            + self.edge_path_runner_arbiter_wrong_updates.nbytes
            + self.edge_path_direct_prototypes.nbytes
            + self.edge_path_direct_counts.nbytes
            + self.edge_path_direct_updates.nbytes
            + self.edge_path_direct_wrong_updates.nbytes
            + self.edge_path_direct_score_checks.nbytes
        )
        if self.prototypes is not None and self.prototype_counts is not None:
            total += int(self.prototypes.nbytes + self.prototype_counts.nbytes)
        if self.base_branch_prototypes is not None and self.base_branch_counts is not None:
            total += int(self.base_branch_prototypes.nbytes + self.base_branch_counts.nbytes)
        if self.role_branch_prototypes is not None and self.role_branch_counts is not None:
            total += int(self.role_branch_prototypes.nbytes + self.role_branch_counts.nbytes)
        if self.joint_rescue_prototypes is not None and self.joint_rescue_counts is not None:
            total += int(self.joint_rescue_prototypes.nbytes + self.joint_rescue_counts.nbytes)
        if self.joint_suppress_prototypes is not None and self.joint_suppress_counts is not None:
            total += int(self.joint_suppress_prototypes.nbytes + self.joint_suppress_counts.nbytes)
        if self.branch_arbiter_prototypes is not None and self.branch_arbiter_counts is not None:
            total += int(self.branch_arbiter_prototypes.nbytes + self.branch_arbiter_counts.nbytes)
        if (
            self.edge_path_runner_arbiter_negative_prototypes is not None
            and self.edge_path_runner_arbiter_negative_counts is not None
        ):
            total += int(
                self.edge_path_runner_arbiter_negative_prototypes.nbytes
                + self.edge_path_runner_arbiter_negative_counts.nbytes
            )
        if self.edge_path_runner_arbiter_gap_codes is not None:
            total += int(self.edge_path_runner_arbiter_gap_codes.nbytes)
        if self.edge_path_homeostasis_trace is not None:
            total += int(self.edge_path_homeostasis_trace.nbytes + self.edge_path_homeostasis_updates.nbytes)
        if (
            self.edge_path_closure_proto_positive is not None
            and self.edge_path_closure_proto_positive_counts is not None
            and self.edge_path_closure_proto_negative is not None
            and self.edge_path_closure_proto_negative_counts is not None
        ):
            total += int(
                self.edge_path_closure_proto_positive.nbytes
                + self.edge_path_closure_proto_positive_counts.nbytes
                + self.edge_path_closure_proto_negative.nbytes
                + self.edge_path_closure_proto_negative_counts.nbytes
                + self.edge_path_closure_proto_updates.nbytes
                + self.edge_path_closure_proto_wrong_updates.nbytes
            )
        if (
            self.edge_path_affinity_positive is not None
            and self.edge_path_affinity_positive_counts is not None
            and self.edge_path_affinity_negative is not None
            and self.edge_path_affinity_negative_counts is not None
        ):
            total += int(
                self.edge_path_affinity_positive.nbytes
                + self.edge_path_affinity_positive_counts.nbytes
                + self.edge_path_affinity_negative.nbytes
                + self.edge_path_affinity_negative_counts.nbytes
                + self.edge_path_affinity_updates.nbytes
                + self.edge_path_affinity_wrong_updates.nbytes
            )
        return total

    def event_cache_stats(self) -> dict[str, Any]:
        lookups = self.role_event_cache_hits + self.role_event_cache_misses
        return {
            "role_event_cache_size": self.role_event_cache_size,
            "role_event_cache_entries": len(self.role_event_cache),
            "role_event_cache_hits": self.role_event_cache_hits,
            "role_event_cache_misses": self.role_event_cache_misses,
            "role_event_cache_hit_rate": (
                self.role_event_cache_hits / lookups if lookups > 0 else 0.0
            ),
            "role_event_cache_bytes": int(len(self.role_event_cache) * self.state_dim * 4),
        }

    def role_score_gate_stats(self) -> dict[str, Any]:
        return {
            "role_score_gate_mode": self.role_score_gate_mode,
            "role_score_gate_checks": self.role_score_gate_checks,
            "role_score_gate_opens": self.role_score_gate_opens,
            "role_score_gate_open_rate": (
                self.role_score_gate_opens / self.role_score_gate_checks
                if self.role_score_gate_checks > 0
                else 1.0
            ),
            "role_score_gate_base_margin": self.role_score_gate_base_margin,
            "role_score_gate_role_margin": self.role_score_gate_role_margin,
        }

    def role_branch_arbiter_stats(self) -> dict[str, Any]:
        return {
            "role_branch_arbiter": self.role_branch_arbiter,
            "role_branch_arbiter_default": self.role_branch_arbiter_default,
            "role_branch_arbiter_slots": self.role_branch_arbiter_slots,
            "role_branch_arbiter_lr": self.role_branch_arbiter_lr,
            "role_branch_arbiter_wrong_lr": self.role_branch_arbiter_wrong_lr,
            "role_branch_arbiter_score_scale": self.role_branch_arbiter_score_scale,
            "role_branch_arbiter_margin": self.role_branch_arbiter_margin,
            "role_branch_arbiter_min_count": self.role_branch_arbiter_min_count,
            "role_branch_arbiter_base_margin": self.role_branch_arbiter_base_margin,
            "role_branch_arbiter_threshold_lr": self.role_branch_arbiter_threshold_lr,
            "role_branch_arbiter_rescue_role_threshold": self.role_branch_arbiter_rescue_role_threshold,
            "role_branch_arbiter_rescue_joint_threshold": self.role_branch_arbiter_rescue_joint_threshold,
            "role_branch_arbiter_joint_variants": self.role_branch_arbiter_joint_variants,
            "role_branch_arbiter_rich_conflict_features": (
                self.role_branch_arbiter_rich_conflict_features
            ),
            "role_branch_arbiter_variants": list(self.branch_arbiter_variants),
            "role_branch_arbiter_threshold": self.branch_arbiter_threshold,
            "role_branch_arbiter_checks": self.branch_arbiter_checks,
            "role_branch_arbiter_updates": self.branch_arbiter_updates,
            "role_branch_arbiter_active_slots": (
                int(np.count_nonzero(self.branch_arbiter_counts))
                if self.branch_arbiter_counts is not None
                else 0
            ),
            "role_branch_arbiter_chosen": {
                name: int(self.branch_arbiter_chosen[idx])
                for idx, name in enumerate(self.branch_arbiter_variants)
            },
            "role_branch_arbiter_target_updates": {
                name: int(self.branch_arbiter_target_updates[idx])
                for idx, name in enumerate(self.branch_arbiter_variants)
            },
        }

    def role_joint_suppress_stats(self) -> dict[str, Any]:
        return {
            "role_joint_suppress_slots": self.role_joint_suppress_slots,
            "role_joint_suppress_lr": self.role_joint_suppress_lr,
            "role_joint_suppress_score_scale": self.role_joint_suppress_score_scale,
            "role_joint_suppress_margin": self.role_joint_suppress_margin,
            "role_joint_suppress_mode": self.role_joint_suppress_mode,
            "role_joint_suppress_direct_threshold": self.role_joint_suppress_direct_threshold,
            "role_joint_suppress_joint_threshold": self.role_joint_suppress_joint_threshold,
            "role_joint_suppress_candidates": self.joint_suppress_candidates,
            "role_joint_suppress_updates": self.joint_suppress_updates,
            "role_joint_suppress_skipped_direct": self.joint_suppress_skipped_direct,
            "role_joint_suppress_skipped_joint": self.joint_suppress_skipped_joint,
            "role_joint_suppress_active_slots": (
                int(np.count_nonzero(self.joint_suppress_counts))
                if self.joint_suppress_counts is not None
                else 0
            ),
        }

    def edge_path_cleanup_stats(self) -> dict[str, Any]:
        avg_candidates = (
            self.edge_path_cleanup_candidates / self.edge_path_cleanup_checks
            if self.edge_path_cleanup_checks > 0
            else 0.0
        )
        return {
            "edge_path_cleanup_answer_slots": self.edge_path_cleanup_answer_slots,
            "edge_path_cleanup_slots": self.edge_path_cleanup_slots,
            "edge_path_cleanup_lr": self.edge_path_cleanup_lr,
            "edge_path_cleanup_wrong_lr": self.edge_path_cleanup_wrong_lr,
            "edge_path_cleanup_score_scale": self.edge_path_cleanup_score_scale,
            "edge_path_cleanup_top_k": self.edge_path_cleanup_top_k,
            "edge_path_cleanup_inhibit": self.edge_path_cleanup_inhibit,
            "edge_path_cleanup_credit_mode": self.edge_path_cleanup_credit_mode,
            "edge_path_soft_top_k": self.edge_path_soft_top_k,
            "edge_path_soft_temperature": self.edge_path_soft_temperature,
            "edge_path_soft_consistency_scale": self.edge_path_soft_consistency_scale,
            "edge_path_soft_learned_scale": self.edge_path_soft_learned_scale,
            "edge_path_closure_score_scale": self.edge_path_closure_score_scale,
            "edge_path_closure_proto_slots": self.edge_path_closure_proto_slots,
            "edge_path_closure_proto_lr": self.edge_path_closure_proto_lr,
            "edge_path_closure_proto_wrong_lr": self.edge_path_closure_proto_wrong_lr,
            "edge_path_closure_proto_score_scale": self.edge_path_closure_proto_score_scale,
            "edge_path_closure_proto_min_count": self.edge_path_closure_proto_min_count,
            "edge_path_closure_proto_positive_active": int(
                0
                if self.edge_path_closure_proto_positive_counts is None
                else np.count_nonzero(self.edge_path_closure_proto_positive_counts)
            ),
            "edge_path_closure_proto_negative_active": int(
                0
                if self.edge_path_closure_proto_negative_counts is None
                else np.count_nonzero(self.edge_path_closure_proto_negative_counts)
            ),
            "edge_path_closure_proto_updates": [
                int(x) for x in self.edge_path_closure_proto_updates
            ],
            "edge_path_closure_proto_wrong_updates": [
                int(x) for x in self.edge_path_closure_proto_wrong_updates
            ],
            "edge_path_closure_proto_score_checks": int(self.edge_path_closure_proto_score_checks),
            "edge_path_closure_proto_score_applied": int(self.edge_path_closure_proto_score_applied),
            "edge_path_closure_proto_score_count_skips": int(
                self.edge_path_closure_proto_score_count_skips
            ),
            "edge_path_closure_proto_positive_max_count": float(
                0.0
                if self.edge_path_closure_proto_positive_counts is None
                else np.max(self.edge_path_closure_proto_positive_counts)
            ),
            "edge_path_closure_proto_negative_max_count": float(
                0.0
                if self.edge_path_closure_proto_negative_counts is None
                else np.max(self.edge_path_closure_proto_negative_counts)
            ),
            "edge_path_affinity_slots": self.edge_path_affinity_slots,
            "edge_path_affinity_lr": self.edge_path_affinity_lr,
            "edge_path_affinity_wrong_lr": self.edge_path_affinity_wrong_lr,
            "edge_path_affinity_score_scale": self.edge_path_affinity_score_scale,
            "edge_path_affinity_min_count": self.edge_path_affinity_min_count,
            "edge_path_affinity_margin_gate": self.edge_path_affinity_margin_gate,
            "edge_path_affinity_learned_dominance": self.edge_path_affinity_learned_dominance,
            "edge_path_affinity_consistency_protect": self.edge_path_affinity_consistency_protect,
            "edge_path_affinity_positive_active": int(
                0
                if self.edge_path_affinity_positive_counts is None
                else np.count_nonzero(self.edge_path_affinity_positive_counts)
            ),
            "edge_path_affinity_negative_active": int(
                0
                if self.edge_path_affinity_negative_counts is None
                else np.count_nonzero(self.edge_path_affinity_negative_counts)
            ),
            "edge_path_affinity_updates": [
                int(x) for x in self.edge_path_affinity_updates
            ],
            "edge_path_affinity_wrong_updates": [
                int(x) for x in self.edge_path_affinity_wrong_updates
            ],
            "edge_path_affinity_score_checks": int(self.edge_path_affinity_score_checks),
            "edge_path_affinity_score_applied": int(self.edge_path_affinity_score_applied),
            "edge_path_affinity_score_count_skips": int(
                self.edge_path_affinity_score_count_skips
            ),
            "edge_path_affinity_gate_checks": int(self.edge_path_affinity_gate_checks),
            "edge_path_affinity_gate_passes": int(self.edge_path_affinity_gate_passes),
            "edge_path_affinity_gate_near_passes": int(self.edge_path_affinity_gate_near_passes),
            "edge_path_affinity_gate_conflict_passes": int(
                self.edge_path_affinity_gate_conflict_passes
            ),
            "edge_path_affinity_gate_skips": int(self.edge_path_affinity_gate_skips),
            "edge_path_affinity_positive_max_count": float(
                0.0
                if self.edge_path_affinity_positive_counts is None
                else np.max(self.edge_path_affinity_positive_counts)
            ),
            "edge_path_affinity_negative_max_count": float(
                0.0
                if self.edge_path_affinity_negative_counts is None
                else np.max(self.edge_path_affinity_negative_counts)
            ),
            "edge_path_cleanup_checks": self.edge_path_cleanup_checks,
            "edge_path_cleanup_avg_candidates": avg_candidates,
            "edge_path_cleanup_active_slots": int(np.count_nonzero(self.edge_path_cleanup_counts)),
            "edge_path_cleanup_updates": [int(x) for x in self.edge_path_cleanup_updates],
            "edge_path_cleanup_wrong_updates": [int(x) for x in self.edge_path_cleanup_wrong_updates],
            "edge_path_cleanup_wins": [int(x) for x in self.edge_path_cleanup_wins],
            "edge_path_transient_inhibit_scale": self.edge_path_transient_inhibit_scale,
            "edge_path_transient_inhibit_lr": self.edge_path_transient_inhibit_lr,
            "edge_path_transient_inhibit_decay": self.edge_path_transient_inhibit_decay,
            "edge_path_transient_inhibit_key": self.edge_path_transient_inhibit_key,
            "edge_path_transient_inhibit_hash_size": self.edge_path_transient_inhibit_hash_size,
            "edge_path_transient_inhibit_active": int(
                np.count_nonzero(self.edge_path_transient_inhibit_trace > 1e-6)
            ),
            "edge_path_transient_inhibit_updates": [
                int(x) for x in self.edge_path_transient_inhibit_updates
            ],
            "edge_path_transient_inhibit_max": float(np.max(self.edge_path_transient_inhibit_trace)),
            "edge_path_transient_boost_scale": self.edge_path_transient_boost_scale,
            "edge_path_transient_boost_lr": self.edge_path_transient_boost_lr,
            "edge_path_transient_boost_support_margin": self.edge_path_transient_boost_support_margin,
            "edge_path_transient_boost_consistency_margin": (
                self.edge_path_transient_boost_consistency_margin
            ),
            "edge_path_transient_boost_runner_learned_max": (
                self.edge_path_transient_boost_runner_learned_max
            ),
            "edge_path_transient_boost_counterfactual_min_gain": (
                self.edge_path_transient_boost_counterfactual_min_gain
            ),
            "edge_path_transient_boost_active": int(
                np.count_nonzero(self.edge_path_transient_boost_trace > 1e-6)
            ),
            "edge_path_transient_boost_updates": [
                int(x) for x in self.edge_path_transient_boost_updates
            ],
            "edge_path_transient_boost_consistency_skips": [
                int(x) for x in self.edge_path_transient_boost_consistency_skips
            ],
            "edge_path_transient_boost_learned_skips": [
                int(x) for x in self.edge_path_transient_boost_learned_skips
            ],
            "edge_path_transient_boost_counterfactual_skips": [
                int(x) for x in self.edge_path_transient_boost_counterfactual_skips
            ],
            "edge_path_transient_boost_max": float(np.max(self.edge_path_transient_boost_trace)),
            "edge_path_homeostasis_scale": self.edge_path_homeostasis_scale,
            "edge_path_homeostasis_lr": self.edge_path_homeostasis_lr,
            "edge_path_homeostasis_decay": self.edge_path_homeostasis_decay,
            "edge_path_homeostasis_min_slot": self.edge_path_homeostasis_min_slot,
            "edge_path_homeostasis_learned_dominance": (
                self.edge_path_homeostasis_learned_dominance
            ),
            "edge_path_homeostasis_structure_margin": self.edge_path_homeostasis_structure_margin,
            "edge_path_homeostasis_soft_mod_scale": self.edge_path_homeostasis_soft_mod_scale,
            "edge_path_homeostasis_soft_mod_floor": self.edge_path_homeostasis_soft_mod_floor,
            "edge_path_homeostasis_trace_threshold": self.edge_path_homeostasis_trace_threshold,
            "edge_path_homeostasis_trace_gain": self.edge_path_homeostasis_trace_gain,
            "edge_path_homeostasis_active": int(
                0
                if self.edge_path_homeostasis_trace is None
                else np.count_nonzero(self.edge_path_homeostasis_trace > 1e-6)
            ),
            "edge_path_homeostasis_updates": [
                int(x) for x in self.edge_path_homeostasis_updates
            ],
            "edge_path_homeostasis_max": float(
                0.0
                if self.edge_path_homeostasis_trace is None
                else np.max(self.edge_path_homeostasis_trace)
            ),
            "edge_path_homeostasis_gate_checks": int(self.edge_path_homeostasis_gate_checks),
            "edge_path_homeostasis_gate_passes": int(self.edge_path_homeostasis_gate_passes),
            "edge_path_homeostasis_gate_skips": int(self.edge_path_homeostasis_gate_skips),
            "edge_path_homeostasis_soft_mod_checks": int(
                self.edge_path_homeostasis_soft_mod_checks
            ),
            "edge_path_homeostasis_soft_mod_mean": float(
                self.edge_path_homeostasis_soft_mod_sum
                / max(self.edge_path_homeostasis_soft_mod_checks, 1)
            ),
            "edge_path_homeostasis_soft_mod_min": float(
                0.0
                if self.edge_path_homeostasis_soft_mod_checks == 0
                else self.edge_path_homeostasis_soft_mod_min
            ),
            "edge_path_homeostasis_soft_mod_max": float(
                0.0
                if self.edge_path_homeostasis_soft_mod_checks == 0
                else self.edge_path_homeostasis_soft_mod_max
            ),
            "edge_path_homeostasis_trace_mod_checks": int(
                self.edge_path_homeostasis_trace_mod_checks
            ),
            "edge_path_homeostasis_trace_mod_raw_mean": float(
                self.edge_path_homeostasis_trace_mod_raw_sum
                / max(self.edge_path_homeostasis_trace_mod_checks, 1)
            ),
            "edge_path_homeostasis_trace_mod_effective_mean": float(
                self.edge_path_homeostasis_trace_mod_effective_sum
                / max(self.edge_path_homeostasis_trace_mod_checks, 1)
            ),
            "edge_path_homeostasis_trace_mod_active_rate": float(
                self.edge_path_homeostasis_trace_mod_active
                / max(self.edge_path_homeostasis_trace_mod_checks, 1)
            ),
            "edge_path_runner_arbiter_slots": self.edge_path_runner_arbiter_slots,
            "edge_path_runner_arbiter_lr": self.edge_path_runner_arbiter_lr,
            "edge_path_runner_arbiter_wrong_lr": self.edge_path_runner_arbiter_wrong_lr,
            "edge_path_runner_arbiter_score_scale": self.edge_path_runner_arbiter_score_scale,
            "edge_path_runner_arbiter_margin": self.edge_path_runner_arbiter_margin,
            "edge_path_runner_arbiter_min_count": self.edge_path_runner_arbiter_min_count,
            "edge_path_runner_arbiter_negative_mode": self.edge_path_runner_arbiter_negative_mode,
            "edge_path_runner_arbiter_feature_mode": self.edge_path_runner_arbiter_feature_mode,
            "edge_path_runner_arbiter_gap_scale": self.edge_path_runner_arbiter_gap_scale,
            "edge_path_runner_arbiter_credit_mode": self.edge_path_runner_arbiter_credit_mode,
            "edge_path_runner_arbiter_active_slots": int(
                np.count_nonzero(self.edge_path_runner_arbiter_counts)
            ),
            "edge_path_runner_arbiter_negative_active_slots": int(
                0
                if self.edge_path_runner_arbiter_negative_counts is None
                else np.count_nonzero(self.edge_path_runner_arbiter_negative_counts)
            ),
            "edge_path_runner_arbiter_updates": [
                int(x) for x in self.edge_path_runner_arbiter_updates
            ],
            "edge_path_runner_arbiter_wrong_updates": [
                int(x) for x in self.edge_path_runner_arbiter_wrong_updates
            ],
            "edge_path_runner_arbiter_score_checks": int(self.edge_path_runner_arbiter_score_checks),
            "edge_path_runner_arbiter_score_applied": int(self.edge_path_runner_arbiter_score_applied),
            "edge_path_runner_arbiter_score_count_skips": int(
                self.edge_path_runner_arbiter_score_count_skips
            ),
            "edge_path_runner_arbiter_max_count": float(np.max(self.edge_path_runner_arbiter_counts)),
            "edge_path_runner_arbiter_negative_max_count": float(
                0.0
                if self.edge_path_runner_arbiter_negative_counts is None
                else np.max(self.edge_path_runner_arbiter_negative_counts)
            ),
        }

    def edge_path_direct_stats(self) -> dict[str, Any]:
        return {
            "edge_path_direct_answer_slots": self.edge_path_direct_answer_slots,
            "edge_path_direct_slots": self.edge_path_direct_slots,
            "edge_path_direct_lr": self.edge_path_direct_lr,
            "edge_path_direct_wrong_lr": self.edge_path_direct_wrong_lr,
            "edge_path_direct_score_scale": self.edge_path_direct_score_scale,
            "edge_path_direct_mode": self.edge_path_direct_mode,
            "edge_path_structured_side_weight": self.edge_path_structured_side_weight,
            "edge_path_structured_path_weight": self.edge_path_structured_path_weight,
            "edge_path_structured_other_weight": self.edge_path_structured_other_weight,
            "edge_path_direct_active_slots": int(np.count_nonzero(self.edge_path_direct_counts)),
            "edge_path_direct_updates": [int(x) for x in self.edge_path_direct_updates],
            "edge_path_direct_wrong_updates": [int(x) for x in self.edge_path_direct_wrong_updates],
            "edge_path_direct_score_checks": [int(x) for x in self.edge_path_direct_score_checks],
        }

    def active_contexts(self) -> int:
        edge_cleanup = int(np.count_nonzero(self.edge_path_cleanup_counts))
        edge_direct = int(np.count_nonzero(self.edge_path_direct_counts))
        edge_transient = int(np.count_nonzero(self.edge_path_transient_inhibit_trace > 1e-6))
        edge_boost = int(np.count_nonzero(self.edge_path_transient_boost_trace > 1e-6))
        edge_homeostasis = int(
            0
            if self.edge_path_homeostasis_trace is None
            else np.count_nonzero(self.edge_path_homeostasis_trace > 1e-6)
        )
        edge_runner_arbiter = int(np.count_nonzero(self.edge_path_runner_arbiter_counts))
        edge_affinity = int(
            0
            if self.edge_path_affinity_positive_counts is None
            else np.count_nonzero(self.edge_path_affinity_positive_counts)
        )
        if self.role_branch_readout:
            base = np.count_nonzero(self.base_branch_counts) if self.base_branch_counts is not None else 0
            role = np.count_nonzero(self.role_branch_counts) if self.role_branch_counts is not None else 0
            joint = np.count_nonzero(self.joint_rescue_counts) if self.joint_rescue_counts is not None else 0
            suppress = (
                np.count_nonzero(self.joint_suppress_counts)
                if self.joint_suppress_counts is not None
                else 0
            )
            arbiter = np.count_nonzero(self.branch_arbiter_counts) if self.branch_arbiter_counts is not None else 0
            return int(
                base
                + role
                + joint
                + suppress
                + arbiter
                + edge_cleanup
                + edge_direct
                + edge_transient
                + edge_boost
                + edge_homeostasis
                + edge_runner_arbiter
                + edge_affinity
            )
        base = int(np.count_nonzero(self.prototype_counts)) if self.prototype_counts is not None else 0
        return int(
            base
            + edge_cleanup
            + edge_direct
            + edge_transient
            + edge_boost
            + edge_homeostasis
            + edge_runner_arbiter
            + edge_affinity
        )


class AnswerSlotReadoutMemory:
    """
    Default-off local answer-position readout.

    The wrapped memory still learns the ordinary next-token stream.  At answer
    token positions, a small slot-specific prototype bank adds local evidence
    for slot 0, slot 1, ... without storing raw prompt text or using BP.
    """

    def __init__(
        self,
        base: Any,
        slot_count: int,
        slots: int,
        lr: float,
        wrong_lr: float,
        score_scale: float,
        margin: float,
        update_base: bool,
        feature_mode: str,
        direct_pre_margin_protect: float,
        direct_margin_min: float,
        coupling_slots: int,
        coupling_lr: float,
        coupling_wrong_lr: float,
        coupling_score_scale: float,
        wrong_cleanup_slots: int,
        wrong_cleanup_lr: float,
        wrong_cleanup_disinhibit_lr: float,
        wrong_cleanup_score_scale: float,
        wrong_cleanup_min_slot: int,
        wrong_cleanup_protect_mode: str,
        wrong_cleanup_protect_threshold: float,
        conflict_rescue_slots: int,
        conflict_rescue_lr: float,
        conflict_rescue_score_scale: float,
        conflict_rescue_top_k: int,
        conflict_rescue_min_slot: int,
        conflict_rescue_min_support: float,
        conflict_rescue_prefix_gate: str,
        conflict_rescue_prefix_margin: float,
        predicted_prefix_credit: str,
        predicted_prefix_skip_teacher_match: bool,
        predicted_prefix_target_top_k: int,
        predicted_prefix_lr_scale: float,
        predicted_prefix_coupling_wrong_credit: bool,
        candidate_arbiter_rank: int,
        candidate_arbiter_lr: float,
        candidate_arbiter_score_scale: float,
        candidate_arbiter_top_k: int,
        candidate_arbiter_min_support: float,
        candidate_arbiter_min_slot: int,
        candidate_arbiter_projection_decay: float,
        candidate_arbiter_clip: float,
        seed: int,
    ) -> None:
        self.base = base
        self.slot_count = max(int(slot_count), 1)
        self.slots = max(int(slots), 1)
        self.lr = float(np.clip(lr, 0.0, 1.0))
        self.wrong_lr = float(np.clip(wrong_lr, 0.0, 1.0))
        self.score_scale = float(max(score_scale, 0.0))
        self.margin = float(max(margin, 0.0))
        self.update_base = bool(update_base)
        self.feature_mode = str(feature_mode)
        self.direct_pre_margin_protect = float(max(direct_pre_margin_protect, 0.0))
        self.direct_margin_min = float(max(direct_margin_min, 0.0))
        self.coupling_slots = max(int(coupling_slots), 1)
        self.coupling_lr = float(np.clip(coupling_lr, 0.0, 1.0))
        self.coupling_wrong_lr = float(np.clip(coupling_wrong_lr, 0.0, 1.0))
        self.coupling_score_scale = float(max(coupling_score_scale, 0.0))
        self.wrong_cleanup_slots = max(int(wrong_cleanup_slots), 1)
        self.wrong_cleanup_lr = float(np.clip(wrong_cleanup_lr, 0.0, 1.0))
        self.wrong_cleanup_disinhibit_lr = float(np.clip(wrong_cleanup_disinhibit_lr, 0.0, 1.0))
        self.wrong_cleanup_score_scale = float(max(wrong_cleanup_score_scale, 0.0))
        self.wrong_cleanup_min_slot = max(int(wrong_cleanup_min_slot), 0)
        self.wrong_cleanup_protect_mode = str(wrong_cleanup_protect_mode)
        self.wrong_cleanup_protect_threshold = float(max(wrong_cleanup_protect_threshold, 0.0))
        if self.wrong_cleanup_protect_mode not in {"none", "positive_delta"}:
            raise ValueError(f"unknown wrong-cleanup protect mode: {self.wrong_cleanup_protect_mode}")
        self.conflict_rescue_slots = max(int(conflict_rescue_slots), 1)
        self.conflict_rescue_lr = float(np.clip(conflict_rescue_lr, 0.0, 1.0))
        self.conflict_rescue_score_scale = float(max(conflict_rescue_score_scale, 0.0))
        self.conflict_rescue_top_k = max(int(conflict_rescue_top_k), 1)
        self.conflict_rescue_min_slot = max(int(conflict_rescue_min_slot), 0)
        self.conflict_rescue_min_support = float(max(conflict_rescue_min_support, 0.0))
        self.conflict_rescue_prefix_gate = str(conflict_rescue_prefix_gate)
        self.conflict_rescue_prefix_margin = float(max(conflict_rescue_prefix_margin, 0.0))
        if self.conflict_rescue_prefix_gate not in {
            "none",
            "observed_pred",
            "margin",
            "observed_pred_margin",
        }:
            raise ValueError(
                f"unknown conflict-rescue prefix gate: {self.conflict_rescue_prefix_gate}"
            )
        self.predicted_prefix_credit = str(predicted_prefix_credit)
        if self.predicted_prefix_credit not in {"none", "coupling", "conflict", "coupling_conflict"}:
            raise ValueError(f"unknown predicted-prefix credit: {self.predicted_prefix_credit}")
        self.predicted_prefix_skip_teacher_match = bool(predicted_prefix_skip_teacher_match)
        self.predicted_prefix_target_top_k = max(int(predicted_prefix_target_top_k), 0)
        self.predicted_prefix_lr_scale = float(
            np.clip(float(predicted_prefix_lr_scale), 0.0, 1.0)
        )
        self.predicted_prefix_coupling_wrong_credit = bool(
            predicted_prefix_coupling_wrong_credit
        )
        self.candidate_arbiter_rank = max(int(candidate_arbiter_rank), 1)
        self.candidate_arbiter_lr = float(np.clip(candidate_arbiter_lr, 0.0, 1.0))
        self.candidate_arbiter_score_scale = float(max(candidate_arbiter_score_scale, 0.0))
        self.candidate_arbiter_top_k = max(int(candidate_arbiter_top_k), 1)
        self.candidate_arbiter_min_support = float(max(candidate_arbiter_min_support, 0.0))
        self.candidate_arbiter_min_slot = max(int(candidate_arbiter_min_slot), 0)
        self.candidate_arbiter_projection_decay = float(
            np.clip(candidate_arbiter_projection_decay, 0.0, 1.0)
        )
        self.candidate_arbiter_clip = float(max(candidate_arbiter_clip, 0.0))
        self.candidate_arbiter_seed = int(seed)
        if self.feature_mode not in {
            "base",
            "role_hop",
            "edge_path",
            "edge_path_wta",
            "edge_path_soft",
            "edge_path_soft_direct",
        }:
            raise ValueError(f"unknown answer-slot feature mode: {self.feature_mode}")
        self.prototypes: np.ndarray | None = None
        self.counts: np.ndarray | None = None
        self.feature_dim = 0
        self.coupling_prototypes: np.ndarray | None = None
        self.coupling_counts: np.ndarray | None = None
        self.coupling_feature_dim = 0
        self.wrong_cleanup_prototypes: np.ndarray | None = None
        self.wrong_cleanup_counts: np.ndarray | None = None
        self.wrong_cleanup_feature_dim = 0
        self.conflict_rescue_prototypes: np.ndarray | None = None
        self.conflict_rescue_counts: np.ndarray | None = None
        self.conflict_rescue_feature_dim = 0
        self.candidate_arbiter_encoder: np.ndarray | None = None
        self.candidate_arbiter_projection: np.ndarray | None = None
        self.candidate_arbiter_feature_dim = 0
        self.candidate_arbiter_code_dim = 0
        self.slot_updates = np.zeros(self.slot_count, dtype=np.int64)
        self.slot_wrong_updates = np.zeros(self.slot_count, dtype=np.int64)
        self.coupling_updates = np.zeros(self.slot_count, dtype=np.int64)
        self.coupling_wrong_updates = np.zeros(self.slot_count, dtype=np.int64)
        self.wrong_cleanup_updates = np.zeros(self.slot_count, dtype=np.int64)
        self.wrong_cleanup_disinhibit_updates = np.zeros(self.slot_count, dtype=np.int64)
        self.conflict_rescue_updates = np.zeros(self.slot_count, dtype=np.int64)
        self.candidate_arbiter_updates = np.zeros(self.slot_count, dtype=np.int64)
        self.candidate_arbiter_score_checks = 0
        self.candidate_arbiter_support_gate_checks = 0
        self.candidate_arbiter_support_gate_blocked = 0
        self.wrong_cleanup_score_checks = 0
        self.wrong_cleanup_score_protected = 0
        self.conflict_rescue_score_checks = 0
        self.conflict_rescue_score_applied = 0
        self.conflict_rescue_support_gate_checks = 0
        self.conflict_rescue_support_gate_blocked = 0
        self.conflict_rescue_prefix_gate_checks = 0
        self.conflict_rescue_prefix_gate_applied = 0
        self.conflict_rescue_prefix_gate_blocked = 0
        self.predicted_prefix_checks = 0
        self.predicted_prefix_updates = 0
        self.predicted_prefix_skipped_teacher_match = 0
        self.predicted_prefix_target_rank_skips = 0
        self.predicted_prefix_coupling_updates = 0
        self.predicted_prefix_coupling_wrong_updates = 0
        self.predicted_prefix_conflict_updates = 0
        self.direct_gate_checks = 0
        self.direct_gate_applied = 0
        self.direct_gate_protected = 0
        self.direct_gate_weak = 0
        self._last_answer_slot_components: dict[str, Any] | None = None
        self._last_scored_answer_slot = -1
        self._last_scored_answer_pred = -1
        self._last_scored_answer_margin = 0.0
        self._last_observed_answer_slot = -1
        self._last_observed_answer_token = -1
        self._last_observed_answer_pred = -1
        self._last_observed_answer_margin = 0.0
        self._last_observed_answer_agree = False
        self._last_conflict_rescue_gate_allowed = True
        self._last_conflict_rescue_gate_reason = "none"
        self._last_conflict_rescue_gate_prev_margin = 0.0
        self._last_conflict_rescue_gate_prev_agree = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def reset_dynamic_state(self) -> None:
        if hasattr(self.base, "reset_dynamic_state"):
            self.base.reset_dynamic_state()
        self._last_scored_answer_slot = -1
        self._last_scored_answer_pred = -1
        self._last_scored_answer_margin = 0.0
        self._last_observed_answer_slot = -1
        self._last_observed_answer_token = -1
        self._last_observed_answer_pred = -1
        self._last_observed_answer_margin = 0.0
        self._last_observed_answer_agree = False
        self._last_conflict_rescue_gate_allowed = True
        self._last_conflict_rescue_gate_reason = "none"
        self._last_conflict_rescue_gate_prev_margin = 0.0
        self._last_conflict_rescue_gate_prev_agree = False

    def record_answer_slot_observation(self, token: int) -> None:
        if self._last_scored_answer_slot < 0:
            return
        self._last_observed_answer_slot = int(self._last_scored_answer_slot)
        self._last_observed_answer_token = int(token)
        self._last_observed_answer_pred = int(self._last_scored_answer_pred)
        self._last_observed_answer_margin = float(self._last_scored_answer_margin)
        self._last_observed_answer_agree = int(token) == int(self._last_scored_answer_pred)

    def observe(self, context: Sequence[int] | np.ndarray, target: int) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(context, target)
        self.record_answer_slot_observation(int(target))

    def slot_index(self, slot: int) -> int:
        return min(max(int(slot), 0), self.slot_count - 1)

    def slot_feature(self, context: Sequence[int] | np.ndarray, slot: int) -> np.ndarray:
        if self.feature_mode != "base" and hasattr(self.base, "answer_slot_feature"):
            feature = self.base.answer_slot_feature(context, int(slot), self.feature_mode)
        elif hasattr(self.base, "feature"):
            feature = self.base.feature(context)
        else:
            feature = self.base.scores(context)
        return phase.normalize_vector(np.asarray(feature, dtype=np.float32))

    def ensure_banks(self, feature: np.ndarray) -> None:
        if self.prototypes is not None and self.counts is not None:
            return
        self.feature_dim = int(feature.shape[0])
        self.prototypes = np.zeros(
            (self.slot_count, self.vocab_size, self.slots, self.feature_dim),
            dtype=np.float32,
        )
        self.counts = np.zeros((self.slot_count, self.vocab_size, self.slots), dtype=np.float32)

    def previous_token_code(self, context: Sequence[int] | np.ndarray) -> np.ndarray | None:
        if not hasattr(self.base, "token_codes"):
            return None
        tokens = list(context)
        if not tokens:
            return None
        prev = int(tokens[-1])
        token_codes = np.asarray(self.base.token_codes, dtype=np.float32)
        if prev < 0 or prev >= token_codes.shape[0]:
            return None
        return token_codes[prev]

    @staticmethod
    def resized_code(code: np.ndarray, dim: int) -> np.ndarray:
        if dim % code.shape[0] == 0:
            repeats = dim // code.shape[0]
            return np.tile(code, repeats)
        return np.resize(code, dim)

    def token_code(self, token: int) -> np.ndarray | None:
        if not hasattr(self.base, "token_codes"):
            return None
        token_codes = np.asarray(self.base.token_codes, dtype=np.float32)
        token = int(token)
        if token < 0 or token >= token_codes.shape[0]:
            return None
        return token_codes[token]

    def coupling_feature(self, context: Sequence[int] | np.ndarray, feature: np.ndarray, slot: int) -> np.ndarray | None:
        if (
            self.coupling_score_scale <= 0.0
            and self.wrong_cleanup_score_scale <= 0.0
            and self.conflict_rescue_score_scale <= 0.0
            and self.candidate_arbiter_score_scale <= 0.0
        ) or int(slot) <= 0:
            return None
        code = self.previous_token_code(context)
        if code is None:
            return None
        if feature.shape[0] % code.shape[0] == 0:
            repeats = feature.shape[0] // code.shape[0]
            code_feature = np.tile(code, repeats)
        else:
            code_feature = np.resize(code, feature.shape[0])
        return phase.normalize_vector((feature * code_feature).astype(np.float32))

    def conflict_pair_feature(self, feature: np.ndarray, winner: int, candidate: int) -> np.ndarray | None:
        winner_code = self.token_code(winner)
        candidate_code = self.token_code(candidate)
        if winner_code is None or candidate_code is None:
            return None
        dim = int(feature.shape[0])
        winner_feature = self.resized_code(winner_code, dim)
        candidate_feature = self.resized_code(candidate_code, dim)
        return phase.normalize_vector((feature * winner_feature * candidate_feature).astype(np.float32))

    def conflict_feature(
        self,
        context: Sequence[int] | np.ndarray,
        feature: np.ndarray,
        slot: int,
    ) -> np.ndarray | None:
        coupling_feature = self.coupling_feature(context, feature, slot)
        if coupling_feature is not None:
            return coupling_feature
        if self.conflict_rescue_score_scale <= 0.0 or int(slot) < self.conflict_rescue_min_slot:
            return None
        return feature

    def ensure_coupling_banks(self, feature: np.ndarray) -> None:
        if self.coupling_prototypes is not None and self.coupling_counts is not None:
            return
        self.coupling_feature_dim = int(feature.shape[0])
        self.coupling_prototypes = np.zeros(
            (self.slot_count, self.vocab_size, self.coupling_slots, self.coupling_feature_dim),
            dtype=np.float32,
        )
        self.coupling_counts = np.zeros(
            (self.slot_count, self.vocab_size, self.coupling_slots),
            dtype=np.float32,
        )

    def ensure_wrong_cleanup_banks(self, feature: np.ndarray) -> None:
        if self.wrong_cleanup_prototypes is not None and self.wrong_cleanup_counts is not None:
            return
        self.wrong_cleanup_feature_dim = int(feature.shape[0])
        self.wrong_cleanup_prototypes = np.zeros(
            (self.slot_count, self.vocab_size, self.wrong_cleanup_slots, self.wrong_cleanup_feature_dim),
            dtype=np.float32,
        )
        self.wrong_cleanup_counts = np.zeros(
            (self.slot_count, self.vocab_size, self.wrong_cleanup_slots),
            dtype=np.float32,
        )

    def ensure_conflict_rescue_banks(self, feature: np.ndarray) -> None:
        if self.conflict_rescue_prototypes is not None and self.conflict_rescue_counts is not None:
            return
        self.conflict_rescue_feature_dim = int(feature.shape[0])
        self.conflict_rescue_prototypes = np.zeros(
            (self.slot_count, self.vocab_size, self.conflict_rescue_slots, self.conflict_rescue_feature_dim),
            dtype=np.float32,
        )
        self.conflict_rescue_counts = np.zeros(
            (self.slot_count, self.vocab_size, self.conflict_rescue_slots),
            dtype=np.float32,
        )

    def candidate_arbiter_feature(
        self,
        context: Sequence[int] | np.ndarray,
        feature: np.ndarray,
        slot: int,
    ) -> np.ndarray:
        coupling_feature = self.coupling_feature(context, feature, slot)
        if coupling_feature is not None:
            return coupling_feature
        return feature.astype(np.float32, copy=False)

    def ensure_candidate_arbiter(self, feature: np.ndarray) -> bool:
        if self.candidate_arbiter_encoder is not None and self.candidate_arbiter_projection is not None:
            return True
        if not hasattr(self.base, "token_codes"):
            return False
        token_codes = np.asarray(self.base.token_codes, dtype=np.float32)
        if token_codes.ndim != 2 or token_codes.shape[0] != self.vocab_size:
            return False
        self.candidate_arbiter_feature_dim = int(feature.shape[0])
        self.candidate_arbiter_code_dim = int(token_codes.shape[1])
        rng = np.random.default_rng(self.candidate_arbiter_seed + 7919)
        encoder = rng.standard_normal(
            (self.candidate_arbiter_rank, self.candidate_arbiter_feature_dim),
            dtype=np.float32,
        )
        self.candidate_arbiter_encoder = phase.normalize_rows(encoder)
        self.candidate_arbiter_projection = np.zeros(
            (self.candidate_arbiter_code_dim, self.candidate_arbiter_rank),
            dtype=np.float32,
        )
        return True

    def candidate_arbiter_latent(self, feature: np.ndarray) -> np.ndarray | None:
        if not self.ensure_candidate_arbiter(feature):
            return None
        assert self.candidate_arbiter_encoder is not None
        if feature.shape[0] != self.candidate_arbiter_feature_dim:
            return None
        latent = self.candidate_arbiter_encoder @ feature.astype(np.float32, copy=False)
        return phase.normalize_vector(latent.astype(np.float32))

    def candidate_arbiter_scores_from_feature(
        self,
        feature: np.ndarray,
        slot: int,
        pre_scores: np.ndarray,
        support_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        if self.candidate_arbiter_score_scale <= 0.0 or int(slot) < self.candidate_arbiter_min_slot:
            return scores
        if not self.ensure_candidate_arbiter(feature):
            return scores
        assert self.candidate_arbiter_projection is not None
        if not np.any(self.candidate_arbiter_projection):
            return scores
        latent = self.candidate_arbiter_latent(feature)
        if latent is None:
            return scores
        residual = self.candidate_arbiter_projection @ latent
        if not np.any(residual):
            return scores
        residual = phase.normalize_vector(residual.astype(np.float32))
        token_codes = np.asarray(self.base.token_codes, dtype=np.float32)
        raw_scores = float(self.candidate_arbiter_score_scale) * (token_codes @ residual)
        candidates = self.top_k_indices(np.asarray(pre_scores, dtype=np.float32), self.candidate_arbiter_top_k)
        if not candidates:
            return scores
        if self.candidate_arbiter_min_support > 0.0 and support_scores is not None:
            support = np.asarray(support_scores, dtype=np.float32)
            kept: list[int] = []
            for candidate in candidates:
                self.candidate_arbiter_support_gate_checks += 1
                if float(support[int(candidate)]) >= self.candidate_arbiter_min_support:
                    kept.append(int(candidate))
                else:
                    self.candidate_arbiter_support_gate_blocked += 1
            candidates = kept
            if not candidates:
                return scores
        local = raw_scores[candidates].astype(np.float32, copy=True)
        if local.size > 1:
            local = local - float(np.mean(local))
        scores[candidates] = local
        self.candidate_arbiter_score_checks += len(candidates)
        return scores.astype(np.float32)

    def update_candidate_arbiter(
        self,
        slot_idx: int,
        target: int,
        feature: np.ndarray,
        scores: np.ndarray,
        support_scores: np.ndarray | None = None,
    ) -> None:
        if (
            self.candidate_arbiter_score_scale <= 0.0
            or self.candidate_arbiter_lr <= 0.0
            or int(slot_idx) < self.candidate_arbiter_min_slot
        ):
            return
        if not self.ensure_candidate_arbiter(feature):
            return
        latent = self.candidate_arbiter_latent(feature)
        if latent is None or not np.any(latent):
            return
        assert self.candidate_arbiter_projection is not None
        token_codes = np.asarray(self.base.token_codes, dtype=np.float32)
        target = int(target)
        if target < 0 or target >= token_codes.shape[0]:
            return
        support: np.ndarray | None = None
        if self.candidate_arbiter_min_support > 0.0 and support_scores is not None:
            support = np.asarray(support_scores, dtype=np.float32)
            self.candidate_arbiter_support_gate_checks += 1
            if float(support[target]) < self.candidate_arbiter_min_support:
                self.candidate_arbiter_support_gate_blocked += 1
                return
        if self.candidate_arbiter_projection_decay < 1.0:
            self.candidate_arbiter_projection *= self.candidate_arbiter_projection_decay
        adjusted = scores.astype(np.float32, copy=True)
        target_score = float(adjusted[target])
        adjusted[target] = -np.inf
        wrongs = self.top_k_indices(adjusted, self.candidate_arbiter_top_k)
        applied = 0
        for wrong in wrongs:
            wrong = int(wrong)
            if wrong < 0 or wrong >= token_codes.shape[0]:
                continue
            if not np.isfinite(float(adjusted[wrong])) or float(adjusted[wrong]) <= target_score:
                continue
            if support is not None:
                self.candidate_arbiter_support_gate_checks += 1
                if float(support[wrong]) < self.candidate_arbiter_min_support:
                    self.candidate_arbiter_support_gate_blocked += 1
                    continue
            direction = phase.normalize_vector(
                (token_codes[target] - token_codes[wrong]).astype(np.float32)
            )
            step = self.candidate_arbiter_lr / max(len(wrongs), 1)
            self.candidate_arbiter_projection += step * np.outer(direction, latent).astype(np.float32)
            applied += 1
        if applied > 0:
            np.clip(
                self.candidate_arbiter_projection,
                -self.candidate_arbiter_clip,
                self.candidate_arbiter_clip,
                out=self.candidate_arbiter_projection,
            )
            self.candidate_arbiter_updates[slot_idx] += applied

    def slot_scores_from_feature(self, feature: np.ndarray, slot: int) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        if self.score_scale <= 0.0:
            return scores
        self.ensure_banks(feature)
        assert self.prototypes is not None
        assert self.counts is not None
        slot_idx = self.slot_index(slot)
        counts = self.counts[slot_idx]
        active = np.flatnonzero(np.any(counts > 0.0, axis=1))
        if active.size == 0:
            return scores
        proto = self.prototypes[slot_idx, active]
        dots = np.einsum("asd,d->as", proto, feature, optimize=True).astype(np.float32)
        dots = np.where(counts[active] > 0.0, dots, -np.inf)
        scores[active] = float(self.score_scale) * np.max(dots, axis=1)
        return scores

    def coupling_scores_from_feature(self, feature: np.ndarray, slot: int) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        if self.coupling_score_scale <= 0.0 or int(slot) <= 0:
            return scores
        self.ensure_coupling_banks(feature)
        assert self.coupling_prototypes is not None
        assert self.coupling_counts is not None
        slot_idx = self.slot_index(slot)
        counts = self.coupling_counts[slot_idx]
        active = np.flatnonzero(np.any(counts > 0.0, axis=1))
        if active.size == 0:
            return scores
        proto = self.coupling_prototypes[slot_idx, active]
        dots = np.einsum("asd,d->as", proto, feature, optimize=True).astype(np.float32)
        dots = np.where(counts[active] > 0.0, dots, -np.inf)
        scores[active] = float(self.coupling_score_scale) * np.max(dots, axis=1)
        return scores

    def wrong_cleanup_scores_from_feature(
        self,
        feature: np.ndarray,
        slot: int,
        protect_scores: np.ndarray | None = None,
    ) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        if self.wrong_cleanup_score_scale <= 0.0 or int(slot) < self.wrong_cleanup_min_slot:
            return scores
        self.ensure_wrong_cleanup_banks(feature)
        assert self.wrong_cleanup_prototypes is not None
        assert self.wrong_cleanup_counts is not None
        slot_idx = self.slot_index(slot)
        counts = self.wrong_cleanup_counts[slot_idx]
        active = np.flatnonzero(np.any(counts > 0.0, axis=1))
        if active.size == 0:
            return scores
        proto = self.wrong_cleanup_prototypes[slot_idx, active]
        dots = np.einsum("asd,d->as", proto, feature, optimize=True).astype(np.float32)
        dots = np.where(counts[active] > 0.0, dots, -np.inf)
        inhibit = np.maximum(np.max(dots, axis=1), 0.0)
        self.wrong_cleanup_score_checks += int(active.size)
        if self.wrong_cleanup_protect_mode == "positive_delta" and protect_scores is not None:
            protect = np.asarray(protect_scores, dtype=np.float32)[active] >= self.wrong_cleanup_protect_threshold
            self.wrong_cleanup_score_protected += int(np.count_nonzero(protect))
            inhibit = np.where(protect, 0.0, inhibit)
        scores[active] = -float(self.wrong_cleanup_score_scale) * inhibit
        return scores

    @staticmethod
    def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
        if k <= 0 or scores.size == 0:
            return []
        k = min(int(k), int(scores.size))
        if k >= scores.size:
            return [int(x) for x in np.argsort(scores)[::-1]]
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [int(x) for x in idx]

    def conflict_rescue_scores_from_feature(
        self,
        feature: np.ndarray,
        slot: int,
        pre_scores: np.ndarray,
        support_scores: np.ndarray,
    ) -> np.ndarray:
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        if self.conflict_rescue_score_scale <= 0.0 or int(slot) < self.conflict_rescue_min_slot:
            return scores
        self.ensure_conflict_rescue_banks(feature)
        assert self.conflict_rescue_prototypes is not None
        assert self.conflict_rescue_counts is not None
        slot_idx = self.slot_index(slot)
        counts = self.conflict_rescue_counts[slot_idx]
        if not np.any(counts > 0.0):
            return scores
        winner = int(np.argmax(pre_scores))
        pool = set(self.top_k_indices(np.asarray(pre_scores, dtype=np.float32), self.conflict_rescue_top_k))
        support_arr = np.asarray(support_scores, dtype=np.float32)
        pool.update(self.top_k_indices(support_arr, self.conflict_rescue_top_k))
        pool.discard(winner)
        for candidate in sorted(pool):
            active = counts[candidate] > 0.0
            if not np.any(active):
                continue
            if self.conflict_rescue_min_support > 0.0:
                self.conflict_rescue_support_gate_checks += 1
                if float(support_arr[candidate]) < self.conflict_rescue_min_support:
                    self.conflict_rescue_support_gate_blocked += 1
                    continue
            pair_feature = self.conflict_pair_feature(feature, winner, candidate)
            if pair_feature is None:
                continue
            dots = self.conflict_rescue_prototypes[slot_idx, candidate] @ pair_feature
            dots = np.where(active, dots, -np.inf)
            best = float(np.max(dots))
            self.conflict_rescue_score_checks += 1
            if best <= 0.0:
                continue
            scores[candidate] = max(scores[candidate], float(self.conflict_rescue_score_scale) * best)
            self.conflict_rescue_score_applied += 1
        return scores

    def update_coupling_target(self, slot_idx: int, target: int, feature: np.ndarray) -> None:
        if self.coupling_lr <= 0.0:
            return
        self.ensure_coupling_banks(feature)
        assert self.coupling_prototypes is not None
        assert self.coupling_counts is not None
        target = int(target)
        counts = self.coupling_counts[slot_idx, target]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            bank_slot = int(empty[0])
            self.coupling_prototypes[slot_idx, target, bank_slot] = feature
            self.coupling_counts[slot_idx, target, bank_slot] = 1.0
            return
        dots = self.coupling_prototypes[slot_idx, target] @ feature
        bank_slot = int(np.argmax(dots))
        self.coupling_prototypes[slot_idx, target, bank_slot] = phase.normalize_vector(
            (1.0 - self.coupling_lr) * self.coupling_prototypes[slot_idx, target, bank_slot]
            + self.coupling_lr * feature
        )
        self.coupling_counts[slot_idx, target, bank_slot] += 1.0

    def update_coupling_wrong(self, slot_idx: int, wrong: int, feature: np.ndarray) -> None:
        if self.coupling_wrong_lr <= 0.0:
            return
        self.ensure_coupling_banks(feature)
        assert self.coupling_prototypes is not None
        assert self.coupling_counts is not None
        wrong = int(wrong)
        active = self.coupling_counts[slot_idx, wrong] > 0.0
        if not np.any(active):
            return
        dots = self.coupling_prototypes[slot_idx, wrong] @ feature
        dots = np.where(active, dots, -np.inf)
        bank_slot = int(np.argmax(dots))
        self.coupling_prototypes[slot_idx, wrong, bank_slot] = phase.normalize_vector(
            self.coupling_prototypes[slot_idx, wrong, bank_slot] - self.coupling_wrong_lr * feature
        )
        self.coupling_wrong_updates[slot_idx] += 1

    def update_wrong_cleanup_wrong(self, slot_idx: int, wrong: int, feature: np.ndarray) -> None:
        if self.wrong_cleanup_score_scale <= 0.0 or self.wrong_cleanup_lr <= 0.0:
            return
        if int(slot_idx) < self.wrong_cleanup_min_slot:
            return
        self.ensure_wrong_cleanup_banks(feature)
        assert self.wrong_cleanup_prototypes is not None
        assert self.wrong_cleanup_counts is not None
        wrong = int(wrong)
        counts = self.wrong_cleanup_counts[slot_idx, wrong]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            bank_slot = int(empty[0])
            self.wrong_cleanup_prototypes[slot_idx, wrong, bank_slot] = feature
            self.wrong_cleanup_counts[slot_idx, wrong, bank_slot] = 1.0
            self.wrong_cleanup_updates[slot_idx] += 1
            return
        dots = self.wrong_cleanup_prototypes[slot_idx, wrong] @ feature
        bank_slot = int(np.argmax(dots))
        self.wrong_cleanup_prototypes[slot_idx, wrong, bank_slot] = phase.normalize_vector(
            (1.0 - self.wrong_cleanup_lr) * self.wrong_cleanup_prototypes[slot_idx, wrong, bank_slot]
            + self.wrong_cleanup_lr * feature
        )
        self.wrong_cleanup_counts[slot_idx, wrong, bank_slot] += 1.0
        self.wrong_cleanup_updates[slot_idx] += 1

    def update_wrong_cleanup_target(self, slot_idx: int, target: int, feature: np.ndarray) -> None:
        if self.wrong_cleanup_score_scale <= 0.0 or self.wrong_cleanup_disinhibit_lr <= 0.0:
            return
        if int(slot_idx) < self.wrong_cleanup_min_slot:
            return
        if self.wrong_cleanup_prototypes is None or self.wrong_cleanup_counts is None:
            return
        target = int(target)
        active = self.wrong_cleanup_counts[slot_idx, target] > 0.0
        if not np.any(active):
            return
        dots = self.wrong_cleanup_prototypes[slot_idx, target] @ feature
        dots = np.where(active, dots, -np.inf)
        bank_slot = int(np.argmax(dots))
        if float(dots[bank_slot]) <= 0.0:
            return
        self.wrong_cleanup_prototypes[slot_idx, target, bank_slot] = phase.normalize_vector(
            self.wrong_cleanup_prototypes[slot_idx, target, bank_slot]
            - self.wrong_cleanup_disinhibit_lr * feature
        )
        self.wrong_cleanup_disinhibit_updates[slot_idx] += 1

    def update_conflict_rescue_target(
        self,
        slot_idx: int,
        winner: int,
        target: int,
        feature: np.ndarray,
    ) -> None:
        if self.conflict_rescue_score_scale <= 0.0 or self.conflict_rescue_lr <= 0.0:
            return
        if int(slot_idx) < self.conflict_rescue_min_slot:
            return
        pair_feature = self.conflict_pair_feature(feature, int(winner), int(target))
        if pair_feature is None:
            return
        self.ensure_conflict_rescue_banks(pair_feature)
        assert self.conflict_rescue_prototypes is not None
        assert self.conflict_rescue_counts is not None
        target = int(target)
        counts = self.conflict_rescue_counts[slot_idx, target]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            bank_slot = int(empty[0])
            self.conflict_rescue_prototypes[slot_idx, target, bank_slot] = pair_feature
            self.conflict_rescue_counts[slot_idx, target, bank_slot] = 1.0
            self.conflict_rescue_updates[slot_idx] += 1
            return
        dots = self.conflict_rescue_prototypes[slot_idx, target] @ pair_feature
        bank_slot = int(np.argmax(dots))
        self.conflict_rescue_prototypes[slot_idx, target, bank_slot] = phase.normalize_vector(
            (1.0 - self.conflict_rescue_lr) * self.conflict_rescue_prototypes[slot_idx, target, bank_slot]
            + self.conflict_rescue_lr * pair_feature
        )
        self.conflict_rescue_counts[slot_idx, target, bank_slot] += 1.0
        self.conflict_rescue_updates[slot_idx] += 1

    @staticmethod
    def score_margin(scores: np.ndarray) -> float:
        if scores.size < 2:
            return float("inf")
        top_two = np.partition(scores.astype(np.float32, copy=False), -2)[-2:]
        return float(np.max(top_two) - np.min(top_two))

    def conflict_rescue_prefix_gate_allows(self, slot: int) -> bool:
        self._last_conflict_rescue_gate_allowed = True
        self._last_conflict_rescue_gate_reason = "none"
        self._last_conflict_rescue_gate_prev_margin = float(self._last_observed_answer_margin)
        self._last_conflict_rescue_gate_prev_agree = bool(self._last_observed_answer_agree)
        if (
            self.conflict_rescue_prefix_gate == "none"
            or self.conflict_rescue_score_scale <= 0.0
            or int(slot) < self.conflict_rescue_min_slot
        ):
            return True
        self.conflict_rescue_prefix_gate_checks += 1
        prev_slot = int(slot) - 1
        if self._last_observed_answer_slot != prev_slot:
            self.conflict_rescue_prefix_gate_blocked += 1
            self._last_conflict_rescue_gate_allowed = False
            self._last_conflict_rescue_gate_reason = "missing_prev_slot"
            return False
        requires_agree = self.conflict_rescue_prefix_gate in {"observed_pred", "observed_pred_margin"}
        if requires_agree and not self._last_observed_answer_agree:
            self.conflict_rescue_prefix_gate_blocked += 1
            self._last_conflict_rescue_gate_allowed = False
            self._last_conflict_rescue_gate_reason = "prev_not_model_pred"
            return False
        requires_margin = self.conflict_rescue_prefix_gate in {"margin", "observed_pred_margin"}
        if requires_margin and self._last_observed_answer_margin < self.conflict_rescue_prefix_margin:
            self.conflict_rescue_prefix_gate_blocked += 1
            self._last_conflict_rescue_gate_allowed = False
            self._last_conflict_rescue_gate_reason = "prev_margin_low"
            return False
        self.conflict_rescue_prefix_gate_applied += 1
        self._last_conflict_rescue_gate_allowed = True
        self._last_conflict_rescue_gate_reason = "applied"
        return True

    @staticmethod
    def direct_delta_margin(delta: np.ndarray) -> tuple[float, float]:
        active = np.flatnonzero(np.abs(delta) > 1e-8)
        if active.size == 0:
            return 0.0, 0.0
        active_delta = delta[active].astype(np.float32, copy=False)
        top = float(np.max(active_delta))
        if active_delta.size < 2:
            return top, top
        top_two = np.partition(active_delta, -2)[-2:]
        return top, float(np.max(top_two) - np.min(top_two))

    def gated_direct_delta(self, pre_direct_scores: np.ndarray, direct_delta: np.ndarray) -> np.ndarray:
        if self.feature_mode != "edge_path_soft_direct":
            return direct_delta
        if self.direct_pre_margin_protect <= 0.0 and self.direct_margin_min <= 0.0:
            return direct_delta
        self.direct_gate_checks += 1
        top_delta, direct_margin = self.direct_delta_margin(direct_delta)
        if top_delta <= 0.0 or (
            self.direct_margin_min > 0.0 and direct_margin < self.direct_margin_min
        ):
            self.direct_gate_weak += 1
            return np.zeros_like(direct_delta)
        pre_margin = self.score_margin(pre_direct_scores)
        if self.direct_pre_margin_protect > 0.0 and pre_margin >= self.direct_pre_margin_protect:
            self.direct_gate_protected += 1
            return np.zeros_like(direct_delta)
        self.direct_gate_applied += 1
        return direct_delta

    @staticmethod
    def target_vs_best_wrong_score(scores: np.ndarray, target: int) -> float:
        arr = np.asarray(scores, dtype=np.float32)
        target = int(target)
        if target < 0 or target >= arr.size:
            return float("nan")
        adjusted = arr.copy()
        adjusted[target] = -np.inf
        return float(arr[target] - np.max(adjusted))

    def counterfactual_answer_slot_scores(
        self,
        context: Sequence[int] | np.ndarray,
        slot: int,
        feature: np.ndarray,
        base_scores: np.ndarray,
    ) -> np.ndarray:
        slot_idx = self.slot_index(slot)
        scores = np.asarray(base_scores, dtype=np.float32).copy()
        slot_delta = self.slot_scores_from_feature(feature, slot_idx)
        scores = scores + slot_delta
        coupling_feature = self.coupling_feature(context, feature, slot_idx)
        if coupling_feature is not None:
            scores = scores + self.coupling_scores_from_feature(coupling_feature, slot_idx)
            protect_scores = None
            if self.wrong_cleanup_protect_mode == "positive_delta":
                protect_scores = np.maximum(slot_delta, 0.0)
            scores = scores + self.wrong_cleanup_scores_from_feature(
                coupling_feature,
                slot_idx,
                protect_scores,
            )
        return scores.astype(np.float32)

    def composed_answer_slot_scores(
        self,
        context: Sequence[int] | np.ndarray,
        slot: int,
        feature: np.ndarray,
    ) -> np.ndarray:
        base_scores = self.base.scores(context).astype(np.float32)
        slot_delta = self.slot_scores_from_feature(feature, slot)
        after_slot_scores = base_scores + slot_delta
        scores = after_slot_scores
        coupling_delta = np.zeros(self.vocab_size, dtype=np.float32)
        coupling_feature = self.coupling_feature(context, feature, slot)
        if coupling_feature is not None:
            coupling_delta = self.coupling_scores_from_feature(coupling_feature, slot)
            scores = scores + coupling_delta
        after_coupling_scores = scores
        wrong_cleanup_delta = np.zeros(self.vocab_size, dtype=np.float32)
        positive_support = np.maximum(slot_delta, 0.0) + np.maximum(coupling_delta, 0.0)
        if coupling_feature is not None:
            protect_scores = None
            if self.wrong_cleanup_protect_mode == "positive_delta":
                protect_scores = positive_support
            wrong_cleanup_delta = self.wrong_cleanup_scores_from_feature(
                coupling_feature,
                slot,
                protect_scores,
            )
            scores = scores + wrong_cleanup_delta
        after_cleanup_scores = scores
        conflict_rescue_delta = np.zeros(self.vocab_size, dtype=np.float32)
        conflict_prefix_allowed = self.conflict_rescue_prefix_gate_allows(slot)
        conflict_feature = self.conflict_feature(context, feature, slot)
        if conflict_feature is not None and conflict_prefix_allowed:
            conflict_rescue_delta = self.conflict_rescue_scores_from_feature(
                conflict_feature,
                slot,
                scores,
                positive_support,
            )
            scores = scores + conflict_rescue_delta
        after_conflict_scores = scores
        direct_delta_raw = np.zeros(self.vocab_size, dtype=np.float32)
        direct_delta = direct_delta_raw
        if hasattr(self.base, "answer_slot_score_delta"):
            direct_delta_raw = self.base.answer_slot_score_delta(context, int(slot), self.feature_mode)
            direct_delta = self.gated_direct_delta(scores, direct_delta_raw)
            scores = scores + direct_delta
        after_direct_scores = scores
        candidate_support = (
            positive_support
            + np.maximum(conflict_rescue_delta, 0.0)
            + np.maximum(direct_delta, 0.0)
        ).astype(np.float32)
        candidate_arbiter_delta = np.zeros(self.vocab_size, dtype=np.float32)
        candidate_feature = self.candidate_arbiter_feature(context, feature, slot)
        candidate_arbiter_delta = self.candidate_arbiter_scores_from_feature(
            candidate_feature,
            slot,
            scores,
            candidate_support,
        )
        scores = scores + candidate_arbiter_delta
        final_scores = scores.astype(np.float32)
        runner_counterfactual_scores = None
        if hasattr(self.base, "edge_path_last_candidate_answer_feature"):
            runner_feature = self.base.edge_path_last_candidate_answer_feature(int(slot), "runner")
            if runner_feature is not None and runner_feature.shape == feature.shape:
                runner_counterfactual_scores = self.counterfactual_answer_slot_scores(
                    context,
                    int(slot),
                    runner_feature,
                    base_scores,
                )
        self._last_answer_slot_components = {
            "slot": int(slot),
            "feature_mode": self.feature_mode,
            "base_scores": base_scores,
            "slot_delta": slot_delta.astype(np.float32),
            "after_slot_scores": after_slot_scores.astype(np.float32),
            "coupling_delta": coupling_delta.astype(np.float32),
            "after_coupling_scores": after_coupling_scores.astype(np.float32),
            "wrong_cleanup_delta": wrong_cleanup_delta.astype(np.float32),
            "after_cleanup_scores": after_cleanup_scores.astype(np.float32),
            "conflict_rescue_delta": conflict_rescue_delta.astype(np.float32),
            "after_conflict_scores": after_conflict_scores.astype(np.float32),
            "direct_delta_raw": direct_delta_raw.astype(np.float32),
            "direct_delta": direct_delta.astype(np.float32),
            "after_direct_scores": after_direct_scores.astype(np.float32),
            "candidate_support": candidate_support.astype(np.float32),
            "candidate_arbiter_delta": candidate_arbiter_delta.astype(np.float32),
            "final_scores": final_scores,
            "runner_counterfactual_scores": (
                runner_counterfactual_scores.astype(np.float32)
                if runner_counterfactual_scores is not None
                else np.zeros(0, dtype=np.float32)
            ),
            "conflict_rescue_prefix_gate_allowed": int(conflict_prefix_allowed),
            "conflict_rescue_prefix_gate_reason": self._last_conflict_rescue_gate_reason,
            "conflict_rescue_prefix_gate_prev_margin": self._last_conflict_rescue_gate_prev_margin,
            "conflict_rescue_prefix_gate_prev_agree": int(self._last_conflict_rescue_gate_prev_agree),
        }
        if hasattr(self.base, "edge_path_last_candidate_metrics"):
            self._last_answer_slot_components.update(
                self.base.edge_path_last_candidate_metrics(int(slot))
            )
        self._last_scored_answer_slot = int(slot)
        self._last_scored_answer_pred = int(np.argmax(final_scores)) if final_scores.size else -1
        self._last_scored_answer_margin = self.score_margin(final_scores)
        return final_scores

    @staticmethod
    def score_component_summary(scores: np.ndarray, target: int, prefix: str) -> dict[str, Any]:
        arr = np.asarray(scores, dtype=np.float32)
        if arr.size == 0:
            return {
                f"{prefix}_pred_id": -1,
                f"{prefix}_pred_score": float("nan"),
                f"{prefix}_target_score": float("nan"),
                f"{prefix}_margin": float("nan"),
                f"{prefix}_target_vs_best_wrong": float("nan"),
                f"{prefix}_target_rank": -1,
                f"{prefix}_active_count": 0,
            }
        target = int(target)
        pred = int(np.argmax(arr))
        pred_score = float(arr[pred])
        target_score = float(arr[target]) if 0 <= target < arr.size else float("nan")
        if arr.size >= 2:
            top_two = np.partition(arr, -2)[-2:]
            margin = float(np.max(top_two) - np.min(top_two))
        else:
            margin = float("inf")
        if 0 <= target < arr.size:
            without_target = arr.copy()
            without_target[target] = -np.inf
            best_wrong = int(np.argmax(without_target))
            target_vs_best = float(target_score - without_target[best_wrong])
            target_rank = int(1 + np.count_nonzero(arr > target_score))
        else:
            target_vs_best = float("nan")
            target_rank = -1
        return {
            f"{prefix}_pred_id": pred,
            f"{prefix}_pred_score": pred_score,
            f"{prefix}_target_score": target_score,
            f"{prefix}_margin": margin,
            f"{prefix}_target_vs_best_wrong": target_vs_best,
            f"{prefix}_target_rank": target_rank,
            f"{prefix}_active_count": int(np.count_nonzero(np.abs(arr) > 1e-8)),
        }

    @staticmethod
    def target_probability(scores: np.ndarray, target: int, temperature: float) -> float:
        target = int(target)
        if target < 0 or target >= len(scores):
            return float("nan")
        probs = phase.softmax(np.asarray(scores, dtype=np.float32), temperature)
        return float(probs[target])

    def last_answer_slot_component_metrics(
        self,
        target: int,
        pred: int,
        target_prob: float,
        temperature: float,
    ) -> dict[str, Any]:
        comp = self._last_answer_slot_components or {}
        row: dict[str, Any] = {
            "slot": int(comp.get("slot", -1)),
            "feature_mode": str(comp.get("feature_mode", "")),
            "target_id": int(target),
            "prediction_id": int(pred),
            "target_prob": float(target_prob),
            "correct": int(pred == int(target)),
        }
        for key, prefix in [
            ("base_scores", "base"),
            ("slot_delta", "slot_delta"),
            ("after_slot_scores", "after_slot"),
            ("coupling_delta", "coupling_delta"),
            ("after_coupling_scores", "after_coupling"),
            ("wrong_cleanup_delta", "wrong_cleanup_delta"),
            ("after_cleanup_scores", "after_cleanup"),
            ("conflict_rescue_delta", "conflict_rescue_delta"),
            ("after_conflict_scores", "after_conflict"),
            ("direct_delta", "direct_delta"),
            ("after_direct_scores", "after_direct"),
            ("candidate_support", "candidate_support"),
            ("candidate_arbiter_delta", "candidate_arbiter_delta"),
            ("runner_counterfactual_scores", "runner_counterfactual"),
            ("final_scores", "final"),
        ]:
            if key in comp:
                row.update(self.score_component_summary(comp[key], int(target), prefix))
        for key, prefix in [
            ("base_scores", "base"),
            ("after_slot_scores", "after_slot"),
            ("after_coupling_scores", "after_coupling"),
            ("after_cleanup_scores", "after_cleanup"),
            ("after_conflict_scores", "after_conflict"),
            ("after_direct_scores", "after_direct"),
            ("final_scores", "final"),
        ]:
            if key in comp:
                row[f"{prefix}_target_prob"] = self.target_probability(comp[key], int(target), temperature)
        if "base_target_prob" in row:
            row["final_target_prob_gain_vs_base"] = float(row.get("final_target_prob", 0.0)) - float(
                row["base_target_prob"]
            )
            row["after_slot_target_prob_gain_vs_base"] = float(
                row.get("after_slot_target_prob", 0.0)
            ) - float(row["base_target_prob"])
            row["after_coupling_target_prob_gain_vs_base"] = float(
                row.get("after_coupling_target_prob", 0.0)
            ) - float(row["base_target_prob"])
            row["after_cleanup_target_prob_gain_vs_base"] = float(
                row.get("after_cleanup_target_prob", 0.0)
            ) - float(row["base_target_prob"])
            row["after_conflict_target_prob_gain_vs_base"] = float(
                row.get("after_conflict_target_prob", 0.0)
            ) - float(row["base_target_prob"])
            row["after_direct_target_prob_gain_vs_base"] = float(
                row.get("after_direct_target_prob", 0.0)
            ) - float(row["base_target_prob"])
        for key in [
            "conflict_rescue_prefix_gate_allowed",
            "conflict_rescue_prefix_gate_reason",
            "conflict_rescue_prefix_gate_prev_margin",
            "conflict_rescue_prefix_gate_prev_agree",
        ]:
            if key in comp:
                row[key] = comp[key]
        for key, value in comp.items():
            if key.startswith("edge_path_"):
                row[key] = value
        return row

    def scores(self, context: Sequence[int] | np.ndarray) -> np.ndarray:
        return self.base.scores(context)

    def scores_with_answer_slot(self, context: Sequence[int] | np.ndarray, slot: int) -> np.ndarray:
        feature = self.slot_feature(context, int(slot))
        return self.composed_answer_slot_scores(context, int(slot), feature)

    def update_bank_target(self, slot_idx: int, target: int, feature: np.ndarray) -> None:
        self.ensure_banks(feature)
        assert self.prototypes is not None
        assert self.counts is not None
        target = int(target)
        counts = self.counts[slot_idx, target]
        empty = np.flatnonzero(counts <= 0.0)
        if empty.size:
            bank_slot = int(empty[0])
            self.prototypes[slot_idx, target, bank_slot] = feature
            self.counts[slot_idx, target, bank_slot] = 1.0
            return
        dots = self.prototypes[slot_idx, target] @ feature
        bank_slot = int(np.argmax(dots))
        self.prototypes[slot_idx, target, bank_slot] = phase.normalize_vector(
            (1.0 - self.lr) * self.prototypes[slot_idx, target, bank_slot] + self.lr * feature
        )
        self.counts[slot_idx, target, bank_slot] += 1.0

    def update_bank_wrong(self, slot_idx: int, wrong: int, feature: np.ndarray) -> None:
        if self.wrong_lr <= 0.0:
            return
        self.ensure_banks(feature)
        assert self.prototypes is not None
        assert self.counts is not None
        wrong = int(wrong)
        active = self.counts[slot_idx, wrong] > 0.0
        if not np.any(active):
            return
        dots = self.prototypes[slot_idx, wrong] @ feature
        dots = np.where(active, dots, -np.inf)
        bank_slot = int(np.argmax(dots))
        self.prototypes[slot_idx, wrong, bank_slot] = phase.normalize_vector(
            self.prototypes[slot_idx, wrong, bank_slot] - self.wrong_lr * feature
        )
        self.slot_wrong_updates[slot_idx] += 1

    def update(self, context: Sequence[int] | np.ndarray, target: int) -> None:
        self.base.update(context, target)

    def update_answer_slot(self, context: Sequence[int] | np.ndarray, target: int, slot: int) -> None:
        target = int(target)
        slot_idx = self.slot_index(slot)
        feature = self.slot_feature(context, slot_idx)
        scores = self.composed_answer_slot_scores(context, slot_idx, feature)
        target_score = float(scores[target])
        adjusted = scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        wrong = int(np.argmax(adjusted))
        should_apply_credit = float(adjusted[wrong]) + self.margin > target_score
        if hasattr(self.base, "update_answer_slot_feature"):
            runner_counterfactual_margin_gain = None
            if self._last_answer_slot_components is not None:
                runner_scores = self._last_answer_slot_components.get("runner_counterfactual_scores")
                if runner_scores is not None and np.asarray(runner_scores).size:
                    current_margin = self.target_vs_best_wrong_score(scores, target)
                    runner_margin = self.target_vs_best_wrong_score(runner_scores, target)
                    if math.isfinite(current_margin) and math.isfinite(runner_margin):
                        runner_counterfactual_margin_gain = float(runner_margin - current_margin)
            self.base.update_answer_slot_feature(
                context,
                target,
                wrong,
                slot_idx,
                should_apply_credit,
                self.feature_mode,
                runner_counterfactual_margin_gain,
            )
        if self.update_base:
            self.base.update(context, target)
        self.update_bank_target(slot_idx, target, feature)
        self.slot_updates[slot_idx] += 1
        if should_apply_credit:
            self.update_bank_wrong(slot_idx, wrong, feature)
        coupling_feature = self.coupling_feature(context, feature, slot_idx)
        if coupling_feature is not None:
            if self.coupling_score_scale > 0.0:
                self.update_coupling_target(slot_idx, target, coupling_feature)
                self.coupling_updates[slot_idx] += 1
                if should_apply_credit:
                    self.update_coupling_wrong(slot_idx, wrong, coupling_feature)
            self.update_wrong_cleanup_target(slot_idx, target, coupling_feature)
            if should_apply_credit:
                self.update_wrong_cleanup_wrong(slot_idx, wrong, coupling_feature)
        conflict_feature = self.conflict_feature(context, feature, slot_idx)
        if should_apply_credit and conflict_feature is not None and self._last_conflict_rescue_gate_allowed:
            self.update_conflict_rescue_target(slot_idx, wrong, target, conflict_feature)
        candidate_feature = self.candidate_arbiter_feature(context, feature, slot_idx)
        candidate_support = None
        if self._last_answer_slot_components is not None:
            candidate_support = self._last_answer_slot_components.get("candidate_support")
        self.update_candidate_arbiter(slot_idx, target, candidate_feature, scores, candidate_support)
        self.record_answer_slot_observation(target)

    def update_answer_slot_predicted_prefix(
        self,
        context: Sequence[int] | np.ndarray,
        target: int,
        slot: int,
        credit_mode: str,
    ) -> None:
        if credit_mode == "none":
            return
        if credit_mode not in {"coupling", "conflict", "coupling_conflict"}:
            raise ValueError(f"unknown predicted-prefix credit mode: {credit_mode}")
        target = int(target)
        slot_idx = self.slot_index(slot)
        if slot_idx <= 0:
            return
        self.predicted_prefix_checks += 1
        feature = self.slot_feature(context, slot_idx)
        scores = self.composed_answer_slot_scores(context, slot_idx, feature)
        target_score = float(scores[target])
        if self.predicted_prefix_target_top_k > 0:
            target_rank = 1 + int(np.count_nonzero(scores > target_score))
            if target_rank > self.predicted_prefix_target_top_k:
                self.predicted_prefix_target_rank_skips += 1
                return
        adjusted = scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        wrong = int(np.argmax(adjusted))
        should_apply_credit = float(adjusted[wrong]) + self.margin > target_score
        coupling_feature = self.coupling_feature(context, feature, slot_idx)
        if coupling_feature is None:
            return
        self.predicted_prefix_updates += 1
        old_coupling_lr = self.coupling_lr
        old_coupling_wrong_lr = self.coupling_wrong_lr
        old_conflict_rescue_lr = self.conflict_rescue_lr
        scale = self.predicted_prefix_lr_scale
        if scale != 1.0:
            self.coupling_lr *= scale
            self.coupling_wrong_lr *= scale
            self.conflict_rescue_lr *= scale
        try:
            if credit_mode in {"coupling", "coupling_conflict"} and self.coupling_score_scale > 0.0:
                self.update_coupling_target(slot_idx, target, coupling_feature)
                self.coupling_updates[slot_idx] += 1
                self.predicted_prefix_coupling_updates += 1
                if should_apply_credit and self.predicted_prefix_coupling_wrong_credit:
                    self.update_coupling_wrong(slot_idx, wrong, coupling_feature)
                    self.predicted_prefix_coupling_wrong_updates += 1
            if credit_mode in {"conflict", "coupling_conflict"} and should_apply_credit:
                self.update_conflict_rescue_target(slot_idx, wrong, target, coupling_feature)
                self.predicted_prefix_conflict_updates += 1
        finally:
            self.coupling_lr = old_coupling_lr
            self.coupling_wrong_lr = old_coupling_wrong_lr
            self.conflict_rescue_lr = old_conflict_rescue_lr

    def state_bytes(self) -> int:
        total = safe_state_bytes(self.base)
        if self.prototypes is not None:
            total += int(self.prototypes.nbytes)
        if self.counts is not None:
            total += int(self.counts.nbytes)
        if self.coupling_prototypes is not None:
            total += int(self.coupling_prototypes.nbytes)
        if self.coupling_counts is not None:
            total += int(self.coupling_counts.nbytes)
        if self.wrong_cleanup_prototypes is not None:
            total += int(self.wrong_cleanup_prototypes.nbytes)
        if self.wrong_cleanup_counts is not None:
            total += int(self.wrong_cleanup_counts.nbytes)
        if self.conflict_rescue_prototypes is not None:
            total += int(self.conflict_rescue_prototypes.nbytes)
        if self.conflict_rescue_counts is not None:
            total += int(self.conflict_rescue_counts.nbytes)
        if self.candidate_arbiter_encoder is not None:
            total += int(self.candidate_arbiter_encoder.nbytes)
        if self.candidate_arbiter_projection is not None:
            total += int(self.candidate_arbiter_projection.nbytes)
        total += int(self.slot_updates.nbytes + self.slot_wrong_updates.nbytes)
        total += int(self.coupling_updates.nbytes + self.coupling_wrong_updates.nbytes)
        total += int(self.wrong_cleanup_updates.nbytes + self.wrong_cleanup_disinhibit_updates.nbytes)
        total += int(self.conflict_rescue_updates.nbytes + self.candidate_arbiter_updates.nbytes)
        return total

    def active_contexts(self) -> int:
        base_active = safe_active_contexts(self.base)
        slot_active = int(np.count_nonzero(self.counts)) if self.counts is not None else 0
        coupling_active = int(np.count_nonzero(self.coupling_counts)) if self.coupling_counts is not None else 0
        cleanup_active = (
            int(np.count_nonzero(self.wrong_cleanup_counts))
            if self.wrong_cleanup_counts is not None
            else 0
        )
        conflict_active = (
            int(np.count_nonzero(self.conflict_rescue_counts))
            if self.conflict_rescue_counts is not None
            else 0
        )
        candidate_active = (
            int(np.count_nonzero(np.abs(self.candidate_arbiter_projection) > 1e-8))
            if self.candidate_arbiter_projection is not None
            else 0
        )
        return int(base_active + slot_active + coupling_active + cleanup_active + conflict_active + candidate_active)

    def answer_slot_stats(self) -> dict[str, Any]:
        return {
            "answer_slot_readout": True,
            "answer_slot_count": self.slot_count,
            "answer_slot_slots": self.slots,
            "answer_slot_lr": self.lr,
            "answer_slot_wrong_lr": self.wrong_lr,
            "answer_slot_score_scale": self.score_scale,
            "answer_slot_margin": self.margin,
            "answer_slot_update_base": self.update_base,
            "answer_slot_feature_mode": self.feature_mode,
            "answer_slot_direct_pre_margin_protect": self.direct_pre_margin_protect,
            "answer_slot_direct_margin_min": self.direct_margin_min,
            "answer_slot_direct_gate_checks": self.direct_gate_checks,
            "answer_slot_direct_gate_applied": self.direct_gate_applied,
            "answer_slot_direct_gate_protected": self.direct_gate_protected,
            "answer_slot_direct_gate_weak": self.direct_gate_weak,
            "answer_slot_coupling_slots": self.coupling_slots,
            "answer_slot_coupling_lr": self.coupling_lr,
            "answer_slot_coupling_wrong_lr": self.coupling_wrong_lr,
            "answer_slot_coupling_score_scale": self.coupling_score_scale,
            "answer_slot_coupling_feature_dim": self.coupling_feature_dim,
            "answer_slot_coupling_active_slots": (
                int(np.count_nonzero(self.coupling_counts))
                if self.coupling_counts is not None
                else 0
            ),
            "answer_slot_coupling_updates": [int(x) for x in self.coupling_updates],
            "answer_slot_coupling_wrong_updates": [int(x) for x in self.coupling_wrong_updates],
            "answer_slot_wrong_cleanup_slots": self.wrong_cleanup_slots,
            "answer_slot_wrong_cleanup_lr": self.wrong_cleanup_lr,
            "answer_slot_wrong_cleanup_disinhibit_lr": self.wrong_cleanup_disinhibit_lr,
            "answer_slot_wrong_cleanup_score_scale": self.wrong_cleanup_score_scale,
            "answer_slot_wrong_cleanup_min_slot": self.wrong_cleanup_min_slot,
            "answer_slot_wrong_cleanup_protect_mode": self.wrong_cleanup_protect_mode,
            "answer_slot_wrong_cleanup_protect_threshold": self.wrong_cleanup_protect_threshold,
            "answer_slot_wrong_cleanup_score_checks": self.wrong_cleanup_score_checks,
            "answer_slot_wrong_cleanup_score_protected": self.wrong_cleanup_score_protected,
            "answer_slot_wrong_cleanup_feature_dim": self.wrong_cleanup_feature_dim,
            "answer_slot_wrong_cleanup_active_slots": (
                int(np.count_nonzero(self.wrong_cleanup_counts))
                if self.wrong_cleanup_counts is not None
                else 0
            ),
            "answer_slot_wrong_cleanup_updates": [int(x) for x in self.wrong_cleanup_updates],
            "answer_slot_wrong_cleanup_disinhibit_updates": [
                int(x) for x in self.wrong_cleanup_disinhibit_updates
            ],
            "answer_slot_conflict_rescue_slots": self.conflict_rescue_slots,
            "answer_slot_conflict_rescue_lr": self.conflict_rescue_lr,
            "answer_slot_conflict_rescue_score_scale": self.conflict_rescue_score_scale,
            "answer_slot_conflict_rescue_top_k": self.conflict_rescue_top_k,
            "answer_slot_conflict_rescue_min_slot": self.conflict_rescue_min_slot,
            "answer_slot_conflict_rescue_min_support": self.conflict_rescue_min_support,
            "answer_slot_conflict_rescue_prefix_gate": self.conflict_rescue_prefix_gate,
            "answer_slot_conflict_rescue_prefix_margin": self.conflict_rescue_prefix_margin,
            "answer_slot_conflict_rescue_feature_dim": self.conflict_rescue_feature_dim,
            "answer_slot_conflict_rescue_active_slots": (
                int(np.count_nonzero(self.conflict_rescue_counts))
                if self.conflict_rescue_counts is not None
                else 0
            ),
            "answer_slot_conflict_rescue_updates": [int(x) for x in self.conflict_rescue_updates],
            "answer_slot_conflict_rescue_score_checks": self.conflict_rescue_score_checks,
            "answer_slot_conflict_rescue_score_applied": self.conflict_rescue_score_applied,
            "answer_slot_conflict_rescue_support_gate_checks": (
                self.conflict_rescue_support_gate_checks
            ),
            "answer_slot_conflict_rescue_support_gate_blocked": (
                self.conflict_rescue_support_gate_blocked
            ),
            "answer_slot_conflict_rescue_prefix_gate_checks": self.conflict_rescue_prefix_gate_checks,
            "answer_slot_conflict_rescue_prefix_gate_applied": self.conflict_rescue_prefix_gate_applied,
            "answer_slot_conflict_rescue_prefix_gate_blocked": self.conflict_rescue_prefix_gate_blocked,
            "answer_slot_predicted_prefix_credit": self.predicted_prefix_credit,
            "answer_slot_predicted_prefix_skip_teacher_match": self.predicted_prefix_skip_teacher_match,
            "answer_slot_predicted_prefix_target_top_k": self.predicted_prefix_target_top_k,
            "answer_slot_predicted_prefix_lr_scale": self.predicted_prefix_lr_scale,
            "answer_slot_predicted_prefix_coupling_wrong_credit": (
                self.predicted_prefix_coupling_wrong_credit
            ),
            "answer_slot_predicted_prefix_checks": self.predicted_prefix_checks,
            "answer_slot_predicted_prefix_updates": self.predicted_prefix_updates,
            "answer_slot_predicted_prefix_skipped_teacher_match": (
                self.predicted_prefix_skipped_teacher_match
            ),
            "answer_slot_predicted_prefix_target_rank_skips": (
                self.predicted_prefix_target_rank_skips
            ),
            "answer_slot_predicted_prefix_coupling_updates": self.predicted_prefix_coupling_updates,
            "answer_slot_predicted_prefix_coupling_wrong_updates": (
                self.predicted_prefix_coupling_wrong_updates
            ),
            "answer_slot_predicted_prefix_conflict_updates": self.predicted_prefix_conflict_updates,
            "answer_candidate_arbiter_rank": self.candidate_arbiter_rank,
            "answer_candidate_arbiter_lr": self.candidate_arbiter_lr,
            "answer_candidate_arbiter_score_scale": self.candidate_arbiter_score_scale,
            "answer_candidate_arbiter_top_k": self.candidate_arbiter_top_k,
            "answer_candidate_arbiter_min_support": self.candidate_arbiter_min_support,
            "answer_candidate_arbiter_min_slot": self.candidate_arbiter_min_slot,
            "answer_candidate_arbiter_projection_decay": self.candidate_arbiter_projection_decay,
            "answer_candidate_arbiter_clip": self.candidate_arbiter_clip,
            "answer_candidate_arbiter_feature_dim": self.candidate_arbiter_feature_dim,
            "answer_candidate_arbiter_code_dim": self.candidate_arbiter_code_dim,
            "answer_candidate_arbiter_score_checks": self.candidate_arbiter_score_checks,
            "answer_candidate_arbiter_support_gate_checks": (
                self.candidate_arbiter_support_gate_checks
            ),
            "answer_candidate_arbiter_support_gate_blocked": (
                self.candidate_arbiter_support_gate_blocked
            ),
            "answer_candidate_arbiter_updates": [int(x) for x in self.candidate_arbiter_updates],
            "answer_candidate_arbiter_active_weights": (
                int(np.count_nonzero(np.abs(self.candidate_arbiter_projection) > 1e-8))
                if self.candidate_arbiter_projection is not None
                else 0
            ),
            "answer_slot_feature_dim": self.feature_dim,
            "answer_slot_active_slots": int(np.count_nonzero(self.counts)) if self.counts is not None else 0,
            "answer_slot_updates": [int(x) for x in self.slot_updates],
            "answer_slot_wrong_updates": [int(x) for x in self.slot_wrong_updates],
        }


def build_memory(args: argparse.Namespace, vocab_size: int) -> tuple[str, Any, dict[str, Any]]:
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

    method = args.method
    if method == "phase_competitive_online":
        memory: Any = stream.OnlineCompetitivePhaseMemory(
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
    elif method == "phase_trace_competitive_online":
        memory = stream.OnlineTraceCompetitivePhaseMemory(
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
    elif method == "phase_trace_dll_local_competitive_online":
        memory = stream.OnlineDLLDeepLocalMemory(
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
    elif method == "phase_trace_noprop_local_competitive_online":
        memory = stream.OnlineNoPropLocalDenoisingMemory(
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
    elif method == "phase_trace_kv_competitive_online":
        memory = stream.OnlineTraceHebbianKVCompetitivePhaseMemory(
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
    elif method == "state_microproto_online":
        memory = OnlineStateMicroPrototypeMemory(
            vocab_size,
            args.state_dim,
            args.state_order,
            args.state_decay,
            args.micro_slots,
            args.micro_lr,
            args.micro_wrong_lr,
            args.micro_score_scale,
            args.phase_bias_weight,
            args.micro_margin,
            args.binding_hops,
            args.binding_window,
            args.binding_query_order,
            args.binding_query_mode,
            args.binding_focus_k,
            args.binding_decay,
            args.binding_bidirectional,
            args.binding_mode,
            args.binding_span_window,
            args.binding_span_top_k,
            args.binding_span_decay,
            args.binding_span_learned_gate,
            args.binding_span_gate_lr,
            args.binding_span_gate_neg_lr,
            args.binding_span_gate_strength,
            args.binding_span_gate_clip,
            args.latent_transition_branch,
            args.transition_window,
            args.transition_passes,
            args.transition_decay,
            args.transition_threshold,
            args.transition_strength,
            args.event_cell_branch,
            args.event_cell_count,
            args.event_cell_window,
            args.event_cell_top_k,
            args.event_cell_lr,
            args.event_cell_credit_lr,
            args.event_cell_neg_lr,
            args.event_cell_query_weight,
            args.event_cell_recency_decay,
            args.seed,
        )
    elif method == "state_event_cleanup_online":
        memory = OnlineQueryEventCleanupMemory(
            vocab_size,
            args.state_dim,
            args.state_order,
            args.state_decay,
            args.micro_slots,
            args.micro_lr,
            args.micro_wrong_lr,
            args.micro_score_scale,
            args.phase_bias_weight,
            args.micro_margin,
            args.binding_query_order,
            args.assembly_hops,
            args.assembly_event_window,
            args.assembly_seed_top_k,
            args.assembly_recency_decay,
            args.assembly_locality_decay,
            args.cleanup_slots,
            args.cleanup_lr,
            args.cleanup_wrong_lr,
            args.cleanup_score_scale,
            args.cleanup_top_k,
            args.cleanup_inhibit,
            args.seed,
        )
    elif method == "state_role_transition_online":
        memory = OnlineLocalRoleTransitionMemory(
            vocab_size,
            args.state_dim,
            args.state_order,
            args.state_decay,
            args.micro_slots,
            args.micro_lr,
            args.micro_wrong_lr,
            args.micro_score_scale,
            args.phase_bias_weight,
            args.micro_margin,
            args.role_query_order,
            args.role_hops,
            args.role_window,
            args.role_top_k,
            args.role_recency_decay,
            args.role_locality_decay,
            args.role_gate_lr,
            args.role_gate_wrong_lr,
            args.role_gate_strength,
            args.role_score_scale,
            args.role_downstream_bonus,
            args.role_channel_gates,
            args.role_final_score_only,
            args.role_score_top_k,
            args.role_score_inhibit,
            args.role_score_gate_mode,
            args.role_score_gate_base_margin,
            args.role_score_gate_role_margin,
            args.role_branch_readout,
            args.role_branch_base_score_scale,
            args.role_branch_role_score_scale,
            args.role_joint_rescue_readout,
            args.role_joint_rescue_score_scale,
            args.role_joint_rescue_top_k,
            args.role_joint_rescue_inhibit,
            args.role_joint_suppress_slots,
            args.role_joint_suppress_lr,
            args.role_joint_suppress_score_scale,
            args.role_joint_suppress_margin,
            args.role_joint_suppress_mode,
            args.role_joint_suppress_direct_threshold,
            args.role_joint_suppress_joint_threshold,
            args.role_branch_arbiter,
            args.role_branch_arbiter_default,
            args.role_branch_arbiter_slots,
            args.role_branch_arbiter_lr,
            args.role_branch_arbiter_wrong_lr,
            args.role_branch_arbiter_score_scale,
            args.role_branch_arbiter_margin,
            args.role_branch_arbiter_min_count,
            args.role_branch_arbiter_base_margin,
            args.role_branch_arbiter_threshold_lr,
            args.role_branch_arbiter_rescue_role_threshold,
            args.role_branch_arbiter_rescue_joint_threshold,
            args.role_branch_arbiter_joint_variants,
            args.role_branch_arbiter_rich_conflict_features,
            args.role_event_cache_size,
            args.edge_path_cleanup_answer_slots,
            args.edge_path_cleanup_slots,
            args.edge_path_cleanup_lr,
            args.edge_path_cleanup_wrong_lr,
            args.edge_path_cleanup_score_scale,
            args.edge_path_cleanup_top_k,
            args.edge_path_cleanup_inhibit,
            args.edge_path_cleanup_credit_mode,
            args.edge_path_margin_gate,
            args.edge_path_margin_min_scale,
            args.edge_path_margin_alt_scale,
            args.edge_path_margin_learned_dominance,
            args.edge_path_margin_escape_scale,
            args.edge_path_transient_inhibit_scale,
            args.edge_path_transient_inhibit_lr,
            args.edge_path_transient_inhibit_decay,
            args.edge_path_transient_inhibit_key,
            args.edge_path_transient_inhibit_hash_size,
            args.edge_path_transient_boost_scale,
            args.edge_path_transient_boost_lr,
            args.edge_path_transient_boost_support_margin,
            args.edge_path_transient_boost_consistency_margin,
            args.edge_path_transient_boost_runner_learned_max,
            args.edge_path_transient_boost_counterfactual_min_gain,
            args.edge_path_homeostasis_scale,
            args.edge_path_homeostasis_lr,
            args.edge_path_homeostasis_decay,
            args.edge_path_homeostasis_min_slot,
            args.edge_path_homeostasis_learned_dominance,
            args.edge_path_homeostasis_structure_margin,
            args.edge_path_homeostasis_soft_mod_scale,
            args.edge_path_homeostasis_soft_mod_floor,
            args.edge_path_homeostasis_trace_threshold,
            args.edge_path_homeostasis_trace_gain,
            args.edge_path_runner_arbiter_slots,
            args.edge_path_runner_arbiter_lr,
            args.edge_path_runner_arbiter_wrong_lr,
            args.edge_path_runner_arbiter_score_scale,
            args.edge_path_runner_arbiter_margin,
            args.edge_path_runner_arbiter_min_count,
            args.edge_path_runner_arbiter_negative_mode,
            args.edge_path_runner_arbiter_feature_mode,
            args.edge_path_runner_arbiter_gap_scale,
            args.edge_path_runner_arbiter_credit_mode,
            args.edge_path_soft_top_k,
            args.edge_path_soft_temperature,
            args.edge_path_soft_consistency_scale,
            args.edge_path_soft_learned_scale,
            args.edge_path_closure_score_scale,
            args.edge_path_closure_proto_slots,
            args.edge_path_closure_proto_lr,
            args.edge_path_closure_proto_wrong_lr,
            args.edge_path_closure_proto_score_scale,
            args.edge_path_closure_proto_min_count,
            args.edge_path_affinity_slots,
            args.edge_path_affinity_lr,
            args.edge_path_affinity_wrong_lr,
            args.edge_path_affinity_score_scale,
            args.edge_path_affinity_min_count,
            args.edge_path_affinity_margin_gate,
            args.edge_path_affinity_learned_dominance,
            args.edge_path_affinity_consistency_protect,
            args.edge_path_direct_answer_slots,
            args.edge_path_direct_slots,
            args.edge_path_direct_lr,
            args.edge_path_direct_wrong_lr,
            args.edge_path_direct_score_scale,
            args.edge_path_direct_mode,
            args.edge_path_structured_side_weight,
            args.edge_path_structured_path_weight,
            args.edge_path_structured_other_weight,
            args.seed,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    name = method
    if args.adaptive_inhibition:
        memory = stream.AdaptiveOutputInhibitionMemory(
            memory,
            strength=args.inhibit_strength,
            decay=args.inhibit_decay,
            lr=args.inhibit_lr,
            disinhibit_lr=args.inhibit_disinhibit_lr,
            top_k=args.inhibit_top_k,
            margin=args.inhibit_margin,
            max_weight=args.inhibit_max_weight,
        )
        name += "_inhib"
    if args.feature_calibration:
        memory = stream.FeatureConditionedCalibrationMemory(
            memory,
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
        name += "_feature_calib"
    if args.readout_gain != 1.0 or args.readout_gain_mode != "fixed":
        memory = stream.ReadoutGainMemory(
            memory,
            gain=args.readout_gain,
            mode=args.readout_gain_mode,
            margin_center=args.readout_gain_margin_center,
            margin_scale=args.readout_gain_margin_scale,
            min_gain=args.readout_gain_min,
            max_gain=args.readout_gain_max,
        )
        name += "_gain"
    if args.answer_slot_readout:
        memory = AnswerSlotReadoutMemory(
            memory,
            args.answer_slot_count,
            args.answer_slot_slots,
            args.answer_slot_lr,
            args.answer_slot_wrong_lr,
            args.answer_slot_score_scale,
            args.answer_slot_margin,
            args.answer_slot_update_base,
            args.answer_slot_feature_mode,
            args.answer_slot_direct_pre_margin_protect,
            args.answer_slot_direct_margin_min,
            args.answer_slot_coupling_slots,
            args.answer_slot_coupling_lr,
            args.answer_slot_coupling_wrong_lr,
            args.answer_slot_coupling_score_scale,
            args.answer_slot_wrong_cleanup_slots,
            args.answer_slot_wrong_cleanup_lr,
            args.answer_slot_wrong_cleanup_disinhibit_lr,
            args.answer_slot_wrong_cleanup_score_scale,
            args.answer_slot_wrong_cleanup_min_slot,
            args.answer_slot_wrong_cleanup_protect_mode,
            args.answer_slot_wrong_cleanup_protect_threshold,
            args.answer_slot_conflict_rescue_slots,
            args.answer_slot_conflict_rescue_lr,
            args.answer_slot_conflict_rescue_score_scale,
            args.answer_slot_conflict_rescue_top_k,
            args.answer_slot_conflict_rescue_min_slot,
            args.answer_slot_conflict_rescue_min_support,
            args.answer_slot_conflict_rescue_prefix_gate,
            args.answer_slot_conflict_rescue_prefix_margin,
            args.answer_slot_predicted_prefix_credit,
            args.answer_slot_predicted_prefix_skip_teacher_match,
            args.answer_slot_predicted_prefix_target_top_k,
            args.answer_slot_predicted_prefix_lr_scale,
            args.answer_slot_predicted_prefix_coupling_wrong_credit,
            args.answer_candidate_arbiter_rank,
            args.answer_candidate_arbiter_lr,
            args.answer_candidate_arbiter_score_scale,
            args.answer_candidate_arbiter_top_k,
            args.answer_candidate_arbiter_min_support,
            args.answer_candidate_arbiter_min_slot,
            args.answer_candidate_arbiter_projection_decay,
            args.answer_candidate_arbiter_clip,
            args.seed,
        )
        name += (
            f"_aslot{args.answer_slot_count}"
            f"_s{args.answer_slot_slots}"
            f"_x{args.answer_slot_score_scale:g}"
        )
        if args.answer_slot_feature_mode != "base":
            name += f"_{args.answer_slot_feature_mode}"
        if args.edge_path_soft_learned_scale != 0.0:
            name += f"_eplearn{args.edge_path_soft_learned_scale:g}"
        if args.edge_path_cleanup_credit_mode != "selected_target":
            name += f"_epcredit_{args.edge_path_cleanup_credit_mode}"
            if args.edge_path_cleanup_credit_mode == "margin_gated_soft_eligibility":
                name += (
                    f"_mg{args.edge_path_margin_gate:g}"
                    f"_ms{args.edge_path_margin_min_scale:g}"
                    f"_ma{args.edge_path_margin_alt_scale:g}"
                )
            if args.edge_path_cleanup_credit_mode == "learned_margin_escape":
                name += (
                    f"_mg{args.edge_path_margin_gate:g}"
                    f"_ms{args.edge_path_margin_min_scale:g}"
                    f"_dom{args.edge_path_margin_learned_dominance:g}"
                    f"_esc{args.edge_path_margin_escape_scale:g}"
                )
            if args.edge_path_cleanup_credit_mode == "transient_inhibit_escape":
                name += (
                    f"_mg{args.edge_path_margin_gate:g}"
                    f"_ms{args.edge_path_margin_min_scale:g}"
                    f"_dom{args.edge_path_margin_learned_dominance:g}"
                    f"_tis{args.edge_path_transient_inhibit_scale:g}"
                    f"_tilr{args.edge_path_transient_inhibit_lr:g}"
                    f"_tid{args.edge_path_transient_inhibit_decay:g}"
                )
                if args.edge_path_transient_inhibit_key != "mid":
                    name += (
                        f"_tik{args.edge_path_transient_inhibit_key}"
                        f"{args.edge_path_transient_inhibit_hash_size}"
                    )
                if args.edge_path_transient_boost_scale > 0.0 or args.edge_path_transient_boost_lr > 0.0:
                    name += (
                        f"_tbs{args.edge_path_transient_boost_scale:g}"
                        f"_tblr{args.edge_path_transient_boost_lr:g}"
                        f"_tbm{args.edge_path_transient_boost_support_margin:g}"
                    )
                    if args.edge_path_transient_boost_consistency_margin >= 0.0:
                        name += f"_tbcm{args.edge_path_transient_boost_consistency_margin:g}"
                    if args.edge_path_transient_boost_runner_learned_max >= 0.0:
                        name += f"_tblmax{args.edge_path_transient_boost_runner_learned_max:g}"
                    if args.edge_path_transient_boost_counterfactual_min_gain > -1.0:
                        name += f"_tbcfg{args.edge_path_transient_boost_counterfactual_min_gain:g}"
                if args.edge_path_homeostasis_scale > 0.0 or args.edge_path_homeostasis_lr > 0.0:
                    name += (
                        f"_ephomeo{args.edge_path_homeostasis_scale:g}"
                        f"_ephlr{args.edge_path_homeostasis_lr:g}"
                        f"_ephd{args.edge_path_homeostasis_decay:g}"
                        f"_ephmin{args.edge_path_homeostasis_min_slot}"
                    )
                    if args.edge_path_homeostasis_learned_dominance > 0.0:
                        name += (
                            f"_ephdom{args.edge_path_homeostasis_learned_dominance:g}"
                            f"_ephsm{args.edge_path_homeostasis_structure_margin:g}"
                        )
                    if args.edge_path_homeostasis_soft_mod_scale > 0.0:
                        name += (
                            f"_ephsoft{args.edge_path_homeostasis_soft_mod_scale:g}"
                            f"_ephfloor{args.edge_path_homeostasis_soft_mod_floor:g}"
                        )
                    if (
                        args.edge_path_homeostasis_trace_threshold > 0.0
                        or abs(args.edge_path_homeostasis_trace_gain - 1.0) > 1e-12
                    ):
                        name += (
                            f"_ephthr{args.edge_path_homeostasis_trace_threshold:g}"
                            f"_ephgain{args.edge_path_homeostasis_trace_gain:g}"
                        )
                if args.edge_path_runner_arbiter_score_scale > 0.0:
                    name += (
                        f"_rab{args.edge_path_runner_arbiter_score_scale:g}"
                        f"_rablr{args.edge_path_runner_arbiter_lr:g}"
                        f"_rabw{args.edge_path_runner_arbiter_wrong_lr:g}"
                        f"_rabm{args.edge_path_runner_arbiter_margin:g}"
                    )
                    if args.edge_path_runner_arbiter_min_count > 0.0:
                        name += f"_rabmc{args.edge_path_runner_arbiter_min_count:g}"
                    if args.edge_path_runner_arbiter_negative_mode != "subtract":
                        name += f"_rabneg{args.edge_path_runner_arbiter_negative_mode}"
                    if args.edge_path_runner_arbiter_feature_mode != "pair":
                        name += (
                            f"_rabf{args.edge_path_runner_arbiter_feature_mode}"
                            f"_rabg{args.edge_path_runner_arbiter_gap_scale:g}"
                        )
                    if args.edge_path_runner_arbiter_credit_mode != "answer_error":
                        name += f"_rabc{args.edge_path_runner_arbiter_credit_mode}"
                if args.edge_path_closure_score_scale > 0.0:
                    name += f"_epclose{args.edge_path_closure_score_scale:g}"
                if args.edge_path_closure_proto_score_scale > 0.0:
                    name += (
                        f"_epcproto{args.edge_path_closure_proto_score_scale:g}"
                        f"_epcplr{args.edge_path_closure_proto_lr:g}"
                        f"_epcpw{args.edge_path_closure_proto_wrong_lr:g}"
                        f"_epcpmc{args.edge_path_closure_proto_min_count:g}"
                    )
                if args.edge_path_affinity_score_scale > 0.0:
                    name += (
                        f"_epaff{args.edge_path_affinity_score_scale:g}"
                        f"_epaffs{args.edge_path_affinity_slots}"
                        f"_epafflr{args.edge_path_affinity_lr:g}"
                        f"_epaffw{args.edge_path_affinity_wrong_lr:g}"
                        f"_epaffmc{args.edge_path_affinity_min_count:g}"
                    )
                    if (
                        args.edge_path_affinity_margin_gate > 0.0
                        or args.edge_path_affinity_learned_dominance > 0.0
                        or args.edge_path_affinity_consistency_protect > 0.0
                    ):
                        name += (
                            f"_epaffmg{args.edge_path_affinity_margin_gate:g}"
                            f"_epaffdom{args.edge_path_affinity_learned_dominance:g}"
                            f"_epaffcp{args.edge_path_affinity_consistency_protect:g}"
                        )
        if args.edge_path_direct_mode != "soft_feature":
            name += f"_edirect_{args.edge_path_direct_mode}"
        if args.answer_slot_direct_pre_margin_protect > 0.0:
            name += f"_dprot{args.answer_slot_direct_pre_margin_protect:g}"
        if args.answer_slot_direct_margin_min > 0.0:
            name += f"_dmin{args.answer_slot_direct_margin_min:g}"
        if args.answer_slot_coupling_score_scale > 0.0:
            name += (
                f"_cslot{args.answer_slot_coupling_slots}"
                f"_cx{args.answer_slot_coupling_score_scale:g}"
            )
        if args.answer_slot_wrong_cleanup_score_scale > 0.0:
            name += (
                f"_wclean{args.answer_slot_wrong_cleanup_slots}"
                f"_wx{args.answer_slot_wrong_cleanup_score_scale:g}"
            )
            if args.answer_slot_wrong_cleanup_protect_mode != "none":
                name += (
                    f"_wprot{args.answer_slot_wrong_cleanup_protect_mode}"
                    f"{args.answer_slot_wrong_cleanup_protect_threshold:g}"
                )
        if args.answer_slot_conflict_rescue_score_scale > 0.0:
            name += (
                f"_conf{args.answer_slot_conflict_rescue_slots}"
                f"_cx{args.answer_slot_conflict_rescue_score_scale:g}"
                f"_k{args.answer_slot_conflict_rescue_top_k}"
            )
            if args.answer_slot_conflict_rescue_prefix_gate != "none":
                name += (
                    f"_pg{args.answer_slot_conflict_rescue_prefix_gate}"
                    f"{args.answer_slot_conflict_rescue_prefix_margin:g}"
                )
            if args.answer_slot_conflict_rescue_min_support > 0.0:
                name += f"_csup{args.answer_slot_conflict_rescue_min_support:g}"
        if args.answer_slot_predicted_prefix_credit != "none":
            name += f"_pp{args.answer_slot_predicted_prefix_credit}"
            if not args.answer_slot_predicted_prefix_skip_teacher_match:
                name += "_all"
            if args.answer_slot_predicted_prefix_target_top_k > 0:
                name += f"_ptop{args.answer_slot_predicted_prefix_target_top_k}"
            if args.answer_slot_predicted_prefix_lr_scale != 1.0:
                name += f"_plr{args.answer_slot_predicted_prefix_lr_scale:g}"
            if not args.answer_slot_predicted_prefix_coupling_wrong_credit:
                name += "_ptargetonly"
        if args.answer_candidate_arbiter_score_scale > 0.0:
            name += (
                f"_carb{args.answer_candidate_arbiter_rank}"
                f"_k{args.answer_candidate_arbiter_top_k}"
                f"_x{args.answer_candidate_arbiter_score_scale:g}"
                f"_lr{args.answer_candidate_arbiter_lr:g}"
            )
            if args.answer_candidate_arbiter_min_support > 0.0:
                name += f"_sup{args.answer_candidate_arbiter_min_support:g}"
            if args.answer_candidate_arbiter_min_slot > 0:
                name += f"_ms{args.answer_candidate_arbiter_min_slot}"
            if args.answer_candidate_arbiter_projection_decay < 1.0:
                name += f"_decay{args.answer_candidate_arbiter_projection_decay:g}"
        if not args.answer_slot_update_base:
            name += "_slotonly"
    if method == "state_microproto_online" and args.binding_hops > 0:
        suffix = f"_bind{args.binding_hops}"
        if args.binding_mode != "pair_apply":
            suffix += f"_{args.binding_mode}"
        if args.binding_bidirectional:
            suffix += "_bidir"
        if args.binding_query_mode != "recent_trace":
            suffix += f"_{args.binding_query_mode}"
        if args.binding_span_learned_gate:
            suffix += "_span_gate"
        name += suffix
    if method == "state_microproto_online" and args.latent_transition_branch:
        name += "_latent_transition"
    if method == "state_microproto_online" and args.event_cell_branch:
        name += "_event_cell"
    if method == "state_event_cleanup_online":
        name += f"_event_cleanup_h{args.assembly_hops}_w{args.assembly_event_window}_k{args.assembly_seed_top_k}"
    if method == "state_role_transition_online":
        name += f"_role_transition_h{args.role_hops}_w{args.role_window}_k{args.role_top_k}"
        if args.role_channel_gates:
            name += "_chan"
        if args.role_final_score_only:
            name += "_final"
        if args.role_score_top_k > 0:
            name += f"_rtop{args.role_score_top_k}"
            if args.role_score_inhibit > 0.0:
                name += f"_i{args.role_score_inhibit:g}"
        if args.role_score_gate_mode != "none":
            name += f"_rsgate_{args.role_score_gate_mode}"
            if args.role_score_gate_base_margin > 0.0:
                name += f"_bm{args.role_score_gate_base_margin:g}"
            if args.role_score_gate_role_margin > 0.0:
                name += f"_rm{args.role_score_gate_role_margin:g}"
        if args.role_branch_readout:
            name += f"_branch_b{args.role_branch_base_score_scale:g}_r{args.role_branch_role_score_scale:g}"
        if args.role_joint_rescue_readout:
            name += f"_joint{args.role_joint_rescue_score_scale:g}"
            if args.role_joint_rescue_top_k > 0:
                name += f"_jtop{args.role_joint_rescue_top_k}"
                if args.role_joint_rescue_inhibit > 0.0:
                    name += f"_ji{args.role_joint_rescue_inhibit:g}"
            if args.role_joint_suppress_score_scale > 0.0:
                name += f"_jsup{args.role_joint_suppress_score_scale:g}_s{args.role_joint_suppress_slots}"
                if args.role_joint_suppress_mode != "all_wrong":
                    name += (
                        f"_{args.role_joint_suppress_mode}"
                        f"_dt{args.role_joint_suppress_direct_threshold:g}"
                        f"_jt{args.role_joint_suppress_joint_threshold:g}"
                    )
        if args.role_branch_arbiter != "none":
            name += (
                f"_arb_{args.role_branch_arbiter}"
                f"_{args.role_branch_arbiter_default}"
                f"_s{args.role_branch_arbiter_slots}"
            )
            if args.role_branch_arbiter_joint_variants:
                name += "_jpaths"
            if args.role_branch_arbiter_rich_conflict_features:
                name += "_rich"
            if args.role_branch_arbiter_min_count > 0.0:
                name += f"_mc{args.role_branch_arbiter_min_count:g}"
            if args.role_branch_arbiter == "base_margin_rescue":
                name += (
                    f"_rr{args.role_branch_arbiter_rescue_role_threshold:g}"
                    f"_jr{args.role_branch_arbiter_rescue_joint_threshold:g}"
                )
        if args.role_event_cache_size > 0:
            name += f"_ecache{args.role_event_cache_size}"

    memory_cfg: dict[str, Any] = {"phase_cfg": asdict(phase_cfg)}
    if method == "state_microproto_online":
        memory_cfg["state_microproto_cfg"] = {
            "state_dim": args.state_dim,
            "state_order": args.state_order,
            "state_decay": args.state_decay,
            "micro_slots": args.micro_slots,
            "micro_lr": args.micro_lr,
            "micro_wrong_lr": args.micro_wrong_lr,
            "micro_score_scale": args.micro_score_scale,
            "micro_margin": args.micro_margin,
            "binding_hops": args.binding_hops,
            "binding_window": args.binding_window,
            "binding_query_order": args.binding_query_order,
            "binding_query_mode": args.binding_query_mode,
            "binding_focus_k": args.binding_focus_k,
            "binding_decay": args.binding_decay,
            "binding_bidirectional": args.binding_bidirectional,
            "binding_mode": args.binding_mode,
            "binding_span_window": args.binding_span_window,
            "binding_span_top_k": args.binding_span_top_k,
            "binding_span_decay": args.binding_span_decay,
            "binding_span_learned_gate": args.binding_span_learned_gate,
            "binding_span_gate_lr": args.binding_span_gate_lr,
            "binding_span_gate_neg_lr": args.binding_span_gate_neg_lr,
            "binding_span_gate_strength": args.binding_span_gate_strength,
            "binding_span_gate_clip": args.binding_span_gate_clip,
            "latent_transition_branch": args.latent_transition_branch,
            "transition_window": args.transition_window,
            "transition_passes": args.transition_passes,
            "transition_decay": args.transition_decay,
            "transition_threshold": args.transition_threshold,
            "transition_strength": args.transition_strength,
            "event_cell_branch": args.event_cell_branch,
            "event_cell_count": args.event_cell_count,
            "event_cell_window": args.event_cell_window,
            "event_cell_top_k": args.event_cell_top_k,
            "event_cell_lr": args.event_cell_lr,
            "event_cell_credit_lr": args.event_cell_credit_lr,
            "event_cell_neg_lr": args.event_cell_neg_lr,
            "event_cell_query_weight": args.event_cell_query_weight,
            "event_cell_recency_decay": args.event_cell_recency_decay,
            "seed": args.seed,
        }
    if method == "state_event_cleanup_online":
        memory_cfg["state_event_cleanup_cfg"] = {
            "state_dim": args.state_dim,
            "state_order": args.state_order,
            "state_decay": args.state_decay,
            "micro_slots": args.micro_slots,
            "micro_lr": args.micro_lr,
            "micro_wrong_lr": args.micro_wrong_lr,
            "micro_score_scale": args.micro_score_scale,
            "micro_margin": args.micro_margin,
            "binding_query_order": args.binding_query_order,
            "assembly_hops": args.assembly_hops,
            "assembly_event_window": args.assembly_event_window,
            "assembly_seed_top_k": args.assembly_seed_top_k,
            "assembly_recency_decay": args.assembly_recency_decay,
            "assembly_locality_decay": args.assembly_locality_decay,
            "cleanup_slots": args.cleanup_slots,
            "cleanup_lr": args.cleanup_lr,
            "cleanup_wrong_lr": args.cleanup_wrong_lr,
            "cleanup_score_scale": args.cleanup_score_scale,
            "cleanup_top_k": args.cleanup_top_k,
            "cleanup_inhibit": args.cleanup_inhibit,
            "seed": args.seed,
        }
    if method == "state_role_transition_online":
        memory_cfg["state_role_transition_cfg"] = {
            "state_dim": args.state_dim,
            "state_order": args.state_order,
            "state_decay": args.state_decay,
            "micro_slots": args.micro_slots,
            "micro_lr": args.micro_lr,
            "micro_wrong_lr": args.micro_wrong_lr,
            "micro_score_scale": args.micro_score_scale,
            "micro_margin": args.micro_margin,
            "role_query_order": args.role_query_order,
            "role_hops": args.role_hops,
            "role_window": args.role_window,
            "role_top_k": args.role_top_k,
            "role_recency_decay": args.role_recency_decay,
            "role_locality_decay": args.role_locality_decay,
            "role_gate_lr": args.role_gate_lr,
            "role_gate_wrong_lr": args.role_gate_wrong_lr,
            "role_gate_strength": args.role_gate_strength,
            "role_score_scale": args.role_score_scale,
            "role_downstream_bonus": args.role_downstream_bonus,
            "role_channel_gates": args.role_channel_gates,
            "role_final_score_only": args.role_final_score_only,
            "role_score_top_k": args.role_score_top_k,
            "role_score_inhibit": args.role_score_inhibit,
            "role_score_gate_mode": args.role_score_gate_mode,
            "role_score_gate_base_margin": args.role_score_gate_base_margin,
            "role_score_gate_role_margin": args.role_score_gate_role_margin,
            "role_branch_readout": args.role_branch_readout,
            "role_branch_base_score_scale": args.role_branch_base_score_scale,
            "role_branch_role_score_scale": args.role_branch_role_score_scale,
            "role_joint_rescue_readout": args.role_joint_rescue_readout,
            "role_joint_rescue_score_scale": args.role_joint_rescue_score_scale,
            "role_joint_rescue_top_k": args.role_joint_rescue_top_k,
            "role_joint_rescue_inhibit": args.role_joint_rescue_inhibit,
            "role_joint_suppress_slots": args.role_joint_suppress_slots,
            "role_joint_suppress_lr": args.role_joint_suppress_lr,
            "role_joint_suppress_score_scale": args.role_joint_suppress_score_scale,
            "role_joint_suppress_margin": args.role_joint_suppress_margin,
            "role_joint_suppress_mode": args.role_joint_suppress_mode,
            "role_joint_suppress_direct_threshold": args.role_joint_suppress_direct_threshold,
            "role_joint_suppress_joint_threshold": args.role_joint_suppress_joint_threshold,
            "role_branch_arbiter": args.role_branch_arbiter,
            "role_branch_arbiter_default": args.role_branch_arbiter_default,
            "role_branch_arbiter_slots": args.role_branch_arbiter_slots,
            "role_branch_arbiter_lr": args.role_branch_arbiter_lr,
            "role_branch_arbiter_wrong_lr": args.role_branch_arbiter_wrong_lr,
            "role_branch_arbiter_score_scale": args.role_branch_arbiter_score_scale,
            "role_branch_arbiter_margin": args.role_branch_arbiter_margin,
            "role_branch_arbiter_min_count": args.role_branch_arbiter_min_count,
            "role_branch_arbiter_base_margin": args.role_branch_arbiter_base_margin,
            "role_branch_arbiter_threshold_lr": args.role_branch_arbiter_threshold_lr,
            "role_branch_arbiter_rescue_role_threshold": args.role_branch_arbiter_rescue_role_threshold,
            "role_branch_arbiter_rescue_joint_threshold": args.role_branch_arbiter_rescue_joint_threshold,
            "role_branch_arbiter_joint_variants": args.role_branch_arbiter_joint_variants,
            "role_branch_arbiter_rich_conflict_features": (
                args.role_branch_arbiter_rich_conflict_features
            ),
            "role_event_cache_size": args.role_event_cache_size,
            "edge_path_cleanup_answer_slots": args.edge_path_cleanup_answer_slots,
            "edge_path_cleanup_slots": args.edge_path_cleanup_slots,
            "edge_path_cleanup_lr": args.edge_path_cleanup_lr,
            "edge_path_cleanup_wrong_lr": args.edge_path_cleanup_wrong_lr,
            "edge_path_cleanup_score_scale": args.edge_path_cleanup_score_scale,
            "edge_path_cleanup_top_k": args.edge_path_cleanup_top_k,
            "edge_path_cleanup_inhibit": args.edge_path_cleanup_inhibit,
            "edge_path_cleanup_credit_mode": args.edge_path_cleanup_credit_mode,
            "edge_path_homeostasis_scale": args.edge_path_homeostasis_scale,
            "edge_path_homeostasis_lr": args.edge_path_homeostasis_lr,
            "edge_path_homeostasis_decay": args.edge_path_homeostasis_decay,
            "edge_path_homeostasis_min_slot": args.edge_path_homeostasis_min_slot,
            "edge_path_homeostasis_learned_dominance": (
                args.edge_path_homeostasis_learned_dominance
            ),
            "edge_path_homeostasis_structure_margin": args.edge_path_homeostasis_structure_margin,
            "edge_path_homeostasis_soft_mod_scale": args.edge_path_homeostasis_soft_mod_scale,
            "edge_path_homeostasis_soft_mod_floor": args.edge_path_homeostasis_soft_mod_floor,
            "edge_path_homeostasis_trace_threshold": args.edge_path_homeostasis_trace_threshold,
            "edge_path_homeostasis_trace_gain": args.edge_path_homeostasis_trace_gain,
            "edge_path_soft_top_k": args.edge_path_soft_top_k,
            "edge_path_soft_temperature": args.edge_path_soft_temperature,
            "edge_path_soft_consistency_scale": args.edge_path_soft_consistency_scale,
            "edge_path_soft_learned_scale": args.edge_path_soft_learned_scale,
            "edge_path_closure_score_scale": args.edge_path_closure_score_scale,
            "edge_path_closure_proto_slots": args.edge_path_closure_proto_slots,
            "edge_path_closure_proto_lr": args.edge_path_closure_proto_lr,
            "edge_path_closure_proto_wrong_lr": args.edge_path_closure_proto_wrong_lr,
            "edge_path_closure_proto_score_scale": args.edge_path_closure_proto_score_scale,
            "edge_path_closure_proto_min_count": args.edge_path_closure_proto_min_count,
            "edge_path_affinity_slots": args.edge_path_affinity_slots,
            "edge_path_affinity_lr": args.edge_path_affinity_lr,
            "edge_path_affinity_wrong_lr": args.edge_path_affinity_wrong_lr,
            "edge_path_affinity_score_scale": args.edge_path_affinity_score_scale,
            "edge_path_affinity_min_count": args.edge_path_affinity_min_count,
            "edge_path_affinity_margin_gate": args.edge_path_affinity_margin_gate,
            "edge_path_affinity_learned_dominance": args.edge_path_affinity_learned_dominance,
            "edge_path_affinity_consistency_protect": args.edge_path_affinity_consistency_protect,
            "edge_path_direct_answer_slots": args.edge_path_direct_answer_slots,
            "edge_path_direct_slots": args.edge_path_direct_slots,
            "edge_path_direct_lr": args.edge_path_direct_lr,
            "edge_path_direct_wrong_lr": args.edge_path_direct_wrong_lr,
            "edge_path_direct_score_scale": args.edge_path_direct_score_scale,
            "edge_path_direct_mode": args.edge_path_direct_mode,
            "edge_path_structured_side_weight": args.edge_path_structured_side_weight,
            "edge_path_structured_path_weight": args.edge_path_structured_path_weight,
            "edge_path_structured_other_weight": args.edge_path_structured_other_weight,
            "seed": args.seed,
        }
    if args.answer_slot_readout:
        memory_cfg["answer_slot_cfg"] = {
            "answer_slot_count": args.answer_slot_count,
            "answer_slot_slots": args.answer_slot_slots,
            "answer_slot_lr": args.answer_slot_lr,
            "answer_slot_wrong_lr": args.answer_slot_wrong_lr,
            "answer_slot_score_scale": args.answer_slot_score_scale,
            "answer_slot_margin": args.answer_slot_margin,
            "answer_slot_update_base": args.answer_slot_update_base,
            "answer_slot_feature_mode": args.answer_slot_feature_mode,
            "answer_slot_direct_pre_margin_protect": args.answer_slot_direct_pre_margin_protect,
            "answer_slot_direct_margin_min": args.answer_slot_direct_margin_min,
            "answer_slot_coupling_slots": args.answer_slot_coupling_slots,
            "answer_slot_coupling_lr": args.answer_slot_coupling_lr,
            "answer_slot_coupling_wrong_lr": args.answer_slot_coupling_wrong_lr,
            "answer_slot_coupling_score_scale": args.answer_slot_coupling_score_scale,
            "answer_slot_wrong_cleanup_slots": args.answer_slot_wrong_cleanup_slots,
            "answer_slot_wrong_cleanup_lr": args.answer_slot_wrong_cleanup_lr,
            "answer_slot_wrong_cleanup_disinhibit_lr": args.answer_slot_wrong_cleanup_disinhibit_lr,
            "answer_slot_wrong_cleanup_score_scale": args.answer_slot_wrong_cleanup_score_scale,
            "answer_slot_wrong_cleanup_min_slot": args.answer_slot_wrong_cleanup_min_slot,
            "answer_slot_wrong_cleanup_protect_mode": args.answer_slot_wrong_cleanup_protect_mode,
            "answer_slot_wrong_cleanup_protect_threshold": args.answer_slot_wrong_cleanup_protect_threshold,
            "answer_slot_conflict_rescue_slots": args.answer_slot_conflict_rescue_slots,
            "answer_slot_conflict_rescue_lr": args.answer_slot_conflict_rescue_lr,
            "answer_slot_conflict_rescue_score_scale": args.answer_slot_conflict_rescue_score_scale,
            "answer_slot_conflict_rescue_top_k": args.answer_slot_conflict_rescue_top_k,
            "answer_slot_conflict_rescue_min_slot": args.answer_slot_conflict_rescue_min_slot,
            "answer_slot_conflict_rescue_min_support": (
                args.answer_slot_conflict_rescue_min_support
            ),
            "answer_slot_conflict_rescue_prefix_gate": args.answer_slot_conflict_rescue_prefix_gate,
            "answer_slot_conflict_rescue_prefix_margin": args.answer_slot_conflict_rescue_prefix_margin,
            "answer_slot_predicted_prefix_credit": args.answer_slot_predicted_prefix_credit,
            "answer_slot_predicted_prefix_skip_teacher_match": (
                args.answer_slot_predicted_prefix_skip_teacher_match
            ),
            "answer_slot_predicted_prefix_target_top_k": (
                args.answer_slot_predicted_prefix_target_top_k
            ),
            "answer_slot_predicted_prefix_lr_scale": args.answer_slot_predicted_prefix_lr_scale,
            "answer_slot_predicted_prefix_coupling_wrong_credit": (
                args.answer_slot_predicted_prefix_coupling_wrong_credit
            ),
            "answer_candidate_arbiter_rank": args.answer_candidate_arbiter_rank,
            "answer_candidate_arbiter_lr": args.answer_candidate_arbiter_lr,
            "answer_candidate_arbiter_score_scale": args.answer_candidate_arbiter_score_scale,
            "answer_candidate_arbiter_top_k": args.answer_candidate_arbiter_top_k,
            "answer_candidate_arbiter_min_support": args.answer_candidate_arbiter_min_support,
            "answer_candidate_arbiter_min_slot": args.answer_candidate_arbiter_min_slot,
            "answer_candidate_arbiter_projection_decay": (
                args.answer_candidate_arbiter_projection_decay
            ),
            "answer_candidate_arbiter_clip": args.answer_candidate_arbiter_clip,
        }
    return name, memory, memory_cfg


def train_sequence(
    memory: Any,
    seq: Sequence[int],
    answer_pos: int | None,
    temperature: float,
    update: bool,
) -> dict[str, Any]:
    reset_dynamic(memory)
    order = int(memory.max_order)
    history: list[int] = []
    loss_sum = 0.0
    correct = 0
    total = 0
    answer_loss = 0.0
    answer_correct = 0
    answer_total = 0
    full_answer_loss = 0.0
    full_answer_correct = 0
    full_answer_total = 0
    full_answer_sequence_correct = 1 if answer_pos is not None else 0
    full_answer_sequence_total = 1 if answer_pos is not None else 0
    for idx in range(len(seq) - 1):
        current = int(seq[idx])
        target = int(seq[idx + 1])
        context = np.array((history + [current])[-order:], dtype=np.int64)
        loss, pred, _ = softmax_loss_and_pred(memory.scores(context), target, temperature)
        loss_sum += loss
        correct += int(pred == target)
        total += 1
        if answer_pos is not None and idx + 1 == answer_pos:
            answer_loss += loss
            answer_correct += int(pred == target)
            answer_total += 1
        if answer_pos is not None and idx + 1 >= answer_pos:
            is_correct = int(pred == target)
            full_answer_loss += loss
            full_answer_correct += is_correct
            full_answer_total += 1
            full_answer_sequence_correct &= is_correct
        if update:
            memory.update(context, target)
        elif hasattr(memory, "observe"):
            memory.observe(context, target)
        history = (history + [current])[-max(order - 1, 0) :]
    return {
        "token_loss_sum": loss_sum,
        "token_correct": correct,
        "token_total": total,
        "answer_loss_sum": answer_loss,
        "answer_correct": answer_correct,
        "answer_total": answer_total,
        "full_answer_loss_sum": full_answer_loss,
        "full_answer_token_correct": full_answer_correct,
        "full_answer_token_total": full_answer_total,
        "full_answer_sequence_correct": full_answer_sequence_correct,
        "full_answer_sequence_total": full_answer_sequence_total,
    }


def score_prompt_answer(
    memory: Any,
    prompt_ids: Sequence[int],
    target_id: int,
    temperature: float,
) -> tuple[float, int, float]:
    reset_dynamic(memory)
    observe_prompt(memory, prompt_ids)
    order = int(memory.max_order)
    context = np.array(list(prompt_ids)[-order:], dtype=np.int64)
    return softmax_loss_and_pred(memory.scores(context), int(target_id), temperature)


def scores_for_answer_slot(memory: Any, context: Sequence[int] | np.ndarray, slot: int) -> np.ndarray:
    if hasattr(memory, "scores_with_answer_slot"):
        return memory.scores_with_answer_slot(context, int(slot))
    return memory.scores(context)


def component_metrics_for_last_score(
    memory: Any,
    phase_name: str,
    slot: int,
    target: int,
    pred: int,
    target_prob: float,
    temperature: float,
) -> dict[str, Any]:
    if hasattr(memory, "last_answer_slot_component_metrics"):
        row = memory.last_answer_slot_component_metrics(
            int(target),
            int(pred),
            float(target_prob),
            float(temperature),
        )
    else:
        row = {
            "slot": int(slot),
            "feature_mode": "base_memory",
            "target_id": int(target),
            "prediction_id": int(pred),
            "target_prob": float(target_prob),
            "correct": int(pred == int(target)),
        }
    row["decode_phase"] = phase_name
    row["slot"] = int(slot)
    return row


def score_top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    if k <= 0 or scores.size == 0:
        return []
    k = min(int(k), int(scores.size))
    if k >= scores.size:
        return [int(x) for x in np.argsort(scores)[::-1]]
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [int(x) for x in idx]


def restore_answer_prefix_state(
    memory: Any,
    prompt_ids: Sequence[int],
    prefix_ids: Sequence[int],
) -> list[int]:
    reset_dynamic(memory)
    observe_prompt(memory, prompt_ids)
    order = int(memory.max_order)
    history = list(int(x) for x in prompt_ids)
    for token in prefix_ids:
        context = np.array(history[-order:], dtype=np.int64)
        if hasattr(memory, "observe"):
            memory.observe(context, int(token))
        history.append(int(token))
    return history


def greedy_lookahead_answer_prediction(
    memory: Any,
    prompt_ids: Sequence[int],
    prefix_ids: Sequence[int],
    slot: int,
    answer_count: int,
    temperature: float,
    lookahead_candidates: int,
    lookahead_weight: float,
) -> tuple[np.ndarray, int, int]:
    history = restore_answer_prefix_state(memory, prompt_ids, prefix_ids)
    order = int(memory.max_order)
    context = np.array(history[-order:], dtype=np.int64)
    scores = scores_for_answer_slot(memory, context, slot)
    probs = phase.softmax(scores, temperature)
    pred = int(np.argmax(probs))
    applied = 0
    if lookahead_candidates <= 0 or slot + 1 >= answer_count:
        return scores, pred, applied

    candidates = score_top_k_indices(scores.astype(np.float32, copy=False), lookahead_candidates)
    if not candidates:
        return scores, pred, applied

    best_pred = pred
    best_score = -float("inf")
    for candidate in candidates:
        candidate = int(candidate)
        candidate_history = restore_answer_prefix_state(
            memory,
            prompt_ids,
            list(prefix_ids) + [candidate],
        )
        next_context = np.array(candidate_history[-order:], dtype=np.int64)
        next_scores = scores_for_answer_slot(memory, next_context, slot + 1)
        next_value = float(np.max(next_scores)) if next_scores.size else 0.0
        joint_score = float(scores[candidate]) + float(lookahead_weight) * next_value
        if joint_score > best_score:
            best_score = joint_score
            best_pred = candidate
    # Restore current-slot state and components after candidate simulations.
    history = restore_answer_prefix_state(memory, prompt_ids, prefix_ids)
    context = np.array(history[-order:], dtype=np.int64)
    scores = scores_for_answer_slot(memory, context, slot)
    return scores, int(best_pred), 1


def score_prompt_answer_sequence(
    memory: Any,
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    temperature: float,
    collect_components: bool = False,
    lookahead_candidates: int = 0,
    lookahead_weight: float = 1.0,
) -> dict[str, Any]:
    reset_dynamic(memory)
    observe_prompt(memory, prompt_ids)
    order = int(memory.max_order)
    history = list(int(x) for x in prompt_ids)
    loss_sum = 0.0
    token_correct = 0
    teacher_forced_preds: list[int] = []
    target_probs: list[float] = []
    component_rows: list[dict[str, Any]] = []
    first_loss = 0.0
    first_rank = 0
    first_margin = 0.0
    answer_top_hits = {k: 0 for k in RANK_DIAGNOSTIC_KS}
    answer_error_top_hits = {k: 0 for k in RANK_DIAGNOSTIC_KS}
    full_rank_sum = 0.0
    full_margin_sum = 0.0
    full_top_hits = {k: 0 for k in RANK_DIAGNOSTIC_KS}
    full_error_top_hits = {k: 0 for k in RANK_DIAGNOSTIC_KS}
    full_error_count = 0
    for slot, target in enumerate(answer_ids):
        context = np.array(history[-order:], dtype=np.int64)
        scores = scores_for_answer_slot(memory, context, slot)
        loss, pred, target_prob = softmax_loss_and_pred(
            scores,
            int(target),
            temperature,
        )
        rank_metrics = candidate_rank_metrics(scores, int(target), int(pred))
        if collect_components:
            component_rows.append(
                component_metrics_for_last_score(
                    memory,
                    "teacher_forced",
                    slot,
                    int(target),
                    int(pred),
                    float(target_prob),
                    temperature,
                )
            )
        loss_sum += loss
        token_correct += int(pred == int(target))
        if not teacher_forced_preds:
            first_loss = loss
            first_rank = int(rank_metrics["target_rank"])
            first_margin = float(rank_metrics["target_margin"])
            for k in RANK_DIAGNOSTIC_KS:
                answer_top_hits[k] += int(rank_metrics[f"top{k}"])
                answer_error_top_hits[k] += int(rank_metrics[f"error_top{k}"])
        teacher_forced_preds.append(pred)
        target_probs.append(target_prob)
        full_rank_sum += float(rank_metrics["target_rank"])
        full_margin_sum += float(rank_metrics["target_margin"])
        full_error_count += int(rank_metrics["error"])
        for k in RANK_DIAGNOSTIC_KS:
            full_top_hits[k] += int(rank_metrics[f"top{k}"])
            full_error_top_hits[k] += int(rank_metrics[f"error_top{k}"])
        if hasattr(memory, "observe"):
            memory.observe(context, int(target))
        history.append(int(target))

    history = list(int(x) for x in prompt_ids)
    greedy_preds: list[int] = []
    greedy_lookahead_applied = 0
    for slot, target in enumerate(answer_ids):
        if lookahead_candidates > 0:
            scores, pred, applied = greedy_lookahead_answer_prediction(
                memory,
                prompt_ids,
                greedy_preds,
                slot,
                len(answer_ids),
                temperature,
                lookahead_candidates,
                lookahead_weight,
            )
            greedy_lookahead_applied += int(applied)
            context = np.array(history[-order:], dtype=np.int64)
        else:
            context = np.array(history[-order:], dtype=np.int64)
            scores = scores_for_answer_slot(memory, context, slot)
            probs = phase.softmax(scores, temperature)
            pred = int(np.argmax(probs))
        probs = phase.softmax(scores, temperature)
        greedy_preds.append(pred)
        if collect_components:
            target = int(target)
            target_prob = float(probs[target]) if 0 <= target < len(probs) else float("nan")
            component_rows.append(
                component_metrics_for_last_score(
                    memory,
                    "greedy",
                    slot,
                    target,
                    pred,
                    target_prob,
                    temperature,
                )
            )
        if hasattr(memory, "observe"):
            memory.observe(context, pred)
        history.append(pred)

    first_correct = int(teacher_forced_preds[0] == int(answer_ids[0])) if answer_ids else 0
    first_prob = float(target_probs[0]) if target_probs else 0.0
    first_pred = int(teacher_forced_preds[0]) if teacher_forced_preds else -1
    metrics: dict[str, Any] = {
        "answer_loss_sum": first_loss if answer_ids else 0.0,
        "answer_correct": first_correct,
        "answer_total": 1 if answer_ids else 0,
        "answer_target_rank": first_rank if answer_ids else 0,
        "answer_target_margin": first_margin if answer_ids else 0.0,
        "answer_target_rank_sum": float(first_rank if answer_ids else 0),
        "answer_target_margin_sum": float(first_margin if answer_ids else 0.0),
        "answer_error_count": int((1 - first_correct) if answer_ids else 0),
        "full_answer_loss_sum": loss_sum,
        "full_answer_token_correct": token_correct,
        "full_answer_token_total": len(answer_ids),
        "full_answer_sequence_correct": int(list(greedy_preds) == [int(x) for x in answer_ids]),
        "full_answer_sequence_total": 1 if answer_ids else 0,
        "full_answer_target_rank_sum": full_rank_sum,
        "full_answer_target_margin_sum": full_margin_sum,
        "full_answer_error_count": int(full_error_count),
        "first_pred": first_pred,
        "first_target_prob": first_prob,
        "teacher_forced_preds": teacher_forced_preds,
        "greedy_preds": greedy_preds,
        "greedy_lookahead_applied": int(greedy_lookahead_applied),
        "component_rows": component_rows,
    }
    for k in RANK_DIAGNOSTIC_KS:
        metrics[f"answer_top{k}_hit"] = int(answer_top_hits[k])
        metrics[f"answer_error_top{k}_hit"] = int(answer_error_top_hits[k])
        metrics[f"full_answer_top{k}_hit"] = int(full_top_hits[k])
        metrics[f"full_answer_error_top{k}_hit"] = int(full_error_top_hits[k])
    return metrics


def train_answer_token(
    memory: Any,
    prompt_ids: Sequence[int],
    target_id: int,
    temperature: float,
) -> dict[str, Any]:
    reset_dynamic(memory)
    observe_prompt(memory, prompt_ids)
    context = np.array(list(prompt_ids)[-int(memory.max_order) :], dtype=np.int64)
    loss, pred, _ = softmax_loss_and_pred(
        scores_for_answer_slot(memory, context, 0),
        int(target_id),
        temperature,
    )
    if hasattr(memory, "update_answer_slot"):
        memory.update_answer_slot(context, int(target_id), 0)
    else:
        memory.update(context, int(target_id))
    correct = int(pred == int(target_id))
    return {
        "token_loss_sum": loss,
        "token_correct": correct,
        "token_total": 1,
        "answer_loss_sum": loss,
        "answer_correct": correct,
        "answer_total": 1,
        "full_answer_loss_sum": loss,
        "full_answer_token_correct": correct,
        "full_answer_token_total": 1,
        "full_answer_sequence_correct": correct,
        "full_answer_sequence_total": 1,
    }


def greedy_answer_prefix_predictions(
    memory: Any,
    prompt_ids: Sequence[int],
    answer_count: int,
    temperature: float,
) -> list[int]:
    reset_dynamic(memory)
    observe_prompt(memory, prompt_ids)
    order = int(memory.max_order)
    history = list(int(x) for x in prompt_ids)
    preds: list[int] = []
    for slot in range(int(answer_count)):
        context = np.array(history[-order:], dtype=np.int64)
        scores = scores_for_answer_slot(memory, context, slot)
        probs = phase.softmax(scores, temperature)
        pred = int(np.argmax(probs))
        preds.append(pred)
        if hasattr(memory, "observe"):
            memory.observe(context, pred)
        history.append(pred)
    return preds


def apply_predicted_prefix_credit(
    memory: Any,
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    predicted_ids: Sequence[int],
    credit_mode: str,
    skip_teacher_match: bool,
) -> None:
    if credit_mode == "none" or not hasattr(memory, "update_answer_slot_predicted_prefix"):
        return
    reset_dynamic(memory)
    observe_prompt(memory, prompt_ids)
    order = int(memory.max_order)
    history = list(int(x) for x in prompt_ids)
    teacher_prefix: list[int] = []
    predicted_prefix: list[int] = []
    for slot, target in enumerate(answer_ids):
        context = np.array(history[-order:], dtype=np.int64)
        if slot > 0:
            prefix_matches = predicted_prefix == teacher_prefix
            if skip_teacher_match and prefix_matches:
                if hasattr(memory, "predicted_prefix_skipped_teacher_match"):
                    memory.predicted_prefix_skipped_teacher_match += 1
            else:
                memory.update_answer_slot_predicted_prefix(
                    context,
                    int(target),
                    int(slot),
                    credit_mode,
                )
        pred = int(predicted_ids[slot]) if slot < len(predicted_ids) else int(target)
        if hasattr(memory, "observe"):
            memory.observe(context, pred)
        history.append(pred)
        predicted_prefix.append(pred)
        teacher_prefix.append(int(target))


def train_prompt_answer_sequence(
    memory: Any,
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    temperature: float,
    update: bool,
    predicted_prefix_credit: str = "none",
    predicted_prefix_skip_teacher_match: bool = True,
) -> dict[str, Any]:
    reset_dynamic(memory)
    predicted_ids: list[int] = []
    if update and predicted_prefix_credit != "none" and len(answer_ids) > 1:
        predicted_ids = greedy_answer_prefix_predictions(
            memory,
            prompt_ids,
            len(answer_ids),
            temperature,
        )
        reset_dynamic(memory)
    order = int(memory.max_order)
    seq = list(int(x) for x in prompt_ids) + list(int(x) for x in answer_ids)
    answer_pos = len(prompt_ids)
    history: list[int] = []
    loss_sum = 0.0
    correct = 0
    total = 0
    answer_loss = 0.0
    answer_correct = 0
    answer_total = 0
    full_answer_loss = 0.0
    full_answer_correct = 0
    full_answer_total = 0
    full_answer_sequence_correct = 1 if answer_ids else 0
    full_answer_sequence_total = 1 if answer_ids else 0
    for idx in range(len(seq) - 1):
        current = int(seq[idx])
        target = int(seq[idx + 1])
        context = np.array((history + [current])[-order:], dtype=np.int64)
        is_answer = idx + 1 >= answer_pos
        slot = (idx + 1) - answer_pos
        scores = scores_for_answer_slot(memory, context, slot) if is_answer else memory.scores(context)
        loss, pred, _ = softmax_loss_and_pred(scores, target, temperature)
        loss_sum += loss
        correct += int(pred == target)
        total += 1
        if idx + 1 == answer_pos:
            answer_loss += loss
            answer_correct += int(pred == target)
            answer_total += 1
        if is_answer:
            is_correct = int(pred == target)
            full_answer_loss += loss
            full_answer_correct += is_correct
            full_answer_total += 1
            full_answer_sequence_correct &= is_correct
        if update:
            if is_answer and hasattr(memory, "update_answer_slot"):
                memory.update_answer_slot(context, target, slot)
            else:
                memory.update(context, target)
        elif hasattr(memory, "observe"):
            memory.observe(context, target)
        history = (history + [current])[-max(order - 1, 0) :]
    if update and predicted_ids:
        apply_predicted_prefix_credit(
            memory,
            prompt_ids,
            answer_ids,
            predicted_ids,
            predicted_prefix_credit,
            predicted_prefix_skip_teacher_match,
        )
    return {
        "token_loss_sum": loss_sum,
        "token_correct": correct,
        "token_total": total,
        "answer_loss_sum": answer_loss,
        "answer_correct": answer_correct,
        "answer_total": answer_total,
        "full_answer_loss_sum": full_answer_loss,
        "full_answer_token_correct": full_answer_correct,
        "full_answer_token_total": full_answer_total,
        "full_answer_sequence_correct": full_answer_sequence_correct,
        "full_answer_sequence_total": full_answer_sequence_total,
    }


def summarize_loss(loss_sum: float, correct: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {"loss": float("nan"), "ppl": float("nan"), "accuracy": 0.0}
    loss = loss_sum / total
    return {
        "loss": float(loss),
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / total,
    }


def decode_one(tokenizer: Any, kept_raw: np.ndarray, compact_id: int) -> str:
    if compact_id < 0 or compact_id >= len(kept_raw):
        return ""
    return tokenizer.decode([int(kept_raw[int(compact_id)])], skip_special_tokens=True)


def decode_many(tokenizer: Any, kept_raw: np.ndarray, compact_ids: Sequence[int]) -> str:
    raw_ids = [
        int(kept_raw[int(compact_id)])
        for compact_id in compact_ids
        if 0 <= int(compact_id) < len(kept_raw)
    ]
    if not raw_ids:
        return ""
    return tokenizer.decode(raw_ids, skip_special_tokens=True).strip()


def split_rows_for_config(args: argparse.Namespace, config: str) -> dict[str, list[dict[str, Any]]]:
    config_dir = args.data_dir / config
    return {
        "train": read_jsonl(config_dir / "train.jsonl", args.train_limit or None),
        "validation": read_jsonl(config_dir / "validation.jsonl", args.eval_limit or None),
        "test": read_jsonl(config_dir / "test.jsonl", args.eval_limit or None),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--configs", nargs="+", default=["en-qa1"])
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "babi_unified_token_qa")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-vocab", type=int, default=512)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument(
        "--prediction-row-limit",
        type=int,
        default=50,
        help="Rows per eval split written to predictions_sample.csv; use 0 to write all evaluated rows.",
    )
    parser.add_argument(
        "--prediction-component-margins",
        action="store_true",
        help="Write per-answer-token component margins/probabilities to prediction_components.csv.",
    )
    parser.add_argument(
        "--answer-only-train",
        action="store_true",
        help="During bAbI tuning, update only the first answer token after the prompt.",
    )
    parser.add_argument("--pretrain-file", type=Path, default=DEFAULT_PRETRAIN_FILE)
    parser.add_argument("--pretrain-chars", type=int, default=0)
    parser.add_argument("--pretrain-token-limit", type=int, default=0)
    parser.add_argument("--require-single-token-answer", action="store_true", default=True)
    parser.add_argument("--allow-multi-token-answer", dest="require_single_token_answer", action="store_false")
    parser.add_argument(
        "--answer-slot-readout",
        action="store_true",
        help="Add a default-off local answer-position prototype readout for multi-token answers.",
    )
    parser.add_argument("--answer-slot-count", type=int, default=2)
    parser.add_argument("--answer-slot-slots", type=int, default=32)
    parser.add_argument("--answer-slot-lr", type=float, default=0.35)
    parser.add_argument("--answer-slot-wrong-lr", type=float, default=0.02)
    parser.add_argument("--answer-slot-score-scale", type=float, default=2.0)
    parser.add_argument("--answer-slot-margin", type=float, default=0.0)
    parser.add_argument(
        "--answer-slot-direct-pre-margin-protect",
        type=float,
        default=0.0,
        help="When direct edge scores are enabled, suppress them if pre-direct answer readout margin is at least this value.",
    )
    parser.add_argument(
        "--answer-slot-direct-margin-min",
        type=float,
        default=0.0,
        help="When direct edge scores are enabled, suppress them unless direct-delta margin reaches this value.",
    )
    parser.add_argument("--answer-slot-coupling-slots", type=int, default=16)
    parser.add_argument("--answer-slot-coupling-lr", type=float, default=0.35)
    parser.add_argument("--answer-slot-coupling-wrong-lr", type=float, default=0.02)
    parser.add_argument("--answer-slot-coupling-score-scale", type=float, default=0.0)
    parser.add_argument("--answer-slot-wrong-cleanup-slots", type=int, default=8)
    parser.add_argument("--answer-slot-wrong-cleanup-lr", type=float, default=0.35)
    parser.add_argument("--answer-slot-wrong-cleanup-disinhibit-lr", type=float, default=0.05)
    parser.add_argument("--answer-slot-wrong-cleanup-score-scale", type=float, default=0.0)
    parser.add_argument("--answer-slot-wrong-cleanup-min-slot", type=int, default=1)
    parser.add_argument(
        "--answer-slot-wrong-cleanup-protect-mode",
        choices=["none", "positive_delta"],
        default="none",
    )
    parser.add_argument("--answer-slot-wrong-cleanup-protect-threshold", type=float, default=0.5)
    parser.add_argument("--answer-slot-conflict-rescue-slots", type=int, default=8)
    parser.add_argument("--answer-slot-conflict-rescue-lr", type=float, default=0.35)
    parser.add_argument("--answer-slot-conflict-rescue-score-scale", type=float, default=0.0)
    parser.add_argument("--answer-slot-conflict-rescue-top-k", type=int, default=4)
    parser.add_argument("--answer-slot-conflict-rescue-min-slot", type=int, default=1)
    parser.add_argument(
        "--answer-slot-conflict-rescue-min-support",
        type=float,
        default=0.0,
        help="Require this much positive slot/coupling support before applying conflict rescue; 0 disables.",
    )
    parser.add_argument(
        "--answer-slot-conflict-rescue-prefix-gate",
        choices=["none", "observed_pred", "margin", "observed_pred_margin"],
        default="none",
        help="Gate conflict rescue using the previous answer-slot prefix consistency/confidence.",
    )
    parser.add_argument(
        "--answer-slot-conflict-rescue-prefix-margin",
        type=float,
        default=0.0,
        help="Previous answer-slot margin required by margin-based conflict-rescue prefix gates.",
    )
    parser.add_argument(
        "--answer-slot-predicted-prefix-credit",
        choices=["none", "coupling", "conflict", "coupling_conflict"],
        default="none",
        help="During answer training, add local credit under the model's own predicted answer prefix.",
    )
    parser.add_argument(
        "--answer-slot-predicted-prefix-include-teacher-match",
        dest="answer_slot_predicted_prefix_skip_teacher_match",
        action="store_false",
        help="Also apply predicted-prefix credit when the predicted prefix equals the teacher-forced prefix.",
    )
    parser.set_defaults(answer_slot_predicted_prefix_skip_teacher_match=True)
    parser.add_argument(
        "--answer-slot-predicted-prefix-target-top-k",
        type=int,
        default=0,
        help="Only apply predicted-prefix credit when the true target is within this rank; 0 disables.",
    )
    parser.add_argument(
        "--answer-slot-predicted-prefix-lr-scale",
        type=float,
        default=1.0,
        help="Temporary local learning-rate multiplier for predicted-prefix credit updates.",
    )
    parser.add_argument(
        "--answer-slot-predicted-prefix-target-only-coupling",
        dest="answer_slot_predicted_prefix_coupling_wrong_credit",
        action="store_false",
        help="For predicted-prefix coupling credit, update the target prototype but skip coupling wrong-credit.",
    )
    parser.set_defaults(answer_slot_predicted_prefix_coupling_wrong_credit=True)
    parser.add_argument("--answer-candidate-arbiter-rank", type=int, default=32)
    parser.add_argument("--answer-candidate-arbiter-lr", type=float, default=0.001)
    parser.add_argument(
        "--answer-candidate-arbiter-score-scale",
        type=float,
        default=0.0,
        help="Add a low-rank local residual over current answer candidates; 0 disables.",
    )
    parser.add_argument("--answer-candidate-arbiter-top-k", type=int, default=4)
    parser.add_argument(
        "--answer-candidate-arbiter-min-support",
        type=float,
        default=0.0,
        help="Require this much positive slot/path support before candidate-arbiter scoring or updates; 0 disables.",
    )
    parser.add_argument("--answer-candidate-arbiter-min-slot", type=int, default=0)
    parser.add_argument("--answer-candidate-arbiter-projection-decay", type=float, default=1.0)
    parser.add_argument("--answer-candidate-arbiter-clip", type=float, default=0.75)
    parser.add_argument(
        "--answer-lookahead-candidates",
        type=int,
        default=0,
        help="During greedy multi-token answer decoding, rescore this many current-slot candidates by one-step future answer consistency; 0 disables.",
    )
    parser.add_argument("--answer-lookahead-weight", type=float, default=1.0)
    parser.add_argument(
        "--answer-slot-feature-mode",
        choices=[
            "base",
            "role_hop",
            "edge_path",
            "edge_path_wta",
            "edge_path_soft",
            "edge_path_soft_direct",
        ],
        default="base",
        help="Feature used by answer-slot prototypes; non-base modes use role-transition traces when available.",
    )
    parser.add_argument(
        "--answer-slot-no-base-update",
        dest="answer_slot_update_base",
        action="store_false",
        help="When answer-slot readout is enabled, update only the slot bank on answer tokens.",
    )
    parser.set_defaults(answer_slot_update_base=True)
    parser.add_argument(
        "--method",
        choices=[
            "phase_competitive_online",
            "phase_trace_competitive_online",
            "phase_trace_dll_local_competitive_online",
            "phase_trace_noprop_local_competitive_online",
            "phase_trace_kv_competitive_online",
            "state_microproto_online",
            "state_event_cleanup_online",
            "state_role_transition_online",
        ],
        default="phase_trace_competitive_online",
    )
    parser.add_argument("--phase-dim", type=int, default=64)
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
    parser.add_argument("--trace-order", type=int, default=96)
    parser.add_argument("--trace-dim", type=int, default=64)
    parser.add_argument("--trace-decay", type=float, default=0.85)
    parser.add_argument("--trace-weight", type=float, default=1.0)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--state-order", type=int, default=128)
    parser.add_argument("--state-decay", type=float, default=0.90)
    parser.add_argument("--micro-slots", type=int, default=64)
    parser.add_argument("--micro-lr", type=float, default=0.35)
    parser.add_argument("--micro-wrong-lr", type=float, default=0.02)
    parser.add_argument("--micro-score-scale", type=float, default=8.0)
    parser.add_argument("--micro-margin", type=float, default=0.0)
    parser.add_argument("--binding-hops", type=int, default=0)
    parser.add_argument("--binding-window", type=int, default=12)
    parser.add_argument("--binding-query-order", type=int, default=8)
    parser.add_argument("--binding-query-mode", choices=["recent_trace", "prefix_overlap"], default="recent_trace")
    parser.add_argument("--binding-focus-k", type=int, default=2)
    parser.add_argument("--binding-decay", type=float, default=0.95)
    parser.add_argument("--binding-bidirectional", action="store_true")
    parser.add_argument("--binding-mode", choices=["pair_apply", "span_sparse"], default="pair_apply")
    parser.add_argument("--binding-span-window", type=int, default=6)
    parser.add_argument("--binding-span-top-k", type=int, default=4)
    parser.add_argument("--binding-span-decay", type=float, default=0.95)
    parser.add_argument("--binding-span-learned-gate", action="store_true")
    parser.add_argument("--binding-span-gate-lr", type=float, default=0.05)
    parser.add_argument("--binding-span-gate-neg-lr", type=float, default=0.05)
    parser.add_argument("--binding-span-gate-strength", type=float, default=0.75)
    parser.add_argument("--binding-span-gate-clip", type=float, default=2.0)
    parser.add_argument("--latent-transition-branch", action="store_true")
    parser.add_argument("--transition-window", type=int, default=6)
    parser.add_argument("--transition-passes", type=int, default=2)
    parser.add_argument("--transition-decay", type=float, default=0.98)
    parser.add_argument("--transition-threshold", type=float, default=0.10)
    parser.add_argument("--transition-strength", type=float, default=0.85)
    parser.add_argument("--event-cell-branch", action="store_true")
    parser.add_argument("--event-cell-count", type=int, default=64)
    parser.add_argument("--event-cell-window", type=int, default=4)
    parser.add_argument("--event-cell-top-k", type=int, default=12)
    parser.add_argument("--event-cell-lr", type=float, default=0.08)
    parser.add_argument("--event-cell-credit-lr", type=float, default=0.05)
    parser.add_argument("--event-cell-neg-lr", type=float, default=0.03)
    parser.add_argument("--event-cell-query-weight", type=float, default=1.0)
    parser.add_argument("--event-cell-recency-decay", type=float, default=0.98)
    parser.add_argument("--assembly-hops", type=int, default=2)
    parser.add_argument("--assembly-event-window", type=int, default=3)
    parser.add_argument("--assembly-seed-top-k", type=int, default=5)
    parser.add_argument("--assembly-recency-decay", type=float, default=0.95)
    parser.add_argument("--assembly-locality-decay", type=float, default=0.90)
    parser.add_argument("--cleanup-slots", type=int, default=0)
    parser.add_argument("--cleanup-lr", type=float, default=0.35)
    parser.add_argument("--cleanup-wrong-lr", type=float, default=0.02)
    parser.add_argument("--cleanup-score-scale", type=float, default=3.0)
    parser.add_argument("--cleanup-top-k", type=int, default=4)
    parser.add_argument("--cleanup-inhibit", type=float, default=0.25)
    parser.add_argument("--role-query-order", type=int, default=16)
    parser.add_argument("--role-hops", type=int, default=2)
    parser.add_argument("--role-window", type=int, default=4)
    parser.add_argument("--role-top-k", type=int, default=6)
    parser.add_argument("--role-recency-decay", type=float, default=0.98)
    parser.add_argument("--role-locality-decay", type=float, default=0.90)
    parser.add_argument("--role-gate-lr", type=float, default=0.08)
    parser.add_argument("--role-gate-wrong-lr", type=float, default=0.04)
    parser.add_argument("--role-gate-strength", type=float, default=1.0)
    parser.add_argument("--role-score-scale", type=float, default=1.5)
    parser.add_argument("--role-downstream-bonus", type=float, default=0.75)
    parser.add_argument("--role-channel-gates", action="store_true")
    parser.add_argument("--role-final-score-only", action="store_true")
    parser.add_argument("--role-score-top-k", type=int, default=0)
    parser.add_argument("--role-score-inhibit", type=float, default=0.0)
    parser.add_argument(
        "--role-score-gate-mode",
        choices=["none", "base_low_margin", "role_high_margin", "base_low_and_role_high"],
        default="none",
    )
    parser.add_argument("--role-score-gate-base-margin", type=float, default=0.0)
    parser.add_argument("--role-score-gate-role-margin", type=float, default=0.0)
    parser.add_argument(
        "--role-branch-readout",
        action="store_true",
        help="Use separate base-state and role-state prototype readouts instead of one concatenated bank.",
    )
    parser.add_argument("--role-branch-base-score-scale", type=float, default=8.0)
    parser.add_argument("--role-branch-role-score-scale", type=float, default=8.0)
    parser.add_argument(
        "--role-joint-rescue-readout",
        action="store_true",
        help="Add a parallel full base+role prototype rescue path on top of branch-separated readout.",
    )
    parser.add_argument("--role-joint-rescue-score-scale", type=float, default=2.0)
    parser.add_argument("--role-joint-rescue-top-k", type=int, default=0)
    parser.add_argument("--role-joint-rescue-inhibit", type=float, default=0.0)
    parser.add_argument("--role-joint-suppress-slots", type=int, default=16)
    parser.add_argument("--role-joint-suppress-lr", type=float, default=0.35)
    parser.add_argument("--role-joint-suppress-score-scale", type=float, default=0.0)
    parser.add_argument("--role-joint-suppress-margin", type=float, default=0.0)
    parser.add_argument(
        "--role-joint-suppress-mode",
        choices=["all_wrong", "protect_direct", "joint_only"],
        default="all_wrong",
        help="Condition joint suppression on local evidence; default preserves the original R173 rule.",
    )
    parser.add_argument("--role-joint-suppress-direct-threshold", type=float, default=0.05)
    parser.add_argument("--role-joint-suppress-joint-threshold", type=float, default=0.0)
    parser.add_argument(
        "--role-branch-arbiter",
        choices=[
            "none",
            "local_proto",
            "base_margin_adaptive",
            "base_margin_rescue",
            "agreement_base_protect",
            "conflict_proto",
        ],
        default="none",
        help="Optional no-BP WTA arbiter over base/role/direct branch score paths.",
    )
    parser.add_argument(
        "--role-branch-arbiter-default",
        choices=[
            "base_only",
            "role_only",
            "base_plus_role",
            "base_plus_direct",
            "base_plus_role_joint",
            "base_plus_direct_joint",
        ],
        default="base_plus_direct",
        help="Branch path used before the local arbiter has any prototype evidence.",
    )
    parser.add_argument("--role-branch-arbiter-slots", type=int, default=16)
    parser.add_argument("--role-branch-arbiter-lr", type=float, default=0.35)
    parser.add_argument("--role-branch-arbiter-wrong-lr", type=float, default=0.05)
    parser.add_argument("--role-branch-arbiter-score-scale", type=float, default=4.0)
    parser.add_argument("--role-branch-arbiter-margin", type=float, default=0.0)
    parser.add_argument(
        "--role-branch-arbiter-min-count",
        type=float,
        default=0.0,
        help="For conflict_proto, require this much repeated base evidence before choosing base_only.",
    )
    parser.add_argument("--role-branch-arbiter-base-margin", type=float, default=0.20)
    parser.add_argument("--role-branch-arbiter-threshold-lr", type=float, default=0.10)
    parser.add_argument("--role-branch-arbiter-rescue-role-threshold", type=float, default=0.05)
    parser.add_argument("--role-branch-arbiter-rescue-joint-threshold", type=float, default=1e9)
    parser.add_argument(
        "--role-branch-arbiter-joint-variants",
        action="store_true",
        help="Let the local branch arbiter choose joint rescue paths in addition to base/role/direct paths.",
    )
    parser.add_argument(
        "--role-branch-arbiter-rich-conflict-features",
        action="store_true",
        help="Add winner-pair and role/joint evidence features to the branch arbiter feature.",
    )
    parser.add_argument(
        "--role-event-cache-size",
        type=int,
        default=0,
        help="Optional transient cache for fixed role event features; stores no raw prompt text.",
    )
    parser.add_argument(
        "--edge-path-cleanup-answer-slots",
        type=int,
        default=2,
        help="Answer positions covered by edge-path WTA cleanup prototypes.",
    )
    parser.add_argument("--edge-path-cleanup-slots", type=int, default=8)
    parser.add_argument("--edge-path-cleanup-lr", type=float, default=0.20)
    parser.add_argument("--edge-path-cleanup-wrong-lr", type=float, default=0.03)
    parser.add_argument("--edge-path-cleanup-score-scale", type=float, default=0.75)
    parser.add_argument("--edge-path-cleanup-top-k", type=int, default=1)
    parser.add_argument("--edge-path-cleanup-inhibit", type=float, default=0.25)
    parser.add_argument(
        "--edge-path-cleanup-credit-mode",
        choices=[
            "selected_target",
            "reward_punish",
            "soft_eligibility",
            "margin_gated_soft_eligibility",
            "learned_margin_escape",
            "transient_inhibit_escape",
        ],
        default="selected_target",
        help="Local cleanup credit rule: old selected-target updates or reward/punish path competition from answer-slot success.",
    )
    parser.add_argument("--edge-path-margin-gate", type=float, default=0.10)
    parser.add_argument("--edge-path-margin-min-scale", type=float, default=0.10)
    parser.add_argument("--edge-path-margin-alt-scale", type=float, default=0.25)
    parser.add_argument("--edge-path-margin-learned-dominance", type=float, default=1.0)
    parser.add_argument("--edge-path-margin-escape-scale", type=float, default=0.50)
    parser.add_argument("--edge-path-transient-inhibit-scale", type=float, default=0.0)
    parser.add_argument("--edge-path-transient-inhibit-lr", type=float, default=0.50)
    parser.add_argument("--edge-path-transient-inhibit-decay", type=float, default=0.90)
    parser.add_argument(
        "--edge-path-transient-inhibit-key",
        choices=["mid", "path_hash", "anchor_path"],
        default="mid",
    )
    parser.add_argument("--edge-path-transient-inhibit-hash-size", type=int, default=512)
    parser.add_argument("--edge-path-transient-boost-scale", type=float, default=0.0)
    parser.add_argument("--edge-path-transient-boost-lr", type=float, default=0.0)
    parser.add_argument("--edge-path-transient-boost-support-margin", type=float, default=0.10)
    parser.add_argument(
        "--edge-path-transient-boost-consistency-margin",
        type=float,
        default=-1.0,
        help="Enable local runner boost only when runner consistency plus this margin reaches selected consistency; negative disables the gate.",
    )
    parser.add_argument(
        "--edge-path-transient-boost-runner-learned-max",
        type=float,
        default=-1.0,
        help="Enable runner boost only for weakly self-confirmed runners with learned score at or below this value; negative disables the gate.",
    )
    parser.add_argument(
        "--edge-path-transient-boost-counterfactual-min-gain",
        type=float,
        default=-1.0,
        help="Enable runner boost only when local runner counterfactual target margin gain reaches this threshold; <= -1 disables the gate.",
    )
    parser.add_argument(
        "--edge-path-homeostasis-scale",
        type=float,
        default=0.0,
        help="Subtract a local usage trace from repeated edge-path winners; 0 disables.",
    )
    parser.add_argument("--edge-path-homeostasis-lr", type=float, default=0.0)
    parser.add_argument("--edge-path-homeostasis-decay", type=float, default=0.98)
    parser.add_argument(
        "--edge-path-homeostasis-min-slot",
        type=int,
        default=0,
        help="Only apply/update edge-path homeostasis for answer slots at or above this index.",
    )
    parser.add_argument(
        "--edge-path-homeostasis-learned-dominance",
        type=float,
        default=0.0,
        help="When >0, apply/update homeostasis only for learned-dominant paths with weak local structure.",
    )
    parser.add_argument(
        "--edge-path-homeostasis-structure-margin",
        type=float,
        default=0.0,
        help="Allowed support/closure/consistency margin before a path is considered structurally weak for homeostasis.",
    )
    parser.add_argument(
        "--edge-path-homeostasis-soft-mod-scale",
        type=float,
        default=0.0,
        help="When >0, replace hard homeostasis gating with a continuous local multiplier from weak-structure/learned pressure.",
    )
    parser.add_argument(
        "--edge-path-homeostasis-soft-mod-floor",
        type=float,
        default=1.0,
        help="Minimum multiplier for soft homeostasis modulation; 1 preserves old behavior.",
    )
    parser.add_argument(
        "--edge-path-homeostasis-trace-threshold",
        type=float,
        default=0.0,
        help="Only the portion of a local usage trace above this threshold contributes to homeostasis penalty.",
    )
    parser.add_argument(
        "--edge-path-homeostasis-trace-gain",
        type=float,
        default=1.0,
        help="Gain applied after trace-threshold compression; 1 preserves old behavior when threshold is 0.",
    )
    parser.add_argument("--edge-path-runner-arbiter-slots", type=int, default=8)
    parser.add_argument("--edge-path-runner-arbiter-lr", type=float, default=0.20)
    parser.add_argument("--edge-path-runner-arbiter-wrong-lr", type=float, default=0.03)
    parser.add_argument("--edge-path-runner-arbiter-score-scale", type=float, default=0.0)
    parser.add_argument("--edge-path-runner-arbiter-margin", type=float, default=0.0)
    parser.add_argument(
        "--edge-path-runner-arbiter-min-count",
        type=float,
        default=0.0,
        help="Require the best positive pair-arbiter prototype to have this count before scoring; 0 preserves the original R240 behavior.",
    )
    parser.add_argument(
        "--edge-path-runner-arbiter-negative-mode",
        choices=["subtract", "separate"],
        default="subtract",
        help="Use old anti-Hebbian subtraction from positive prototypes, or a separate local negative prototype bank.",
    )
    parser.add_argument(
        "--edge-path-runner-arbiter-feature-mode",
        choices=["pair", "rich_gaps"],
        default="pair",
        help="Pair feature for selected-vs-runner arbiter: original path pair or path pair plus bounded score/support/learned/consistency gaps.",
    )
    parser.add_argument(
        "--edge-path-runner-arbiter-gap-scale",
        type=float,
        default=0.50,
        help="Scale for bounded gap channels when --edge-path-runner-arbiter-feature-mode=rich_gaps.",
    )
    parser.add_argument(
        "--edge-path-runner-arbiter-credit-mode",
        choices=["answer_error", "counterfactual_positive"],
        default="answer_error",
        help="Third-factor rule for no-BP selected-vs-runner pair arbiter updates.",
    )
    parser.add_argument("--edge-path-soft-top-k", type=int, default=6)
    parser.add_argument("--edge-path-soft-temperature", type=float, default=0.20)
    parser.add_argument("--edge-path-soft-consistency-scale", type=float, default=0.50)
    parser.add_argument("--edge-path-soft-learned-scale", type=float, default=0.0)
    parser.add_argument(
        "--edge-path-closure-score-scale",
        type=float,
        default=0.0,
        help="Add a local dendritic closure score between path_feature and source/destination agreement during edge-path ranking.",
    )
    parser.add_argument("--edge-path-closure-proto-slots", type=int, default=8)
    parser.add_argument("--edge-path-closure-proto-lr", type=float, default=0.20)
    parser.add_argument("--edge-path-closure-proto-wrong-lr", type=float, default=0.03)
    parser.add_argument(
        "--edge-path-closure-proto-score-scale",
        type=float,
        default=0.0,
        help="Score edge-path candidates with local learned positive-minus-negative closure prototypes.",
    )
    parser.add_argument(
        "--edge-path-closure-proto-min-count",
        type=float,
        default=0.0,
        help="Require the best positive closure prototype to have this count before adding closure-prototype score.",
    )
    parser.add_argument("--edge-path-affinity-slots", type=int, default=2)
    parser.add_argument("--edge-path-affinity-lr", type=float, default=0.20)
    parser.add_argument("--edge-path-affinity-wrong-lr", type=float, default=0.03)
    parser.add_argument(
        "--edge-path-affinity-score-scale",
        type=float,
        default=0.0,
        help="Score edge-path candidates with local compact support/closure affinity prototypes.",
    )
    parser.add_argument(
        "--edge-path-affinity-min-count",
        type=float,
        default=0.0,
        help="Require the best positive affinity prototype to have this count before scoring.",
    )
    parser.add_argument(
        "--edge-path-affinity-margin-gate",
        type=float,
        default=0.0,
        help="When >0, score/update affinity only for candidates within this support margin of the strongest alternative.",
    )
    parser.add_argument(
        "--edge-path-affinity-learned-dominance",
        type=float,
        default=0.0,
        help="When >0, also open the affinity gate for learned-dominant candidates whose closure/consistency evidence is weaker than an alternative.",
    )
    parser.add_argument(
        "--edge-path-affinity-consistency-protect",
        type=float,
        default=0.0,
        help="Tolerance before treating closure/consistency as structurally weaker for learned-dominance affinity gating.",
    )
    parser.add_argument("--edge-path-direct-answer-slots", type=int, default=2)
    parser.add_argument("--edge-path-direct-slots", type=int, default=16)
    parser.add_argument("--edge-path-direct-lr", type=float, default=0.35)
    parser.add_argument("--edge-path-direct-wrong-lr", type=float, default=0.03)
    parser.add_argument("--edge-path-direct-score-scale", type=float, default=1.0)
    parser.add_argument(
        "--edge-path-direct-mode",
        choices=["soft_feature", "candidate_scores", "structured_scores"],
        default="soft_feature",
        help="For edge_path_soft_direct, score averaged path features, each candidate path, or structured source/path/other evidence.",
    )
    parser.add_argument("--edge-path-structured-side-weight", type=float, default=0.50)
    parser.add_argument("--edge-path-structured-path-weight", type=float, default=0.35)
    parser.add_argument("--edge-path-structured-other-weight", type=float, default=0.15)
    parser.add_argument("--kv-order", type=int, default=128)
    parser.add_argument("--kv-dim", type=int, default=64)
    parser.add_argument("--kv-trace-decay", type=float, default=0.95)
    parser.add_argument("--kv-weight", type=float, default=0.50)
    parser.add_argument("--kv-score-weight", type=float, default=0.50)
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
    parser.add_argument("--dll-hidden-dims", type=int, nargs="+", default=[256])
    parser.add_argument("--dll-label-dim", type=int, default=128)
    parser.add_argument("--dll-lr", type=float, default=0.02)
    parser.add_argument("--dll-bias-lr", type=float, default=0.002)
    parser.add_argument("--dll-delta-clip", type=float, default=1.0)
    parser.add_argument("--dll-activation", choices=["tanh", "linear"], default="tanh")
    parser.add_argument("--dll-disable-row-normalize", action="store_true")
    parser.add_argument("--noprop-hidden-dims", type=int, nargs="+", default=[256])
    parser.add_argument("--noprop-label-dim", type=int, default=128)
    parser.add_argument("--noprop-alpha-start", type=float, default=0.85)
    parser.add_argument("--noprop-alpha-end", type=float, default=0.35)
    parser.add_argument("--noprop-lr", type=float, default=0.02)
    parser.add_argument("--noprop-denoise-lr", type=float, default=0.02)
    parser.add_argument("--noprop-bias-lr", type=float, default=0.002)
    parser.add_argument("--noprop-delta-clip", type=float, default=1.0)
    parser.add_argument("--noprop-activation", choices=["tanh", "linear"], default="tanh")
    parser.add_argument("--noprop-disable-row-normalize", action="store_true")
    parser.add_argument("--adaptive-inhibition", action="store_true")
    parser.add_argument("--inhibit-strength", type=float, default=0.15)
    parser.add_argument("--inhibit-decay", type=float, default=0.85)
    parser.add_argument("--inhibit-lr", type=float, default=0.005)
    parser.add_argument("--inhibit-disinhibit-lr", type=float, default=0.004)
    parser.add_argument("--inhibit-top-k", type=int, default=1)
    parser.add_argument("--inhibit-margin", type=float, default=0.0)
    parser.add_argument("--inhibit-max-weight", type=float, default=2.5)
    parser.add_argument("--feature-calibration", action="store_true")
    parser.add_argument("--feature-calibration-strength", type=float, default=1.0)
    parser.add_argument("--feature-calibration-lr", type=float, default=0.02)
    parser.add_argument("--feature-calibration-decay", type=float, default=1.0)
    parser.add_argument("--feature-calibration-clip", type=float, default=2.0)
    parser.add_argument("--feature-calibration-dim", type=int, default=64)
    parser.add_argument("--feature-calibration-gate-decay", type=float, default=0.50)
    parser.add_argument("--feature-calibration-threshold", type=float, default=0.0)
    parser.add_argument("--feature-calibration-derived-codes", action="store_true")
    parser.add_argument("--readout-gain", type=float, default=1.0)
    parser.add_argument("--readout-gain-mode", choices=["fixed", "margin"], default="fixed")
    parser.add_argument("--readout-gain-margin-center", type=float, default=1.0)
    parser.add_argument("--readout-gain-margin-scale", type=float, default=1.0)
    parser.add_argument("--readout-gain-min", type=float, default=0.5)
    parser.add_argument("--readout-gain-max", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.branch_orders) != len(args.branch_weights):
        raise ValueError("--branch-orders and --branch-weights must have the same length")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    splits_by_config = {config: split_rows_for_config(args, config) for config in args.configs}
    train_rows = [
        row
        for config in args.configs
        for row in splits_by_config[config]["train"]
    ]

    answer_raw_tokens: dict[str, list[int]] = {}
    forced_raw: list[int] = []
    for row in train_rows:
        answer = str(row["answer"])
        ids = answer_token_ids(tokenizer, answer)
        answer_raw_tokens.setdefault(answer, ids)
        forced_raw.extend(ids)
    prompt_marker_raw = encode(tokenizer, "Context:\nQuestion: Answer:")
    forced_raw.extend(int(x) for x in prompt_marker_raw)

    pretrain_text = read_prefix(args.pretrain_file, args.pretrain_chars)
    train_text = pretrain_text + "".join(row_train_text(row) for row in train_rows)
    train_raw = encode(tokenizer, train_text)
    kept_raw, raw_to_compact = build_compact_vocab_with_forced(train_raw, forced_raw, args.max_vocab)
    vocab_size = int(len(kept_raw))
    method_name, memory, memory_cfg = build_memory(args, vocab_size)

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    if args.pretrain_chars > 0:
        pretrain_raw = encode(tokenizer, pretrain_text)
        if args.pretrain_token_limit > 0:
            pretrain_raw = pretrain_raw[: args.pretrain_token_limit]
        pretrain_ids = to_compact(pretrain_raw, raw_to_compact)
        pre = train_sequence(memory, pretrain_ids, None, args.temperature, update=True)
        token_summary = summarize_loss(pre["token_loss_sum"], pre["token_correct"], pre["token_total"])
        summary_rows.append(
            {
                "method": method_name,
                "config": "pretrain",
                "split": "pretrain_text",
                "examples": 0,
                "evaluated_examples": 0,
                "skipped_examples": 0,
                "answer_accuracy": 0.0,
                "answer_loss": float("nan"),
                "answer_ppl": float("nan"),
                "token_accuracy": token_summary["accuracy"],
                "token_loss": token_summary["loss"],
                "token_ppl": token_summary["ppl"],
                "token_targets": pre["token_total"],
                "state_bytes": safe_state_bytes(memory),
                "active_contexts": safe_active_contexts(memory),
                "vocab_size": vocab_size,
                "stores_raw_text": False,
                "task_format": "unified_next_token_prompt",
            }
        )

    train_answer_loss = 0.0
    train_answer_correct = 0
    train_answer_total = 0
    train_token_loss = 0.0
    train_token_correct = 0
    train_token_total = 0
    train_full_answer_loss = 0.0
    train_full_answer_token_correct = 0
    train_full_answer_token_total = 0
    train_full_answer_sequence_correct = 0
    train_full_answer_sequence_total = 0
    train_skipped = 0
    for _ in range(max(args.train_epochs, 1)):
        for row in train_rows:
            prompt_ids = to_compact(encode(tokenizer, row_prompt(row)), raw_to_compact)
            answer_ids_raw = answer_token_ids(tokenizer, str(row["answer"]))
            if args.require_single_token_answer and len(answer_ids_raw) != 1:
                train_skipped += 1
                continue
            answer_ids = to_compact(answer_ids_raw, raw_to_compact)
            if not prompt_ids or not answer_ids or len(answer_ids) != len(answer_ids_raw):
                train_skipped += 1
                continue
            if args.answer_only_train:
                metrics = train_answer_token(memory, prompt_ids, int(answer_ids[0]), args.temperature)
            elif hasattr(memory, "update_answer_slot"):
                metrics = train_prompt_answer_sequence(
                    memory,
                    prompt_ids,
                    answer_ids,
                    args.temperature,
                    update=True,
                    predicted_prefix_credit=args.answer_slot_predicted_prefix_credit,
                    predicted_prefix_skip_teacher_match=(
                        args.answer_slot_predicted_prefix_skip_teacher_match
                    ),
                )
            else:
                seq = prompt_ids + answer_ids
                metrics = train_sequence(memory, seq, len(prompt_ids), args.temperature, update=True)
            train_answer_loss += metrics["answer_loss_sum"]
            train_answer_correct += metrics["answer_correct"]
            train_answer_total += metrics["answer_total"]
            train_token_loss += metrics["token_loss_sum"]
            train_token_correct += metrics["token_correct"]
            train_token_total += metrics["token_total"]
            train_full_answer_loss += metrics["full_answer_loss_sum"]
            train_full_answer_token_correct += metrics["full_answer_token_correct"]
            train_full_answer_token_total += metrics["full_answer_token_total"]
            train_full_answer_sequence_correct += metrics["full_answer_sequence_correct"]
            train_full_answer_sequence_total += metrics["full_answer_sequence_total"]

    train_answer_summary = summarize_loss(train_answer_loss, train_answer_correct, train_answer_total)
    train_token_summary = summarize_loss(train_token_loss, train_token_correct, train_token_total)
    train_full_token_summary = summarize_loss(
        train_full_answer_loss,
        train_full_answer_token_correct,
        train_full_answer_token_total,
    )
    summary_rows.append(
        {
            "method": method_name,
            "config": ",".join(args.configs),
            "split": "train_online",
            "examples": len(train_rows) * max(args.train_epochs, 1),
            "evaluated_examples": train_answer_total,
            "skipped_examples": train_skipped,
            "answer_accuracy": train_answer_summary["accuracy"],
            "answer_loss": train_answer_summary["loss"],
            "answer_ppl": train_answer_summary["ppl"],
            "full_answer_accuracy": (
                train_full_answer_sequence_correct / train_full_answer_sequence_total
                if train_full_answer_sequence_total > 0
                else 0.0
            ),
            "full_answer_loss": train_full_token_summary["loss"],
            "full_answer_ppl": train_full_token_summary["ppl"],
            "full_answer_token_accuracy": train_full_token_summary["accuracy"],
            "full_answer_token_targets": train_full_answer_token_total,
            "full_answer_sequences": train_full_answer_sequence_total,
            "token_accuracy": train_token_summary["accuracy"],
            "token_loss": train_token_summary["loss"],
            "token_ppl": train_token_summary["ppl"],
            "token_targets": train_token_total,
            "state_bytes": safe_state_bytes(memory),
            "active_contexts": safe_active_contexts(memory),
            "vocab_size": vocab_size,
            "stores_raw_text": False,
            "task_format": "unified_next_token_prompt",
        }
    )

    for config in args.configs:
        eval_splits = [
            ("train_post", splits_by_config[config]["train"]),
            ("validation", splits_by_config[config]["validation"]),
            ("test", splits_by_config[config]["test"]),
        ]
        for split, rows in eval_splits:
            loss_sum = 0.0
            correct = 0
            total = 0
            full_answer_loss = 0.0
            full_answer_token_correct = 0
            full_answer_token_total = 0
            full_answer_sequence_correct = 0
            full_answer_sequence_total = 0
            greedy_lookahead_applied_total = 0
            answer_rank_sum = 0.0
            answer_margin_sum = 0.0
            answer_error_count = 0
            answer_top_hits = {k: 0 for k in RANK_DIAGNOSTIC_KS}
            answer_error_top_hits = {k: 0 for k in RANK_DIAGNOSTIC_KS}
            full_answer_rank_sum = 0.0
            full_answer_margin_sum = 0.0
            full_answer_error_count = 0
            full_answer_top_hits = {k: 0 for k in RANK_DIAGNOSTIC_KS}
            full_answer_error_top_hits = {k: 0 for k in RANK_DIAGNOSTIC_KS}
            skipped = 0
            for idx, row in enumerate(rows):
                answer_ids_raw = answer_token_ids(tokenizer, str(row["answer"]))
                if args.require_single_token_answer and len(answer_ids_raw) != 1:
                    skipped += 1
                    continue
                prompt_ids = to_compact(encode(tokenizer, row_prompt(row)), raw_to_compact)
                answer_ids = to_compact(answer_ids_raw, raw_to_compact)
                if not prompt_ids or not answer_ids or len(answer_ids) != len(answer_ids_raw):
                    skipped += 1
                    continue
                target = int(answer_ids[0])
                should_write_prediction = args.prediction_row_limit <= 0 or idx < args.prediction_row_limit
                metrics = score_prompt_answer_sequence(
                    memory,
                    prompt_ids,
                    answer_ids,
                    args.temperature,
                    collect_components=args.prediction_component_margins and should_write_prediction,
                    lookahead_candidates=args.answer_lookahead_candidates,
                    lookahead_weight=args.answer_lookahead_weight,
                )
                pred = int(metrics["first_pred"])
                target_prob = float(metrics["first_target_prob"])
                loss_sum += metrics["answer_loss_sum"]
                correct += metrics["answer_correct"]
                total += 1
                full_answer_loss += metrics["full_answer_loss_sum"]
                full_answer_token_correct += metrics["full_answer_token_correct"]
                full_answer_token_total += metrics["full_answer_token_total"]
                full_answer_sequence_correct += metrics["full_answer_sequence_correct"]
                full_answer_sequence_total += metrics["full_answer_sequence_total"]
                greedy_lookahead_applied_total += int(metrics["greedy_lookahead_applied"])
                answer_rank_sum += float(metrics["answer_target_rank_sum"])
                answer_margin_sum += float(metrics["answer_target_margin_sum"])
                answer_error_count += int(metrics["answer_error_count"])
                full_answer_rank_sum += float(metrics["full_answer_target_rank_sum"])
                full_answer_margin_sum += float(metrics["full_answer_target_margin_sum"])
                full_answer_error_count += int(metrics["full_answer_error_count"])
                for k in RANK_DIAGNOSTIC_KS:
                    answer_top_hits[k] += int(metrics[f"answer_top{k}_hit"])
                    answer_error_top_hits[k] += int(metrics[f"answer_error_top{k}_hit"])
                    full_answer_top_hits[k] += int(metrics[f"full_answer_top{k}_hit"])
                    full_answer_error_top_hits[k] += int(metrics[f"full_answer_error_top{k}_hit"])
                if should_write_prediction:
                    prediction_rows.append(
                        {
                            "method": method_name,
                            "config": config,
                            "split": split,
                            "example_index": idx,
                            "question": row["question"],
                            "target_answer": row["answer"],
                            "target_token": decode_one(tokenizer, kept_raw, target),
                            "prediction_token": decode_one(tokenizer, kept_raw, pred),
                            "target_answer_decoded": decode_many(tokenizer, kept_raw, answer_ids),
                            "prediction_answer_decoded": decode_many(
                                tokenizer,
                                kept_raw,
                                metrics["greedy_preds"],
                            ),
                            "target_prob": target_prob,
                            "target_rank": metrics["answer_target_rank"],
                            "target_margin": metrics["answer_target_margin"],
                            "target_top2": metrics["answer_top2_hit"],
                            "target_top4": metrics["answer_top4_hit"],
                            "target_top8": metrics["answer_top8_hit"],
                            "error_target_top2": metrics["answer_error_top2_hit"],
                            "error_target_top4": metrics["answer_error_top4_hit"],
                            "error_target_top8": metrics["answer_error_top8_hit"],
                            "correct": int(pred == target),
                            "full_correct": metrics["full_answer_sequence_correct"],
                            "full_answer_token_correct": metrics["full_answer_token_correct"],
                            "full_answer_token_total": metrics["full_answer_token_total"],
                            "greedy_lookahead_applied": metrics["greedy_lookahead_applied"],
                            "prompt_compact_tokens": len(prompt_ids),
                            "raw_answer_token_count": len(answer_ids_raw),
                        }
                    )
                    if args.prediction_component_margins:
                        for comp in metrics.get("component_rows", []):
                            comp_row = {
                                "method": method_name,
                                "config": config,
                                "split": split,
                                "example_index": idx,
                                "question": row["question"],
                                "target_answer": row["answer"],
                                "target_answer_decoded": decode_many(tokenizer, kept_raw, answer_ids),
                                "raw_answer_token_count": len(answer_ids_raw),
                            }
                            comp_row.update(comp)
                            comp_row["target_token"] = decode_one(
                                tokenizer,
                                kept_raw,
                                int(comp_row.get("target_id", -1)),
                            )
                            comp_row["prediction_token"] = decode_one(
                                tokenizer,
                                kept_raw,
                                int(comp_row.get("prediction_id", -1)),
                            )
                            component_rows.append(comp_row)
            answer_summary = summarize_loss(loss_sum, correct, total)
            full_token_summary = summarize_loss(
                full_answer_loss,
                full_answer_token_correct,
                full_answer_token_total,
            )
            summary_rows.append(
                {
                    "method": method_name,
                    "config": config,
                    "split": split,
                    "examples": len(rows),
                    "evaluated_examples": total,
                    "skipped_examples": skipped,
                    "answer_accuracy": answer_summary["accuracy"],
                    "answer_loss": answer_summary["loss"],
                    "answer_ppl": answer_summary["ppl"],
                    "full_answer_accuracy": (
                        full_answer_sequence_correct / full_answer_sequence_total
                        if full_answer_sequence_total > 0
                        else 0.0
                    ),
                    "full_answer_loss": full_token_summary["loss"],
                    "full_answer_ppl": full_token_summary["ppl"],
                    "full_answer_token_accuracy": full_token_summary["accuracy"],
                    "full_answer_token_targets": full_answer_token_total,
                    "full_answer_sequences": full_answer_sequence_total,
                    "greedy_lookahead_applied": greedy_lookahead_applied_total,
                    "token_accuracy": float("nan"),
                    "token_loss": float("nan"),
                    "token_ppl": float("nan"),
                    "token_targets": 0,
                    "state_bytes": safe_state_bytes(memory),
                    "active_contexts": safe_active_contexts(memory),
                    "vocab_size": vocab_size,
                    "stores_raw_text": False,
                    "task_format": "unified_next_token_prompt",
                    **summarize_rank_diagnostics(
                        "answer",
                        answer_rank_sum,
                        answer_margin_sum,
                        answer_top_hits,
                        answer_error_top_hits,
                        answer_error_count,
                        total,
                    ),
                    **summarize_rank_diagnostics(
                        "full_answer",
                        full_answer_rank_sum,
                        full_answer_margin_sum,
                        full_answer_top_hits,
                        full_answer_error_top_hits,
                        full_answer_error_count,
                        full_answer_token_total,
                    ),
                }
            )

    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "predictions_sample.csv", prediction_rows)
    if args.prediction_component_margins:
        write_csv(args.out_dir / "prediction_components.csv", component_rows)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "method": method_name,
                "memory_cfg": memory_cfg,
                "vocab_size": vocab_size,
                "kept_raw_token_count": int(len(kept_raw)),
                "answer_raw_tokens": answer_raw_tokens,
                "wall_seconds": time.perf_counter() - start_time,
                "event_cache_stats": safe_event_cache_stats(memory),
                "role_score_gate_stats": safe_role_score_gate_stats(memory),
                "role_branch_arbiter_stats": safe_role_branch_arbiter_stats(memory),
                "role_joint_suppress_stats": safe_role_joint_suppress_stats(memory),
                "edge_path_cleanup_stats": safe_edge_path_cleanup_stats(memory),
                "edge_path_direct_stats": safe_edge_path_direct_stats(memory),
                "answer_slot_stats": safe_answer_slot_stats(memory),
                "model_stores_raw_text": False,
                "artifact_contains_decoded_text": True,
                "note": "QA is evaluated as next-token prediction after a prompt, not as a task-specific answer head.",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
