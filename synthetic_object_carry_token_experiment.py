#!/usr/bin/env python3
"""
Parser-free synthetic object-carry token QA.

The task isolates the bAbI QA2 transition that current unified token models do
not learn: an object is picked up by a carrier, the carrier moves, and the model
must answer the object's final location as the next token after "Answer:".

This script uses raw token IDs and the same full-vocabulary no-BP
micro-prototype memory used by the unified bAbI evaluator.  It does not add a
parser, symbolic state, QA head, BP, raw replay, or test-answer updates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from babi_unified_token_qa_experiment import (
    OnlineStateMicroPrototypeMemory,
    softmax_loss_and_pred,
    summarize_loss,
    write_csv,
)
import phase_binding_token_experiment as phase


SCRIPT_DIR = Path(__file__).resolve().parent
ARBITRATED_VARIANTS = {"span_gate"}
EVENT_ASSEMBLY_VARIANTS = {"event_assembly"}
EVENT_CLEANUP_VARIANTS = {"event_cleanup"}


def build_vocab(num_persons: int, num_objects: int, num_locations: int) -> tuple[list[str], dict[str, int]]:
    tokens = [
        "Context:",
        "Question:",
        "Answer:",
        ".",
        "where",
        "is",
        "?",
        "at",
        "pick",
        "move",
    ]
    tokens.extend(f"p{i}" for i in range(num_persons))
    tokens.extend(f"o{i}" for i in range(num_objects))
    tokens.extend(f"l{i}" for i in range(num_locations))
    return tokens, {token: idx for idx, token in enumerate(tokens)}


def encode(tokens: list[str], token_to_id: dict[str, int]) -> list[int]:
    return [int(token_to_id[token]) for token in tokens]


def make_example(rng: np.random.Generator, args: argparse.Namespace) -> dict[str, Any]:
    persons = [f"p{i}" for i in range(args.num_persons)]
    objects = [f"o{i}" for i in range(args.num_objects)]
    locations = [f"l{i}" for i in range(args.num_locations)]
    carrier = str(rng.choice(persons))
    distractor_person = str(rng.choice([p for p in persons if p != carrier]))
    obj = str(rng.choice(objects))
    distractor_object = str(rng.choice([o for o in objects if o != obj]))

    story: list[str] = ["Context:"]
    carrier_loc = str(rng.choice(locations))
    distractor_loc = str(rng.choice(locations))
    object_loc = str(rng.choice(locations))
    story.extend([carrier, "at", carrier_loc, "."])
    story.extend([distractor_person, "at", distractor_loc, "."])
    story.extend([obj, "at", object_loc, "."])
    story.extend([distractor_object, "at", str(rng.choice(locations)), "."])
    story.extend([carrier, "pick", obj, "."])

    for _ in range(args.carrier_moves):
        carrier_loc = str(rng.choice(locations))
        story.extend([carrier, "move", carrier_loc, "."])
        distractor_loc = str(rng.choice(locations))
        story.extend([distractor_person, "move", distractor_loc, "."])

    for _ in range(args.extra_distractors):
        if rng.random() < 0.5:
            story.extend([distractor_person, "move", str(rng.choice(locations)), "."])
        else:
            story.extend([distractor_object, "at", str(rng.choice(locations)), "."])

    prompt = story + ["Question:", "where", "is", obj, "?", "Answer:"]
    return {
        "prompt_tokens": prompt,
        "answer": carrier_loc,
        "carrier": carrier,
        "object": obj,
    }


def make_dataset(args: argparse.Namespace, split: str, count: int, seed_offset: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed + seed_offset)
    return [make_example(rng, args) for _ in range(count)]


def top2_margin(scores: np.ndarray) -> float:
    if scores.size < 2:
        return 0.0
    top_two = np.partition(scores.astype(np.float32, copy=False), -2)[-2:]
    return float(np.max(top_two) - np.min(top_two))


def sigmoid_gate(x: float) -> float:
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-x)))


def build_state_memory(args: argparse.Namespace, vocab_size: int, variant: str) -> OnlineStateMicroPrototypeMemory:
    binding_hops = args.span_binding_hops if variant in {"span", "span_event_cell"} else 0
    event_cell_branch = variant in {"event_cell", "span_event_cell"}
    return OnlineStateMicroPrototypeMemory(
        vocab_size=vocab_size,
        state_dim=args.state_dim,
        state_order=args.state_order,
        state_decay=args.state_decay,
        slots=args.micro_slots,
        lr=args.micro_lr,
        wrong_lr=args.micro_wrong_lr,
        score_scale=args.micro_score_scale,
        bias_weight=args.bias_weight,
        margin=args.micro_margin,
        binding_hops=binding_hops,
        binding_window=args.binding_window,
        binding_query_order=args.binding_query_order,
        binding_query_mode="prefix_overlap",
        binding_focus_k=args.binding_focus_k,
        binding_decay=args.binding_decay,
        binding_bidirectional=False,
        binding_mode="span_sparse",
        binding_span_window=args.binding_span_window,
        binding_span_top_k=args.binding_span_top_k,
        binding_span_decay=args.binding_span_decay,
        binding_span_learned_gate=False,
        binding_span_gate_lr=0.0,
        binding_span_gate_neg_lr=0.0,
        binding_span_gate_strength=0.0,
        binding_span_gate_clip=0.0,
        latent_transition_branch=False,
        transition_window=1,
        transition_passes=1,
        transition_decay=1.0,
        transition_threshold=0.0,
        transition_strength=1.0,
        event_cell_branch=event_cell_branch,
        event_cell_count=args.event_cell_count,
        event_cell_window=args.event_cell_window,
        event_cell_top_k=args.event_cell_top_k,
        event_cell_lr=args.event_cell_lr,
        event_cell_credit_lr=args.event_cell_credit_lr,
        event_cell_neg_lr=args.event_cell_neg_lr,
        event_cell_query_weight=args.event_cell_query_weight,
        event_cell_recency_decay=args.event_cell_recency_decay,
        seed=args.seed,
    )


class LocalSpanArbitrationMemory:
    """
    Parallel no-BP baseline/span memories with local confidence arbitration.

    Both branches learn online from the same target.  At readout time a scalar
    gate derived from branch score margins decides how much span score to mix
    into the base recurrent trace score.  The gate does not use labels, BP, or
    test-set feedback.
    """

    def __init__(self, args: argparse.Namespace, vocab_size: int) -> None:
        self.base = build_state_memory(args, vocab_size, "baseline")
        self.span = build_state_memory(args, vocab_size, "span")
        self.max_order = max(int(self.base.max_order), int(self.span.max_order))
        self.gate_mode = str(args.arbitration_gate_mode)
        self.margin_threshold = float(args.arbitration_margin_threshold)
        self.temperature = max(float(args.arbitration_temperature), 1e-6)
        self.span_gain = float(args.arbitration_span_gain)
        self.last_gate_weight = 0.0
        self.last_base_margin = 0.0
        self.last_span_margin = 0.0

    def gate_weight(self, base_margin: float, span_margin: float) -> float:
        if self.gate_mode == "hard_low_margin":
            return 1.0 if base_margin < self.margin_threshold else 0.0
        if self.gate_mode == "soft_low_margin":
            return sigmoid_gate((self.margin_threshold - base_margin) / self.temperature)
        if self.gate_mode == "soft_span_advantage":
            return sigmoid_gate((span_margin - base_margin - self.margin_threshold) / self.temperature)
        raise ValueError(f"unknown arbitration gate mode: {self.gate_mode}")

    def scores(self, context: list[int] | np.ndarray) -> np.ndarray:
        base_scores = self.base.scores(context)
        span_scores = self.span.scores(context)
        base_margin = top2_margin(base_scores)
        span_margin = top2_margin(span_scores)
        gate = self.gate_weight(base_margin, span_margin)
        self.last_gate_weight = gate
        self.last_base_margin = base_margin
        self.last_span_margin = span_margin
        return ((1.0 - gate) * base_scores + gate * self.span_gain * span_scores).astype(np.float32)

    def update(self, context: list[int] | np.ndarray, target: int) -> None:
        self.base.update(context, int(target))
        self.span.update(context, int(target))

    def state_bytes(self) -> int:
        return int(self.base.state_bytes() + self.span.state_bytes())

    def active_contexts(self) -> int:
        return int(self.base.active_contexts() + self.span.active_contexts())

    def config(self) -> dict[str, Any]:
        return {
            "arbitration_gate_mode": self.gate_mode,
            "arbitration_margin_threshold": self.margin_threshold,
            "arbitration_temperature": self.temperature,
            "arbitration_span_gain": self.span_gain,
        }


class QueryEventAssemblyMemory:
    """
    Query-seeded local event assembly with full-vocabulary local readout.

    The assembly starts from query tokens that also appear in the prompt prefix,
    gathers local neighbor events around those seeds, then repeats seed selection
    for a small number of hops.  It uses token identity and local windows only;
    there is no grammar parser, symbolic object state, BP, or raw replay.
    """

    def __init__(self, args: argparse.Namespace, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.state_dim = max(int(args.state_dim), 1)
        self.max_order = max(int(args.state_order), 1)
        self.state_decay = float(np.clip(args.state_decay, 0.0, 0.999))
        self.slots = max(int(args.micro_slots), 1)
        self.lr = float(np.clip(args.micro_lr, 0.0, 1.0))
        self.wrong_lr = float(np.clip(args.micro_wrong_lr, 0.0, 1.0))
        self.score_scale = float(args.micro_score_scale)
        self.bias_weight = float(args.bias_weight)
        self.margin = float(args.micro_margin)
        self.query_order = max(int(args.binding_query_order), 1)
        self.hops = max(int(args.assembly_hops), 1)
        self.event_window = max(int(args.assembly_event_window), 1)
        self.seed_top_k = max(int(args.assembly_seed_top_k), 1)
        self.recency_decay = float(np.clip(args.assembly_recency_decay, 0.0, 1.0))
        self.locality_decay = float(np.clip(args.assembly_locality_decay, 0.0, 1.0))
        self.feature_dim = self.state_dim * (1 + self.hops)
        rng = np.random.default_rng(args.seed + 7919)
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
        self.unigram_counts = np.ones(self.vocab_size, dtype=np.float32)
        self.output_bias = np.full(self.vocab_size, -math.log(max(self.vocab_size, 1)), dtype=np.float32)
        self.last_seed_counts: list[int] = []
        self.last_event_feature = np.zeros(self.state_dim * self.hops, dtype=np.float32)

    def recurrent_state(self, tokens: list[int]) -> np.ndarray:
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
        prefix_set = set(prefix)
        seen: set[int] = set()
        seeds: list[int] = []
        for token in reversed(query):
            token = int(token)
            if token not in prefix_set or token in seen:
                continue
            seen.add(token)
            seeds.append(token)
            if len(seeds) >= self.seed_top_k:
                break
        return list(reversed(seeds))

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

    def feature(self, context: list[int] | np.ndarray) -> np.ndarray:
        tokens = [int(token) for token in list(context)[-self.max_order :]]
        prefix, query = self.prefix_and_query(tokens)
        seeds = self.initial_query_seeds(prefix, query)
        excluded = set(seeds)
        pieces = [self.recurrent_state(tokens)]
        seed_counts: list[int] = []
        for _ in range(self.hops):
            seed_counts.append(len(seeds))
            state = self.local_event_state(prefix, seeds)
            pieces.append(state)
            new_seeds = self.select_seeds(prefix, state, excluded)
            excluded.update(new_seeds)
            seeds = new_seeds
        self.last_seed_counts = seed_counts
        self.last_event_feature = phase.normalize_vector(
            np.concatenate(pieces[1:]).astype(np.float32)
        )
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

    def scores(self, context: list[int] | np.ndarray) -> np.ndarray:
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

    def update(self, context: list[int] | np.ndarray, target: int) -> None:
        target = int(target)
        feature = self.feature(context)
        scores = self.scores_from_feature(feature)
        target_score = float(scores[target])
        adjusted = scores.astype(np.float32, copy=True)
        adjusted[target] = -np.inf
        wrong = int(np.argmax(adjusted))
        self.update_target_slot(target, feature)
        if self.wrong_lr > 0.0 and float(adjusted[wrong]) + self.margin > target_score:
            self.update_wrong_slot(wrong, feature)
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
            + self.unigram_counts.nbytes
            + self.output_bias.nbytes
        )

    def active_contexts(self) -> int:
        return int(np.count_nonzero(self.prototype_counts))

    def config(self) -> dict[str, Any]:
        return {
            "assembly_hops": self.hops,
            "assembly_event_window": self.event_window,
            "assembly_seed_top_k": self.seed_top_k,
            "assembly_recency_decay": self.recency_decay,
            "assembly_locality_decay": self.locality_decay,
        }


class EventCleanupAssemblyMemory(QueryEventAssemblyMemory):
    """
    Event assembly plus an event-only local WTA cleanup readout.

    The cleanup branch learns prototypes only over event-hop states.  At scoring
    time it adds a candidate score and can inhibit non-winners among cleanup
    candidates.  Updates are local target/wrong-winner updates, not BP.
    """

    def __init__(self, args: argparse.Namespace, vocab_size: int) -> None:
        super().__init__(args, vocab_size)
        self.cleanup_dim = self.state_dim * self.hops
        self.cleanup_slots = int(args.cleanup_slots) if int(args.cleanup_slots) > 0 else self.slots
        self.cleanup_lr = float(np.clip(args.cleanup_lr, 0.0, 1.0))
        self.cleanup_wrong_lr = float(np.clip(args.cleanup_wrong_lr, 0.0, 1.0))
        self.cleanup_score_scale = float(args.cleanup_score_scale)
        self.cleanup_top_k = max(int(args.cleanup_top_k), 0)
        self.cleanup_inhibit = float(max(args.cleanup_inhibit, 0.0))
        self.cleanup_prototypes = np.zeros(
            (self.vocab_size, self.cleanup_slots, self.cleanup_dim),
            dtype=np.float32,
        )
        self.cleanup_counts = np.zeros((self.vocab_size, self.cleanup_slots), dtype=np.float32)

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

    def scores(self, context: list[int] | np.ndarray) -> np.ndarray:
        feature = self.feature(context)
        return self.combined_scores(feature, self.last_event_feature)

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

    def update(self, context: list[int] | np.ndarray, target: int) -> None:
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
        return int(super().state_bytes() + self.cleanup_prototypes.nbytes + self.cleanup_counts.nbytes)

    def active_contexts(self) -> int:
        return int(super().active_contexts() + np.count_nonzero(self.cleanup_counts))

    def config(self) -> dict[str, Any]:
        cfg = super().config()
        cfg.update(
            {
                "cleanup_slots": self.cleanup_slots,
                "cleanup_lr": self.cleanup_lr,
                "cleanup_wrong_lr": self.cleanup_wrong_lr,
                "cleanup_score_scale": self.cleanup_score_scale,
                "cleanup_top_k": self.cleanup_top_k,
                "cleanup_inhibit": self.cleanup_inhibit,
            }
        )
        return cfg


def build_memory(args: argparse.Namespace, vocab_size: int, variant: str) -> Any:
    if variant in ARBITRATED_VARIANTS:
        return LocalSpanArbitrationMemory(args, vocab_size)
    if variant in EVENT_ASSEMBLY_VARIANTS:
        return QueryEventAssemblyMemory(args, vocab_size)
    if variant in EVENT_CLEANUP_VARIANTS:
        return EventCleanupAssemblyMemory(args, vocab_size)
    return build_state_memory(args, vocab_size, variant)


def gate_snapshot(memory: Any) -> dict[str, float] | None:
    if not hasattr(memory, "last_gate_weight"):
        return None
    return {
        "gate_weight": float(memory.last_gate_weight),
        "base_margin": float(memory.last_base_margin),
        "span_margin": float(memory.last_span_margin),
    }


def score_answer(memory: Any, prompt_ids: list[int], target: int, temperature: float) -> tuple[float, int, float]:
    context = np.array(prompt_ids[-int(memory.max_order) :], dtype=np.int64)
    return softmax_loss_and_pred(memory.scores(context), int(target), temperature)


def train_answer(memory: Any, prompt_ids: list[int], target: int, temperature: float) -> dict[str, Any]:
    loss, pred, _ = score_answer(memory, prompt_ids, target, temperature)
    context = np.array(prompt_ids[-int(memory.max_order) :], dtype=np.int64)
    memory.update(context, int(target))
    metrics = {
        "loss": loss,
        "correct": int(pred == int(target)),
        "total": 1,
    }
    snapshot = gate_snapshot(memory)
    if snapshot is not None:
        metrics.update(snapshot)
    return metrics


def evaluate(
    memory: Any,
    rows: list[dict[str, Any]],
    token_to_id: dict[str, int],
    id_to_token: list[str],
    split: str,
    method: str,
    temperature: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loss_sum = 0.0
    correct = 0
    total = 0
    gate_count = 0
    gate_weight_sum = 0.0
    gate_hard_count = 0
    base_margin_sum = 0.0
    span_margin_sum = 0.0
    preds: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt_ids = encode(row["prompt_tokens"], token_to_id)
        target = int(token_to_id[row["answer"]])
        loss, pred, target_prob = score_answer(memory, prompt_ids, target, temperature)
        snapshot = gate_snapshot(memory)
        loss_sum += loss
        correct += int(pred == target)
        total += 1
        if snapshot is not None:
            gate_count += 1
            gate_weight_sum += snapshot["gate_weight"]
            gate_hard_count += int(snapshot["gate_weight"] >= 0.5)
            base_margin_sum += snapshot["base_margin"]
            span_margin_sum += snapshot["span_margin"]
        if idx < 50:
            pred_row = {
                "method": method,
                "split": split,
                "example_index": idx,
                "carrier": row["carrier"],
                "object": row["object"],
                "target_answer": row["answer"],
                "prediction": id_to_token[pred],
                "target_prob": target_prob,
                "correct": int(pred == target),
                "prompt": " ".join(row["prompt_tokens"]),
            }
            if snapshot is not None:
                pred_row.update(snapshot)
            preds.append(pred_row)
    summary = summarize_loss(loss_sum, correct, total)
    row = {
        "method": method,
        "split": split,
        "examples": len(rows),
        "answer_accuracy": summary["accuracy"],
        "answer_loss": summary["loss"],
        "answer_ppl": summary["ppl"],
        "state_bytes": memory.state_bytes(),
        "active_contexts": memory.active_contexts(),
        "stores_raw_text": False,
        "task_format": "synthetic_object_carry_next_token",
    }
    if gate_count > 0:
        row.update(
            {
                "gate_rate": gate_hard_count / gate_count,
                "mean_gate_weight": gate_weight_sum / gate_count,
                "mean_base_margin": base_margin_sum / gate_count,
                "mean_span_margin": span_margin_sum / gate_count,
            }
        )
    return row, preds


def run_variant(
    args: argparse.Namespace,
    variant: str,
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    id_to_token: list[str],
    token_to_id: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    memory = build_memory(args, len(id_to_token), variant)
    start = time.perf_counter()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_gate_count = 0
    train_gate_weight_sum = 0.0
    train_gate_hard_count = 0
    train_base_margin_sum = 0.0
    train_span_margin_sum = 0.0
    for _ in range(max(args.train_epochs, 1)):
        for row in train_rows:
            prompt_ids = encode(row["prompt_tokens"], token_to_id)
            target = int(token_to_id[row["answer"]])
            metrics = train_answer(memory, prompt_ids, target, args.temperature)
            train_loss += metrics["loss"]
            train_correct += metrics["correct"]
            train_total += metrics["total"]
            if "gate_weight" in metrics:
                train_gate_count += 1
                train_gate_weight_sum += float(metrics["gate_weight"])
                train_gate_hard_count += int(float(metrics["gate_weight"]) >= 0.5)
                train_base_margin_sum += float(metrics["base_margin"])
                train_span_margin_sum += float(metrics["span_margin"])
    train_summary = summarize_loss(train_loss, train_correct, train_total)
    train_row = {
        "method": variant,
        "split": "train_online",
        "examples": len(train_rows) * max(args.train_epochs, 1),
        "answer_accuracy": train_summary["accuracy"],
        "answer_loss": train_summary["loss"],
        "answer_ppl": train_summary["ppl"],
        "state_bytes": memory.state_bytes(),
        "active_contexts": memory.active_contexts(),
        "stores_raw_text": False,
        "task_format": "synthetic_object_carry_next_token",
    }
    if train_gate_count > 0:
        train_row.update(
            {
                "gate_rate": train_gate_hard_count / train_gate_count,
                "mean_gate_weight": train_gate_weight_sum / train_gate_count,
                "mean_base_margin": train_base_margin_sum / train_gate_count,
                "mean_span_margin": train_span_margin_sum / train_gate_count,
            }
        )
    rows = [train_row]
    prediction_rows: list[dict[str, Any]] = []
    for split, split_rows in [("train_post", train_rows), ("validation", valid_rows), ("test", test_rows)]:
        summary, preds = evaluate(memory, split_rows, token_to_id, id_to_token, split, variant, args.temperature)
        rows.append(summary)
        prediction_rows.extend(preds)
    cfg = {
        "variant": variant,
        "wall_seconds": time.perf_counter() - start,
        "state_bytes": memory.state_bytes(),
    }
    if hasattr(memory, "config"):
        cfg.update(memory.config())
    return rows, prediction_rows, cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "synthetic_object_carry_token")
    parser.add_argument("--methods", nargs="+", default=["baseline", "span", "event_cell", "span_event_cell"])
    parser.add_argument("--train-examples", type=int, default=1000)
    parser.add_argument("--valid-examples", type=int, default=200)
    parser.add_argument("--test-examples", type=int, default=1000)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--num-persons", type=int, default=6)
    parser.add_argument("--num-objects", type=int, default=6)
    parser.add_argument("--num-locations", type=int, default=8)
    parser.add_argument("--carrier-moves", type=int, default=2)
    parser.add_argument("--extra-distractors", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=96)
    parser.add_argument("--state-order", type=int, default=96)
    parser.add_argument("--state-decay", type=float, default=0.90)
    parser.add_argument("--micro-slots", type=int, default=32)
    parser.add_argument("--micro-lr", type=float, default=0.35)
    parser.add_argument("--micro-wrong-lr", type=float, default=0.02)
    parser.add_argument("--micro-score-scale", type=float, default=8.0)
    parser.add_argument("--micro-margin", type=float, default=0.0)
    parser.add_argument("--bias-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--binding-window", type=int, default=12)
    parser.add_argument("--span-binding-hops", type=int, default=2)
    parser.add_argument("--binding-query-order", type=int, default=8)
    parser.add_argument("--binding-focus-k", type=int, default=2)
    parser.add_argument("--binding-decay", type=float, default=0.95)
    parser.add_argument("--binding-span-window", type=int, default=5)
    parser.add_argument("--binding-span-top-k", type=int, default=4)
    parser.add_argument("--binding-span-decay", type=float, default=0.95)
    parser.add_argument(
        "--arbitration-gate-mode",
        choices=["hard_low_margin", "soft_low_margin", "soft_span_advantage"],
        default="soft_low_margin",
    )
    parser.add_argument("--arbitration-margin-threshold", type=float, default=0.5)
    parser.add_argument("--arbitration-temperature", type=float, default=0.25)
    parser.add_argument("--arbitration-span-gain", type=float, default=1.0)
    parser.add_argument("--assembly-hops", type=int, default=2)
    parser.add_argument("--assembly-event-window", type=int, default=2)
    parser.add_argument("--assembly-seed-top-k", type=int, default=3)
    parser.add_argument("--assembly-recency-decay", type=float, default=0.95)
    parser.add_argument("--assembly-locality-decay", type=float, default=0.90)
    parser.add_argument("--cleanup-slots", type=int, default=0)
    parser.add_argument("--cleanup-lr", type=float, default=0.35)
    parser.add_argument("--cleanup-wrong-lr", type=float, default=0.02)
    parser.add_argument("--cleanup-score-scale", type=float, default=3.0)
    parser.add_argument("--cleanup-top-k", type=int, default=4)
    parser.add_argument("--cleanup-inhibit", type=float, default=0.0)
    parser.add_argument("--event-cell-count", type=int, default=48)
    parser.add_argument("--event-cell-window", type=int, default=3)
    parser.add_argument("--event-cell-top-k", type=int, default=8)
    parser.add_argument("--event-cell-lr", type=float, default=0.08)
    parser.add_argument("--event-cell-credit-lr", type=float, default=0.05)
    parser.add_argument("--event-cell-neg-lr", type=float, default=0.03)
    parser.add_argument("--event-cell-query-weight", type=float, default=1.0)
    parser.add_argument("--event-cell-recency-decay", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    id_to_token, token_to_id = build_vocab(args.num_persons, args.num_objects, args.num_locations)
    train_rows = make_dataset(args, "train", args.train_examples, 0)
    valid_rows = make_dataset(args, "validation", args.valid_examples, 100_000)
    test_rows = make_dataset(args, "test", args.test_examples, 200_000)

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    variant_cfg: list[dict[str, Any]] = []
    for variant in args.methods:
        rows, preds, cfg = run_variant(args, variant, train_rows, valid_rows, test_rows, id_to_token, token_to_id)
        summary_rows.extend(rows)
        prediction_rows.extend(preds)
        variant_cfg.append(cfg)

    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "predictions_sample.csv", prediction_rows)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "vocab": id_to_token,
                "variants": variant_cfg,
                "model_stores_raw_text": False,
                "note": "Synthetic object-carry QA is evaluated as next-token prediction over the full toy vocabulary.",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
