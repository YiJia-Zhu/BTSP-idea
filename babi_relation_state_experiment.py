#!/usr/bin/env python3
"""
Pure no-BP relation-state experiments for bAbI QA18 and QA19.

QA18 tests size/containment transitive reasoning.  QA19 tests two-hop path
finding.  The main models use fixed random entity codes plus local
delta-Hebbian writes into relation matrices.  They do not store raw examples,
use BP/BPTT, or load any pretrained model.

Regex parsers are used only as the first local sensory front-end, matching the
earlier R137 stage before learned front-ends and delayed QA-credit are added.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from babi_no_bp_qa_experiment import (
    DEFAULT_DATA_DIR,
    HashedLookupBaseline,
    MajorityBaseline,
    PhaseDendriticQALearner,
    PhaseQAConfig,
    RawRetrievalBaseline,
    build_answer_vocab,
    evaluate_method,
    normalize_vector,
    read_jsonl,
    stable_hash_int,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent

DIRECTIONS = ("north", "south", "east", "west")
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
SIZE_STATEMENT_TYPES = ("fit_inside", "bigger_than")
SIZE_STATEMENT_TO_IDX = {name: idx for idx, name in enumerate(SIZE_STATEMENT_TYPES)}
SIZE_QUERY_TYPES = ("fit_in", "bigger_than")
SIZE_QUERY_TO_IDX = {name: idx for idx, name in enumerate(SIZE_QUERY_TYPES)}
DIRECTION_TO_IDX = {name: idx for idx, name in enumerate(DIRECTIONS)}


def clean_phrase(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_size_statement_detail(sentence: str) -> dict[str, str] | None:
    text = clean_phrase(sentence)
    fit = re.fullmatch(r"the (.+?) fits inside the (.+)", text)
    if fit:
        left = clean_phrase(fit.group(1))
        right = clean_phrase(fit.group(2))
        return {
            "event": "fit_inside",
            "left": left,
            "right": right,
            "smaller": left,
            "larger": right,
        }
    bigger = re.fullmatch(r"the (.+?) is bigger than the (.+)", text)
    if bigger:
        left = clean_phrase(bigger.group(1))
        right = clean_phrase(bigger.group(2))
        return {
            "event": "bigger_than",
            "left": left,
            "right": right,
            "smaller": right,
            "larger": left,
        }
    return None


def parse_size_statement(sentence: str) -> tuple[str, str] | None:
    """Return (smaller, larger) if a size relation is present."""
    detail = parse_size_statement_detail(sentence)
    if detail is None:
        return None
    return detail["smaller"], detail["larger"]


def parse_size_query_detail(question: str) -> dict[str, str] | None:
    text = clean_phrase(question)
    fit = re.fullmatch(r"does the (.+?) fit in the (.+)", text)
    if fit:
        left = clean_phrase(fit.group(1))
        right = clean_phrase(fit.group(2))
        return {
            "query": "fit_in",
            "left": left,
            "right": right,
            "smaller": left,
            "larger": right,
        }
    bigger = re.fullmatch(r"is the (.+?) bigger than the (.+)", text)
    if bigger:
        left = clean_phrase(bigger.group(1))
        right = clean_phrase(bigger.group(2))
        return {
            "query": "bigger_than",
            "left": left,
            "right": right,
            "smaller": right,
            "larger": left,
        }
    return None


def parse_size_query(question: str) -> tuple[str, str] | None:
    """Return desired (smaller, larger) relation for a yes answer."""
    detail = parse_size_query_detail(question)
    if detail is None:
        return None
    return detail["smaller"], detail["larger"]


def parse_path_statement_detail(sentence: str) -> dict[str, str] | None:
    text = clean_phrase(sentence)
    match = re.fullmatch(r"the (.+?) is (north|south|east|west) of the (.+)", text)
    if not match:
        return None
    relative = clean_phrase(match.group(1))
    direction = match.group(2)
    anchor = clean_phrase(match.group(3))
    return {
        "source": anchor,
        "direction": direction,
        "target": relative,
        "relative": relative,
        "anchor": anchor,
    }


def parse_path_statement(sentence: str) -> tuple[str, str, str] | None:
    """Return (source, direction, target) where direction moves source->target."""
    detail = parse_path_statement_detail(sentence)
    if detail is None:
        return None
    return detail["source"], detail["direction"], detail["target"]


def parse_path_query_detail(question: str) -> dict[str, str] | None:
    text = clean_phrase(question)
    match = re.fullmatch(r"how do you go from the (.+?) to the (.+)", text)
    if not match:
        return None
    return {"source": clean_phrase(match.group(1)), "target": clean_phrase(match.group(2))}


def parse_path_query(question: str) -> tuple[str, str] | None:
    detail = parse_path_query_detail(question)
    if detail is None:
        return None
    return detail["source"], detail["target"]


def phrase_feature(
    words: list[str],
    token_code_fn: Any,
    role: str,
    dim: int,
) -> np.ndarray:
    if not words:
        return np.zeros(dim, dtype=np.float32)
    feature = np.zeros(dim, dtype=np.float32)
    for idx, word in enumerate(words):
        feature += 1.25 * token_code_fn(word, f"{role}:tok:{idx}")
        feature += 0.75 * token_code_fn(word, f"{role}:bag")
    return normalize_vector(feature)


def strip_initial_article(words: list[str]) -> list[str]:
    if words and words[0] in {"the", "a", "an"}:
        return words[1:]
    return words


def span_after_marker(words: list[str], marker: str) -> list[str]:
    if marker not in words:
        return []
    idx = words.index(marker) + 1
    while idx < len(words) and words[idx] in {"the", "a", "an"}:
        idx += 1
    return words[idx:]


def span_between(words: list[str], start_after: str | None, stop_words: set[str]) -> list[str]:
    start = 0
    if start_after is not None:
        if start_after not in words:
            return []
        start = words.index(start_after) + 1
    while start < len(words) and words[start] in {"the", "a", "an"}:
        start += 1
    end = start
    while end < len(words) and words[end] not in stop_words:
        end += 1
    return words[start:end]


def stable_code(token: str, role: str, dim: int, seed: int) -> np.ndarray:
    code_seed = stable_hash_int(f"{seed}:relation-state:{role}:{token}", bits=64)
    rng = np.random.default_rng(code_seed)
    return normalize_vector(rng.normal(0.0, 1.0, dim).astype(np.float32))


def associate(matrix: np.ndarray, key: np.ndarray, value: np.ndarray, lr: float) -> None:
    pred = matrix @ key
    matrix += lr * np.outer(value - pred, key).astype(np.float32)


def recurrent_path_score(
    matrix: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    max_hops: int,
    hop_decay: float,
) -> float:
    value = start.astype(np.float32, copy=True)
    best = -1.0
    decay = 1.0
    for _ in range(max_hops):
        value = matrix @ value
        norm = float(np.linalg.norm(value))
        if norm <= 1e-8:
            break
        value = (value / norm).astype(np.float32)
        best = max(best, decay * float(goal @ value))
        decay *= hop_decay
    return best


@dataclass
class RelationStateConfig:
    dim: int = 96
    lr: float = 1.0
    score_scale: float = 8.0
    max_hops: int = 4
    hop_decay: float = 0.95
    seed: int = 0


@dataclass
class LocalDetectorConfig:
    dim: int = 64
    lr: float = 0.08
    epochs: int = 3
    score_scale: float = 6.0
    seed: int = 0


class LocalPrototypeDetector:
    def __init__(self, cfg: LocalDetectorConfig, name: str) -> None:
        self.cfg = cfg
        self.name = name
        self.rng = np.random.default_rng(cfg.seed)
        self._token_code_cache: dict[tuple[str, str], np.ndarray] = {}

    def token_code(self, token: str, slot: str) -> np.ndarray:
        token = clean_phrase(token)
        cache_key = (token, slot)
        cached = self._token_code_cache.get(cache_key)
        if cached is not None:
            return cached
        code_seed = stable_hash_int(f"{self.cfg.seed}:{self.name}:{slot}:{token}", bits=64)
        rng = np.random.default_rng(code_seed)
        code = normalize_vector(rng.normal(0.0, 1.0, self.cfg.dim).astype(np.float32))
        self._token_code_cache[cache_key] = code
        return code

    def text_feature(self, text: str, role: str) -> np.ndarray:
        words = clean_phrase(text).split()
        if not words:
            return np.zeros(self.cfg.dim, dtype=np.float32)
        feature = np.zeros(self.cfg.dim, dtype=np.float32)
        for idx, word in enumerate(words):
            feature += self.token_code(word, f"{role}:tok:{idx}")
            feature += 0.5 * self.token_code(word, f"{role}:bag")
        return normalize_vector(feature)

    def update_prototype(
        self,
        prototypes: dict[str, np.ndarray],
        counts: Counter[str],
        label: str | None,
        feature: np.ndarray,
    ) -> None:
        if label is None:
            return
        label = clean_phrase(label)
        counts[label] += 1
        if label not in prototypes:
            prototypes[label] = feature.copy()
            return
        eta = 1.0 / float(counts[label])
        prototypes[label] = normalize_vector((1.0 - eta) * prototypes[label] + eta * feature)

    def decode_slot(self, feature: np.ndarray, prototypes: dict[str, np.ndarray]) -> tuple[str | None, float]:
        if not prototypes:
            return None, 0.0
        best_label: str | None = None
        best_score = -1e9
        for label, proto in prototypes.items():
            score = float(proto @ feature)
            if score > best_score:
                best_label = label
                best_score = score
        return best_label, best_score


class LearnedSizeStatementDetector(LocalPrototypeDetector):
    def __init__(self, cfg: LocalDetectorConfig) -> None:
        super().__init__(cfg, "size-statement")
        self.event_weights = np.zeros((len(SIZE_STATEMENT_TYPES), cfg.dim), dtype=np.float32)
        self.left_prototypes: dict[str, np.ndarray] = {}
        self.right_prototypes: dict[str, np.ndarray] = {}
        self.left_counts: Counter[str] = Counter()
        self.right_counts: Counter[str] = Counter()

    def slot_feature(self, sentence: str, slot: str) -> np.ndarray:
        words = clean_phrase(sentence).split()
        if slot == "left":
            span = span_between(words, None, {"fits", "is"})
        else:
            if "inside" in words:
                span = span_after_marker(words, "inside")
            else:
                span = span_after_marker(words, "than")
        return phrase_feature(span, self.token_code, f"size-statement:{slot}", self.cfg.dim)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples: list[tuple[str, dict[str, str]]] = []
        for row in rows:
            for item in row["context"]:
                detail = parse_size_statement_detail(str(item["text"]))
                if detail is not None:
                    examples.append((str(item["text"]), detail))
        if not examples:
            return
        for _ in range(self.cfg.epochs):
            for idx in self.rng.permutation(len(examples)):
                sentence, detail = examples[int(idx)]
                feature = self.text_feature(sentence, "event")
                target = SIZE_STATEMENT_TO_IDX[detail["event"]]
                scores = self.event_weights @ feature
                pred = int(np.argmax(scores))
                if pred != target:
                    self.event_weights[target] += self.cfg.lr * feature
                    self.event_weights[pred] -= self.cfg.lr * feature
                    self.event_weights[target] = normalize_vector(self.event_weights[target])
                    self.event_weights[pred] = normalize_vector(self.event_weights[pred])
                self.update_prototype(
                    self.left_prototypes,
                    self.left_counts,
                    detail["left"],
                    self.slot_feature(sentence, "left"),
                )
                self.update_prototype(
                    self.right_prototypes,
                    self.right_counts,
                    detail["right"],
                    self.slot_feature(sentence, "right"),
                )

    def predict(self, sentence: str) -> dict[str, str | float | None]:
        feature = self.text_feature(sentence, "event")
        scores = self.cfg.score_scale * (self.event_weights @ feature)
        event = SIZE_STATEMENT_TYPES[int(np.argmax(scores))]
        left, left_conf = self.decode_slot(self.slot_feature(sentence, "left"), self.left_prototypes)
        right, right_conf = self.decode_slot(self.slot_feature(sentence, "right"), self.right_prototypes)
        if event == "fit_inside":
            smaller, larger = left, right
        else:
            smaller, larger = right, left
        return {
            "event": event,
            "left": left,
            "right": right,
            "smaller": smaller,
            "larger": larger,
            "left_confidence": left_conf,
            "right_confidence": right_conf,
        }

    def metrics(self, rows: list[dict[str, Any]]) -> dict[str, float | int]:
        total = event_correct = left_total = left_correct = right_total = right_correct = 0
        for row in rows:
            for item in row["context"]:
                detail = parse_size_statement_detail(str(item["text"]))
                if detail is None:
                    continue
                pred = self.predict(str(item["text"]))
                total += 1
                event_correct += int(pred["event"] == detail["event"])
                left_total += 1
                right_total += 1
                left_correct += int(pred["left"] == detail["left"])
                right_correct += int(pred["right"] == detail["right"])
        return {
            "examples": total,
            "event_accuracy": event_correct / max(total, 1),
            "left_accuracy": left_correct / max(left_total, 1),
            "right_accuracy": right_correct / max(right_total, 1),
            "state_bytes": self.state_bytes(),
        }

    def state_bytes(self) -> int:
        state = {
            "cfg": asdict(self.cfg),
            "event_weights": self.event_weights,
            "left_prototypes": self.left_prototypes,
            "right_prototypes": self.right_prototypes,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


class LearnedSizeQueryDetector(LocalPrototypeDetector):
    def __init__(self, cfg: LocalDetectorConfig) -> None:
        super().__init__(cfg, "size-query")
        self.query_weights = np.zeros((len(SIZE_QUERY_TYPES), cfg.dim), dtype=np.float32)
        self.left_prototypes: dict[str, np.ndarray] = {}
        self.right_prototypes: dict[str, np.ndarray] = {}
        self.left_counts: Counter[str] = Counter()
        self.right_counts: Counter[str] = Counter()

    def slot_feature(self, question: str, slot: str) -> np.ndarray:
        words = clean_phrase(question).split()
        if slot == "left":
            span = span_between(words, None, {"fit", "bigger"})
            if span and span[0] in {"does", "is"}:
                span = strip_initial_article(span[1:])
        else:
            if "in" in words:
                span = span_after_marker(words, "in")
            else:
                span = span_after_marker(words, "than")
        return phrase_feature(span, self.token_code, f"size-query:{slot}", self.cfg.dim)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples = [(str(row["question"]), parse_size_query_detail(str(row["question"]))) for row in rows]
        examples = [(q, d) for q, d in examples if d is not None]
        if not examples:
            return
        for _ in range(self.cfg.epochs):
            for idx in self.rng.permutation(len(examples)):
                question, detail = examples[int(idx)]
                feature = self.text_feature(question, "query")
                target = SIZE_QUERY_TO_IDX[detail["query"]]
                scores = self.query_weights @ feature
                pred = int(np.argmax(scores))
                if pred != target:
                    self.query_weights[target] += self.cfg.lr * feature
                    self.query_weights[pred] -= self.cfg.lr * feature
                    self.query_weights[target] = normalize_vector(self.query_weights[target])
                    self.query_weights[pred] = normalize_vector(self.query_weights[pred])
                self.update_prototype(
                    self.left_prototypes,
                    self.left_counts,
                    detail["left"],
                    self.slot_feature(question, "left"),
                )
                self.update_prototype(
                    self.right_prototypes,
                    self.right_counts,
                    detail["right"],
                    self.slot_feature(question, "right"),
                )

    def predict(self, question: str) -> dict[str, str | float | None]:
        feature = self.text_feature(question, "query")
        scores = self.cfg.score_scale * (self.query_weights @ feature)
        query = SIZE_QUERY_TYPES[int(np.argmax(scores))]
        left, left_conf = self.decode_slot(self.slot_feature(question, "left"), self.left_prototypes)
        right, right_conf = self.decode_slot(self.slot_feature(question, "right"), self.right_prototypes)
        if query == "fit_in":
            smaller, larger = left, right
        else:
            smaller, larger = right, left
        return {
            "query": query,
            "left": left,
            "right": right,
            "smaller": smaller,
            "larger": larger,
            "left_confidence": left_conf,
            "right_confidence": right_conf,
        }

    def metrics(self, rows: list[dict[str, Any]]) -> dict[str, float | int]:
        total = query_correct = left_correct = right_correct = 0
        for row in rows:
            detail = parse_size_query_detail(str(row["question"]))
            if detail is None:
                continue
            pred = self.predict(str(row["question"]))
            total += 1
            query_correct += int(pred["query"] == detail["query"])
            left_correct += int(pred["left"] == detail["left"])
            right_correct += int(pred["right"] == detail["right"])
        return {
            "examples": total,
            "query_accuracy": query_correct / max(total, 1),
            "left_accuracy": left_correct / max(total, 1),
            "right_accuracy": right_correct / max(total, 1),
            "state_bytes": self.state_bytes(),
        }

    def state_bytes(self) -> int:
        state = {
            "cfg": asdict(self.cfg),
            "query_weights": self.query_weights,
            "left_prototypes": self.left_prototypes,
            "right_prototypes": self.right_prototypes,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


class LearnedPathStatementDetector(LocalPrototypeDetector):
    def __init__(self, cfg: LocalDetectorConfig) -> None:
        super().__init__(cfg, "path-statement")
        self.direction_weights = np.zeros((len(DIRECTIONS), cfg.dim), dtype=np.float32)
        self.source_prototypes: dict[str, np.ndarray] = {}
        self.target_prototypes: dict[str, np.ndarray] = {}
        self.source_counts: Counter[str] = Counter()
        self.target_counts: Counter[str] = Counter()

    def slot_feature(self, sentence: str, slot: str) -> np.ndarray:
        words = clean_phrase(sentence).split()
        if slot == "target":
            span = span_between(words, None, {"is"})
        else:
            span = span_after_marker(words, "of")
        return phrase_feature(span, self.token_code, f"path-statement:{slot}", self.cfg.dim)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples: list[tuple[str, dict[str, str]]] = []
        for row in rows:
            for item in row["context"]:
                detail = parse_path_statement_detail(str(item["text"]))
                if detail is not None:
                    examples.append((str(item["text"]), detail))
        if not examples:
            return
        for _ in range(self.cfg.epochs):
            for idx in self.rng.permutation(len(examples)):
                sentence, detail = examples[int(idx)]
                feature = self.text_feature(sentence, "direction")
                target = DIRECTION_TO_IDX[detail["direction"]]
                scores = self.direction_weights @ feature
                pred = int(np.argmax(scores))
                if pred != target:
                    self.direction_weights[target] += self.cfg.lr * feature
                    self.direction_weights[pred] -= self.cfg.lr * feature
                    self.direction_weights[target] = normalize_vector(self.direction_weights[target])
                    self.direction_weights[pred] = normalize_vector(self.direction_weights[pred])
                self.update_prototype(
                    self.source_prototypes,
                    self.source_counts,
                    detail["source"],
                    self.slot_feature(sentence, "source"),
                )
                self.update_prototype(
                    self.target_prototypes,
                    self.target_counts,
                    detail["target"],
                    self.slot_feature(sentence, "target"),
                )

    def predict(self, sentence: str) -> dict[str, str | float | None]:
        feature = self.text_feature(sentence, "direction")
        scores = self.cfg.score_scale * (self.direction_weights @ feature)
        direction = DIRECTIONS[int(np.argmax(scores))]
        source, source_conf = self.decode_slot(self.slot_feature(sentence, "source"), self.source_prototypes)
        target, target_conf = self.decode_slot(self.slot_feature(sentence, "target"), self.target_prototypes)
        return {
            "source": source,
            "direction": direction,
            "target": target,
            "source_confidence": source_conf,
            "target_confidence": target_conf,
        }

    def metrics(self, rows: list[dict[str, Any]]) -> dict[str, float | int]:
        total = direction_correct = source_correct = target_correct = 0
        for row in rows:
            for item in row["context"]:
                detail = parse_path_statement_detail(str(item["text"]))
                if detail is None:
                    continue
                pred = self.predict(str(item["text"]))
                total += 1
                direction_correct += int(pred["direction"] == detail["direction"])
                source_correct += int(pred["source"] == detail["source"])
                target_correct += int(pred["target"] == detail["target"])
        return {
            "examples": total,
            "direction_accuracy": direction_correct / max(total, 1),
            "source_accuracy": source_correct / max(total, 1),
            "target_accuracy": target_correct / max(total, 1),
            "state_bytes": self.state_bytes(),
        }

    def state_bytes(self) -> int:
        state = {
            "cfg": asdict(self.cfg),
            "direction_weights": self.direction_weights,
            "source_prototypes": self.source_prototypes,
            "target_prototypes": self.target_prototypes,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


class LearnedPathQueryDetector(LocalPrototypeDetector):
    def __init__(self, cfg: LocalDetectorConfig) -> None:
        super().__init__(cfg, "path-query")
        self.source_prototypes: dict[str, np.ndarray] = {}
        self.target_prototypes: dict[str, np.ndarray] = {}
        self.source_counts: Counter[str] = Counter()
        self.target_counts: Counter[str] = Counter()

    def slot_feature(self, question: str, slot: str) -> np.ndarray:
        words = clean_phrase(question).split()
        if slot == "source":
            span = span_between(words, "from", {"to"})
        else:
            span = span_after_marker(words, "to")
        return phrase_feature(span, self.token_code, f"path-query:{slot}", self.cfg.dim)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples = [(str(row["question"]), parse_path_query_detail(str(row["question"]))) for row in rows]
        examples = [(q, d) for q, d in examples if d is not None]
        for _ in range(self.cfg.epochs):
            for idx in self.rng.permutation(len(examples)):
                question, detail = examples[int(idx)]
                self.update_prototype(
                    self.source_prototypes,
                    self.source_counts,
                    detail["source"],
                    self.slot_feature(question, "source"),
                )
                self.update_prototype(
                    self.target_prototypes,
                    self.target_counts,
                    detail["target"],
                    self.slot_feature(question, "target"),
                )

    def predict(self, question: str) -> dict[str, str | float | None]:
        source, source_conf = self.decode_slot(self.slot_feature(question, "source"), self.source_prototypes)
        target, target_conf = self.decode_slot(self.slot_feature(question, "target"), self.target_prototypes)
        return {
            "source": source,
            "target": target,
            "source_confidence": source_conf,
            "target_confidence": target_conf,
        }

    def metrics(self, rows: list[dict[str, Any]]) -> dict[str, float | int]:
        total = source_correct = target_correct = 0
        for row in rows:
            detail = parse_path_query_detail(str(row["question"]))
            if detail is None:
                continue
            pred = self.predict(str(row["question"]))
            total += 1
            source_correct += int(pred["source"] == detail["source"])
            target_correct += int(pred["target"] == detail["target"])
        return {
            "examples": total,
            "source_accuracy": source_correct / max(total, 1),
            "target_accuracy": target_correct / max(total, 1),
            "state_bytes": self.state_bytes(),
        }

    def state_bytes(self) -> int:
        state = {
            "cfg": asdict(self.cfg),
            "source_prototypes": self.source_prototypes,
            "target_prototypes": self.target_prototypes,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


class SizeRelationStateQALearner:
    """
    Local matrix model for QA18.

    One matrix maps smaller-object codes to larger-object codes.  Transitive
    reasoning is performed by repeated local retrieval through the same matrix.
    """

    def __init__(
        self,
        answer_to_idx: dict[str, int],
        fallback: MajorityBaseline,
        cfg: RelationStateConfig,
        statement_detector: LearnedSizeStatementDetector | None = None,
        query_detector: LearnedSizeQueryDetector | None = None,
        statement_mode: str = "regex",
        query_mode: str = "regex",
    ) -> None:
        self.answer_to_idx = answer_to_idx
        self.fallback = fallback
        self.cfg = cfg
        self.statement_detector = statement_detector
        self.query_detector = query_detector
        self.statement_mode = statement_mode
        self.query_mode = query_mode
        self._code_cache: dict[str, np.ndarray] = {}

    def fit(self, rows: list[dict[str, Any]]) -> None:
        if self.statement_detector is not None:
            self.statement_detector.fit(rows)
        if self.query_detector is not None:
            self.query_detector.fit(rows)

    def code(self, token: str) -> np.ndarray:
        token = clean_phrase(token)
        cached = self._code_cache.get(token)
        if cached is not None:
            return cached
        code = stable_code(token, "size-entity", self.cfg.dim, self.cfg.seed)
        self._code_cache[token] = code
        return code

    def new_state(self) -> np.ndarray:
        return np.zeros((self.cfg.dim, self.cfg.dim), dtype=np.float32)

    def observe(self, matrix: np.ndarray, sentence: str) -> None:
        parsed = self.detect_statement(sentence)
        if parsed is None:
            return
        smaller, larger = parsed
        associate(matrix, self.code(smaller), self.code(larger), self.cfg.lr)

    def detect_statement(self, sentence: str) -> tuple[str, str] | None:
        if self.statement_detector is None or self.statement_mode == "regex":
            return parse_size_statement(sentence)
        pred = self.statement_detector.predict(sentence)
        smaller = pred.get("smaller")
        larger = pred.get("larger")
        if isinstance(smaller, str) and isinstance(larger, str):
            return smaller, larger
        if self.statement_mode == "hybrid":
            return parse_size_statement(sentence)
        return None

    def read_context(self, row: dict[str, Any]) -> np.ndarray:
        matrix = self.new_state()
        for item in row["context"]:
            self.observe(matrix, str(item["text"]))
        return matrix

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        query = self.detect_query(str(row["question"]))
        if query is None or "yes" not in self.answer_to_idx or "no" not in self.answer_to_idx:
            return self.fallback.scores(row)
        smaller, larger = query
        matrix = self.read_context(row)
        yes = recurrent_path_score(
            matrix,
            self.code(smaller),
            self.code(larger),
            self.cfg.max_hops,
            self.cfg.hop_decay,
        )
        no = recurrent_path_score(
            matrix,
            self.code(larger),
            self.code(smaller),
            self.cfg.max_hops,
            self.cfg.hop_decay,
        )
        if yes < -0.5 and no < -0.5:
            return self.fallback.scores(row)
        scores = np.full(len(self.answer_to_idx), -4.0, dtype=np.float32)
        scores[self.answer_to_idx["yes"]] = self.cfg.score_scale * yes
        scores[self.answer_to_idx["no"]] = self.cfg.score_scale * no
        return scores

    def detect_query(self, question: str) -> tuple[str, str] | None:
        if self.query_detector is None or self.query_mode == "regex":
            return parse_size_query(question)
        pred = self.query_detector.predict(question)
        smaller = pred.get("smaller")
        larger = pred.get("larger")
        if isinstance(smaller, str) and isinstance(larger, str):
            return smaller, larger
        if self.query_mode == "hybrid":
            return parse_size_query(question)
        return None

    def state_bytes(self) -> int:
        detector_bytes = 0
        if self.statement_detector is not None:
            detector_bytes += self.statement_detector.state_bytes()
        if self.query_detector is not None:
            detector_bytes += self.query_detector.state_bytes()
        return int(self.cfg.dim * self.cfg.dim * np.dtype(np.float32).itemsize) + detector_bytes


class PathRelationStateQALearner:
    """
    Local matrix model for QA19.

    Each direction owns a transition matrix.  Statements update the forward and
    reverse transition with local delta-Hebbian writes.  Candidate answers are
    scored by applying the direction sequence to the source code and comparing
    the result with the target code.
    """

    def __init__(
        self,
        answer_to_idx: dict[str, int],
        fallback: MajorityBaseline,
        cfg: RelationStateConfig,
        statement_detector: LearnedPathStatementDetector | None = None,
        query_detector: LearnedPathQueryDetector | None = None,
        statement_mode: str = "regex",
        query_mode: str = "regex",
    ) -> None:
        self.answer_to_idx = answer_to_idx
        self.fallback = fallback
        self.cfg = cfg
        self.statement_detector = statement_detector
        self.query_detector = query_detector
        self.statement_mode = statement_mode
        self.query_mode = query_mode
        self._code_cache: dict[str, np.ndarray] = {}

    def fit(self, rows: list[dict[str, Any]]) -> None:
        if self.statement_detector is not None:
            self.statement_detector.fit(rows)
        if self.query_detector is not None:
            self.query_detector.fit(rows)

    def code(self, token: str) -> np.ndarray:
        token = clean_phrase(token)
        cached = self._code_cache.get(token)
        if cached is not None:
            return cached
        code = stable_code(token, "path-place", self.cfg.dim, self.cfg.seed)
        self._code_cache[token] = code
        return code

    def new_state(self) -> dict[str, np.ndarray]:
        return {
            direction: np.zeros((self.cfg.dim, self.cfg.dim), dtype=np.float32)
            for direction in DIRECTIONS
        }

    def observe(self, matrices: dict[str, np.ndarray], sentence: str) -> None:
        parsed = self.detect_statement(sentence)
        if parsed is None:
            return
        source, direction, target = parsed
        associate(matrices[direction], self.code(source), self.code(target), self.cfg.lr)
        associate(matrices[OPPOSITE[direction]], self.code(target), self.code(source), self.cfg.lr)

    def detect_statement(self, sentence: str) -> tuple[str, str, str] | None:
        if self.statement_detector is None or self.statement_mode == "regex":
            return parse_path_statement(sentence)
        pred = self.statement_detector.predict(sentence)
        source = pred.get("source")
        direction = pred.get("direction")
        target = pred.get("target")
        if isinstance(source, str) and isinstance(direction, str) and isinstance(target, str):
            return source, direction, target
        if self.statement_mode == "hybrid":
            return parse_path_statement(sentence)
        return None

    def read_context(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        matrices = self.new_state()
        for item in row["context"]:
            self.observe(matrices, str(item["text"]))
        return matrices

    def sequence_score(
        self,
        matrices: dict[str, np.ndarray],
        source: str,
        target: str,
        directions: list[str],
    ) -> float:
        value = self.code(source)
        for direction in directions:
            if direction not in matrices:
                return -1.0
            value = matrices[direction] @ value
            norm = float(np.linalg.norm(value))
            if norm <= 1e-8:
                return -1.0
            value = (value / norm).astype(np.float32)
        return float(self.code(target) @ value)

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        query = self.detect_query(str(row["question"]))
        if query is None:
            return self.fallback.scores(row)
        source, target = query
        matrices = self.read_context(row)
        scores = np.full(len(self.answer_to_idx), -4.0, dtype=np.float32)
        any_valid = False
        for answer, idx in self.answer_to_idx.items():
            directions = answer.split()
            if not directions or any(direction not in DIRECTIONS for direction in directions):
                continue
            score = self.sequence_score(matrices, source, target, directions)
            scores[idx] = self.cfg.score_scale * score
            any_valid = True
        if not any_valid:
            return self.fallback.scores(row)
        return scores

    def detect_query(self, question: str) -> tuple[str, str] | None:
        if self.query_detector is None or self.query_mode == "regex":
            return parse_path_query(question)
        pred = self.query_detector.predict(question)
        source = pred.get("source")
        target = pred.get("target")
        if isinstance(source, str) and isinstance(target, str):
            return source, target
        if self.query_mode == "hybrid":
            return parse_path_query(question)
        return None

    def state_bytes(self) -> int:
        detector_bytes = 0
        if self.statement_detector is not None:
            detector_bytes += self.statement_detector.state_bytes()
        if self.query_detector is not None:
            detector_bytes += self.query_detector.state_bytes()
        return int(len(DIRECTIONS) * self.cfg.dim * self.cfg.dim * np.dtype(np.float32).itemsize) + detector_bytes


class SizeSymbolicGraphBaseline:
    """Diagnostic symbolic upper bound for QA18, not the main method."""

    def __init__(self, answer_to_idx: dict[str, int], fallback: MajorityBaseline) -> None:
        self.answer_to_idx = answer_to_idx
        self.fallback = fallback

    def fit(self, rows: list[dict[str, Any]]) -> None:
        del rows

    def path_exists(self, edges: dict[str, set[str]], start: str, goal: str) -> bool:
        frontier = [start]
        seen = {start}
        while frontier:
            node = frontier.pop(0)
            for nxt in edges.get(node, set()):
                if nxt == goal:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return False

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        query = parse_size_query(str(row["question"]))
        if query is None or "yes" not in self.answer_to_idx or "no" not in self.answer_to_idx:
            return self.fallback.scores(row)
        edges: dict[str, set[str]] = {}
        for item in row["context"]:
            parsed = parse_size_statement(str(item["text"]))
            if parsed is None:
                continue
            smaller, larger = parsed
            edges.setdefault(smaller, set()).add(larger)
        smaller, larger = query
        yes = self.path_exists(edges, smaller, larger)
        no = self.path_exists(edges, larger, smaller)
        scores = np.full(len(self.answer_to_idx), -4.0, dtype=np.float32)
        if yes or not no:
            scores[self.answer_to_idx["yes"]] = 4.0
        if no or not yes:
            scores[self.answer_to_idx["no"]] = 4.0
        return scores

    def state_bytes(self) -> int:
        return 0


class PathSymbolicGraphBaseline:
    """Diagnostic symbolic upper bound for QA19, not the main method."""

    def __init__(self, answer_to_idx: dict[str, int], fallback: MajorityBaseline) -> None:
        self.answer_to_idx = answer_to_idx
        self.fallback = fallback

    def fit(self, rows: list[dict[str, Any]]) -> None:
        del rows

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        query = parse_path_query(str(row["question"]))
        if query is None:
            return self.fallback.scores(row)
        source, target = query
        graph: dict[str, list[tuple[str, str]]] = {}
        for item in row["context"]:
            parsed = parse_path_statement(str(item["text"]))
            if parsed is None:
                continue
            src, direction, dst = parsed
            graph.setdefault(src, []).append((direction, dst))
            graph.setdefault(dst, []).append((OPPOSITE[direction], src))
        scores = np.full(len(self.answer_to_idx), -4.0, dtype=np.float32)
        for first_dir, mid in graph.get(source, []):
            for second_dir, dst in graph.get(mid, []):
                if dst != target:
                    continue
                answer = f"{first_dir} {second_dir}"
                if answer in self.answer_to_idx:
                    scores[self.answer_to_idx[answer]] = 4.0
        if float(np.max(scores)) <= -3.9:
            return self.fallback.scores(row)
        return scores

    def state_bytes(self) -> int:
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--configs", nargs="+", default=["en-qa18", "en-qa19"])
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "babi_relation_state")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--lookup-ngrams", type=int, default=2)
    parser.add_argument("--relation-dim", type=int, default=96)
    parser.add_argument("--relation-lr", type=float, default=1.0)
    parser.add_argument("--relation-score-scale", type=float, default=8.0)
    parser.add_argument("--relation-max-hops", type=int, default=4)
    parser.add_argument("--relation-hop-decay", type=float, default=0.95)
    parser.add_argument("--detector-dim", type=int, default=64)
    parser.add_argument("--detector-lr", type=float, default=0.08)
    parser.add_argument("--detector-epochs", type=int, default=3)
    parser.add_argument("--detector-score-scale", type=float, default=6.0)
    parser.add_argument("--phase-dim", type=int, default=64)
    parser.add_argument("--phase-lr", type=float, default=0.08)
    parser.add_argument("--phase-wrong-lr", type=float, default=0.03)
    parser.add_argument("--phase-epochs", type=int, default=8)
    parser.add_argument("--phase-score-scale", type=float, default=6.0)
    parser.add_argument("--phase-temperature", type=float, default=1.0)
    parser.add_argument("--branch-agreement", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_splits(data_dir: Path, config: str, train_limit: int, eval_limit: int) -> dict[str, list[dict[str, Any]]]:
    config_dir = data_dir / config
    return {
        "train": read_jsonl(config_dir / "train.jsonl", train_limit or None),
        "validation": read_jsonl(config_dir / "validation.jsonl", eval_limit or None),
        "test": read_jsonl(config_dir / "test.jsonl", eval_limit or None),
    }


def detector_metric_rows(
    config: str,
    splits: dict[str, list[dict[str, Any]]],
    detectors: list[tuple[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detector_name, detector in detectors:
        for split, split_rows in splits.items():
            metrics = detector.metrics(split_rows)
            rows.append(
                {
                    "config": config,
                    "detector": detector_name,
                    "split": split,
                    **metrics,
                }
            )
    return rows


def run_config(
    args: argparse.Namespace,
    config: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    splits = load_splits(args.data_dir, config, args.train_limit, args.eval_limit)
    answer_vocab = build_answer_vocab(splits["train"], splits["validation"], splits["test"])
    answer_to_idx = {answer: idx for idx, answer in enumerate(answer_vocab)}

    majority = MajorityBaseline(answer_to_idx)
    majority.fit(splits["train"])

    retrieval = RawRetrievalBaseline(answer_to_idx)
    retrieval.fit(splits["train"])

    lookup = HashedLookupBaseline(answer_to_idx, args.hash_bits, args.lookup_ngrams, args.seed + 17)
    lookup.fit(splits["train"])

    relation_cfg = RelationStateConfig(
        dim=args.relation_dim,
        lr=args.relation_lr,
        score_scale=args.relation_score_scale,
        max_hops=args.relation_max_hops,
        hop_decay=args.relation_hop_decay,
        seed=args.seed + 503,
    )
    phase_cfg = PhaseQAConfig(
        phase_dim=args.phase_dim,
        lr=args.phase_lr,
        wrong_lr=args.phase_wrong_lr,
        epochs=args.phase_epochs,
        score_scale=args.phase_score_scale,
        temperature=args.phase_temperature,
        branch_agreement=args.branch_agreement,
        seed=args.seed + 607,
    )
    phase = PhaseDendriticQALearner(answer_to_idx, phase_cfg)
    phase.fit(splits["train"])
    detector_cfg = LocalDetectorConfig(
        dim=args.detector_dim,
        lr=args.detector_lr,
        epochs=args.detector_epochs,
        score_scale=args.detector_score_scale,
        seed=args.seed + 701,
    )

    methods: list[tuple[str, Any, bool, str]] = [
        ("majority_no_memory", majority, False, "baseline"),
        ("raw_lexical_retrieval", retrieval, True, "diagnostic_raw_retrieval"),
        ("hashed_lookup_diagnostic", lookup, False, "diagnostic_statistical_lookup"),
        ("phase_dendritic_no_bp", phase, False, "pure_no_bp_neural_answer_classifier"),
    ]

    if config == "en-qa18":
        symbolic = SizeSymbolicGraphBaseline(answer_to_idx, majority)
        relation = SizeRelationStateQALearner(answer_to_idx, majority, relation_cfg)
        learned_statement = LearnedSizeStatementDetector(detector_cfg)
        learned_query = LearnedSizeQueryDetector(
            LocalDetectorConfig(
                dim=args.detector_dim,
                lr=args.detector_lr,
                epochs=args.detector_epochs,
                score_scale=args.detector_score_scale,
                seed=args.seed + 709,
            )
        )
        learned_relation = SizeRelationStateQALearner(
            answer_to_idx,
            majority,
            relation_cfg,
            statement_detector=learned_statement,
            query_detector=learned_query,
            statement_mode="learned",
            query_mode="learned",
        )
        methods.extend(
            [
                ("symbolic_size_graph_upper", symbolic, False, "diagnostic_symbolic_upper_bound"),
                ("size_relation_state_no_bp", relation, False, "pure_no_bp_relation_state"),
                (
                    "size_learned_relation_state_no_bp",
                    learned_relation,
                    False,
                    "pure_no_bp_relation_state_learned_frontend",
                ),
            ]
        )
        learned_detectors: list[tuple[str, Any]] = [
            ("size_statement", learned_statement),
            ("size_query", learned_query),
        ]
    elif config == "en-qa19":
        symbolic = PathSymbolicGraphBaseline(answer_to_idx, majority)
        relation = PathRelationStateQALearner(answer_to_idx, majority, relation_cfg)
        learned_statement = LearnedPathStatementDetector(detector_cfg)
        learned_query = LearnedPathQueryDetector(
            LocalDetectorConfig(
                dim=args.detector_dim,
                lr=args.detector_lr,
                epochs=args.detector_epochs,
                score_scale=args.detector_score_scale,
                seed=args.seed + 719,
            )
        )
        learned_relation = PathRelationStateQALearner(
            answer_to_idx,
            majority,
            relation_cfg,
            statement_detector=learned_statement,
            query_detector=learned_query,
            statement_mode="learned",
            query_mode="learned",
        )
        methods.extend(
            [
                ("symbolic_path_graph_upper", symbolic, False, "diagnostic_symbolic_upper_bound"),
                ("path_relation_state_no_bp", relation, False, "pure_no_bp_relation_state"),
                (
                    "path_learned_relation_state_no_bp",
                    learned_relation,
                    False,
                    "pure_no_bp_relation_state_learned_frontend",
                ),
            ]
        )
        learned_detectors = [
            ("path_statement", learned_statement),
            ("path_query", learned_query),
        ]
    else:
        learned_detectors = []

    all_summary: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for name, model, stores_raw, method_type in methods:
        if hasattr(model, "fit") and name.startswith(("symbolic_", "size_", "path_")):
            model.fit(splits["train"])
        summary, predictions = evaluate_method(
            name,
            model,
            splits,
            answer_to_idx,
            args.phase_temperature,
            stores_raw,
            method_type,
        )
        for row in summary:
            row["config"] = config
            row["answer_count"] = len(answer_vocab)
        for row in predictions:
            row["config"] = config
        all_summary.extend(summary)
        all_predictions.extend(predictions)
    all_detector_metrics = detector_metric_rows(config, splits, learned_detectors)
    return all_summary, all_predictions, all_detector_metrics


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_summary: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_detector_metrics: list[dict[str, Any]] = []
    for config in args.configs:
        summary, predictions, detector_metrics = run_config(args, config)
        all_summary.extend(summary)
        all_predictions.extend(predictions)
        all_detector_metrics.extend(detector_metrics)

    write_csv(args.out_dir / "summary.csv", all_summary)
    write_csv(args.out_dir / "predictions_sample.csv", all_predictions)
    write_csv(args.out_dir / "detector_metrics.csv", all_detector_metrics)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "relation_cfg": {
                    "dim": args.relation_dim,
                    "lr": args.relation_lr,
                    "score_scale": args.relation_score_scale,
                    "max_hops": args.relation_max_hops,
                    "hop_decay": args.relation_hop_decay,
                    "seed": args.seed + 503,
                },
                "phase_cfg": {
                    "phase_dim": args.phase_dim,
                    "lr": args.phase_lr,
                    "wrong_lr": args.phase_wrong_lr,
                    "epochs": args.phase_epochs,
                    "score_scale": args.phase_score_scale,
                    "temperature": args.phase_temperature,
                    "branch_agreement": args.branch_agreement,
                    "seed": args.seed + 607,
                },
                "detector_cfg": {
                    "dim": args.detector_dim,
                    "lr": args.detector_lr,
                    "epochs": args.detector_epochs,
                    "score_scale": args.detector_score_scale,
                    "statement_seed": args.seed + 701,
                },
            },
            f,
            indent=2,
            default=str,
            sort_keys=True,
        )

    print("Summary:")
    for row in all_summary:
        if row["split"] == "test":
            print(
                f"  {row['config']} {row['method']}: "
                f"test_acc={row['accuracy']:.3f} test_loss={row['loss']:.3f} "
                f"bytes={row['state_bytes']}"
            )
    print(f"wrote {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
