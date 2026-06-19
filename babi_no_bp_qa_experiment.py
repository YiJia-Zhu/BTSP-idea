#!/usr/bin/env python3
"""
Pure no-BP bAbI QA answer-selection experiment.

The first target is bAbI QA1.  The model reads a context/question pair and
selects one answer id.  Training uses only local target/wrong-winner updates:
no BP, no BPTT, no pretrained encoder, and no API.

Statistical lookup and symbolic trackers are reported as diagnostics/bounds,
not as final biomimetic methods.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "babi_qa_processed"


LOCATION_WORDS = {"bathroom", "bedroom", "garden", "hallway", "kitchen", "office"}
COLOR_WORDS = {"gray", "green", "white", "yellow"}
IRREGULAR_SINGULARS = {
    "cats": "cat",
    "frogs": "frog",
    "lions": "lion",
    "mice": "mouse",
    "rhinos": "rhino",
    "sheep": "sheep",
    "swans": "swan",
    "wolves": "wolf",
}
MOVE_VERBS = {
    "moved",
    "went",
    "journeyed",
    "travelled",
    "traveled",
    "returned",
}
PICKUP_VERBS = {"got", "grabbed", "took", "picked"}
DROP_VERBS = {"dropped", "discarded", "left", "put"}
NULL_OWNER = "__none__"
EVENT_TYPES = ["none", "move", "pickup", "drop"]
EVENT_TO_IDX = {event: idx for idx, event in enumerate(EVENT_TYPES)}
QUERY_TYPES = ["where_is", "where_before"]
QUERY_TO_IDX = {query: idx for idx, query in enumerate(QUERY_TYPES)}
ATTRIBUTE_EVENT_TYPES = ["none", "class_afraid", "entity_class", "entity_color"]
ATTRIBUTE_EVENT_TO_IDX = {event: idx for idx, event in enumerate(ATTRIBUTE_EVENT_TYPES)}
ATTRIBUTE_QUERY_TYPES = ["none", "afraid_of", "color"]
ATTRIBUTE_QUERY_TO_IDX = {query: idx for idx, query in enumerate(ATTRIBUTE_QUERY_TYPES)}


@dataclass
class PhaseQAConfig:
    phase_dim: int = 64
    lr: float = 0.08
    wrong_lr: float = 0.03
    epochs: int = 8
    score_scale: float = 6.0
    temperature: float = 1.0
    branch_agreement: float = 0.05
    seed: int = 0


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
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def row_context_text(row: dict[str, Any]) -> str:
    return " ".join(str(item["text"]) for item in row["context"])


def row_text(row: dict[str, Any]) -> str:
    return row_context_text(row) + " " + str(row["question"])


def token_set(row: dict[str, Any]) -> set[str]:
    return set(tokens(row_text(row)))


def stable_hash_int(text: str, bits: int = 64) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=bits // 8).digest()
    return int.from_bytes(digest, "little")


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = scores / max(float(temperature), 1e-6)
    z = z - float(np.max(z))
    exp_z = np.exp(z)
    return (exp_z / np.sum(exp_z)).astype(np.float32)


def evaluate_scores(
    rows: list[dict[str, Any]],
    answer_to_idx: dict[str, int],
    score_fn: Any,
    temperature: float,
) -> dict[str, Any]:
    losses: list[float] = []
    correct = 0
    predictions: list[dict[str, Any]] = []
    idx_to_answer = {idx: answer for answer, idx in answer_to_idx.items()}
    for idx, row in enumerate(rows):
        target = answer_to_idx[row["answer"]]
        scores = score_fn(row).astype(np.float32, copy=False)
        probs = softmax(scores, temperature)
        pred_idx = int(np.argmax(probs))
        pred = idx_to_answer[pred_idx]
        losses.append(-math.log(float(probs[target]) + 1e-9))
        ok = int(pred_idx == target)
        correct += ok
        if idx < 50:
            predictions.append(
                {
                    "example_index": idx,
                    "question": row["question"],
                    "target": row["answer"],
                    "prediction": pred,
                    "correct": ok,
                }
            )
    total = len(rows)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


class MajorityBaseline:
    def __init__(self, answer_to_idx: dict[str, int]) -> None:
        self.answer_to_idx = answer_to_idx
        self.counts = np.ones(len(answer_to_idx), dtype=np.float32)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.counts[self.answer_to_idx[row["answer"]]] += 1.0

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        del row
        probs = self.counts / float(np.sum(self.counts))
        return np.log(np.maximum(probs, 1e-9)).astype(np.float32)

    def state_bytes(self) -> int:
        return len(pickle.dumps({"counts": self.counts}, protocol=pickle.HIGHEST_PROTOCOL))


class RawRetrievalBaseline:
    """Raw-example retrieval baseline. Strong diagnostic, violates no-raw-data."""

    def __init__(self, answer_to_idx: dict[str, int]) -> None:
        self.answer_to_idx = answer_to_idx
        self.examples: list[tuple[str, set[str], int]] = []

    def fit(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.examples.append((row_text(row), token_set(row), self.answer_to_idx[row["answer"]]))

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        query = token_set(row)
        scores = np.zeros(len(self.answer_to_idx), dtype=np.float32)
        for _, example_tokens, answer_idx in self.examples:
            denom = len(query | example_tokens) or 1
            score = len(query & example_tokens) / denom
            scores[answer_idx] = max(scores[answer_idx], float(score))
        return scores

    def state_bytes(self) -> int:
        return len(pickle.dumps(self.examples, protocol=pickle.HIGHEST_PROTOCOL))


class HashedLookupBaseline:
    """Hashed sparse feature count table. Diagnostic statistical baseline only."""

    def __init__(self, answer_to_idx: dict[str, int], hash_bits: int, ngrams: int, seed: int) -> None:
        self.answer_to_idx = answer_to_idx
        self.hash_bits = hash_bits
        self.ngrams = ngrams
        self.seed = seed
        self.tables: dict[int, dict[int, float]] = {}
        self.answer_counts = np.ones(len(answer_to_idx), dtype=np.float32)

    def features(self, row: dict[str, Any]) -> list[int]:
        all_tokens = tokens(row_text(row))
        feats: set[int] = set()
        for n in range(1, self.ngrams + 1):
            for idx in range(len(all_tokens) - n + 1):
                gram = " ".join(all_tokens[idx : idx + n])
                feats.add(stable_hash_int(f"{self.seed}:{n}:{gram}", bits=self.hash_bits))
        return sorted(feats)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            answer_idx = self.answer_to_idx[row["answer"]]
            self.answer_counts[answer_idx] += 1.0
            for feature in self.features(row):
                table = self.tables.setdefault(feature, {})
                table[answer_idx] = table.get(answer_idx, 0.0) + 1.0

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        scores = 0.02 * self.answer_counts.astype(np.float32, copy=True)
        for feature in self.features(row):
            table = self.tables.get(feature)
            if table is None:
                continue
            for answer_idx, count in table.items():
                scores[int(answer_idx)] += float(count)
        return scores

    def active_features(self) -> int:
        return len(self.tables)

    def state_bytes(self) -> int:
        state = {
            "tables": self.tables,
            "answer_counts": self.answer_counts,
            "hash_bits": self.hash_bits,
            "ngrams": self.ngrams,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


def parse_question_subject(question: str) -> str | None:
    match = re.search(r"where\s+(?:is|was)\s+(?:the\s+)?([a-z0-9]+)", question.lower())
    if match:
        return match.group(1)
    return None


def parse_before_question(question: str) -> tuple[str, str] | None:
    match = re.search(
        r"where\s+was\s+(?:the\s+)?([a-z0-9]+)\s+before\s+(?:the\s+)?([a-z0-9]+)",
        question.lower(),
    )
    if match and match.group(2) in LOCATION_WORDS:
        return match.group(1), match.group(2)
    return None


def parse_movement(sentence: str) -> tuple[str, str] | None:
    words = tokens(sentence)
    if len(words) < 4:
        return None
    subject = words[0]
    if not any(verb in words for verb in MOVE_VERBS):
        return None
    for idx, word in enumerate(words):
        if word == "to" and idx + 2 < len(words) and words[idx + 1] == "the":
            location = words[idx + 2]
            if location in LOCATION_WORDS:
                return subject, location
    return None


def parse_object_after_the(words: list[str]) -> str | None:
    for idx, word in enumerate(words):
        if word == "the" and idx + 1 < len(words):
            candidate = words[idx + 1]
            if candidate not in LOCATION_WORDS:
                return candidate
    return None


def parse_pickup(sentence: str) -> tuple[str, str] | None:
    words = tokens(sentence)
    if len(words) < 3:
        return None
    subject = words[0]
    if not any(verb in words for verb in PICKUP_VERBS):
        return None
    obj = parse_object_after_the(words)
    if obj is None:
        return None
    return subject, obj


def parse_drop(sentence: str) -> tuple[str, str] | None:
    words = tokens(sentence)
    if len(words) < 3:
        return None
    subject = words[0]
    if not any(verb in words for verb in DROP_VERBS):
        return None
    obj = parse_object_after_the(words)
    if obj is None:
        return None
    return subject, obj


def parse_event(sentence: str) -> dict[str, str | None]:
    movement = parse_movement(sentence)
    if movement is not None:
        person, location = movement
        return {"event": "move", "person": person, "object": None, "location": location}
    pickup = parse_pickup(sentence)
    if pickup is not None:
        person, obj = pickup
        return {"event": "pickup", "person": person, "object": obj, "location": None}
    dropped = parse_drop(sentence)
    if dropped is not None:
        person, obj = dropped
        return {"event": "drop", "person": person, "object": obj, "location": None}
    return {"event": "none", "person": None, "object": None, "location": None}


def parse_query(question: str) -> dict[str, str | None]:
    before = parse_before_question(question)
    if before is not None:
        obj, destination = before
        return {"query": "where_before", "subject": obj, "destination": destination}
    subject = parse_question_subject(question)
    if subject is not None:
        return {"query": "where_is", "subject": subject, "destination": None}
    return {"query": "where_is", "subject": None, "destination": None}


def singularize_noun(word: str) -> str:
    word = word.lower()
    if word in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[word]
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 1:
        return word[:-1]
    return word


def parse_attribute_statement(sentence: str) -> dict[str, str | None]:
    words = tokens(sentence)
    if len(words) >= 5 and words[1] == "are" and words[2] == "afraid" and words[3] == "of":
        return {
            "event": "class_afraid",
            "entity": singularize_noun(words[0]),
            "value": singularize_noun(words[4]),
        }
    if len(words) >= 4 and words[1] == "is" and words[2] == "a":
        return {
            "event": "entity_class",
            "entity": words[0],
            "value": singularize_noun(words[3]),
        }
    if len(words) >= 3 and words[1] == "is" and words[2] in COLOR_WORDS:
        return {"event": "entity_color", "entity": words[0], "value": words[2]}
    return {"event": "none", "entity": None, "value": None}


def parse_attribute_query(question: str) -> dict[str, str | None]:
    words = tokens(question)
    if len(words) >= 5 and words[0] == "what" and words[1] == "is" and words[3] == "afraid":
        return {"query": "afraid_of", "subject": words[2]}
    if len(words) >= 4 and words[0] == "what" and words[1] == "color" and words[2] == "is":
        return {"query": "color", "subject": words[3]}
    return {"query": "none", "subject": None}


class LearnedEventDetector:
    """
    Local no-BP detector for bAbI event type and role slots.

    Training labels are derived from sentence-local event structure only, never
    from QA answers.  Updates are perceptron-style target/wrong-winner writes
    over fixed random token/position features and prototype averaging for slot
    values.
    """

    def __init__(
        self,
        dim: int = 64,
        lr: float = 0.08,
        epochs: int = 3,
        score_scale: float = 6.0,
        seed: int = 0,
        confidence_threshold: float = 0.15,
    ) -> None:
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.score_scale = score_scale
        self.seed = seed
        self.confidence_threshold = confidence_threshold
        self.rng = np.random.default_rng(seed)
        self.event_weights = np.zeros((len(EVENT_TYPES), dim), dtype=np.float32)
        self.person_prototypes: dict[str, np.ndarray] = {}
        self.object_prototypes: dict[str, np.ndarray] = {}
        self.location_prototypes: dict[str, np.ndarray] = {}
        self.person_counts: Counter[str] = Counter()
        self.object_counts: Counter[str] = Counter()
        self.location_counts: Counter[str] = Counter()
        self._token_code_cache: dict[tuple[str, str], np.ndarray] = {}

    def token_code(self, token: str, slot: str) -> np.ndarray:
        cache_key = (token, slot)
        cached = self._token_code_cache.get(cache_key)
        if cached is not None:
            return cached
        seed = stable_hash_int(f"{self.seed}:event-detector:{slot}:{token}", bits=64)
        rng = np.random.default_rng(seed)
        code = normalize_vector(rng.normal(0.0, 1.0, self.dim).astype(np.float32))
        self._token_code_cache[cache_key] = code
        return code

    def sentence_feature(self, sentence: str) -> np.ndarray:
        words = tokens(sentence)
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        feature = np.zeros(self.dim, dtype=np.float32)
        for idx, word in enumerate(words):
            feature += self.token_code(word, f"tok:{idx}")
            feature += 0.5 * self.token_code(word, "bag")
        return normalize_vector(feature)

    def slot_feature(self, sentence: str, slot: Literal["person", "object", "location"]) -> np.ndarray:
        words = tokens(sentence)
        feature = np.zeros(self.dim, dtype=np.float32)
        for idx, word in enumerate(words):
            if slot == "person":
                if idx == 0:
                    feature += 1.5 * self.token_code(word, "slot-person-first")
                feature += 0.25 * self.token_code(word, "slot-person-bag")
            elif slot == "object":
                if idx > 0 and words[idx - 1] == "the":
                    feature += 1.5 * self.token_code(word, "slot-object-after-the")
                feature += 0.25 * self.token_code(word, "slot-object-bag")
            else:
                if idx > 1 and words[idx - 1] == "the" and words[idx - 2] == "to":
                    feature += 1.5 * self.token_code(word, "slot-location-after-to-the")
                if idx > 1 and words[idx - 1] == "the" and words[idx - 2] == "before":
                    feature += 1.0 * self.token_code(word, "slot-location-after-before-the")
                feature += 0.25 * self.token_code(word, "slot-location-bag")
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
        counts[label] += 1
        if label not in prototypes:
            prototypes[label] = feature.copy()
        else:
            eta = 1.0 / float(counts[label])
            prototypes[label] = normalize_vector((1.0 - eta) * prototypes[label] + eta * feature)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples: list[tuple[str, dict[str, str | None]]] = []
        for row in rows:
            for item in row["context"]:
                sentence = str(item["text"])
                parsed = parse_event(sentence)
                examples.append((sentence, parsed))
        if not examples:
            return
        for _ in range(self.epochs):
            order = self.rng.permutation(len(examples))
            for idx in order:
                sentence, parsed = examples[int(idx)]
                feature = self.sentence_feature(sentence)
                target = EVENT_TO_IDX[str(parsed["event"])]
                scores = self.event_weights @ feature
                pred = int(np.argmax(scores))
                if pred != target:
                    self.event_weights[target] += self.lr * feature
                    self.event_weights[pred] -= self.lr * feature
                    self.event_weights[target] = normalize_vector(self.event_weights[target])
                    self.event_weights[pred] = normalize_vector(self.event_weights[pred])
                self.update_prototype(
                    self.person_prototypes,
                    self.person_counts,
                    parsed["person"],
                    self.slot_feature(sentence, "person"),
                )
                self.update_prototype(
                    self.object_prototypes,
                    self.object_counts,
                    parsed["object"],
                    self.slot_feature(sentence, "object"),
                )
                self.update_prototype(
                    self.location_prototypes,
                    self.location_counts,
                    parsed["location"],
                    self.slot_feature(sentence, "location"),
                )

    def decode_slot(self, feature: np.ndarray, prototypes: dict[str, np.ndarray]) -> tuple[str | None, float]:
        if not prototypes:
            return None, 0.0
        best_label: str | None = None
        best_score = -1e9
        for label, proto in prototypes.items():
            score = float(proto @ feature)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    def predict(self, sentence: str) -> dict[str, str | float | None]:
        feature = self.sentence_feature(sentence)
        raw_scores = self.event_weights @ feature
        scores = self.score_scale * raw_scores
        event_idx = int(np.argmax(scores))
        sorted_scores = np.sort(scores)
        margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) >= 2 else 0.0
        event = EVENT_TYPES[event_idx]
        person, person_conf = self.decode_slot(self.slot_feature(sentence, "person"), self.person_prototypes)
        obj, obj_conf = self.decode_slot(self.slot_feature(sentence, "object"), self.object_prototypes)
        location, location_conf = self.decode_slot(self.slot_feature(sentence, "location"), self.location_prototypes)
        if margin < self.confidence_threshold:
            event = "none"
        return {
            "event": event,
            "person": person,
            "object": obj,
            "location": location,
            "event_confidence": margin,
            "person_confidence": person_conf,
            "object_confidence": obj_conf,
            "location_confidence": location_conf,
        }

    def state_bytes(self) -> int:
        state = {
            "dim": self.dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "score_scale": self.score_scale,
            "confidence_threshold": self.confidence_threshold,
            "event_weights": self.event_weights,
            "person_prototypes": self.person_prototypes,
            "object_prototypes": self.object_prototypes,
            "location_prototypes": self.location_prototypes,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))

    def event_metrics(self, rows: list[dict[str, Any]], limit: int = 0) -> dict[str, Any]:
        total = 0
        event_correct = 0
        person_total = person_correct = 0
        object_total = object_correct = 0
        location_total = location_correct = 0
        confusion = Counter()
        for row in rows:
            for item in row["context"]:
                if limit and total >= limit:
                    break
                sentence = str(item["text"])
                target = parse_event(sentence)
                pred = self.predict(sentence)
                target_event = str(target["event"])
                pred_event = str(pred["event"])
                event_correct += int(pred_event == target_event)
                confusion[(target_event, pred_event)] += 1
                if target["person"] is not None:
                    person_total += 1
                    person_correct += int(pred["person"] == target["person"])
                if target["object"] is not None:
                    object_total += 1
                    object_correct += int(pred["object"] == target["object"])
                if target["location"] is not None:
                    location_total += 1
                    location_correct += int(pred["location"] == target["location"])
                total += 1
            if limit and total >= limit:
                break
        return {
            "sentences": total,
            "event_accuracy": event_correct / max(total, 1),
            "person_accuracy": person_correct / max(person_total, 1),
            "object_accuracy": object_correct / max(object_total, 1),
            "location_accuracy": location_correct / max(location_total, 1),
            "person_total": person_total,
            "object_total": object_total,
            "location_total": location_total,
            "confusion": {f"{a}->{b}": c for (a, b), c in sorted(confusion.items())},
        }


class LearnedQueryDetector:
    """
    Local no-BP question parser for query type and slots.

    Labels come from question-local structure only, not from the answer.  The
    detector uses fixed random token/position features, perceptron target/wrong
    updates for query type, and prototype averaging for subject/destination.
    """

    def __init__(
        self,
        dim: int = 64,
        lr: float = 0.08,
        epochs: int = 3,
        score_scale: float = 6.0,
        seed: int = 0,
        confidence_threshold: float = 0.0,
    ) -> None:
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.score_scale = score_scale
        self.seed = seed
        self.confidence_threshold = confidence_threshold
        self.rng = np.random.default_rng(seed)
        self.query_weights = np.zeros((len(QUERY_TYPES), dim), dtype=np.float32)
        self.subject_prototypes: dict[str, np.ndarray] = {}
        self.destination_prototypes: dict[str, np.ndarray] = {}
        self.subject_counts: Counter[str] = Counter()
        self.destination_counts: Counter[str] = Counter()
        self._token_code_cache: dict[tuple[str, str], np.ndarray] = {}

    def token_code(self, token: str, slot: str) -> np.ndarray:
        cache_key = (token, slot)
        cached = self._token_code_cache.get(cache_key)
        if cached is not None:
            return cached
        seed = stable_hash_int(f"{self.seed}:query-detector:{slot}:{token}", bits=64)
        rng = np.random.default_rng(seed)
        code = normalize_vector(rng.normal(0.0, 1.0, self.dim).astype(np.float32))
        self._token_code_cache[cache_key] = code
        return code

    def question_feature(self, question: str) -> np.ndarray:
        words = tokens(question)
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        feature = np.zeros(self.dim, dtype=np.float32)
        for idx, word in enumerate(words):
            feature += self.token_code(word, f"tok:{idx}")
            feature += 0.5 * self.token_code(word, "bag")
        return normalize_vector(feature)

    def slot_feature(self, question: str, slot: Literal["subject", "destination"]) -> np.ndarray:
        words = tokens(question)
        feature = np.zeros(self.dim, dtype=np.float32)
        for idx, word in enumerate(words):
            if slot == "subject":
                if idx > 0 and words[idx - 1] == "is":
                    feature += 1.5 * self.token_code(word, "slot-subject-after-is")
                if idx > 1 and words[idx - 1] == "the" and words[idx - 2] in {"is", "was"}:
                    feature += 1.5 * self.token_code(word, "slot-subject-after-is-the")
                feature += 0.25 * self.token_code(word, "slot-subject-bag")
            else:
                if idx > 1 and words[idx - 1] == "the" and words[idx - 2] == "before":
                    feature += 1.5 * self.token_code(word, "slot-destination-after-before-the")
                feature += 0.25 * self.token_code(word, "slot-destination-bag")
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
        counts[label] += 1
        if label not in prototypes:
            prototypes[label] = feature.copy()
        else:
            eta = 1.0 / float(counts[label])
            prototypes[label] = normalize_vector((1.0 - eta) * prototypes[label] + eta * feature)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples = [(str(row["question"]), parse_query(str(row["question"]))) for row in rows]
        if not examples:
            return
        for _ in range(self.epochs):
            order = self.rng.permutation(len(examples))
            for idx in order:
                question, parsed = examples[int(idx)]
                feature = self.question_feature(question)
                target = QUERY_TO_IDX[str(parsed["query"])]
                scores = self.query_weights @ feature
                pred = int(np.argmax(scores))
                if pred != target:
                    self.query_weights[target] += self.lr * feature
                    self.query_weights[pred] -= self.lr * feature
                    self.query_weights[target] = normalize_vector(self.query_weights[target])
                    self.query_weights[pred] = normalize_vector(self.query_weights[pred])
                self.update_prototype(
                    self.subject_prototypes,
                    self.subject_counts,
                    parsed["subject"],
                    self.slot_feature(question, "subject"),
                )
                self.update_prototype(
                    self.destination_prototypes,
                    self.destination_counts,
                    parsed["destination"],
                    self.slot_feature(question, "destination"),
                )

    def decode_slot(self, feature: np.ndarray, prototypes: dict[str, np.ndarray]) -> tuple[str | None, float]:
        if not prototypes:
            return None, 0.0
        best_label: str | None = None
        best_score = -1e9
        for label, proto in prototypes.items():
            score = float(proto @ feature)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    def predict(self, question: str) -> dict[str, str | float | None]:
        feature = self.question_feature(question)
        raw_scores = self.query_weights @ feature
        scores = self.score_scale * raw_scores
        query_idx = int(np.argmax(scores))
        sorted_scores = np.sort(scores)
        margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) >= 2 else 0.0
        query = QUERY_TYPES[query_idx]
        subject, subject_conf = self.decode_slot(self.slot_feature(question, "subject"), self.subject_prototypes)
        destination, destination_conf = self.decode_slot(
            self.slot_feature(question, "destination"), self.destination_prototypes
        )
        if margin < self.confidence_threshold:
            query = "where_is"
        return {
            "query": query,
            "subject": subject,
            "destination": destination,
            "query_confidence": margin,
            "subject_confidence": subject_conf,
            "destination_confidence": destination_conf,
        }

    def state_bytes(self) -> int:
        state = {
            "dim": self.dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "score_scale": self.score_scale,
            "confidence_threshold": self.confidence_threshold,
            "query_weights": self.query_weights,
            "subject_prototypes": self.subject_prototypes,
            "destination_prototypes": self.destination_prototypes,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))

    def query_metrics(self, rows: list[dict[str, Any]], limit: int = 0) -> dict[str, Any]:
        total = 0
        query_correct = 0
        subject_total = subject_correct = 0
        destination_total = destination_correct = 0
        confusion = Counter()
        for row in rows:
            if limit and total >= limit:
                break
            question = str(row["question"])
            target = parse_query(question)
            pred = self.predict(question)
            target_query = str(target["query"])
            pred_query = str(pred["query"])
            query_correct += int(pred_query == target_query)
            confusion[(target_query, pred_query)] += 1
            if target["subject"] is not None:
                subject_total += 1
                subject_correct += int(pred["subject"] == target["subject"])
            if target["destination"] is not None:
                destination_total += 1
                destination_correct += int(pred["destination"] == target["destination"])
            total += 1
        return {
            "questions": total,
            "query_accuracy": query_correct / max(total, 1),
            "subject_accuracy": subject_correct / max(subject_total, 1),
            "destination_accuracy": destination_correct / max(destination_total, 1),
            "subject_total": subject_total,
            "destination_total": destination_total,
            "confusion": {f"{a}->{b}": c for (a, b), c in sorted(confusion.items())},
        }


class LearnedAttributeStatementDetector:
    """
    Local no-BP statement parser for QA15/QA16 attribute facts.

    Labels are sentence-local grammar labels, not answer labels.  Event type is
    trained with fixed random token/position features and perceptron-style
    target/wrong updates; entity/value slots use local prototype averaging.
    """

    def __init__(
        self,
        dim: int = 64,
        lr: float = 0.08,
        epochs: int = 3,
        score_scale: float = 6.0,
        seed: int = 0,
        confidence_threshold: float = 0.0,
    ) -> None:
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.score_scale = score_scale
        self.seed = seed
        self.confidence_threshold = confidence_threshold
        self.rng = np.random.default_rng(seed)
        self.event_weights = np.zeros((len(ATTRIBUTE_EVENT_TYPES), dim), dtype=np.float32)
        self.entity_prototypes: dict[str, np.ndarray] = {}
        self.value_prototypes: dict[str, np.ndarray] = {}
        self.entity_counts: Counter[str] = Counter()
        self.value_counts: Counter[str] = Counter()
        self._token_code_cache: dict[tuple[str, str], np.ndarray] = {}

    def token_code(self, token: str, slot: str) -> np.ndarray:
        cache_key = (token, slot)
        cached = self._token_code_cache.get(cache_key)
        if cached is not None:
            return cached
        seed = stable_hash_int(f"{self.seed}:attribute-statement:{slot}:{token}", bits=64)
        rng = np.random.default_rng(seed)
        code = normalize_vector(rng.normal(0.0, 1.0, self.dim).astype(np.float32))
        self._token_code_cache[cache_key] = code
        return code

    def sentence_feature(self, sentence: str) -> np.ndarray:
        words = tokens(sentence)
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        feature = np.zeros(self.dim, dtype=np.float32)
        for idx, word in enumerate(words):
            feature += self.token_code(word, f"tok:{idx}")
            feature += 0.5 * self.token_code(word, "bag")
        return normalize_vector(feature)

    def slot_feature(self, sentence: str, slot: Literal["entity", "value"]) -> np.ndarray:
        words = tokens(sentence)
        feature = np.zeros(self.dim, dtype=np.float32)
        for idx, word in enumerate(words):
            if slot == "entity":
                if idx == 0:
                    feature += 1.75 * self.token_code(singularize_noun(word), "slot-entity-first")
                feature += 0.2 * self.token_code(singularize_noun(word), "slot-entity-bag")
            else:
                value_word = singularize_noun(word)
                if idx > 0 and words[idx - 1] in {"a", "of"}:
                    feature += 1.75 * self.token_code(value_word, "slot-value-after-a-or-of")
                if idx > 0 and words[idx - 1] == "is" and word in COLOR_WORDS:
                    feature += 1.75 * self.token_code(word, "slot-value-color-after-is")
                if idx == len(words) - 1:
                    feature += 0.75 * self.token_code(value_word, "slot-value-last")
                feature += 0.2 * self.token_code(value_word, "slot-value-bag")
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
        counts[label] += 1
        if label not in prototypes:
            prototypes[label] = feature.copy()
        else:
            eta = 1.0 / float(counts[label])
            prototypes[label] = normalize_vector((1.0 - eta) * prototypes[label] + eta * feature)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples: list[tuple[str, dict[str, str | None]]] = []
        for row in rows:
            for item in row["context"]:
                sentence = str(item["text"])
                examples.append((sentence, parse_attribute_statement(sentence)))
        if not examples:
            return
        for _ in range(self.epochs):
            order = self.rng.permutation(len(examples))
            for idx in order:
                sentence, parsed = examples[int(idx)]
                feature = self.sentence_feature(sentence)
                target = ATTRIBUTE_EVENT_TO_IDX[str(parsed["event"])]
                scores = self.event_weights @ feature
                pred = int(np.argmax(scores))
                if pred != target:
                    self.event_weights[target] += self.lr * feature
                    self.event_weights[pred] -= self.lr * feature
                    self.event_weights[target] = normalize_vector(self.event_weights[target])
                    self.event_weights[pred] = normalize_vector(self.event_weights[pred])
                self.update_prototype(
                    self.entity_prototypes,
                    self.entity_counts,
                    parsed["entity"],
                    self.slot_feature(sentence, "entity"),
                )
                self.update_prototype(
                    self.value_prototypes,
                    self.value_counts,
                    parsed["value"],
                    self.slot_feature(sentence, "value"),
                )

    def decode_slot(self, feature: np.ndarray, prototypes: dict[str, np.ndarray]) -> tuple[str | None, float]:
        if not prototypes:
            return None, 0.0
        best_label: str | None = None
        best_score = -1e9
        for label, proto in prototypes.items():
            score = float(proto @ feature)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    def predict(self, sentence: str) -> dict[str, str | float | None]:
        feature = self.sentence_feature(sentence)
        raw_scores = self.event_weights @ feature
        scores = self.score_scale * raw_scores
        event_idx = int(np.argmax(scores))
        sorted_scores = np.sort(scores)
        margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) >= 2 else 0.0
        event = ATTRIBUTE_EVENT_TYPES[event_idx]
        if margin < self.confidence_threshold:
            event = "none"
        entity, entity_conf = self.decode_slot(self.slot_feature(sentence, "entity"), self.entity_prototypes)
        value, value_conf = self.decode_slot(self.slot_feature(sentence, "value"), self.value_prototypes)
        return {
            "event": event,
            "entity": entity,
            "value": value,
            "event_confidence": margin,
            "entity_confidence": entity_conf,
            "value_confidence": value_conf,
        }

    def state_bytes(self) -> int:
        state = {
            "dim": self.dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "score_scale": self.score_scale,
            "confidence_threshold": self.confidence_threshold,
            "event_weights": self.event_weights,
            "entity_prototypes": self.entity_prototypes,
            "value_prototypes": self.value_prototypes,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))

    def statement_metrics(self, rows: list[dict[str, Any]], limit: int = 0) -> dict[str, Any]:
        total = 0
        event_correct = 0
        entity_total = entity_correct = 0
        value_total = value_correct = 0
        confusion = Counter()
        for row in rows:
            for item in row["context"]:
                if limit and total >= limit:
                    break
                sentence = str(item["text"])
                target = parse_attribute_statement(sentence)
                pred = self.predict(sentence)
                target_event = str(target["event"])
                pred_event = str(pred["event"])
                event_correct += int(pred_event == target_event)
                confusion[(target_event, pred_event)] += 1
                if target["entity"] is not None:
                    entity_total += 1
                    entity_correct += int(pred["entity"] == target["entity"])
                if target["value"] is not None:
                    value_total += 1
                    value_correct += int(pred["value"] == target["value"])
                total += 1
            if limit and total >= limit:
                break
        return {
            "sentences": total,
            "event_accuracy": event_correct / max(total, 1),
            "entity_accuracy": entity_correct / max(entity_total, 1),
            "value_accuracy": value_correct / max(value_total, 1),
            "entity_total": entity_total,
            "value_total": value_total,
            "confusion": {f"{a}->{b}": c for (a, b), c in sorted(confusion.items())},
        }


class LearnedAttributeQueryDetector:
    """Local no-BP query parser for QA15/QA16 attribute questions."""

    def __init__(
        self,
        dim: int = 64,
        lr: float = 0.08,
        epochs: int = 3,
        score_scale: float = 6.0,
        seed: int = 0,
        confidence_threshold: float = 0.0,
    ) -> None:
        self.dim = dim
        self.lr = lr
        self.epochs = epochs
        self.score_scale = score_scale
        self.seed = seed
        self.confidence_threshold = confidence_threshold
        self.rng = np.random.default_rng(seed)
        self.query_weights = np.zeros((len(ATTRIBUTE_QUERY_TYPES), dim), dtype=np.float32)
        self.subject_prototypes: dict[str, np.ndarray] = {}
        self.subject_counts: Counter[str] = Counter()
        self._token_code_cache: dict[tuple[str, str], np.ndarray] = {}

    def token_code(self, token: str, slot: str) -> np.ndarray:
        cache_key = (token, slot)
        cached = self._token_code_cache.get(cache_key)
        if cached is not None:
            return cached
        seed = stable_hash_int(f"{self.seed}:attribute-query:{slot}:{token}", bits=64)
        rng = np.random.default_rng(seed)
        code = normalize_vector(rng.normal(0.0, 1.0, self.dim).astype(np.float32))
        self._token_code_cache[cache_key] = code
        return code

    def question_feature(self, question: str) -> np.ndarray:
        words = tokens(question)
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        feature = np.zeros(self.dim, dtype=np.float32)
        for idx, word in enumerate(words):
            feature += self.token_code(word, f"tok:{idx}")
            feature += 0.5 * self.token_code(word, "bag")
        return normalize_vector(feature)

    def subject_feature(self, question: str) -> np.ndarray:
        words = tokens(question)
        feature = np.zeros(self.dim, dtype=np.float32)
        for idx, word in enumerate(words):
            if idx > 0 and words[idx - 1] == "is":
                feature += 1.75 * self.token_code(word, "slot-subject-after-is")
            if idx > 1 and words[idx - 1] == "is" and words[idx - 2] == "color":
                feature += 1.75 * self.token_code(word, "slot-subject-after-color-is")
            feature += 0.2 * self.token_code(word, "slot-subject-bag")
        return normalize_vector(feature)

    def update_prototype(self, label: str | None, feature: np.ndarray) -> None:
        if label is None:
            return
        self.subject_counts[label] += 1
        if label not in self.subject_prototypes:
            self.subject_prototypes[label] = feature.copy()
        else:
            eta = 1.0 / float(self.subject_counts[label])
            self.subject_prototypes[label] = normalize_vector(
                (1.0 - eta) * self.subject_prototypes[label] + eta * feature
            )

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples = [(str(row["question"]), parse_attribute_query(str(row["question"]))) for row in rows]
        if not examples:
            return
        for _ in range(self.epochs):
            order = self.rng.permutation(len(examples))
            for idx in order:
                question, parsed = examples[int(idx)]
                feature = self.question_feature(question)
                target = ATTRIBUTE_QUERY_TO_IDX[str(parsed["query"])]
                scores = self.query_weights @ feature
                pred = int(np.argmax(scores))
                if pred != target:
                    self.query_weights[target] += self.lr * feature
                    self.query_weights[pred] -= self.lr * feature
                    self.query_weights[target] = normalize_vector(self.query_weights[target])
                    self.query_weights[pred] = normalize_vector(self.query_weights[pred])
                self.update_prototype(parsed["subject"], self.subject_feature(question))

    def decode_subject(self, feature: np.ndarray) -> tuple[str | None, float]:
        if not self.subject_prototypes:
            return None, 0.0
        best_label: str | None = None
        best_score = -1e9
        for label, proto in self.subject_prototypes.items():
            score = float(proto @ feature)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    def predict(self, question: str) -> dict[str, str | float | None]:
        feature = self.question_feature(question)
        raw_scores = self.query_weights @ feature
        scores = self.score_scale * raw_scores
        query_idx = int(np.argmax(scores))
        sorted_scores = np.sort(scores)
        margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) >= 2 else 0.0
        query = ATTRIBUTE_QUERY_TYPES[query_idx]
        if margin < self.confidence_threshold:
            query = "none"
        subject, subject_conf = self.decode_subject(self.subject_feature(question))
        return {
            "query": query,
            "subject": subject,
            "query_confidence": margin,
            "subject_confidence": subject_conf,
        }

    def state_bytes(self) -> int:
        state = {
            "dim": self.dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "score_scale": self.score_scale,
            "confidence_threshold": self.confidence_threshold,
            "query_weights": self.query_weights,
            "subject_prototypes": self.subject_prototypes,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))

    def query_metrics(self, rows: list[dict[str, Any]], limit: int = 0) -> dict[str, Any]:
        total = 0
        query_correct = 0
        subject_total = subject_correct = 0
        confusion = Counter()
        for row in rows:
            if limit and total >= limit:
                break
            question = str(row["question"])
            target = parse_attribute_query(question)
            pred = self.predict(question)
            target_query = str(target["query"])
            pred_query = str(pred["query"])
            query_correct += int(pred_query == target_query)
            confusion[(target_query, pred_query)] += 1
            if target["subject"] is not None:
                subject_total += 1
                subject_correct += int(pred["subject"] == target["subject"])
            total += 1
        return {
            "questions": total,
            "query_accuracy": query_correct / max(total, 1),
            "subject_accuracy": subject_correct / max(subject_total, 1),
            "subject_total": subject_total,
            "confusion": {f"{a}->{b}": c for (a, b), c in sorted(confusion.items())},
        }


class SymbolicLocationTracker:
    """Task-specific diagnostic upper bound for location QA, not a learned method."""

    def __init__(self, answer_to_idx: dict[str, int], fallback: MajorityBaseline) -> None:
        self.answer_to_idx = answer_to_idx
        self.fallback = fallback

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        subject = parse_question_subject(str(row["question"]))
        locations: dict[str, str] = {}
        for item in row["context"]:
            parsed = parse_movement(str(item["text"]))
            if parsed is not None:
                name, location = parsed
                locations[name] = location
        if subject is not None and subject in locations and locations[subject] in self.answer_to_idx:
            scores = np.full(len(self.answer_to_idx), -4.0, dtype=np.float32)
            scores[self.answer_to_idx[locations[subject]]] = 4.0
            return scores
        return self.fallback.scores(row)

    def state_bytes(self) -> int:
        return 0


class RoleBindingStateQALearner:
    """
    Online role/state binding model with local no-BP association updates.

    It treats context sentences as an input stream and maintains four local
    associative states:
    - person -> location;
    - object -> current owner;
    - object -> current location;
    - (object, destination) -> previous location.

    The matrices are updated by local delta-Hebbian writes and reset per story.
    No answer labels, BP, pretrained encoders, raw example replay, or statistical
    token-count lookup are used by this model.
    """

    def __init__(
        self,
        answer_to_idx: dict[str, int],
        fallback: MajorityBaseline,
        event_detector: LearnedEventDetector | None = None,
        event_mode: str = "regex",
        query_detector: LearnedQueryDetector | None = None,
        query_mode: str = "regex",
        dim: int = 64,
        lr: float = 1.0,
        score_scale: float = 8.0,
        carry_threshold: float = 0.35,
        seed: int = 0,
    ) -> None:
        self.answer_to_idx = answer_to_idx
        self.fallback = fallback
        self.event_detector = event_detector
        self.event_mode = event_mode
        self.query_detector = query_detector
        self.query_mode = query_mode
        self.dim = dim
        self.lr = lr
        self.score_scale = score_scale
        self.carry_threshold = carry_threshold
        self.seed = seed
        self._code_cache: dict[tuple[str, str], np.ndarray] = {}
        self.answer_locations = [answer for answer in answer_to_idx if answer in LOCATION_WORDS]
        self.location_codes = {
            location: self.code(location, "location") for location in sorted(LOCATION_WORDS | set(answer_to_idx))
        }
        self.null_owner_code = self.code(NULL_OWNER, "person")

    def fit(self, rows: list[dict[str, Any]]) -> None:
        if self.event_detector is not None:
            self.event_detector.fit(rows)
        if self.query_detector is not None:
            self.query_detector.fit(rows)

    def code(self, token: str, role: str) -> np.ndarray:
        cache_key = (token, role)
        cached = self._code_cache.get(cache_key)
        if cached is not None:
            return cached
        seed = stable_hash_int(f"{self.seed}:role-binding:{role}:{token}", bits=64)
        rng = np.random.default_rng(seed)
        code = normalize_vector(rng.normal(0.0, 1.0, self.dim).astype(np.float32))
        self._code_cache[cache_key] = code
        return code

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.dim % 2 == 0:
            return complex_bind(a, b)
        return normalize_vector(a * b)

    def associate(self, matrix: np.ndarray, key: np.ndarray, value: np.ndarray) -> None:
        pred = matrix @ key
        matrix += self.lr * np.outer(value - pred, key).astype(np.float32)

    def retrieve(self, matrix: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, float]:
        value = matrix @ key
        conf = float(np.linalg.norm(value))
        if conf <= 1e-8:
            return value.astype(np.float32), 0.0
        return (value / conf).astype(np.float32), conf

    def decode_location(self, value: np.ndarray) -> tuple[str | None, float, np.ndarray]:
        if not self.answer_locations:
            return None, 0.0, np.zeros(len(self.answer_to_idx), dtype=np.float32)
        scores = np.full(len(self.answer_to_idx), -4.0, dtype=np.float32)
        best_location: str | None = None
        best_score = -1e9
        for location in self.answer_locations:
            idx = self.answer_to_idx[location]
            score = float(self.location_codes[location] @ value)
            scores[idx] = self.score_scale * score
            if score > best_score:
                best_score = score
                best_location = location
        return best_location, best_score, scores

    def set_object_location(
        self,
        state: dict[str, Any],
        obj: str,
        location: str,
    ) -> None:
        obj_code = self.code(obj, "object")
        location_code = self.location_codes[location]
        prev_value, prev_conf = self.retrieve(state["object_location"], obj_code)
        prev_location, prev_score, _ = self.decode_location(prev_value)
        if prev_conf > 0.05 and prev_location is not None and prev_location != location and prev_score > 0.15:
            before_key = self.bind(obj_code, location_code)
            self.associate(state["before_location"], before_key, self.location_codes[prev_location])
        self.associate(state["object_location"], obj_code, location_code)
        state["objects"].add(obj)

    def set_person_location(self, state: dict[str, Any], person: str, location: str) -> None:
        person_code = self.code(person, "person")
        location_code = self.location_codes[location]
        self.associate(state["person_location"], person_code, location_code)
        state["people"].add(person)
        carried = []
        for obj in sorted(state["objects"]):
            owner_value, owner_conf = self.retrieve(state["object_owner"], self.code(obj, "object"))
            owner_score = float(owner_value @ person_code) if owner_conf > 0.0 else 0.0
            if owner_score >= self.carry_threshold:
                carried.append(obj)
        for obj in carried:
            self.set_object_location(state, obj, location)

    def person_location(self, state: dict[str, Any], person: str) -> str | None:
        value, conf = self.retrieve(state["person_location"], self.code(person, "person"))
        if conf <= 0.05:
            return None
        location, score, _ = self.decode_location(value)
        if score <= 0.0:
            return None
        return location

    def detect_event(self, sentence: str) -> dict[str, str | None]:
        if self.event_detector is None or self.event_mode == "regex":
            return parse_event(sentence)
        detected = self.event_detector.predict(sentence)
        event = str(detected["event"])
        parsed = {
            "event": event,
            "person": detected["person"] if isinstance(detected["person"], str) else None,
            "object": detected["object"] if isinstance(detected["object"], str) else None,
            "location": detected["location"] if isinstance(detected["location"], str) else None,
        }
        if self.event_mode == "hybrid" and parsed["event"] == "none":
            return parse_event(sentence)
        return parsed

    def observe(self, state: dict[str, Any], sentence: str) -> None:
        event = self.detect_event(sentence)
        if event["event"] == "move":
            person = event["person"]
            location = event["location"]
            if person is None or location is None:
                return
            self.set_person_location(state, person, location)
            return

        if event["event"] == "pickup":
            person = event["person"]
            obj = event["object"]
            if person is None or obj is None:
                return
            state["people"].add(person)
            state["objects"].add(obj)
            person_code = self.code(person, "person")
            obj_code = self.code(obj, "object")
            self.associate(state["object_owner"], obj_code, person_code)
            location = self.person_location(state, person)
            if location is not None:
                self.set_object_location(state, obj, location)
            return

        if event["event"] == "drop":
            person = event["person"]
            obj = event["object"]
            if person is None or obj is None:
                return
            state["people"].add(person)
            state["objects"].add(obj)
            location = self.person_location(state, person)
            if location is not None:
                self.set_object_location(state, obj, location)
            self.associate(state["object_owner"], self.code(obj, "object"), self.null_owner_code)

    def new_state(self) -> dict[str, Any]:
        return {
            "person_location": np.zeros((self.dim, self.dim), dtype=np.float32),
            "object_owner": np.zeros((self.dim, self.dim), dtype=np.float32),
            "object_location": np.zeros((self.dim, self.dim), dtype=np.float32),
            "before_location": np.zeros((self.dim, self.dim), dtype=np.float32),
            "people": set(),
            "objects": set(),
        }

    def read_context(self, row: dict[str, Any]) -> dict[str, Any]:
        state = self.new_state()
        for item in row["context"]:
            self.observe(state, str(item["text"]))
        return state

    def detect_query(self, question: str) -> dict[str, str | None]:
        if self.query_detector is None or self.query_mode == "regex":
            return parse_query(question)
        detected = self.query_detector.predict(question)
        parsed = {
            "query": str(detected["query"]),
            "subject": detected["subject"] if isinstance(detected["subject"], str) else None,
            "destination": detected["destination"] if isinstance(detected["destination"], str) else None,
        }
        if self.query_mode == "hybrid" and parsed["subject"] is None:
            return parse_query(question)
        return parsed

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        state = self.read_context(row)
        query = self.detect_query(str(row["question"]))
        if query["query"] == "where_before":
            obj = query["subject"]
            destination = query["destination"]
            if obj is None or destination is None or destination not in self.location_codes:
                return self.fallback.scores(row)
            key = self.bind(self.code(obj, "object"), self.location_codes[destination])
            value, conf = self.retrieve(state["before_location"], key)
            if conf > 0.05:
                _, _, scores = self.decode_location(value)
                return scores
            return self.fallback.scores(row)

        subject = query["subject"]
        if subject is None:
            return self.fallback.scores(row)
        if subject in state["objects"]:
            value, conf = self.retrieve(state["object_location"], self.code(subject, "object"))
        else:
            value, conf = self.retrieve(state["person_location"], self.code(subject, "person"))
        if conf <= 0.05:
            return self.fallback.scores(row)
        _, _, scores = self.decode_location(value)
        return scores

    def state_bytes(self) -> int:
        matrix_bytes = 4 * self.dim * self.dim * np.dtype(np.float32).itemsize
        state = {
            "dim": self.dim,
            "lr": self.lr,
            "score_scale": self.score_scale,
            "carry_threshold": self.carry_threshold,
            "seed": self.seed,
            "event_mode": self.event_mode,
            "event_detector_bytes": self.event_detector.state_bytes() if self.event_detector is not None else 0,
            "query_mode": self.query_mode,
            "query_detector_bytes": self.query_detector.state_bytes() if self.query_detector is not None else 0,
            "answer_locations": self.answer_locations,
            "ephemeral_matrix_bytes": matrix_bytes,
        }
        return matrix_bytes + len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


class AttributeBindingStateQALearner:
    """
    Online attribute/category binding model for bAbI QA15/QA16-style tasks.

    The model keeps local associative matrices for:
    - entity -> category;
    - category -> afraid-of answer;
    - entity -> color answer;
    - category -> color answer inferred from observed entity/category/color.

    Context state is rebuilt per story from local delta-Hebbian writes.  The
    model stores no raw examples and uses no answer labels while reading a
    story; the grammar front-end is deliberately task-specific, matching the
    first role-binding baseline before learned event/query detectors.
    """

    def __init__(
        self,
        answer_to_idx: dict[str, int],
        fallback: MajorityBaseline,
        statement_detector: LearnedAttributeStatementDetector | None = None,
        statement_mode: str = "regex",
        query_detector: LearnedAttributeQueryDetector | None = None,
        query_mode: str = "regex",
        dim: int = 64,
        lr: float = 1.0,
        score_scale: float = 8.0,
        confidence_threshold: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.answer_to_idx = answer_to_idx
        self.fallback = fallback
        self.statement_detector = statement_detector
        self.statement_mode = statement_mode
        self.query_detector = query_detector
        self.query_mode = query_mode
        self.dim = dim
        self.lr = lr
        self.score_scale = score_scale
        self.confidence_threshold = confidence_threshold
        self.seed = seed
        self._code_cache: dict[tuple[str, str], np.ndarray] = {}
        self.answer_codes = {answer: self.code(answer, "answer") for answer in answer_to_idx}

    def fit(self, rows: list[dict[str, Any]]) -> None:
        if self.statement_detector is not None:
            self.statement_detector.fit(rows)
        if self.query_detector is not None:
            self.query_detector.fit(rows)

    def code(self, token: str, role: str) -> np.ndarray:
        cache_key = (token, role)
        cached = self._code_cache.get(cache_key)
        if cached is not None:
            return cached
        seed = stable_hash_int(f"{self.seed}:attribute-binding:{role}:{token}", bits=64)
        rng = np.random.default_rng(seed)
        code = normalize_vector(rng.normal(0.0, 1.0, self.dim).astype(np.float32))
        self._code_cache[cache_key] = code
        return code

    def associate(self, matrix: np.ndarray, key: np.ndarray, value: np.ndarray) -> None:
        pred = matrix @ key
        matrix += self.lr * np.outer(value - pred, key).astype(np.float32)

    def retrieve(self, matrix: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, float]:
        value = matrix @ key
        conf = float(np.linalg.norm(value))
        if conf <= 1e-8:
            return value.astype(np.float32), 0.0
        return (value / conf).astype(np.float32), conf

    def decode_category(self, state: dict[str, Any], value: np.ndarray) -> tuple[str | None, float]:
        best_label: str | None = None
        best_score = -1e9
        for label in sorted(state["categories"]):
            score = float(self.code(label, "category") @ value)
            if score > best_score:
                best_score = score
                best_label = label
        if best_score < 0.0:
            return None, best_score
        return best_label, best_score

    def decode_answer_scores(self, value: np.ndarray) -> tuple[str | None, float, np.ndarray]:
        scores = np.full(len(self.answer_to_idx), -4.0, dtype=np.float32)
        best_answer: str | None = None
        best_score = -1e9
        for answer, idx in self.answer_to_idx.items():
            score = float(self.answer_codes[answer] @ value)
            scores[idx] = self.score_scale * score
            if score > best_score:
                best_score = score
                best_answer = answer
        return best_answer, best_score, scores

    def maybe_link_category_color(
        self,
        state: dict[str, Any],
        entity: str,
        category: str | None = None,
        color: str | None = None,
    ) -> None:
        if category is None:
            cat_value, cat_conf = self.retrieve(state["entity_category"], self.code(entity, "entity"))
            if cat_conf <= self.confidence_threshold:
                return
            category, cat_score = self.decode_category(state, cat_value)
            if category is None or cat_score <= 0.0:
                return
        if color is None:
            if entity not in state["colored_entities"]:
                return
            color_value, color_conf = self.retrieve(state["entity_color"], self.code(entity, "entity"))
            if color_conf <= self.confidence_threshold:
                return
            color, color_score, _ = self.decode_answer_scores(color_value)
            if color is None or color_score <= 0.0:
                return
        if color not in self.answer_codes:
            return
        self.associate(state["category_color"], self.code(category, "category"), self.answer_codes[color])

    def detect_statement(self, sentence: str) -> dict[str, str | None]:
        if self.statement_detector is None or self.statement_mode == "regex":
            return parse_attribute_statement(sentence)
        detected = self.statement_detector.predict(sentence)
        parsed = {
            "event": str(detected["event"]),
            "entity": detected["entity"] if isinstance(detected["entity"], str) else None,
            "value": detected["value"] if isinstance(detected["value"], str) else None,
        }
        if self.statement_mode == "hybrid" and (parsed["event"] == "none" or parsed["entity"] is None):
            return parse_attribute_statement(sentence)
        return parsed

    def observe(self, state: dict[str, Any], sentence: str) -> None:
        event = self.detect_statement(sentence)
        event_type = event["event"]
        entity = event["entity"]
        value = event["value"]
        if entity is None or value is None:
            return
        if event_type == "class_afraid":
            category = entity
            answer = value
            state["categories"].add(category)
            if answer in self.answer_codes:
                self.associate(
                    state["category_afraid"],
                    self.code(category, "category"),
                    self.answer_codes[answer],
                )
            return
        if event_type == "entity_class":
            category = value
            state["entities"].add(entity)
            state["categories"].add(category)
            self.associate(
                state["entity_category"],
                self.code(entity, "entity"),
                self.code(category, "category"),
            )
            self.maybe_link_category_color(state, entity, category)
            return
        if event_type == "entity_color":
            color = value
            if color not in self.answer_codes:
                return
            state["entities"].add(entity)
            state["colored_entities"].add(entity)
            self.associate(state["entity_color"], self.code(entity, "entity"), self.answer_codes[color])
            self.maybe_link_category_color(state, entity, color=color)

    def new_state(self) -> dict[str, Any]:
        return {
            "entity_category": np.zeros((self.dim, self.dim), dtype=np.float32),
            "category_afraid": np.zeros((self.dim, self.dim), dtype=np.float32),
            "entity_color": np.zeros((self.dim, self.dim), dtype=np.float32),
            "category_color": np.zeros((self.dim, self.dim), dtype=np.float32),
            "entities": set(),
            "colored_entities": set(),
            "categories": set(),
        }

    def read_context(self, row: dict[str, Any]) -> dict[str, Any]:
        state = self.new_state()
        for item in row["context"]:
            self.observe(state, str(item["text"]))
        return state

    def category_for_entity(self, state: dict[str, Any], entity: str) -> str | None:
        value, conf = self.retrieve(state["entity_category"], self.code(entity, "entity"))
        if conf <= self.confidence_threshold:
            return None
        category, score = self.decode_category(state, value)
        if category is None or score <= 0.0:
            return None
        return category

    def detect_query(self, question: str) -> dict[str, str | None]:
        if self.query_detector is None or self.query_mode == "regex":
            return parse_attribute_query(question)
        detected = self.query_detector.predict(question)
        parsed = {
            "query": str(detected["query"]),
            "subject": detected["subject"] if isinstance(detected["subject"], str) else None,
        }
        if self.query_mode == "hybrid" and (parsed["query"] == "none" or parsed["subject"] is None):
            return parse_attribute_query(question)
        return parsed

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        state = self.read_context(row)
        query = self.detect_query(str(row["question"]))
        subject = query["subject"]
        if subject is None:
            return self.fallback.scores(row)
        if query["query"] == "afraid_of":
            category = self.category_for_entity(state, subject)
            if category is None:
                return self.fallback.scores(row)
            value, conf = self.retrieve(state["category_afraid"], self.code(category, "category"))
            if conf <= self.confidence_threshold:
                return self.fallback.scores(row)
            _, score, scores = self.decode_answer_scores(value)
            return scores if score > 0.0 else self.fallback.scores(row)
        if query["query"] == "color":
            if subject in state["colored_entities"]:
                direct_value, direct_conf = self.retrieve(state["entity_color"], self.code(subject, "entity"))
                if direct_conf > self.confidence_threshold:
                    _, direct_score, direct_scores = self.decode_answer_scores(direct_value)
                    if direct_score > 0.0:
                        return direct_scores
            category = self.category_for_entity(state, subject)
            if category is None:
                return self.fallback.scores(row)
            value, conf = self.retrieve(state["category_color"], self.code(category, "category"))
            if conf <= self.confidence_threshold:
                return self.fallback.scores(row)
            _, score, scores = self.decode_answer_scores(value)
            return scores if score > 0.0 else self.fallback.scores(row)
        return self.fallback.scores(row)

    def state_bytes(self) -> int:
        matrix_bytes = 4 * self.dim * self.dim * np.dtype(np.float32).itemsize
        state = {
            "dim": self.dim,
            "lr": self.lr,
            "score_scale": self.score_scale,
            "confidence_threshold": self.confidence_threshold,
            "seed": self.seed,
            "statement_mode": self.statement_mode,
            "statement_detector_bytes": self.statement_detector.state_bytes()
            if self.statement_detector is not None
            else 0,
            "query_mode": self.query_mode,
            "query_detector_bytes": self.query_detector.state_bytes() if self.query_detector is not None else 0,
            "answer_vocab": sorted(self.answer_to_idx),
            "ephemeral_matrix_bytes": matrix_bytes,
        }
        detector_bytes = 0
        if self.statement_detector is not None:
            detector_bytes += self.statement_detector.state_bytes()
        if self.query_detector is not None:
            detector_bytes += self.query_detector.state_bytes()
        return matrix_bytes + detector_bytes + len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


def phase_identity(dim: int) -> np.ndarray:
    out = np.zeros(2 * dim, dtype=np.float32)
    out[0::2] = 1.0
    return out / (np.linalg.norm(out) + 1e-8)


def normalize_vector(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x) + 1e-8)).astype(np.float32)


def complex_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty_like(a)
    out[0::2] = a[0::2] * b[0::2] - a[1::2] * b[1::2]
    out[1::2] = a[1::2] * b[0::2] + a[0::2] * b[1::2]
    return normalize_vector(out)


class PhaseDendriticQALearner:
    """
    Fixed random phase features plus local branch-wise WTA readout updates.

    Branches:
    - question branch;
    - full context branch with recency weighting;
    - subject-gated context branch for lines mentioning the question entity.
    """

    def __init__(self, answer_to_idx: dict[str, int], cfg: PhaseQAConfig) -> None:
        self.answer_to_idx = answer_to_idx
        self.cfg = cfg
        self.branch_count = 3
        self.feature_dim = 2 * cfg.phase_dim
        self.weights = np.zeros((len(answer_to_idx), self.branch_count, self.feature_dim), dtype=np.float32)
        self.answer_counts = np.ones(len(answer_to_idx), dtype=np.float32)
        self.rng = np.random.default_rng(cfg.seed)
        self.branch_gains = np.ones(self.branch_count, dtype=np.float32)

    def token_code(self, token: str, branch: int) -> np.ndarray:
        seed = stable_hash_int(f"{self.cfg.seed}:phase:{branch}:{token}", bits=64)
        rng = np.random.default_rng(seed)
        phases = rng.uniform(-math.pi, math.pi, self.cfg.phase_dim).astype(np.float32)
        out = np.empty(self.feature_dim, dtype=np.float32)
        out[0::2] = np.cos(phases)
        out[1::2] = np.sin(phases)
        return normalize_vector(out)

    def encode_tokens(self, seq: list[str], branch: int, decay: float = 1.0) -> np.ndarray:
        if not seq:
            return phase_identity(self.cfg.phase_dim)
        additive = np.zeros(self.feature_dim, dtype=np.float32)
        bound = phase_identity(self.cfg.phase_dim)
        weight = 1.0
        for token in reversed(seq):
            code = self.token_code(token, branch)
            additive += weight * code
            bound = complex_bind(code, bound)
            weight *= decay
        return normalize_vector(additive + 0.35 * bound)

    def features(self, row: dict[str, Any]) -> np.ndarray:
        question_tokens = tokens(str(row["question"]))
        context_lines = [str(item["text"]) for item in row["context"]]
        context_tokens = tokens(" ".join(context_lines))
        subject = parse_question_subject(str(row["question"]))
        if subject is not None:
            subject_lines = [line for line in context_lines if subject in tokens(line)]
        else:
            subject_lines = []
        subject_tokens = tokens(" ".join(subject_lines if subject_lines else context_lines))
        return np.stack(
            [
                self.encode_tokens(question_tokens, 0, decay=1.0),
                self.encode_tokens(context_tokens, 1, decay=0.98),
                self.encode_tokens(subject_tokens, 2, decay=0.92),
            ],
            axis=0,
        )

    def branch_scores(self, feats: np.ndarray) -> np.ndarray:
        return np.einsum("abd,bd->ab", self.weights, feats).astype(np.float32)

    def scores_from_features(self, feats: np.ndarray) -> np.ndarray:
        branch_scores = self.branch_scores(feats)
        scores = self.cfg.score_scale * (branch_scores * self.branch_gains[None, :]).sum(axis=1)
        if self.cfg.branch_agreement > 0.0:
            centered = branch_scores - np.mean(branch_scores, axis=1, keepdims=True)
            agreement = -np.var(centered, axis=1)
            scores = scores + self.cfg.branch_agreement * agreement
        return scores.astype(np.float32)

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        return self.scores_from_features(self.features(row))

    def fit(self, rows: list[dict[str, Any]]) -> None:
        for _ in range(self.cfg.epochs):
            order = self.rng.permutation(len(rows))
            for idx in order:
                row = rows[int(idx)]
                target = self.answer_to_idx[row["answer"]]
                feats = self.features(row)
                scores = self.scores_from_features(feats)
                probs = softmax(scores, self.cfg.temperature)
                pred = int(np.argmax(probs))
                target_scale = self.cfg.lr * (1.0 - float(probs[target]))
                wrong_scale = self.cfg.wrong_lr * float(probs[pred])
                self.weights[target] += target_scale * feats
                if pred != target:
                    self.weights[pred] -= wrong_scale * feats
                self.weights[target] = self.normalize_answer(self.weights[target])
                if pred != target:
                    self.weights[pred] = self.normalize_answer(self.weights[pred])
                self.answer_counts[target] += 1.0

    def normalize_answer(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        return (matrix / norms).astype(np.float32)

    def state_bytes(self) -> int:
        state = {
            "weights": self.weights,
            "answer_counts": self.answer_counts,
            "cfg": asdict(self.cfg),
            "branch_gains": self.branch_gains,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


def build_answer_vocab(*splits: list[dict[str, Any]]) -> list[str]:
    answers: set[str] = set()
    for rows in splits:
        answers.update(row["answer"] for row in rows)
    return sorted(answers)


def evaluate_method(
    name: str,
    model: Any,
    splits: dict[str, list[dict[str, Any]]],
    answer_to_idx: dict[str, int],
    temperature: float,
    stores_raw_text: bool,
    method_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for split, rows in splits.items():
        metrics = evaluate_scores(rows, answer_to_idx, model.scores, temperature)
        summary_rows.append(
            {
                "method": name,
                "method_type": method_type,
                "split": split,
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "state_bytes": model.state_bytes(),
                "stores_raw_text": stores_raw_text,
                "active_features": model.active_features() if hasattr(model, "active_features") else "",
            }
        )
        for row in metrics["predictions"]:
            pred_rows.append({"method": name, "split": split, **row})
    return summary_rows, pred_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", default="en-qa1")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "babi_no_bp_qa1")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--lookup-ngrams", type=int, default=2)
    parser.add_argument("--phase-dim", type=int, default=64)
    parser.add_argument("--phase-lr", type=float, default=0.08)
    parser.add_argument("--phase-wrong-lr", type=float, default=0.03)
    parser.add_argument("--phase-epochs", type=int, default=8)
    parser.add_argument("--phase-score-scale", type=float, default=6.0)
    parser.add_argument("--phase-temperature", type=float, default=1.0)
    parser.add_argument("--branch-agreement", type=float, default=0.05)
    parser.add_argument("--role-dim", type=int, default=64)
    parser.add_argument("--role-lr", type=float, default=1.0)
    parser.add_argument("--role-score-scale", type=float, default=8.0)
    parser.add_argument("--role-carry-threshold", type=float, default=0.35)
    parser.add_argument("--attr-dim", type=int, default=64)
    parser.add_argument("--attr-lr", type=float, default=1.0)
    parser.add_argument("--attr-score-scale", type=float, default=8.0)
    parser.add_argument("--attr-confidence-threshold", type=float, default=0.05)
    parser.add_argument("--attr-statement-mode", choices=["regex", "learned", "hybrid"], default="regex")
    parser.add_argument("--attr-statement-dim", type=int, default=64)
    parser.add_argument("--attr-statement-lr", type=float, default=0.08)
    parser.add_argument("--attr-statement-epochs", type=int, default=3)
    parser.add_argument("--attr-statement-score-scale", type=float, default=6.0)
    parser.add_argument("--attr-statement-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--attr-statement-eval-limit", type=int, default=0)
    parser.add_argument("--attr-query-mode", choices=["regex", "learned", "hybrid"], default="regex")
    parser.add_argument("--attr-query-dim", type=int, default=64)
    parser.add_argument("--attr-query-lr", type=float, default=0.08)
    parser.add_argument("--attr-query-epochs", type=int, default=3)
    parser.add_argument("--attr-query-score-scale", type=float, default=6.0)
    parser.add_argument("--attr-query-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--attr-query-eval-limit", type=int, default=0)
    parser.add_argument("--role-event-mode", choices=["regex", "learned", "hybrid"], default="regex")
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--event-lr", type=float, default=0.08)
    parser.add_argument("--event-epochs", type=int, default=3)
    parser.add_argument("--event-score-scale", type=float, default=6.0)
    parser.add_argument("--event-confidence-threshold", type=float, default=0.15)
    parser.add_argument("--event-eval-limit", type=int, default=0)
    parser.add_argument("--role-query-mode", choices=["regex", "learned", "hybrid"], default="regex")
    parser.add_argument("--query-dim", type=int, default=64)
    parser.add_argument("--query-lr", type=float, default=0.08)
    parser.add_argument("--query-epochs", type=int, default=3)
    parser.add_argument("--query-score-scale", type=float, default=6.0)
    parser.add_argument("--query-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--query-eval-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config_dir = args.data_dir / args.config
    train_rows = read_jsonl(config_dir / "train.jsonl", args.train_limit or None)
    validation_rows = read_jsonl(config_dir / "validation.jsonl", args.eval_limit or None)
    test_rows = read_jsonl(config_dir / "test.jsonl", args.eval_limit or None)
    answer_vocab = build_answer_vocab(train_rows, validation_rows, test_rows)
    answer_to_idx = {answer: idx for idx, answer in enumerate(answer_vocab)}
    splits = {"train": train_rows, "validation": validation_rows, "test": test_rows}

    majority = MajorityBaseline(answer_to_idx)
    majority.fit(train_rows)

    retrieval = RawRetrievalBaseline(answer_to_idx)
    retrieval.fit(train_rows)

    lookup = HashedLookupBaseline(answer_to_idx, args.hash_bits, args.lookup_ngrams, args.seed + 17)
    lookup.fit(train_rows)

    symbolic = SymbolicLocationTracker(answer_to_idx, majority)
    event_detector = None
    if args.role_event_mode in {"learned", "hybrid"}:
        event_detector = LearnedEventDetector(
            dim=args.event_dim,
            lr=args.event_lr,
            epochs=args.event_epochs,
            score_scale=args.event_score_scale,
            seed=args.seed + 101,
            confidence_threshold=args.event_confidence_threshold,
        )
    query_detector = None
    if args.role_query_mode in {"learned", "hybrid"}:
        query_detector = LearnedQueryDetector(
            dim=args.query_dim,
            lr=args.query_lr,
            epochs=args.query_epochs,
            score_scale=args.query_score_scale,
            seed=args.seed + 211,
            confidence_threshold=args.query_confidence_threshold,
        )
    role_binding = RoleBindingStateQALearner(
        answer_to_idx,
        majority,
        event_detector=event_detector,
        event_mode=args.role_event_mode,
        query_detector=query_detector,
        query_mode=args.role_query_mode,
        dim=args.role_dim,
        lr=args.role_lr,
        score_scale=args.role_score_scale,
        carry_threshold=args.role_carry_threshold,
        seed=args.seed,
    )
    role_binding.fit(train_rows)

    attr_statement_detector = None
    if args.attr_statement_mode in {"learned", "hybrid"}:
        attr_statement_detector = LearnedAttributeStatementDetector(
            dim=args.attr_statement_dim,
            lr=args.attr_statement_lr,
            epochs=args.attr_statement_epochs,
            score_scale=args.attr_statement_score_scale,
            seed=args.seed + 317,
            confidence_threshold=args.attr_statement_confidence_threshold,
        )
    attr_query_detector = None
    if args.attr_query_mode in {"learned", "hybrid"}:
        attr_query_detector = LearnedAttributeQueryDetector(
            dim=args.attr_query_dim,
            lr=args.attr_query_lr,
            epochs=args.attr_query_epochs,
            score_scale=args.attr_query_score_scale,
            seed=args.seed + 331,
            confidence_threshold=args.attr_query_confidence_threshold,
        )
    attribute_binding = AttributeBindingStateQALearner(
        answer_to_idx,
        majority,
        statement_detector=attr_statement_detector,
        statement_mode=args.attr_statement_mode,
        query_detector=attr_query_detector,
        query_mode=args.attr_query_mode,
        dim=args.attr_dim,
        lr=args.attr_lr,
        score_scale=args.attr_score_scale,
        confidence_threshold=args.attr_confidence_threshold,
        seed=args.seed + 307,
    )
    attribute_binding.fit(train_rows)

    phase_cfg = PhaseQAConfig(
        phase_dim=args.phase_dim,
        lr=args.phase_lr,
        wrong_lr=args.phase_wrong_lr,
        epochs=args.phase_epochs,
        score_scale=args.phase_score_scale,
        temperature=args.phase_temperature,
        branch_agreement=args.branch_agreement,
        seed=args.seed,
    )
    phase = PhaseDendriticQALearner(answer_to_idx, phase_cfg)
    phase.fit(train_rows)

    all_summary: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    if event_detector is not None:
        for split, rows in splits.items():
            metrics = event_detector.event_metrics(rows, args.event_eval_limit)
            event_rows.append(
                {
                    "split": split,
                    "sentences": metrics["sentences"],
                    "event_accuracy": metrics["event_accuracy"],
                    "person_accuracy": metrics["person_accuracy"],
                    "object_accuracy": metrics["object_accuracy"],
                    "location_accuracy": metrics["location_accuracy"],
                    "person_total": metrics["person_total"],
                    "object_total": metrics["object_total"],
                    "location_total": metrics["location_total"],
                    "state_bytes": event_detector.state_bytes(),
                }
            )
    query_rows: list[dict[str, Any]] = []
    if query_detector is not None:
        for split, rows in splits.items():
            metrics = query_detector.query_metrics(rows, args.query_eval_limit)
            query_rows.append(
                {
                    "split": split,
                    "questions": metrics["questions"],
                    "query_accuracy": metrics["query_accuracy"],
                    "subject_accuracy": metrics["subject_accuracy"],
                    "destination_accuracy": metrics["destination_accuracy"],
                    "subject_total": metrics["subject_total"],
                    "destination_total": metrics["destination_total"],
                    "state_bytes": query_detector.state_bytes(),
                }
            )
    attr_statement_rows: list[dict[str, Any]] = []
    if attr_statement_detector is not None:
        for split, rows in splits.items():
            metrics = attr_statement_detector.statement_metrics(rows, args.attr_statement_eval_limit)
            attr_statement_rows.append(
                {
                    "split": split,
                    "sentences": metrics["sentences"],
                    "event_accuracy": metrics["event_accuracy"],
                    "entity_accuracy": metrics["entity_accuracy"],
                    "value_accuracy": metrics["value_accuracy"],
                    "entity_total": metrics["entity_total"],
                    "value_total": metrics["value_total"],
                    "state_bytes": attr_statement_detector.state_bytes(),
                }
            )
    attr_query_rows: list[dict[str, Any]] = []
    if attr_query_detector is not None:
        for split, rows in splits.items():
            metrics = attr_query_detector.query_metrics(rows, args.attr_query_eval_limit)
            attr_query_rows.append(
                {
                    "split": split,
                    "questions": metrics["questions"],
                    "query_accuracy": metrics["query_accuracy"],
                    "subject_accuracy": metrics["subject_accuracy"],
                    "subject_total": metrics["subject_total"],
                    "state_bytes": attr_query_detector.state_bytes(),
                }
            )
    methods = [
        ("majority_no_memory", majority, False, "baseline"),
        ("raw_lexical_retrieval", retrieval, True, "diagnostic_raw_retrieval"),
        ("hashed_lookup_diagnostic", lookup, False, "diagnostic_statistical_lookup"),
        ("symbolic_location_tracker", symbolic, False, "diagnostic_symbolic_upper_bound"),
        ("role_binding_state_no_bp", role_binding, False, "pure_no_bp_state_binding"),
        ("attribute_binding_state_no_bp", attribute_binding, False, "pure_no_bp_attribute_binding"),
        ("phase_dendritic_no_bp", phase, False, "pure_no_bp_neural"),
    ]
    for name, model, stores_raw, method_type in methods:
        summary, predictions = evaluate_method(
            name,
            model,
            splits,
            answer_to_idx,
            args.phase_temperature,
            stores_raw,
            method_type,
        )
        all_summary.extend(summary)
        all_predictions.extend(predictions)

    write_csv(args.out_dir / "summary.csv", all_summary)
    write_csv(args.out_dir / "predictions_sample.csv", all_predictions)
    write_csv(args.out_dir / "event_metrics.csv", event_rows)
    write_csv(args.out_dir / "query_metrics.csv", query_rows)
    write_csv(args.out_dir / "attribute_statement_metrics.csv", attr_statement_rows)
    write_csv(args.out_dir / "attribute_query_metrics.csv", attr_query_rows)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "answer_vocab": answer_vocab,
                "phase_cfg": asdict(phase_cfg),
                "role_binding_cfg": {
                    "dim": args.role_dim,
                    "lr": args.role_lr,
                    "score_scale": args.role_score_scale,
                    "carry_threshold": args.role_carry_threshold,
                    "event_mode": args.role_event_mode,
                    "query_mode": args.role_query_mode,
                    "seed": args.seed,
                },
                "attribute_binding_cfg": {
                    "dim": args.attr_dim,
                    "lr": args.attr_lr,
                    "score_scale": args.attr_score_scale,
                    "confidence_threshold": args.attr_confidence_threshold,
                    "statement_mode": args.attr_statement_mode,
                    "query_mode": args.attr_query_mode,
                    "seed": args.seed + 307,
                },
                "attribute_statement_detector_cfg": {
                    "dim": args.attr_statement_dim,
                    "lr": args.attr_statement_lr,
                    "epochs": args.attr_statement_epochs,
                    "score_scale": args.attr_statement_score_scale,
                    "confidence_threshold": args.attr_statement_confidence_threshold,
                    "seed": args.seed + 317,
                    "state_bytes": attr_statement_detector.state_bytes()
                    if attr_statement_detector is not None
                    else 0,
                },
                "attribute_query_detector_cfg": {
                    "dim": args.attr_query_dim,
                    "lr": args.attr_query_lr,
                    "epochs": args.attr_query_epochs,
                    "score_scale": args.attr_query_score_scale,
                    "confidence_threshold": args.attr_query_confidence_threshold,
                    "seed": args.seed + 331,
                    "state_bytes": attr_query_detector.state_bytes() if attr_query_detector is not None else 0,
                },
                "event_detector_cfg": {
                    "dim": args.event_dim,
                    "lr": args.event_lr,
                    "epochs": args.event_epochs,
                    "score_scale": args.event_score_scale,
                    "confidence_threshold": args.event_confidence_threshold,
                    "seed": args.seed + 101,
                    "state_bytes": event_detector.state_bytes() if event_detector is not None else 0,
                },
                "query_detector_cfg": {
                    "dim": args.query_dim,
                    "lr": args.query_lr,
                    "epochs": args.query_epochs,
                    "score_scale": args.query_score_scale,
                    "confidence_threshold": args.query_confidence_threshold,
                    "seed": args.seed + 211,
                    "state_bytes": query_detector.state_bytes() if query_detector is not None else 0,
                },
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "test_rows": len(test_rows),
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
                f"  {row['method']}: test_acc={row['accuracy']:.3f} "
                f"test_loss={row['loss']:.3f} bytes={row['state_bytes']}"
            )
    print(f"wrote {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
