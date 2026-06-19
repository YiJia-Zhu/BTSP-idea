#!/usr/bin/env python3
"""
Delayed answer-credit experiment for no-BP bAbI role binding.

R097 showed that the learned event/query front-end is fragile under strong
paraphrase.  This script tests a stricter training signal: event/query detector
updates are driven by final QA answer error instead of sentence-local parser
labels.  Structural-label models are kept only as baselines/initializers.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from babi_no_bp_qa_experiment import (
    DEFAULT_DATA_DIR,
    EVENT_TO_IDX,
    EVENT_TYPES,
    LOCATION_WORDS,
    QUERY_TO_IDX,
    QUERY_TYPES,
    MajorityBaseline,
    RoleBindingStateQALearner,
    build_answer_vocab,
    evaluate_method,
    normalize_vector,
    read_jsonl,
    softmax,
    tokens,
    write_csv,
)
from babi_paraphrase_stress_experiment import (
    StressEventDetector,
    StressQueryDetector,
    build_learned_role_model,
    paraphrase_rows,
)


SCRIPT_DIR = Path(__file__).resolve().parent

STOPWORDS = {
    "a",
    "an",
    "and",
    "back",
    "before",
    "contains",
    "held",
    "in",
    "is",
    "of",
    "room",
    "the",
    "there",
    "to",
    "was",
    "where",
    "which",
}

DEFAULT_LEARNED_NEAR_RISK_THRESHOLD_CANDIDATES = [-0.8, -0.6, -0.4, -0.2, -0.1, 0.0, 0.1]


def answer_objective(scores: np.ndarray, target_idx: int, temperature: float) -> float:
    probs = softmax(scores.astype(np.float32, copy=False), temperature)
    return math.log(float(probs[target_idx]) + 1e-9)


def answer_loss(scores: np.ndarray, target_idx: int, temperature: float) -> float:
    probs = softmax(scores.astype(np.float32, copy=False), temperature)
    return -math.log(float(probs[target_idx]) + 1e-9)


def event_key(event: dict[str, str | None]) -> tuple[str | None, str | None, str | None, str | None]:
    return (event.get("event"), event.get("person"), event.get("object"), event.get("location"))


def query_key(query: dict[str, str | None]) -> tuple[str | None, str | None, str | None]:
    return (query.get("query"), query.get("subject"), query.get("destination"))


class CreditEventDetector(StressEventDetector):
    """Stress detector with third-factor answer-credit updates."""

    def credit_update(self, sentence: str, target_event: dict[str, str | None], scale: float) -> float:
        feature = self.sentence_feature(sentence)
        raw_scores = self.event_weights @ feature
        pred_idx = int(np.argmax(raw_scores))
        target_idx = EVENT_TO_IDX[str(target_event["event"])]
        applied = 0.0
        eta = self.lr * max(float(scale), 0.05)
        if pred_idx != target_idx:
            self.event_weights[target_idx] += eta * feature
            self.event_weights[pred_idx] -= 0.5 * eta * feature
            self.event_weights[target_idx] = normalize_vector(self.event_weights[target_idx])
            self.event_weights[pred_idx] = normalize_vector(self.event_weights[pred_idx])
            applied += float(eta) * float(np.linalg.norm(feature))
        self.update_prototype(
            self.person_prototypes,
            self.person_counts,
            target_event.get("person"),
            self.slot_feature(sentence, "person"),
        )
        self.update_prototype(
            self.object_prototypes,
            self.object_counts,
            target_event.get("object"),
            self.slot_feature(sentence, "object"),
        )
        self.update_prototype(
            self.location_prototypes,
            self.location_counts,
            target_event.get("location"),
            self.slot_feature(sentence, "location"),
        )
        return applied


class CreditQueryDetector(StressQueryDetector):
    """Stress query detector with answer-credit updates."""

    def __init__(
        self,
        *args: Any,
        before_relation_slot_features: bool = False,
        enable_query_subject_wta: bool = False,
        query_subject_wta_bonus: float = 1.0,
        query_subject_wta_min_margin: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.before_relation_slot_features = before_relation_slot_features
        self.enable_query_subject_wta = enable_query_subject_wta
        self.query_subject_wta_bonus = query_subject_wta_bonus
        self.query_subject_wta_min_margin = query_subject_wta_min_margin

    def slot_feature(self, question: str, slot: Any) -> np.ndarray:
        feature = super().slot_feature(question, slot)
        if not self.before_relation_slot_features:
            return feature
        words = tokens(question)
        extra = np.zeros(self.dim, dtype=np.float32)
        if slot == "subject":
            for idx, word in enumerate(words):
                if idx > 0 and words[idx - 1] in {"held", "contains"}:
                    extra += 1.5 * self.token_code(word, "slot-before-subject-after-relation")
                if idx > 1 and words[idx - 1] == "the" and words[idx - 2] in {"held", "contains"}:
                    extra += 2.0 * self.token_code(word, "slot-before-subject-after-relation-the")
        elif slot == "destination":
            for idx, word in enumerate(words):
                if idx > 0 and words[idx - 1] == "before":
                    extra += 1.0 * self.token_code(word, "slot-before-destination-after-before")
                if idx > 1 and words[idx - 1] == "the" and words[idx - 2] == "before":
                    extra += 1.5 * self.token_code(word, "slot-before-destination-after-before-the")
        if not np.any(extra):
            return feature
        return normalize_vector(feature + extra)

    def subject_wta_candidate_labels(self, question: str) -> list[str]:
        words = set(tokens(question))
        candidates: list[str] = []
        for label in self.subject_prototypes:
            label_tokens = tokens(label)
            if not label_tokens:
                continue
            if any(token in words and token not in STOPWORDS and token not in LOCATION_WORDS for token in label_tokens):
                candidates.append(label)
        return sorted(set(candidates))

    def query_subject_wta(self, question: str) -> tuple[str | None, float]:
        candidates = self.subject_wta_candidate_labels(question)
        if not candidates:
            return None, 0.0
        words = set(tokens(question))
        feature = self.slot_feature(question, "subject")
        scored: list[tuple[float, str]] = []
        for label in candidates:
            proto = self.subject_prototypes[label]
            lexical_bonus = self.query_subject_wta_bonus if any(token in words for token in tokens(label)) else 0.0
            scored.append((float(proto @ feature) + lexical_bonus, label))
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_label = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        if margin < self.query_subject_wta_min_margin:
            return None, margin
        return best_label, margin

    def predict(self, question: str) -> dict[str, str | float | None]:
        pred = super().predict(question)
        if not self.enable_query_subject_wta:
            return pred
        words = set(tokens(question))
        if pred.get("query") != "where_before" and "before" not in words:
            return pred
        subject, margin = self.query_subject_wta(question)
        if subject is None:
            return pred
        pred["subject"] = subject
        pred["subject_confidence"] = max(float(pred.get("subject_confidence") or 0.0), float(margin))
        pred["subject_wta_margin"] = float(margin)
        return pred

    def credit_update(self, question: str, target_query: dict[str, str | None], scale: float) -> float:
        feature = self.question_feature(question)
        raw_scores = self.query_weights @ feature
        pred_idx = int(np.argmax(raw_scores))
        target_idx = QUERY_TO_IDX[str(target_query["query"])]
        applied = 0.0
        eta = self.lr * max(float(scale), 0.05)
        if pred_idx != target_idx:
            self.query_weights[target_idx] += eta * feature
            self.query_weights[pred_idx] -= 0.5 * eta * feature
            self.query_weights[target_idx] = normalize_vector(self.query_weights[target_idx])
            self.query_weights[pred_idx] = normalize_vector(self.query_weights[pred_idx])
            applied += float(eta) * float(np.linalg.norm(feature))
        self.update_prototype(
            self.subject_prototypes,
            self.subject_counts,
            target_query.get("subject"),
            self.slot_feature(question, "subject"),
        )
        self.update_prototype(
            self.destination_prototypes,
            self.destination_counts,
            target_query.get("destination"),
            self.slot_feature(question, "destination"),
        )
        return applied


def apply_event_to_state(
    model: RoleBindingStateQALearner,
    state: dict[str, Any],
    event: dict[str, str | None],
) -> None:
    event_type = event.get("event")
    if event_type == "move":
        person = event.get("person")
        location = event.get("location")
        if person is not None and location is not None:
            model.set_person_location(state, person, location)
        return
    if event_type == "pickup":
        person = event.get("person")
        obj = event.get("object")
        if person is None or obj is None:
            return
        state["people"].add(person)
        state["objects"].add(obj)
        model.associate(state["object_owner"], model.code(obj, "object"), model.code(person, "person"))
        location = model.person_location(state, person)
        if location is not None:
            model.set_object_location(state, obj, location)
        return
    if event_type == "drop":
        person = event.get("person")
        obj = event.get("object")
        if person is None or obj is None:
            return
        state["people"].add(person)
        state["objects"].add(obj)
        location = model.person_location(state, person)
        if location is not None:
            model.set_object_location(state, obj, location)
        model.associate(state["object_owner"], model.code(obj, "object"), model.null_owner_code)


def scores_for_query(
    model: RoleBindingStateQALearner,
    state: dict[str, Any],
    row: dict[str, Any],
    query: dict[str, str | None],
) -> np.ndarray:
    if query.get("query") == "where_before":
        obj = query.get("subject")
        destination = query.get("destination")
        if obj is None or destination is None or destination not in model.location_codes:
            base_scores = model.fallback.scores(row)
            blender = getattr(model, "blend_before_credit_scores", None)
            if callable(blender):
                return blender(state, row, query, base_scores)
            return base_scores
        key = model.bind(model.code(obj, "object"), model.location_codes[destination])
        value, conf = model.retrieve(state["before_location"], key)
        if conf > 0.05:
            _, _, scores = model.decode_location(value)
            base_scores = scores
        else:
            base_scores = model.fallback.scores(row)
        blender = getattr(model, "blend_before_credit_scores", None)
        if callable(blender):
            return blender(state, row, query, base_scores)
        return base_scores

    subject = query.get("subject")
    if subject is None:
        return model.fallback.scores(row)
    if subject in state["objects"]:
        value, conf = model.retrieve(state["object_location"], model.code(subject, "object"))
    else:
        value, conf = model.retrieve(state["person_location"], model.code(subject, "person"))
    if conf <= 0.05:
        return model.fallback.scores(row)
    _, _, scores = model.decode_location(value)
    return scores


def scores_with_overrides(
    model: RoleBindingStateQALearner,
    row: dict[str, Any],
    event_overrides: dict[int, dict[str, str | None]] | None = None,
    query_override: dict[str, str | None] | None = None,
) -> np.ndarray:
    state = model.new_state()
    overrides = event_overrides or {}
    for idx, item in enumerate(row["context"]):
        event = overrides.get(idx)
        if event is None:
            event = model.detect_event(str(item["text"]))
        apply_event_to_state(model, state, event)
    query = query_override if query_override is not None else model.detect_query(str(row["question"]))
    return scores_for_query(model, state, row, query)


def predicted_event_sequence(model: RoleBindingStateQALearner, row: dict[str, Any]) -> list[dict[str, str | None]]:
    return [model.detect_event(str(item["text"])) for item in row["context"]]


def state_from_events(
    model: RoleBindingStateQALearner,
    row: dict[str, Any],
    events: list[dict[str, str | None]],
    replace_idx: int | None = None,
    replacement: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    state = model.new_state()
    for idx, event in enumerate(events):
        if replace_idx is not None and idx == replace_idx and replacement is not None:
            apply_event_to_state(model, state, replacement)
        else:
            apply_event_to_state(model, state, event)
    return state


def scores_from_cached_events(
    model: RoleBindingStateQALearner,
    row: dict[str, Any],
    events: list[dict[str, str | None]],
    query: dict[str, str | None],
    replace_idx: int | None = None,
    replacement: dict[str, str | None] | None = None,
) -> np.ndarray:
    state = state_from_events(model, row, events, replace_idx, replacement)
    return scores_for_query(model, state, row, query)


class BeforeLocationCreditRoleBindingStateQALearner(RoleBindingStateQALearner):
    """Adds a persistent local before-location correction matrix.

    The update key is an eligibility-style relation state derived from the
    current story state and the detected `(object, destination)` query.  The
    matrix is updated only from final answer credit; it stores no raw text.
    """

    def __init__(
        self,
        *args: Any,
        before_credit_lr: float = 0.5,
        before_credit_weight: float = 1.0,
        before_credit_threshold: float = 0.05,
        before_credit_before_weight: float = 0.75,
        before_credit_current_weight: float = 0.35,
        before_credit_gate_mode: str = "always",
        before_credit_gate_margin: float = 0.0,
        before_credit_confidence_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.before_credit_lr = before_credit_lr
        self.before_credit_weight = before_credit_weight
        self.before_credit_threshold = before_credit_threshold
        self.before_credit_before_weight = before_credit_before_weight
        self.before_credit_current_weight = before_credit_current_weight
        self.before_credit_gate_mode = before_credit_gate_mode
        self.before_credit_gate_margin = before_credit_gate_margin
        self.before_credit_confidence_scale = before_credit_confidence_scale
        self.before_credit_matrix = np.zeros((self.dim, self.dim), dtype=np.float32)

    def before_relation_key(
        self,
        state: dict[str, Any],
        obj: str,
        destination: str,
    ) -> np.ndarray:
        obj_code = self.code(obj, "object")
        destination_code = self.location_codes[destination]
        relation_key = self.bind(obj_code, destination_code)
        key = relation_key.astype(np.float32, copy=True)
        before_value, before_conf = self.retrieve(state["before_location"], relation_key)
        if before_conf > self.before_credit_threshold:
            key += self.before_credit_before_weight * before_value
        current_value, current_conf = self.retrieve(state["object_location"], obj_code)
        if current_conf > self.before_credit_threshold:
            key += self.before_credit_current_weight * current_value
        return normalize_vector(key)

    def location_score_margin(self, scores: np.ndarray) -> float:
        loc_scores = [float(scores[self.answer_to_idx[location]]) for location in self.answer_locations]
        if len(loc_scores) < 2:
            return float("inf")
        loc_scores.sort(reverse=True)
        return loc_scores[0] - loc_scores[1]

    def top_location_from_scores(self, scores: np.ndarray) -> str | None:
        if not self.answer_locations:
            return None
        best_location: str | None = None
        best_score = -1e9
        for location in self.answer_locations:
            score = float(scores[self.answer_to_idx[location]])
            if score > best_score:
                best_score = score
                best_location = location
        return best_location

    def before_credit_gate_weight(self, confidence: float) -> float:
        if self.before_credit_gate_mode != "confidence":
            return 1.0
        scale = max(float(self.before_credit_confidence_scale), 1e-6)
        return float(np.clip((float(confidence) - self.before_credit_threshold) / scale, 0.0, 1.0))

    def blend_before_credit_scores(
        self,
        state: dict[str, Any],
        row: dict[str, Any],
        query: dict[str, str | None],
        base_scores: np.ndarray,
    ) -> np.ndarray:
        del row
        obj = query.get("subject")
        destination = query.get("destination")
        if obj is None or destination is None or destination not in self.location_codes:
            return base_scores
        if self.before_credit_gate_mode == "low_margin":
            margin = self.location_score_margin(base_scores)
            if margin >= self.before_credit_gate_margin:
                return base_scores
        key = self.before_relation_key(state, obj, destination)
        value, conf = self.retrieve(self.before_credit_matrix, key)
        if conf <= self.before_credit_threshold:
            return base_scores
        _, _, credit_scores = self.decode_location(value)
        if self.before_credit_gate_mode == "agree_top":
            if self.top_location_from_scores(base_scores) != self.top_location_from_scores(credit_scores):
                return base_scores
        gate_weight = self.before_credit_gate_weight(conf)
        if gate_weight <= 0.0:
            return base_scores
        return (base_scores + self.before_credit_weight * gate_weight * credit_scores).astype(np.float32)

    def before_credit_update(
        self,
        state: dict[str, Any],
        query: dict[str, str | None],
        target_answer: str,
        scale: float,
    ) -> float:
        obj = query.get("subject")
        destination = query.get("destination")
        if (
            query.get("query") != "where_before"
            or obj is None
            or destination is None
            or destination not in self.location_codes
            or target_answer not in self.location_codes
        ):
            return 0.0
        key = self.before_relation_key(state, obj, destination)
        target = self.location_codes[target_answer]
        pred = self.before_credit_matrix @ key
        eta = self.before_credit_lr * max(float(scale), 0.05)
        self.before_credit_matrix += eta * np.outer(target - pred, key).astype(np.float32)
        return float(eta) * float(np.linalg.norm(target - pred))

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        state = self.read_context(row)
        query = self.detect_query(str(row["question"]))
        return scores_for_query(self, state, row, query)

    def state_bytes(self) -> int:
        state = {
            "before_credit_lr": self.before_credit_lr,
            "before_credit_weight": self.before_credit_weight,
            "before_credit_threshold": self.before_credit_threshold,
            "before_credit_gate_mode": self.before_credit_gate_mode,
            "before_credit_gate_margin": self.before_credit_gate_margin,
            "before_credit_confidence_scale": self.before_credit_confidence_scale,
            "before_credit_matrix": self.before_credit_matrix,
        }
        return super().state_bytes() + self.before_credit_matrix.nbytes + len(
            pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        )


def candidate_object_tokens(words: list[str], object_vocab: set[str]) -> list[str]:
    candidates: list[str] = []
    for idx, word in enumerate(words):
        if word in LOCATION_WORDS or word in STOPWORDS:
            continue
        if idx > 0 and words[idx - 1] == "the":
            candidates.append(word)
        elif word in object_vocab:
            candidates.append(word)
    return sorted(set(candidates))[:3]


def candidate_events(sentence: str, object_vocab: set[str]) -> list[dict[str, str | None]]:
    words = tokens(sentence)
    if not words:
        return [{"event": "none", "person": None, "object": None, "location": None}]
    person = words[0] if words[0] not in STOPWORDS and words[0] not in LOCATION_WORDS else None
    locations = [word for word in words if word in LOCATION_WORDS]
    objects = candidate_object_tokens(words, object_vocab)
    candidates = [{"event": "none", "person": None, "object": None, "location": None}]
    if person is not None:
        for location in locations:
            candidates.append({"event": "move", "person": person, "object": None, "location": location})
        for obj in objects:
            candidates.append({"event": "pickup", "person": person, "object": obj, "location": None})
            candidates.append({"event": "drop", "person": person, "object": obj, "location": None})
    unique: dict[tuple[str | None, str | None, str | None, str | None], dict[str, str | None]] = {}
    for candidate in candidates:
        unique[event_key(candidate)] = candidate
    return list(unique.values())


def query_subject_candidates(question: str, object_vocab: set[str]) -> list[str]:
    words = tokens(question)
    candidates = [
        word
        for word in words
        if word not in STOPWORDS and word not in LOCATION_WORDS and (word in object_vocab or len(word) > 2)
    ]
    return sorted(set(candidates))[:4]


def candidate_queries(question: str, object_vocab: set[str]) -> list[dict[str, str | None]]:
    words = tokens(question)
    subjects = query_subject_candidates(question, object_vocab)
    destinations = [word for word in words if word in LOCATION_WORDS]
    candidates: list[dict[str, str | None]] = []
    for subject in subjects:
        candidates.append({"query": "where_is", "subject": subject, "destination": None})
        for destination in destinations:
            candidates.append({"query": "where_before", "subject": subject, "destination": destination})
    if not candidates:
        candidates.append({"query": "where_is", "subject": None, "destination": None})
    unique: dict[tuple[str | None, str | None, str | None], dict[str, str | None]] = {}
    for candidate in candidates:
        unique[query_key(candidate)] = candidate
    return list(unique.values())


def credit_sentence_indices(
    row: dict[str, Any],
    object_vocab: set[str],
    max_sentences: int,
) -> list[int]:
    if max_sentences <= 0 or max_sentences >= len(row["context"]):
        return list(range(len(row["context"])))
    question_terms = set(query_subject_candidates(str(row["question"]), object_vocab))
    answer = str(row["answer"])
    scored: list[tuple[int, int]] = []
    for idx, item in enumerate(row["context"]):
        words = set(tokens(str(item["text"])))
        score = 0
        if answer in words:
            score += 4
        if words & question_terms:
            score += 3
        if words & object_vocab:
            score += 2
        if words & LOCATION_WORDS:
            score += 1
        if score > 0:
            scored.append((score, idx))
    if not scored:
        return list(range(min(max_sentences, len(row["context"]))))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return sorted(idx for _, idx in scored[:max_sentences])


def build_object_vocab(rows: list[dict[str, Any]]) -> set[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for word in query_subject_candidates(str(row["question"]), set()):
            counts[word] += 2
        for item in row["context"]:
            words = tokens(str(item["text"]))
            for idx, word in enumerate(words):
                if word in LOCATION_WORDS or word in STOPWORDS:
                    continue
                if idx > 0 and words[idx - 1] == "the":
                    counts[word] += 1
    return {word for word, count in counts.items() if count >= 1}


class AnswerCreditTrainer:
    def __init__(
        self,
        model: RoleBindingStateQALearner,
        answer_to_idx: dict[str, int],
        object_vocab: set[str],
        temperature: float,
        min_gain: float,
        max_event_updates: int,
        max_credit_sentences: int,
        candidate_cache_mode: str,
        error_only: bool,
        max_credit_scale: float,
        disable_query_credit: bool,
        disable_event_credit: bool,
        enable_before_relation_credit: bool,
        restrict_before_query_credit: bool,
        protect_query_credit_confidence: float,
        seed: int,
    ) -> None:
        self.model = model
        self.answer_to_idx = answer_to_idx
        self.object_vocab = object_vocab
        self.temperature = temperature
        self.min_gain = min_gain
        self.max_event_updates = max_event_updates
        self.max_credit_sentences = max_credit_sentences
        self.candidate_cache_mode = candidate_cache_mode
        self.error_only = error_only
        self.max_credit_scale = max_credit_scale
        self.disable_query_credit = disable_query_credit
        self.disable_event_credit = disable_event_credit
        self.enable_before_relation_credit = enable_before_relation_credit
        self.restrict_before_query_credit = restrict_before_query_credit
        self.protect_query_credit_confidence = protect_query_credit_confidence
        self.rng = np.random.default_rng(seed)

    def credit_scale(self, gain: float) -> float:
        if self.max_credit_scale > 0.0:
            return min(float(gain), self.max_credit_scale)
        return float(gain)

    def current_objective(self, row: dict[str, Any]) -> float:
        target = self.answer_to_idx[row["answer"]]
        scores = scores_with_overrides(self.model, row)
        return answer_objective(scores, target, self.temperature)

    def query_candidates_for_row(self, row: dict[str, Any]) -> list[dict[str, str | None]]:
        candidates = candidate_queries(str(row["question"]), self.object_vocab)
        if not self.restrict_before_query_credit:
            return candidates
        question_words = set(tokens(str(row["question"])))
        if "before" not in question_words or not (question_words & LOCATION_WORDS):
            return candidates
        before_candidates = [query for query in candidates if query.get("query") == "where_before"]
        return before_candidates or candidates

    def should_protect_query_update(
        self,
        query_detector: CreditQueryDetector,
        question: str,
        target_query: dict[str, str | None],
    ) -> bool:
        if self.protect_query_credit_confidence <= 0.0:
            return False
        pred = query_detector.predict(question)
        if (
            pred.get("query") != target_query.get("query")
            or pred.get("subject") != target_query.get("subject")
            or pred.get("destination") != target_query.get("destination")
        ):
            return False
        subject_conf = float(pred.get("subject_confidence") or 0.0)
        destination_conf = float(pred.get("destination_confidence") or 0.0)
        return min(subject_conf, destination_conf) >= self.protect_query_credit_confidence

    def before_relation_credit_step(
        self,
        row: dict[str, Any],
        query: dict[str, str | None],
        target: int,
        current: float,
        cached_events: list[dict[str, str | None]],
        use_cached: bool,
        cached_state: dict[str, Any] | None = None,
    ) -> tuple[float, float, bool]:
        updater = getattr(self.model, "before_credit_update", None)
        matrix = getattr(self.model, "before_credit_matrix", None)
        if not self.enable_before_relation_credit or not callable(updater) or matrix is None:
            return current, 0.0, False
        if query.get("query") != "where_before":
            return current, 0.0, False
        if use_cached:
            state = cached_state if cached_state is not None else state_from_events(self.model, row, cached_events)
        else:
            state = self.model.read_context(row)
        old_matrix = matrix.copy()
        scale = self.credit_scale(max(-current, 0.0))
        update_norm = updater(state, query, str(row["answer"]), scale)
        if update_norm <= 0.0:
            return current, 0.0, False
        if use_cached:
            after_scores = scores_for_query(self.model, state, row, query)
        else:
            after_scores = scores_with_overrides(self.model, row, query_override=query)
        after = answer_objective(after_scores, target, self.temperature)
        if after - current <= self.min_gain:
            matrix[...] = old_matrix
            return current, 0.0, False
        return after, update_norm, True

    def train(self, rows: list[dict[str, Any]], epochs: int) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        if self.model.event_detector is None or self.model.query_detector is None:
            return metrics
        event_detector = self.model.event_detector
        query_detector = self.model.query_detector
        if not isinstance(event_detector, CreditEventDetector) or not isinstance(query_detector, CreditQueryDetector):
            return metrics

        for epoch in range(epochs):
            order = self.rng.permutation(len(rows))
            query_updates = 0
            event_updates = 0
            before_relation_updates = 0
            total_gain = 0.0
            update_norm = 0.0
            before_update_norm = 0.0
            skipped_correct = 0
            for raw_idx in order:
                row = rows[int(raw_idx)]
                target = self.answer_to_idx[row["answer"]]
                use_cached = self.candidate_cache_mode == "cached"
                cached_events: list[dict[str, str | None]] = []
                cached_query: dict[str, str | None] = {"query": "where_is", "subject": None, "destination": None}
                cached_state: dict[str, Any] | None = None
                if use_cached:
                    cached_events = predicted_event_sequence(self.model, row)
                    cached_query = self.model.detect_query(str(row["question"]))
                    cached_state = state_from_events(self.model, row, cached_events)
                    current_scores = scores_for_query(self.model, cached_state, row, cached_query)
                else:
                    current_scores = scores_with_overrides(self.model, row)
                current = answer_objective(current_scores, target, self.temperature)
                if self.error_only and int(np.argmax(current_scores)) == target:
                    skipped_correct += 1
                    continue

                if not self.disable_query_credit:
                    best_query: dict[str, str | None] | None = None
                    best_query_obj = current
                    for query in self.query_candidates_for_row(row):
                        if use_cached:
                            objective = answer_objective(
                                scores_for_query(self.model, cached_state or state_from_events(self.model, row, cached_events), row, query),
                                target,
                                self.temperature,
                            )
                        else:
                            objective = answer_objective(
                                scores_with_overrides(self.model, row, query_override=query),
                                target,
                                self.temperature,
                            )
                        if objective > best_query_obj:
                            best_query_obj = objective
                            best_query = query
                    if best_query is not None and best_query_obj - current > self.min_gain:
                        gain = best_query_obj - current
                        if not self.should_protect_query_update(query_detector, str(row["question"]), best_query):
                            update_norm += query_detector.credit_update(
                                str(row["question"]), best_query, self.credit_scale(gain)
                            )
                            query_updates += 1
                            total_gain += gain
                        current = best_query_obj
                        if use_cached:
                            cached_query = best_query

                before_current, before_norm, before_applied = self.before_relation_credit_step(
                    row,
                    cached_query if use_cached else self.model.detect_query(str(row["question"])),
                    target,
                    current,
                    cached_events,
                    use_cached,
                    cached_state,
                )
                if before_applied:
                    before_gain = before_current - current
                    before_relation_updates += 1
                    before_update_norm += before_norm
                    total_gain += before_gain
                    current = before_current

                candidate_indices = credit_sentence_indices(row, self.object_vocab, self.max_credit_sentences)
                event_update_budget = 0 if self.disable_event_credit else self.max_event_updates
                for _ in range(event_update_budget):
                    best_idx = -1
                    best_event: dict[str, str | None] | None = None
                    best_event_obj = current
                    for sent_idx in candidate_indices:
                        item = row["context"][sent_idx]
                        sentence = str(item["text"])
                        for event in candidate_events(sentence, self.object_vocab):
                            if use_cached:
                                objective = answer_objective(
                                    scores_from_cached_events(
                                        self.model,
                                        row,
                                        cached_events,
                                        cached_query,
                                        replace_idx=sent_idx,
                                        replacement=event,
                                    ),
                                    target,
                                    self.temperature,
                                )
                            else:
                                objective = answer_objective(
                                    scores_with_overrides(self.model, row, event_overrides={sent_idx: event}),
                                    target,
                                    self.temperature,
                                )
                            if objective > best_event_obj:
                                best_event_obj = objective
                                best_event = event
                                best_idx = sent_idx
                    if best_event is None or best_event_obj - current <= self.min_gain:
                        break
                    sentence = str(row["context"][best_idx]["text"])
                    gain = best_event_obj - current
                    update_norm += event_detector.credit_update(sentence, best_event, self.credit_scale(gain))
                    event_updates += 1
                    total_gain += gain
                    current = best_event_obj
                    if use_cached:
                        cached_events[best_idx] = best_event

            metrics.append(
                {
                    "epoch": epoch,
                    "rows": len(rows),
                    "query_updates": query_updates,
                    "event_updates": event_updates,
                    "before_relation_updates": before_relation_updates,
                    "total_gain": total_gain,
                    "mean_gain_per_row": total_gain / max(len(rows), 1),
                    "update_norm": update_norm,
                    "before_update_norm": before_update_norm,
                    "skipped_correct": skipped_correct,
                    "error_only": self.error_only,
                    "max_credit_scale": self.max_credit_scale,
                    "disable_query_credit": self.disable_query_credit,
                    "disable_event_credit": self.disable_event_credit,
                    "enable_before_relation_credit": self.enable_before_relation_credit,
                    "restrict_before_query_credit": self.restrict_before_query_credit,
                    "protect_query_credit_confidence": self.protect_query_credit_confidence,
                }
            )
        return metrics


def build_credit_model(
    answer_to_idx: dict[str, int],
    majority: MajorityBaseline,
    seed_rows: list[dict[str, Any]] | None,
    credit_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    model_seed: int,
    train_credit: bool = True,
) -> tuple[RoleBindingStateQALearner, list[dict[str, Any]]]:
    event_detector = CreditEventDetector(
        dim=args.event_dim,
        lr=args.event_lr,
        epochs=args.event_epochs,
        score_scale=args.event_score_scale,
        seed=model_seed + 101,
        confidence_threshold=args.event_confidence_threshold,
    )
    query_detector = CreditQueryDetector(
        dim=args.query_dim,
        lr=args.query_lr,
        epochs=args.query_epochs,
        score_scale=args.query_score_scale,
        seed=model_seed + 211,
        confidence_threshold=args.query_confidence_threshold,
        before_relation_slot_features=args.before_relation_slot_features,
        enable_query_subject_wta=args.enable_query_subject_wta,
        query_subject_wta_bonus=args.query_subject_wta_bonus,
        query_subject_wta_min_margin=args.query_subject_wta_min_margin,
    )
    if seed_rows is not None:
        event_detector.fit(seed_rows)
        query_detector.fit(seed_rows)
    model_cls = BeforeLocationCreditRoleBindingStateQALearner if args.enable_before_relation_credit else RoleBindingStateQALearner
    model = model_cls(
        answer_to_idx,
        majority,
        event_detector=event_detector,
        event_mode="learned",
        query_detector=query_detector,
        query_mode="learned",
        dim=args.role_dim,
        lr=args.role_lr,
        score_scale=args.role_score_scale,
        carry_threshold=args.role_carry_threshold,
        seed=model_seed,
        **(
            {
                "before_credit_lr": args.before_credit_lr,
                "before_credit_weight": args.before_credit_weight,
                "before_credit_threshold": args.before_credit_threshold,
                "before_credit_before_weight": args.before_credit_before_weight,
                "before_credit_current_weight": args.before_credit_current_weight,
                "before_credit_gate_mode": args.before_credit_gate_mode,
                "before_credit_gate_margin": args.before_credit_gate_margin,
                "before_credit_confidence_scale": args.before_credit_confidence_scale,
            }
            if args.enable_before_relation_credit
            else {}
        ),
    )
    trainer = AnswerCreditTrainer(
        model,
        answer_to_idx,
        build_object_vocab(credit_rows),
        args.temperature,
        args.min_credit_gain,
        args.max_event_updates_per_row,
        args.max_credit_sentences_per_row,
        args.candidate_cache_mode,
        args.credit_error_only,
        args.max_credit_scale,
        args.disable_query_credit,
        args.disable_event_credit,
        args.enable_before_relation_credit,
        args.restrict_before_query_credit,
        args.protect_query_credit_confidence,
        model_seed + 503,
    )
    if not train_credit:
        return model, []
    return model, trainer.train(credit_rows, args.credit_epochs)


def detector_summary_rows(
    method: str,
    model: RoleBindingStateQALearner,
    splits: dict[str, list[dict[str, Any]]],
    config: str,
    train_strength: str,
    eval_strength: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if model.event_detector is None or model.query_detector is None:
        return rows
    for split, split_rows in splits.items():
        event_metrics = model.event_detector.event_metrics(split_rows, args.event_eval_limit)
        query_metrics = model.query_detector.query_metrics(split_rows, args.query_eval_limit)
        rows.append(
            {
                "method": method,
                "config": config,
                "train_strength": train_strength,
                "eval_strength": eval_strength,
                "split": split,
                "event_accuracy": event_metrics["event_accuracy"],
                "person_accuracy": event_metrics["person_accuracy"],
                "object_accuracy": event_metrics["object_accuracy"],
                "location_accuracy": event_metrics["location_accuracy"],
                "query_accuracy": query_metrics["query_accuracy"],
                "subject_accuracy": query_metrics["subject_accuracy"],
                "destination_accuracy": query_metrics["destination_accuracy"],
                "event_sentences": event_metrics["sentences"],
                "questions": query_metrics["questions"],
                "event_state_bytes": model.event_detector.state_bytes(),
                "query_state_bytes": model.query_detector.state_bytes(),
            }
        )
    return rows


def before_credit_probe_details(
    model: RoleBindingStateQALearner,
    row: dict[str, Any],
    pre_pred_idx: int | None = None,
    post_pred_idx: int | None = None,
) -> dict[str, Any]:
    if not isinstance(model, BeforeLocationCreditRoleBindingStateQALearner):
        return {"confidence": 0.0, "gate_weight": 0.0, "credit_margin": 0.0, "status": "not_before_model"}
    state = model.read_context(row)
    query = model.detect_query(str(row["question"]))
    if query.get("query") != "where_before":
        return {"confidence": 0.0, "gate_weight": 0.0, "credit_margin": 0.0, "status": "not_before_query"}
    obj = query.get("subject")
    destination = query.get("destination")
    if obj is None or destination is None or destination not in model.location_codes:
        return {"confidence": 0.0, "gate_weight": 0.0, "credit_margin": 0.0, "status": "invalid_before_query"}
    key = model.before_relation_key(state, obj, destination)
    value, confidence = model.retrieve(model.before_credit_matrix, key)
    if confidence <= model.before_credit_threshold:
        return {"confidence": confidence, "gate_weight": 0.0, "credit_margin": 0.0, "status": "below_threshold"}
    _, _, credit_scores = model.decode_location(value)
    detail: dict[str, Any] = {
        "confidence": confidence,
        "gate_weight": model.before_credit_gate_weight(confidence),
        "credit_margin": model.location_score_margin(credit_scores),
        "status": "active",
    }
    if pre_pred_idx is not None and post_pred_idx is not None:
        top_idx = int(np.argmax(credit_scores))
        top_score = float(credit_scores[top_idx])
        pre_score = float(credit_scores[int(pre_pred_idx)])
        post_score = float(credit_scores[int(post_pred_idx)])
        detail.update(
            {
                "pre_credit_score": pre_score,
                "post_credit_score": post_score,
                "post_minus_pre_credit_score": post_score - pre_score,
                "pre_credit_gap_to_top": top_score - pre_score,
                "post_credit_gap_to_top": top_score - post_score,
                "pre_is_credit_top": int(int(pre_pred_idx) == top_idx),
                "post_is_credit_top": int(int(post_pred_idx) == top_idx),
            }
        )
    return detail


def before_credit_probe(
    model: RoleBindingStateQALearner,
    row: dict[str, Any],
) -> tuple[float, float, float, str]:
    detail = before_credit_probe_details(model, row)
    return (
        float(detail["confidence"]),
        float(detail["gate_weight"]),
        float(detail["credit_margin"]),
        str(detail["status"]),
    )


def confidence_bucket(confidence: float, threshold: float) -> str:
    if confidence <= 0.0:
        return "zero"
    if confidence <= threshold:
        return "below_threshold"
    if confidence < 0.25:
        return "low"
    if confidence < 0.50:
        return "mid"
    if confidence < 0.75:
        return "high"
    return "very_high"


def gate_bucket(gate_weight: float) -> str:
    if gate_weight <= 0.0:
        return "zero"
    if gate_weight < 0.25:
        return "low"
    if gate_weight < 0.50:
        return "mid"
    if gate_weight < 0.75:
        return "high"
    return "full"


def answer_margin(scores: np.ndarray, answer_to_idx: dict[str, int]) -> float:
    if len(answer_to_idx) < 2:
        return float("inf")
    ranked = sorted((float(scores[idx]) for idx in answer_to_idx.values()), reverse=True)
    return ranked[0] - ranked[1]


def classify_flip(
    pre_correct: bool,
    post_correct: bool,
    prediction_changed: bool,
    delta_loss: float,
) -> str:
    improved = delta_loss < -1e-9
    if prediction_changed:
        if not pre_correct and post_correct:
            return "helpful_flip"
        if pre_correct and not post_correct:
            return "harmful_flip"
        return "wrong_flip_ce_improved" if improved else "wrong_flip_ce_worse"
    if pre_correct and post_correct:
        return "same_correct_ce_improved" if improved else "same_correct_ce_worse"
    return "same_wrong_ce_improved" if improved else "same_wrong_ce_worse"


def flip_diagnostic_rows(
    config: str,
    train_strength: str,
    eval_strength: str,
    split: str,
    rows: list[dict[str, Any]],
    pre_model: RoleBindingStateQALearner,
    post_model: RoleBindingStateQALearner,
    answer_to_idx: dict[str, int],
    temperature: float,
) -> list[dict[str, Any]]:
    idx_to_answer = {idx: answer for answer, idx in answer_to_idx.items()}
    threshold = float(getattr(post_model, "before_credit_threshold", 0.05))
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        target_idx = answer_to_idx[str(row["answer"])]
        pre_scores = pre_model.scores(row).astype(np.float32, copy=False)
        post_scores = post_model.scores(row).astype(np.float32, copy=False)
        pre_pred_idx = int(np.argmax(pre_scores))
        post_pred_idx = int(np.argmax(post_scores))
        pre_loss = answer_loss(pre_scores, target_idx, temperature)
        post_loss = answer_loss(post_scores, target_idx, temperature)
        pre_correct = pre_pred_idx == target_idx
        post_correct = post_pred_idx == target_idx
        changed = pre_pred_idx != post_pred_idx
        delta_loss = post_loss - pre_loss
        confidence, gate_weight, credit_margin, probe_status = before_credit_probe(post_model, row)
        pre_margin = answer_margin(pre_scores, answer_to_idx)
        post_margin = answer_margin(post_scores, answer_to_idx)
        out.append(
            {
                "config": config,
                "train_strength": train_strength,
                "eval_strength": eval_strength,
                "split": split,
                "example_index": idx,
                "target": row["answer"],
                "pre_prediction": idx_to_answer[pre_pred_idx],
                "post_prediction": idx_to_answer[post_pred_idx],
                "pre_correct": int(pre_correct),
                "post_correct": int(post_correct),
                "prediction_changed": int(changed),
                "pre_loss": pre_loss,
                "post_loss": post_loss,
                "delta_loss": delta_loss,
                "pre_margin": pre_margin,
                "post_margin": post_margin,
                "margin_delta": post_margin - pre_margin,
                "before_confidence": confidence,
                "before_gate_weight": gate_weight,
                "before_credit_margin": credit_margin,
                "probe_status": probe_status,
                "confidence_bucket": confidence_bucket(confidence, threshold),
                "gate_bucket": gate_bucket(gate_weight),
                "flip_type": classify_flip(pre_correct, post_correct, changed, delta_loss),
            }
        )
    return out


def flip_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        common = (
            str(row["config"]),
            str(row["train_strength"]),
            str(row["eval_strength"]),
            str(row["split"]),
        )
        groups[(*common, "all")].append(row)
        groups[(*common, str(row["confidence_bucket"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (config, train_strength, eval_strength, split, bucket), bucket_rows in sorted(groups.items()):
        total = len(bucket_rows)
        counts = Counter(str(row["flip_type"]) for row in bucket_rows)
        pre_correct = sum(int(row["pre_correct"]) for row in bucket_rows)
        post_correct = sum(int(row["post_correct"]) for row in bucket_rows)
        changed = sum(int(row["prediction_changed"]) for row in bucket_rows)
        summary.append(
            {
                "config": config,
                "train_strength": train_strength,
                "eval_strength": eval_strength,
                "split": split,
                "confidence_bucket": bucket,
                "total": total,
                "pre_accuracy": pre_correct / max(total, 1),
                "post_accuracy": post_correct / max(total, 1),
                "accuracy_delta": (post_correct - pre_correct) / max(total, 1),
                "mean_delta_loss": float(np.mean([float(row["delta_loss"]) for row in bucket_rows])) if total else 0.0,
                "prediction_changed": changed,
                "helpful_flip": counts["helpful_flip"],
                "harmful_flip": counts["harmful_flip"],
                "wrong_flip_ce_improved": counts["wrong_flip_ce_improved"],
                "wrong_flip_ce_worse": counts["wrong_flip_ce_worse"],
                "same_correct_ce_improved": counts["same_correct_ce_improved"],
                "same_correct_ce_worse": counts["same_correct_ce_worse"],
                "same_wrong_ce_improved": counts["same_wrong_ce_improved"],
                "same_wrong_ce_worse": counts["same_wrong_ce_worse"],
            }
        )
    return summary


class FlipGateReadoutModel:
    """Train-split local gate that arbitrates pre/post-credit WTA flips.

    The gate is a small perceptron-style inhibitory microcircuit.  It sees only
    local score margins and before-credit confidence features.  Training uses
    final answer loss on the credit-train split as a third factor; evaluation
    labels are never used to update the gate.
    """

    base_feature_names = [
        "bias",
        "pre_margin",
        "post_margin",
        "margin_delta",
        "before_confidence",
        "before_gate_weight",
        "before_credit_margin",
        "post_flip_margin",
        "pre_flip_margin",
    ]

    compat_feature_names = [
        "before_credit_post_score",
        "before_credit_pre_score",
        "before_credit_post_minus_pre",
        "before_credit_post_gap_to_top",
        "before_credit_pre_gap_to_top",
        "before_credit_post_is_top",
        "before_credit_pre_is_top",
    ]

    compat_channel_feature_names = [
        "bias",
        "pre_margin",
        "post_margin",
        "margin_delta",
        "before_confidence",
        "before_gate_weight",
        "before_credit_margin",
        "before_credit_post_score",
        "before_credit_pre_score",
        "before_credit_post_minus_pre",
        "before_credit_post_gap_to_top",
        "before_credit_pre_gap_to_top",
        "before_credit_post_is_top",
        "before_credit_pre_is_top",
        "post_flip_margin",
        "pre_flip_margin",
    ]

    near_risk_channel_feature_names = [
        "bias",
        "risk_margin",
        "risk_margin_ratio",
        "risk_distance_ratio",
        "compat_score",
        "compat_allow",
        "one_class_margin",
        "positive_margin",
        "class_allow",
        "pre_margin",
        "post_margin",
        "margin_delta",
        "before_confidence",
        "before_gate_weight",
        "before_credit_margin",
        "post_flip_margin",
        "pre_flip_margin",
    ]

    def __init__(
        self,
        pre_model: RoleBindingStateQALearner,
        post_model: RoleBindingStateQALearner,
        answer_to_idx: dict[str, int],
        temperature: float,
        lr: float,
        epochs: int,
        threshold: float,
        init_bias: float,
        min_loss_gain: float,
        feature_scale: float,
        gate_mode: str,
        radius_scale: float,
        risk_radius_scale: float,
        counterfactual_top_k: int,
        counterfactual_margin: float,
        identity_features: bool,
        risk_class_prototypes: bool,
        risk_radius_quantile: float,
        risk_micro_prototypes: int,
        before_compat_features: bool,
        before_compat_filter: str,
        before_compat_margin: float,
        before_compat_channel: bool,
        risk_near_margin: float,
        risk_near_compat_threshold: float,
        risk_near_radius_fraction: float,
        risk_near_blocks_rescue: bool,
        learned_near_risk_channel: bool,
        learned_near_risk_lr: float,
        learned_near_risk_epochs: int,
        learned_near_risk_threshold: float,
        learned_near_risk_init_bias: float,
        learned_near_risk_boundary_fraction: float,
        learned_near_risk_balance_mode: str,
        learned_near_risk_inhibit_per_allow: float,
        learned_near_risk_source_channels: str,
        learned_near_risk_auto_threshold: str,
        learned_near_risk_auto_threshold_candidates: list[float],
        seed: int,
    ) -> None:
        self.pre_model = pre_model
        self.post_model = post_model
        self.answer_to_idx = answer_to_idx
        self.idx_to_answer = {idx: answer for answer, idx in answer_to_idx.items()}
        self.temperature = temperature
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.min_loss_gain = min_loss_gain
        self.feature_scale = max(float(feature_scale), 1e-6)
        self.gate_mode = gate_mode
        self.radius_scale = max(float(radius_scale), 0.0)
        self.risk_radius_scale = max(float(risk_radius_scale), 0.0)
        self.counterfactual_top_k = max(int(counterfactual_top_k), 0)
        self.counterfactual_margin = max(float(counterfactual_margin), 1e-6)
        self.identity_features = bool(identity_features)
        self.risk_class_prototypes_enabled = bool(risk_class_prototypes)
        self.risk_radius_quantile = float(np.clip(float(risk_radius_quantile), 0.0, 1.0))
        self.risk_micro_prototypes = max(int(risk_micro_prototypes), 1)
        self.before_compat_features = bool(before_compat_features)
        self.before_compat_filter = before_compat_filter
        self.before_compat_margin = float(before_compat_margin)
        self.before_compat_channel = bool(before_compat_channel)
        self.risk_near_margin = max(float(risk_near_margin), 0.0)
        self.risk_near_compat_threshold = float(risk_near_compat_threshold)
        self.risk_near_radius_fraction = max(float(risk_near_radius_fraction), 0.0)
        self.risk_near_blocks_rescue = bool(risk_near_blocks_rescue)
        self.learned_near_risk_channel = bool(learned_near_risk_channel)
        self.learned_near_risk_lr = float(learned_near_risk_lr)
        self.learned_near_risk_epochs = max(int(learned_near_risk_epochs), 0)
        self.learned_near_risk_threshold = float(learned_near_risk_threshold)
        self.learned_near_risk_init_bias = float(learned_near_risk_init_bias)
        self.learned_near_risk_boundary_fraction = max(float(learned_near_risk_boundary_fraction), 0.0)
        self.learned_near_risk_balance_mode = learned_near_risk_balance_mode
        self.learned_near_risk_inhibit_per_allow = max(float(learned_near_risk_inhibit_per_allow), 0.0)
        self.learned_near_risk_source_channels = learned_near_risk_source_channels
        self.learned_near_risk_auto_threshold = learned_near_risk_auto_threshold
        if learned_near_risk_auto_threshold_candidates:
            self.learned_near_risk_auto_threshold_candidates = [
                float(value) for value in learned_near_risk_auto_threshold_candidates
            ]
        else:
            self.learned_near_risk_auto_threshold_candidates = list(
                DEFAULT_LEARNED_NEAR_RISK_THRESHOLD_CANDIDATES
            )
        self.learned_near_risk_auto_selected_threshold = self.learned_near_risk_threshold
        self.learned_near_risk_auto_calibration_samples = 0
        self.learned_near_risk_auto_allow_samples = 0
        self.learned_near_risk_auto_inhibit_samples = 0
        self.learned_near_risk_auto_balanced_accuracy = 0.0
        self.learned_near_risk_auto_allow_recall = 0.0
        self.learned_near_risk_auto_inhibit_recall = 0.0
        self.learned_near_risk_auto_false_inhibit = 0
        self.learned_near_risk_auto_false_allow = 0
        self.feature_names = list(self.base_feature_names)
        if self.before_compat_features:
            self.feature_names.extend(self.compat_feature_names)
        self.answer_feature_indices = sorted(answer_to_idx.items(), key=lambda item: item[1])
        if self.identity_features:
            for answer, _ in self.answer_feature_indices:
                self.feature_names.append(f"pre_is_{answer}")
            for answer, _ in self.answer_feature_indices:
                self.feature_names.append(f"post_is_{answer}")
        self.weights = np.zeros(len(self.feature_names), dtype=np.float32)
        self.weights[0] = float(init_bias)
        self.compat_channel_weights = np.zeros(len(self.compat_channel_feature_names), dtype=np.float32)
        self.compat_channel_weights[0] = float(init_bias)
        self.learned_near_risk_weights = np.zeros(len(self.near_risk_channel_feature_names), dtype=np.float32)
        self.learned_near_risk_weights[0] = self.learned_near_risk_init_bias
        self.learned_near_risk_source_weights: dict[str, np.ndarray] = {}
        self.learned_near_risk_source_thresholds: dict[str, float] = {}
        self.learned_near_risk_source_counts: dict[str, Any] = {}
        self.learned_near_risk_source_updates: dict[str, int] = {}
        self.learned_near_risk_source_calibration: dict[str, Any] = {}
        self.learned_near_risk_samples = 0
        self.learned_near_risk_boundary_samples = 0
        self.learned_near_risk_epoch_samples = 0
        self.learned_near_risk_updates = 0
        self.positive_prototype = np.zeros(len(self.feature_names) - 1, dtype=np.float32)
        self.positive_radius = 0.0
        self.positive_count = 0
        self.risk_prototype = np.zeros(len(self.feature_names) - 1, dtype=np.float32)
        self.risk_radius = 0.0
        self.risk_count = 0
        self.counterfactual_risk_count = 0
        self.risk_class_prototypes: dict[str, np.ndarray] = {}
        self.risk_class_radii: dict[str, Any] = {}
        self.risk_class_counts: dict[str, Any] = {}
        self.class_prototypes: dict[str, np.ndarray] = {}
        self.class_radii: dict[str, float] = {}
        self.class_counts: dict[str, int] = {}
        self.rng = np.random.default_rng(seed)
        self._score_cache: dict[int, dict[str, Any]] = {}

    def scaled_margin(self, value: float) -> float:
        return float(np.clip(float(value) / self.feature_scale, -2.0, 2.0))

    def features(self, row: dict[str, Any], pre_scores: np.ndarray, post_scores: np.ndarray) -> np.ndarray:
        pre_margin = answer_margin(pre_scores, self.answer_to_idx)
        post_margin = answer_margin(post_scores, self.answer_to_idx)
        pre_pred = int(np.argmax(pre_scores))
        post_pred = int(np.argmax(post_scores))
        before_detail = before_credit_probe_details(self.post_model, row, pre_pred, post_pred)
        confidence = float(before_detail["confidence"])
        gate_weight = float(before_detail["gate_weight"])
        credit_margin = float(before_detail["credit_margin"])
        post_flip_margin = float(post_scores[post_pred] - post_scores[pre_pred])
        pre_flip_margin = float(pre_scores[pre_pred] - pre_scores[post_pred])
        values = [
            1.0,
            self.scaled_margin(pre_margin),
            self.scaled_margin(post_margin),
            self.scaled_margin(post_margin - pre_margin),
            float(np.clip(confidence, 0.0, 2.0)),
            float(np.clip(gate_weight, 0.0, 2.0)),
            self.scaled_margin(credit_margin),
            self.scaled_margin(post_flip_margin),
            self.scaled_margin(pre_flip_margin),
        ]
        if self.before_compat_features:
            values.extend(
                [
                    self.scaled_margin(float(before_detail.get("post_credit_score", 0.0))),
                    self.scaled_margin(float(before_detail.get("pre_credit_score", 0.0))),
                    self.scaled_margin(float(before_detail.get("post_minus_pre_credit_score", 0.0))),
                    self.scaled_margin(float(before_detail.get("post_credit_gap_to_top", 0.0))),
                    self.scaled_margin(float(before_detail.get("pre_credit_gap_to_top", 0.0))),
                    float(before_detail.get("post_is_credit_top", 0)),
                    float(before_detail.get("pre_is_credit_top", 0)),
                ]
            )
        if self.identity_features:
            values.extend(1.0 if pre_pred == idx else 0.0 for _, idx in self.answer_feature_indices)
            values.extend(1.0 if post_pred == idx else 0.0 for _, idx in self.answer_feature_indices)
        return np.asarray(values, dtype=np.float32)

    def compat_channel_features(
        self,
        row: dict[str, Any],
        pre_scores: np.ndarray,
        post_scores: np.ndarray,
    ) -> np.ndarray:
        pre_margin = answer_margin(pre_scores, self.answer_to_idx)
        post_margin = answer_margin(post_scores, self.answer_to_idx)
        pre_pred = int(np.argmax(pre_scores))
        post_pred = int(np.argmax(post_scores))
        before_detail = before_credit_probe_details(self.post_model, row, pre_pred, post_pred)
        post_flip_margin = float(post_scores[post_pred] - post_scores[pre_pred])
        pre_flip_margin = float(pre_scores[pre_pred] - pre_scores[post_pred])
        values = [
            1.0,
            self.scaled_margin(pre_margin),
            self.scaled_margin(post_margin),
            self.scaled_margin(post_margin - pre_margin),
            float(np.clip(float(before_detail["confidence"]), 0.0, 2.0)),
            float(np.clip(float(before_detail["gate_weight"]), 0.0, 2.0)),
            self.scaled_margin(float(before_detail["credit_margin"])),
            self.scaled_margin(float(before_detail.get("post_credit_score", 0.0))),
            self.scaled_margin(float(before_detail.get("pre_credit_score", 0.0))),
            self.scaled_margin(float(before_detail.get("post_minus_pre_credit_score", 0.0))),
            self.scaled_margin(float(before_detail.get("post_credit_gap_to_top", 0.0))),
            self.scaled_margin(float(before_detail.get("pre_credit_gap_to_top", 0.0))),
            float(before_detail.get("post_is_credit_top", 0)),
            float(before_detail.get("pre_is_credit_top", 0)),
            self.scaled_margin(post_flip_margin),
            self.scaled_margin(pre_flip_margin),
        ]
        return np.asarray(values, dtype=np.float32)

    def gate_score(self, features: np.ndarray) -> float:
        return float(self.weights @ features)

    def compat_channel_score(self, features: np.ndarray) -> float:
        return float(self.compat_channel_weights @ features)

    def scaled_finite(self, value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return self.scaled_margin(value)

    def near_risk_channel_features(self, features: np.ndarray, compat_score: float) -> np.ndarray:
        risk_detail = self.nearest_risk_detail(features)
        positive_detail = self.nearest_positive_detail(features)
        risk_margin = float(risk_detail.get("risk_margin", float("-inf")))
        risk_distance = float(risk_detail.get("risk_distance", float("inf")))
        risk_radius = float(risk_detail.get("risk_scaled_radius", 0.0))
        if np.isfinite(risk_radius) and risk_radius > 1e-6:
            risk_margin_ratio = float(np.clip(risk_margin / risk_radius, -2.0, 2.0))
            risk_distance_ratio = float(np.clip(risk_distance / risk_radius, 0.0, 4.0))
        else:
            risk_margin_ratio = -2.0
            risk_distance_ratio = 4.0
        values = [
            1.0,
            self.scaled_finite(risk_margin),
            risk_margin_ratio,
            risk_distance_ratio,
            float(np.clip(compat_score, -2.0, 2.0)),
            float(compat_score >= self.threshold),
            self.scaled_finite(float(positive_detail.get("one_class_margin", float("-inf")))),
            self.scaled_finite(float(positive_detail.get("positive_margin", float("-inf")))),
            float(bool(positive_detail.get("class_allow", False))),
            float(features[1]) if len(features) > 1 else 0.0,
            float(features[2]) if len(features) > 2 else 0.0,
            float(features[3]) if len(features) > 3 else 0.0,
            float(features[4]) if len(features) > 4 else 0.0,
            float(features[5]) if len(features) > 5 else 0.0,
            float(features[6]) if len(features) > 6 else 0.0,
            float(features[7]) if len(features) > 7 else 0.0,
            float(features[8]) if len(features) > 8 else 0.0,
        ]
        return np.asarray(values, dtype=np.float32)

    def near_risk_channel_score(self, features: np.ndarray, compat_score: float) -> float:
        near_features = self.near_risk_channel_features(features, compat_score)
        return float(self.learned_near_risk_weights @ near_features)

    def near_risk_channel_score_with_weights(
        self,
        features: np.ndarray,
        compat_score: float,
        weights: np.ndarray,
    ) -> float:
        near_features = self.near_risk_channel_features(features, compat_score)
        return float(weights @ near_features)

    def learned_near_risk_source_group(self, source: str) -> str:
        base = source.removesuffix("_boundary")
        if base.startswith("counterfactual_"):
            return "counterfactual"
        return "natural"

    def learned_near_risk_source_scores(
        self,
        features: np.ndarray,
        compat_score: float,
    ) -> dict[str, float]:
        if not self.learned_near_risk_source_weights:
            return {}
        near_features = self.near_risk_channel_features(features, compat_score)
        return {
            group: float(weights @ near_features)
            for group, weights in self.learned_near_risk_source_weights.items()
        }

    def learned_near_risk_match_from_features(
        self,
        features: np.ndarray,
        compat_score: float,
        raw_risk_match: bool,
    ) -> bool:
        if not self.learned_near_risk_channel:
            return False
        if raw_risk_match:
            return False
        if self.learned_near_risk_source_channels != "off" and self.learned_near_risk_source_weights:
            for group, score in self.learned_near_risk_source_scores(features, compat_score).items():
                threshold = self.learned_near_risk_source_thresholds.get(group, self.learned_near_risk_threshold)
                if score >= threshold:
                    return True
            return False
        return self.near_risk_channel_score(features, compat_score) >= self.learned_near_risk_threshold

    def one_class_distance(self, features: np.ndarray) -> float:
        if self.positive_count <= 0:
            return float("inf")
        local_features = features[1:].astype(np.float32, copy=False)
        return float(np.linalg.norm(local_features - self.positive_prototype))

    def one_class_allow(self, features: np.ndarray) -> bool:
        if self.positive_count <= 0:
            return False
        return self.one_class_distance(features) <= self.positive_radius * self.radius_scale + 1e-6

    def class_prototype_allow(self, features: np.ndarray) -> bool:
        if not self.class_prototypes:
            return False
        local_features = features[1:].astype(np.float32, copy=False)
        for label, prototype in self.class_prototypes.items():
            radius = self.class_radii.get(label, 0.0) * self.radius_scale
            if float(np.linalg.norm(local_features - prototype)) <= radius + 1e-6:
                return True
        return False

    def risk_distance(self, features: np.ndarray) -> float:
        if self.risk_count <= 0:
            return float("inf")
        local_features = features[1:].astype(np.float32, copy=False)
        return float(np.linalg.norm(local_features - self.risk_prototype))

    def risk_match(self, features: np.ndarray) -> bool:
        if self.risk_count <= 0:
            return False
        if self.risk_class_prototypes_enabled and self.risk_class_prototypes:
            local_features = features[1:].astype(np.float32, copy=False)
            for label, prototype in self.risk_class_prototypes.items():
                prototypes = np.asarray(prototype, dtype=np.float32)
                if prototypes.ndim == 1:
                    prototypes = prototypes[None, :]
                raw_radii = self.risk_class_radii.get(label, 0.0)
                if isinstance(raw_radii, list):
                    radii = [float(value) for value in raw_radii]
                else:
                    radii = [float(raw_radii)] * int(prototypes.shape[0])
                for local_prototype, radius in zip(prototypes, radii, strict=False):
                    scaled_radius = radius * self.risk_radius_scale
                    if float(np.linalg.norm(local_features - local_prototype)) <= scaled_radius + 1e-6:
                        return True
            return False
        return self.risk_distance(features) <= self.risk_radius * self.risk_radius_scale + 1e-6

    def nearest_risk_detail(self, features: np.ndarray) -> dict[str, Any]:
        local_features = features[1:].astype(np.float32, copy=False)
        best = {
            "risk_label": "",
            "risk_proto_index": -1,
            "risk_proto_count": 0,
            "risk_distance": float("inf"),
            "risk_radius": 0.0,
            "risk_scaled_radius": 0.0,
            "risk_margin": float("-inf"),
            "risk_match": False,
        }
        if self.risk_count <= 0:
            return best
        if self.risk_class_prototypes_enabled and self.risk_class_prototypes:
            for label, prototype in self.risk_class_prototypes.items():
                prototypes = np.asarray(prototype, dtype=np.float32)
                if prototypes.ndim == 1:
                    prototypes = prototypes[None, :]
                raw_radii = self.risk_class_radii.get(label, 0.0)
                if isinstance(raw_radii, list):
                    radii = [float(value) for value in raw_radii]
                else:
                    radii = [float(raw_radii)] * int(prototypes.shape[0])
                raw_counts = self.risk_class_counts.get(label, 0)
                if isinstance(raw_counts, list):
                    counts = [int(value) for value in raw_counts]
                else:
                    counts = [int(raw_counts)] * int(prototypes.shape[0])
                for idx, (local_prototype, radius) in enumerate(zip(prototypes, radii, strict=False)):
                    distance = float(np.linalg.norm(local_features - local_prototype))
                    scaled_radius = float(radius) * self.risk_radius_scale
                    margin = scaled_radius - distance
                    if distance < float(best["risk_distance"]):
                        best = {
                            "risk_label": label,
                            "risk_proto_index": idx,
                            "risk_proto_count": counts[idx] if idx < len(counts) else 0,
                            "risk_distance": distance,
                            "risk_radius": float(radius),
                            "risk_scaled_radius": scaled_radius,
                            "risk_margin": margin,
                            "risk_match": margin >= -1e-6,
                        }
            return best

        distance = self.risk_distance(features)
        scaled_radius = self.risk_radius * self.risk_radius_scale
        return {
            "risk_label": "global",
            "risk_proto_index": 0,
            "risk_proto_count": self.risk_count,
            "risk_distance": distance,
            "risk_radius": self.risk_radius,
            "risk_scaled_radius": scaled_radius,
            "risk_margin": scaled_radius - distance,
            "risk_match": distance <= scaled_radius + 1e-6,
        }

    def nearest_positive_detail(self, features: np.ndarray) -> dict[str, Any]:
        one_radius = self.positive_radius * self.radius_scale
        one_distance = self.one_class_distance(features)
        best = {
            "one_class_distance": one_distance,
            "one_class_scaled_radius": one_radius,
            "one_class_margin": one_radius - one_distance,
            "one_class_allow": self.one_class_allow(features),
            "positive_label": "",
            "positive_count": 0,
            "positive_distance": float("inf"),
            "positive_radius": 0.0,
            "positive_scaled_radius": 0.0,
            "positive_margin": float("-inf"),
            "class_allow": False,
        }
        if not self.class_prototypes:
            return best
        local_features = features[1:].astype(np.float32, copy=False)
        for label, prototype in self.class_prototypes.items():
            radius = float(self.class_radii.get(label, 0.0))
            scaled_radius = radius * self.radius_scale
            distance = float(np.linalg.norm(local_features - prototype))
            margin = scaled_radius - distance
            if distance < float(best["positive_distance"]):
                best.update(
                    {
                        "positive_label": label,
                        "positive_count": int(self.class_counts.get(label, 0)),
                        "positive_distance": distance,
                        "positive_radius": radius,
                        "positive_scaled_radius": scaled_radius,
                        "positive_margin": margin,
                        "class_allow": margin >= -1e-6,
                    }
                )
        return best

    def risk_near_buffer_from_detail(self, risk_detail: dict[str, Any]) -> float:
        buffer = self.risk_near_margin
        if self.risk_near_radius_fraction > 0.0:
            scaled_radius = float(risk_detail.get("risk_scaled_radius", 0.0))
            if np.isfinite(scaled_radius) and scaled_radius > 0.0:
                buffer = max(buffer, self.risk_near_radius_fraction * scaled_radius)
        return float(buffer)

    def risk_near_match_from_detail(
        self,
        risk_detail: dict[str, Any],
        compat_score: float,
        raw_risk_match: bool | None = None,
    ) -> bool:
        buffer = self.risk_near_buffer_from_detail(risk_detail)
        if buffer <= 0.0:
            return False
        if raw_risk_match is None:
            raw_risk_match = bool(risk_detail.get("risk_match", False))
        if raw_risk_match:
            return False
        margin = float(risk_detail.get("risk_margin", float("-inf")))
        if not np.isfinite(margin):
            return False
        return margin >= -buffer - 1e-6 and compat_score >= self.risk_near_compat_threshold

    def before_compat_allow(self, row: dict[str, Any], pre_scores: np.ndarray, post_scores: np.ndarray) -> bool:
        if self.before_compat_filter == "off":
            return True
        pre_pred = int(np.argmax(pre_scores))
        post_pred = int(np.argmax(post_scores))
        detail = before_credit_probe_details(self.post_model, row, pre_pred, post_pred)
        if detail.get("status") != "active":
            return True
        post_is_top = bool(int(detail.get("post_is_credit_top", 0)))
        post_minus_pre = float(detail.get("post_minus_pre_credit_score", 0.0))
        post_gap = float(detail.get("post_credit_gap_to_top", 0.0))
        margin = self.before_compat_margin
        if self.before_compat_filter == "post_top":
            return post_is_top or post_gap <= margin
        if self.before_compat_filter == "post_better":
            return post_minus_pre >= margin
        if self.before_compat_filter == "post_close":
            return post_gap <= margin
        return True

    def prototype_radius(self, distances: np.ndarray) -> float:
        if len(distances) == 0:
            return 0.0
        if self.risk_radius_quantile >= 1.0:
            return float(np.max(distances))
        return float(np.quantile(distances, self.risk_radius_quantile))

    def risk_micro_cluster(self, stacked: np.ndarray) -> tuple[np.ndarray, list[float], list[int]]:
        if self.risk_micro_prototypes <= 1 or len(stacked) <= 1:
            prototype = np.mean(stacked, axis=0, keepdims=True).astype(np.float32)
            distances = np.linalg.norm(stacked - prototype[0][None, :], axis=1)
            return prototype, [self.prototype_radius(distances)], [int(len(stacked))]

        centroid = np.mean(stacked, axis=0).astype(np.float32)
        first_idx = int(np.argmin(np.linalg.norm(stacked - centroid[None, :], axis=1)))
        centers = [stacked[first_idx].astype(np.float32, copy=True)]
        max_k = min(self.risk_micro_prototypes, len(stacked))
        for _ in range(1, max_k):
            dist_to_centers = np.min(
                np.linalg.norm(stacked[:, None, :] - np.stack(centers)[None, :, :], axis=2),
                axis=1,
            )
            next_idx = int(np.argmax(dist_to_centers))
            if float(dist_to_centers[next_idx]) <= 1e-8:
                break
            centers.append(stacked[next_idx].astype(np.float32, copy=True))

        centers_arr = np.stack(centers).astype(np.float32)
        assignments = np.zeros(len(stacked), dtype=np.int64)
        for _ in range(3):
            distances = np.linalg.norm(stacked[:, None, :] - centers_arr[None, :, :], axis=2)
            assignments = np.argmin(distances, axis=1)
            new_centers = centers_arr.copy()
            for idx in range(len(centers_arr)):
                members = stacked[assignments == idx]
                if len(members):
                    new_centers[idx] = np.mean(members, axis=0).astype(np.float32)
            if np.allclose(new_centers, centers_arr):
                break
            centers_arr = new_centers

        radii: list[float] = []
        counts: list[int] = []
        for idx in range(len(centers_arr)):
            members = stacked[assignments == idx]
            counts.append(int(len(members)))
            if len(members):
                distances = np.linalg.norm(members - centers_arr[idx][None, :], axis=1)
                radii.append(self.prototype_radius(distances))
            else:
                radii.append(0.0)
        return centers_arr.astype(np.float32), radii, counts

    def allow_post(self, row: dict[str, Any], pre_scores: np.ndarray, post_scores: np.ndarray) -> tuple[bool, float]:
        features = self.features(row, pre_scores, post_scores)
        score = self.gate_score(features)
        perceptron_allow = score >= self.threshold
        one_class_allow = self.one_class_allow(features)
        class_allow = self.class_prototype_allow(features)
        compat_allow = self.before_compat_allow(row, pre_scores, post_scores)
        compat_score = 0.0
        compat_channel_allow = False
        if self.before_compat_channel:
            compat_features = self.compat_channel_features(row, pre_scores, post_scores)
            compat_score = self.compat_channel_score(compat_features)
            compat_channel_allow = compat_score >= self.threshold
        risk_detail = self.nearest_risk_detail(features)
        risk_match = self.risk_match(features)
        risk_near_match = self.risk_near_match_from_detail(risk_detail, compat_score, risk_match)
        learned_near_match = self.learned_near_risk_match_from_features(features, compat_score, risk_match)
        effective_risk_match = risk_match or risk_near_match or learned_near_match
        rescue_veto = self.risk_near_blocks_rescue and (risk_near_match or learned_near_match)
        if self.gate_mode == "one_class":
            return one_class_allow and compat_allow, score
        if self.gate_mode == "class_prototype":
            return class_allow and compat_allow, score
        if self.gate_mode == "risk_only":
            return (not effective_risk_match) and compat_allow, score
        if self.gate_mode == "risk_compat_rescue":
            rescue_allow = compat_channel_allow and not rescue_veto
            return ((not effective_risk_match) or rescue_allow) and compat_allow, compat_score
        if self.gate_mode == "risk_compat_positive_rescue":
            rescue_allow = compat_channel_allow and one_class_allow
            if rescue_veto:
                rescue_allow = False
            return ((not effective_risk_match) or rescue_allow) and compat_allow, compat_score
        if self.gate_mode == "risk_compat_class_rescue":
            rescue_allow = compat_channel_allow and class_allow
            if rescue_veto:
                rescue_allow = False
            return ((not effective_risk_match) or rescue_allow) and compat_allow, compat_score
        if self.gate_mode == "risk_prototype":
            return one_class_allow and (not effective_risk_match) and compat_allow, score
        if self.gate_mode == "class_risk":
            return class_allow and (not effective_risk_match) and compat_allow, score
        if self.gate_mode == "hybrid":
            return perceptron_allow and one_class_allow and compat_allow, score
        return perceptron_allow and compat_allow, score

    def blocked_scores(self, pre_scores: np.ndarray, post_scores: np.ndarray) -> np.ndarray:
        pre_pred = int(np.argmax(pre_scores))
        post_pred = int(np.argmax(post_scores))
        blocked = post_scores.astype(np.float32, copy=True)
        blocked[pre_pred] = max(float(blocked[pre_pred]), float(blocked[post_pred]) + 1e-3)
        return blocked

    def row_record(self, row: dict[str, Any]) -> dict[str, Any]:
        cache_key = id(row)
        cached = self._score_cache.get(cache_key)
        if cached is not None:
            return cached
        pre_scores = self.pre_model.scores(row).astype(np.float32, copy=False)
        post_scores = self.post_model.scores(row).astype(np.float32, copy=False)
        pre_pred = int(np.argmax(pre_scores))
        post_pred = int(np.argmax(post_scores))
        allow = True
        score = 0.0
        if pre_pred != post_pred:
            allow, score = self.allow_post(row, pre_scores, post_scores)
        gate_scores = post_scores if pre_pred == post_pred or allow else self.blocked_scores(pre_scores, post_scores)
        record = {
            "pre_scores": pre_scores,
            "post_scores": post_scores,
            "gate_scores": gate_scores.astype(np.float32, copy=False),
            "pre_pred": pre_pred,
            "post_pred": post_pred,
            "gate_pred": int(np.argmax(gate_scores)),
            "allow": allow,
            "gate_score": score,
        }
        self._score_cache[cache_key] = record
        return record

    def forced_post_scores(self, post_scores: np.ndarray, candidate_idx: int) -> np.ndarray:
        forced = post_scores.astype(np.float32, copy=True)
        forced[candidate_idx] = float(np.max(forced)) + self.counterfactual_margin
        return forced

    def counterfactual_risk_features(
        self,
        row: dict[str, Any],
        pre_scores: np.ndarray,
        post_scores: np.ndarray,
        target_idx: int,
        pre_loss: float,
    ) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
        if self.counterfactual_top_k <= 0:
            return []
        pre_pred = int(np.argmax(pre_scores))
        order = np.argsort(post_scores)[::-1]
        out: list[tuple[str, np.ndarray, np.ndarray | None]] = []
        considered = 0
        for candidate_idx in order:
            candidate_idx = int(candidate_idx)
            if candidate_idx == pre_pred:
                continue
            forced_scores = self.forced_post_scores(post_scores, candidate_idx)
            if int(np.argmax(forced_scores)) != candidate_idx:
                continue
            considered += 1
            forced_loss = answer_loss(forced_scores, target_idx, self.temperature)
            if forced_loss > pre_loss + self.min_loss_gain:
                label = f"counterfactual_{self.idx_to_answer[candidate_idx]}"
                compat_feature = (
                    self.compat_channel_features(row, pre_scores, forced_scores).astype(np.float32, copy=True)
                    if self.before_compat_channel
                    else None
                )
                out.append(
                    (
                        label,
                        self.features(row, pre_scores, forced_scores).astype(np.float32, copy=True),
                        compat_feature,
                    )
                )
            if considered >= self.counterfactual_top_k:
                break
        return out

    def nearest_risk_anchor(self, features: np.ndarray) -> tuple[np.ndarray | None, float]:
        if self.risk_count <= 0:
            return None, 0.0
        local_features = features[1:].astype(np.float32, copy=False)
        best_prototype: np.ndarray | None = None
        best_distance = float("inf")
        best_radius = 0.0
        if self.risk_class_prototypes_enabled and self.risk_class_prototypes:
            for label, prototype in self.risk_class_prototypes.items():
                prototypes = np.asarray(prototype, dtype=np.float32)
                if prototypes.ndim == 1:
                    prototypes = prototypes[None, :]
                raw_radii = self.risk_class_radii.get(label, 0.0)
                if isinstance(raw_radii, list):
                    radii = [float(value) for value in raw_radii]
                else:
                    radii = [float(raw_radii)] * int(prototypes.shape[0])
                for local_prototype, radius in zip(prototypes, radii, strict=False):
                    distance = float(np.linalg.norm(local_features - local_prototype))
                    if distance < best_distance:
                        best_distance = distance
                        best_prototype = local_prototype.astype(np.float32, copy=True)
                        best_radius = float(radius) * self.risk_radius_scale
            return best_prototype, best_radius
        return self.risk_prototype.astype(np.float32, copy=True), self.risk_radius * self.risk_radius_scale

    def boundary_risk_feature(self, features: np.ndarray) -> np.ndarray | None:
        prototype, radius = self.nearest_risk_anchor(features)
        if prototype is None or radius <= 1e-6:
            return None
        local_features = features[1:].astype(np.float32, copy=False)
        direction = local_features - prototype
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            direction = self.rng.normal(size=prototype.shape).astype(np.float32)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                return None
        boundary = features.astype(np.float32, copy=True)
        target_distance = radius * (1.0 + self.learned_near_risk_boundary_fraction)
        boundary[1:] = prototype + (direction / norm) * target_distance
        return boundary.astype(np.float32, copy=False)

    def learned_near_risk_balanced_subset(
        self,
        train_samples: list[tuple[np.ndarray, np.ndarray | None, bool, str]],
    ) -> list[tuple[np.ndarray, np.ndarray | None, bool, str]]:
        allow_samples = [sample for sample in train_samples if not sample[2]]
        inhibit_samples = [sample for sample in train_samples if sample[2]]
        if not allow_samples or not inhibit_samples:
            return list(train_samples)
        if self.learned_near_risk_inhibit_per_allow <= 0.0:
            return list(allow_samples) + list(inhibit_samples)
        inhibit_limit = max(1, int(round(len(allow_samples) * self.learned_near_risk_inhibit_per_allow)))
        if inhibit_limit >= len(inhibit_samples):
            selected_inhibit = list(inhibit_samples)
        else:
            selected = self.rng.choice(len(inhibit_samples), size=inhibit_limit, replace=False)
            selected_inhibit = [inhibit_samples[int(idx)] for idx in selected]
        return list(allow_samples) + selected_inhibit

    def learned_near_risk_sample_scores(
        self,
        samples: list[tuple[np.ndarray, np.ndarray | None, bool, str]],
        weights: np.ndarray | None = None,
    ) -> list[tuple[float, bool, str]]:
        if weights is None:
            weights = self.learned_near_risk_weights
        out: list[tuple[float, bool, str]] = []
        for features, compat_features, target_inhibit, source in samples:
            compat_score = self.compat_channel_score(compat_features) if compat_features is not None else 0.0
            near_features = self.near_risk_channel_features(features, compat_score)
            score = float(weights @ near_features)
            out.append((score, bool(target_inhibit), source))
        return out

    def calibrate_learned_near_risk_threshold(
        self,
        train_samples: list[tuple[np.ndarray, np.ndarray | None, bool, str]],
    ) -> dict[str, Any]:
        candidates = sorted({float(value) for value in self.learned_near_risk_auto_threshold_candidates})
        if not candidates:
            candidates = list(DEFAULT_LEARNED_NEAR_RISK_THRESHOLD_CANDIDATES)
        base_metrics: dict[str, Any] = {
            "learned_near_risk_auto_threshold_mode": self.learned_near_risk_auto_threshold,
            "learned_near_risk_auto_threshold_candidates": json.dumps(candidates),
            "learned_near_risk_auto_selected_threshold": self.learned_near_risk_threshold,
            "learned_near_risk_auto_calibration_samples": 0,
            "learned_near_risk_auto_allow_samples": 0,
            "learned_near_risk_auto_inhibit_samples": 0,
            "learned_near_risk_auto_balanced_accuracy": 0.0,
            "learned_near_risk_auto_accuracy": 0.0,
            "learned_near_risk_auto_allow_recall": 0.0,
            "learned_near_risk_auto_inhibit_recall": 0.0,
            "learned_near_risk_auto_false_inhibit": 0,
            "learned_near_risk_auto_false_allow": 0,
            "learned_near_risk_auto_threshold_metrics": "[]",
        }
        if self.learned_near_risk_auto_threshold == "off":
            return base_metrics

        if self.learned_near_risk_auto_threshold != "balanced_train":
            raise ValueError(f"unknown learned near-risk auto threshold mode: {self.learned_near_risk_auto_threshold}")

        calibration_samples = self.learned_near_risk_balanced_subset(train_samples)
        scored = self.learned_near_risk_sample_scores(calibration_samples)
        if not scored:
            return base_metrics

        allow_count = sum(1 for _score, target_inhibit, _source in scored if not target_inhibit)
        inhibit_count = sum(1 for _score, target_inhibit, _source in scored if target_inhibit)
        total = len(scored)
        threshold_rows: list[dict[str, Any]] = []
        best_row: dict[str, Any] | None = None
        best_key: tuple[float, float, float, float] | None = None
        for threshold in candidates:
            true_inhibit = 0
            false_inhibit = 0
            true_allow = 0
            false_allow = 0
            for score, target_inhibit, _source in scored:
                pred_inhibit = score >= threshold
                if target_inhibit and pred_inhibit:
                    true_inhibit += 1
                elif target_inhibit and not pred_inhibit:
                    false_allow += 1
                elif not target_inhibit and pred_inhibit:
                    false_inhibit += 1
                else:
                    true_allow += 1
            allow_recall = true_allow / max(allow_count, 1)
            inhibit_recall = true_inhibit / max(inhibit_count, 1)
            if allow_count and inhibit_count:
                balanced_accuracy = 0.5 * (allow_recall + inhibit_recall)
            else:
                balanced_accuracy = (true_allow + true_inhibit) / max(total, 1)
            accuracy = (true_allow + true_inhibit) / max(total, 1)
            row = {
                "threshold": threshold,
                "samples": total,
                "allow_samples": allow_count,
                "inhibit_samples": inhibit_count,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": accuracy,
                "allow_recall": allow_recall,
                "inhibit_recall": inhibit_recall,
                "false_inhibit": false_inhibit,
                "false_allow": false_allow,
            }
            threshold_rows.append(row)
            key = (balanced_accuracy, allow_recall, inhibit_recall, threshold)
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
        if best_row is None:
            return base_metrics

        selected_threshold = float(best_row["threshold"])
        self.learned_near_risk_threshold = selected_threshold
        self.learned_near_risk_auto_selected_threshold = selected_threshold
        self.learned_near_risk_auto_calibration_samples = int(best_row["samples"])
        self.learned_near_risk_auto_allow_samples = int(best_row["allow_samples"])
        self.learned_near_risk_auto_inhibit_samples = int(best_row["inhibit_samples"])
        self.learned_near_risk_auto_balanced_accuracy = float(best_row["balanced_accuracy"])
        self.learned_near_risk_auto_allow_recall = float(best_row["allow_recall"])
        self.learned_near_risk_auto_inhibit_recall = float(best_row["inhibit_recall"])
        self.learned_near_risk_auto_false_inhibit = int(best_row["false_inhibit"])
        self.learned_near_risk_auto_false_allow = int(best_row["false_allow"])

        base_metrics.update(
            {
                "learned_near_risk_auto_selected_threshold": selected_threshold,
                "learned_near_risk_auto_calibration_samples": int(best_row["samples"]),
                "learned_near_risk_auto_allow_samples": int(best_row["allow_samples"]),
                "learned_near_risk_auto_inhibit_samples": int(best_row["inhibit_samples"]),
                "learned_near_risk_auto_balanced_accuracy": float(best_row["balanced_accuracy"]),
                "learned_near_risk_auto_accuracy": float(best_row["accuracy"]),
                "learned_near_risk_auto_allow_recall": float(best_row["allow_recall"]),
                "learned_near_risk_auto_inhibit_recall": float(best_row["inhibit_recall"]),
                "learned_near_risk_auto_false_inhibit": int(best_row["false_inhibit"]),
                "learned_near_risk_auto_false_allow": int(best_row["false_allow"]),
                "learned_near_risk_auto_threshold_metrics": json.dumps(threshold_rows, sort_keys=True),
            }
        )
        return base_metrics

    def learned_near_risk_threshold_metrics_for_weights(
        self,
        train_samples: list[tuple[np.ndarray, np.ndarray | None, bool, str]],
        weights: np.ndarray,
    ) -> dict[str, Any]:
        candidates = sorted({float(value) for value in self.learned_near_risk_auto_threshold_candidates})
        if not candidates:
            candidates = list(DEFAULT_LEARNED_NEAR_RISK_THRESHOLD_CANDIDATES)
        calibration_samples = self.learned_near_risk_balanced_subset(train_samples)
        scored = self.learned_near_risk_sample_scores(calibration_samples, weights)
        allow_count = sum(1 for _score, target_inhibit, _source in scored if not target_inhibit)
        inhibit_count = sum(1 for _score, target_inhibit, _source in scored if target_inhibit)
        total = len(scored)
        if total <= 0:
            return {
                "selected_threshold": self.learned_near_risk_threshold,
                "samples": 0,
                "allow_samples": 0,
                "inhibit_samples": 0,
                "balanced_accuracy": 0.0,
                "accuracy": 0.0,
                "allow_recall": 0.0,
                "inhibit_recall": 0.0,
                "false_inhibit": 0,
                "false_allow": 0,
                "threshold_metrics": [],
            }
        threshold_rows: list[dict[str, Any]] = []
        best_row: dict[str, Any] | None = None
        best_key: tuple[float, float, float, float] | None = None
        for threshold in candidates:
            true_inhibit = 0
            false_inhibit = 0
            true_allow = 0
            false_allow = 0
            for score, target_inhibit, _source in scored:
                pred_inhibit = score >= threshold
                if target_inhibit and pred_inhibit:
                    true_inhibit += 1
                elif target_inhibit and not pred_inhibit:
                    false_allow += 1
                elif not target_inhibit and pred_inhibit:
                    false_inhibit += 1
                else:
                    true_allow += 1
            allow_recall = true_allow / max(allow_count, 1)
            inhibit_recall = true_inhibit / max(inhibit_count, 1)
            if allow_count and inhibit_count:
                balanced_accuracy = 0.5 * (allow_recall + inhibit_recall)
            else:
                balanced_accuracy = (true_allow + true_inhibit) / max(total, 1)
            accuracy = (true_allow + true_inhibit) / max(total, 1)
            row = {
                "threshold": threshold,
                "samples": total,
                "allow_samples": allow_count,
                "inhibit_samples": inhibit_count,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": accuracy,
                "allow_recall": allow_recall,
                "inhibit_recall": inhibit_recall,
                "false_inhibit": false_inhibit,
                "false_allow": false_allow,
            }
            threshold_rows.append(row)
            key = (balanced_accuracy, allow_recall, inhibit_recall, threshold)
            if best_key is None or key > best_key:
                best_key = key
                best_row = row
        assert best_row is not None
        return {
            "selected_threshold": float(best_row["threshold"]),
            "samples": int(best_row["samples"]),
            "allow_samples": int(best_row["allow_samples"]),
            "inhibit_samples": int(best_row["inhibit_samples"]),
            "balanced_accuracy": float(best_row["balanced_accuracy"]),
            "accuracy": float(best_row["accuracy"]),
            "allow_recall": float(best_row["allow_recall"]),
            "inhibit_recall": float(best_row["inhibit_recall"]),
            "false_inhibit": int(best_row["false_inhibit"]),
            "false_allow": int(best_row["false_allow"]),
            "threshold_metrics": threshold_rows,
        }

    def fit_learned_near_risk_source_channels(
        self,
        train_samples: list[tuple[np.ndarray, np.ndarray | None, bool, str]],
    ) -> dict[str, Any]:
        self.learned_near_risk_source_weights = {}
        self.learned_near_risk_source_thresholds = {}
        self.learned_near_risk_source_counts = {}
        self.learned_near_risk_source_updates = {}
        self.learned_near_risk_source_calibration = {}
        base_metrics = {
            "learned_near_risk_source_channel_mode": self.learned_near_risk_source_channels,
            "learned_near_risk_source_channel_counts": "{}",
            "learned_near_risk_source_channel_thresholds": "{}",
            "learned_near_risk_source_channel_updates": "{}",
            "learned_near_risk_source_channel_calibration": "{}",
        }
        if self.learned_near_risk_source_channels == "off":
            return base_metrics
        if self.learned_near_risk_source_channels != "risk_source":
            raise ValueError(f"unknown learned near-risk source channel mode: {self.learned_near_risk_source_channels}")

        allow_samples = [sample for sample in train_samples if not sample[2]]
        inhibit_by_group: dict[str, list[tuple[np.ndarray, np.ndarray | None, bool, str]]] = defaultdict(list)
        for sample in train_samples:
            if not sample[2]:
                continue
            inhibit_by_group[self.learned_near_risk_source_group(sample[3])].append(sample)

        for group, inhibit_samples in sorted(inhibit_by_group.items()):
            if not allow_samples or not inhibit_samples:
                continue
            group_samples = list(allow_samples) + list(inhibit_samples)
            weights = np.zeros(len(self.near_risk_channel_feature_names), dtype=np.float32)
            weights[0] = self.learned_near_risk_init_bias
            updates = 0
            epoch_sample_count = 0
            target_inhibit_count = 0
            pred_inhibit_count = 0
            for _ in range(self.learned_near_risk_epochs):
                epoch_samples = self.learned_near_risk_balanced_subset(group_samples)
                epoch_sample_count += len(epoch_samples)
                order = self.rng.permutation(len(epoch_samples))
                for raw_idx in order:
                    features, compat_features, target_inhibit, _source = epoch_samples[int(raw_idx)]
                    compat_score = self.compat_channel_score(compat_features) if compat_features is not None else 0.0
                    near_features = self.near_risk_channel_features(features, compat_score)
                    score = float(weights @ near_features)
                    pred_inhibit = score >= self.learned_near_risk_threshold
                    target_inhibit_count += int(target_inhibit)
                    pred_inhibit_count += int(pred_inhibit)
                    if pred_inhibit != target_inhibit:
                        direction = 1.0 if target_inhibit else -1.0
                        weights += (self.learned_near_risk_lr * direction * near_features).astype(np.float32)
                        updates += 1

            calibration = self.learned_near_risk_threshold_metrics_for_weights(group_samples, weights)
            selected_threshold = float(calibration["selected_threshold"])
            self.learned_near_risk_source_weights[group] = weights
            self.learned_near_risk_source_thresholds[group] = selected_threshold
            self.learned_near_risk_source_counts[group] = {
                "samples": len(group_samples),
                "allow_samples": len(allow_samples),
                "inhibit_samples": len(inhibit_samples),
                "epoch_samples": epoch_sample_count,
                "target_inhibit": target_inhibit_count,
                "pred_inhibit_before_update": pred_inhibit_count,
            }
            self.learned_near_risk_source_updates[group] = updates
            self.learned_near_risk_source_calibration[group] = calibration

        base_metrics.update(
            {
                "learned_near_risk_source_channel_counts": json.dumps(
                    self.learned_near_risk_source_counts, sort_keys=True
                ),
                "learned_near_risk_source_channel_thresholds": json.dumps(
                    self.learned_near_risk_source_thresholds, sort_keys=True
                ),
                "learned_near_risk_source_channel_updates": json.dumps(
                    self.learned_near_risk_source_updates, sort_keys=True
                ),
                "learned_near_risk_source_channel_calibration": json.dumps(
                    self.learned_near_risk_source_calibration, sort_keys=True
                ),
            }
        )
        return base_metrics

    def fit_learned_near_risk_channel(
        self,
        samples: list[tuple[np.ndarray, np.ndarray | None, bool, str]],
    ) -> dict[str, Any]:
        if not self.learned_near_risk_channel or self.learned_near_risk_epochs <= 0:
            return {
                "learned_near_risk_channel": int(self.learned_near_risk_channel),
                "learned_near_risk_samples": 0,
                "learned_near_risk_boundary_samples": 0,
                "learned_near_risk_updates": 0,
                "learned_near_risk_threshold": self.learned_near_risk_threshold,
                "learned_near_risk_lr": self.learned_near_risk_lr,
                "learned_near_risk_epochs": self.learned_near_risk_epochs,
                "learned_near_risk_boundary_fraction": self.learned_near_risk_boundary_fraction,
                "learned_near_risk_balance_mode": self.learned_near_risk_balance_mode,
                "learned_near_risk_inhibit_per_allow": self.learned_near_risk_inhibit_per_allow,
                "learned_near_risk_auto_threshold_mode": self.learned_near_risk_auto_threshold,
                "learned_near_risk_auto_threshold_candidates": json.dumps(
                    self.learned_near_risk_auto_threshold_candidates
                ),
                "learned_near_risk_auto_selected_threshold": self.learned_near_risk_threshold,
                "learned_near_risk_auto_calibration_samples": 0,
                "learned_near_risk_auto_allow_samples": 0,
                "learned_near_risk_auto_inhibit_samples": 0,
                "learned_near_risk_auto_balanced_accuracy": 0.0,
                "learned_near_risk_auto_accuracy": 0.0,
                "learned_near_risk_auto_allow_recall": 0.0,
                "learned_near_risk_auto_inhibit_recall": 0.0,
                "learned_near_risk_auto_false_inhibit": 0,
                "learned_near_risk_auto_false_allow": 0,
                "learned_near_risk_auto_threshold_metrics": "[]",
                "learned_near_risk_source_channel_mode": self.learned_near_risk_source_channels,
                "learned_near_risk_source_channel_counts": "{}",
                "learned_near_risk_source_channel_thresholds": "{}",
                "learned_near_risk_source_channel_updates": "{}",
                "learned_near_risk_source_channel_calibration": "{}",
            }
        train_samples = list(samples)
        boundary_samples = 0
        if self.learned_near_risk_boundary_fraction > 0.0:
            for features, compat_features, target_inhibit, source in samples:
                if not target_inhibit:
                    continue
                boundary_features = self.boundary_risk_feature(features)
                if boundary_features is None:
                    continue
                train_samples.append((boundary_features, compat_features, True, f"{source}_boundary"))
                boundary_samples += 1
        allow_samples = [sample for sample in train_samples if not sample[2]]
        inhibit_samples = [sample for sample in train_samples if sample[2]]
        updates = 0
        target_inhibit_count = 0
        pred_inhibit_count = 0
        epoch_sample_count = 0
        for _ in range(self.learned_near_risk_epochs):
            epoch_samples = train_samples
            if (
                self.learned_near_risk_balance_mode == "resample"
                and allow_samples
                and inhibit_samples
                and self.learned_near_risk_inhibit_per_allow > 0.0
            ):
                inhibit_limit = max(1, int(round(len(allow_samples) * self.learned_near_risk_inhibit_per_allow)))
                if inhibit_limit < len(inhibit_samples):
                    selected = self.rng.choice(len(inhibit_samples), size=inhibit_limit, replace=False)
                    epoch_samples = list(allow_samples) + [inhibit_samples[int(idx)] for idx in selected]
            epoch_sample_count += len(epoch_samples)
            order = self.rng.permutation(len(epoch_samples))
            for raw_idx in order:
                features, compat_features, target_inhibit, _source = epoch_samples[int(raw_idx)]
                if compat_features is not None:
                    compat_score = self.compat_channel_score(compat_features)
                else:
                    compat_score = 0.0
                near_features = self.near_risk_channel_features(features, compat_score)
                score = float(self.learned_near_risk_weights @ near_features)
                pred_inhibit = score >= self.learned_near_risk_threshold
                target_inhibit_count += int(target_inhibit)
                pred_inhibit_count += int(pred_inhibit)
                if pred_inhibit != target_inhibit:
                    direction = 1.0 if target_inhibit else -1.0
                    self.learned_near_risk_weights += (
                        self.learned_near_risk_lr * direction * near_features
                    ).astype(np.float32)
                    updates += 1
        self.learned_near_risk_samples = len(train_samples)
        self.learned_near_risk_boundary_samples = boundary_samples
        self.learned_near_risk_epoch_samples = epoch_sample_count
        self.learned_near_risk_updates = updates
        auto_metrics = self.calibrate_learned_near_risk_threshold(train_samples)
        source_metrics = self.fit_learned_near_risk_source_channels(train_samples)
        return {
            "learned_near_risk_channel": 1,
            "learned_near_risk_samples": len(train_samples),
            "learned_near_risk_boundary_samples": boundary_samples,
            "learned_near_risk_epoch_samples": epoch_sample_count,
            "learned_near_risk_target_inhibit": target_inhibit_count,
            "learned_near_risk_pred_inhibit_before_update": pred_inhibit_count,
            "learned_near_risk_updates": updates,
            "learned_near_risk_threshold": self.learned_near_risk_threshold,
            "learned_near_risk_lr": self.learned_near_risk_lr,
            "learned_near_risk_epochs": self.learned_near_risk_epochs,
            "learned_near_risk_boundary_fraction": self.learned_near_risk_boundary_fraction,
            "learned_near_risk_balance_mode": self.learned_near_risk_balance_mode,
            "learned_near_risk_inhibit_per_allow": self.learned_near_risk_inhibit_per_allow,
            **auto_metrics,
            **source_metrics,
        }

    def fit(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        positive_features: list[np.ndarray] = []
        risk_features: list[np.ndarray] = []
        risk_by_class: dict[str, list[np.ndarray]] = defaultdict(list)
        counterfactual_risk_count = 0
        positive_by_class: dict[str, list[np.ndarray]] = defaultdict(list)
        near_risk_samples: list[tuple[np.ndarray, np.ndarray | None, bool, str]] = []
        for epoch in range(self.epochs):
            self._score_cache.clear()
            order = self.rng.permutation(len(rows))
            updates = 0
            compat_channel_updates = 0
            flip_candidates = 0
            target_allow = 0
            target_block = 0
            pred_allow = 0
            compat_channel_pred_allow = 0
            compat_channel_counterfactual_pred_allow = 0
            compat_channel_counterfactual_updates = 0
            for raw_idx in order:
                row = rows[int(raw_idx)]
                target_idx = self.answer_to_idx[str(row["answer"])]
                pre_scores = self.pre_model.scores(row).astype(np.float32, copy=False)
                post_scores = self.post_model.scores(row).astype(np.float32, copy=False)
                pre_pred_idx = int(np.argmax(pre_scores))
                post_pred_idx = int(np.argmax(post_scores))
                pre_loss = answer_loss(pre_scores, target_idx, self.temperature)
                if epoch == 0:
                    counterfactual = self.counterfactual_risk_features(
                        row,
                        pre_scores,
                        post_scores,
                        target_idx,
                        pre_loss,
                    )
                    for label, feature, compat_feature in counterfactual:
                        risk_features.append(feature[1:].astype(np.float32, copy=True))
                        risk_by_class[label].append(feature[1:].astype(np.float32, copy=True))
                        near_risk_samples.append((feature, compat_feature, True, label))
                        if compat_feature is not None:
                            compat_allow_pred = self.compat_channel_score(compat_feature) >= self.threshold
                            compat_channel_counterfactual_pred_allow += int(compat_allow_pred)
                            if compat_allow_pred:
                                self.compat_channel_weights -= (self.lr * compat_feature).astype(np.float32)
                                compat_channel_counterfactual_updates += 1
                    counterfactual_risk_count += len(counterfactual)
                if pre_pred_idx == post_pred_idx:
                    continue
                flip_candidates += 1
                post_loss = answer_loss(post_scores, target_idx, self.temperature)
                should_allow = post_loss + self.min_loss_gain < pre_loss
                features = self.features(row, pre_scores, post_scores)
                if epoch == 0 and should_allow:
                    pre_correct = pre_pred_idx == target_idx
                    post_correct = post_pred_idx == target_idx
                    flip_type = classify_flip(pre_correct, post_correct, True, post_loss - pre_loss)
                    positive_features.append(features[1:].astype(np.float32, copy=True))
                    positive_by_class[flip_type].append(features[1:].astype(np.float32, copy=True))
                    compat_feature = (
                        self.compat_channel_features(row, pre_scores, post_scores).astype(np.float32, copy=True)
                        if self.before_compat_channel
                        else None
                    )
                    near_risk_samples.append((features.astype(np.float32, copy=True), compat_feature, False, flip_type))
                if epoch == 0 and not should_allow:
                    pre_correct = pre_pred_idx == target_idx
                    post_correct = post_pred_idx == target_idx
                    flip_type = classify_flip(pre_correct, post_correct, True, post_loss - pre_loss)
                    risk_feature = features[1:].astype(np.float32, copy=True)
                    risk_features.append(risk_feature)
                    risk_by_class[f"natural_{flip_type}"].append(risk_feature)
                    compat_feature = (
                        self.compat_channel_features(row, pre_scores, post_scores).astype(np.float32, copy=True)
                        if self.before_compat_channel
                        else None
                    )
                    near_risk_samples.append((features.astype(np.float32, copy=True), compat_feature, True, flip_type))
                score = self.gate_score(features)
                allow = score >= self.threshold
                if self.before_compat_channel:
                    compat_features = self.compat_channel_features(row, pre_scores, post_scores)
                    compat_score = self.compat_channel_score(compat_features)
                    compat_allow_pred = compat_score >= self.threshold
                    compat_channel_pred_allow += int(compat_allow_pred)
                    if compat_allow_pred != should_allow:
                        direction = 1.0 if should_allow else -1.0
                        self.compat_channel_weights += (self.lr * direction * compat_features).astype(np.float32)
                        compat_channel_updates += 1
                target_allow += int(should_allow)
                target_block += int(not should_allow)
                pred_allow += int(allow)
                if allow != should_allow:
                    direction = 1.0 if should_allow else -1.0
                    self.weights += (self.lr * direction * features).astype(np.float32)
                    updates += 1
            metrics.append(
                {
                    "epoch": epoch,
                    "rows": len(rows),
                    "flip_candidates": flip_candidates,
                    "target_allow": target_allow,
                    "target_block": target_block,
                    "pred_allow_before_update": pred_allow,
                    "updates": updates,
                    "compat_channel_pred_allow_before_update": compat_channel_pred_allow,
                    "compat_channel_updates": compat_channel_updates,
                    "compat_channel_counterfactual_pred_allow": compat_channel_counterfactual_pred_allow,
                    "compat_channel_counterfactual_updates": compat_channel_counterfactual_updates,
                    "lr": self.lr,
                    "threshold": self.threshold,
                    "min_loss_gain": self.min_loss_gain,
                    "gate_mode": self.gate_mode,
                    "radius_scale": self.radius_scale,
                    "risk_radius_scale": self.risk_radius_scale,
                    "counterfactual_top_k": self.counterfactual_top_k,
                    "counterfactual_risk_count": self.counterfactual_risk_count,
                    "identity_features": int(self.identity_features),
                    "risk_class_prototypes": int(self.risk_class_prototypes_enabled),
                    "risk_radius_quantile": self.risk_radius_quantile,
                    "risk_micro_prototypes": self.risk_micro_prototypes,
                    "before_compat_features": int(self.before_compat_features),
                    "before_compat_filter": self.before_compat_filter,
                    "before_compat_margin": self.before_compat_margin,
                    "before_compat_channel": int(self.before_compat_channel),
                    "risk_near_margin": self.risk_near_margin,
                    "risk_near_compat_threshold": self.risk_near_compat_threshold,
                    "risk_near_radius_fraction": self.risk_near_radius_fraction,
                    "risk_near_blocks_rescue": int(self.risk_near_blocks_rescue),
                    "learned_near_risk_channel": int(self.learned_near_risk_channel),
                    "learned_near_risk_samples": self.learned_near_risk_samples,
                    "learned_near_risk_boundary_samples": self.learned_near_risk_boundary_samples,
                    "learned_near_risk_epoch_samples": self.learned_near_risk_epoch_samples,
                    "learned_near_risk_updates": self.learned_near_risk_updates,
                    "learned_near_risk_threshold": self.learned_near_risk_threshold,
                    "learned_near_risk_lr": self.learned_near_risk_lr,
                    "learned_near_risk_epochs": self.learned_near_risk_epochs,
                    "learned_near_risk_boundary_fraction": self.learned_near_risk_boundary_fraction,
                    "learned_near_risk_balance_mode": self.learned_near_risk_balance_mode,
                    "learned_near_risk_inhibit_per_allow": self.learned_near_risk_inhibit_per_allow,
                    "learned_near_risk_auto_threshold_mode": self.learned_near_risk_auto_threshold,
                    "learned_near_risk_auto_threshold_candidates": json.dumps(
                        self.learned_near_risk_auto_threshold_candidates
                    ),
                    "learned_near_risk_auto_selected_threshold": self.learned_near_risk_auto_selected_threshold,
                    "learned_near_risk_auto_calibration_samples": self.learned_near_risk_auto_calibration_samples,
                    "learned_near_risk_auto_allow_samples": self.learned_near_risk_auto_allow_samples,
                    "learned_near_risk_auto_inhibit_samples": self.learned_near_risk_auto_inhibit_samples,
                    "learned_near_risk_auto_balanced_accuracy": self.learned_near_risk_auto_balanced_accuracy,
                    "learned_near_risk_auto_allow_recall": self.learned_near_risk_auto_allow_recall,
                    "learned_near_risk_auto_inhibit_recall": self.learned_near_risk_auto_inhibit_recall,
                    "learned_near_risk_auto_false_inhibit": self.learned_near_risk_auto_false_inhibit,
                    "learned_near_risk_auto_false_allow": self.learned_near_risk_auto_false_allow,
                    "learned_near_risk_source_channel_mode": self.learned_near_risk_source_channels,
                    "learned_near_risk_source_channel_counts": json.dumps(
                        self.learned_near_risk_source_counts, sort_keys=True
                    ),
                    "learned_near_risk_source_channel_thresholds": json.dumps(
                        self.learned_near_risk_source_thresholds, sort_keys=True
                    ),
                    "learned_near_risk_source_channel_updates": json.dumps(
                        self.learned_near_risk_source_updates, sort_keys=True
                    ),
                    "learned_near_risk_source_channel_calibration": json.dumps(
                        self.learned_near_risk_source_calibration, sort_keys=True
                    ),
                    "positive_count": self.positive_count,
                    "positive_radius": self.positive_radius,
                    "risk_count": self.risk_count,
                    "risk_radius": self.risk_radius,
                    "risk_class_counts": json.dumps(self.risk_class_counts, sort_keys=True),
                    "risk_class_radii": json.dumps(self.risk_class_radii, sort_keys=True),
                    "class_counts": json.dumps(self.class_counts, sort_keys=True),
                    "class_radii": json.dumps(self.class_radii, sort_keys=True),
                }
            )
        if positive_features:
            stacked = np.stack(positive_features).astype(np.float32)
            self.positive_prototype = np.mean(stacked, axis=0).astype(np.float32)
            distances = np.linalg.norm(stacked - self.positive_prototype[None, :], axis=1)
            self.positive_radius = float(np.max(distances)) if len(distances) else 0.0
            self.positive_count = int(len(positive_features))
            for row in metrics:
                row["positive_count"] = self.positive_count
                row["positive_radius"] = self.positive_radius
        if risk_features:
            stacked = np.stack(risk_features).astype(np.float32)
            self.risk_prototype = np.mean(stacked, axis=0).astype(np.float32)
            distances = np.linalg.norm(stacked - self.risk_prototype[None, :], axis=1)
            self.risk_radius = self.prototype_radius(distances)
            self.risk_count = int(len(risk_features))
            self.counterfactual_risk_count = int(counterfactual_risk_count)
            for row in metrics:
                row["risk_count"] = self.risk_count
                row["risk_radius"] = self.risk_radius
                row["counterfactual_risk_count"] = self.counterfactual_risk_count
        self.risk_class_prototypes = {}
        self.risk_class_radii = {}
        self.risk_class_counts = {}
        for label, values in risk_by_class.items():
            stacked = np.stack(values).astype(np.float32)
            prototypes, radii, counts = self.risk_micro_cluster(stacked)
            if len(prototypes) == 1:
                self.risk_class_prototypes[label] = prototypes[0]
                self.risk_class_radii[label] = radii[0]
                self.risk_class_counts[label] = counts[0]
            else:
                self.risk_class_prototypes[label] = prototypes
                self.risk_class_radii[label] = [float(value) for value in radii]
                self.risk_class_counts[label] = [int(value) for value in counts]
        self.class_prototypes = {}
        self.class_radii = {}
        self.class_counts = {}
        for label, values in positive_by_class.items():
            stacked = np.stack(values).astype(np.float32)
            prototype = np.mean(stacked, axis=0).astype(np.float32)
            distances = np.linalg.norm(stacked - prototype[None, :], axis=1)
            self.class_prototypes[label] = prototype
            self.class_radii[label] = float(np.max(distances)) if len(distances) else 0.0
            self.class_counts[label] = int(len(values))
        near_metrics = self.fit_learned_near_risk_channel(near_risk_samples)
        for row in metrics:
            row["class_counts"] = json.dumps(self.class_counts, sort_keys=True)
            row["class_radii"] = json.dumps(self.class_radii, sort_keys=True)
            row["risk_class_counts"] = json.dumps(self.risk_class_counts, sort_keys=True)
            row["risk_class_radii"] = json.dumps(self.risk_class_radii, sort_keys=True)
            row.update(near_metrics)
        self._score_cache.clear()
        return metrics

    def scores(self, row: dict[str, Any]) -> np.ndarray:
        return self.row_record(row)["gate_scores"]

    def gate_summary_rows(self, splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        rows_out: list[dict[str, Any]] = []
        for split, rows in splits.items():
            pre_correct = 0
            post_correct = 0
            gate_correct = 0
            pre_losses: list[float] = []
            post_losses: list[float] = []
            gate_losses: list[float] = []
            flip_candidates = 0
            gate_allowed = 0
            gate_blocked = 0
            target_allow = 0
            target_block = 0
            helpful_allowed = 0
            helpful_blocked = 0
            harmful_allowed = 0
            harmful_blocked = 0
            wrong_improved_allowed = 0
            wrong_improved_blocked = 0
            wrong_worse_allowed = 0
            wrong_worse_blocked = 0
            for row in rows:
                target_idx = self.answer_to_idx[str(row["answer"])]
                record = self.row_record(row)
                pre_scores = record["pre_scores"]
                post_scores = record["post_scores"]
                gate_scores = record["gate_scores"]
                pre_pred = int(record["pre_pred"])
                post_pred = int(record["post_pred"])
                pre_loss = answer_loss(pre_scores, target_idx, self.temperature)
                post_loss = answer_loss(post_scores, target_idx, self.temperature)
                pre_losses.append(pre_loss)
                post_losses.append(post_loss)
                pre_correct_flag = pre_pred == target_idx
                post_correct_flag = post_pred == target_idx
                pre_correct += int(pre_correct_flag)
                post_correct += int(post_correct_flag)
                if pre_pred != post_pred:
                    flip_candidates += 1
                    should_allow = post_loss + self.min_loss_gain < pre_loss
                    allow = bool(record["allow"])
                    target_allow += int(should_allow)
                    target_block += int(not should_allow)
                    gate_allowed += int(allow)
                    gate_blocked += int(not allow)
                    flip_type = classify_flip(pre_correct_flag, post_correct_flag, True, post_loss - pre_loss)
                    if flip_type == "helpful_flip":
                        helpful_allowed += int(allow)
                        helpful_blocked += int(not allow)
                    elif flip_type == "harmful_flip":
                        harmful_allowed += int(allow)
                        harmful_blocked += int(not allow)
                    elif flip_type == "wrong_flip_ce_improved":
                        wrong_improved_allowed += int(allow)
                        wrong_improved_blocked += int(not allow)
                    elif flip_type == "wrong_flip_ce_worse":
                        wrong_worse_allowed += int(allow)
                        wrong_worse_blocked += int(not allow)
                gate_pred = int(record["gate_pred"])
                gate_correct += int(gate_pred == target_idx)
                gate_losses.append(answer_loss(gate_scores, target_idx, self.temperature))
            total = len(rows)
            rows_out.append(
                {
                    "method": "qa_credit_seeded_flip_gate",
                    "split": split,
                    "rows": total,
                    "pre_accuracy": pre_correct / max(total, 1),
                    "post_accuracy": post_correct / max(total, 1),
                    "gate_accuracy": gate_correct / max(total, 1),
                    "pre_loss": float(np.mean(pre_losses)) if pre_losses else 0.0,
                    "post_loss": float(np.mean(post_losses)) if post_losses else 0.0,
                    "gate_loss": float(np.mean(gate_losses)) if gate_losses else 0.0,
                    "flip_candidates": flip_candidates,
                    "target_allow": target_allow,
                    "target_block": target_block,
                    "gate_allowed": gate_allowed,
                    "gate_blocked": gate_blocked,
                    "helpful_allowed": helpful_allowed,
                    "helpful_blocked": helpful_blocked,
                    "harmful_allowed": harmful_allowed,
                    "harmful_blocked": harmful_blocked,
                    "wrong_improved_allowed": wrong_improved_allowed,
                    "wrong_improved_blocked": wrong_improved_blocked,
                    "wrong_worse_allowed": wrong_worse_allowed,
                    "wrong_worse_blocked": wrong_worse_blocked,
                    "gate_mode": self.gate_mode,
                    "radius_scale": self.radius_scale,
                    "risk_radius_scale": self.risk_radius_scale,
                    "counterfactual_top_k": self.counterfactual_top_k,
                    "counterfactual_risk_count": self.counterfactual_risk_count,
                    "identity_features": int(self.identity_features),
                    "risk_class_prototypes": int(self.risk_class_prototypes_enabled),
                    "risk_radius_quantile": self.risk_radius_quantile,
                    "risk_micro_prototypes": self.risk_micro_prototypes,
                    "before_compat_features": int(self.before_compat_features),
                    "before_compat_filter": self.before_compat_filter,
                    "before_compat_margin": self.before_compat_margin,
                    "before_compat_channel": int(self.before_compat_channel),
                    "risk_near_margin": self.risk_near_margin,
                    "risk_near_compat_threshold": self.risk_near_compat_threshold,
                    "risk_near_radius_fraction": self.risk_near_radius_fraction,
                    "risk_near_blocks_rescue": int(self.risk_near_blocks_rescue),
                    "learned_near_risk_channel": int(self.learned_near_risk_channel),
                    "learned_near_risk_samples": self.learned_near_risk_samples,
                    "learned_near_risk_boundary_samples": self.learned_near_risk_boundary_samples,
                    "learned_near_risk_epoch_samples": self.learned_near_risk_epoch_samples,
                    "learned_near_risk_updates": self.learned_near_risk_updates,
                    "learned_near_risk_threshold": self.learned_near_risk_threshold,
                    "learned_near_risk_lr": self.learned_near_risk_lr,
                    "learned_near_risk_epochs": self.learned_near_risk_epochs,
                    "learned_near_risk_boundary_fraction": self.learned_near_risk_boundary_fraction,
                    "learned_near_risk_balance_mode": self.learned_near_risk_balance_mode,
                    "learned_near_risk_inhibit_per_allow": self.learned_near_risk_inhibit_per_allow,
                    "learned_near_risk_auto_threshold_mode": self.learned_near_risk_auto_threshold,
                    "learned_near_risk_auto_threshold_candidates": json.dumps(
                        self.learned_near_risk_auto_threshold_candidates
                    ),
                    "learned_near_risk_auto_selected_threshold": self.learned_near_risk_auto_selected_threshold,
                    "learned_near_risk_auto_calibration_samples": self.learned_near_risk_auto_calibration_samples,
                    "learned_near_risk_auto_allow_samples": self.learned_near_risk_auto_allow_samples,
                    "learned_near_risk_auto_inhibit_samples": self.learned_near_risk_auto_inhibit_samples,
                    "learned_near_risk_auto_balanced_accuracy": self.learned_near_risk_auto_balanced_accuracy,
                    "learned_near_risk_auto_allow_recall": self.learned_near_risk_auto_allow_recall,
                    "learned_near_risk_auto_inhibit_recall": self.learned_near_risk_auto_inhibit_recall,
                    "learned_near_risk_auto_false_inhibit": self.learned_near_risk_auto_false_inhibit,
                    "learned_near_risk_auto_false_allow": self.learned_near_risk_auto_false_allow,
                    "learned_near_risk_source_channel_mode": self.learned_near_risk_source_channels,
                    "learned_near_risk_source_channel_counts": json.dumps(
                        self.learned_near_risk_source_counts, sort_keys=True
                    ),
                    "learned_near_risk_source_channel_thresholds": json.dumps(
                        self.learned_near_risk_source_thresholds, sort_keys=True
                    ),
                    "learned_near_risk_source_channel_updates": json.dumps(
                        self.learned_near_risk_source_updates, sort_keys=True
                    ),
                    "learned_near_risk_source_channel_calibration": json.dumps(
                        self.learned_near_risk_source_calibration, sort_keys=True
                    ),
                    "positive_count": self.positive_count,
                    "positive_radius": self.positive_radius,
                    "risk_count": self.risk_count,
                    "risk_radius": self.risk_radius,
                    "risk_class_counts": json.dumps(self.risk_class_counts, sort_keys=True),
                    "risk_class_radii": json.dumps(self.risk_class_radii, sort_keys=True),
                    "class_counts": json.dumps(self.class_counts, sort_keys=True),
                    "class_radii": json.dumps(self.class_radii, sort_keys=True),
                }
            )
        return rows_out

    def gate_diagnostic_rows(
        self,
        config: str,
        train_strength: str,
        eval_strength: str,
        split: str,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for example_index, row in enumerate(rows):
            target_idx = self.answer_to_idx[str(row["answer"])]
            record = self.row_record(row)
            pre_scores = record["pre_scores"]
            post_scores = record["post_scores"]
            gate_scores = record["gate_scores"]
            pre_pred = int(record["pre_pred"])
            post_pred = int(record["post_pred"])
            if pre_pred == post_pred:
                continue
            gate_pred = int(record["gate_pred"])
            pre_loss = answer_loss(pre_scores, target_idx, self.temperature)
            post_loss = answer_loss(post_scores, target_idx, self.temperature)
            gate_loss = answer_loss(gate_scores, target_idx, self.temperature)
            pre_correct = pre_pred == target_idx
            post_correct = post_pred == target_idx
            gate_correct = gate_pred == target_idx
            features = self.features(row, pre_scores, post_scores)
            compat_features = self.compat_channel_features(row, pre_scores, post_scores)
            compat_score = self.compat_channel_score(compat_features)
            risk_detail = self.nearest_risk_detail(features)
            raw_risk_match = self.risk_match(features)
            risk_near_match = self.risk_near_match_from_detail(risk_detail, compat_score, raw_risk_match)
            learned_near_score = self.near_risk_channel_score(features, compat_score)
            source_scores = self.learned_near_risk_source_scores(features, compat_score)
            matched_source_groups = [
                group
                for group, score in source_scores.items()
                if score >= self.learned_near_risk_source_thresholds.get(group, self.learned_near_risk_threshold)
            ]
            source_max_score = max(source_scores.values()) if source_scores else 0.0
            learned_near_match = self.learned_near_risk_match_from_features(features, compat_score, raw_risk_match)
            risk_near_buffer = self.risk_near_buffer_from_detail(risk_detail)
            out.append(
                {
                    "config": config,
                    "train_strength": train_strength,
                    "eval_strength": eval_strength,
                    "split": split,
                    "example_index": example_index,
                    "target": row["answer"],
                    "pre_prediction": self.idx_to_answer[pre_pred],
                    "post_prediction": self.idx_to_answer[post_pred],
                    "gate_prediction": self.idx_to_answer[gate_pred],
                    "pre_correct": int(pre_correct),
                    "post_correct": int(post_correct),
                    "gate_correct": int(gate_correct),
                    "gate_allowed": int(bool(record["allow"])),
                    "should_allow": int(post_loss + self.min_loss_gain < pre_loss),
                    "flip_type": classify_flip(pre_correct, post_correct, True, post_loss - pre_loss),
                    "pre_loss": pre_loss,
                    "post_loss": post_loss,
                    "gate_loss": gate_loss,
                    "post_minus_pre_loss": post_loss - pre_loss,
                    "gate_minus_pre_loss": gate_loss - pre_loss,
                    "pre_margin": answer_margin(pre_scores, self.answer_to_idx),
                    "post_margin": answer_margin(post_scores, self.answer_to_idx),
                    "compat_score": compat_score,
                    "compat_channel_allow": int(compat_score >= self.threshold),
                    "compat_filter_allow": int(self.before_compat_allow(row, pre_scores, post_scores)),
                    "raw_risk_match": int(raw_risk_match),
                    "risk_near_match": int(risk_near_match),
                    "learned_near_risk_score": learned_near_score,
                    "learned_near_risk_source_scores": json.dumps(source_scores, sort_keys=True),
                    "learned_near_risk_source_max_score": source_max_score,
                    "learned_near_risk_source_matched_groups": json.dumps(matched_source_groups),
                    "learned_near_risk_match": int(learned_near_match),
                    "effective_risk_match": int(raw_risk_match or risk_near_match or learned_near_match),
                    "risk_near_buffer": risk_near_buffer,
                    "risk_near_margin_threshold": self.risk_near_margin,
                    "risk_near_compat_threshold": self.risk_near_compat_threshold,
                    "risk_near_radius_fraction": self.risk_near_radius_fraction,
                    "risk_near_blocks_rescue": int(self.risk_near_blocks_rescue),
                    "learned_near_risk_channel": int(self.learned_near_risk_channel),
                    "learned_near_risk_threshold": self.learned_near_risk_threshold,
                    "learned_near_risk_boundary_fraction": self.learned_near_risk_boundary_fraction,
                    "learned_near_risk_balance_mode": self.learned_near_risk_balance_mode,
                    "learned_near_risk_inhibit_per_allow": self.learned_near_risk_inhibit_per_allow,
                    "learned_near_risk_auto_threshold_mode": self.learned_near_risk_auto_threshold,
                    "learned_near_risk_auto_selected_threshold": self.learned_near_risk_auto_selected_threshold,
                    "learned_near_risk_auto_calibration_samples": self.learned_near_risk_auto_calibration_samples,
                    "learned_near_risk_auto_allow_samples": self.learned_near_risk_auto_allow_samples,
                    "learned_near_risk_auto_inhibit_samples": self.learned_near_risk_auto_inhibit_samples,
                    "learned_near_risk_auto_balanced_accuracy": self.learned_near_risk_auto_balanced_accuracy,
                    "learned_near_risk_auto_allow_recall": self.learned_near_risk_auto_allow_recall,
                    "learned_near_risk_auto_inhibit_recall": self.learned_near_risk_auto_inhibit_recall,
                    "learned_near_risk_source_channel_mode": self.learned_near_risk_source_channels,
                    "learned_near_risk_source_channel_thresholds": json.dumps(
                        self.learned_near_risk_source_thresholds, sort_keys=True
                    ),
                    "gate_mode": self.gate_mode,
                    "threshold": self.threshold,
                    "radius_scale": self.radius_scale,
                    "risk_radius_scale": self.risk_radius_scale,
                    **risk_detail,
                    **self.nearest_positive_detail(features),
                }
            )
        return out

    def state_bytes(self) -> int:
        state = {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "compat_channel_feature_names": self.compat_channel_feature_names,
            "compat_channel_weights": self.compat_channel_weights,
            "near_risk_channel_feature_names": self.near_risk_channel_feature_names,
            "learned_near_risk_weights": self.learned_near_risk_weights,
            "learned_near_risk_channel": self.learned_near_risk_channel,
            "learned_near_risk_samples": self.learned_near_risk_samples,
            "learned_near_risk_boundary_samples": self.learned_near_risk_boundary_samples,
            "learned_near_risk_epoch_samples": self.learned_near_risk_epoch_samples,
            "learned_near_risk_updates": self.learned_near_risk_updates,
            "learned_near_risk_threshold": self.learned_near_risk_threshold,
            "learned_near_risk_lr": self.learned_near_risk_lr,
            "learned_near_risk_epochs": self.learned_near_risk_epochs,
            "learned_near_risk_boundary_fraction": self.learned_near_risk_boundary_fraction,
            "learned_near_risk_balance_mode": self.learned_near_risk_balance_mode,
            "learned_near_risk_inhibit_per_allow": self.learned_near_risk_inhibit_per_allow,
            "learned_near_risk_auto_threshold": self.learned_near_risk_auto_threshold,
            "learned_near_risk_auto_threshold_candidates": self.learned_near_risk_auto_threshold_candidates,
            "learned_near_risk_auto_selected_threshold": self.learned_near_risk_auto_selected_threshold,
            "learned_near_risk_auto_calibration_samples": self.learned_near_risk_auto_calibration_samples,
            "learned_near_risk_auto_allow_samples": self.learned_near_risk_auto_allow_samples,
            "learned_near_risk_auto_inhibit_samples": self.learned_near_risk_auto_inhibit_samples,
            "learned_near_risk_auto_balanced_accuracy": self.learned_near_risk_auto_balanced_accuracy,
            "learned_near_risk_auto_allow_recall": self.learned_near_risk_auto_allow_recall,
            "learned_near_risk_auto_inhibit_recall": self.learned_near_risk_auto_inhibit_recall,
            "learned_near_risk_auto_false_inhibit": self.learned_near_risk_auto_false_inhibit,
            "learned_near_risk_auto_false_allow": self.learned_near_risk_auto_false_allow,
            "learned_near_risk_source_channels": self.learned_near_risk_source_channels,
            "learned_near_risk_source_weights": self.learned_near_risk_source_weights,
            "learned_near_risk_source_thresholds": self.learned_near_risk_source_thresholds,
            "learned_near_risk_source_counts": self.learned_near_risk_source_counts,
            "learned_near_risk_source_updates": self.learned_near_risk_source_updates,
            "learned_near_risk_source_calibration": self.learned_near_risk_source_calibration,
            "positive_prototype": self.positive_prototype,
            "positive_radius": self.positive_radius,
            "positive_count": self.positive_count,
            "risk_prototype": self.risk_prototype,
            "risk_radius": self.risk_radius,
            "risk_count": self.risk_count,
            "counterfactual_risk_count": self.counterfactual_risk_count,
            "risk_class_prototypes": self.risk_class_prototypes,
            "risk_class_radii": self.risk_class_radii,
            "risk_class_counts": self.risk_class_counts,
            "class_prototypes": self.class_prototypes,
            "class_radii": self.class_radii,
            "class_counts": self.class_counts,
            "gate_mode": self.gate_mode,
            "radius_scale": self.radius_scale,
            "risk_radius_scale": self.risk_radius_scale,
            "counterfactual_top_k": self.counterfactual_top_k,
            "counterfactual_margin": self.counterfactual_margin,
            "identity_features": self.identity_features,
            "risk_class_prototypes_enabled": self.risk_class_prototypes_enabled,
            "risk_radius_quantile": self.risk_radius_quantile,
            "risk_micro_prototypes": self.risk_micro_prototypes,
            "before_compat_features": self.before_compat_features,
            "before_compat_filter": self.before_compat_filter,
            "before_compat_margin": self.before_compat_margin,
            "before_compat_channel": self.before_compat_channel,
            "risk_near_margin": self.risk_near_margin,
            "risk_near_compat_threshold": self.risk_near_compat_threshold,
            "risk_near_radius_fraction": self.risk_near_radius_fraction,
            "risk_near_blocks_rescue": self.risk_near_blocks_rescue,
            "threshold": self.threshold,
            "min_loss_gain": self.min_loss_gain,
            "feature_scale": self.feature_scale,
        }
        return self.pre_model.state_bytes() + self.post_model.state_bytes() + len(
            pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        )

    def active_features(self) -> str:
        return (
            f"flip_gate_features={len(self.feature_names)};mode={self.gate_mode};"
            f"compat_channel={int(self.before_compat_channel)};"
            f"risk_near_margin={self.risk_near_margin:.3f};"
            f"risk_near_radius_fraction={self.risk_near_radius_fraction:.3f};"
            f"risk_near_blocks_rescue={int(self.risk_near_blocks_rescue)};"
            f"learned_near_risk={int(self.learned_near_risk_channel)};"
            f"learned_near_balance={self.learned_near_risk_balance_mode};"
            f"learned_near_auto={self.learned_near_risk_auto_threshold};"
            f"learned_near_source={self.learned_near_risk_source_channels};"
            f"learned_near_threshold={self.learned_near_risk_threshold:.3f};"
            f"risk_count={self.risk_count};cf_risk={self.counterfactual_risk_count}"
        )


def run_one(
    config: str,
    train_strength: str,
    eval_strength: str,
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    config_dir = args.data_dir / config
    train_base = read_jsonl(config_dir / "train.jsonl", args.train_limit or None)
    val_base = read_jsonl(config_dir / "validation.jsonl", args.eval_limit or None)
    test_base = read_jsonl(config_dir / "test.jsonl", args.eval_limit or None)
    calibration_base: list[dict[str, Any]] = []
    train_fit_base = train_base
    calibration_ratio = float(np.clip(float(args.train_calibration_ratio), 0.0, 0.9))
    if calibration_ratio > 0.0 and len(train_base) > 1:
        rng = np.random.default_rng(args.seed + 1701)
        calibration_count = int(round(len(train_base) * calibration_ratio))
        calibration_count = min(max(calibration_count, 1), len(train_base) - 1)
        calibration_indices = set(int(idx) for idx in rng.permutation(len(train_base))[:calibration_count])
        train_fit_base = [row for idx, row in enumerate(train_base) if idx not in calibration_indices]
        calibration_base = [row for idx, row in enumerate(train_base) if idx in calibration_indices]
    original_train = paraphrase_rows(train_fit_base, "none")
    credit_train = paraphrase_rows(train_fit_base, train_strength)
    train_calibration_rows = paraphrase_rows(calibration_base, train_strength) if calibration_base else []
    validation_rows = paraphrase_rows(val_base, eval_strength)
    test_rows = paraphrase_rows(test_base, eval_strength)
    answer_vocab = build_answer_vocab(original_train, credit_train, train_calibration_rows, validation_rows, test_rows)
    answer_to_idx = {answer: idx for idx, answer in enumerate(answer_vocab)}
    splits = {"train": credit_train, "validation": validation_rows, "test": test_rows}
    if train_calibration_rows:
        splits["train_calibration"] = train_calibration_rows

    majority = MajorityBaseline(answer_to_idx)
    majority.fit(credit_train)

    methods: list[tuple[str, RoleBindingStateQALearner | MajorityBaseline, bool, str]] = [
        ("majority_no_memory", majority, False, "baseline"),
    ]
    detector_models: list[tuple[str, RoleBindingStateQALearner]] = []
    credit_rows: list[dict[str, Any]] = []
    flip_gate_rows: list[dict[str, Any]] = []
    flip_gate_diag_rows: list[dict[str, Any]] = []
    seeded_pre_for_flip: RoleBindingStateQALearner | None = None
    seeded_post_for_flip: RoleBindingStateQALearner | None = None

    original_seed = build_learned_role_model(answer_to_idx, majority, original_train, args, aware=False)
    methods.append(("r097_original_structural_seed", original_seed, False, "local_structural_seed_baseline"))
    detector_models.append(("r097_original_structural_seed", original_seed))

    if args.include_structural_upper:
        strong_upper = build_learned_role_model(answer_to_idx, majority, credit_train, args, aware=False)
        methods.append(("r097_same_surface_structural_upper", strong_upper, False, "local_structural_upper_bound"))
        detector_models.append(("r097_same_surface_structural_upper", strong_upper))

    if args.seed_mode in {"cold", "both"}:
        cold_model, cold_credit = build_credit_model(answer_to_idx, majority, None, credit_train, args, args.seed + 701)
        methods.append(("qa_credit_cold_answer_only", cold_model, False, "pure_no_bp_answer_credit"))
        detector_models.append(("qa_credit_cold_answer_only", cold_model))
        for row in cold_credit:
            credit_rows.append(
                {
                    **row,
                    "method": "qa_credit_cold_answer_only",
                    "config": config,
                    "train_strength": train_strength,
                    "eval_strength": eval_strength,
                    "seeded_with_structural_labels": False,
                }
            )

    if args.seed_mode in {"seeded", "both"}:
        seeded_pre_model, _ = build_credit_model(
            answer_to_idx,
            majority,
            original_train,
            credit_train,
            args,
            args.seed + 907,
            train_credit=False,
        )
        seeded_pre_for_flip = seeded_pre_model
        methods.append(("qa_credit_seeded_pre_credit", seeded_pre_model, False, "same_seed_structural_seed"))
        detector_models.append(("qa_credit_seeded_pre_credit", seeded_pre_model))
        seeded_model, seeded_credit = build_credit_model(
            answer_to_idx,
            majority,
            original_train,
            credit_train,
            args,
            args.seed + 907,
        )
        seeded_post_for_flip = seeded_model
        methods.append(("qa_credit_seeded_answer_only", seeded_model, False, "answer_credit_after_original_seed"))
        detector_models.append(("qa_credit_seeded_answer_only", seeded_model))
        for row in seeded_credit:
            credit_rows.append(
                {
                    **row,
                    "method": "qa_credit_seeded_answer_only",
                    "config": config,
                    "train_strength": train_strength,
                    "eval_strength": eval_strength,
                    "seeded_with_structural_labels": True,
                }
            )
        if args.enable_flip_gate:
            flip_gate_model = FlipGateReadoutModel(
                seeded_pre_model,
                seeded_model,
                answer_to_idx,
                args.temperature,
                args.flip_gate_lr,
                args.flip_gate_epochs,
                args.flip_gate_threshold,
                args.flip_gate_init_bias,
                args.flip_gate_min_loss_gain,
                args.flip_gate_feature_scale,
                args.flip_gate_mode,
                args.flip_gate_radius_scale,
                args.flip_gate_risk_radius_scale,
                args.flip_gate_counterfactual_top_k,
                args.flip_gate_counterfactual_margin,
                args.flip_gate_identity_features,
                args.flip_gate_risk_class_prototypes,
                args.flip_gate_risk_radius_quantile,
                args.flip_gate_risk_micro_prototypes,
                args.flip_gate_before_compat_features,
                args.flip_gate_before_compat_filter,
                args.flip_gate_before_compat_margin,
                args.flip_gate_before_compat_channel,
                args.flip_gate_risk_near_margin,
                args.flip_gate_risk_near_compat_threshold,
                args.flip_gate_risk_near_radius_fraction,
                args.flip_gate_risk_near_blocks_rescue,
                args.flip_gate_learned_near_risk_channel,
                args.flip_gate_learned_near_risk_lr,
                args.flip_gate_learned_near_risk_epochs,
                args.flip_gate_learned_near_risk_threshold,
                args.flip_gate_learned_near_risk_init_bias,
                args.flip_gate_learned_near_risk_boundary_fraction,
                args.flip_gate_learned_near_risk_balance_mode,
                args.flip_gate_learned_near_risk_inhibit_per_allow,
                args.flip_gate_learned_near_risk_source_channels,
                args.flip_gate_learned_near_risk_auto_threshold,
                args.flip_gate_learned_near_risk_auto_threshold_candidates,
                args.seed + 1301,
            )
            for row in flip_gate_model.fit(credit_train):
                flip_gate_rows.append(
                    {
                        **row,
                        "method": "qa_credit_seeded_flip_gate",
                        "metric_type": "train_gate",
                        "config": config,
                        "train_strength": train_strength,
                        "eval_strength": eval_strength,
                    }
                )
            for row in flip_gate_model.gate_summary_rows(splits):
                flip_gate_rows.append(
                    {
                        **row,
                        "metric_type": "eval_gate",
                        "config": config,
                        "train_strength": train_strength,
                        "eval_strength": eval_strength,
                    }
                )
            if args.flip_gate_risk_near_radius_fraction_sweep:
                original_fraction = flip_gate_model.risk_near_radius_fraction
                for fraction in args.flip_gate_risk_near_radius_fraction_sweep:
                    flip_gate_model.risk_near_radius_fraction = max(float(fraction), 0.0)
                    flip_gate_model._score_cache.clear()
                    for row in flip_gate_model.gate_summary_rows(splits):
                        flip_gate_rows.append(
                            {
                                **row,
                                "metric_type": "eval_gate_sweep",
                                "sweep_parameter": "risk_near_radius_fraction",
                                "sweep_value": flip_gate_model.risk_near_radius_fraction,
                                "config": config,
                                "train_strength": train_strength,
                                "eval_strength": eval_strength,
                            }
                        )
                flip_gate_model.risk_near_radius_fraction = original_fraction
                flip_gate_model._score_cache.clear()
            if args.flip_gate_learned_near_risk_threshold_sweep:
                original_threshold = flip_gate_model.learned_near_risk_threshold
                for threshold in args.flip_gate_learned_near_risk_threshold_sweep:
                    flip_gate_model.learned_near_risk_threshold = float(threshold)
                    flip_gate_model._score_cache.clear()
                    for row in flip_gate_model.gate_summary_rows(splits):
                        flip_gate_rows.append(
                            {
                                **row,
                                "metric_type": "eval_gate_sweep",
                                "sweep_parameter": "learned_near_risk_threshold",
                                "sweep_value": flip_gate_model.learned_near_risk_threshold,
                                "config": config,
                                "train_strength": train_strength,
                                "eval_strength": eval_strength,
                            }
                        )
                flip_gate_model.learned_near_risk_threshold = original_threshold
                flip_gate_model._score_cache.clear()
            if args.write_flip_gate_diagnostics:
                selected_splits = (
                    splits
                    if args.flip_diagnostic_split == "all"
                    else {args.flip_diagnostic_split: splits[args.flip_diagnostic_split]}
                )
                for split_name, split_rows in selected_splits.items():
                    flip_gate_diag_rows.extend(
                        flip_gate_model.gate_diagnostic_rows(
                            config,
                            train_strength,
                            eval_strength,
                            split_name,
                            split_rows,
                        )
                    )
            methods.append(("qa_credit_seeded_flip_gate", flip_gate_model, False, "train_split_local_flip_gate"))

    summary_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for name, model, stores_raw, method_type in methods:
        summary, preds = evaluate_method(name, model, splits, answer_to_idx, args.temperature, stores_raw, method_type)
        for row in summary:
            row.update({"config": config, "train_strength": train_strength, "eval_strength": eval_strength})
        for row in preds:
            row.update({"config": config, "train_strength": train_strength, "eval_strength": eval_strength})
        summary_rows.extend(summary)
        pred_rows.extend(preds)

    detector_rows: list[dict[str, Any]] = []
    for name, model in detector_models:
        detector_rows.extend(detector_summary_rows(name, model, splits, config, train_strength, eval_strength, args))

    flip_rows: list[dict[str, Any]] = []
    flip_summary: list[dict[str, Any]] = []
    if args.write_flip_diagnostics and seeded_pre_for_flip is not None and seeded_post_for_flip is not None:
        selected_splits = splits if args.flip_diagnostic_split == "all" else {args.flip_diagnostic_split: splits[args.flip_diagnostic_split]}
        for split_name, split_rows in selected_splits.items():
            flip_rows.extend(
                flip_diagnostic_rows(
                    config,
                    train_strength,
                    eval_strength,
                    split_name,
                    split_rows,
                    seeded_pre_for_flip,
                    seeded_post_for_flip,
                    answer_to_idx,
                    args.temperature,
                )
            )
        flip_summary = flip_summary_rows(flip_rows)

    config_rows = [
        {
            "config": config,
            "train_strength": train_strength,
            "eval_strength": eval_strength,
            "train_rows": len(credit_train),
            "train_calibration_rows": len(train_calibration_rows),
            "train_calibration_ratio": calibration_ratio,
            "validation_rows": len(validation_rows),
            "test_rows": len(test_rows),
        }
    ]
    return (
        summary_rows,
        pred_rows,
        detector_rows,
        credit_rows,
        config_rows,
        flip_rows,
        flip_summary,
        flip_gate_rows,
        flip_gate_diag_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--configs", nargs="+", default=["en-qa2", "en-qa3"])
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "babi_delayed_credit")
    parser.add_argument("--train-strength", choices=["none", "mild", "strong"], default="strong")
    parser.add_argument("--eval-strength", choices=["none", "mild", "strong"], default="strong")
    parser.add_argument("--seed-mode", choices=["cold", "seeded", "both"], default="both")
    parser.add_argument("--include-structural-upper", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--train-calibration-ratio", type=float, default=0.0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--credit-epochs", type=int, default=2)
    parser.add_argument("--max-event-updates-per-row", type=int, default=2)
    parser.add_argument("--max-credit-sentences-per-row", type=int, default=12)
    parser.add_argument("--candidate-cache-mode", choices=["exact", "cached"], default="exact")
    parser.add_argument("--credit-error-only", action="store_true")
    parser.add_argument("--max-credit-scale", type=float, default=0.0)
    parser.add_argument("--disable-query-credit", action="store_true")
    parser.add_argument("--disable-event-credit", action="store_true")
    parser.add_argument("--enable-before-relation-credit", action="store_true")
    parser.add_argument("--restrict-before-query-credit", action="store_true")
    parser.add_argument("--protect-query-credit-confidence", type=float, default=0.0)
    parser.add_argument("--min-credit-gain", type=float, default=1e-4)
    parser.add_argument("--role-dim", type=int, default=64)
    parser.add_argument("--role-lr", type=float, default=1.0)
    parser.add_argument("--role-score-scale", type=float, default=8.0)
    parser.add_argument("--role-carry-threshold", type=float, default=0.35)
    parser.add_argument("--before-credit-lr", type=float, default=0.5)
    parser.add_argument("--before-credit-weight", type=float, default=1.0)
    parser.add_argument("--before-credit-threshold", type=float, default=0.05)
    parser.add_argument("--before-credit-before-weight", type=float, default=0.75)
    parser.add_argument("--before-credit-current-weight", type=float, default=0.35)
    parser.add_argument(
        "--before-credit-gate-mode",
        choices=["always", "low_margin", "agree_top", "confidence"],
        default="always",
    )
    parser.add_argument("--before-credit-gate-margin", type=float, default=0.0)
    parser.add_argument("--before-credit-confidence-scale", type=float, default=1.0)
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--event-lr", type=float, default=0.08)
    parser.add_argument("--event-epochs", type=int, default=3)
    parser.add_argument("--event-score-scale", type=float, default=6.0)
    parser.add_argument("--event-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--event-eval-limit", type=int, default=0)
    parser.add_argument("--query-dim", type=int, default=64)
    parser.add_argument("--query-lr", type=float, default=0.08)
    parser.add_argument("--query-epochs", type=int, default=3)
    parser.add_argument("--query-score-scale", type=float, default=6.0)
    parser.add_argument("--query-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--before-relation-slot-features", action="store_true")
    parser.add_argument("--enable-query-subject-wta", action="store_true")
    parser.add_argument("--query-subject-wta-bonus", type=float, default=1.0)
    parser.add_argument("--query-subject-wta-min-margin", type=float, default=0.0)
    parser.add_argument("--query-eval-limit", type=int, default=0)
    parser.add_argument("--write-flip-diagnostics", action="store_true")
    parser.add_argument("--write-flip-gate-diagnostics", action="store_true")
    parser.add_argument("--flip-diagnostic-split", choices=["train", "validation", "test", "all"], default="test")
    parser.add_argument("--enable-flip-gate", action="store_true")
    parser.add_argument("--flip-gate-epochs", type=int, default=4)
    parser.add_argument("--flip-gate-lr", type=float, default=0.25)
    parser.add_argument("--flip-gate-threshold", type=float, default=0.0)
    parser.add_argument("--flip-gate-init-bias", type=float, default=-0.25)
    parser.add_argument("--flip-gate-min-loss-gain", type=float, default=0.0)
    parser.add_argument("--flip-gate-feature-scale", type=float, default=4.0)
    parser.add_argument(
        "--flip-gate-mode",
        choices=[
            "perceptron",
            "one_class",
            "class_prototype",
            "risk_only",
            "risk_compat_rescue",
            "risk_compat_positive_rescue",
            "risk_compat_class_rescue",
            "risk_prototype",
            "class_risk",
            "hybrid",
        ],
        default="perceptron",
    )
    parser.add_argument("--flip-gate-radius-scale", type=float, default=1.0)
    parser.add_argument("--flip-gate-risk-radius-scale", type=float, default=1.0)
    parser.add_argument("--flip-gate-counterfactual-top-k", type=int, default=0)
    parser.add_argument("--flip-gate-counterfactual-margin", type=float, default=1e-3)
    parser.add_argument("--flip-gate-identity-features", action="store_true")
    parser.add_argument("--flip-gate-risk-class-prototypes", action="store_true")
    parser.add_argument("--flip-gate-risk-radius-quantile", type=float, default=1.0)
    parser.add_argument("--flip-gate-risk-micro-prototypes", type=int, default=1)
    parser.add_argument("--flip-gate-before-compat-features", action="store_true")
    parser.add_argument(
        "--flip-gate-before-compat-filter",
        choices=["off", "post_top", "post_better", "post_close"],
        default="off",
    )
    parser.add_argument("--flip-gate-before-compat-margin", type=float, default=0.0)
    parser.add_argument("--flip-gate-before-compat-channel", action="store_true")
    parser.add_argument("--flip-gate-risk-near-margin", type=float, default=0.0)
    parser.add_argument("--flip-gate-risk-near-compat-threshold", type=float, default=0.25)
    parser.add_argument("--flip-gate-risk-near-radius-fraction", type=float, default=0.0)
    parser.add_argument("--flip-gate-risk-near-blocks-rescue", action="store_true")
    parser.add_argument("--flip-gate-risk-near-radius-fraction-sweep", nargs="*", type=float, default=[])
    parser.add_argument("--flip-gate-learned-near-risk-channel", action="store_true")
    parser.add_argument("--flip-gate-learned-near-risk-lr", type=float, default=0.05)
    parser.add_argument("--flip-gate-learned-near-risk-epochs", type=int, default=1)
    parser.add_argument("--flip-gate-learned-near-risk-threshold", type=float, default=0.0)
    parser.add_argument("--flip-gate-learned-near-risk-init-bias", type=float, default=-0.25)
    parser.add_argument("--flip-gate-learned-near-risk-boundary-fraction", type=float, default=0.2)
    parser.add_argument("--flip-gate-learned-near-risk-balance-mode", choices=["off", "resample"], default="off")
    parser.add_argument("--flip-gate-learned-near-risk-inhibit-per-allow", type=float, default=4.0)
    parser.add_argument(
        "--flip-gate-learned-near-risk-source-channels",
        choices=["off", "risk_source"],
        default="off",
    )
    parser.add_argument(
        "--flip-gate-learned-near-risk-auto-threshold",
        choices=["off", "balanced_train"],
        default="off",
    )
    parser.add_argument(
        "--flip-gate-learned-near-risk-auto-threshold-candidates",
        nargs="*",
        type=float,
        default=[],
    )
    parser.add_argument("--flip-gate-learned-near-risk-threshold-sweep", nargs="*", type=float, default=[])
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, Any]] = []
    all_preds: list[dict[str, Any]] = []
    all_detectors: list[dict[str, Any]] = []
    all_credit: list[dict[str, Any]] = []
    all_configs: list[dict[str, Any]] = []
    all_flip_rows: list[dict[str, Any]] = []
    all_flip_summary: list[dict[str, Any]] = []
    all_flip_gate: list[dict[str, Any]] = []
    all_flip_gate_diag: list[dict[str, Any]] = []
    for config in args.configs:
        summary, preds, detectors, credit, config_rows, flip_rows, flip_summary, flip_gate, flip_gate_diag = run_one(
            config,
            args.train_strength,
            args.eval_strength,
            args,
        )
        all_summary.extend(summary)
        all_preds.extend(preds)
        all_detectors.extend(detectors)
        all_credit.extend(credit)
        all_configs.extend(config_rows)
        all_flip_rows.extend(flip_rows)
        all_flip_summary.extend(flip_summary)
        all_flip_gate.extend(flip_gate)
        all_flip_gate_diag.extend(flip_gate_diag)

    write_csv(args.out_dir / "summary.csv", all_summary)
    write_csv(args.out_dir / "predictions_sample.csv", all_preds)
    write_csv(args.out_dir / "detector_metrics.csv", all_detectors)
    write_csv(args.out_dir / "credit_metrics.csv", all_credit)
    write_csv(args.out_dir / "run_configs.csv", all_configs)
    if args.write_flip_diagnostics:
        write_csv(args.out_dir / "flip_diagnostics.csv", all_flip_rows)
        write_csv(args.out_dir / "flip_summary.csv", all_flip_summary)
    if args.enable_flip_gate:
        write_csv(args.out_dir / "flip_gate_metrics.csv", all_flip_gate)
    if args.write_flip_gate_diagnostics:
        write_csv(args.out_dir / "flip_gate_diagnostics.csv", all_flip_gate_diag)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str, sort_keys=True)

    print("Summary:")
    for row in all_summary:
        if row["split"] == "test":
            print(
                f"  {row['config']} {row['method']}: "
                f"acc={float(row['accuracy']):.3f} loss={float(row['loss']):.3f}"
            )
    print(f"wrote {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
