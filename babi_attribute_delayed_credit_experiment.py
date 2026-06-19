#!/usr/bin/env python3
"""
Delayed answer-credit experiment for no-BP bAbI QA15/QA16 attribute binding.

R139 showed that the learned attribute front-end is fragile under unseen strong
paraphrases.  This script tests whether final QA answer error can adapt the
statement/query detectors on the paraphrased train stream without reading
paraphrase-local structure labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from babi_attribute_paraphrase_stress_experiment import (
    StressAttributeQueryDetector,
    StressAttributeStatementDetector,
    paraphrase_rows,
)
from babi_no_bp_qa_experiment import (
    ATTRIBUTE_EVENT_TO_IDX,
    ATTRIBUTE_QUERY_TO_IDX,
    COLOR_WORDS,
    DEFAULT_DATA_DIR,
    AttributeBindingStateQALearner,
    MajorityBaseline,
    build_answer_vocab,
    evaluate_method,
    normalize_vector,
    read_jsonl,
    singularize_noun,
    softmax,
    tokens,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
STOPWORDS = {
    "a",
    "an",
    "are",
    "as",
    "classified",
    "color",
    "colored",
    "fear",
    "is",
    "of",
    "scared",
    "the",
    "what",
    "which",
}


def answer_objective(scores: np.ndarray, target_idx: int, temperature: float) -> float:
    probs = softmax(scores.astype(np.float32, copy=False), temperature)
    return math.log(float(probs[target_idx]) + 1e-9)


def candidate_key(event: dict[str, str | None]) -> tuple[str | None, str | None, str | None]:
    return (event.get("event"), event.get("entity"), event.get("value"))


def query_key(query: dict[str, str | None]) -> tuple[str | None, str | None]:
    return (query.get("query"), query.get("subject"))


class CreditAttributeStatementDetector(StressAttributeStatementDetector):
    """Attribute statement detector with third-factor answer-credit updates."""

    def credit_update(self, sentence: str, target_event: dict[str, str | None], scale: float) -> float:
        feature = self.sentence_feature(sentence)
        raw_scores = self.event_weights @ feature
        pred_idx = int(np.argmax(raw_scores))
        target_idx = ATTRIBUTE_EVENT_TO_IDX[str(target_event["event"])]
        eta = self.lr * max(float(scale), 0.05)
        applied = 0.0
        if pred_idx != target_idx:
            self.event_weights[target_idx] += eta * feature
            self.event_weights[pred_idx] -= 0.5 * eta * feature
            self.event_weights[target_idx] = normalize_vector(self.event_weights[target_idx])
            self.event_weights[pred_idx] = normalize_vector(self.event_weights[pred_idx])
            applied += float(eta) * float(np.linalg.norm(feature))
        self.update_prototype(
            self.entity_prototypes,
            self.entity_counts,
            target_event.get("entity"),
            self.slot_feature(sentence, "entity"),
        )
        self.update_prototype(
            self.value_prototypes,
            self.value_counts,
            target_event.get("value"),
            self.slot_feature(sentence, "value"),
        )
        return applied

    def slot_consolidation_update(self, sentence: str, target_event: dict[str, str | None]) -> float:
        before = len(self.entity_prototypes) + len(self.value_prototypes)
        self.update_prototype(
            self.entity_prototypes,
            self.entity_counts,
            target_event.get("entity"),
            self.slot_feature(sentence, "entity"),
        )
        self.update_prototype(
            self.value_prototypes,
            self.value_counts,
            target_event.get("value"),
            self.slot_feature(sentence, "value"),
        )
        after = len(self.entity_prototypes) + len(self.value_prototypes)
        return float(max(after - before, 0))


class CreditAttributeQueryDetector(StressAttributeQueryDetector):
    """Attribute query detector with third-factor answer-credit updates."""

    def credit_update(self, question: str, target_query: dict[str, str | None], scale: float) -> float:
        feature = self.question_feature(question)
        raw_scores = self.query_weights @ feature
        pred_idx = int(np.argmax(raw_scores))
        target_idx = ATTRIBUTE_QUERY_TO_IDX[str(target_query["query"])]
        eta = self.lr * max(float(scale), 0.05)
        applied = 0.0
        if pred_idx != target_idx:
            self.query_weights[target_idx] += eta * feature
            self.query_weights[pred_idx] -= 0.5 * eta * feature
            self.query_weights[target_idx] = normalize_vector(self.query_weights[target_idx])
            self.query_weights[pred_idx] = normalize_vector(self.query_weights[pred_idx])
            applied += float(eta) * float(np.linalg.norm(feature))
        self.update_prototype(target_query.get("subject"), self.subject_feature(question))
        return applied


def first_token(words: list[str]) -> str | None:
    if not words:
        return None
    return words[0]


def last_value(words: list[str]) -> str | None:
    for word in reversed(words):
        if word not in STOPWORDS:
            return singularize_noun(word)
    return None


def candidate_attribute_events(sentence: str) -> list[dict[str, str | None]]:
    words = tokens(sentence)
    entity = first_token(words)
    value = last_value(words)
    candidates: list[dict[str, str | None]] = []
    has_fear_cue = bool({"fear", "afraid"} & set(words))
    has_class_cue = "classified" in words or " a " in f" {' '.join(words)} "
    has_color_cue = "colored" in words or bool(set(words) & COLOR_WORDS)
    if entity is not None and value is not None and has_fear_cue:
        candidates.append({"event": "class_afraid", "entity": singularize_noun(entity), "value": value})
    if entity is not None and value is not None and has_class_cue and not has_color_cue:
        candidates.append({"event": "entity_class", "entity": entity, "value": value})
    if entity is not None and has_color_cue:
        for color in sorted(set(words) & COLOR_WORDS):
            candidates.append({"event": "entity_color", "entity": entity, "value": color})
    unique: dict[tuple[str | None, str | None, str | None], dict[str, str | None]] = {}
    for event in candidates:
        unique[candidate_key(event)] = event
    return list(unique.values())


def subject_candidates(question: str) -> list[str]:
    words = tokens(question)
    candidates: list[str] = []
    for idx, word in enumerate(words):
        if word in STOPWORDS or word in COLOR_WORDS:
            continue
        if idx > 0 and words[idx - 1] == "is":
            candidates.append(word)
        elif len(word) > 2:
            candidates.append(word)
    return sorted(set(candidates))[:4]


def candidate_attribute_queries(question: str) -> list[dict[str, str | None]]:
    candidates: list[dict[str, str | None]] = []
    for subject in subject_candidates(question):
        candidates.append({"query": "afraid_of", "subject": subject})
        candidates.append({"query": "color", "subject": subject})
    if not candidates:
        candidates.append({"query": "none", "subject": None})
    unique: dict[tuple[str | None, str | None], dict[str, str | None]] = {}
    for query in candidates:
        unique[query_key(query)] = query
    return list(unique.values())


def apply_attribute_event(
    model: AttributeBindingStateQALearner,
    state: dict[str, Any],
    event: dict[str, str | None],
) -> None:
    event_type = event.get("event")
    entity = event.get("entity")
    value = event.get("value")
    if entity is None or value is None:
        return
    if event_type == "class_afraid":
        category = entity
        answer = value
        state["categories"].add(category)
        if answer in model.answer_codes:
            model.associate(state["category_afraid"], model.code(category, "category"), model.answer_codes[answer])
        return
    if event_type == "entity_class":
        category = value
        state["entities"].add(entity)
        state["categories"].add(category)
        model.associate(state["entity_category"], model.code(entity, "entity"), model.code(category, "category"))
        model.maybe_link_category_color(state, entity, category)
        return
    if event_type == "entity_color":
        color = value
        if color not in model.answer_codes:
            return
        state["entities"].add(entity)
        state["colored_entities"].add(entity)
        model.associate(state["entity_color"], model.code(entity, "entity"), model.answer_codes[color])
        model.maybe_link_category_color(state, entity, color=color)


def read_context_with_overrides(
    model: AttributeBindingStateQALearner,
    row: dict[str, Any],
    event_overrides: dict[int, dict[str, str | None]] | None = None,
) -> dict[str, Any]:
    state = model.new_state()
    overrides = event_overrides or {}
    for idx, item in enumerate(row["context"]):
        if idx in overrides:
            event = overrides[idx]
        else:
            event = model.detect_statement(str(item["text"]))
        apply_attribute_event(model, state, event)
    return state


def scores_for_query(
    model: AttributeBindingStateQALearner,
    state: dict[str, Any],
    row: dict[str, Any],
    query: dict[str, str | None],
) -> np.ndarray:
    subject = query.get("subject")
    if subject is None:
        return model.fallback.scores(row)
    if query.get("query") == "afraid_of":
        category = model.category_for_entity(state, subject)
        if category is None:
            return model.fallback.scores(row)
        value, conf = model.retrieve(state["category_afraid"], model.code(category, "category"))
        if conf <= model.confidence_threshold:
            return model.fallback.scores(row)
        _, score, scores = model.decode_answer_scores(value)
        return scores if score > 0.0 else model.fallback.scores(row)
    if query.get("query") == "color":
        if subject in state["colored_entities"]:
            direct_value, direct_conf = model.retrieve(state["entity_color"], model.code(subject, "entity"))
            if direct_conf > model.confidence_threshold:
                _, direct_score, direct_scores = model.decode_answer_scores(direct_value)
                if direct_score > 0.0:
                    return direct_scores
        category = model.category_for_entity(state, subject)
        if category is None:
            return model.fallback.scores(row)
        value, conf = model.retrieve(state["category_color"], model.code(category, "category"))
        if conf <= model.confidence_threshold:
            return model.fallback.scores(row)
        _, score, scores = model.decode_answer_scores(value)
        return scores if score > 0.0 else model.fallback.scores(row)
    return model.fallback.scores(row)


def scores_with_overrides(
    model: AttributeBindingStateQALearner,
    row: dict[str, Any],
    event_overrides: dict[int, dict[str, str | None]] | None = None,
    query_override: dict[str, str | None] | None = None,
) -> np.ndarray:
    state = read_context_with_overrides(model, row, event_overrides)
    query = query_override if query_override is not None else model.detect_query(str(row["question"]))
    return scores_for_query(model, state, row, query)


def credit_sentence_indices(row: dict[str, Any], max_sentences: int) -> list[int]:
    if max_sentences <= 0 or max_sentences >= len(row["context"]):
        return list(range(len(row["context"])))
    question_terms = set(subject_candidates(str(row["question"])))
    answer = str(row["answer"])
    scored: list[tuple[int, int]] = []
    for idx, item in enumerate(row["context"]):
        words = set(tokens(str(item["text"])))
        score = 0
        if answer in words or singularize_noun(answer) in {singularize_noun(word) for word in words}:
            score += 4
        if words & question_terms:
            score += 3
        if words & COLOR_WORDS:
            score += 2
        if score > 0:
            scored.append((score, idx))
    if not scored:
        return list(range(min(max_sentences, len(row["context"]))))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return sorted(idx for _, idx in scored[:max_sentences])


class AttributeAnswerCreditTrainer:
    def __init__(
        self,
        model: AttributeBindingStateQALearner,
        answer_to_idx: dict[str, int],
        temperature: float,
        min_gain: float,
        max_credit_scale: float,
        max_statement_updates: int,
        max_credit_sentences: int,
        enable_pair_statement_credit: bool,
        slot_consolidation_mode: str,
        disable_query_credit: bool,
        disable_statement_credit: bool,
        error_only: bool,
        seed: int,
    ) -> None:
        self.model = model
        self.answer_to_idx = answer_to_idx
        self.temperature = temperature
        self.min_gain = min_gain
        self.max_credit_scale = max_credit_scale
        self.max_statement_updates = max_statement_updates
        self.max_credit_sentences = max_credit_sentences
        self.enable_pair_statement_credit = enable_pair_statement_credit
        self.slot_consolidation_mode = slot_consolidation_mode
        self.disable_query_credit = disable_query_credit
        self.disable_statement_credit = disable_statement_credit
        self.error_only = error_only
        self.rng = np.random.default_rng(seed)

    def credit_scale(self, gain: float) -> float:
        if self.max_credit_scale > 0.0:
            return min(float(gain), self.max_credit_scale)
        return float(gain)

    def consolidate_slots(
        self,
        row: dict[str, Any],
        statement_detector: CreditAttributeStatementDetector,
        candidate_entries: list[tuple[int, dict[str, str | None]]],
    ) -> int:
        unique: dict[
            tuple[int, str | None, str | None, str | None],
            tuple[int, dict[str, str | None]],
        ] = {}
        for sent_idx, event in candidate_entries:
            unique[(sent_idx, event.get("event"), event.get("entity"), event.get("value"))] = (sent_idx, event)
        updates = 0
        for _, (sent_idx, event) in sorted(unique.items(), key=lambda item: item[0]):
            sentence = str(row["context"][sent_idx]["text"])
            statement_detector.slot_consolidation_update(sentence, event)
            updates += 1
        return updates

    def train(self, rows: list[dict[str, Any]], epochs: int) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        statement_detector = self.model.statement_detector
        query_detector = self.model.query_detector
        if not isinstance(statement_detector, CreditAttributeStatementDetector):
            return metrics
        if not isinstance(query_detector, CreditAttributeQueryDetector):
            return metrics
        for epoch in range(epochs):
            order = self.rng.permutation(len(rows))
            statement_updates = 0
            query_updates = 0
            total_gain = 0.0
            update_norm = 0.0
            skipped_correct = 0
            searched_statement_candidates = 0
            searched_pair_candidates = 0
            pair_credit_updates = 0
            slot_consolidation_updates = 0
            for raw_idx in order:
                row = rows[int(raw_idx)]
                target = self.answer_to_idx[row["answer"]]
                current_scores = scores_with_overrides(self.model, row)
                current = answer_objective(current_scores, target, self.temperature)
                row_correct = int(np.argmax(current_scores)) == target

                candidate_indices = credit_sentence_indices(row, self.max_credit_sentences)
                candidate_entries: list[tuple[int, dict[str, str | None]]] = []
                for sent_idx in candidate_indices:
                    sentence = str(row["context"][sent_idx]["text"])
                    for event in candidate_attribute_events(sentence):
                        candidate_entries.append((sent_idx, event))

                if self.slot_consolidation_mode == "all":
                    slot_consolidation_updates += self.consolidate_slots(row, statement_detector, candidate_entries)

                if self.error_only and row_correct:
                    skipped_correct += 1
                    continue

                if not self.disable_query_credit:
                    best_query: dict[str, str | None] | None = None
                    best_query_obj = current
                    for query in candidate_attribute_queries(str(row["question"])):
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
                        update_norm += query_detector.credit_update(
                            str(row["question"]), best_query, self.credit_scale(gain)
                        )
                        query_updates += 1
                        total_gain += gain
                        current = best_query_obj

                if self.disable_statement_credit:
                    continue
                if self.slot_consolidation_mode == "error":
                    slot_consolidation_updates += self.consolidate_slots(row, statement_detector, candidate_entries)
                accepted_statement_credit = False
                for _ in range(self.max_statement_updates):
                    best_replacements: list[tuple[int, dict[str, str | None]]] = []
                    best_event_obj = current
                    for sent_idx, event in candidate_entries:
                        searched_statement_candidates += 1
                        objective = answer_objective(
                            scores_with_overrides(self.model, row, event_overrides={sent_idx: event}),
                            target,
                            self.temperature,
                        )
                        if objective > best_event_obj:
                            best_event_obj = objective
                            best_replacements = [(sent_idx, event)]
                    if self.enable_pair_statement_credit and len(candidate_entries) >= 2:
                        for left_idx in range(len(candidate_entries)):
                            sent_i, event_i = candidate_entries[left_idx]
                            for sent_j, event_j in candidate_entries[left_idx + 1 :]:
                                if sent_i == sent_j:
                                    continue
                                searched_pair_candidates += 1
                                objective = answer_objective(
                                    scores_with_overrides(
                                        self.model,
                                        row,
                                        event_overrides={sent_i: event_i, sent_j: event_j},
                                    ),
                                    target,
                                    self.temperature,
                                )
                                if objective > best_event_obj:
                                    best_event_obj = objective
                                    best_replacements = [(sent_i, event_i), (sent_j, event_j)]
                    if not best_replacements or best_event_obj - current <= self.min_gain:
                        break
                    gain = best_event_obj - current
                    for best_idx, best_event in best_replacements:
                        sentence = str(row["context"][best_idx]["text"])
                        update_norm += statement_detector.credit_update(sentence, best_event, self.credit_scale(gain))
                        statement_updates += 1
                        accepted_statement_credit = True
                    if len(best_replacements) > 1:
                        pair_credit_updates += 1
                    total_gain += gain
                    current = best_event_obj
                if self.slot_consolidation_mode == "credit" and accepted_statement_credit:
                    slot_consolidation_updates += self.consolidate_slots(row, statement_detector, candidate_entries)
            metrics.append(
                {
                    "epoch": epoch,
                    "rows": len(rows),
                    "statement_updates": statement_updates,
                    "query_updates": query_updates,
                    "total_gain": total_gain,
                    "mean_gain_per_row": total_gain / max(len(rows), 1),
                    "update_norm": update_norm,
                    "skipped_correct": skipped_correct,
                    "searched_statement_candidates": searched_statement_candidates,
                    "searched_pair_candidates": searched_pair_candidates,
                    "pair_credit_updates": pair_credit_updates,
                    "slot_consolidation_updates": slot_consolidation_updates,
                    "error_only": self.error_only,
                    "max_credit_scale": self.max_credit_scale,
                    "max_statement_updates": self.max_statement_updates,
                    "max_credit_sentences": self.max_credit_sentences,
                    "enable_pair_statement_credit": self.enable_pair_statement_credit,
                    "slot_consolidation_mode": self.slot_consolidation_mode,
                    "disable_query_credit": self.disable_query_credit,
                    "disable_statement_credit": self.disable_statement_credit,
                }
            )
        return metrics


def build_credit_model(
    answer_to_idx: dict[str, int],
    majority: MajorityBaseline,
    seed_rows: list[dict[str, Any]],
    credit_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    model_seed: int,
    train_credit: bool,
) -> tuple[AttributeBindingStateQALearner, list[dict[str, Any]]]:
    statement_detector = CreditAttributeStatementDetector(
        dim=args.statement_dim,
        lr=args.statement_lr,
        epochs=args.statement_epochs,
        score_scale=args.statement_score_scale,
        seed=model_seed + 317,
        confidence_threshold=args.statement_confidence_threshold,
    )
    query_detector = CreditAttributeQueryDetector(
        dim=args.query_dim,
        lr=args.query_lr,
        epochs=args.query_epochs,
        score_scale=args.query_score_scale,
        seed=model_seed + 331,
        confidence_threshold=args.query_confidence_threshold,
    )
    statement_detector.fit(seed_rows)
    query_detector.fit(seed_rows)
    model = AttributeBindingStateQALearner(
        answer_to_idx,
        majority,
        statement_detector=statement_detector,
        statement_mode="learned",
        query_detector=query_detector,
        query_mode="learned",
        dim=args.attr_dim,
        lr=args.attr_lr,
        score_scale=args.attr_score_scale,
        confidence_threshold=args.attr_confidence_threshold,
        seed=model_seed + 307,
    )
    trainer = AttributeAnswerCreditTrainer(
        model,
        answer_to_idx,
        args.temperature,
        args.min_credit_gain,
        args.max_credit_scale,
        args.max_statement_updates_per_row,
        args.max_credit_sentences_per_row,
        args.enable_pair_statement_credit,
        args.slot_consolidation_mode,
        args.disable_query_credit,
        args.disable_statement_credit,
        args.credit_error_only,
        model_seed + 503,
    )
    if not train_credit:
        return model, []
    return model, trainer.train(credit_rows, args.credit_epochs)


def build_structural_model(
    answer_to_idx: dict[str, int],
    majority: MajorityBaseline,
    train_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    model_seed: int,
) -> AttributeBindingStateQALearner:
    statement_detector = StressAttributeStatementDetector(
        dim=args.statement_dim,
        lr=args.statement_lr,
        epochs=args.statement_epochs,
        score_scale=args.statement_score_scale,
        seed=model_seed + 317,
        confidence_threshold=args.statement_confidence_threshold,
    )
    query_detector = StressAttributeQueryDetector(
        dim=args.query_dim,
        lr=args.query_lr,
        epochs=args.query_epochs,
        score_scale=args.query_score_scale,
        seed=model_seed + 331,
        confidence_threshold=args.query_confidence_threshold,
    )
    model = AttributeBindingStateQALearner(
        answer_to_idx,
        majority,
        statement_detector=statement_detector,
        statement_mode="learned",
        query_detector=query_detector,
        query_mode="learned",
        dim=args.attr_dim,
        lr=args.attr_lr,
        score_scale=args.attr_score_scale,
        confidence_threshold=args.attr_confidence_threshold,
        seed=model_seed + 307,
    )
    model.fit(train_rows)
    return model


def detector_summary_rows(
    method: str,
    model: AttributeBindingStateQALearner,
    splits: dict[str, list[dict[str, Any]]],
    config: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if model.statement_detector is None or model.query_detector is None:
        return rows
    for split, split_rows in splits.items():
        statement_metrics = model.statement_detector.statement_metrics(split_rows, args.statement_eval_limit)
        query_metrics = model.query_detector.query_metrics(split_rows, args.query_eval_limit)
        rows.append(
            {
                "method": method,
                "config": config,
                "split": split,
                "statement_event_accuracy": statement_metrics["event_accuracy"],
                "statement_entity_accuracy": statement_metrics["entity_accuracy"],
                "statement_value_accuracy": statement_metrics["value_accuracy"],
                "query_accuracy": query_metrics["query_accuracy"],
                "subject_accuracy": query_metrics["subject_accuracy"],
                "sentences": statement_metrics["sentences"],
                "questions": query_metrics["questions"],
                "statement_state_bytes": model.statement_detector.state_bytes(),
                "query_state_bytes": model.query_detector.state_bytes(),
            }
        )
    return rows


def run_one(config: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config_dir = args.data_dir / config
    train_base = read_jsonl(config_dir / "train.jsonl", args.train_limit or None)
    val_base = read_jsonl(config_dir / "validation.jsonl", args.eval_limit or None)
    test_base = read_jsonl(config_dir / "test.jsonl", args.eval_limit or None)
    seed_rows = paraphrase_rows(train_base, args.seed_strength)
    credit_rows = paraphrase_rows(train_base, args.credit_strength)
    validation_rows = paraphrase_rows(val_base, args.eval_strength)
    test_rows = paraphrase_rows(test_base, args.eval_strength)
    answer_vocab = build_answer_vocab(seed_rows, credit_rows, validation_rows, test_rows)
    answer_to_idx = {answer: idx for idx, answer in enumerate(answer_vocab)}
    majority = MajorityBaseline(answer_to_idx)
    majority.fit(seed_rows)
    splits = {"credit_train": credit_rows, "validation": validation_rows, "test": test_rows}

    pre_model, _ = build_credit_model(answer_to_idx, majority, seed_rows, credit_rows, args, args.seed, train_credit=False)
    credit_model, credit_metrics = build_credit_model(
        answer_to_idx, majority, seed_rows, credit_rows, args, args.seed, train_credit=True
    )
    structural_model = build_structural_model(answer_to_idx, majority, credit_rows, args, args.seed + 1009)

    summary_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    detector_rows: list[dict[str, Any]] = []
    for name, model, method_type in [
        ("seeded_pre_credit", pre_model, "pure_no_bp_seeded_frontend_no_credit"),
        ("answer_credit", credit_model, "pure_no_bp_answer_credit"),
        ("strong_structural_upper", structural_model, "local_structure_label_upper"),
    ]:
        summary, preds = evaluate_method(name, model, splits, answer_to_idx, args.temperature, False, method_type)
        for row in summary:
            row.update(
                {
                    "config": config,
                    "seed_strength": args.seed_strength,
                    "credit_strength": args.credit_strength,
                    "eval_strength": args.eval_strength,
                }
            )
        for row in preds:
            row.update({"config": config})
        summary_rows.extend(summary)
        pred_rows.extend(preds)
        detector_rows.extend(detector_summary_rows(name, model, splits, config, args))

    for row in credit_metrics:
        row.update(
            {
                "method": "answer_credit",
                "config": config,
                "seed_strength": args.seed_strength,
                "credit_strength": args.credit_strength,
                "eval_strength": args.eval_strength,
            }
        )
    config_rows = [
        {
            "config": config,
            "seed_strength": args.seed_strength,
            "credit_strength": args.credit_strength,
            "eval_strength": args.eval_strength,
            "seed_rows": len(seed_rows),
            "credit_rows": len(credit_rows),
            "validation_rows": len(validation_rows),
            "test_rows": len(test_rows),
        }
    ]
    return summary_rows, pred_rows, detector_rows, credit_metrics + config_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--configs", nargs="+", default=["en-qa15", "en-qa16"])
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "babi_attribute_delayed_credit")
    parser.add_argument("--seed-strength", choices=["none", "mild", "strong"], default="none")
    parser.add_argument("--credit-strength", choices=["none", "mild", "strong"], default="strong")
    parser.add_argument("--eval-strength", choices=["none", "mild", "strong"], default="strong")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--attr-dim", type=int, default=64)
    parser.add_argument("--attr-lr", type=float, default=1.0)
    parser.add_argument("--attr-score-scale", type=float, default=8.0)
    parser.add_argument("--attr-confidence-threshold", type=float, default=0.05)
    parser.add_argument("--statement-dim", type=int, default=64)
    parser.add_argument("--statement-lr", type=float, default=0.08)
    parser.add_argument("--statement-epochs", type=int, default=3)
    parser.add_argument("--statement-score-scale", type=float, default=6.0)
    parser.add_argument("--statement-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--statement-eval-limit", type=int, default=0)
    parser.add_argument("--query-dim", type=int, default=64)
    parser.add_argument("--query-lr", type=float, default=0.08)
    parser.add_argument("--query-epochs", type=int, default=3)
    parser.add_argument("--query-score-scale", type=float, default=6.0)
    parser.add_argument("--query-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--query-eval-limit", type=int, default=0)
    parser.add_argument("--credit-epochs", type=int, default=2)
    parser.add_argument("--min-credit-gain", type=float, default=1e-4)
    parser.add_argument("--max-credit-scale", type=float, default=1.0)
    parser.add_argument("--max-statement-updates-per-row", type=int, default=2)
    parser.add_argument("--max-credit-sentences-per-row", type=int, default=4)
    parser.add_argument("--enable-pair-statement-credit", action="store_true")
    parser.add_argument("--slot-consolidation-mode", choices=["off", "credit", "error", "all"], default="off")
    parser.add_argument("--disable-query-credit", action="store_true", default=True)
    parser.add_argument("--enable-query-credit", dest="disable_query_credit", action="store_false")
    parser.add_argument("--disable-statement-credit", action="store_true")
    parser.add_argument("--credit-error-only", action="store_true", default=True)
    parser.add_argument("--credit-all-rows", dest="credit_error_only", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, Any]] = []
    all_preds: list[dict[str, Any]] = []
    all_detectors: list[dict[str, Any]] = []
    all_credit: list[dict[str, Any]] = []
    for config in args.configs:
        summary, preds, detectors, credit = run_one(config, args)
        all_summary.extend(summary)
        all_preds.extend(preds)
        all_detectors.extend(detectors)
        all_credit.extend(credit)

    write_csv(args.out_dir / "summary.csv", all_summary)
    write_csv(args.out_dir / "predictions_sample.csv", all_preds)
    write_csv(args.out_dir / "detector_metrics.csv", all_detectors)
    write_csv(args.out_dir / "credit_metrics.csv", all_credit)
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
