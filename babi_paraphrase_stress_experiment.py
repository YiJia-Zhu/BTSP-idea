#!/usr/bin/env python3
"""
Paraphrase/noise stress test for learned no-BP bAbI role binding.

The script rewrites bAbI surface forms while preserving answers and world-state
semantics.  It compares learned event/query front-ends trained on original text
against the same local no-BP front-ends trained on paraphrased text.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from babi_no_bp_qa_experiment import (
    DEFAULT_DATA_DIR,
    HashedLookupBaseline,
    LearnedEventDetector,
    LearnedQueryDetector,
    MajorityBaseline,
    PhaseDendriticQALearner,
    PhaseQAConfig,
    RoleBindingStateQALearner,
    RawRetrievalBaseline,
    build_answer_vocab,
    evaluate_method,
    parse_event,
    parse_query,
    read_jsonl,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent


MOVE_REWRITES = [
    (re.compile(r"^([A-Z][a-z]+) moved to the ([a-z]+)\.$"), r"\1 relocated to the \2."),
    (re.compile(r"^([A-Z][a-z]+) went to the ([a-z]+)\.$"), r"\1 headed to the \2."),
    (re.compile(r"^([A-Z][a-z]+) went back to the ([a-z]+)\.$"), r"\1 returned to the \2."),
    (re.compile(r"^([A-Z][a-z]+) journeyed to the ([a-z]+)\.$"), r"\1 proceeded to the \2."),
    (re.compile(r"^([A-Z][a-z]+) travelled to the ([a-z]+)\.$"), r"\1 walked to the \2."),
]
PICKUP_REWRITES = [
    (re.compile(r"^([A-Z][a-z]+) got the ([a-z]+)( there)?\.$"), r"\1 acquired the \2."),
    (re.compile(r"^([A-Z][a-z]+) grabbed the ([a-z]+)( there)?\.$"), r"\1 collected the \2."),
    (re.compile(r"^([A-Z][a-z]+) took the ([a-z]+)( there)?\.$"), r"\1 picked up the \2."),
    (re.compile(r"^([A-Z][a-z]+) picked up the ([a-z]+)( there)?\.$"), r"\1 lifted the \2."),
]
DROP_REWRITES = [
    (re.compile(r"^([A-Z][a-z]+) dropped the ([a-z]+)( there)?\.$"), r"\1 released the \2."),
    (re.compile(r"^([A-Z][a-z]+) discarded the ([a-z]+)( there)?\.$"), r"\1 set down the \2."),
    (re.compile(r"^([A-Z][a-z]+) left the ([a-z]+)( there)?\.$"), r"\1 put aside the \2."),
    (re.compile(r"^([A-Z][a-z]+) put down the ([a-z]+)( there)?\.$"), r"\1 deposited the \2."),
]
QUESTION_REWRITES = [
    (re.compile(r"^Where is ([A-Z][a-z]+)\?$"), r"Which room is \1 in?"),
    (re.compile(r"^Where is the ([a-z]+)\?$"), r"Which room contains the \1?"),
    (re.compile(r"^Where was the ([a-z]+) before the ([a-z]+)\?$"), r"Before the \2, which room held the \1?"),
]


def rewrite_text(text: str, rules: list[tuple[re.Pattern[str], str]]) -> str:
    for pattern, repl in rules:
        if pattern.match(text):
            return pattern.sub(repl, text)
    return text


def paraphrase_sentence(sentence: str, strength: str) -> str:
    if strength == "none":
        return sentence
    out = rewrite_text(sentence, MOVE_REWRITES)
    out = rewrite_text(out, PICKUP_REWRITES)
    out = rewrite_text(out, DROP_REWRITES)
    if strength == "mild":
        return out
    if out != sentence:
        return out.replace(".", " .")
    return out


def paraphrase_question(question: str, strength: str) -> str:
    if strength == "none":
        return question
    out = rewrite_text(question, QUESTION_REWRITES)
    if strength == "strong":
        out = out.replace("?", " ?")
    return out


def paraphrase_rows(rows: list[dict[str, Any]], strength: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        new_row = deepcopy(row)
        new_row["_source_question"] = str(row["question"])
        new_row["question"] = paraphrase_question(str(row["question"]), strength)
        new_context = []
        for item in row["context"]:
            new_item = dict(item)
            new_item["_source_text"] = str(item["text"])
            new_item["text"] = paraphrase_sentence(str(item["text"]), strength)
            new_context.append(new_item)
        new_row["context"] = new_context
        out.append(new_row)
    return out


class StressEventDetector(LearnedEventDetector):
    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples: list[tuple[str, dict[str, str | None]]] = []
        for row in rows:
            for item in row["context"]:
                sentence = str(item["text"])
                source_sentence = str(item.get("_source_text", sentence))
                examples.append((sentence, parse_event(source_sentence)))
        if not examples:
            return
        for _ in range(self.epochs):
            order = self.rng.permutation(len(examples))
            for idx in order:
                sentence, parsed = examples[int(idx)]
                feature = self.sentence_feature(sentence)
                target = {"none": 0, "move": 1, "pickup": 2, "drop": 3}[str(parsed["event"])]
                scores = self.event_weights @ feature
                pred = int(np.argmax(scores))
                if pred != target:
                    self.event_weights[target] += self.lr * feature
                    self.event_weights[pred] -= self.lr * feature
                    self.event_weights[target] = self.event_weights[target] / (
                        np.linalg.norm(self.event_weights[target]) + 1e-8
                    )
                    self.event_weights[pred] = self.event_weights[pred] / (
                        np.linalg.norm(self.event_weights[pred]) + 1e-8
                    )
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

    def event_metrics(self, rows: list[dict[str, Any]], limit: int = 0) -> dict[str, Any]:
        total = 0
        event_correct = 0
        person_total = person_correct = 0
        object_total = object_correct = 0
        location_total = location_correct = 0
        confusion: dict[tuple[str, str], int] = {}
        for row in rows:
            for item in row["context"]:
                if limit and total >= limit:
                    break
                sentence = str(item["text"])
                target = parse_event(str(item.get("_source_text", sentence)))
                pred = self.predict(sentence)
                target_event = str(target["event"])
                pred_event = str(pred["event"])
                event_correct += int(pred_event == target_event)
                confusion[(target_event, pred_event)] = confusion.get((target_event, pred_event), 0) + 1
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


class StressQueryDetector(LearnedQueryDetector):
    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples = [
            (str(row["question"]), parse_query(str(row.get("_source_question", row["question"])))) for row in rows
        ]
        if not examples:
            return
        for _ in range(self.epochs):
            order = self.rng.permutation(len(examples))
            for idx in order:
                question, parsed = examples[int(idx)]
                feature = self.question_feature(question)
                target = {"where_is": 0, "where_before": 1}[str(parsed["query"])]
                scores = self.query_weights @ feature
                pred = int(np.argmax(scores))
                if pred != target:
                    self.query_weights[target] += self.lr * feature
                    self.query_weights[pred] -= self.lr * feature
                    self.query_weights[target] = self.query_weights[target] / (
                        np.linalg.norm(self.query_weights[target]) + 1e-8
                    )
                    self.query_weights[pred] = self.query_weights[pred] / (
                        np.linalg.norm(self.query_weights[pred]) + 1e-8
                    )
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

    def query_metrics(self, rows: list[dict[str, Any]], limit: int = 0) -> dict[str, Any]:
        total = 0
        query_correct = 0
        subject_total = subject_correct = 0
        destination_total = destination_correct = 0
        confusion: dict[tuple[str, str], int] = {}
        for row in rows:
            if limit and total >= limit:
                break
            question = str(row["question"])
            target = parse_query(str(row.get("_source_question", question)))
            pred = self.predict(question)
            target_query = str(target["query"])
            pred_query = str(pred["query"])
            query_correct += int(pred_query == target_query)
            confusion[(target_query, pred_query)] = confusion.get((target_query, pred_query), 0) + 1
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


class ParaphraseAwareEventDetector(StressEventDetector):
    def normalize_sentence(self, sentence: str) -> str:
        replacements = {
            "relocated": "moved",
            "headed": "went",
            "returned": "went",
            "proceeded": "journeyed",
            "walked": "travelled",
            "acquired": "got",
            "collected": "grabbed",
            "lifted": "picked",
            "released": "dropped",
            "deposited": "dropped",
            "aside": "",
            "set down": "dropped",
        }
        out = sentence.lower()
        for src, dst in replacements.items():
            out = out.replace(src, dst)
        out = out.replace("put the", "put down the")
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def sentence_feature(self, sentence: str) -> np.ndarray:
        return super().sentence_feature(self.normalize_sentence(sentence))

    def slot_feature(self, sentence: str, slot: Any) -> np.ndarray:
        return super().slot_feature(self.normalize_sentence(sentence), slot)

    def predict(self, sentence: str) -> dict[str, str | float | None]:
        return super().predict(self.normalize_sentence(sentence))


class ParaphraseAwareQueryDetector(StressQueryDetector):
    def normalize_question(self, question: str) -> str:
        lower = question.lower().replace(" ?", "?")
        match = re.match(r"which room is ([a-z]+) in\?", lower)
        if match:
            return f"where is {match.group(1)}?"
        match = re.match(r"which room contains the ([a-z]+)\?", lower)
        if match:
            return f"where is the {match.group(1)}?"
        match = re.match(r"before the ([a-z]+), which room held the ([a-z]+)\?", lower)
        if match:
            return f"where was the {match.group(2)} before the {match.group(1)}?"
        return lower

    def question_feature(self, question: str) -> np.ndarray:
        return super().question_feature(self.normalize_question(question))

    def slot_feature(self, question: str, slot: Any) -> np.ndarray:
        return super().slot_feature(self.normalize_question(question), slot)

    def predict(self, question: str) -> dict[str, str | float | None]:
        return super().predict(self.normalize_question(question))


def build_learned_role_model(
    answer_to_idx: dict[str, int],
    majority: MajorityBaseline,
    train_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    aware: bool,
) -> RoleBindingStateQALearner:
    event_cls = ParaphraseAwareEventDetector if aware else StressEventDetector
    query_cls = ParaphraseAwareQueryDetector if aware else StressQueryDetector
    event_detector = event_cls(
        dim=args.event_dim,
        lr=args.event_lr,
        epochs=args.event_epochs,
        score_scale=args.event_score_scale,
        seed=args.seed + 101,
        confidence_threshold=args.event_confidence_threshold,
    )
    query_detector = query_cls(
        dim=args.query_dim,
        lr=args.query_lr,
        epochs=args.query_epochs,
        score_scale=args.query_score_scale,
        seed=args.seed + 211,
        confidence_threshold=args.query_confidence_threshold,
    )
    model = RoleBindingStateQALearner(
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
        seed=args.seed,
    )
    model.fit(train_rows)
    return model


def run_one(
    config: str,
    train_strength: str,
    eval_strength: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config_dir = args.data_dir / config
    train_base = read_jsonl(config_dir / "train.jsonl", args.train_limit or None)
    val_base = read_jsonl(config_dir / "validation.jsonl", args.eval_limit or None)
    test_base = read_jsonl(config_dir / "test.jsonl", args.eval_limit or None)
    train_rows = paraphrase_rows(train_base, train_strength)
    validation_rows = paraphrase_rows(val_base, eval_strength)
    test_rows = paraphrase_rows(test_base, eval_strength)
    answer_vocab = build_answer_vocab(train_rows, validation_rows, test_rows)
    answer_to_idx = {answer: idx for idx, answer in enumerate(answer_vocab)}
    splits = {"train": train_rows, "validation": validation_rows, "test": test_rows}

    majority = MajorityBaseline(answer_to_idx)
    majority.fit(train_rows)

    lookup = HashedLookupBaseline(answer_to_idx, args.hash_bits, args.lookup_ngrams, args.seed + 17)
    lookup.fit(train_rows)

    method_set = set(args.methods)
    run_all = "all" in method_set

    raw = None
    if run_all or "raw" in method_set:
        raw = RawRetrievalBaseline(answer_to_idx)
        raw.fit(train_rows)

    phase = None
    if run_all or "phase" in method_set:
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

    learned = build_learned_role_model(answer_to_idx, majority, train_rows, args, aware=False)
    aware = build_learned_role_model(answer_to_idx, majority, train_rows, args, aware=True)

    summary_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    methods = []
    if run_all or "majority" in method_set:
        methods.append(("majority_no_memory", majority, False, "baseline"))
    if raw is not None:
        methods.append(("raw_lexical_retrieval", raw, True, "diagnostic_raw_retrieval"))
    if run_all or "hashed" in method_set:
        methods.append(("hashed_lookup_diagnostic", lookup, False, "diagnostic_statistical_lookup"))
    if phase is not None:
        methods.append(("phase_dendritic_no_bp", phase, False, "pure_no_bp_neural"))
    if run_all or "learned" in method_set:
        methods.append(("learned_event_query_role_binding", learned, False, "pure_no_bp_learned_frontend"))
    if run_all or "aware" in method_set:
        methods.append(("aware_event_query_role_binding", aware, False, "diagnostic_normalized_frontend"))
    for name, model, stores_raw, method_type in methods:
        summary, preds = evaluate_method(name, model, splits, answer_to_idx, args.phase_temperature, stores_raw, method_type)
        for row in summary:
            row.update({"config": config, "train_strength": train_strength, "eval_strength": eval_strength})
        for row in preds:
            row.update({"config": config, "train_strength": train_strength, "eval_strength": eval_strength})
        summary_rows.extend(summary)
        pred_rows.extend(preds)

    detector_rows: list[dict[str, Any]] = []
    for method_name, model in [
        ("learned_event_query_role_binding", learned),
        ("aware_event_query_role_binding", aware),
    ]:
        event_detector = model.event_detector
        query_detector = model.query_detector
        if event_detector is None or query_detector is None:
            continue
        for split, rows in splits.items():
            event_metrics = event_detector.event_metrics(rows, args.event_eval_limit)
            query_metrics = query_detector.query_metrics(rows, args.query_eval_limit)
            detector_rows.append(
                {
                    "method": method_name,
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
                    "event_state_bytes": event_detector.state_bytes(),
                    "query_state_bytes": query_detector.state_bytes(),
                }
            )

    config_rows = [
        {
            "config": config,
            "train_strength": train_strength,
            "eval_strength": eval_strength,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "test_rows": len(test_rows),
        }
    ]
    return summary_rows, pred_rows, detector_rows, config_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--configs", nargs="+", default=["en-qa2", "en-qa3"])
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "babi_paraphrase_stress")
    parser.add_argument("--train-strengths", nargs="+", default=["none", "strong"])
    parser.add_argument("--eval-strength", choices=["none", "mild", "strong"], default="strong")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["all", "majority", "raw", "hashed", "phase", "learned", "aware"],
    )
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--lookup-ngrams", type=int, default=2)
    parser.add_argument("--phase-dim", type=int, default=32)
    parser.add_argument("--phase-lr", type=float, default=0.08)
    parser.add_argument("--phase-wrong-lr", type=float, default=0.03)
    parser.add_argument("--phase-epochs", type=int, default=2)
    parser.add_argument("--phase-score-scale", type=float, default=6.0)
    parser.add_argument("--phase-temperature", type=float, default=1.0)
    parser.add_argument("--branch-agreement", type=float, default=0.05)
    parser.add_argument("--role-dim", type=int, default=64)
    parser.add_argument("--role-lr", type=float, default=1.0)
    parser.add_argument("--role-score-scale", type=float, default=8.0)
    parser.add_argument("--role-carry-threshold", type=float, default=0.35)
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
    parser.add_argument("--query-eval-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, Any]] = []
    all_preds: list[dict[str, Any]] = []
    all_detectors: list[dict[str, Any]] = []
    all_configs: list[dict[str, Any]] = []

    for config in args.configs:
        for train_strength in args.train_strengths:
            summary, preds, detectors, config_rows = run_one(config, train_strength, args.eval_strength, args)
            all_summary.extend(summary)
            all_preds.extend(preds)
            all_detectors.extend(detectors)
            all_configs.extend(config_rows)

    write_csv(args.out_dir / "summary.csv", all_summary)
    write_csv(args.out_dir / "predictions_sample.csv", all_preds)
    write_csv(args.out_dir / "detector_metrics.csv", all_detectors)
    write_csv(args.out_dir / "run_configs.csv", all_configs)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str, sort_keys=True)

    print("Summary:")
    for row in all_summary:
        if row["split"] == "test" and row["method"] in {
            "learned_event_query_role_binding",
            "aware_event_query_role_binding",
            "hashed_lookup_diagnostic",
        }:
            print(
                f"  {row['config']} train={row['train_strength']} eval={row['eval_strength']} "
                f"{row['method']}: acc={float(row['accuracy']):.3f} loss={float(row['loss']):.3f}"
            )
    print(f"wrote {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
