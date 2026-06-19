#!/usr/bin/env python3
"""
Paraphrase stress test for no-BP bAbI QA15/QA16 attribute binding.

This mirrors the QA2/QA3 paraphrase stress script but targets the R138
attribute/category path.  It rewrites surface forms while preserving the source
answer labels and compares original-surface training against paraphrase-surface
training for local no-BP statement/query detectors.
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
    ATTRIBUTE_EVENT_TO_IDX,
    ATTRIBUTE_QUERY_TO_IDX,
    DEFAULT_DATA_DIR,
    AttributeBindingStateQALearner,
    HashedLookupBaseline,
    LearnedAttributeQueryDetector,
    LearnedAttributeStatementDetector,
    MajorityBaseline,
    PhaseDendriticQALearner,
    PhaseQAConfig,
    RawRetrievalBaseline,
    build_answer_vocab,
    evaluate_method,
    normalize_vector,
    parse_attribute_query,
    parse_attribute_statement,
    read_jsonl,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent

ATTRIBUTE_STATEMENT_REWRITES = [
    (re.compile(r"^([A-Za-z]+) are afraid of ([a-z]+)\.$"), r"\1 fear \2."),
    (re.compile(r"^([A-Z][a-z]+) is a ([a-z]+)\.$"), r"\1 is classified as a \2."),
    (re.compile(r"^([A-Z][a-z]+) is (gray|green|white|yellow)\.$"), r"\1 is colored \2."),
]
ATTRIBUTE_QUERY_REWRITES = [
    (re.compile(r"^What is ([A-Za-z]+) afraid of\?$"), r"What is \1 scared of?"),
    (re.compile(r"^What color is ([A-Za-z]+)\?$"), r"Which color is \1?"),
]


def rewrite_text(text: str, rules: list[tuple[re.Pattern[str], str]]) -> str:
    for pattern, repl in rules:
        if pattern.match(text):
            return pattern.sub(repl, text)
    return text


def paraphrase_sentence(sentence: str, strength: str) -> str:
    if strength == "none":
        return sentence
    out = rewrite_text(sentence, ATTRIBUTE_STATEMENT_REWRITES)
    if strength == "mild":
        return out
    return out.replace(".", " .") if out != sentence else out


def paraphrase_question(question: str, strength: str) -> str:
    if strength == "none":
        return question
    out = rewrite_text(question, ATTRIBUTE_QUERY_REWRITES)
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


class StressAttributeStatementDetector(LearnedAttributeStatementDetector):
    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples: list[tuple[str, dict[str, str | None]]] = []
        for row in rows:
            for item in row["context"]:
                sentence = str(item["text"])
                source_sentence = str(item.get("_source_text", sentence))
                examples.append((sentence, parse_attribute_statement(source_sentence)))
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

    def statement_metrics(self, rows: list[dict[str, Any]], limit: int = 0) -> dict[str, Any]:
        total = 0
        event_correct = 0
        entity_total = entity_correct = 0
        value_total = value_correct = 0
        confusion: dict[tuple[str, str], int] = {}
        for row in rows:
            for item in row["context"]:
                if limit and total >= limit:
                    break
                sentence = str(item["text"])
                target = parse_attribute_statement(str(item.get("_source_text", sentence)))
                pred = self.predict(sentence)
                target_event = str(target["event"])
                pred_event = str(pred["event"])
                event_correct += int(pred_event == target_event)
                confusion[(target_event, pred_event)] = confusion.get((target_event, pred_event), 0) + 1
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


class StressAttributeQueryDetector(LearnedAttributeQueryDetector):
    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples = [
            (str(row["question"]), parse_attribute_query(str(row.get("_source_question", row["question"]))))
            for row in rows
        ]
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

    def query_metrics(self, rows: list[dict[str, Any]], limit: int = 0) -> dict[str, Any]:
        total = 0
        query_correct = 0
        subject_total = subject_correct = 0
        confusion: dict[tuple[str, str], int] = {}
        for row in rows:
            if limit and total >= limit:
                break
            question = str(row["question"])
            target = parse_attribute_query(str(row.get("_source_question", question)))
            pred = self.predict(question)
            target_query = str(target["query"])
            pred_query = str(pred["query"])
            query_correct += int(pred_query == target_query)
            confusion[(target_query, pred_query)] = confusion.get((target_query, pred_query), 0) + 1
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


class ParaphraseAwareAttributeStatementDetector(StressAttributeStatementDetector):
    def normalize_sentence(self, sentence: str) -> str:
        text = sentence.replace(" .", ".")
        match = re.match(r"^([A-Za-z]+) fear ([a-z]+)\.$", text)
        if match:
            return f"{match.group(1)} are afraid of {match.group(2)}."
        match = re.match(r"^([A-Z][a-z]+) is classified as a ([a-z]+)\.$", text)
        if match:
            return f"{match.group(1)} is a {match.group(2)}."
        match = re.match(r"^([A-Z][a-z]+) is colored ([a-z]+)\.$", text)
        if match:
            return f"{match.group(1)} is {match.group(2)}."
        return text

    def sentence_feature(self, sentence: str) -> np.ndarray:
        return super().sentence_feature(self.normalize_sentence(sentence))

    def slot_feature(self, sentence: str, slot: Any) -> np.ndarray:
        return super().slot_feature(self.normalize_sentence(sentence), slot)

    def predict(self, sentence: str) -> dict[str, str | float | None]:
        return super().predict(self.normalize_sentence(sentence))


class ParaphraseAwareAttributeQueryDetector(StressAttributeQueryDetector):
    def normalize_question(self, question: str) -> str:
        text = question.replace(" ?", "?")
        match = re.match(r"^What is ([A-Za-z]+) scared of\?$", text)
        if match:
            return f"What is {match.group(1)} afraid of?"
        match = re.match(r"^Which color is ([A-Za-z]+)\?$", text)
        if match:
            return f"What color is {match.group(1)}?"
        return text

    def question_feature(self, question: str) -> np.ndarray:
        return super().question_feature(self.normalize_question(question))

    def subject_feature(self, question: str) -> np.ndarray:
        return super().subject_feature(self.normalize_question(question))

    def predict(self, question: str) -> dict[str, str | float | None]:
        return super().predict(self.normalize_question(question))


def build_attribute_model(
    answer_to_idx: dict[str, int],
    majority: MajorityBaseline,
    train_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    aware: bool,
) -> AttributeBindingStateQALearner:
    statement_cls = ParaphraseAwareAttributeStatementDetector if aware else StressAttributeStatementDetector
    query_cls = ParaphraseAwareAttributeQueryDetector if aware else StressAttributeQueryDetector
    statement_detector = statement_cls(
        dim=args.statement_dim,
        lr=args.statement_lr,
        epochs=args.statement_epochs,
        score_scale=args.statement_score_scale,
        seed=args.seed + 317,
        confidence_threshold=args.statement_confidence_threshold,
    )
    query_detector = query_cls(
        dim=args.query_dim,
        lr=args.query_lr,
        epochs=args.query_epochs,
        score_scale=args.query_score_scale,
        seed=args.seed + 331,
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
        seed=args.seed + 307,
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

    learned = build_attribute_model(answer_to_idx, majority, train_rows, args, aware=False)
    aware = build_attribute_model(answer_to_idx, majority, train_rows, args, aware=True)

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
        methods.append(("learned_attribute_binding", learned, False, "pure_no_bp_learned_frontend"))
    if run_all or "aware" in method_set:
        methods.append(("aware_attribute_binding", aware, False, "diagnostic_normalized_frontend"))

    summary_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
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
        ("learned_attribute_binding", learned),
        ("aware_attribute_binding", aware),
    ]:
        statement_detector = model.statement_detector
        query_detector = model.query_detector
        if statement_detector is None or query_detector is None:
            continue
        for split, rows in splits.items():
            statement_metrics = statement_detector.statement_metrics(rows, args.statement_eval_limit)
            query_metrics = query_detector.query_metrics(rows, args.query_eval_limit)
            detector_rows.append(
                {
                    "method": method_name,
                    "config": config,
                    "train_strength": train_strength,
                    "eval_strength": eval_strength,
                    "split": split,
                    "statement_event_accuracy": statement_metrics["event_accuracy"],
                    "statement_entity_accuracy": statement_metrics["entity_accuracy"],
                    "statement_value_accuracy": statement_metrics["value_accuracy"],
                    "query_accuracy": query_metrics["query_accuracy"],
                    "subject_accuracy": query_metrics["subject_accuracy"],
                    "sentences": statement_metrics["sentences"],
                    "questions": query_metrics["questions"],
                    "statement_state_bytes": statement_detector.state_bytes(),
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
    parser.add_argument("--configs", nargs="+", default=["en-qa15", "en-qa16"])
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "babi_attribute_paraphrase_stress")
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
            "learned_attribute_binding",
            "aware_attribute_binding",
            "hashed_lookup_diagnostic",
        }:
            print(
                f"  {row['config']} train={row['train_strength']} eval={row['eval_strength']} "
                f"{row['method']}: acc={float(row['accuracy']):.3f} loss={float(row['loss']):.3f}"
            )
    print(f"wrote {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
