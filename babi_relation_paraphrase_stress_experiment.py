#!/usr/bin/env python3
"""
Paraphrase stress test for no-BP bAbI QA18/QA19 relation-state learning.

R144 learned the canonical relation front-end for size/path reasoning.  This
script rewrites the surface form while preserving the answer labels and tests
whether local no-BP detectors transfer or adapt to the new surface.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from babi_no_bp_qa_experiment import (
    DEFAULT_DATA_DIR,
    MajorityBaseline,
    build_answer_vocab,
    evaluate_method,
    read_jsonl,
    write_csv,
)
from babi_relation_state_experiment import (
    LearnedPathQueryDetector,
    LearnedPathStatementDetector,
    LearnedSizeQueryDetector,
    LearnedSizeStatementDetector,
    LocalDetectorConfig,
    PathRelationStateQALearner,
    RelationStateConfig,
    SizeRelationStateQALearner,
    parse_path_query_detail,
    parse_path_statement_detail,
    parse_size_query_detail,
    parse_size_statement_detail,
)


SCRIPT_DIR = Path(__file__).resolve().parent

SIZE_STATEMENT_REWRITES = [
    (
        re.compile(r"^The (.+?) fits inside the (.+?)\.$"),
        r"The \1 can fit within the \2.",
    ),
    (
        re.compile(r"^The (.+?) is bigger than the (.+?)\.$"),
        r"The \1 is larger than the \2.",
    ),
]
SIZE_QUERY_REWRITES = [
    (
        re.compile(r"^Does the (.+?) fit in the (.+?)\?$"),
        r"Can the \1 fit inside the \2?",
    ),
    (
        re.compile(r"^Is the (.+?) bigger than the (.+?)\?$"),
        r"Is the \1 larger than the \2?",
    ),
]
PATH_STATEMENT_REWRITE = re.compile(r"^The (.+?) is (north|south|east|west) of the (.+?)\.$")
PATH_QUERY_REWRITE = re.compile(r"^How do you go from the (.+?) to the (.+?)\?$")
PATH_DIRECTION_ALIAS = {
    "north": "above",
    "south": "below",
    "east": "to the right",
    "west": "to the left",
}


def rewrite_text(text: str, rules: list[tuple[re.Pattern[str], str]]) -> str:
    for pattern, repl in rules:
        if pattern.match(text):
            return pattern.sub(repl, text)
    return text


def paraphrase_sentence(sentence: str, config: str, strength: str) -> str:
    if strength == "none":
        return sentence
    if config == "en-qa18":
        return rewrite_text(sentence, SIZE_STATEMENT_REWRITES)
    if config == "en-qa19":
        match = PATH_STATEMENT_REWRITE.match(sentence)
        if match:
            return f"The {match.group(1)} is {PATH_DIRECTION_ALIAS[match.group(2)]} of the {match.group(3)}."
    return sentence


def paraphrase_question(question: str, config: str, strength: str) -> str:
    if strength == "none":
        return question
    if config == "en-qa18":
        return rewrite_text(question, SIZE_QUERY_REWRITES)
    if config == "en-qa19":
        match = PATH_QUERY_REWRITE.match(question)
        if match:
            return f"What path takes you from the {match.group(1)} to the {match.group(2)}?"
    return question


def paraphrase_rows(rows: list[dict[str, Any]], config: str, strength: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = deepcopy(row)
        new_row["_source_question"] = str(row["question"])
        new_row["question"] = paraphrase_question(str(row["question"]), config, strength)
        new_context = []
        for item in row["context"]:
            new_item = dict(item)
            new_item["_source_text"] = str(item["text"])
            new_item["text"] = paraphrase_sentence(str(item["text"]), config, strength)
            new_context.append(new_item)
        new_row["context"] = new_context
        out.append(new_row)
    return out


class StressSizeStatementDetector(LearnedSizeStatementDetector):
    def slot_feature(self, sentence: str, slot: str):  # type: ignore[no-untyped-def]
        text = sentence.replace(" can fit within ", " fits inside ")
        text = text.replace(" is larger than ", " is bigger than ")
        return super().slot_feature(text, slot)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples: list[tuple[str, dict[str, str]]] = []
        for row in rows:
            for item in row["context"]:
                sentence = str(item["text"])
                source = str(item.get("_source_text", sentence))
                detail = parse_size_statement_detail(source)
                if detail is not None:
                    examples.append((sentence, detail))
        self._fit_examples(examples)

    def _fit_examples(self, examples: list[tuple[str, dict[str, str]]]) -> None:
        if not examples:
            return
        for _ in range(self.cfg.epochs):
            for idx in self.rng.permutation(len(examples)):
                sentence, detail = examples[int(idx)]
                feature = self.text_feature(sentence, "event")
                target = {"fit_inside": 0, "bigger_than": 1}[detail["event"]]
                scores = self.event_weights @ feature
                pred = int(scores.argmax())
                if pred != target:
                    self.event_weights[target] += self.cfg.lr * feature
                    self.event_weights[pred] -= self.cfg.lr * feature
                self.update_prototype(self.left_prototypes, self.left_counts, detail["left"], self.slot_feature(sentence, "left"))
                self.update_prototype(self.right_prototypes, self.right_counts, detail["right"], self.slot_feature(sentence, "right"))

    def metrics(self, rows: list[dict[str, Any]]) -> dict[str, float | int]:
        total = event_correct = left_correct = right_correct = 0
        for row in rows:
            for item in row["context"]:
                sentence = str(item["text"])
                target = parse_size_statement_detail(str(item.get("_source_text", sentence)))
                if target is None:
                    continue
                pred = self.predict(sentence)
                total += 1
                event_correct += int(pred["event"] == target["event"])
                left_correct += int(pred["left"] == target["left"])
                right_correct += int(pred["right"] == target["right"])
        return {
            "examples": total,
            "event_accuracy": event_correct / max(total, 1),
            "left_accuracy": left_correct / max(total, 1),
            "right_accuracy": right_correct / max(total, 1),
            "state_bytes": self.state_bytes(),
        }


class StressSizeQueryDetector(LearnedSizeQueryDetector):
    def slot_feature(self, question: str, slot: str):  # type: ignore[no-untyped-def]
        text = question.replace("Can the ", "Does the ")
        text = text.replace(" fit inside ", " fit in ")
        text = text.replace(" larger than ", " bigger than ")
        return super().slot_feature(text, slot)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples = []
        for row in rows:
            question = str(row["question"])
            source = str(row.get("_source_question", question))
            detail = parse_size_query_detail(source)
            if detail is not None:
                examples.append((question, detail))
        if not examples:
            return
        for _ in range(self.cfg.epochs):
            for idx in self.rng.permutation(len(examples)):
                question, detail = examples[int(idx)]
                feature = self.text_feature(question, "query")
                target = {"fit_in": 0, "bigger_than": 1}[detail["query"]]
                scores = self.query_weights @ feature
                pred = int(scores.argmax())
                if pred != target:
                    self.query_weights[target] += self.cfg.lr * feature
                    self.query_weights[pred] -= self.cfg.lr * feature
                self.update_prototype(self.left_prototypes, self.left_counts, detail["left"], self.slot_feature(question, "left"))
                self.update_prototype(self.right_prototypes, self.right_counts, detail["right"], self.slot_feature(question, "right"))

    def metrics(self, rows: list[dict[str, Any]]) -> dict[str, float | int]:
        total = query_correct = left_correct = right_correct = 0
        for row in rows:
            question = str(row["question"])
            target = parse_size_query_detail(str(row.get("_source_question", question)))
            if target is None:
                continue
            pred = self.predict(question)
            total += 1
            query_correct += int(pred["query"] == target["query"])
            left_correct += int(pred["left"] == target["left"])
            right_correct += int(pred["right"] == target["right"])
        return {
            "examples": total,
            "query_accuracy": query_correct / max(total, 1),
            "left_accuracy": left_correct / max(total, 1),
            "right_accuracy": right_correct / max(total, 1),
            "state_bytes": self.state_bytes(),
        }


class StressPathStatementDetector(LearnedPathStatementDetector):
    def slot_feature(self, sentence: str, slot: str):  # type: ignore[no-untyped-def]
        text = sentence.replace(" lies ", " is ")
        return super().slot_feature(text, slot)

    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples: list[tuple[str, dict[str, str]]] = []
        for row in rows:
            for item in row["context"]:
                sentence = str(item["text"])
                source = str(item.get("_source_text", sentence))
                detail = parse_path_statement_detail(source)
                if detail is not None:
                    examples.append((sentence, detail))
        if not examples:
            return
        for _ in range(self.cfg.epochs):
            for idx in self.rng.permutation(len(examples)):
                sentence, detail = examples[int(idx)]
                feature = self.text_feature(sentence, "direction")
                target = {"north": 0, "south": 1, "east": 2, "west": 3}[detail["direction"]]
                scores = self.direction_weights @ feature
                pred = int(scores.argmax())
                if pred != target:
                    self.direction_weights[target] += self.cfg.lr * feature
                    self.direction_weights[pred] -= self.cfg.lr * feature
                self.update_prototype(self.source_prototypes, self.source_counts, detail["source"], self.slot_feature(sentence, "source"))
                self.update_prototype(self.target_prototypes, self.target_counts, detail["target"], self.slot_feature(sentence, "target"))

    def metrics(self, rows: list[dict[str, Any]]) -> dict[str, float | int]:
        total = direction_correct = source_correct = target_correct = 0
        for row in rows:
            for item in row["context"]:
                sentence = str(item["text"])
                target = parse_path_statement_detail(str(item.get("_source_text", sentence)))
                if target is None:
                    continue
                pred = self.predict(sentence)
                total += 1
                direction_correct += int(pred["direction"] == target["direction"])
                source_correct += int(pred["source"] == target["source"])
                target_correct += int(pred["target"] == target["target"])
        return {
            "examples": total,
            "direction_accuracy": direction_correct / max(total, 1),
            "source_accuracy": source_correct / max(total, 1),
            "target_accuracy": target_correct / max(total, 1),
            "state_bytes": self.state_bytes(),
        }


class StressPathQueryDetector(LearnedPathQueryDetector):
    def fit(self, rows: list[dict[str, Any]]) -> None:
        examples = []
        for row in rows:
            question = str(row["question"])
            source = str(row.get("_source_question", question))
            detail = parse_path_query_detail(source)
            if detail is not None:
                examples.append((question, detail))
        if not examples:
            return
        for _ in range(self.cfg.epochs):
            for idx in self.rng.permutation(len(examples)):
                question, detail = examples[int(idx)]
                self.update_prototype(self.source_prototypes, self.source_counts, detail["source"], self.slot_feature(question, "source"))
                self.update_prototype(self.target_prototypes, self.target_counts, detail["target"], self.slot_feature(question, "target"))

    def metrics(self, rows: list[dict[str, Any]]) -> dict[str, float | int]:
        total = source_correct = target_correct = 0
        for row in rows:
            question = str(row["question"])
            target = parse_path_query_detail(str(row.get("_source_question", question)))
            if target is None:
                continue
            pred = self.predict(question)
            total += 1
            source_correct += int(pred["source"] == target["source"])
            target_correct += int(pred["target"] == target["target"])
        return {
            "examples": total,
            "source_accuracy": source_correct / max(total, 1),
            "target_accuracy": target_correct / max(total, 1),
            "state_bytes": self.state_bytes(),
        }


def build_model(
    config: str,
    answer_to_idx: dict[str, int],
    majority: MajorityBaseline,
    relation_cfg: RelationStateConfig,
    detector_cfg: LocalDetectorConfig,
) -> tuple[Any, list[tuple[str, Any]]]:
    if config == "en-qa18":
        statement = StressSizeStatementDetector(detector_cfg)
        query = StressSizeQueryDetector(detector_cfg)
        model = SizeRelationStateQALearner(
            answer_to_idx,
            majority,
            relation_cfg,
            statement_detector=statement,
            query_detector=query,
            statement_mode="learned",
            query_mode="learned",
        )
        return model, [("size_statement", statement), ("size_query", query)]
    statement = StressPathStatementDetector(detector_cfg)
    query = StressPathQueryDetector(detector_cfg)
    model = PathRelationStateQALearner(
        answer_to_idx,
        majority,
        relation_cfg,
        statement_detector=statement,
        query_detector=query,
        statement_mode="learned",
        query_mode="learned",
    )
    return model, [("path_statement", statement), ("path_query", query)]


def run_condition(
    config: str,
    condition: str,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    relation_cfg: RelationStateConfig,
    detector_cfg: LocalDetectorConfig,
    temperature: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    answer_vocab = build_answer_vocab(train_rows, validation_rows, test_rows)
    answer_to_idx = {answer: idx for idx, answer in enumerate(answer_vocab)}
    majority = MajorityBaseline(answer_to_idx)
    majority.fit(train_rows)
    model, detectors = build_model(config, answer_to_idx, majority, relation_cfg, detector_cfg)
    model.fit(train_rows)
    splits = {"train": train_rows, "validation": validation_rows, "test": test_rows}
    summary, _ = evaluate_method(
        condition,
        model,
        splits,
        answer_to_idx,
        temperature,
        False,
        "pure_no_bp_relation_state_learned_frontend_stress",
    )
    detector_rows = []
    for detector_name, detector in detectors:
        for split, rows in splits.items():
            detector_rows.append(
                {
                    "config": config,
                    "condition": condition,
                    "detector": detector_name,
                    "split": split,
                    **detector.metrics(rows),
                }
            )
    for row in summary:
        row["config"] = config
    return summary, detector_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--configs", nargs="+", default=["en-qa18", "en-qa19"])
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "babi_relation_paraphrase_stress")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--relation-dim", type=int, default=128)
    parser.add_argument("--relation-lr", type=float, default=1.0)
    parser.add_argument("--relation-score-scale", type=float, default=8.0)
    parser.add_argument("--relation-max-hops", type=int, default=4)
    parser.add_argument("--relation-hop-decay", type=float, default=0.95)
    parser.add_argument("--detector-dim", type=int, default=64)
    parser.add_argument("--detector-lr", type=float, default=0.08)
    parser.add_argument("--detector-epochs", type=int, default=3)
    parser.add_argument("--detector-score-scale", type=float, default=6.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def read_split(data_dir: Path, config: str, split: str, limit: int) -> list[dict[str, Any]]:
    return read_jsonl(data_dir / config / f"{split}.jsonl", limit or None)


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    relation_cfg = RelationStateConfig(
        dim=args.relation_dim,
        lr=args.relation_lr,
        score_scale=args.relation_score_scale,
        max_hops=args.relation_max_hops,
        hop_decay=args.relation_hop_decay,
        seed=args.seed + 503,
    )
    detector_cfg = LocalDetectorConfig(
        dim=args.detector_dim,
        lr=args.detector_lr,
        epochs=args.detector_epochs,
        score_scale=args.detector_score_scale,
        seed=args.seed + 701,
    )
    all_summary: list[dict[str, Any]] = []
    all_detector_rows: list[dict[str, Any]] = []
    for config in args.configs:
        train_rows = read_split(args.data_dir, config, "train", args.train_limit)
        validation_rows = read_split(args.data_dir, config, "validation", args.eval_limit)
        test_rows = read_split(args.data_dir, config, "test", args.eval_limit)
        strong_train = paraphrase_rows(train_rows, config, "strong")
        strong_validation = paraphrase_rows(validation_rows, config, "strong")
        strong_test = paraphrase_rows(test_rows, config, "strong")
        for condition, cond_train in [
            ("learned_original_train_strong_test", train_rows),
            ("learned_strong_train_strong_test", strong_train),
        ]:
            summary, detector_rows = run_condition(
                config,
                condition,
                cond_train,
                strong_validation,
                strong_test,
                relation_cfg,
                detector_cfg,
                args.temperature,
            )
            all_summary.extend(summary)
            all_detector_rows.extend(detector_rows)
    write_csv(args.out_dir / "summary.csv", all_summary)
    write_csv(args.out_dir / "detector_metrics.csv", all_detector_rows)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args)}, f, indent=2, default=str, sort_keys=True)

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
