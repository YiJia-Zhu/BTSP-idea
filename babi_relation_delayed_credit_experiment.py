#!/usr/bin/env python3
"""
Delayed final-answer credit for bAbI QA18/QA19 relation paraphrases.

R145 showed that strong paraphrases mainly break relation type/direction
channels, while slots transfer.  This script starts from canonical learned
front-ends, streams strong paraphrased train rows, and updates only local
relation/direction detectors using answer-error-gated third factors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from babi_no_bp_qa_experiment import (
    DEFAULT_DATA_DIR,
    MajorityBaseline,
    build_answer_vocab,
    evaluate_method,
    normalize_vector,
    read_jsonl,
    softmax,
    write_csv,
)
from babi_relation_paraphrase_stress_experiment import (
    StressPathQueryDetector,
    StressPathStatementDetector,
    StressSizeQueryDetector,
    StressSizeStatementDetector,
    paraphrase_rows,
)
from babi_relation_state_experiment import (
    LocalDetectorConfig,
    PathRelationStateQALearner,
    RelationStateConfig,
    SizeRelationStateQALearner,
    associate,
    recurrent_path_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent

SIZE_FIT_STRONG = re.compile(r"^The (.+?) can fit within the (.+?)\.$")
SIZE_BIGGER_STRONG = re.compile(r"^The (.+?) is larger than the (.+?)\.$")
SIZE_FIT_QUERY_STRONG = re.compile(r"^Can the (.+?) fit inside the (.+?)\?$")
SIZE_BIGGER_QUERY_STRONG = re.compile(r"^Is the (.+?) larger than the (.+?)\?$")
PATH_ALIAS_STATEMENT = re.compile(r"^The (.+?) is (above|below|to the right|to the left) of the (.+?)\.$")
PATH_ALIAS_TO_DIRECTION = {
    "above": "north",
    "below": "south",
    "to the right": "east",
    "to the left": "west",
}
DIRECTION_TO_IDX = {"north": 0, "south": 1, "east": 2, "west": 3}


def answer_objective(scores: np.ndarray, target_idx: int, temperature: float) -> float:
    probs = softmax(scores.astype(np.float32, copy=False), temperature)
    return math.log(float(probs[target_idx]) + 1e-9)


def size_statement_surface_detail(sentence: str) -> dict[str, str] | None:
    fit = SIZE_FIT_STRONG.match(sentence)
    if fit:
        left, right = fit.group(1).lower(), fit.group(2).lower()
        return {"event": "fit_inside", "left": left, "right": right, "smaller": left, "larger": right}
    bigger = SIZE_BIGGER_STRONG.match(sentence)
    if bigger:
        left, right = bigger.group(1).lower(), bigger.group(2).lower()
        return {"event": "bigger_than", "left": left, "right": right, "smaller": right, "larger": left}
    return None


def size_query_surface_detail(question: str) -> dict[str, str] | None:
    fit = SIZE_FIT_QUERY_STRONG.match(question)
    if fit:
        left, right = fit.group(1).lower(), fit.group(2).lower()
        return {"query": "fit_in", "left": left, "right": right, "smaller": left, "larger": right}
    bigger = SIZE_BIGGER_QUERY_STRONG.match(question)
    if bigger:
        left, right = bigger.group(1).lower(), bigger.group(2).lower()
        return {"query": "bigger_than", "left": left, "right": right, "smaller": right, "larger": left}
    return None


def path_statement_surface_detail(sentence: str) -> dict[str, str] | None:
    match = PATH_ALIAS_STATEMENT.match(sentence)
    if not match:
        return None
    target = match.group(1).lower()
    direction = PATH_ALIAS_TO_DIRECTION[match.group(2)]
    source = match.group(3).lower()
    return {"source": source, "direction": direction, "target": target}


class CreditSizeStatementDetector(StressSizeStatementDetector):
    def credit_update(self, sentence: str, detail: dict[str, str], scale: float) -> float:
        feature = self.text_feature(sentence, "event")
        target = {"fit_inside": 0, "bigger_than": 1}[detail["event"]]
        pred = int(np.argmax(self.event_weights @ feature))
        eta = self.cfg.lr * max(float(scale), 0.05)
        if pred != target:
            self.event_weights[target] += eta * feature
            self.event_weights[pred] -= 0.5 * eta * feature
            self.event_weights[target] = normalize_vector(self.event_weights[target])
            self.event_weights[pred] = normalize_vector(self.event_weights[pred])
        self.update_prototype(self.left_prototypes, self.left_counts, detail["left"], self.slot_feature(sentence, "left"))
        self.update_prototype(self.right_prototypes, self.right_counts, detail["right"], self.slot_feature(sentence, "right"))
        return float(eta)


class CreditSizeQueryDetector(StressSizeQueryDetector):
    def credit_update(self, question: str, detail: dict[str, str], scale: float) -> float:
        feature = self.text_feature(question, "query")
        target = {"fit_in": 0, "bigger_than": 1}[detail["query"]]
        pred = int(np.argmax(self.query_weights @ feature))
        eta = self.cfg.lr * max(float(scale), 0.05)
        if pred != target:
            self.query_weights[target] += eta * feature
            self.query_weights[pred] -= 0.5 * eta * feature
            self.query_weights[target] = normalize_vector(self.query_weights[target])
            self.query_weights[pred] = normalize_vector(self.query_weights[pred])
        self.update_prototype(self.left_prototypes, self.left_counts, detail["left"], self.slot_feature(question, "left"))
        self.update_prototype(self.right_prototypes, self.right_counts, detail["right"], self.slot_feature(question, "right"))
        return float(eta)


class CreditPathStatementDetector(StressPathStatementDetector):
    def credit_update(self, sentence: str, detail: dict[str, str], scale: float) -> float:
        feature = self.text_feature(sentence, "direction")
        target = DIRECTION_TO_IDX[detail["direction"]]
        pred = int(np.argmax(self.direction_weights @ feature))
        eta = self.cfg.lr * max(float(scale), 0.05)
        if pred != target:
            self.direction_weights[target] += eta * feature
            self.direction_weights[pred] -= 0.5 * eta * feature
            self.direction_weights[target] = normalize_vector(self.direction_weights[target])
            self.direction_weights[pred] = normalize_vector(self.direction_weights[pred])
        self.update_prototype(self.source_prototypes, self.source_counts, detail["source"], self.slot_feature(sentence, "source"))
        self.update_prototype(self.target_prototypes, self.target_counts, detail["target"], self.slot_feature(sentence, "target"))
        return float(eta)


def build_size_model(
    answer_to_idx: dict[str, int],
    majority: MajorityBaseline,
    relation_cfg: RelationStateConfig,
    detector_cfg: LocalDetectorConfig,
) -> tuple[SizeRelationStateQALearner, CreditSizeStatementDetector, CreditSizeQueryDetector]:
    statement = CreditSizeStatementDetector(detector_cfg)
    query = CreditSizeQueryDetector(detector_cfg)
    model = SizeRelationStateQALearner(
        answer_to_idx,
        majority,
        relation_cfg,
        statement_detector=statement,
        query_detector=query,
        statement_mode="learned",
        query_mode="learned",
    )
    return model, statement, query


def build_path_model(
    answer_to_idx: dict[str, int],
    majority: MajorityBaseline,
    relation_cfg: RelationStateConfig,
    detector_cfg: LocalDetectorConfig,
) -> tuple[PathRelationStateQALearner, CreditPathStatementDetector, StressPathQueryDetector]:
    statement = CreditPathStatementDetector(detector_cfg)
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
    return model, statement, query


def forced_size_scores(
    model: SizeRelationStateQALearner,
    row: dict[str, Any],
    forced_statement: tuple[int, dict[str, str]] | None,
    forced_query: dict[str, str] | None,
) -> np.ndarray:
    matrix = model.new_state()
    for idx, item in enumerate(row["context"]):
        if forced_statement is not None and idx == forced_statement[0]:
            detail = forced_statement[1]
            parsed = (detail["smaller"], detail["larger"])
        else:
            parsed = model.detect_statement(str(item["text"]))
        if parsed is None:
            continue
        smaller, larger = parsed
        associate(matrix, model.code(smaller), model.code(larger), model.cfg.lr)
    if forced_query is not None:
        query = (forced_query["smaller"], forced_query["larger"])
    else:
        query = model.detect_query(str(row["question"]))
    if query is None or "yes" not in model.answer_to_idx or "no" not in model.answer_to_idx:
        return model.fallback.scores(row)
    smaller, larger = query
    yes = recurrent_path_score(matrix, model.code(smaller), model.code(larger), model.cfg.max_hops, model.cfg.hop_decay)
    no = recurrent_path_score(matrix, model.code(larger), model.code(smaller), model.cfg.max_hops, model.cfg.hop_decay)
    scores = np.full(len(model.answer_to_idx), -4.0, dtype=np.float32)
    scores[model.answer_to_idx["yes"]] = model.cfg.score_scale * yes
    scores[model.answer_to_idx["no"]] = model.cfg.score_scale * no
    return scores


def forced_path_scores(
    model: PathRelationStateQALearner,
    row: dict[str, Any],
    forced_statement: tuple[int, dict[str, str]] | None,
) -> np.ndarray:
    matrices = model.new_state()
    for idx, item in enumerate(row["context"]):
        if forced_statement is not None and idx == forced_statement[0]:
            detail = forced_statement[1]
            parsed = (detail["source"], detail["direction"], detail["target"])
        else:
            parsed = model.detect_statement(str(item["text"]))
        if parsed is None:
            continue
        source, direction, target = parsed
        associate(matrices[direction], model.code(source), model.code(target), model.cfg.lr)
        associate(matrices[{"north": "south", "south": "north", "east": "west", "west": "east"}[direction]], model.code(target), model.code(source), model.cfg.lr)
    query = model.detect_query(str(row["question"]))
    if query is None:
        return model.fallback.scores(row)
    source, target = query
    scores = np.full(len(model.answer_to_idx), -4.0, dtype=np.float32)
    for answer, answer_idx in model.answer_to_idx.items():
        directions = answer.split()
        if not directions:
            continue
        scores[answer_idx] = model.cfg.score_scale * model.sequence_score(matrices, source, target, directions)
    return scores


def apply_answer_credit(
    config: str,
    model: Any,
    statement_detector: Any,
    query_detector: Any,
    train_rows: list[dict[str, Any]],
    answer_to_idx: dict[str, int],
    temperature: float,
    credit_epochs: int,
    credit_gate: str,
    min_gain: float,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for epoch in range(credit_epochs):
        rows_seen = wrong_rows = statement_updates = query_updates = 0
        gain_total = 0.0
        for row in train_rows:
            rows_seen += 1
            target = answer_to_idx[row["answer"]]
            base_scores = model.scores(row)
            pred = int(np.argmax(softmax(base_scores, temperature)))
            if pred == target:
                continue
            wrong_rows += 1
            base_obj = answer_objective(base_scores, target, temperature)
            if config == "en-qa18":
                for idx, item in enumerate(row["context"]):
                    detail = size_statement_surface_detail(str(item["text"]))
                    if detail is None:
                        continue
                    forced = forced_size_scores(model, row, (idx, detail), None)
                    gain = answer_objective(forced, target, temperature) - base_obj
                    if credit_gate == "error" or gain > min_gain:
                        statement_detector.credit_update(str(item["text"]), detail, max(gain, 0.1))
                        statement_updates += 1
                        gain_total += max(gain, 0.0)
                query_detail = size_query_surface_detail(str(row["question"]))
                if query_detail is not None:
                    forced = forced_size_scores(model, row, None, query_detail)
                    gain = answer_objective(forced, target, temperature) - base_obj
                    if credit_gate == "error" or gain > min_gain:
                        query_detector.credit_update(str(row["question"]), query_detail, max(gain, 0.1))
                        query_updates += 1
                        gain_total += max(gain, 0.0)
            else:
                for idx, item in enumerate(row["context"]):
                    detail = path_statement_surface_detail(str(item["text"]))
                    if detail is None:
                        continue
                    forced = forced_path_scores(model, row, (idx, detail))
                    gain = answer_objective(forced, target, temperature) - base_obj
                    if credit_gate == "error" or gain > min_gain:
                        statement_detector.credit_update(str(item["text"]), detail, max(gain, 0.1))
                        statement_updates += 1
                        gain_total += max(gain, 0.0)
        summary.append(
            {
                "config": config,
                "epoch": epoch,
                "rows_seen": rows_seen,
                "wrong_rows": wrong_rows,
                "statement_updates": statement_updates,
                "query_updates": query_updates,
                "gain_total": gain_total,
            }
        )
    return summary


def detector_rows(config: str, condition: str, detectors: list[tuple[str, Any]], splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for detector_name, detector in detectors:
        for split, split_rows in splits.items():
            rows.append({"config": config, "condition": condition, "detector": detector_name, "split": split, **detector.metrics(split_rows)})
    return rows


def run_config(args: argparse.Namespace, config: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = read_jsonl(args.data_dir / config / "train.jsonl", args.train_limit or None)
    validation_rows = read_jsonl(args.data_dir / config / "validation.jsonl", args.eval_limit or None)
    test_rows = read_jsonl(args.data_dir / config / "test.jsonl", args.eval_limit or None)
    strong_train = paraphrase_rows(train_rows, config, "strong")
    strong_validation = paraphrase_rows(validation_rows, config, "strong")
    strong_test = paraphrase_rows(test_rows, config, "strong")
    answer_vocab = build_answer_vocab(train_rows, validation_rows, test_rows)
    answer_to_idx = {answer: idx for idx, answer in enumerate(answer_vocab)}
    majority = MajorityBaseline(answer_to_idx)
    majority.fit(train_rows)
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
    if config == "en-qa18":
        model, statement, query = build_size_model(answer_to_idx, majority, relation_cfg, detector_cfg)
        detectors = [("size_statement", statement), ("size_query", query)]
    else:
        model, statement, query = build_path_model(answer_to_idx, majority, relation_cfg, detector_cfg)
        detectors = [("path_statement", statement), ("path_query", query)]
    model.fit(train_rows)
    splits = {"train": strong_train, "validation": strong_validation, "test": strong_test}
    summary_rows: list[dict[str, Any]] = []
    detector_summary: list[dict[str, Any]] = []
    credit_summary: list[dict[str, Any]] = []
    pre_summary, _ = evaluate_method(
        "seeded_pre_credit",
        model,
        splits,
        answer_to_idx,
        args.temperature,
        False,
        "pure_no_bp_relation_state_answer_credit",
    )
    for row in pre_summary:
        row["config"] = config
    summary_rows.extend(pre_summary)
    detector_summary.extend(detector_rows(config, "seeded_pre_credit", detectors, splits))
    credit_summary.extend(
        apply_answer_credit(
            config,
            model,
            statement,
            query,
            strong_train,
            answer_to_idx,
            args.temperature,
            args.credit_epochs,
            args.credit_gate,
            args.min_credit_gain,
        )
    )
    post_summary, _ = evaluate_method(
        "answer_credit",
        model,
        splits,
        answer_to_idx,
        args.temperature,
        False,
        "pure_no_bp_relation_state_answer_credit",
    )
    for row in post_summary:
        row["config"] = config
    summary_rows.extend(post_summary)
    detector_summary.extend(detector_rows(config, "answer_credit", detectors, splits))
    return summary_rows, detector_summary, credit_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--configs", nargs="+", default=["en-qa18", "en-qa19"])
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "babi_relation_delayed_credit")
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
    parser.add_argument("--credit-epochs", type=int, default=1)
    parser.add_argument("--credit-gate", choices=["error", "improvement"], default="error")
    parser.add_argument("--min-credit-gain", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, Any]] = []
    all_detector: list[dict[str, Any]] = []
    all_credit: list[dict[str, Any]] = []
    for config in args.configs:
        summary, detector, credit = run_config(args, config)
        all_summary.extend(summary)
        all_detector.extend(detector)
        all_credit.extend(credit)
    write_csv(args.out_dir / "summary.csv", all_summary)
    write_csv(args.out_dir / "detector_metrics.csv", all_detector)
    write_csv(args.out_dir / "credit_summary.csv", all_credit)
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
