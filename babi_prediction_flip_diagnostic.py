#!/usr/bin/env python3
"""Compare two bAbI unified-QA prediction CSVs example by example."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def row_key(row: dict[str, str]) -> tuple[str, str, int, str]:
    return (
        row.get("config", ""),
        row.get("split", ""),
        to_int(row.get("example_index", "")),
        row.get("target_answer", ""),
    )


def flip_type(base_full: int, cand_full: int) -> str:
    if base_full == 0 and cand_full == 1:
        return "helpful_exact"
    if base_full == 1 and cand_full == 0:
        return "harmful_exact"
    if base_full == 1 and cand_full == 1:
        return "both_correct"
    return "both_wrong"


def build_flip_rows(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    baseline_name: str,
    candidate_name: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    baseline_by_key = {row_key(row): row for row in baseline_rows}
    candidate_by_key = {row_key(row): row for row in candidate_rows}
    keys = sorted(set(baseline_by_key) & set(candidate_by_key), key=lambda item: (item[0], item[1], item[2], item[3]))
    missing_baseline = [str(key) for key in sorted(set(candidate_by_key) - set(baseline_by_key))]
    missing_candidate = [str(key) for key in sorted(set(baseline_by_key) - set(candidate_by_key))]
    rows: list[dict[str, Any]] = []
    for key in keys:
        base = baseline_by_key[key]
        cand = candidate_by_key[key]
        base_full = to_int(base.get("full_correct", ""))
        cand_full = to_int(cand.get("full_correct", ""))
        base_first = to_int(base.get("correct", ""))
        cand_first = to_int(cand.get("correct", ""))
        base_tok = to_int(base.get("full_answer_token_correct", ""))
        cand_tok = to_int(cand.get("full_answer_token_correct", ""))
        total_tok = max(
            to_int(base.get("full_answer_token_total", "")),
            to_int(cand.get("full_answer_token_total", "")),
        )
        rows.append(
            {
                "config": key[0],
                "split": key[1],
                "example_index": key[2],
                "question": base.get("question", ""),
                "target_answer": base.get("target_answer", ""),
                "baseline": baseline_name,
                "candidate": candidate_name,
                "baseline_prediction": base.get("prediction_answer_decoded", ""),
                "candidate_prediction": cand.get("prediction_answer_decoded", ""),
                "baseline_first_token": base.get("prediction_token", ""),
                "candidate_first_token": cand.get("prediction_token", ""),
                "baseline_full_correct": base_full,
                "candidate_full_correct": cand_full,
                "baseline_first_correct": base_first,
                "candidate_first_correct": cand_first,
                "baseline_token_correct": base_tok,
                "candidate_token_correct": cand_tok,
                "full_answer_token_total": total_tok,
                "token_correct_delta": cand_tok - base_tok,
                "target_prob_delta": to_float(cand.get("target_prob", "")) - to_float(base.get("target_prob", "")),
                "flip_type": flip_type(base_full, cand_full),
                "first_token_flip": int(base_first != cand_first),
            }
        )
    return rows, missing_baseline, missing_candidate


def build_summary_rows(flip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flip_rows:
        grouped[str(row["split"])].append(row)
        grouped["ALL"].append(row)
    summary: list[dict[str, Any]] = []
    for split in sorted(grouped, key=lambda name: (name != "ALL", name)):
        rows = grouped[split]
        total = len(rows)
        if total <= 0:
            continue
        counts = defaultdict(int)
        base_full = 0
        cand_full = 0
        base_tok = 0
        cand_tok = 0
        tok_total = 0
        prob_delta = 0.0
        first_flips = 0
        for row in rows:
            counts[str(row["flip_type"])] += 1
            base_full += int(row["baseline_full_correct"])
            cand_full += int(row["candidate_full_correct"])
            base_tok += int(row["baseline_token_correct"])
            cand_tok += int(row["candidate_token_correct"])
            tok_total += int(row["full_answer_token_total"])
            prob_delta += float(row["target_prob_delta"])
            first_flips += int(row["first_token_flip"])
        summary.append(
            {
                "split": split,
                "examples": total,
                "baseline_full_accuracy": base_full / total,
                "candidate_full_accuracy": cand_full / total,
                "full_accuracy_delta": (cand_full - base_full) / total,
                "baseline_token_accuracy": base_tok / tok_total if tok_total else 0.0,
                "candidate_token_accuracy": cand_tok / tok_total if tok_total else 0.0,
                "token_accuracy_delta": (cand_tok - base_tok) / tok_total if tok_total else 0.0,
                "helpful_exact": counts["helpful_exact"],
                "harmful_exact": counts["harmful_exact"],
                "both_correct": counts["both_correct"],
                "both_wrong": counts["both_wrong"],
                "net_helpful_exact": counts["helpful_exact"] - counts["harmful_exact"],
                "first_token_flip_rate": first_flips / total,
                "mean_target_prob_delta": prob_delta / total,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    flip_rows, missing_baseline, missing_candidate = build_flip_rows(
        read_csv(args.baseline),
        read_csv(args.candidate),
        args.baseline_name,
        args.candidate_name,
    )
    summary_rows = build_summary_rows(flip_rows)
    write_csv(args.out_dir / "flip_rows.csv", flip_rows)
    write_csv(args.out_dir / "flip_summary.csv", summary_rows)
    write_csv(
        args.out_dir / "missing_rows.csv",
        [
            {"missing_from": "baseline", "key": key}
            for key in missing_baseline
        ]
        + [
            {"missing_from": "candidate", "key": key}
            for key in missing_candidate
        ],
    )


if __name__ == "__main__":
    main()
