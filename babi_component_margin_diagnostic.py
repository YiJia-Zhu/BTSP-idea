#!/usr/bin/env python3
"""Compare per-token bAbI component-margin diagnostics."""

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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value or ""))
    except ValueError:
        return default


def to_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or "")
    except ValueError:
        return default


def row_key(row: dict[str, str]) -> tuple[str, str, int, str, int, int, str]:
    return (
        row.get("config", ""),
        row.get("split", ""),
        to_int(row.get("example_index")),
        row.get("decode_phase", ""),
        to_int(row.get("slot")),
        to_int(row.get("target_id")),
        row.get("target_answer", ""),
    )


def compare_rows(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    baseline_name: str,
    candidate_name: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    baseline_by_key = {row_key(row): row for row in baseline_rows}
    candidate_by_key = {row_key(row): row for row in candidate_rows}
    keys = sorted(set(baseline_by_key) & set(candidate_by_key))
    missing_baseline = [str(key) for key in sorted(set(candidate_by_key) - set(baseline_by_key))]
    missing_candidate = [str(key) for key in sorted(set(baseline_by_key) - set(candidate_by_key))]

    rows: list[dict[str, Any]] = []
    for key in keys:
        base = baseline_by_key[key]
        cand = candidate_by_key[key]
        base_correct = to_int(base.get("correct"))
        cand_correct = to_int(cand.get("correct"))
        base_prob = to_float(base.get("final_target_prob"))
        cand_prob = to_float(cand.get("final_target_prob"))
        base_margin = to_float(base.get("final_margin"))
        cand_margin = to_float(cand.get("final_margin"))
        base_target_vs_best = to_float(base.get("final_target_vs_best_wrong"))
        cand_target_vs_best = to_float(cand.get("final_target_vs_best_wrong"))
        coupling_active_count = to_int(cand.get("coupling_delta_active_count"))
        wrong_cleanup_active_count = to_int(cand.get("wrong_cleanup_delta_active_count"))
        conflict_rescue_active_count = to_int(cand.get("conflict_rescue_delta_active_count"))
        rows.append(
            {
                "config": key[0],
                "split": key[1],
                "example_index": key[2],
                "decode_phase": key[3],
                "slot": key[4],
                "target_answer": base.get("target_answer", ""),
                "target_token": base.get("target_token", ""),
                "baseline": baseline_name,
                "candidate": candidate_name,
                "baseline_prediction_token": base.get("prediction_token", ""),
                "candidate_prediction_token": cand.get("prediction_token", ""),
                "baseline_correct": base_correct,
                "candidate_correct": cand_correct,
                "helpful_flip": int(base_correct == 0 and cand_correct == 1),
                "harmful_flip": int(base_correct == 1 and cand_correct == 0),
                "both_wrong": int(base_correct == 0 and cand_correct == 0),
                "both_correct": int(base_correct == 1 and cand_correct == 1),
                "baseline_final_target_prob": base_prob,
                "candidate_final_target_prob": cand_prob,
                "final_target_prob_delta": cand_prob - base_prob,
                "baseline_final_target_vs_best_wrong": base_target_vs_best,
                "candidate_final_target_vs_best_wrong": cand_target_vs_best,
                "final_target_vs_best_delta": cand_target_vs_best - base_target_vs_best,
                "baseline_final_margin": base_margin,
                "candidate_final_margin": cand_margin,
                "final_margin_delta": cand_margin - base_margin,
                "candidate_coupling_delta_active_count": coupling_active_count,
                "candidate_coupling_delta_active": int(coupling_active_count > 0),
                "candidate_coupling_delta_target_score": to_float(cand.get("coupling_delta_target_score")),
                "candidate_coupling_delta_target_vs_best_wrong": to_float(
                    cand.get("coupling_delta_target_vs_best_wrong")
                ),
                "candidate_after_coupling_target_prob_gain_vs_base": to_float(
                    cand.get("after_coupling_target_prob_gain_vs_base")
                ),
                "candidate_wrong_cleanup_delta_active_count": wrong_cleanup_active_count,
                "candidate_wrong_cleanup_delta_active": int(wrong_cleanup_active_count > 0),
                "candidate_wrong_cleanup_delta_target_score": to_float(
                    cand.get("wrong_cleanup_delta_target_score")
                ),
                "candidate_wrong_cleanup_delta_target_vs_best_wrong": to_float(
                    cand.get("wrong_cleanup_delta_target_vs_best_wrong")
                ),
                "candidate_after_cleanup_target_prob_gain_vs_base": to_float(
                    cand.get("after_cleanup_target_prob_gain_vs_base")
                ),
                "candidate_conflict_rescue_delta_active_count": conflict_rescue_active_count,
                "candidate_conflict_rescue_delta_active": int(conflict_rescue_active_count > 0),
                "candidate_conflict_rescue_delta_target_score": to_float(
                    cand.get("conflict_rescue_delta_target_score")
                ),
                "candidate_conflict_rescue_delta_target_vs_best_wrong": to_float(
                    cand.get("conflict_rescue_delta_target_vs_best_wrong")
                ),
                "candidate_after_conflict_target_prob_gain_vs_base": to_float(
                    cand.get("after_conflict_target_prob_gain_vs_base")
                ),
                "candidate_final_target_rank": to_int(cand.get("final_target_rank"), -1),
                "candidate_high_margin_wrong_0p20": int(cand_correct == 0 and cand_margin >= 0.20),
                "candidate_high_margin_wrong_0p50": int(cand_correct == 0 and cand_margin >= 0.50),
                "both_wrong_prob_up": int(base_correct == 0 and cand_correct == 0 and cand_prob > base_prob),
                "both_wrong_margin_up": int(
                    base_correct == 0 and cand_correct == 0 and cand_target_vs_best > base_target_vs_best
                ),
            }
        )
    return rows, missing_baseline, missing_candidate


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row[key]) for row in rows)


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        split = str(row["split"])
        phase = str(row["decode_phase"])
        slot = int(row["slot"])
        grouped[(split, phase, "slot", slot)].append(row)
        grouped[(split, phase, "all_slots", -1)].append(row)
        grouped[("ALL", phase, "slot", slot)].append(row)
        grouped[("ALL", phase, "all_slots", -1)].append(row)
    summary: list[dict[str, Any]] = []
    for (split, phase, slot_group, slot), group_rows in sorted(grouped.items()):
        n = len(group_rows)
        if n <= 0:
            continue
        base_correct = count(group_rows, "baseline_correct")
        cand_correct = count(group_rows, "candidate_correct")
        both_wrong = count(group_rows, "both_wrong")
        summary.append(
            {
                "split": split,
                "decode_phase": phase,
                "slot_group": slot_group,
                "slot": slot,
                "rows": n,
                "baseline_accuracy": base_correct / n,
                "candidate_accuracy": cand_correct / n,
                "accuracy_delta": (cand_correct - base_correct) / n,
                "helpful_flips": count(group_rows, "helpful_flip"),
                "harmful_flips": count(group_rows, "harmful_flip"),
                "net_flips": count(group_rows, "helpful_flip") - count(group_rows, "harmful_flip"),
                "both_wrong": both_wrong,
                "both_correct": count(group_rows, "both_correct"),
                "baseline_mean_final_target_prob": mean(group_rows, "baseline_final_target_prob"),
                "candidate_mean_final_target_prob": mean(group_rows, "candidate_final_target_prob"),
                "mean_final_target_prob_delta": mean(group_rows, "final_target_prob_delta"),
                "mean_final_target_vs_best_delta": mean(group_rows, "final_target_vs_best_delta"),
                "mean_final_margin_delta": mean(group_rows, "final_margin_delta"),
                "coupling_active_rate": count(group_rows, "candidate_coupling_delta_active") / n,
                "mean_coupling_delta_active_count": mean(group_rows, "candidate_coupling_delta_active_count"),
                "mean_coupling_delta_target_score": mean(group_rows, "candidate_coupling_delta_target_score"),
                "mean_coupling_delta_target_vs_best_wrong": mean(
                    group_rows,
                    "candidate_coupling_delta_target_vs_best_wrong",
                ),
                "wrong_cleanup_active_rate": count(group_rows, "candidate_wrong_cleanup_delta_active") / n,
                "mean_wrong_cleanup_delta_active_count": mean(
                    group_rows,
                    "candidate_wrong_cleanup_delta_active_count",
                ),
                "mean_wrong_cleanup_delta_target_score": mean(
                    group_rows,
                    "candidate_wrong_cleanup_delta_target_score",
                ),
                "mean_wrong_cleanup_delta_target_vs_best_wrong": mean(
                    group_rows,
                    "candidate_wrong_cleanup_delta_target_vs_best_wrong",
                ),
                "conflict_rescue_active_rate": count(group_rows, "candidate_conflict_rescue_delta_active") / n,
                "mean_conflict_rescue_delta_active_count": mean(
                    group_rows,
                    "candidate_conflict_rescue_delta_active_count",
                ),
                "mean_conflict_rescue_delta_target_score": mean(
                    group_rows,
                    "candidate_conflict_rescue_delta_target_score",
                ),
                "mean_conflict_rescue_delta_target_vs_best_wrong": mean(
                    group_rows,
                    "candidate_conflict_rescue_delta_target_vs_best_wrong",
                ),
                "both_wrong_prob_up": count(group_rows, "both_wrong_prob_up"),
                "both_wrong_prob_up_rate_among_both_wrong": (
                    count(group_rows, "both_wrong_prob_up") / both_wrong if both_wrong else 0.0
                ),
                "both_wrong_margin_up": count(group_rows, "both_wrong_margin_up"),
                "both_wrong_margin_up_rate_among_both_wrong": (
                    count(group_rows, "both_wrong_margin_up") / both_wrong if both_wrong else 0.0
                ),
                "candidate_high_margin_wrong_0p20": count(group_rows, "candidate_high_margin_wrong_0p20"),
                "candidate_high_margin_wrong_0p50": count(group_rows, "candidate_high_margin_wrong_0p50"),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-components", type=Path, required=True)
    parser.add_argument("--candidate-components", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    args = parser.parse_args()

    rows, missing_baseline, missing_candidate = compare_rows(
        read_csv(args.baseline_components),
        read_csv(args.candidate_components),
        args.baseline_name,
        args.candidate_name,
    )
    write_csv(args.out_dir / "component_comparison_rows.csv", rows)
    write_csv(args.out_dir / "component_summary.csv", build_summary(rows))
    write_csv(
        args.out_dir / "missing_rows.csv",
        [{"missing_from": "baseline", "key": key} for key in missing_baseline]
        + [{"missing_from": "candidate", "key": key} for key in missing_candidate],
    )


if __name__ == "__main__":
    main()
