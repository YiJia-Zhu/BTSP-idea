#!/usr/bin/env python3
"""
Branch arbitration diagnostic for unified bAbI role-transition memory.

This tool trains the branch-separated role-transition memory, then scores each
prompt with separate base and role branch readouts.  It is diagnostic only: no
branch arbitration learned here is used for training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

import babi_unified_token_qa_experiment as babi


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


def build_parser() -> argparse.ArgumentParser:
    parser = babi.build_parser()
    parser.set_defaults(
        configs=["en-qa14", "en-qa17", "en-qa18"],
        out_dir=Path("output") / "babi_branch_arbitration_diagnostic",
        max_vocab=256,
        method="state_role_transition_online",
        answer_only_train=True,
        state_dim=64,
        state_order=224,
        micro_slots=64,
        micro_lr=0.35,
        micro_wrong_lr=0.02,
        micro_score_scale=8.0,
        role_query_order=16,
        role_hops=2,
        role_window=4,
        role_top_k=6,
        role_recency_decay=0.98,
        role_locality_decay=0.9,
        role_gate_lr=0.08,
        role_gate_wrong_lr=0.04,
        role_gate_strength=1.0,
        role_score_scale=1.5,
        role_downstream_bonus=0.75,
        role_channel_gates=True,
        role_final_score_only=True,
        role_event_cache_size=4096,
        role_branch_readout=True,
        role_branch_base_score_scale=8.0,
        role_branch_role_score_scale=8.0,
    )
    parser.add_argument(
        "--diagnostic-splits",
        nargs="+",
        default=["validation", "test"],
        choices=["train", "validation", "test"],
    )
    return parser


def compact_prompt_and_target(
    tokenizer: Any,
    raw_to_compact: np.ndarray,
    row: dict[str, Any],
) -> tuple[list[int], int] | None:
    prompt_ids = babi.to_compact(babi.encode(tokenizer, babi.row_prompt(row)), raw_to_compact)
    answer_ids_raw = babi.answer_token_ids(tokenizer, str(row["answer"]))
    if len(answer_ids_raw) != 1:
        return None
    answer_ids = babi.to_compact(answer_ids_raw, raw_to_compact)
    if not prompt_ids or not answer_ids:
        return None
    return prompt_ids, int(answer_ids[0])


def top_margin(scores: np.ndarray) -> float:
    if scores.size < 2:
        return 0.0
    top2 = np.partition(scores.astype(np.float64, copy=False), -2)[-2:]
    return float(top2[1] - top2[0])


def loss_pred_prob(scores: np.ndarray, target: int, temperature: float) -> tuple[float, int, float, float]:
    loss, pred, prob = babi.softmax_loss_and_pred(scores, target, temperature)
    return loss, pred, prob, top_margin(scores)


def component_scores(memory: Any, prompt_ids: list[int], target: int, temperature: float) -> dict[str, Any]:
    if not getattr(memory, "role_branch_readout", False):
        raise TypeError("branch diagnostic requires --role-branch-readout")
    babi.reset_dynamic(memory)
    babi.observe_prompt(memory, prompt_ids)
    context = np.array(list(prompt_ids)[-int(memory.max_order) :], dtype=np.int64)
    feature = memory.feature(context)

    components = memory.branch_component_scores(feature)
    full_scores = memory.scores_from_feature(feature)
    variants = {name: scores for name, scores in components.items()}
    variants["full"] = full_scores
    out: dict[str, Any] = {}
    for name, scores in variants.items():
        loss, pred, prob, margin = loss_pred_prob(scores, target, temperature)
        out[f"{name}_loss"] = loss
        out[f"{name}_pred"] = pred
        out[f"{name}_prob"] = prob
        out[f"{name}_margin"] = margin
        out[f"{name}_correct"] = int(pred == target)
    if "base_only_pred" in out and "role_only_pred" in out:
        out["base_role_agree"] = int(out["base_only_pred"] == out["role_only_pred"])
    if "base_only_pred" in out and "full_pred" in out:
        out["base_full_agree"] = int(out["base_only_pred"] == out["full_pred"])
    if "role_only_pred" in out and "full_pred" in out:
        out["role_full_agree"] = int(out["role_only_pred"] == out["full_pred"])
    return out


def row_variants(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "base_only",
        "role_only",
        "base_plus_role",
        "base_plus_direct",
        "base_plus_role_joint",
        "base_plus_direct_joint",
        "full",
    ]
    found = {key[: -len("_correct")] for row in rows for key in row if key.endswith("_correct")}
    ordered = [name for name in preferred if name in found]
    ordered.extend(sorted(found.difference(ordered)))
    return ordered


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    agreements: list[dict[str, Any]] = []
    variants = row_variants(rows)
    candidate_pairs = [
        ("base_only", "base_plus_direct"),
        ("base_only", "base_plus_direct_joint"),
        ("base_plus_direct", "base_plus_direct_joint"),
        ("base_plus_role_joint", "base_plus_direct_joint"),
        ("base_only", "full"),
    ]
    keys = sorted({(row["config"], row["split"]) for row in rows})
    for config, split in keys:
        subset = [row for row in rows if row["config"] == config and row["split"] == split]
        if not subset:
            continue
        for variant in variants:
            total = len(subset)
            correct = sum(int(row[f"{variant}_correct"]) for row in subset)
            loss = float(np.mean([row[f"{variant}_loss"] for row in subset]))
            margin = float(np.mean([row[f"{variant}_margin"] for row in subset]))
            summary.append(
                {
                    "config": config,
                    "split": split,
                    "variant": variant,
                    "examples": total,
                    "accuracy": correct / total,
                    "loss": loss,
                    "ppl": float(math.exp(min(loss, 20.0))),
                    "mean_margin": margin,
                }
            )
        total = len(subset)
        flips.append(
            {
                "config": config,
                "split": split,
                "examples": total,
                "base_to_full_helpful": sum(
                    row["base_only_correct"] == 0 and row["full_correct"] == 1 for row in subset
                ),
                "base_to_full_harmful": sum(
                    row["base_only_correct"] == 1 and row["full_correct"] == 0 for row in subset
                ),
                "base_full_same": sum(row["base_only_pred"] == row["full_pred"] for row in subset),
                "base_role_agree_rate": float(np.mean([row["base_role_agree"] for row in subset])),
                "base_full_agree_rate": float(np.mean([row["base_full_agree"] for row in subset])),
                "role_full_agree_rate": float(np.mean([row["role_full_agree"] for row in subset])),
            }
        )
        for left, right in candidate_pairs:
            if left not in variants or right not in variants:
                continue
            disagree = [row for row in subset if row[f"{left}_pred"] != row[f"{right}_pred"]]
            agreements.append(
                {
                    "config": config,
                    "split": split,
                    "left": left,
                    "right": right,
                    "examples": total,
                    "agree": total - len(disagree),
                    "agree_rate": (total - len(disagree)) / total,
                    "left_accuracy": float(np.mean([row[f"{left}_correct"] for row in subset])),
                    "right_accuracy": float(np.mean([row[f"{right}_correct"] for row in subset])),
                    "disagree_examples": len(disagree),
                    "left_wins_on_disagree": sum(
                        row[f"{left}_correct"] == 1 and row[f"{right}_correct"] == 0
                        for row in disagree
                    ),
                    "right_wins_on_disagree": sum(
                        row[f"{left}_correct"] == 0 and row[f"{right}_correct"] == 1
                        for row in disagree
                    ),
                    "both_wrong_on_disagree": sum(
                        row[f"{left}_correct"] == 0 and row[f"{right}_correct"] == 0
                        for row in disagree
                    ),
                    "both_correct_on_disagree": sum(
                        row[f"{left}_correct"] == 1 and row[f"{right}_correct"] == 1
                        for row in disagree
                    ),
                }
            )
    return summary, flips, agreements


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    splits_by_config = {config: babi.split_rows_for_config(args, config) for config in args.configs}
    train_rows = [row for config in args.configs for row in splits_by_config[config]["train"]]

    forced_raw: list[int] = []
    answer_raw_tokens: dict[str, list[int]] = {}
    for row in train_rows:
        answer = str(row["answer"])
        ids = babi.answer_token_ids(tokenizer, answer)
        answer_raw_tokens.setdefault(answer, ids)
        forced_raw.extend(ids[:1])
    forced_raw.extend(int(x) for x in babi.encode(tokenizer, "Context:\nQuestion: Answer:"))

    train_text = "".join(babi.row_train_text(row) for row in train_rows)
    train_raw = babi.encode(tokenizer, train_text)
    kept_raw, raw_to_compact = babi.build_compact_vocab_with_forced(
        train_raw,
        forced_raw,
        args.max_vocab,
    )
    method_name, memory, memory_cfg = babi.build_memory(args, int(len(kept_raw)))

    start = time.perf_counter()
    for _ in range(max(args.train_epochs, 1)):
        for row in train_rows:
            compact = compact_prompt_and_target(tokenizer, raw_to_compact, row)
            if compact is None:
                continue
            prompt_ids, target = compact
            babi.train_answer_token(memory, prompt_ids, target, args.temperature)

    diagnostic_rows: list[dict[str, Any]] = []
    for config in args.configs:
        for split in args.diagnostic_splits:
            for idx, row in enumerate(splits_by_config[config][split]):
                compact = compact_prompt_and_target(tokenizer, raw_to_compact, row)
                if compact is None:
                    continue
                prompt_ids, target = compact
                row_scores = component_scores(memory, prompt_ids, target, args.temperature)
                diagnostic_rows.append(
                    {
                        "config": config,
                        "split": split,
                        "example_index": idx,
                        "target": target,
                        **row_scores,
                    }
                )

    summary_rows, flip_rows, agreement_rows = summarize(diagnostic_rows)
    write_csv(args.out_dir / "branch_rows.csv", diagnostic_rows)
    write_csv(args.out_dir / "branch_component_summary.csv", summary_rows)
    write_csv(args.out_dir / "branch_flip_summary.csv", flip_rows)
    write_csv(args.out_dir / "branch_pair_agreement_summary.csv", agreement_rows)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "method": method_name,
                "memory_cfg": memory_cfg,
                "vocab_size": int(len(kept_raw)),
                "wall_seconds": time.perf_counter() - start,
                "event_cache_stats": babi.safe_event_cache_stats(memory),
                "role_score_gate_stats": babi.safe_role_score_gate_stats(memory),
                "model_stores_raw_text": False,
                "artifact_contains_decoded_text": False,
                "note": "Branch arbitration diagnostic only; no branch rule is used for training.",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
