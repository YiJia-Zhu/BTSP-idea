#!/usr/bin/env python3
"""
Center-difference diagnostic for the unified bAbI role-gate update.

This is an analysis tool only.  It compares the local no-BP role-gate update
with a finite-difference loss descent direction.  The finite-difference
direction is never used for training.
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


def vector_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x.astype(np.float64, copy=False)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = vector_norm(a) * vector_norm(b)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a.ravel().astype(np.float64), b.ravel().astype(np.float64)) / denom)


def sign_agreement(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    aa = a.ravel()
    bb = b.ravel()
    mask = (np.abs(aa) > eps) & (np.abs(bb) > eps)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.sign(aa[mask]) == np.sign(bb[mask])))


def scaled_like(direction: np.ndarray, target_norm: float) -> np.ndarray:
    norm = vector_norm(direction)
    if norm <= 0.0 or target_norm <= 0.0:
        return np.zeros_like(direction)
    return (direction * (target_norm / norm)).astype(np.float64)


def normalized_gate_step(params: np.ndarray, step: np.ndarray) -> np.ndarray:
    updated = (params + step).astype(np.float64, copy=True)
    for idx in range(updated.shape[0]):
        norm = vector_norm(updated[idx])
        if norm > 0.0:
            updated[idx] /= norm
    return updated


def selected_center_difference_direction(
    params: np.ndarray,
    loss_fn: Any,
    selected: np.ndarray,
    eps: float,
) -> np.ndarray:
    flat = params.astype(np.float64, copy=True).ravel()
    grad = np.zeros_like(flat)
    for idx in selected:
        idx = int(idx)
        original = float(flat[idx])
        flat[idx] = original + eps
        plus = loss_fn(flat.reshape(params.shape))
        flat[idx] = original - eps
        minus = loss_fn(flat.reshape(params.shape))
        flat[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)
    return (-grad).reshape(params.shape)


def build_parser() -> argparse.ArgumentParser:
    parser = babi.build_parser()
    parser.set_defaults(
        configs=["en-qa2"],
        out_dir=Path("output") / "babi_role_gate_alignment_r162",
        max_vocab=256,
        train_limit=80,
        eval_limit=0,
        method="state_role_transition_online",
        answer_only_train=True,
        state_dim=32,
        state_order=160,
        micro_slots=16,
        micro_score_scale=8.0,
        role_query_order=16,
        role_hops=2,
        role_window=4,
        role_top_k=6,
        role_score_scale=1.5,
        role_gate_strength=1.0,
        role_gate_lr=0.08,
        role_gate_wrong_lr=0.04,
        role_downstream_bonus=0.75,
        role_channel_gates=True,
        role_final_score_only=True,
    )
    parser.add_argument("--diagnostic-limit", type=int, default=12)
    parser.add_argument("--center-eps", type=float, default=1e-3)
    parser.add_argument("--max-diff-dims", type=int, default=64)
    return parser


def compact_prompt_and_target(
    tokenizer: Any,
    raw_to_compact: np.ndarray,
    row: dict[str, Any],
) -> tuple[list[int], int] | None:
    prompt_ids = babi.to_compact(babi.encode(tokenizer, babi.row_prompt(row)), raw_to_compact)
    answer_ids_raw = babi.answer_token_ids(tokenizer, str(row["answer"]))
    answer_ids = babi.to_compact(answer_ids_raw[:1], raw_to_compact)
    if not prompt_ids or not answer_ids:
        return None
    return prompt_ids, int(answer_ids[0])


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    return [
        {
            "rows": len(rows),
            "cosine_mean": float(np.mean([row["cosine_local_vs_center_diff"] for row in rows])),
            "sign_agreement_mean": float(np.mean([row["sign_agreement"] for row in rows])),
            "loss_change_local_mean": float(np.mean([row["loss_change_local"] for row in rows])),
            "loss_change_center_diff_mean": float(
                np.mean([row["loss_change_center_diff_scaled"] for row in rows])
            ),
            "loss_change_random_mean": float(np.mean([row["loss_change_random_scaled"] for row in rows])),
            "local_improves_count": int(sum(row["loss_change_local"] < 0.0 for row in rows)),
            "center_diff_improves_count": int(
                sum(row["loss_change_center_diff_scaled"] < 0.0 for row in rows)
            ),
            "random_improves_count": int(sum(row["loss_change_random_scaled"] < 0.0 for row in rows)),
        }
    ]


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    rows_by_config = {
        config: babi.read_jsonl(args.data_dir / config / "train.jsonl", args.train_limit or None)
        for config in args.configs
    }
    train_rows = [row for config in args.configs for row in rows_by_config[config]]

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
    method_name, memory, _ = babi.build_memory(args, int(len(kept_raw)))
    if not hasattr(memory, "compute_role_gate_delta"):
        raise TypeError("selected method does not expose role-gate deltas")

    start = time.perf_counter()
    result_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + 65537)
    for example_idx, row in enumerate(train_rows):
        compact = compact_prompt_and_target(tokenizer, raw_to_compact, row)
        if compact is None:
            continue
        prompt_ids, target = compact
        context = np.array(list(prompt_ids)[-int(memory.max_order) :], dtype=np.int64)
        tokens = [int(token) for token in context.tolist()]

        feature = memory.feature(context)
        scores = memory.scores_from_feature(feature)
        loss_before, pred_before, _ = babi.softmax_loss_and_pred(scores, target, args.temperature)
        adjusted = scores.astype(np.float32, copy=True)
        adjusted[int(target)] = -np.inf
        wrong = int(np.argmax(adjusted))
        target_score = float(scores[int(target)])
        should_credit = float(adjusted[wrong]) + float(memory.margin) > target_score
        local_delta = memory.compute_role_gate_delta(tokens, target, wrong, should_credit)

        if np.any(local_delta) and len(result_rows) < args.diagnostic_limit:
            params = memory.role_gate_weights.astype(np.float64, copy=True)
            local_flat = local_delta.ravel().astype(np.float64)
            if args.max_diff_dims > 0 and local_flat.size > args.max_diff_dims:
                selected = np.argsort(-np.abs(local_flat))[: args.max_diff_dims]
            else:
                selected = np.arange(local_flat.size)

            def loss_for_gate(gate_params: np.ndarray) -> float:
                old = memory.role_gate_weights.copy()
                memory.role_gate_weights = gate_params.astype(np.float32, copy=True)
                try:
                    loss, _, _ = babi.score_prompt_answer(memory, prompt_ids, target, args.temperature)
                finally:
                    memory.role_gate_weights = old
                return float(loss)

            cd_direction = selected_center_difference_direction(
                params,
                loss_for_gate,
                selected,
                args.center_eps,
            )
            selected_mask = np.zeros(local_flat.size, dtype=bool)
            selected_mask[selected] = True
            local_selected = local_flat[selected_mask]
            cd_selected = cd_direction.ravel()[selected_mask]
            local_norm = vector_norm(local_delta)
            cd_step = scaled_like(cd_direction, local_norm)
            random_direction = rng.normal(0.0, 1.0, params.shape).astype(np.float64)
            random_step = scaled_like(random_direction, local_norm)
            local_after = loss_for_gate(normalized_gate_step(params, local_delta))
            cd_after = loss_for_gate(normalized_gate_step(params, cd_step))
            random_after = loss_for_gate(normalized_gate_step(params, random_step))
            result_rows.append(
                {
                    "example_index": example_idx,
                    "target_compact": int(target),
                    "wrong_compact": int(wrong),
                    "pred_before": int(pred_before),
                    "should_credit": bool(should_credit),
                    "selected_dims": int(selected.size),
                    "loss_before": float(loss_before),
                    "local_delta_norm": local_norm,
                    "center_diff_norm": vector_norm(cd_direction),
                    "cosine_local_vs_center_diff": cosine_similarity(local_selected, cd_selected),
                    "sign_agreement": sign_agreement(local_selected, cd_selected),
                    "loss_change_local": float(local_after - loss_before),
                    "loss_change_center_diff_scaled": float(cd_after - loss_before),
                    "loss_change_random_scaled": float(random_after - loss_before),
                }
            )

        memory.update(context, target)
        if len(result_rows) >= args.diagnostic_limit:
            break

    summary_rows = summarize(result_rows)
    write_csv(args.out_dir / "role_gate_alignment_rows.csv", result_rows)
    write_csv(args.out_dir / "role_gate_alignment_summary.csv", summary_rows)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "method": method_name,
                "answer_raw_tokens": answer_raw_tokens,
                "kept_raw_token_count": int(len(kept_raw)),
                "rows": len(result_rows),
                "wall_seconds": time.perf_counter() - start,
                "note": "Finite differences are diagnostic only and are never used as training updates.",
            },
            f,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
