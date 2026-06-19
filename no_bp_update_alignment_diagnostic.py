#!/usr/bin/env python3
"""
Center-difference diagnostics for local no-BP updates.

This script is an analysis tool only.  It estimates a local finite-difference
loss direction and compares it with the update produced by no-BP rules.  The
finite-difference direction is never used as the training method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from babi_no_bp_qa_experiment import (
    DEFAULT_DATA_DIR,
    PhaseDendriticQALearner,
    PhaseQAConfig,
    build_answer_vocab,
    read_jsonl,
    softmax,
)
from compositional_cue_experiment import (
    SplitConfig,
    complex_bind_phase_vectors,
    conjugate_phase_vector,
    make_split,
    normalize_rows,
    normalize_vector,
    target_for_pair,
)


SCRIPT_DIR = Path(__file__).resolve().parent


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
    if denom == 0.0:
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
    if norm == 0.0 or target_norm == 0.0:
        return np.zeros_like(direction)
    return (direction * (target_norm / norm)).astype(np.float64)


def center_difference_direction(
    params: np.ndarray,
    loss_fn: Callable[[np.ndarray], float],
    eps: float,
) -> np.ndarray:
    flat = params.astype(np.float64, copy=True).ravel()
    grad = np.zeros_like(flat)
    for idx in range(flat.size):
        original = float(flat[idx])
        flat[idx] = original + eps
        plus = loss_fn(flat)
        flat[idx] = original - eps
        minus = loss_fn(flat)
        flat[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)
    return (-grad).reshape(params.shape)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["variant"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (task, variant), group in sorted(groups.items()):
        out.append(
            {
                "task": task,
                "variant": variant,
                "runs": len(group),
                "cosine_mean": float(np.mean([row["cosine_local_vs_center_diff"] for row in group])),
                "sign_agreement_mean": float(np.mean([row["sign_agreement"] for row in group])),
                "loss_change_local_mean": float(np.mean([row["loss_change_local"] for row in group])),
                "loss_change_center_diff_mean": float(
                    np.mean([row["loss_change_center_diff_scaled"] for row in group])
                ),
                "loss_change_random_mean": float(np.mean([row["loss_change_random_scaled"] for row in group])),
                "local_improves_count": int(sum(row["loss_change_local"] < 0.0 for row in group)),
                "center_diff_improves_count": int(
                    sum(row["loss_change_center_diff_scaled"] < 0.0 for row in group)
                ),
                "random_improves_count": int(sum(row["loss_change_random_scaled"] < 0.0 for row in group)),
            }
        )
    return out


def phase_vector(k: int, harmonics: int, phase_index: int) -> np.ndarray:
    phase = 2.0 * math.pi * (phase_index % k) / k
    values: list[float] = []
    for harmonic in range(1, harmonics + 1):
        values.append(math.cos(harmonic * phase))
        values.append(math.sin(harmonic * phase))
    return normalize_vector(np.array(values, dtype=np.float32))


def decode_phase_params(params: np.ndarray, k: int, feature_dim: int) -> tuple[np.ndarray, np.ndarray]:
    first = normalize_rows(params[: k * feature_dim].reshape(k, feature_dim).astype(np.float32))
    second = normalize_rows(params[k * feature_dim :].reshape(k, feature_dim).astype(np.float32))
    return first, second


def compositional_loss(
    params: np.ndarray,
    k: int,
    harmonics: int,
    pairs: list[tuple[int, int]],
    logit_scale: float,
) -> float:
    feature_dim = 2 * harmonics
    first, second = decode_phase_params(params, k, feature_dim)
    target_phases = np.stack([phase_vector(k, harmonics, target) for target in range(k)], axis=0)
    losses = []
    for a, b in pairs:
        target = target_for_pair(a, b, k)
        feature = complex_bind_phase_vectors(first[a], second[b])
        probs = softmax(logit_scale * (target_phases @ feature))
        losses.append(-math.log(float(probs[target]) + 1e-9))
    return float(np.mean(losses)) if losses else 0.0


def local_compositional_update(
    params: np.ndarray,
    k: int,
    harmonics: int,
    pairs: list[tuple[int, int]],
    lr: float,
) -> np.ndarray:
    feature_dim = 2 * harmonics
    first, second = decode_phase_params(params, k, feature_dim)
    target_phases = np.stack([phase_vector(k, harmonics, target) for target in range(k)], axis=0)
    for a, b in pairs:
        target_phase = target_phases[target_for_pair(a, b, k)]
        first_old = first[a].copy()
        second_old = second[b].copy()
        desired_first = complex_bind_phase_vectors(target_phase, conjugate_phase_vector(second_old))
        first_new = normalize_vector((1.0 - lr) * first_old + lr * desired_first)
        desired_second = complex_bind_phase_vectors(conjugate_phase_vector(first_new), target_phase)
        second_new = normalize_vector((1.0 - lr) * second_old + lr * desired_second)
        first[a] = first_new
        second[b] = second_new
    return np.concatenate([first.ravel(), second.ravel()]).astype(np.float64)


def run_compositional_diagnostic(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in args.k_values:
        harmonics = args.phase_harmonics if args.phase_harmonics > 0 else max(1, k // 2)
        feature_dim = 2 * harmonics
        target_phases = np.stack([phase_vector(k, harmonics, target) for target in range(k)], axis=0)
        del target_phases
        for seed in args.seeds:
            split_cfg = SplitConfig(k=k, heldout_fraction=args.heldout_fraction, seed=seed)
            train_pairs, heldout_pairs = make_split(split_cfg)
            batch = train_pairs[: min(args.batch_size, len(train_pairs))]
            rng = np.random.default_rng(seed + 1009)
            first = normalize_rows(rng.normal(0.0, 1.0, (k, feature_dim)).astype(np.float32))
            second = normalize_rows(rng.normal(0.0, 1.0, (k, feature_dim)).astype(np.float32))
            params = np.concatenate([first.ravel(), second.ravel()]).astype(np.float64)
            loss_fn = lambda flat: compositional_loss(flat, k, harmonics, batch, args.phase_logit_scale)

            before = loss_fn(params)
            local_params = local_compositional_update(params, k, harmonics, batch, args.target_phase_lr)
            local_delta = local_params - params
            cd_direction = center_difference_direction(params, loss_fn, args.center_eps).ravel()
            local_delta_flat = local_delta.ravel()
            local_norm = vector_norm(local_delta_flat)
            cd_step = scaled_like(cd_direction, local_norm)
            random_direction = rng.normal(0.0, 1.0, params.shape).astype(np.float64)
            random_step = scaled_like(random_direction, local_norm)

            local_after = loss_fn(params + local_delta_flat)
            cd_after = loss_fn(params + cd_step)
            random_after = loss_fn(params + random_step)
            rows.append(
                {
                    "task": "compositional_cue",
                    "variant": "target_only_phase_codes",
                    "seed": seed,
                    "k": k,
                    "batch_size": len(batch),
                    "heldout_pairs": len(heldout_pairs),
                    "parameter_count": int(params.size),
                    "center_eps": args.center_eps,
                    "loss_before": before,
                    "loss_after_local": local_after,
                    "loss_after_center_diff_scaled": cd_after,
                    "loss_after_random_scaled": random_after,
                    "loss_change_local": local_after - before,
                    "loss_change_center_diff_scaled": cd_after - before,
                    "loss_change_random_scaled": random_after - before,
                    "local_update_norm": local_norm,
                    "center_diff_direction_norm": vector_norm(cd_direction),
                    "cosine_local_vs_center_diff": cosine_similarity(local_delta_flat, cd_direction),
                    "sign_agreement": sign_agreement(local_delta_flat, cd_direction),
                }
            )
    return rows


def batch_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return (exp_scores / np.sum(exp_scores, axis=1, keepdims=True)).astype(np.float64)


def qa_scores_from_weights(
    weights: np.ndarray,
    feats: np.ndarray,
    cfg: PhaseQAConfig,
    branch_gains: np.ndarray,
) -> np.ndarray:
    branch_scores = np.einsum("abd,nbd->nab", weights.astype(np.float32), feats.astype(np.float32))
    scores = cfg.score_scale * (branch_scores * branch_gains[None, None, :]).sum(axis=2)
    if cfg.branch_agreement > 0.0:
        centered = branch_scores - np.mean(branch_scores, axis=2, keepdims=True)
        scores = scores + cfg.branch_agreement * (-np.var(centered, axis=2))
    return scores.astype(np.float64)


def qa_loss_from_weights(
    weights: np.ndarray,
    feats: np.ndarray,
    targets: np.ndarray,
    cfg: PhaseQAConfig,
    branch_gains: np.ndarray,
) -> float:
    probs = batch_softmax(qa_scores_from_weights(weights, feats, cfg, branch_gains))
    losses = -np.log(np.maximum(probs[np.arange(len(targets)), targets], 1e-9))
    return float(np.mean(losses)) if len(losses) else 0.0


def local_qa_update(
    model: PhaseDendriticQALearner,
    rows: list[dict[str, Any]],
    answer_to_idx: dict[str, int],
) -> np.ndarray:
    original = model.weights.copy()
    model.weights = original.copy()
    for row in rows:
        target = answer_to_idx[row["answer"]]
        feats = model.features(row)
        scores = model.scores_from_features(feats)
        probs = softmax(scores, model.cfg.temperature)
        pred = int(np.argmax(probs))
        target_scale = model.cfg.lr * (1.0 - float(probs[target]))
        wrong_scale = model.cfg.wrong_lr * float(probs[pred])
        model.weights[target] += target_scale * feats
        if pred != target:
            model.weights[pred] -= wrong_scale * feats
        model.weights[target] = model.normalize_answer(model.weights[target])
        if pred != target:
            model.weights[pred] = model.normalize_answer(model.weights[pred])
    updated = model.weights.copy()
    model.weights = original
    return updated


def run_babi_qa_diagnostic(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config_dir = args.data_dir / args.qa_config
    train_rows = read_jsonl(config_dir / "train.jsonl", None)
    validation_rows = read_jsonl(config_dir / "validation.jsonl", None)
    answer_vocab = build_answer_vocab(train_rows, validation_rows)
    answer_to_idx = {answer: idx for idx, answer in enumerate(answer_vocab)}
    for seed in args.seeds:
        cfg = PhaseQAConfig(
            phase_dim=args.qa_phase_dim,
            lr=args.qa_phase_lr,
            wrong_lr=args.qa_phase_wrong_lr,
            epochs=args.qa_pretrain_epochs,
            score_scale=args.qa_score_scale,
            temperature=args.qa_temperature,
            branch_agreement=args.qa_branch_agreement,
            seed=seed,
        )
        model = PhaseDendriticQALearner(answer_to_idx, cfg)
        pretrain_rows = train_rows[: min(args.qa_pretrain_rows, len(train_rows))]
        if args.qa_pretrain_epochs > 0 and pretrain_rows:
            model.fit(pretrain_rows)
        start = min(len(pretrain_rows), len(train_rows) - 1)
        batch = train_rows[start : start + args.batch_size]
        if len(batch) < args.batch_size:
            batch = train_rows[: min(args.batch_size, len(train_rows))]
        feats = np.stack([model.features(row) for row in batch], axis=0)
        targets = np.array([answer_to_idx[row["answer"]] for row in batch], dtype=np.int64)
        weights = model.weights.astype(np.float64, copy=True)
        shape = weights.shape
        loss_fn = lambda flat: qa_loss_from_weights(
            flat.reshape(shape),
            feats,
            targets,
            cfg,
            model.branch_gains,
        )

        before = loss_fn(weights.ravel())
        local_weights = local_qa_update(model, batch, answer_to_idx).astype(np.float64)
        local_delta = (local_weights - weights).ravel()
        cd_direction = center_difference_direction(weights.ravel(), loss_fn, args.center_eps).ravel()
        local_norm = vector_norm(local_delta)
        cd_step = scaled_like(cd_direction, local_norm)
        rng = np.random.default_rng(seed + 2027)
        random_step = scaled_like(rng.normal(0.0, 1.0, local_delta.shape), local_norm)

        local_after = loss_fn(weights.ravel() + local_delta)
        cd_after = loss_fn(weights.ravel() + cd_step)
        random_after = loss_fn(weights.ravel() + random_step)
        rows.append(
            {
                "task": "babi_qa",
                "variant": args.qa_config,
                "seed": seed,
                "k": len(answer_vocab),
                "batch_size": len(batch),
                "heldout_pairs": "",
                "parameter_count": int(weights.size),
                "center_eps": args.center_eps,
                "loss_before": before,
                "loss_after_local": local_after,
                "loss_after_center_diff_scaled": cd_after,
                "loss_after_random_scaled": random_after,
                "loss_change_local": local_after - before,
                "loss_change_center_diff_scaled": cd_after - before,
                "loss_change_random_scaled": random_after - before,
                "local_update_norm": local_norm,
                "center_diff_direction_norm": vector_norm(cd_direction),
                "cosine_local_vs_center_diff": cosine_similarity(local_delta, cd_direction),
                "sign_agreement": sign_agreement(local_delta, cd_direction),
                "qa_pretrain_rows": len(pretrain_rows),
                "qa_pretrain_epochs": args.qa_pretrain_epochs,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "no_bp_update_alignment")
    parser.add_argument("--tasks", nargs="+", default=["compositional", "babi"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--center-eps", type=float, default=1e-3)

    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--phase-harmonics", type=int, default=0)
    parser.add_argument("--phase-logit-scale", type=float, default=8.0)
    parser.add_argument("--target-phase-lr", type=float, default=0.20)

    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--qa-config", default="en-qa1")
    parser.add_argument("--qa-phase-dim", type=int, default=8)
    parser.add_argument("--qa-phase-lr", type=float, default=0.08)
    parser.add_argument("--qa-phase-wrong-lr", type=float, default=0.03)
    parser.add_argument("--qa-pretrain-epochs", type=int, default=1)
    parser.add_argument("--qa-pretrain-rows", type=int, default=120)
    parser.add_argument("--qa-score-scale", type=float, default=6.0)
    parser.add_argument("--qa-temperature", type=float, default=1.0)
    parser.add_argument("--qa-branch-agreement", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    task_set = set(args.tasks)
    if "all" in task_set or "compositional" in task_set:
        rows.extend(run_compositional_diagnostic(args))
    if "all" in task_set or "babi" in task_set:
        rows.extend(run_babi_qa_diagnostic(args))

    summary_rows = aggregate(rows)
    write_csv(args.out_dir / "summary.csv", rows)
    write_csv(args.out_dir / "aggregate.csv", summary_rows)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "note": "center-difference is used only as a diagnostic direction, not as an optimizer",
                "phase_qa_config": asdict(
                    PhaseQAConfig(
                        phase_dim=args.qa_phase_dim,
                        lr=args.qa_phase_lr,
                        wrong_lr=args.qa_phase_wrong_lr,
                        epochs=args.qa_pretrain_epochs,
                        score_scale=args.qa_score_scale,
                        temperature=args.qa_temperature,
                        branch_agreement=args.qa_branch_agreement,
                        seed=args.seeds[0] if args.seeds else 0,
                    )
                ),
            },
            f,
            indent=2,
            default=str,
            sort_keys=True,
        )

    print("Aggregate:")
    for row in summary_rows:
        print(
            f"  {row['task']} {row['variant']}: "
            f"cos={row['cosine_mean']:.3f} sign={row['sign_agreement_mean']:.3f} "
            f"local_dloss={row['loss_change_local_mean']:.3f} "
            f"cd_dloss={row['loss_change_center_diff_mean']:.3f}"
        )
    print(f"wrote {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
