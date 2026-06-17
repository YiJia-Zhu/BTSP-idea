#!/usr/bin/env python3
"""
Prequential TinyStories stream experiment for no-raw-data online learning.

This script reuses the existing no-BP memory implementations and evaluates them
in a strictly streaming setup:
  - warm up on the train prefix as an online stream
  - continue on the valid prefix without storing raw samples
  - report pre-update, online-update, and post-update CE/acc on each chunk
  - track memory growth and approximate serialized state size
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

import tinystories_llama_token_experiment as base


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = base.DEFAULT_TRAIN
DEFAULT_VALID = base.DEFAULT_VALID
DEFAULT_TOKENIZER = base.DEFAULT_TOKENIZER


@dataclass
class CombinedContextState:
    max_order: int
    semantic_weight: float


class CombinedContextMemory:
    """Exact sparse memory plus semantic bucket memory, updated online together."""

    def __init__(
        self,
        vocab_size: int,
        exact_cfg: base.SparseHebbianContextConfig,
        semantic_cfg: base.SemanticHebbianConfig,
        semantic_weight: float,
    ) -> None:
        self.exact = base.SparseHebbianContextMemory(vocab_size, exact_cfg)
        self.semantic = base.SemanticHebbianMemory(vocab_size, semantic_cfg)
        self.cfg = CombinedContextState(
            max_order=max(exact_cfg.max_order, semantic_cfg.order),
            semantic_weight=semantic_weight,
        )
        self.semantic_weight = semantic_weight
        self.vocab_size = vocab_size

    def scores(self, context: tuple[int, ...]) -> np.ndarray:
        exact_context = context[-self.exact.cfg.max_order :]
        semantic_context = context[-self.semantic.cfg.order :]
        return self.exact.scores(exact_context) + self.semantic_weight * self.semantic.scores(semantic_context)

    def update(self, context: tuple[int, ...], target: int) -> None:
        self.exact.update(context, target)
        self.semantic.update(context, target)

    def active_contexts(self) -> int:
        return self.exact.active_contexts() + self.semantic.active_contexts()


def decay_row_map(row_map: dict, decay: float) -> None:
    if decay >= 1.0:
        return
    for key in list(row_map.keys()):
        row_map[key] *= decay


def prune_row_map(row_map: dict, min_count: float) -> None:
    if min_count <= 0.0:
        return
    for key in list(row_map.keys()):
        if float(row_map[key]) < min_count:
            del row_map[key]


def prune_dense_counts(counts: np.ndarray, min_count: float, decay: float) -> None:
    if decay < 1.0:
        counts *= decay
    if min_count > 0.0:
        counts[counts < min_count] = 0.0


def prune_sparse_by_cap(memory: base.SparseHebbianContextMemory, max_contexts: int) -> None:
    if max_contexts <= 0:
        return
    scored: list[tuple[float, int, tuple[int, ...]]] = []
    for order in range(1, len(memory.tables)):
        for key, row in memory.tables[order].items():
            scored.append((float(sum(row.values())), order, key))
    if len(scored) <= max_contexts:
        return
    keep = {(order, key) for _, order, key in sorted(scored, reverse=True)[:max_contexts]}
    for order in range(1, len(memory.tables)):
        table = memory.tables[order]
        for key in list(table.keys()):
            if (order, key) not in keep:
                del table[key]


def prune_sparse_memory(
    memory: base.SparseHebbianContextMemory,
    min_count: float,
    decay: float,
    max_contexts: int,
) -> None:
    prune_dense_counts(memory.unigram, min_count, decay)
    if min_count <= 0.0 and decay >= 1.0 and max_contexts <= 0:
        return
    for table in memory.tables:
        empty_keys: list[tuple[int, ...]] = []
        for key, row in table.items():
            if decay < 1.0:
                decay_row_map(row, decay)
            prune_row_map(row, min_count)
            if not row:
                empty_keys.append(key)
        for key in empty_keys:
            del table[key]
    prune_sparse_by_cap(memory, max_contexts)


def prune_semantic_by_cap(memory: base.SemanticHebbianMemory, max_contexts: int) -> None:
    if max_contexts <= 0 or len(memory.tables) <= max_contexts:
        return
    scored = [(float(sum(row.values())), key) for key, row in memory.tables.items()]
    keep = {key for _, key in sorted(scored, reverse=True)[:max_contexts]}
    for key in list(memory.tables.keys()):
        if key not in keep:
            del memory.tables[key]


def prune_semantic_memory(
    memory: base.SemanticHebbianMemory,
    min_count: float,
    decay: float,
    max_contexts: int,
) -> None:
    prune_dense_counts(memory.unigram, min_count, decay)
    if min_count <= 0.0 and decay >= 1.0 and max_contexts <= 0:
        return
    empty_keys: list[int] = []
    for key, row in memory.tables.items():
        if decay < 1.0:
            decay_row_map(row, decay)
        prune_row_map(row, min_count)
        if not row:
            empty_keys.append(key)
    for key in empty_keys:
        del memory.tables[key]
    prune_semantic_by_cap(memory, max_contexts)


def rebuild_continuation_sets(memory: base.ContinuationBackoffMemory) -> None:
    memory.continuation = {token: set() for token in range(memory.vocab_size)}
    memory.prev_token_pairs = set()
    for order in range(1, memory.cfg.max_order + 1):
        for key, row in memory.exact_tables[order].items():
            for token in row:
                memory.continuation[token].add(key)
                if order == 1 and len(key) == 1:
                    memory.prev_token_pairs.add((key[-1], token))


def prune_continuation_by_cap(memory: base.ContinuationBackoffMemory, max_contexts: int) -> None:
    if max_contexts <= 0:
        return
    scored: list[tuple[float, int, tuple[int, ...]]] = []
    for order in range(1, memory.cfg.max_order + 1):
        for key in memory.exact_tables[order]:
            scored.append((float(memory.context_totals[order].get(key, 0.0)), order, key))
    if len(scored) <= max_contexts:
        return
    keep = {(order, key) for _, order, key in sorted(scored, reverse=True)[:max_contexts]}
    for order in range(1, memory.cfg.max_order + 1):
        table = memory.exact_tables[order]
        totals = memory.context_totals[order]
        for key in list(table.keys()):
            if (order, key) not in keep:
                del table[key]
                totals.pop(key, None)
    rebuild_continuation_sets(memory)


def prune_continuation_memory(
    memory: base.ContinuationBackoffMemory,
    min_count: float,
    decay: float,
    max_contexts: int,
) -> None:
    prune_dense_counts(memory.unigram, min_count, decay)
    if min_count <= 0.0 and decay >= 1.0 and max_contexts <= 0:
        return
    for order in range(1, memory.cfg.max_order + 1):
        table = memory.exact_tables[order]
        totals = memory.context_totals[order]
        empty_keys: list[tuple[int, ...]] = []
        for key, row in table.items():
            if decay < 1.0:
                decay_row_map(row, decay)
            prune_row_map(row, min_count)
            if not row:
                empty_keys.append(key)
                continue
            totals[key] = float(sum(row.values()))
        for key in empty_keys:
            del table[key]
            totals.pop(key, None)
    prune_continuation_by_cap(memory, max_contexts)
    if max_contexts <= 0:
        rebuild_continuation_sets(memory)


def prune_any_memory(memory, min_count: float, decay: float, max_contexts: int) -> None:
    if hasattr(memory, "exact") and hasattr(memory, "semantic"):
        component_cap = max_contexts // 2 if max_contexts > 1 else max_contexts
        prune_sparse_memory(memory.exact, min_count, decay, component_cap)
        prune_semantic_memory(memory.semantic, min_count, decay, component_cap)
        return
    if isinstance(memory, base.SparseHebbianContextMemory):
        prune_sparse_memory(memory, min_count, decay, max_contexts)
        return
    if isinstance(memory, base.SemanticHebbianMemory):
        prune_semantic_memory(memory, min_count, decay, max_contexts)
        return
    if isinstance(memory, base.ContinuationBackoffMemory):
        prune_continuation_memory(memory, min_count, decay, max_contexts)
        return


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def truncate_history(tokens: Sequence[int], keep: int) -> list[int]:
    if keep <= 0:
        return []
    return list(tokens[-keep:])


def memory_order(memory) -> int:
    cfg = getattr(memory, "cfg", None)
    if cfg is None:
        raise AttributeError("memory has no cfg")
    if hasattr(cfg, "max_order"):
        return int(getattr(cfg, "max_order"))
    if hasattr(cfg, "order"):
        return int(getattr(cfg, "order"))
    raise AttributeError("memory cfg has no max_order/order")


def segment_windows(ids: np.ndarray, segment_tokens: int) -> list[tuple[int, int]]:
    if segment_tokens < 2:
        raise ValueError("segment_tokens must be >= 2")
    stride = max(segment_tokens - 1, 1)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < len(ids) - 1:
        end = min(len(ids), start + segment_tokens)
        if end - start < 2:
            break
        windows.append((start, end))
        if end >= len(ids):
            break
        start += stride
    return windows


def memory_state_bytes(memory) -> int:
    return len(pickle.dumps(memory, protocol=pickle.HIGHEST_PROTOCOL))


def summarize_losses(loss_sum: float, correct: int, total: int) -> dict:
    if total <= 0:
        return {"loss": float("nan"), "ppl": float("nan"), "accuracy": 0.0}
    loss = loss_sum / total
    return {
        "loss": float(loss),
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / total,
    }


def run_stream_phase(
    method: str,
    memory,
    ids: np.ndarray,
    segment_tokens: int,
    start_history: Sequence[int],
    temperature: float,
    update: bool,
    pass_name: str,
    show_progress: bool,
    prune_every_segments: int = 0,
    prune_min_count: float = 0.0,
    prune_decay: float = 1.0,
    prune_max_contexts: int = 0,
) -> tuple[dict, list[int], list[dict]]:
    order = memory_order(memory)
    history = truncate_history(start_history, max(order - 1, 0))
    windows = segment_windows(ids, segment_tokens)
    total_targets = sum(end - start - 1 for start, end in windows)
    progress = base.ProgressBar(f"{method} {pass_name}", max(total_targets, 1), show_progress)

    rows: list[dict] = []
    global_loss_sum = 0.0
    global_correct = 0
    global_total = 0
    segment_idx = 0
    processed_targets = 0

    for start, end in windows:
        segment = ids[start:end]
        segment_history = list(history)
        segment_loss_sum = 0.0
        segment_correct = 0
        segment_total = 0

        for idx in range(len(segment) - 1):
            current = int(segment[idx])
            target = int(segment[idx + 1])
            context = tuple(truncate_history(segment_history + [current], order))
            loss, pred = base.cross_entropy_from_scores(memory.scores(context), target, temperature)
            segment_loss_sum += loss
            segment_correct += int(pred == target)
            segment_total += 1
            if update:
                memory.update(context, target)
            segment_history = truncate_history(segment_history + [current], max(order - 1, 0))

        history = segment_history
        pruned = False
        if update and prune_every_segments > 0 and (segment_idx + 1) % prune_every_segments == 0:
            prune_any_memory(memory, prune_min_count, prune_decay, prune_max_contexts)
            pruned = True
        processed_targets += segment_total
        global_loss_sum += segment_loss_sum
        global_correct += segment_correct
        global_total += segment_total
        state_bytes = memory_state_bytes(memory)
        segment_summary = summarize_losses(segment_loss_sum, segment_correct, segment_total)
        rows.append(
            {
                "method": method,
                "pass": pass_name,
                "segment_idx": segment_idx,
                "segment_start": start,
                "segment_end": end,
                "segment_tokens": len(segment),
                "target_tokens": segment_total,
                "loss": segment_summary["loss"],
                "ppl": segment_summary["ppl"],
                "accuracy": segment_summary["accuracy"],
                "active_contexts": memory.active_contexts() if hasattr(memory, "active_contexts") else 0,
                "state_bytes": state_bytes,
                "bytes_per_target": state_bytes / max(processed_targets, 1),
                "pruned": pruned,
                "prune_min_count": prune_min_count,
                "prune_decay": prune_decay,
                "prune_max_contexts": prune_max_contexts,
            }
        )
        progress.update(processed_targets)
        segment_idx += 1

    progress.close()
    summary = summarize_losses(global_loss_sum, global_correct, global_total)
    summary.update(
        {
            "method": method,
            "pass": pass_name,
            "segment_count": segment_idx,
            "target_tokens": global_total,
            "active_contexts": memory.active_contexts() if hasattr(memory, "active_contexts") else 0,
            "state_bytes": memory_state_bytes(memory),
            "history_tokens": len(history),
            "bytes_per_target": memory_state_bytes(memory) / max(global_total, 1),
        }
    )
    return summary, history, rows


def evaluate_memory_sequence(
    memory,
    ids: np.ndarray,
    temperature: float,
    start_history: Sequence[int] | None = None,
) -> dict:
    order = memory_order(memory)
    history = truncate_history(start_history or [], max(order - 1, 0))
    losses: list[float] = []
    correct = 0
    total = 0
    for idx in range(len(ids) - 1):
        current = int(ids[idx])
        target = int(ids[idx + 1])
        context = tuple(truncate_history(history + [current], order))
        loss, pred = base.cross_entropy_from_scores(memory.scores(context), target, temperature)
        losses.append(loss)
        correct += int(pred == target)
        total += 1
        history = truncate_history(history + [current], max(order - 1, 0))
    summary = summarize_losses(float(np.sum(losses)), correct, total)
    summary["target_tokens"] = total
    return summary


def sample_context_memory(
    memory,
    tokenizer,
    kept_raw: np.ndarray,
    prompt: str,
    length: int,
    decode_cfg: base.DecodingConfig | None = None,
    seed: int = 0,
) -> str:
    ids = base.compact_prompt_ids(tokenizer, kept_raw, prompt)
    generated = list(ids)
    rng = np.random.default_rng(seed)
    order = memory_order(memory)

    for _ in range(length):
        context = tuple(generated[-order:])
        scores = memory.scores(context)
        current = (
            base.choose_next_token(scores, generated, decode_cfg, rng)
            if decode_cfg is not None
            else int(np.argmax(scores))
        )
        generated.append(current)

    return base.decode_compact_ids(tokenizer, kept_raw, generated)


def plot_stream_curves(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    fig, axes = plt.subplots(len(methods), 1, figsize=(8, 3.4 * len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        method_rows = [row for row in rows if row["method"] == method]
        passes = list(dict.fromkeys(str(row["pass"]) for row in method_rows))
        for pass_name in passes:
            pass_rows = [row for row in method_rows if row["pass"] == pass_name]
            xs = [int(row["segment_idx"]) for row in pass_rows]
            ys = [float(row["loss"]) for row in pass_rows]
            ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.5, label=pass_name)
        ax.set_title(method)
        ax.set_ylabel("segment CE")
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("segment index")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "tinystories_online_stream")
    parser.add_argument("--train-chars", type=int, default=50_000)
    parser.add_argument("--valid-chars", type=int, default=10_000)
    parser.add_argument("--max-vocab", type=int, default=256)
    parser.add_argument("--warmup-token-limit", type=int, default=0)
    parser.add_argument("--stream-token-limit", type=int, default=0)
    parser.add_argument("--retention-token-limit", type=int, default=1024)
    parser.add_argument("--segment-tokens", type=int, default=256)
    parser.add_argument("--prune-every-segments", type=int, default=0)
    parser.add_argument("--prune-min-count", type=float, default=0.0)
    parser.add_argument("--prune-decay", type=float, default=1.0)
    parser.add_argument("--prune-max-contexts", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sample-len", type=int, default=24)
    parser.add_argument("--sample-temperature", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=0.45)
    parser.add_argument("--no-repeat-ngram", type=int, default=4)
    parser.add_argument("--decode-top-k", type=int, default=0)
    parser.add_argument("--decode-temperature", type=float, default=0.9)
    parser.add_argument("--context-max-order", type=int, default=3)
    parser.add_argument("--context-alpha", type=float, default=0.05)
    parser.add_argument("--context-unigram-weight", type=float, default=0.15)
    parser.add_argument("--context-order-weight", type=float, default=1.0)
    parser.add_argument("--context-order-weight-growth", type=float, default=1.6)
    parser.add_argument("--context-score-mode", choices=["additive", "normalized"], default="additive")
    parser.add_argument("--context-smoothing", type=float, default=0.05)
    parser.add_argument("--context-backoff", type=float, default=0.35)
    parser.add_argument("--semantic-order", type=int, default=8)
    parser.add_argument("--semantic-dim", type=int, default=64)
    parser.add_argument("--semantic-hash-bits", type=int, default=12)
    parser.add_argument("--semantic-alpha", type=float, default=0.05)
    parser.add_argument("--semantic-bucket-weight", type=float, default=1.0)
    parser.add_argument("--semantic-unigram-weight", type=float, default=0.15)
    parser.add_argument("--semantic-combine-weight", type=float, default=0.5)
    parser.add_argument("--continuation-max-order", type=int, default=3)
    parser.add_argument("--continuation-discount", type=float, default=0.75)
    parser.add_argument("--continuation-unigram-weight", type=float, default=0.10)
    parser.add_argument("--continuation-weight", type=float, default=1.0)
    parser.add_argument("--continuation-exact-weight", type=float, default=1.0)
    parser.add_argument("--continuation-exact-backoff", type=float, default=0.4)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def write_config(path: Path, args: argparse.Namespace) -> None:
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["sparse_hebbian_context"] = {
        "max_order": args.context_max_order,
        "alpha": args.context_alpha,
        "unigram_weight": args.context_unigram_weight,
        "order_weight": args.context_order_weight,
        "order_weight_growth": args.context_order_weight_growth,
        "score_mode": args.context_score_mode,
        "smoothing": args.context_smoothing,
        "backoff": args.context_backoff,
    }
    config["semantic_hebbian"] = {
        "order": args.semantic_order,
        "dim": args.semantic_dim,
        "hash_bits": args.semantic_hash_bits,
        "alpha": args.semantic_alpha,
        "bucket_weight": args.semantic_bucket_weight,
        "unigram_weight": args.semantic_unigram_weight,
        "seed": args.seed + 101,
    }
    config["continuation_backoff"] = {
        "max_order": args.continuation_max_order,
        "discount": args.continuation_discount,
        "unigram_weight": args.continuation_unigram_weight,
        "continuation_weight": args.continuation_weight,
        "exact_weight": args.continuation_exact_weight,
        "exact_backoff": args.continuation_exact_backoff,
    }
    config["stream_pruning"] = {
        "prune_every_segments": args.prune_every_segments,
        "prune_min_count": args.prune_min_count,
        "prune_decay": args.prune_decay,
        "prune_max_contexts": args.prune_max_contexts,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    train_text = base.read_prefix(args.train_file, args.train_chars)
    valid_text = base.read_prefix(args.valid_file, args.valid_chars)
    train_raw = base.encode_text(tokenizer, train_text)
    valid_raw = base.encode_text(tokenizer, valid_text)
    kept_raw, train_ids, valid_ids = base.build_compact_vocab(train_raw, valid_raw, args.max_vocab)
    args.max_vocab = int(len(kept_raw))

    if args.warmup_token_limit > 0:
        train_ids = train_ids[: args.warmup_token_limit]
    if args.stream_token_limit > 0:
        valid_ids = valid_ids[: args.stream_token_limit]
    if len(train_ids) <= 1 or len(valid_ids) <= 1:
        raise ValueError("Not enough compact-vocab tokens for streaming evaluation.")

    show_progress = not args.no_progress
    print(f"tokenizer: {args.tokenizer_dir}")
    print(f"raw tokens: train={len(train_raw):,}, valid={len(valid_raw):,}")
    print(f"compact vocab: {args.max_vocab}")
    print(f"compact tokens: train={len(train_ids):,}, valid={len(valid_ids):,}")
    print(f"segment tokens: {args.segment_tokens}, stride={max(args.segment_tokens - 1, 1)}")

    sparse_cfg = base.SparseHebbianContextConfig(
        max_order=args.context_max_order,
        alpha=args.context_alpha,
        unigram_weight=args.context_unigram_weight,
        order_weight=args.context_order_weight,
        order_weight_growth=args.context_order_weight_growth,
        score_mode=args.context_score_mode,
        smoothing=args.context_smoothing,
        backoff=args.context_backoff,
    )
    semantic_cfg = base.SemanticHebbianConfig(
        order=args.semantic_order,
        dim=args.semantic_dim,
        hash_bits=args.semantic_hash_bits,
        alpha=args.semantic_alpha,
        bucket_weight=args.semantic_bucket_weight,
        unigram_weight=args.semantic_unigram_weight,
        seed=args.seed + 101,
    )
    continuation_cfg = base.ContinuationBackoffConfig(
        max_order=args.continuation_max_order,
        discount=args.continuation_discount,
        unigram_weight=args.continuation_unigram_weight,
        continuation_weight=args.continuation_weight,
        exact_weight=args.continuation_exact_weight,
        exact_backoff=args.continuation_exact_backoff,
    )
    decode_cfg = base.DecodingConfig(
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram=args.no_repeat_ngram,
        top_k=args.decode_top_k,
        temperature=args.decode_temperature,
    )

    method_builders = {
        "sparse_hebbian_context": lambda: base.SparseHebbianContextMemory(args.max_vocab, sparse_cfg),
        "combined_context": lambda: CombinedContextMemory(args.max_vocab, sparse_cfg, semantic_cfg, args.semantic_combine_weight),
        "continuation_backoff": lambda: base.ContinuationBackoffMemory(args.max_vocab, continuation_cfg),
    }

    warmup_summaries: dict[str, dict] = {}
    stream_pre_summaries: dict[str, dict] = {}
    stream_online_summaries: dict[str, dict] = {}
    stream_post_summaries: dict[str, dict] = {}
    retention_before_summaries: dict[str, dict] = {}
    retention_after_summaries: dict[str, dict] = {}
    final_histories: dict[str, list[int]] = {}
    stream_rows: list[dict] = []
    method_states: dict[str, object] = {}

    for method_name, builder in method_builders.items():
        method_states[method_name] = builder()

    warmup_rows: list[dict] = []
    warmup_final_histories: dict[str, list[int]] = {}
    warmup_seconds: dict[str, float] = {}
    for method_name, memory in method_states.items():
        with base.Timer() as warmup_timer:
            summary, history, rows = run_stream_phase(
                method_name,
                memory,
                train_ids,
                args.segment_tokens,
                [],
                args.temperature,
                update=True,
                pass_name="warmup_online",
                show_progress=show_progress,
                prune_every_segments=args.prune_every_segments,
                prune_min_count=args.prune_min_count,
                prune_decay=args.prune_decay,
                prune_max_contexts=args.prune_max_contexts,
            )
        summary["seconds"] = warmup_timer.elapsed
        warmup_summaries[method_name] = summary
        warmup_final_histories[method_name] = history
        warmup_seconds[method_name] = warmup_timer.elapsed
        warmup_rows.extend(rows)
    stream_rows.extend(warmup_rows)
    retention_ids = train_ids[: min(len(train_ids), args.retention_token_limit)]

    # Stream the validation prefix with the warmed-up memories.
    for method_name, memory in method_states.items():
        start_history = warmup_final_histories[method_name]
        retention_before_summaries[method_name] = (
            evaluate_memory_sequence(memory, retention_ids, args.temperature)
            if len(retention_ids) > 1 and args.retention_token_limit > 0
            else {"loss": float("nan"), "accuracy": 0.0, "target_tokens": 0}
        )
        with base.Timer() as pre_timer:
            pre_summary, _, pre_rows = run_stream_phase(
                method_name,
                memory,
                valid_ids,
                args.segment_tokens,
                start_history,
                args.temperature,
                update=False,
                pass_name="stream_pre",
                show_progress=show_progress,
            )
        with base.Timer() as online_timer:
            online_summary, final_history, online_rows = run_stream_phase(
                method_name,
                memory,
                valid_ids,
                args.segment_tokens,
                start_history,
                args.temperature,
                update=True,
                pass_name="stream_online",
                show_progress=show_progress,
                prune_every_segments=args.prune_every_segments,
                prune_min_count=args.prune_min_count,
                prune_decay=args.prune_decay,
                prune_max_contexts=args.prune_max_contexts,
            )
        with base.Timer() as post_timer:
            post_summary, _, post_rows = run_stream_phase(
                method_name,
                memory,
                valid_ids,
                args.segment_tokens,
                start_history,
                args.temperature,
                update=False,
                pass_name="stream_post",
                show_progress=show_progress,
            )

        stream_pre_summaries[method_name] = pre_summary
        stream_online_summaries[method_name] = online_summary
        stream_post_summaries[method_name] = post_summary
        retention_after_summaries[method_name] = (
            evaluate_memory_sequence(memory, retention_ids, args.temperature)
            if len(retention_ids) > 1 and args.retention_token_limit > 0
            else {"loss": float("nan"), "accuracy": 0.0, "target_tokens": 0}
        )
        final_histories[method_name] = final_history
        stream_rows.extend(pre_rows)
        stream_rows.extend(online_rows)
        stream_rows.extend(post_rows)

        pre_summary["seconds"] = pre_timer.elapsed
        online_summary["seconds"] = online_timer.elapsed
        post_summary["seconds"] = post_timer.elapsed

    rows: list[dict] = []
    for method_name, memory in method_states.items():
        warmup_summary = warmup_summaries[method_name]
        pre_summary = stream_pre_summaries[method_name]
        online_summary = stream_online_summaries[method_name]
        post_summary = stream_post_summaries[method_name]
        retention_before = retention_before_summaries[method_name]
        retention_after = retention_after_summaries[method_name]
        state_bytes = memory_state_bytes(memory)
        rows.append(
            {
                "method": method_name,
                "warmup_loss": warmup_summary["loss"],
                "warmup_acc": warmup_summary["accuracy"],
                "stream_pre_loss": pre_summary["loss"],
                "stream_pre_acc": pre_summary["accuracy"],
                "stream_online_loss": online_summary["loss"],
                "stream_online_acc": online_summary["accuracy"],
                "stream_post_loss": post_summary["loss"],
                "stream_post_acc": post_summary["accuracy"],
                "stream_delta_loss": pre_summary["loss"] - post_summary["loss"],
                "online_to_post_delta": online_summary["loss"] - post_summary["loss"],
                "retention_before_loss": retention_before["loss"],
                "retention_after_loss": retention_after["loss"],
                "retention_delta_loss": retention_after["loss"] - retention_before["loss"],
                "retention_before_acc": retention_before["accuracy"],
                "retention_after_acc": retention_after["accuracy"],
                "retention_tokens": retention_after["target_tokens"],
                "warmup_targets": warmup_summary["target_tokens"],
                "stream_targets": post_summary["target_tokens"],
                "stream_pre_seconds": pre_summary["seconds"],
                "stream_online_seconds": online_summary["seconds"],
                "stream_post_seconds": post_summary["seconds"],
                "active_contexts": memory.active_contexts() if hasattr(memory, "active_contexts") else 0,
                "state_bytes": state_bytes,
                "bytes_per_target": state_bytes / max(warmup_summary["target_tokens"] + post_summary["target_tokens"], 1),
                "final_history_tokens": len(final_histories[method_name]),
                "warmup_seconds": warmup_seconds[method_name],
                "prune_every_segments": args.prune_every_segments,
                "prune_min_count": args.prune_min_count,
                "prune_decay": args.prune_decay,
                "prune_max_contexts": args.prune_max_contexts,
            }
        )

    base.write_csv(args.out_dir / "summary.csv", rows)
    base.write_csv(args.out_dir / "segment_metrics.csv", stream_rows)
    plot_stream_curves(stream_rows, args.out_dir / "segment_curve.png")
    write_config(args.out_dir / "config.json", args)
    write_json(args.out_dir / "summary.json", {"rows": rows})

    prompts = (
        base.read_prompt_file(args.prompt_file)
        if args.prompt_file is not None
        else base.build_heldout_prompts(train_text, valid_text, tokenizer, kept_raw, count=3)
    )
    if not prompts:
        prompts = list(base.FALLBACK_SAMPLE_PROMPTS[:3])
    prompt_checks = [base.prompt_status(tokenizer, kept_raw, prompt, train_text) for prompt in prompts]
    comparison_path = args.out_dir / "greedy_completions.txt"
    generation_rows: list[dict] = []
    with comparison_path.open("w", encoding="utf-8") as f:
        for prompt_idx, (prompt, check) in enumerate(zip(prompts, prompt_checks)):
            f.write(
                f"PROMPT: {prompt!r} | in_train_prefix={check['in_train_prefix']} "
                f"| compact_tokens={check['compact_tokens']}/{check['raw_tokens']}\n"
            )
            for method_name, memory in method_states.items():
                text = sample_context_memory(
                    memory,
                    tokenizer,
                    kept_raw,
                    prompt,
                    args.sample_len,
                    decode_cfg,
                    args.seed + 100 + prompt_idx,
                )
                metric_row = base.generation_quality_metrics(method_name, prompt, text)
                metric_row["prompt_index"] = prompt_idx
                generation_rows.append(metric_row)
                f.write(f"\n{method_name} greedy:\n{text}\n")
            f.write("\n" + "=" * 80 + "\n\n")
    base.write_csv(args.out_dir / "generation_metrics.csv", generation_rows)
    write_json(
        args.out_dir / "run_report.json",
        {
            "rows": rows,
            "prompts": prompt_checks,
            "methods": list(method_builders.keys()),
        },
    )

    print("\nSummary:")
    for row in rows:
        print(
            f"  {row['method']}: warmup_ce={row['warmup_loss']:.3f} "
            f"stream_pre_ce={row['stream_pre_loss']:.3f} "
            f"stream_post_ce={row['stream_post_loss']:.3f} "
            f"acc={row['stream_post_acc']:.3f} "
            f"bytes={row['state_bytes']:,}"
        )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")
    print(f"wrote segment metrics: {args.out_dir / 'segment_metrics.csv'}")
    print(f"wrote segment curve: {args.out_dir / 'segment_curve.png'}")
    print(f"wrote completions: {comparison_path}")
    print(f"wrote generation metrics: {args.out_dir / 'generation_metrics.csv'}")
    print(f"wrote config: {args.out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
