#!/usr/bin/env python3
"""
U010 random-init Llama-1B BP baseline on the same data as U009.

This is a comparison experiment only.  It uses the local Llama tokenizer and
Llama-3.2-1B config, but does not load pretrained weights.  The model is trained
from random initialization with ordinary backpropagation to check whether the
repetition/format-attractor problem also appears in a standard Llama decoder
after one complete epoch over the same small mixed corpus.
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
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

import packed_lm_data
import phase_binding_token_experiment as phase
import u009_full_vocab_local_predictive_stream as u009


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = phase.DEFAULT_TRAIN
DEFAULT_VALID = phase.DEFAULT_VALID
DEFAULT_TOKENIZER = phase.DEFAULT_TOKENIZER
DEFAULT_GSM_TRAIN = u009.DEFAULT_GSM_TRAIN
DEFAULT_GSM_VALID = u009.DEFAULT_GSM_VALID


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def count_pairs(docs: list[np.ndarray]) -> int:
    return int(sum(max(int(doc.size) - 1, 0) for doc in docs))


def make_lm_sequences(docs: list[np.ndarray], seq_len: int) -> list[np.ndarray]:
    seq_len = max(int(seq_len), 1)
    sequences: list[np.ndarray] = []
    for doc in docs:
        if doc.size < 2:
            continue
        for start in range(0, int(doc.size) - 1, seq_len):
            chunk = doc[start : min(start + seq_len + 1, int(doc.size))]
            if chunk.size >= 2:
                sequences.append(chunk.astype(np.int64, copy=False))
    return sequences


def batch_sequences(
    sequences: list[np.ndarray],
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(sequences), batch_size):
        group = sequences[start : start + batch_size]
        max_len = max(int(seq.size) - 1 for seq in group)
        input_ids = torch.full((len(group), max_len), int(pad_id), dtype=torch.long)
        labels = torch.full((len(group), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(group), max_len), dtype=torch.long)
        for row, seq in enumerate(group):
            n = int(seq.size) - 1
            input_ids[row, :n] = torch.from_numpy(seq[:-1].copy())
            labels[row, :n] = torch.from_numpy(seq[1:].copy())
            attention_mask[row, :n] = 1
        batches.append(
            {
                "input_ids": input_ids.to(device, non_blocking=True),
                "labels": labels.to(device, non_blocking=True),
                "attention_mask": attention_mask.to(device, non_blocking=True),
            }
        )
    return batches


def summarize(loss_sum: float, correct: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"loss": float("nan"), "ppl": float("nan"), "accuracy": 0.0, "tokens": 0}
    loss = loss_sum / float(total)
    return {
        "loss": float(loss),
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": float(correct / float(total)),
        "tokens": int(total),
    }


def build_model_config(args: argparse.Namespace, tokenizer_len: int) -> Any:
    if args.model_scale == "llama1b":
        cfg = AutoConfig.from_pretrained(str(args.tokenizer), local_files_only=True)
        cfg.vocab_size = int(tokenizer_len)
        cfg.use_cache = False
        cfg.pad_token_id = int(args.pad_id)
        return cfg
    if args.model_scale == "debug":
        return LlamaConfig(
            vocab_size=int(tokenizer_len),
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=max(int(args.seq_len), 128),
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            tie_word_embeddings=True,
            pad_token_id=int(args.pad_id),
            bos_token_id=128000,
            eos_token_id=128009,
            use_cache=False,
        )
    raise ValueError(f"unknown model_scale: {args.model_scale}")


def instantiate_model(args: argparse.Namespace, tokenizer_len: int, device: torch.device) -> torch.nn.Module:
    cfg = build_model_config(args, tokenizer_len)
    if args.model_scale == "llama1b":
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = LlamaForCausalLM(cfg)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
    model.to(device=device, dtype=dtype)
    model.train()
    return model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    autocast_dtype: torch.dtype | None,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for batch in batches:
        labels = batch["labels"]
        active = labels.ne(-100)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            out = model(**batch)
        token_count = int(active.sum().item())
        loss_sum += float(out.loss.detach().float().item()) * token_count
        preds = out.logits.detach().argmax(dim=-1)
        correct += int((preds.eq(labels) & active).sum().item())
        total += token_count
    model.train()
    return summarize(loss_sum, correct, total)


def train_one_epoch(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
    autocast_dtype: torch.dtype | None,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    correct = 0
    total = 0
    chunk_loss = 0.0
    chunk_correct = 0
    chunk_total = 0
    chunks: list[dict[str, float | int]] = []
    accum = max(int(args.grad_accum), 1)
    for step, batch in enumerate(batches):
        labels = batch["labels"]
        active = labels.ne(-100)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            out = model(**batch)
            loss = out.loss / float(accum)
        loss.backward()
        if (step + 1) % accum == 0 or step + 1 == len(batches):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        token_count = int(active.sum().item())
        batch_loss = float(out.loss.detach().float().item())
        preds = out.logits.detach().argmax(dim=-1)
        hits = int((preds.eq(labels) & active).sum().item())
        loss_sum += batch_loss * token_count
        correct += hits
        total += token_count
        chunk_loss += batch_loss * token_count
        chunk_correct += hits
        chunk_total += token_count
        if args.chunk_tokens > 0 and chunk_total >= args.chunk_tokens:
            chunks.append(
                {
                    "chunk": int(len(chunks)),
                    "tokens_seen": int(total),
                    "loss": float(chunk_loss / float(chunk_total)),
                    "accuracy": float(chunk_correct / float(chunk_total)),
                    "tokens": int(chunk_total),
                }
            )
            chunk_loss = 0.0
            chunk_correct = 0
            chunk_total = 0
    if chunk_total > 0:
        chunks.append(
            {
                "chunk": int(len(chunks)),
                "tokens_seen": int(total),
                "loss": float(chunk_loss / float(chunk_total)),
                "accuracy": float(chunk_correct / float(chunk_total)),
                "tokens": int(chunk_total),
            }
        )
    return summarize(loss_sum, correct, total), chunks


@torch.no_grad()
def generate_one(
    model: torch.nn.Module,
    prompt: np.ndarray,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    pad_id: int,
    device: torch.device,
) -> list[int]:
    model.eval()
    input_ids = torch.tensor(prompt[None, :], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    kwargs: dict[str, Any] = {
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": int(pad_id),
        "use_cache": True,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
    out = model.generate(input_ids=input_ids, **kwargs)
    model.train()
    return [int(x) for x in out[0].detach().cpu().tolist()]


def generate_samples(
    model: torch.nn.Module,
    docs: list[np.ndarray],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    usable = [doc for doc in docs if int(doc.size) > max(int(args.sample_prompt_tokens), 1)]
    for idx, doc in enumerate(usable[: max(int(args.sample_count), 0)]):
        prompt = doc[: max(int(args.sample_prompt_tokens), 1)]
        greedy = generate_one(
            model,
            prompt,
            args.sample_new_tokens,
            do_sample=False,
            temperature=args.sample_temperature,
            pad_id=args.pad_id,
            device=device,
        )
        sampled = generate_one(
            model,
            prompt,
            args.sample_new_tokens,
            do_sample=True,
            temperature=args.sample_temperature,
            pad_id=args.pad_id,
            device=device,
        )
        rows.append(
            {
                "sample": int(idx),
                "prompt_tokens": int(prompt.size),
                "generated_tokens": int(args.sample_new_tokens),
                "prompt": tokenizer.decode([int(x) for x in prompt.tolist()], skip_special_tokens=False),
                "greedy_completion": tokenizer.decode(greedy[int(prompt.size) :], skip_special_tokens=False),
                "sampled_completion": tokenizer.decode(sampled[int(prompt.size) :], skip_special_tokens=False),
                "greedy_full_text": tokenizer.decode(greedy, skip_special_tokens=False),
                "sampled_full_text": tokenizer.decode(sampled, skip_special_tokens=False),
            }
        )
    return rows


def write_samples(path: Path, samples: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(f"Sample {row['sample']}\n")
            f.write(f"Prompt tokens: {row['prompt_tokens']}  Generated tokens: {row['generated_tokens']}\n")
            f.write("Prompt:\n")
            f.write(str(row["prompt"]).strip() + "\n")
            f.write("Greedy completion:\n")
            f.write(str(row["greedy_completion"]).strip() + "\n")
            f.write("Sampled completion:\n")
            f.write(str(row["sampled_completion"]).strip() + "\n")
            f.write("Greedy full:\n")
            f.write(str(row["greedy_full_text"]).strip() + "\n")
            f.write("Sampled full:\n")
            f.write(str(row["sampled_full_text"]).strip() + "\n\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tinystories", "gsm8k", "mix"], default="mix")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--gsm-train-file", type=Path, default=DEFAULT_GSM_TRAIN)
    parser.add_argument("--gsm-valid-file", type=Path, default=DEFAULT_GSM_VALID)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "u010_llama1b_bp_same_data")
    parser.add_argument("--train-chars", type=int, default=100_000)
    parser.add_argument("--valid-chars", type=int, default=12_000)
    parser.add_argument("--doc-chars", type=int, default=1_200)
    parser.add_argument("--gsm-train-items", type=int, default=128)
    parser.add_argument("--gsm-valid-items", type=int, default=32)
    parser.add_argument("--model-scale", choices=["debug", "llama1b"], default="llama1b")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--sequence-mode", choices=["packed", "per_doc"], default="packed")
    parser.add_argument("--pack-eos-id", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--chunk-tokens", type=int, default=1000)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--sample-prompt-tokens", type=int, default=20)
    parser.add_argument("--sample-new-tokens", type=int, default=64)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad-id", type=int, default=128009)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else args.pad_id)
    if int(args.pack_eos_id) < 0:
        args.pack_eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else args.pad_id)

    train_text_docs, valid_text_docs = u009.load_text_documents(args)
    train_docs = u009.tokenize_documents(tokenizer, train_text_docs)
    valid_docs = u009.tokenize_documents(tokenizer, valid_text_docs)
    if args.sequence_mode == "packed":
        train_sequences = packed_lm_data.pack_documents(train_docs, args.seq_len, args.pack_eos_id)
        valid_sequences = packed_lm_data.pack_documents(valid_docs, args.seq_len, args.pack_eos_id)
    else:
        train_sequences = make_lm_sequences(train_docs, args.seq_len)
        valid_sequences = make_lm_sequences(valid_docs, args.seq_len)
    train_batches = batch_sequences(train_sequences, args.batch_size, args.pad_id, device)
    valid_batches = batch_sequences(valid_sequences, args.batch_size, args.pad_id, device)

    model = instantiate_model(args, int(len(tokenizer)), device)
    param_count = int(sum(param.numel() for param in model.parameters()))
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" and device.type == "cuda" else (
        torch.float16 if args.dtype == "fp16" and device.type == "cuda" else None
    )

    valid_pre = evaluate(model, valid_batches, autocast_dtype)
    start = time.perf_counter()
    train_summary, train_chunks = train_one_epoch(model, train_batches, args, autocast_dtype)
    train_seconds = time.perf_counter() - start
    valid_post = evaluate(model, valid_batches, autocast_dtype)
    samples = generate_samples(model, valid_docs, tokenizer, args, device)
    write_samples(args.out_dir / "samples.txt", samples)
    write_csv(args.out_dir / "train_chunks.csv", train_chunks)

    max_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    summary = {
        "task": args.task,
        "seed": int(args.seed),
        "model_scale": args.model_scale,
        "tokenizer": str(args.tokenizer),
        "tokenizer_len": int(len(tokenizer)),
        "train_docs": int(len(train_docs)),
        "valid_docs": int(len(valid_docs)),
        "train_pairs": int(count_pairs(train_docs)),
        "valid_pairs": int(count_pairs(valid_docs)),
        "train_sequences": int(len(train_sequences)),
        "valid_sequences": int(len(valid_sequences)),
        "sequence_mode": args.sequence_mode,
        "seq_len": int(args.seq_len),
        "pack_eos_id": int(args.pack_eos_id),
        "packed_train_targets": int(packed_lm_data.count_sequence_targets(train_sequences)),
        "packed_valid_targets": int(packed_lm_data.count_sequence_targets(valid_sequences)),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "lr": float(args.lr),
        "parameters": int(param_count),
        "valid_pre_loss": valid_pre["loss"],
        "valid_pre_acc": valid_pre["accuracy"],
        "train_loss": train_summary["loss"],
        "train_acc": train_summary["accuracy"],
        "train_tokens": train_summary["tokens"],
        "valid_post_loss": valid_post["loss"],
        "valid_post_acc": valid_post["accuracy"],
        "train_seconds": float(train_seconds),
        "tokens_per_second": float(train_summary["tokens"] / max(train_seconds, 1e-9)),
        "max_cuda_mem_bytes": int(max_mem),
    }
    write_csv(args.out_dir / "summary.csv", [summary])
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "summary": summary,
                "train_chunks": train_chunks,
                "samples": samples,
            },
            f,
            indent=2,
        )

    first_chunk = train_chunks[0]["loss"] if train_chunks else float("nan")
    last_chunk = train_chunks[-1]["loss"] if train_chunks else float("nan")
    print("Summary:")
    print(
        f"  scale={args.model_scale} params={param_count:,} train_pairs={count_pairs(train_docs)} "
        f"valid_pairs={count_pairs(valid_docs)}"
    )
    print(
        f"  train={train_summary['loss']:.4f}/{train_summary['accuracy']:.4f} "
        f"chunks={first_chunk:.4f}->{last_chunk:.4f} speed={summary['tokens_per_second']:.2f} tok/s"
    )
    print(f"  valid={valid_pre['loss']:.4f}->{valid_post['loss']:.4f}/{valid_post['accuracy']:.4f}")
    print(f"  max_cuda_mem={max_mem / (1024 ** 3):.2f} GiB")
    print(f"  samples: {args.out_dir / 'samples.txt'}")
    print(f"  summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
