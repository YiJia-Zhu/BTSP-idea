#!/usr/bin/env python3
"""
U006 strict streaming adapter for the cloned Error-Neuron-Microcircuits model.

The reference microcircuit remains the learning core.  This file only adapts
plain token documents into the interface expected by `errormc_model`:

    current token + position -> r0 input rate
    next token -> output-layer target rate
    reference errormc_model.evolve_system(...)

Unlike U005, this does not average a context window into one static vector.
The model is stepped through the document token by token, and each token can be
held for several reference dt steps.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

import phase_binding_token_experiment as phase
from u005_reference_errormc_llm_adapter import (
    DEFAULT_GSM_TRAIN,
    DEFAULT_GSM_VALID,
    DEFAULT_TOKENIZER,
    DEFAULT_TRAIN,
    DEFAULT_VALID,
    SCRIPT_DIR,
    build_reference_params,
    init_MC,
    sinusoidal_positions,
    softmax_loss_and_pred,
    summarize,
    write_csv,
)


def split_text_documents(text: str, fallback_chars: int) -> list[str]:
    docs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if len(docs) > 1 or fallback_chars <= 0:
        return docs if docs else ([text] if text.strip() else [])
    stripped = text.strip()
    return [stripped[idx : idx + fallback_chars] for idx in range(0, len(stripped), fallback_chars)]


def gsm_records_to_docs(path: Path, max_items: int) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("question", [])
    cots = data.get("cot", [])
    answers = data.get("answer", [])
    count = min(max(int(max_items), 0), len(questions), len(cots), len(answers))
    docs = []
    for idx in range(count):
        docs.append(f"Question: {questions[idx]}\nReasoning: {cots[idx]}\nAnswer: {answers[idx]}\n")
    return docs


def interleave_doc_groups(groups: list[list[str]]) -> list[str]:
    out: list[str] = []
    max_len = max((len(group) for group in groups), default=0)
    for idx in range(max_len):
        for group in groups:
            if idx < len(group):
                out.append(group[idx])
    return out


def load_text_documents(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    train_groups: list[list[str]] = []
    valid_groups: list[list[str]] = []
    if args.task in {"tinystories", "mix"}:
        train_groups.append(
            split_text_documents(phase.read_prefix(args.train_file, args.train_chars), args.doc_chars)
        )
        valid_groups.append(
            split_text_documents(phase.read_prefix(args.valid_file, args.valid_chars), args.doc_chars)
        )
    if args.task in {"gsm8k", "mix"}:
        train_groups.append(gsm_records_to_docs(args.gsm_train_file, args.gsm_train_items))
        valid_groups.append(gsm_records_to_docs(args.gsm_valid_file, args.gsm_valid_items))
    return interleave_doc_groups(train_groups), interleave_doc_groups(valid_groups)


def tokenize_documents(tokenizer: Any, docs: list[str]) -> list[np.ndarray]:
    arrays = [phase.encode_text(tokenizer, doc) for doc in docs if doc.strip()]
    return [arr for arr in arrays if arr.size >= 2]


def build_compact_document_vocab(
    train_raw_docs: list[np.ndarray],
    valid_raw_docs: list[np.ndarray],
    max_vocab: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int, np.ndarray]:
    if not train_raw_docs:
        raise ValueError("no training documents after tokenization")
    train_raw = np.concatenate(train_raw_docs)
    if train_raw.size == 0:
        raise ValueError("empty training token stream")
    counts = np.bincount(train_raw)
    kept_count = min(max(int(max_vocab), 1), counts.shape[0])
    kept_raw = np.argsort(-counts)[:kept_count]
    kept_raw = kept_raw[np.argsort(kept_raw)]
    max_seen = int(train_raw.max())
    if valid_raw_docs:
        valid_max = max((int(doc.max()) for doc in valid_raw_docs if doc.size), default=0)
        max_seen = max(max_seen, valid_max)
    raw_to_compact = np.full(max(max_seen + 1, int(kept_raw.max()) + 1), -1, dtype=np.int64)
    raw_to_compact[kept_raw] = np.arange(len(kept_raw), dtype=np.int64)

    def map_docs(raw_docs: list[np.ndarray]) -> list[np.ndarray]:
        mapped_docs: list[np.ndarray] = []
        for doc in raw_docs:
            compact = raw_to_compact[doc]
            compact = compact[compact >= 0].astype(np.int64)
            if compact.size >= 2:
                mapped_docs.append(compact)
        return mapped_docs

    return map_docs(train_raw_docs), map_docs(valid_raw_docs), int(len(kept_raw)), kept_raw.astype(np.int64)


def load_document_ids(args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray], int, np.ndarray, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    train_text_docs, valid_text_docs = load_text_documents(args)
    train_raw_docs = tokenize_documents(tokenizer, train_text_docs)
    valid_raw_docs = tokenize_documents(tokenizer, valid_text_docs)
    train_docs, valid_docs, vocab_size, kept_raw = build_compact_document_vocab(
        train_raw_docs, valid_raw_docs, args.max_vocab
    )
    return train_docs, valid_docs, vocab_size, kept_raw, tokenizer


class ReferenceStreamErrorMCLLMAdapter:
    def __init__(self, vocab_size: int, args: argparse.Namespace) -> None:
        self.vocab_size = int(vocab_size)
        self.args = args
        self.context_len = max(int(args.context_len), 1)
        self.d_model = max(int(args.d_model), 1)
        self.presentation_steps = (
            max(int(args.presentation_steps), 1)
            if args.presentation_steps > 0
            else max(int(round(args.tpres / args.dt)), 1)
        )
        self.predict_steps = max(int(args.predict_steps), 0)
        self.rng = np.random.default_rng(args.seed + 601)
        self.token_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (self.vocab_size, self.d_model)).astype(np.float32)
        )
        self.position_codes = sinusoidal_positions(self.context_len, self.d_model)
        self.params = build_reference_params(args, self.vocab_size)
        self.mc = init_MC(self.params, [args.seed], teacher=False)[0]
        self.output_counts = np.ones(self.vocab_size, dtype=np.float32)
        self.output_bias = np.zeros(self.vocab_size, dtype=np.float32)
        self._refresh_output_bias()
        self.reset_voltages()

    def _refresh_output_bias(self) -> None:
        if self.args.output_bias_mode == "unigram":
            probs = self.output_counts / float(np.sum(self.output_counts))
            self.output_bias = np.log(np.maximum(probs, 1e-9)).astype(np.float32)
        else:
            self.output_bias = np.zeros(self.vocab_size, dtype=np.float32)

    def encode_token(self, token: int, position: int) -> np.ndarray:
        token_code = self.token_codes[int(token)]
        pos_code = self.position_codes[int(position) % self.context_len]
        return phase.normalize_vector((token_code + argsafe_position_scale(self.args) * pos_code).astype(np.float32))

    def target_vector(self, target: int) -> np.ndarray:
        target = int(target)
        if self.args.target_mode == "onehot":
            vec = np.zeros(self.vocab_size, dtype=np.float32)
            vec[target] = 1.0
            return vec
        if self.args.target_mode == "centered":
            if self.vocab_size <= 1:
                return np.zeros(self.vocab_size, dtype=np.float32)
            vec = np.full(self.vocab_size, -1.0 / float(self.vocab_size - 1), dtype=np.float32)
            vec[target] = 1.0
            return vec
        if self.args.target_mode == "smoothed":
            eps = min(max(float(self.args.label_smoothing), 0.0), 1.0)
            if self.vocab_size <= 1:
                return np.ones(self.vocab_size, dtype=np.float32)
            vec = np.full(self.vocab_size, eps / float(self.vocab_size - 1), dtype=np.float32)
            vec[target] = 1.0 - eps
            return vec.astype(np.float32)
        raise ValueError(f"unknown target_mode: {self.args.target_mode}")

    def read_output(self) -> np.ndarray:
        if self.args.readout_mode == "rate":
            out = self.mc.rP_breve[-1]
        elif self.args.readout_mode == "voltage":
            out = self.mc.uP_breve[-1]
        else:
            raise ValueError(f"unknown readout_mode: {self.args.readout_mode}")
        return (self.args.logit_scale * out + self.output_bias).astype(np.float32)

    def evolve_current_token(
        self,
        token: int,
        position: int,
        steps: int,
        target: int | None,
        learn: bool,
    ) -> None:
        if steps <= 0:
            return
        r0 = self.encode_token(int(token), int(position))
        u_tgt = [self.target_vector(int(target))] if target is not None else None
        for _ in range(int(steps)):
            self.mc.evolve_system(
                r0=r0,
                u_tgt=u_tgt,
                learn_weights=learn,
                learn_lat_weights=learn and self.args.learn_lat_weights,
                learn_bw_weights=learn and self.args.learn_bw_weights,
                record=False,
            )

    def predict_next(self, token: int, position: int) -> np.ndarray:
        self.evolve_current_token(token, position, self.predict_steps, target=None, learn=False)
        return self.read_output()

    def observe(self, token: int, target: int, position: int, learn: bool = True) -> None:
        self.evolve_current_token(token, position, self.presentation_steps, target=target, learn=learn)
        if learn and self.args.output_bias_mode == "unigram":
            self.output_counts[int(target)] += 1.0
            self._refresh_output_bias()

    def reset_voltages(self) -> None:
        for name in ("uP", "uP_old", "uP_breve", "uI", "uI_old", "uI_breve"):
            if hasattr(self.mc, name):
                values = getattr(self.mc, name)
                setattr(self.mc, name, [np.zeros_like(value) for value in values])
        for name in ("vbas", "vden", "vapi", "vapi_old", "vapi_noise", "noise", "dWPP_LO"):
            if hasattr(self.mc, name):
                values = getattr(self.mc, name)
                setattr(self.mc, name, [np.zeros_like(value) for value in values])
        self.mc.rP_breve = [self.mc.activation[i](self.mc.uP_breve[i]) for i in range(len(self.mc.uP_breve))]
        self.mc.rP_breve_old = [np.zeros_like(value) for value in self.mc.rP_breve]
        self.mc.rI_breve = [self.mc.error_activation[i](self.mc.uI_breve[i]) for i in range(len(self.mc.uI_breve))]
        self.mc.rI_breve_old = [np.zeros_like(value) for value in self.mc.rI_breve]
        self.mc.d_rP_breve = [self.mc.d_activation[i](self.mc.uP_breve[i]) for i in range(len(self.mc.uP_breve))]
        self.mc.r0 = np.zeros(self.d_model, dtype=np.float32)
        self.mc.r0_old = np.zeros(self.d_model, dtype=np.float32)
        if hasattr(self.mc, "rP_breve_HI"):
            self.mc.rP_breve_HI = [np.zeros_like(value) for value in self.mc.rP_breve]
        if hasattr(self.mc, "dWPP_post_LO_old"):
            self.mc.dWPP_post_LO_old = [np.zeros_like(value) for value in self.mc.dWPP_post_LO_old]
        if hasattr(self.mc, "r0_LO_old"):
            self.mc.r0_LO_old = np.zeros_like(self.mc.r0_LO_old)
        if hasattr(self.mc, "r_LO_old"):
            self.mc.r_LO_old = [np.zeros_like(value) for value in self.mc.r_LO_old]
        self.mc.Time = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "token_codes": self.token_codes.copy(),
            "position_codes": self.position_codes.copy(),
            "output_counts": self.output_counts.copy(),
            "output_bias": self.output_bias.copy(),
            "WPP": [w.copy() for w in self.mc.WPP],
            "WIP": [w.copy() for w in self.mc.WIP],
            "BPI": [w.copy() for w in self.mc.BPI],
            "BII": [w.copy() for w in self.mc.BII],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.token_codes = state["token_codes"].copy()
        self.position_codes = state["position_codes"].copy()
        self.output_counts = state["output_counts"].copy()
        self.output_bias = state["output_bias"].copy()
        self.mc.WPP = [w.copy() for w in state["WPP"]]
        self.mc.WIP = [w.copy() for w in state["WIP"]]
        self.mc.BPI = [w.copy() for w in state["BPI"]]
        self.mc.BII = [w.copy() for w in state["BII"]]

    def state_bytes(self) -> int:
        arrays: list[np.ndarray] = [self.token_codes, self.position_codes, self.output_bias, self.output_counts]
        arrays.extend(self.mc.WPP)
        arrays.extend(self.mc.WIP)
        arrays.extend(self.mc.BPI)
        arrays.extend(self.mc.BII)
        arrays.extend(self.mc.uP)
        arrays.extend(self.mc.uI)
        return int(sum(array.nbytes for array in arrays))

    def parameter_count(self) -> int:
        arrays: list[np.ndarray] = [self.token_codes, self.position_codes, self.output_bias, self.output_counts]
        arrays.extend(self.mc.WPP)
        arrays.extend(self.mc.WIP)
        arrays.extend(self.mc.BPI)
        arrays.extend(self.mc.BII)
        return int(sum(array.size for array in arrays))


def argsafe_position_scale(args: argparse.Namespace) -> float:
    return float(getattr(args, "position_scale", 1.0))


def run_documents(
    model: ReferenceStreamErrorMCLLMAdapter,
    docs: list[np.ndarray],
    update: bool,
    limit: int,
) -> dict[str, float | int]:
    loss_sum = 0.0
    correct = 0
    total = 0
    for doc in docs:
        if model.args.reset_doc_state:
            model.reset_voltages()
        for pos in range(len(doc) - 1):
            token = int(doc[pos])
            target = int(doc[pos + 1])
            logits = model.predict_next(token, pos)
            loss, pred = softmax_loss_and_pred(logits, target, model.args.temperature)
            loss_sum += loss
            correct += int(pred == target)
            total += 1
            if update:
                model.observe(token, target, pos, learn=True)
            if limit > 0 and total >= limit:
                return summarize(loss_sum, correct, total)
    return summarize(loss_sum, correct, total)


def count_document_pairs(docs: list[np.ndarray]) -> int:
    return int(sum(max(len(doc) - 1, 0) for doc in docs))


def decode_compact_tokens(tokenizer: Any, kept_raw: np.ndarray, compact_ids: list[int] | np.ndarray) -> str:
    raw_ids = [int(kept_raw[int(token)]) for token in compact_ids if 0 <= int(token) < len(kept_raw)]
    if not raw_ids:
        return ""
    return tokenizer.decode(raw_ids, skip_special_tokens=False)


def generate_compact_tokens(
    model: ReferenceStreamErrorMCLLMAdapter,
    prompt: np.ndarray,
    max_new_tokens: int,
) -> list[int]:
    prompt_ids = [int(token) for token in prompt]
    if not prompt_ids:
        return []
    model.reset_voltages()
    for pos, token in enumerate(prompt_ids[:-1]):
        model.evolve_current_token(token, pos, model.predict_steps, target=None, learn=False)
    out = prompt_ids.copy()
    current = prompt_ids[-1]
    for _ in range(max(int(max_new_tokens), 0)):
        logits = model.predict_next(current, len(out) - 1)
        current = int(np.argmax(logits))
        out.append(current)
    model.reset_voltages()
    return out


def generate_samples(
    model: ReferenceStreamErrorMCLLMAdapter,
    docs: list[np.ndarray],
    tokenizer: Any,
    kept_raw: np.ndarray,
    sample_count: int,
    prompt_tokens: int,
    new_tokens: int,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    usable_docs = [doc for doc in docs if len(doc) > max(int(prompt_tokens), 1)]
    for idx, doc in enumerate(usable_docs[: max(int(sample_count), 0)]):
        prompt = doc[: max(int(prompt_tokens), 1)]
        generated = generate_compact_tokens(model, prompt, new_tokens)
        rows.append(
            {
                "sample": int(idx),
                "prompt_tokens": int(len(prompt)),
                "generated_tokens": int(max(len(generated) - len(prompt), 0)),
                "prompt": decode_compact_tokens(tokenizer, kept_raw, prompt),
                "completion": decode_compact_tokens(tokenizer, kept_raw, generated[len(prompt) :]),
                "full_text": decode_compact_tokens(tokenizer, kept_raw, generated),
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
            f.write("Completion:\n")
            f.write(str(row["completion"]).strip() + "\n")
            f.write("Full:\n")
            f.write(str(row["full_text"]).strip() + "\n\n")


def estimate_reference_dense_params(vocab_size: int, d_model: int, blocks: int, bw_connection_mode: str) -> int:
    vocab = max(int(vocab_size), 1)
    d = max(int(d_model), 1)
    b = max(int(blocks), 1)
    dims = [d for _ in range(b)] + [vocab]
    wpp = b * d * d + vocab * d
    wip_bpi = 2 * sum(dim * dim for dim in dims)
    bii = sum(lower * upper for lower, upper in zip(dims[:-1], dims[1:]))
    if bw_connection_mode == "skip":
        bii = 0
        error_dims = [0] + dims
        for idx, lower in enumerate(error_dims):
            for upper in error_dims[idx + 1 :]:
                bii += lower * upper
    token_codes = vocab * d + d * 4096
    output_bridge = 2 * vocab
    return int(wpp + wip_bpi + bii + token_codes + output_bridge)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tinystories", "gsm8k", "mix"], default="tinystories")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--gsm-train-file", type=Path, default=DEFAULT_GSM_TRAIN)
    parser.add_argument("--gsm-valid-file", type=Path, default=DEFAULT_GSM_VALID)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "u006_reference_stream_errormc_llm")
    parser.add_argument("--train-chars", type=int, default=12_000)
    parser.add_argument("--valid-chars", type=int, default=3_000)
    parser.add_argument("--doc-chars", type=int, default=1_000)
    parser.add_argument("--gsm-train-items", type=int, default=64)
    parser.add_argument("--gsm-valid-items", type=int, default=32)
    parser.add_argument("--max-vocab", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--model-type", choices=["FA", "BP"], default="FA")
    parser.add_argument("--bw-connection-mode", choices=["layered", "skip"], default="layered")
    parser.add_argument("--target-mode", choices=["onehot", "centered", "smoothed"], default="smoothed")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--readout-mode", choices=["rate", "voltage"], default="rate")
    parser.add_argument("--output-bias-mode", choices=["none", "unigram"], default="none")
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--eta-fw", type=float, default=0.02)
    parser.add_argument("--eta-output", type=float, default=0.05)
    parser.add_argument("--eta-ip", type=float, default=0.0)
    parser.add_argument("--eta-pi", type=float, default=0.0)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--tpres", type=float, default=0.2)
    parser.add_argument("--presentation-steps", type=int, default=0)
    parser.add_argument("--predict-steps", type=int, default=1)
    parser.add_argument("--settling-time", type=float, default=0.0)
    parser.add_argument("--logit-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--wt-noise", type=float, default=0.0)
    parser.add_argument("--dWPP-use-activation", action="store_true")
    parser.add_argument("--no-varphi-transfer", action="store_true")
    parser.add_argument("--learn-lat-weights", action="store_true")
    parser.add_argument("--learn-bw-weights", action="store_true")
    parser.add_argument("--no-reset-doc-state", dest="reset_doc_state", action="store_false")
    parser.set_defaults(reset_doc_state=True)
    parser.add_argument("--eval-token-limit", type=int, default=600)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--sample-prompt-tokens", type=int, default=16)
    parser.add_argument("--sample-new-tokens", type=int, default=48)
    parser.add_argument("--large-vocab-size", type=int, default=50_000)
    parser.add_argument("--large-d-model", type=int, default=1536)
    parser.add_argument("--large-blocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_docs, valid_docs, vocab_size, kept_raw, tokenizer = load_document_ids(args)
    model = ReferenceStreamErrorMCLLMAdapter(vocab_size, args)
    start = time.perf_counter()
    train_summary = run_documents(model, train_docs, update=True, limit=args.eval_token_limit)
    train_seconds = time.perf_counter() - start
    samples = generate_samples(
        model,
        valid_docs,
        tokenizer,
        kept_raw,
        args.sample_count,
        args.sample_prompt_tokens,
        args.sample_new_tokens,
    )
    write_samples(args.out_dir / "greedy_samples.txt", samples)
    valid_pre = run_documents(model, valid_docs, update=False, limit=args.eval_token_limit)
    valid_online = run_documents(model, valid_docs, update=True, limit=args.eval_token_limit)
    valid_post = run_documents(model, valid_docs, update=False, limit=args.eval_token_limit)
    large_estimate = estimate_reference_dense_params(
        args.large_vocab_size,
        args.large_d_model,
        args.large_blocks,
        args.bw_connection_mode,
    )
    summary = {
        "task": args.task,
        "seed": int(args.seed),
        "model_type": args.model_type,
        "bw_connection_mode": args.bw_connection_mode,
        "target_mode": args.target_mode,
        "readout_mode": args.readout_mode,
        "output_bias_mode": args.output_bias_mode,
        "train_docs": int(len(train_docs)),
        "valid_docs": int(len(valid_docs)),
        "train_available_tokens": int(count_document_pairs(train_docs)),
        "valid_available_tokens": int(count_document_pairs(valid_docs)),
        "vocab_size": int(vocab_size),
        "context_len": int(args.context_len),
        "d_model": int(args.d_model),
        "blocks": int(args.blocks),
        "presentation_steps": int(model.presentation_steps),
        "predict_steps": int(model.predict_steps),
        "actual_params": int(model.parameter_count()),
        "actual_state_bytes": int(model.state_bytes()),
        "large_reference_dense_param_estimate": int(large_estimate),
        "train_loss": train_summary["loss"],
        "train_acc": train_summary["accuracy"],
        "train_tokens": train_summary["tokens"],
        "valid_pre_loss": valid_pre["loss"],
        "valid_pre_acc": valid_pre["accuracy"],
        "valid_pre_tokens": valid_pre["tokens"],
        "valid_online_loss": valid_online["loss"],
        "valid_online_acc": valid_online["accuracy"],
        "valid_online_tokens": valid_online["tokens"],
        "valid_post_loss": valid_post["loss"],
        "valid_post_acc": valid_post["accuracy"],
        "valid_post_tokens": valid_post["tokens"],
        "sample_count": int(len(samples)),
        "train_seconds": float(train_seconds),
    }
    write_csv(args.out_dir / "summary.csv", [summary])
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "summary": summary,
                "samples": samples,
            },
            f,
            indent=2,
        )
    print("Summary:")
    print(
        f"  task={args.task} model={args.model_type}/{args.bw_connection_mode} "
        f"docs={len(train_docs)}/{len(valid_docs)} steps={model.presentation_steps} "
        f"params={model.parameter_count():,} valid_post={valid_post['loss']:.3f}/{valid_post['accuracy']:.3f}"
    )
    print(f"  large reference-dense estimate={large_estimate:,}")
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
