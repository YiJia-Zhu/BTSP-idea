#!/usr/bin/env python3
"""
U005 direct adapter around the cloned Error-Neuron-Microcircuits implementation.

This script imports the reference `errormc_model` through `init_MC` and uses its
own `evolve_system`, `calc_dendritic_updates`, and `evolve_synapses` methods.
Only the LLM data adapter is local: context tokens are encoded into the model's
input vector `r0`, and the next token is encoded as a one-hot rate target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

import phase_binding_token_experiment as phase


SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPT_DIR / "Error-Neuron-Microcircuits" / "numpy_model"
DEFAULT_TRAIN = phase.DEFAULT_TRAIN
DEFAULT_VALID = phase.DEFAULT_VALID
DEFAULT_TOKENIZER = phase.DEFAULT_TOKENIZER
DEFAULT_GSM_TRAIN = SCRIPT_DIR / "data" / "GSM8k-Aug" / "gsm8k_aug_train.json"
DEFAULT_GSM_VALID = SCRIPT_DIR / "data" / "GSM8k-Aug" / "gsm8k_test.json"

sys.path.insert(0, str(REFERENCE_DIR.resolve()))
from src.init_MC import init_MC  # noqa: E402


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


def summarize(loss_sum: float, correct: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"loss": float("nan"), "ppl": float("nan"), "accuracy": 0.0, "tokens": 0}
    loss = loss_sum / total
    return {
        "loss": float(loss),
        "ppl": float(math.exp(min(loss, 20.0))),
        "accuracy": correct / total,
        "tokens": int(total),
    }


def softmax_loss_and_pred(logits: np.ndarray, target: int, temperature: float) -> tuple[float, int]:
    probs = phase.softmax(logits.astype(np.float32), temperature)
    return -math.log(float(probs[int(target)]) + 1e-9), int(np.argmax(probs))


def sinusoidal_positions(context_len: int, dim: int) -> np.ndarray:
    positions = np.arange(max(int(context_len), 1), dtype=np.float32)[:, None]
    half = max(int(dim) // 2, 1)
    div = np.exp(np.arange(half, dtype=np.float32) * (-math.log(10000.0) / max(half, 1)))
    angles = positions * div[None, :]
    codes = np.zeros((context_len, dim), dtype=np.float32)
    codes[:, 0 : 2 * half : 2] = np.sin(angles[:, : codes[:, 0::2].shape[1]])
    codes[:, 1 : 2 * half : 2] = np.cos(angles[:, : codes[:, 1::2].shape[1]])
    return phase.normalize_rows(codes)


def gsm_records_to_text(path: Path, max_items: int) -> str:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("question", [])
    cots = data.get("cot", [])
    answers = data.get("answer", [])
    count = min(max(int(max_items), 0), len(questions), len(cots), len(answers))
    docs = []
    for idx in range(count):
        docs.append(f"Question: {questions[idx]}\nReasoning: {cots[idx]}\nAnswer: {answers[idx]}\n")
    return "\n".join(docs)


def load_ids(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int]:
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    train_text = ""
    valid_text = ""
    if args.task in {"tinystories", "mix"}:
        train_text += phase.read_prefix(args.train_file, args.train_chars)
        valid_text += phase.read_prefix(args.valid_file, args.valid_chars)
    if args.task in {"gsm8k", "mix"}:
        train_text += "\n" + gsm_records_to_text(args.gsm_train_file, args.gsm_train_items)
        valid_text += "\n" + gsm_records_to_text(args.gsm_valid_file, args.gsm_valid_items)
    train_raw = phase.encode_text(tokenizer, train_text)
    valid_raw = phase.encode_text(tokenizer, valid_text)
    _, train_ids, valid_ids = phase.build_compact_vocab(train_raw, valid_raw, args.max_vocab)
    vocab_size = int(min(args.max_vocab, len(set(train_ids.tolist()) | set(valid_ids.tolist()))))
    return train_ids, valid_ids, vocab_size


def build_reference_params(args: argparse.Namespace, vocab_size: int) -> dict[str, Any]:
    layers = [args.d_model] + [args.d_model for _ in range(args.blocks)] + [vocab_size]
    eta_fw: list[float] | list[list[float]] = [args.eta_fw for _ in range(len(layers) - 2)] + [args.eta_output]
    return {
        "mc_model": "errormc",
        "teacher_mc_model": "ann",
        "dt": args.dt,
        "Tpres": args.tpres,
        "dtxi": args.dt,
        "tauHP": 1e-1,
        "tauLO": 1e2,
        "tauxi": 1e-1,
        "dataset_size": 1,
        "epochs": 1,
        "settling_time": args.settling_time,
        "layers": layers,
        "error_layers": [0] + layers[1:],
        "activation": ["tanh" for _ in range(args.blocks)] + ["linear"],
        "error_activation": "linear",
        "model_type": args.model_type,
        "WT_noise": args.wt_noise,
        "noise_type": "OU",
        "noise_scale": [0.0],
        "noise_deg": 30,
        "noise_mode": "const",
        "gl": 0.03,
        "gbas": 0.1,
        "gapi": 0.06,
        "gden": 0.1,
        "gnI": 0.06,
        "gntgt": 0.06,
        "eta_fw": eta_fw,
        "eta_bw": [],
        "eta_IP": [args.eta_ip for _ in range(len(layers) - 1)],
        "eta_PI": [args.eta_pi for _ in range(len(layers) - 1)],
        "eta_fw_conversion": "fill_diag",
        "target_rate": True,
        "init_WIP_identity": True,
        "init_BPI_identity": True,
        "init_realistic_connectivity": False,
        "init_WPP_range": [-args.init_scale, args.init_scale],
        "init_BII_range": [-args.init_scale, args.init_scale],
        "init_BPI_range": [0, 0],
        "init_WIP_range": [0, 0],
        "init_teacher_WPP_range": [-args.init_scale, args.init_scale],
        "init_teacher_BPI_range": [0, 0],
        "init_teacher_WIP_range": [0, 0],
        "init_teacher_BPP_range": [-args.init_scale, args.init_scale],
        "fw_connection_mode": "layered",
        "WPP_skip_connection_range": [0, 0],
        "bw_connection_mode": args.bw_connection_mode,
        "BII_skip_connection_range": [0, 0],
        "alpha": [0.0 for _ in range(len(layers) - 1)],
        "random_seed": [args.seed],
        "init_in_SPS": False,
        "inter_low_pass": False,
        "pyr_hi_pass": False,
        "dWPP_low_pass": False,
        "dWPP_r_low_pass": False,
        "dWPP_post_low_pass": False,
        "gate_regularizer": False,
        "dWPP_use_activation": args.dWPP_use_activation,
        "varphi_transfer": not args.no_varphi_transfer,
        "input_signal": "step",
        "copy_teacher_weights": False,
        "copy_teacher_voltages": False,
        "rec_per_steps": 10**9,
        "rec_MSE": False,
        "rec_rate_MSE": True,
        "rec_error": False,
        "rec_input": False,
        "rec_target": False,
        "rec_WPP": False,
        "rec_WIP": False,
        "rec_BPP": False,
        "rec_BII": False,
        "rec_BPI": False,
        "rec_dWPP": False,
        "rec_dWIP": False,
        "rec_dBPP": False,
        "rec_dBII": False,
        "rec_dBPI": False,
        "rec_uP": False,
        "rec_uP_breve": False,
        "rec_rP_breve": False,
        "rec_rP_breve_HI": False,
        "rec_uI": False,
        "rec_uI_breve": False,
        "rec_rI_breve": False,
        "rec_vbas": False,
        "rec_vapi": False,
        "rec_vapi_noise": False,
        "rec_noise": False,
        "rec_epsilon": False,
        "rec_epsilon_LO": False,
        "rec_lat_mismatch": False,
    }


class ReferenceErrorMCLLMAdapter:
    def __init__(self, vocab_size: int, args: argparse.Namespace) -> None:
        self.vocab_size = int(vocab_size)
        self.args = args
        self.context_len = max(int(args.context_len), 1)
        self.d_model = max(int(args.d_model), 1)
        self.rng = np.random.default_rng(args.seed + 501)
        self.token_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (self.vocab_size, self.d_model)).astype(np.float32)
        )
        self.position_codes = sinusoidal_positions(self.context_len, self.d_model)
        self.params = build_reference_params(args, self.vocab_size)
        self.mc = init_MC(self.params, [args.seed], teacher=False)[0]
        self.output_counts = np.ones(self.vocab_size, dtype=np.float32)
        self.output_bias = np.full(self.vocab_size, -math.log(max(self.vocab_size, 1)), dtype=np.float32)

    def context_feature(self, context: np.ndarray) -> np.ndarray:
        clipped = np.asarray(context[-self.context_len :], dtype=np.int64)
        offset = self.context_len - len(clipped)
        x = self.token_codes[clipped] + self.position_codes[offset:]
        state = np.mean(x, axis=0).astype(np.float32)
        return phase.normalize_vector(state)

    def target_vector(self, target: int) -> np.ndarray:
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        vec[int(target)] = 1.0
        return vec

    def logits(self) -> np.ndarray:
        return (self.args.logit_scale * self.mc.rP_breve[-1] + self.output_bias).astype(np.float32)

    def observe(self, context: np.ndarray, target: int, learn: bool) -> np.ndarray:
        r0 = self.context_feature(context)
        u_tgt = [self.target_vector(int(target))] if learn else None
        self.mc.evolve_system(
            r0=r0,
            u_tgt=u_tgt,
            learn_weights=learn,
            learn_lat_weights=learn and self.args.learn_lat_weights,
            learn_bw_weights=learn and self.args.learn_bw_weights,
            record=False,
        )
        if learn:
            self.output_counts[int(target)] += 1.0
            probs = self.output_counts / float(np.sum(self.output_counts))
            self.output_bias = np.log(np.maximum(probs, 1e-9)).astype(np.float32)
        return self.logits()

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


def run_pass(
    model: ReferenceErrorMCLLMAdapter,
    ids: np.ndarray,
    update: bool,
    limit: int,
) -> dict[str, float | int]:
    order = model.context_len
    if len(ids) <= order:
        return summarize(0.0, 0, 0)
    max_idx = len(ids)
    if limit > 0:
        max_idx = min(max_idx, order + limit)
    loss_sum = 0.0
    correct = 0
    total = 0
    for idx in range(order, max_idx):
        target = int(ids[idx])
        logits = model.observe(ids[idx - order : idx], target, learn=update)
        loss, pred = softmax_loss_and_pred(logits, target, model.args.temperature)
        loss_sum += loss
        correct += int(pred == target)
        total += 1
    return summarize(loss_sum, correct, total)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tinystories", "gsm8k", "mix"], default="tinystories")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--gsm-train-file", type=Path, default=DEFAULT_GSM_TRAIN)
    parser.add_argument("--gsm-valid-file", type=Path, default=DEFAULT_GSM_VALID)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "u005_reference_errormc_llm")
    parser.add_argument("--train-chars", type=int, default=20_000)
    parser.add_argument("--valid-chars", type=int, default=5_000)
    parser.add_argument("--gsm-train-items", type=int, default=256)
    parser.add_argument("--gsm-valid-items", type=int, default=128)
    parser.add_argument("--max-vocab", type=int, default=128)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--model-type", choices=["FA", "BP"], default="FA")
    parser.add_argument("--bw-connection-mode", choices=["layered", "skip"], default="layered")
    parser.add_argument("--eta-fw", type=float, default=0.02)
    parser.add_argument("--eta-output", type=float, default=0.05)
    parser.add_argument("--eta-ip", type=float, default=0.0)
    parser.add_argument("--eta-pi", type=float, default=0.0)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--tpres", type=float, default=0.2)
    parser.add_argument("--settling-time", type=float, default=0.0)
    parser.add_argument("--logit-scale", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--wt-noise", type=float, default=0.0)
    parser.add_argument("--dWPP-use-activation", action="store_true")
    parser.add_argument("--no-varphi-transfer", action="store_true")
    parser.add_argument("--learn-lat-weights", action="store_true")
    parser.add_argument("--learn-bw-weights", action="store_true")
    parser.add_argument("--eval-token-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_ids, valid_ids, vocab_size = load_ids(args)
    model = ReferenceErrorMCLLMAdapter(vocab_size, args)
    start = time.perf_counter()
    train_summary = run_pass(model, train_ids, update=True, limit=args.eval_token_limit)
    train_seconds = time.perf_counter() - start
    valid_pre = run_pass(model, valid_ids, update=False, limit=args.eval_token_limit)
    valid_online = run_pass(model, valid_ids, update=True, limit=args.eval_token_limit)
    valid_post = run_pass(model, valid_ids, update=False, limit=args.eval_token_limit)
    summary = {
        "task": args.task,
        "seed": int(args.seed),
        "model_type": args.model_type,
        "bw_connection_mode": args.bw_connection_mode,
        "vocab_size": int(vocab_size),
        "context_len": int(args.context_len),
        "d_model": int(args.d_model),
        "blocks": int(args.blocks),
        "actual_params": int(model.parameter_count()),
        "actual_state_bytes": int(model.state_bytes()),
        "train_loss": train_summary["loss"],
        "train_acc": train_summary["accuracy"],
        "valid_pre_loss": valid_pre["loss"],
        "valid_pre_acc": valid_pre["accuracy"],
        "valid_online_loss": valid_online["loss"],
        "valid_online_acc": valid_online["accuracy"],
        "valid_post_loss": valid_post["loss"],
        "valid_post_acc": valid_post["accuracy"],
        "train_seconds": float(train_seconds),
    }
    write_csv(args.out_dir / "summary.csv", [summary])
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "summary": summary,
            },
            f,
            indent=2,
        )
    print("Summary:")
    print(
        f"  task={args.task} model={args.model_type}/{args.bw_connection_mode} "
        f"params={model.parameter_count():,} valid_post={valid_post['loss']:.3f}/{valid_post['accuracy']:.3f}"
    )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
