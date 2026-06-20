#!/usr/bin/env python3
"""
U007 strict LLM adapter for the reference dPC / Rao-Ballard-style predictive
coding model in Error-Neuron-Microcircuits.

This file does not reimplement predictive coding.  It calls the cloned source
through `init_MC(... mc_model="dPC" ...)` and only adapts token documents into
the reference interface:

    current token + position -> r0
    next token -> u_tgt
    reference dPC_model.evolve_system(...)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

import phase_binding_token_experiment as phase
from u005_reference_errormc_llm_adapter import (
    DEFAULT_GSM_TRAIN,
    DEFAULT_GSM_VALID,
    DEFAULT_TOKENIZER,
    DEFAULT_TRAIN,
    DEFAULT_VALID,
    SCRIPT_DIR,
    init_MC,
    sinusoidal_positions,
    softmax_loss_and_pred,
    summarize,
    write_csv,
)
from u006_reference_stream_errormc_llm_adapter import (
    count_document_pairs,
    decode_compact_tokens,
    generate_samples,
    load_document_ids,
    write_samples,
)


def build_reference_dpc_params(args: argparse.Namespace, vocab_size: int) -> dict[str, Any]:
    layers = [args.d_model] + [args.d_model for _ in range(args.blocks)] + [vocab_size]
    eta_fw = [args.eta_fw for _ in range(len(layers) - 2)] + [args.eta_output]
    return {
        "mc_model": "dPC",
        "teacher_mc_model": "ann",
        "dt": args.dt,
        "Tpres": args.tpres,
        "dataset_size": 1,
        "epochs": 1,
        "settling_time": args.settling_time,
        "layers": layers,
        "error_layers": [0] + layers[1:],
        "activation": "linear",
        "model_type": args.model_type,
        "WT_noise": args.wt_noise,
        "gl": args.gl,
        "gbas": args.gbas,
        "gapi": args.gapi,
        "gden": args.gden,
        "gnI": 0.0,
        "gntgt": args.gntgt,
        "eta_fw": eta_fw,
        "eta_bw": [args.eta_bw for _ in range(max(len(layers) - 2, 0))],
        "eta_IP": [args.eta_ip for _ in range(max(len(layers) - 2, 0))],
        "eta_PI": [args.eta_pi for _ in range(max(len(layers) - 2, 0))],
        "dWPP_use_activation": args.dWPP_use_activation,
        "init_WPP_range": [-args.init_scale, args.init_scale],
        "init_WIP_range": [-args.init_scale, args.init_scale],
        "init_BPP_range": [-args.init_scale, args.init_scale],
        "init_BPI_range": [-args.init_scale, args.init_scale],
        "init_teacher_WPP_range": [-args.init_scale, args.init_scale],
        "init_teacher_WIP_range": [-args.init_scale, args.init_scale],
        "init_teacher_BPP_range": [-args.init_scale, args.init_scale],
        "init_teacher_BPI_range": [-args.init_scale, args.init_scale],
        "fw_connection_mode": "layered",
        "bw_connection_mode": "layered",
        "input_signal": "step",
        "random_seed": [args.seed],
        "copy_teacher_weights": False,
        "copy_teacher_voltages": False,
        "init_in_SPS": args.init_in_sps,
        "rec_per_steps": 10**9,
        "rec_MSE": False,
        "rec_rate_MSE": True,
        "rec_error": False,
        "rec_input": False,
        "rec_target": False,
        "rec_WPP": False,
        "rec_dWPP": False,
        "rec_WIP": False,
        "rec_dWIP": False,
        "rec_BPP": False,
        "rec_dBPP": False,
        "rec_BPI": False,
        "rec_dBPI": False,
        "rec_uP": False,
        "rec_uP_breve": False,
        "rec_rP_breve": False,
        "rec_uI": False,
        "rec_uI_breve": False,
        "rec_rI_breve": False,
        "rec_vapi": False,
        "rec_vbas": False,
        "rec_lat_mismatch": False,
    }


class ReferenceDPCLLMAdapter:
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
        self.rng = np.random.default_rng(args.seed + 701)
        self.token_codes = phase.normalize_rows(
            self.rng.normal(0.0, 1.0, (self.vocab_size, self.d_model)).astype(np.float32)
        )
        self.position_codes = sinusoidal_positions(self.context_len, self.d_model)
        self.params = build_reference_dpc_params(args, self.vocab_size)
        self.mc = init_MC(self.params, [args.seed], teacher=False)[0]
        self.output_counts = np.ones(self.vocab_size, dtype=np.float32)
        self.output_bias = np.zeros(self.vocab_size, dtype=np.float32)
        self._refresh_output_bias()
        if args.self_predicting_init:
            self.mc.set_self_predicting_state()
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
        return phase.normalize_vector((token_code + self.args.position_scale * pos_code).astype(np.float32))

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
            if learn and self.args.self_predicting_each_step:
                self.mc.set_self_predicting_state()

    def predict_next(self, token: int, position: int) -> np.ndarray:
        self.evolve_current_token(token, position, self.predict_steps, target=None, learn=False)
        return self.read_output()

    def observe(self, token: int, target: int, position: int, learn: bool = True) -> None:
        self.evolve_current_token(token, position, self.presentation_steps, target=target, learn=learn)
        if learn and self.args.output_bias_mode == "unigram":
            self.output_counts[int(target)] += 1.0
            self._refresh_output_bias()

    def reset_voltages(self) -> None:
        for name in ("uP", "uP_old", "uP_breve", "uI", "uI_old"):
            if hasattr(self.mc, name):
                values = getattr(self.mc, name)
                setattr(self.mc, name, [np.zeros_like(value) for value in values])
        for name in ("vbas", "vden", "vapi", "vapi_old"):
            if hasattr(self.mc, name):
                values = getattr(self.mc, name)
                setattr(self.mc, name, [np.zeros_like(value) for value in values])
        self.mc.rP_breve = [self.mc.activation[i](self.mc.uP_breve[i]) for i in range(len(self.mc.uP_breve))]
        self.mc.rP_breve_old = [np.zeros_like(value) for value in self.mc.rP_breve]
        self.mc.uI_breve = [value.copy() for value in self.mc.rP_breve]
        self.mc.rI_breve = self.mc.uI_breve
        self.mc.rI_breve_old = [np.zeros_like(value) for value in self.mc.rI_breve]
        self.mc.r0 = np.zeros(self.d_model, dtype=np.float32)
        self.mc.r0_old = np.zeros(self.d_model, dtype=np.float32)
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
            "BPP": [w.copy() for w in self.mc.BPP],
            "BPI": [w.copy() for w in self.mc.BPI],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.token_codes = state["token_codes"].copy()
        self.position_codes = state["position_codes"].copy()
        self.output_counts = state["output_counts"].copy()
        self.output_bias = state["output_bias"].copy()
        self.mc.WPP = [w.copy() for w in state["WPP"]]
        self.mc.WIP = [w.copy() for w in state["WIP"]]
        self.mc.BPP = [w.copy() for w in state["BPP"]]
        self.mc.BPI = [w.copy() for w in state["BPI"]]

    def state_bytes(self) -> int:
        arrays: list[np.ndarray] = [self.token_codes, self.position_codes, self.output_bias, self.output_counts]
        arrays.extend(self.mc.WPP)
        arrays.extend(self.mc.WIP)
        arrays.extend(self.mc.BPP)
        arrays.extend(self.mc.BPI)
        arrays.extend(self.mc.uP)
        arrays.extend(self.mc.uI)
        return int(sum(array.nbytes for array in arrays))

    def parameter_count(self) -> int:
        arrays: list[np.ndarray] = [self.token_codes, self.position_codes, self.output_bias, self.output_counts]
        arrays.extend(self.mc.WPP)
        arrays.extend(self.mc.WIP)
        arrays.extend(self.mc.BPP)
        arrays.extend(self.mc.BPI)
        return int(sum(array.size for array in arrays))


def run_documents(
    model: ReferenceDPCLLMAdapter,
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


def estimate_dpc_dense_params(vocab_size: int, d_model: int, blocks: int, context_len: int) -> int:
    vocab = max(int(vocab_size), 1)
    d = max(int(d_model), 1)
    b = max(int(blocks), 1)
    dims = [d for _ in range(b)] + [vocab]
    wpp = b * d * d + vocab * d
    recurrent_predictive = 0
    for lower, upper in zip(dims[:-1], dims[1:]):
        recurrent_predictive += 3 * lower * upper
    token_codes = vocab * d + max(int(context_len), 1) * d
    output_bridge = 2 * vocab
    return int(wpp + recurrent_predictive + token_codes + output_bridge)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tinystories", "gsm8k", "mix"], default="tinystories")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--valid-file", type=Path, default=DEFAULT_VALID)
    parser.add_argument("--gsm-train-file", type=Path, default=DEFAULT_GSM_TRAIN)
    parser.add_argument("--gsm-valid-file", type=Path, default=DEFAULT_GSM_VALID)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "u007_reference_dpc_llm")
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
    parser.add_argument("--target-mode", choices=["onehot", "centered"], default="onehot")
    parser.add_argument("--readout-mode", choices=["rate", "voltage"], default="rate")
    parser.add_argument("--output-bias-mode", choices=["none", "unigram"], default="none")
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--eta-fw", type=float, default=0.02)
    parser.add_argument("--eta-output", type=float, default=0.05)
    parser.add_argument("--eta-bw", type=float, default=0.0)
    parser.add_argument("--eta-ip", type=float, default=0.0)
    parser.add_argument("--eta-pi", type=float, default=0.0)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--tpres", type=float, default=0.2)
    parser.add_argument("--presentation-steps", type=int, default=0)
    parser.add_argument("--predict-steps", type=int, default=5)
    parser.add_argument("--settling-time", type=float, default=0.0)
    parser.add_argument("--logit-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--wt-noise", type=float, default=0.0)
    parser.add_argument("--gl", type=float, default=0.03)
    parser.add_argument("--gbas", type=float, default=0.10)
    parser.add_argument("--gapi", type=float, default=0.06)
    parser.add_argument("--gden", type=float, default=0.10)
    parser.add_argument("--gntgt", type=float, default=0.06)
    parser.add_argument("--dWPP-use-activation", action="store_true")
    parser.add_argument("--learn-lat-weights", action="store_true")
    parser.add_argument("--learn-bw-weights", action="store_true")
    parser.add_argument("--init-in-sps", action="store_true")
    parser.add_argument("--self-predicting-init", action="store_true")
    parser.add_argument("--self-predicting-each-step", action="store_true")
    parser.add_argument("--no-reset-doc-state", dest="reset_doc_state", action="store_false")
    parser.set_defaults(reset_doc_state=True)
    parser.add_argument("--eval-token-limit", type=int, default=600)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--sample-prompt-tokens", type=int, default=16)
    parser.add_argument("--sample-new-tokens", type=int, default=48)
    parser.add_argument("--large-vocab-size", type=int, default=50_000)
    parser.add_argument("--large-d-model", type=int, default=1536)
    parser.add_argument("--large-blocks", type=int, default=16)
    parser.add_argument("--large-context-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_docs, valid_docs, vocab_size, kept_raw, tokenizer = load_document_ids(args)
    model = ReferenceDPCLLMAdapter(vocab_size, args)
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
    large_estimate = estimate_dpc_dense_params(
        args.large_vocab_size,
        args.large_d_model,
        args.large_blocks,
        args.large_context_len,
    )
    summary = {
        "task": args.task,
        "seed": int(args.seed),
        "mc_model": "dPC",
        "model_type": args.model_type,
        "target_mode": args.target_mode,
        "readout_mode": args.readout_mode,
        "output_bias_mode": args.output_bias_mode,
        "self_predicting_init": bool(args.self_predicting_init),
        "self_predicting_each_step": bool(args.self_predicting_each_step),
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
        "large_dpc_dense_param_estimate": int(large_estimate),
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
        f"  task={args.task} model=dPC/{args.model_type} docs={len(train_docs)}/{len(valid_docs)} "
        f"steps={model.presentation_steps} params={model.parameter_count():,} "
        f"valid_pre={valid_pre['loss']:.3f}/{valid_pre['accuracy']:.3f} "
        f"valid_post={valid_post['loss']:.3f}/{valid_post['accuracy']:.3f}"
    )
    print(f"  large dPC dense estimate={large_estimate:,}")
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
