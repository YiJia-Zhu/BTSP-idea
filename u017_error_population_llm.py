#!/usr/bin/env python3
"""
U017 error-population local predictive learner.

This keeps the U012/U015 LLM adapter, tokenizer, batching, and evaluation code,
but replaces the shared code_error hidden update with a layer-wise error stream:

  delta_L comes from the output population.
  delta_l is projected from delta_{l+1} through B_l.
  W_l is updated locally with delta_l^T @ r_{l-1}.

No autograd or BP is used.  In row-vector tensor notation, the BP-aligned
setting stores B_l as the row-vector equivalent of the paper's transpose
error-projection matrix.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import u012_u009_torch_fast_variants as u012


class TorchU017ErrorPopulationModel(u012.TorchU009LocalModel):
    def __init__(
        self,
        vocab_size: int,
        cfg: u012.U012Config,
        device: torch.device,
        error_mode: str,
        error_noise: float,
    ) -> None:
        super().__init__(vocab_size, cfg, device)
        self.error_mode = str(error_mode)
        self.error_noise_scale = float(error_noise)
        rng = np.random.default_rng(cfg.seed + 17017)
        self.error_noises: list[torch.Tensor] = []
        self.error_weights: list[torch.Tensor] = []
        for _ in range(max(self.blocks - 1, 0)):
            if self.error_noise_scale > 0.0:
                noise = rng.normal(
                    0.0,
                    self.error_noise_scale / math.sqrt(self.d_model),
                    (self.d_model, self.d_model),
                ).astype(np.float32)
                self.error_noises.append(torch.tensor(noise, device=device))
            else:
                self.error_noises.append(torch.zeros((self.d_model, self.d_model), device=device))
            self.error_weights.append(torch.empty((self.d_model, self.d_model), device=device))
        if self.error_mode == "fixed":
            for idx in range(len(self.error_weights)):
                values = rng.normal(
                    0.0, 1.0 / math.sqrt(self.d_model), (self.d_model, self.d_model)
                ).astype(np.float32)
                self.error_weights[idx] = torch.tensor(values, device=device)
        elif self.error_mode == "transpose":
            self.refresh_error_weights()
        else:
            raise ValueError(f"unknown error_mode {self.error_mode!r}")

    def refresh_error_weights(self) -> None:
        if self.error_mode != "transpose":
            return
        # The official code sets BPP = WPP.T in column-vector notation.  This
        # code uses row vectors with hidden = input @ W.T, so the equivalent
        # backward projection is multiplication by W, stored directly here.
        for idx in range(len(self.error_weights)):
            self.error_weights[idx].copy_(self.ff_weights[idx + 1] + self.error_noises[idx])

    def parameter_count(self) -> int:
        return int(super().parameter_count() + sum(t.numel() for t in self.error_weights))

    def state_bytes(self) -> int:
        return int(super().state_bytes() + sum(t.numel() * t.element_size() for t in self.error_weights))

    def observe_exact(self, context: np.ndarray, target: int) -> tuple[float, int]:
        loss, correct, _ = self.observe_chunk([context], np.asarray([target], dtype=np.int64))
        feature, _ = self.forward_one(context, collect=False)
        pred = int(torch.argmax(torch.softmax(self.logits_from_feature(feature), dim=0)).item())
        return float(loss), pred if correct else pred

    def observe_chunk(self, contexts: list[np.ndarray], targets_np: np.ndarray) -> tuple[float, int, int]:
        h, traces_by_block, current_tokens = self.forward_batch(contexts, collect=True)
        targets = torch.as_tensor(targets_np, dtype=torch.long, device=self.device)
        logits = self.logits_from_features(h)
        probs = torch.softmax(logits / max(float(self.cfg.temperature), 1e-6), dim=1)
        target_probs = probs.gather(1, targets[:, None]).squeeze(1)
        losses = -torch.log(torch.clamp(target_probs, min=1e-12))
        preds = torch.argmax(probs, dim=1)
        correct = int((preds == targets).sum().item())

        output_error = -probs
        output_error[torch.arange(targets.numel(), device=self.device), targets] += 1.0
        final_error = float(self.cfg.logit_scale) * (output_error @ self.output_weights)

        self.output_weights.addmm_(output_error.T, h, beta=1.0, alpha=float(self.cfg.output_lr))
        self.update_count_bias_chunk(targets)

        layer_deltas: list[torch.Tensor] = [torch.empty_like(final_error) for _ in range(self.blocks)]
        next_error = final_error
        for block_idx in range(self.blocks - 1, -1, -1):
            if block_idx < self.blocks - 1:
                projected = next_error @ self.error_weights[block_idx]
            else:
                projected = next_error
            hidden = traces_by_block[block_idx]["hidden"]
            delta = projected * (1.0 - hidden.square())
            layer_deltas[block_idx] = delta
            next_error = delta

        for block_idx, delta in enumerate(layer_deltas):
            local_inputs = traces_by_block[block_idx]["input"]
            self.ff_weights[block_idx].addmm_(delta.T, local_inputs, beta=1.0, alpha=float(self.cfg.hidden_lr))
            if self.cfg.hidden_bias_lr > 0.0:
                self.ff_biases[block_idx].add_(delta.sum(dim=0), alpha=float(self.cfg.hidden_bias_lr))

        if self.cfg.embedding_lr > 0.0 and layer_deltas:
            first_error = layer_deltas[0]
            for token, code_error in zip(current_tokens, first_error):
                self.input_codes[token] = u012.torch_normalize_vector(
                    self.input_codes[token] + float(self.cfg.embedding_lr) * code_error
                )

        self.step += int(targets.numel())
        if self.cfg.row_norm_interval > 0 and self.step // int(self.cfg.row_norm_interval) != (
            (self.step - int(targets.numel())) // int(self.cfg.row_norm_interval)
        ):
            self.ff_weights = [u012.torch_normalize_rows(weight) for weight in self.ff_weights]
            self.refresh_error_weights()
        elif self.error_mode == "transpose":
            self.refresh_error_weights()

        return float(losses.sum().item()), correct, int(targets.numel())


def build_parser() -> argparse.ArgumentParser:
    parser = u012.build_parser()
    parser.set_defaults(out_dir=u012.SCRIPT_DIR / "output" / "u017_error_population_llm")
    parser.add_argument("--error-mode", choices=["transpose", "fixed"], default="transpose")
    parser.add_argument("--error-noise", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = u012.AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    if int(args.pack_eos_id) < 0:
        args.pack_eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else len(tokenizer) - 1)
    train_text_docs, valid_text_docs = u012.u009.load_text_documents(args)
    train_docs = u012.u009.tokenize_documents(tokenizer, train_text_docs)
    valid_docs = u012.u009.tokenize_documents(tokenizer, valid_text_docs)
    cfg = u012.U012Config(
        context_len=args.context_len,
        d_model=args.d_model,
        blocks=args.blocks,
        attn_rank=args.attn_rank,
        output_lr=args.output_lr,
        hidden_lr=args.hidden_lr,
        hidden_bias_lr=args.hidden_bias_lr,
        embedding_lr=args.embedding_lr,
        logit_scale=args.logit_scale,
        attention_scale=args.attention_scale,
        ff_scale=args.ff_scale,
        position_scale=args.position_scale,
        temperature=args.temperature,
        row_norm_interval=args.row_norm_interval,
        bias_alpha=args.bias_alpha,
        seed=args.seed,
    )
    model = TorchU017ErrorPopulationModel(
        int(len(tokenizer)),
        cfg,
        device,
        error_mode=args.error_mode,
        error_noise=args.error_noise,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.skip_train_probes:
        pre_train_probe = {"loss": float("nan"), "accuracy": float("nan"), "tokens": 0, "ppl": float("nan")}
    else:
        eval_mode = "packed" if args.mode == "packed" else "exact"
        pre_train_probe, _ = u012.run_documents(
            model,
            train_docs,
            update=False,
            mode=eval_mode,
            chunk_size=args.chunk_size,
            chunk_tokens=0,
            eos_id=args.pack_eos_id,
        )
    eval_mode = "packed" if args.mode == "packed" else "exact"
    valid_pre, _ = u012.run_documents(
        model,
        valid_docs,
        update=False,
        mode=eval_mode,
        chunk_size=args.chunk_size,
        chunk_tokens=0,
        eos_id=args.pack_eos_id,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = u012.time.perf_counter()
    train_summary, train_chunks = u012.run_documents(
        model,
        train_docs,
        update=True,
        mode=args.mode,
        chunk_size=args.chunk_size,
        chunk_tokens=args.chunk_tokens,
        eos_id=args.pack_eos_id,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = u012.time.perf_counter() - start
    if args.skip_train_probes:
        post_train_probe = {"loss": float("nan"), "accuracy": float("nan"), "tokens": 0, "ppl": float("nan")}
    else:
        post_train_probe, _ = u012.run_documents(
            model,
            train_docs,
            update=False,
            mode=eval_mode,
            chunk_size=args.chunk_size,
            chunk_tokens=0,
            eos_id=args.pack_eos_id,
        )
    valid_post, _ = u012.run_documents(
        model,
        valid_docs,
        update=False,
        mode=eval_mode,
        chunk_size=args.chunk_size,
        chunk_tokens=0,
        eos_id=args.pack_eos_id,
    )
    valid_unigram = u012.evaluate_unigram(valid_docs, u012.unigram_from_docs(train_docs, int(len(tokenizer))))
    samples = u012.generate_samples(
        model,
        valid_docs,
        tokenizer,
        args.sample_count,
        args.sample_prompt_tokens,
        args.sample_new_tokens,
        args.sample_temperature,
    )
    u012.write_samples(args.out_dir / "samples.txt", samples)
    u012.write_csv(args.out_dir / "train_chunks.csv", train_chunks)

    max_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    train_packed_sequences = u012.packed_lm_data.pack_documents(train_docs, args.context_len, args.pack_eos_id)
    valid_packed_sequences = u012.packed_lm_data.pack_documents(valid_docs, args.context_len, args.pack_eos_id)
    summary = {
        "task": args.task,
        "mode": args.mode,
        "error_mode": args.error_mode,
        "error_noise": float(args.error_noise),
        "seed": int(args.seed),
        "device": str(device),
        "tokenizer_len": int(len(tokenizer)),
        "train_docs": int(len(train_docs)),
        "valid_docs": int(len(valid_docs)),
        "train_pairs": int(u012.count_document_pairs(train_docs)),
        "valid_pairs": int(u012.count_document_pairs(valid_docs)),
        "packed_train_sequences": int(len(train_packed_sequences)),
        "packed_valid_sequences": int(len(valid_packed_sequences)),
        "packed_train_targets": int(u012.packed_lm_data.count_sequence_targets(train_packed_sequences)),
        "packed_valid_targets": int(u012.packed_lm_data.count_sequence_targets(valid_packed_sequences)),
        "context_len": int(args.context_len),
        "pack_eos_id": int(args.pack_eos_id),
        "d_model": int(args.d_model),
        "blocks": int(args.blocks),
        "attn_rank": int(args.attn_rank),
        "chunk_size": int(args.chunk_size),
        "parameters": int(model.parameter_count()),
        "state_bytes": int(model.state_bytes()),
        "pre_train_probe_loss": pre_train_probe["loss"],
        "valid_pre_loss": valid_pre["loss"],
        "train_loss": train_summary["loss"],
        "train_acc": train_summary["accuracy"],
        "train_tokens": train_summary["tokens"],
        "post_train_probe_loss": post_train_probe["loss"],
        "valid_post_loss": valid_post["loss"],
        "valid_post_acc": valid_post["accuracy"],
        "valid_unigram_loss": valid_unigram["loss"],
        "train_seconds": float(train_seconds),
        "tokens_per_second": float(train_summary["tokens"] / max(train_seconds, 1e-9)),
        "max_cuda_mem_bytes": int(max_mem),
    }
    u012.write_csv(args.out_dir / "summary.csv", [summary])
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        u012.json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "config": asdict(cfg),
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
        f"  mode={args.mode} task={args.task} error={args.error_mode} tokenizer_len={len(tokenizer)} "
        f"d={args.d_model} blocks={args.blocks} params={model.parameter_count():,} "
        f"state={model.state_bytes() / (1024 ** 2):.1f} MiB"
    )
    print(
        f"  train={train_summary['loss']:.4f}/{train_summary['accuracy']:.4f} "
        f"chunks={first_chunk:.4f}->{last_chunk:.4f} tokens={train_summary['tokens']} "
        f"speed={summary['tokens_per_second']:.2f} tok/s"
    )
    print(
        f"  valid={valid_pre['loss']:.4f}->{valid_post['loss']:.4f} "
        f"valid_unigram={valid_unigram['loss']:.4f} max_mem={max_mem / (1024 ** 3):.2f} GiB"
    )
    print(f"  samples: {args.out_dir / 'samples.txt'}")
    print(f"  summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
