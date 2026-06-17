#!/usr/bin/env python3
"""
Compositional cue experiment for pure no-BP recurrent learning.

Task:
    C_a, F, C_b, F...F, Q -> T_((a + b) mod K)

The train split withholds cue pairs, so target-position accuracy on held-out
pairs tests whether a method learned a compositional rule rather than a pair
lookup table.  The no-BP method is an e-prop-style three-factor recurrent
network with fixed random feedback and eligibility traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class SplitConfig:
    k: int = 4
    heldout_fraction: float = 0.25
    seed: int = 0


@dataclass
class RNNConfig:
    hidden_dim: int = 32
    recurrent_scale: float = 0.55
    input_scale: float = 0.80
    output_scale: float = 0.10
    lr_hidden: float = 0.010
    lr_out: float = 0.045
    eligibility_decay: float = 0.92
    epochs: int = 700
    grad_clip: float = 2.0
    seed: int = 0


@dataclass
class ReservoirConfig:
    hidden_dim: int = 64
    recurrent_scale: float = 0.95
    input_scale: float = 0.85
    ridge: float = 1e-2
    seed: int = 0


@dataclass
class PhaseBindingConfig:
    harmonics: int = 0
    logit_scale: float = 8.0
    code_epochs: int = 20
    code_lr: float = 0.35
    code_noise: float = 0.02
    seed: int = 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp_logits = np.exp(shifted)
    return (exp_logits / np.sum(exp_logits)).astype(np.float32)


def spectral_normalize(matrix: np.ndarray, scale: float) -> np.ndarray:
    eigvals = np.linalg.eigvals(matrix.astype(np.float64))
    radius = float(np.max(np.abs(eigvals)))
    if radius == 0.0:
        return matrix.astype(np.float32)
    return (matrix * (scale / radius)).astype(np.float32)


def make_split(cfg: SplitConfig) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rng = np.random.default_rng(cfg.seed)
    pairs = [(a, b) for a in range(cfg.k) for b in range(cfg.k)]
    heldout_count = max(1, int(round(len(pairs) * cfg.heldout_fraction)))
    for _ in range(10_000):
        shuffled = pairs.copy()
        rng.shuffle(shuffled)
        heldout = sorted(shuffled[:heldout_count])
        train = sorted(shuffled[heldout_count:])
        train_a = {a for a, _ in train}
        train_b = {b for _, b in train}
        train_targets = {(a + b) % cfg.k for a, b in train}
        heldout_targets = {(a + b) % cfg.k for a, b in heldout}
        if (
            len(train_a) == cfg.k
            and len(train_b) == cfg.k
            and len(train_targets) == cfg.k
            and heldout_targets <= train_targets
        ):
            return train, heldout
    raise RuntimeError("could not construct balanced held-out split")


def input_dim(k: int) -> int:
    return k + 2


def encode_sequence(a: int, b: int, k: int, filler_steps: int) -> list[np.ndarray]:
    dim = input_dim(k)
    cue_a = np.zeros(dim, dtype=np.float32)
    cue_a[a] = 1.0
    cue_b = np.zeros(dim, dtype=np.float32)
    cue_b[b] = 1.0
    filler = np.zeros(dim, dtype=np.float32)
    filler[k] = 1.0
    query = np.zeros(dim, dtype=np.float32)
    query[k + 1] = 1.0
    return [cue_a, filler, cue_b] + [filler.copy() for _ in range(filler_steps)] + [query]


def target_for_pair(a: int, b: int, k: int) -> int:
    return (a + b) % k


class ReservoirReadout:
    def __init__(self, k: int, cfg: ReservoirConfig) -> None:
        rng = np.random.default_rng(cfg.seed)
        self.k = k
        self.cfg = cfg
        self.w_in = rng.normal(0.0, cfg.input_scale / math.sqrt(input_dim(k)), (cfg.hidden_dim, input_dim(k))).astype(np.float32)
        raw_rec = rng.normal(0.0, 1.0 / math.sqrt(cfg.hidden_dim), (cfg.hidden_dim, cfg.hidden_dim)).astype(np.float32)
        self.w_rec = spectral_normalize(raw_rec, cfg.recurrent_scale)
        self.w_out = np.zeros((k, cfg.hidden_dim + 1), dtype=np.float32)

    def final_state(self, a: int, b: int, filler_steps: int) -> np.ndarray:
        h = np.zeros(self.cfg.hidden_dim, dtype=np.float32)
        for x in encode_sequence(a, b, self.k, filler_steps):
            h = np.tanh(self.w_in @ x + self.w_rec @ h).astype(np.float32)
        return np.concatenate([h, np.ones(1, dtype=np.float32)])

    def train(self, pairs: list[tuple[int, int]], filler_steps: int) -> None:
        x_rows = []
        y_rows = []
        for a, b in pairs:
            x_rows.append(self.final_state(a, b, filler_steps))
            y = np.zeros(self.k, dtype=np.float32)
            y[target_for_pair(a, b, self.k)] = 1.0
            y_rows.append(y)
        x_mat = np.stack(x_rows, axis=0)
        y_mat = np.stack(y_rows, axis=0)
        reg = self.cfg.ridge * np.eye(x_mat.shape[1], dtype=np.float32)
        self.w_out = (np.linalg.solve(x_mat.T @ x_mat + reg, x_mat.T @ y_mat).T).astype(np.float32)

    def predict(self, a: int, b: int, filler_steps: int) -> np.ndarray:
        return softmax(self.w_out @ self.final_state(a, b, filler_steps))


class PairLookupHebbian:
    """Hebbian prototype lookup over whole cue pairs; useful as a non-compositional control."""

    def __init__(self, k: int, logit_scale: float = 8.0) -> None:
        self.k = k
        self.logit_scale = logit_scale
        self.prototypes = np.zeros((k, k * k), dtype=np.float32)

    def feature(self, a: int, b: int) -> np.ndarray:
        x = np.zeros(self.k * self.k, dtype=np.float32)
        x[a * self.k + b] = 1.0
        return x

    def train(self, pairs: list[tuple[int, int]], filler_steps: int) -> None:
        del filler_steps
        counts = np.zeros(self.k, dtype=np.float32)
        for a, b in pairs:
            target = target_for_pair(a, b, self.k)
            self.prototypes[target] += self.feature(a, b)
            counts[target] += 1.0
        self.prototypes /= np.maximum(counts[:, None], 1.0)

    def predict(self, a: int, b: int, filler_steps: int) -> np.ndarray:
        del filler_steps
        scores = self.prototypes @ self.feature(a, b)
        return softmax(self.logit_scale * scores)


class PhaseBindingHebbian:
    """
    Oscillatory phase-binding memory with Hebbian prototype readout.

    The working memory binds C_a and C_b by adding phase on a K-state ring.  A
    local target-gated Hebbian update stores one prototype per target class.  It
    is deliberately not a token-frequency model and uses no BP.
    """

    def __init__(self, k: int, cfg: PhaseBindingConfig, scramble_second_cue: bool = False) -> None:
        self.k = k
        self.cfg = cfg
        self.harmonics = cfg.harmonics if cfg.harmonics > 0 else max(1, k // 2)
        self.scramble_second_cue = scramble_second_cue
        rng = np.random.default_rng(cfg.seed)
        self.second_cue_code = rng.permutation(k) if scramble_second_cue else np.arange(k)
        self.prototypes = np.zeros((k, 2 * self.harmonics), dtype=np.float32)

    def feature(self, a: int, b: int) -> np.ndarray:
        phase_index = (a + int(self.second_cue_code[b])) % self.k
        phase = 2.0 * math.pi * phase_index / self.k
        values: list[float] = []
        for harmonic in range(1, self.harmonics + 1):
            values.append(math.cos(harmonic * phase))
            values.append(math.sin(harmonic * phase))
        x = np.array(values, dtype=np.float32)
        return x / (np.linalg.norm(x) + 1e-8)

    def train(self, pairs: list[tuple[int, int]], filler_steps: int) -> None:
        del filler_steps
        counts = np.zeros(self.k, dtype=np.float32)
        for a, b in pairs:
            target = target_for_pair(a, b, self.k)
            self.prototypes[target] += self.feature(a, b)
            counts[target] += 1.0
        self.prototypes /= np.maximum(counts[:, None], 1.0)
        norms = np.linalg.norm(self.prototypes, axis=1, keepdims=True)
        self.prototypes /= np.maximum(norms, 1e-8)

    def predict(self, a: int, b: int, filler_steps: int) -> np.ndarray:
        del filler_steps
        scores = self.prototypes @ self.feature(a, b)
        return softmax(self.cfg.logit_scale * scores)


class LearnedPhaseBindingHebbian:
    """
    Locally learned cue-to-phase code plus Hebbian phase-binding readout.

    Training uses target-gated local attraction:
      - first cue C_a is pulled toward phase a;
      - second cue C_b is pulled toward phase (target - a) mod K;
      - target readout stores the bound phase feature by Hebbian averaging.
    """

    def __init__(self, k: int, cfg: PhaseBindingConfig) -> None:
        self.k = k
        self.cfg = cfg
        self.harmonics = cfg.harmonics if cfg.harmonics > 0 else max(1, k // 2)
        self.rng = np.random.default_rng(cfg.seed)
        self.first_codes = normalize_rows(
            self.rng.normal(0.0, 1.0, (k, 2 * self.harmonics)).astype(np.float32)
        )
        self.second_codes = normalize_rows(
            self.rng.normal(0.0, 1.0, (k, 2 * self.harmonics)).astype(np.float32)
        )
        self.prototypes = np.zeros((k, 2 * self.harmonics), dtype=np.float32)

    def phase_vector(self, phase_index: int) -> np.ndarray:
        phase = 2.0 * math.pi * (phase_index % self.k) / self.k
        values: list[float] = []
        for harmonic in range(1, self.harmonics + 1):
            values.append(math.cos(harmonic * phase))
            values.append(math.sin(harmonic * phase))
        return normalize_vector(np.array(values, dtype=np.float32))

    def update_code(self, codes: np.ndarray, cue: int, phase_index: int) -> None:
        target = self.phase_vector(phase_index)
        noise = self.rng.normal(0.0, self.cfg.code_noise, codes[cue].shape).astype(np.float32)
        codes[cue] = normalize_vector((1.0 - self.cfg.code_lr) * codes[cue] + self.cfg.code_lr * target + noise)

    def feature(self, a: int, b: int) -> np.ndarray:
        return complex_bind_phase_vectors(self.first_codes[a], self.second_codes[b])

    def train(self, pairs: list[tuple[int, int]], filler_steps: int) -> None:
        del filler_steps
        for _ in range(self.cfg.code_epochs):
            shuffled = pairs.copy()
            self.rng.shuffle(shuffled)
            for a, b in shuffled:
                target = target_for_pair(a, b, self.k)
                self.update_code(self.first_codes, a, a)
                self.update_code(self.second_codes, b, (target - a) % self.k)

        counts = np.zeros(self.k, dtype=np.float32)
        for a, b in pairs:
            target = target_for_pair(a, b, self.k)
            self.prototypes[target] += self.feature(a, b)
            counts[target] += 1.0
        self.prototypes /= np.maximum(counts[:, None], 1.0)
        self.prototypes = normalize_rows(self.prototypes)

    def predict(self, a: int, b: int, filler_steps: int) -> np.ndarray:
        del filler_steps
        scores = self.prototypes @ self.feature(a, b)
        return softmax(self.cfg.logit_scale * scores)


class TargetOnlyPhaseBindingHebbian:
    """
    Target-only local phase factorization.

    Unlike LearnedPhaseBindingHebbian, this variant never receives a direct
    cue->phase label such as C_a -> phase a.  The only teaching signal is the
    final target class.  Cue codes are updated by local complex binding rules:
    if code_a * code_b should match target_phase, then code_a is attracted to
    target_phase * conj(code_b), and code_b to conj(code_a) * target_phase.
    """

    def __init__(self, k: int, cfg: PhaseBindingConfig) -> None:
        self.k = k
        self.cfg = cfg
        self.harmonics = cfg.harmonics if cfg.harmonics > 0 else max(1, k // 2)
        self.rng = np.random.default_rng(cfg.seed)
        self.first_codes = normalize_rows(
            self.rng.normal(0.0, 1.0, (k, 2 * self.harmonics)).astype(np.float32)
        )
        self.second_codes = normalize_rows(
            self.rng.normal(0.0, 1.0, (k, 2 * self.harmonics)).astype(np.float32)
        )
        self.target_phases = np.stack([self.phase_vector(target) for target in range(k)], axis=0)
        self.prototypes = np.zeros((k, 2 * self.harmonics), dtype=np.float32)

    def phase_vector(self, phase_index: int) -> np.ndarray:
        phase = 2.0 * math.pi * (phase_index % self.k) / self.k
        values: list[float] = []
        for harmonic in range(1, self.harmonics + 1):
            values.append(math.cos(harmonic * phase))
            values.append(math.sin(harmonic * phase))
        return normalize_vector(np.array(values, dtype=np.float32))

    def feature(self, a: int, b: int) -> np.ndarray:
        return complex_bind_phase_vectors(self.first_codes[a], self.second_codes[b])

    def train(self, pairs: list[tuple[int, int]], filler_steps: int) -> None:
        del filler_steps
        for _ in range(self.cfg.code_epochs):
            shuffled = pairs.copy()
            self.rng.shuffle(shuffled)
            for a, b in shuffled:
                target_phase = self.target_phases[target_for_pair(a, b, self.k)]
                first_old = self.first_codes[a]
                second_old = self.second_codes[b]
                desired_first = complex_bind_phase_vectors(target_phase, conjugate_phase_vector(second_old))
                first_noise = self.rng.normal(0.0, self.cfg.code_noise, first_old.shape).astype(np.float32)
                first_new = normalize_vector(
                    (1.0 - self.cfg.code_lr) * first_old + self.cfg.code_lr * desired_first + first_noise
                )
                desired_second = complex_bind_phase_vectors(conjugate_phase_vector(first_new), target_phase)
                second_noise = self.rng.normal(0.0, self.cfg.code_noise, second_old.shape).astype(np.float32)
                second_new = normalize_vector(
                    (1.0 - self.cfg.code_lr) * second_old + self.cfg.code_lr * desired_second + second_noise
                )
                self.first_codes[a] = first_new
                self.second_codes[b] = second_new

        counts = np.zeros(self.k, dtype=np.float32)
        for a, b in pairs:
            target = target_for_pair(a, b, self.k)
            self.prototypes[target] += self.feature(a, b)
            counts[target] += 1.0
        self.prototypes /= np.maximum(counts[:, None], 1.0)
        self.prototypes = normalize_rows(self.prototypes)

    def predict(self, a: int, b: int, filler_steps: int) -> np.ndarray:
        del filler_steps
        scores = self.prototypes @ self.feature(a, b)
        return softmax(self.cfg.logit_scale * scores)


class ThreeFactorRNN:
    def __init__(self, k: int, cfg: RNNConfig, resample_feedback: bool = False) -> None:
        rng = np.random.default_rng(cfg.seed)
        self.k = k
        self.cfg = cfg
        self.rng = rng
        self.resample_feedback = resample_feedback
        self.w_in = rng.normal(0.0, cfg.input_scale / math.sqrt(input_dim(k)), (cfg.hidden_dim, input_dim(k))).astype(np.float32)
        raw_rec = rng.normal(0.0, 1.0 / math.sqrt(cfg.hidden_dim), (cfg.hidden_dim, cfg.hidden_dim)).astype(np.float32)
        self.w_rec = spectral_normalize(raw_rec, cfg.recurrent_scale)
        self.w_out = rng.normal(0.0, cfg.output_scale / math.sqrt(cfg.hidden_dim), (k, cfg.hidden_dim)).astype(np.float32)
        self.feedback = rng.normal(0.0, 1.0 / math.sqrt(k), (cfg.hidden_dim, k)).astype(np.float32)

    def forward_with_traces(
        self, a: int, b: int, filler_steps: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
        h = np.zeros(self.cfg.hidden_dim, dtype=np.float32)
        e_in = np.zeros_like(self.w_in)
        e_rec = np.zeros_like(self.w_rec)
        caches: list[dict[str, np.ndarray]] = []
        for x in encode_sequence(a, b, self.k, filler_steps):
            h_prev = h
            z = self.w_in @ x + self.w_rec @ h_prev
            h = np.tanh(z).astype(np.float32)
            deriv = (1.0 - h * h).astype(np.float32)
            e_in = self.cfg.eligibility_decay * e_in + deriv[:, None] * x[None, :]
            e_rec = self.cfg.eligibility_decay * e_rec + deriv[:, None] * h_prev[None, :]
            caches.append({"x": x, "h_prev": h_prev, "h": h, "deriv": deriv})
        logits = self.w_out @ h
        return logits, h, e_in.astype(np.float32), e_rec.astype(np.float32), caches

    def train_eprop(
        self,
        pairs: list[tuple[int, int]],
        filler_steps: int,
        epochs: int,
    ) -> list[float]:
        history: list[float] = []
        for _ in range(epochs):
            shuffled = pairs.copy()
            self.rng.shuffle(shuffled)
            losses = []
            for a, b in shuffled:
                target = target_for_pair(a, b, self.k)
                logits, h, e_in, e_rec, _ = self.forward_with_traces(a, b, filler_steps)
                probs = softmax(logits)
                losses.append(-math.log(float(probs[target]) + 1e-8))
                error = probs
                error[target] -= 1.0
                feedback = (
                    self.rng.normal(0.0, 1.0 / math.sqrt(self.k), self.feedback.shape).astype(np.float32)
                    if self.resample_feedback
                    else self.feedback
                )
                learning_signal = feedback @ error
                delta_out = np.outer(error, h)
                delta_in = learning_signal[:, None] * e_in
                delta_rec = learning_signal[:, None] * e_rec
                scale = clip_scale([delta_out, delta_in, delta_rec], self.cfg.grad_clip)
                self.w_out -= self.cfg.lr_out * scale * delta_out
                self.w_in -= self.cfg.lr_hidden * scale * delta_in
                self.w_rec -= self.cfg.lr_hidden * scale * delta_rec
                np.clip(self.w_rec, -1.5, 1.5, out=self.w_rec)
            history.append(float(np.mean(losses)))
        return history

    def train_bptt(
        self,
        pairs: list[tuple[int, int]],
        filler_steps: int,
        epochs: int,
    ) -> list[float]:
        history: list[float] = []
        for _ in range(epochs):
            shuffled = pairs.copy()
            self.rng.shuffle(shuffled)
            losses = []
            for a, b in shuffled:
                target = target_for_pair(a, b, self.k)
                logits, h, _, _, caches = self.forward_with_traces(a, b, filler_steps)
                probs = softmax(logits)
                losses.append(-math.log(float(probs[target]) + 1e-8))
                error = probs
                error[target] -= 1.0
                grad_w_out = np.outer(error, h)
                dh = self.w_out.T @ error
                grad_w_in = np.zeros_like(self.w_in)
                grad_w_rec = np.zeros_like(self.w_rec)
                for cache in reversed(caches):
                    dz = dh * cache["deriv"]
                    grad_w_in += np.outer(dz, cache["x"])
                    grad_w_rec += np.outer(dz, cache["h_prev"])
                    dh = self.w_rec.T @ dz
                scale = clip_scale([grad_w_out, grad_w_in, grad_w_rec], self.cfg.grad_clip)
                self.w_out -= self.cfg.lr_out * scale * grad_w_out
                self.w_in -= self.cfg.lr_hidden * scale * grad_w_in
                self.w_rec -= self.cfg.lr_hidden * scale * grad_w_rec
                np.clip(self.w_rec, -1.5, 1.5, out=self.w_rec)
            history.append(float(np.mean(losses)))
        return history

    def predict(self, a: int, b: int, filler_steps: int) -> np.ndarray:
        logits, _, _, _, _ = self.forward_with_traces(a, b, filler_steps)
        return softmax(logits)


def clip_scale(arrays: Iterable[np.ndarray], max_norm: float) -> float:
    norm = math.sqrt(sum(float(np.sum(array * array)) for array in arrays))
    if norm <= max_norm or norm == 0.0:
        return 1.0
    return max_norm / (norm + 1e-8)


def normalize_vector(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x) + 1e-8)).astype(np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def conjugate_phase_vector(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    out[1::2] *= -1.0
    return out


def complex_bind_phase_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    bound = np.zeros_like(a)
    for idx in range(0, len(a), 2):
        ar, ai = float(a[idx]), float(a[idx + 1])
        br, bi = float(b[idx]), float(b[idx + 1])
        bound[idx] = ar * br - ai * bi
        bound[idx + 1] = ai * br + ar * bi
    return normalize_vector(bound)


def evaluate_model(model: Any, pairs: list[tuple[int, int]], k: int, filler_steps: int) -> dict[str, Any]:
    losses = []
    correct = 0
    confusion = np.zeros((k, k), dtype=np.int32)
    for a, b in pairs:
        target = target_for_pair(a, b, k)
        probs = model.predict(a, b, filler_steps)
        pred = int(np.argmax(probs))
        correct += int(pred == target)
        losses.append(-math.log(float(probs[target]) + 1e-8))
        confusion[target, pred] += 1
    return {
        "accuracy": correct / max(len(pairs), 1),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "confusion": confusion.tolist(),
    }


def run_one(
    k: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split_cfg = SplitConfig(k=k, heldout_fraction=args.heldout_fraction, seed=seed)
    train_pairs, heldout_pairs = make_split(split_cfg)
    methods: list[tuple[str, Any, list[float], float]] = []
    selected_methods = set(args.methods)

    def should_run(method: str) -> bool:
        return "all" in selected_methods or method in selected_methods

    if should_run("pair_lookup_hebbian"):
        start = time.perf_counter()
        pair_lookup = PairLookupHebbian(k, args.phase_logit_scale)
        pair_lookup.train(train_pairs, args.filler_steps)
        methods.append(("pair_lookup_hebbian", pair_lookup, [], time.perf_counter() - start))

    if should_run("phase_binding_hebbian"):
        start = time.perf_counter()
        phase_binding = PhaseBindingHebbian(
            k,
            PhaseBindingConfig(args.phase_harmonics, args.phase_logit_scale, seed=seed),
            scramble_second_cue=False,
        )
        phase_binding.train(train_pairs, args.filler_steps)
        methods.append(("phase_binding_hebbian", phase_binding, [], time.perf_counter() - start))

    if should_run("learned_phase_binding_hebbian"):
        start = time.perf_counter()
        learned_phase = LearnedPhaseBindingHebbian(
            k,
            PhaseBindingConfig(
                harmonics=args.phase_harmonics,
                logit_scale=args.phase_logit_scale,
                code_epochs=args.phase_code_epochs,
                code_lr=args.phase_code_lr,
                code_noise=args.phase_code_noise,
                seed=seed,
            ),
        )
        learned_phase.train(train_pairs, args.filler_steps)
        methods.append(("learned_phase_binding_hebbian", learned_phase, [], time.perf_counter() - start))

    if should_run("target_only_phase_binding_hebbian"):
        start = time.perf_counter()
        target_only_phase = TargetOnlyPhaseBindingHebbian(
            k,
            PhaseBindingConfig(
                harmonics=args.phase_harmonics,
                logit_scale=args.phase_logit_scale,
                code_epochs=args.target_phase_epochs,
                code_lr=args.target_phase_lr,
                code_noise=args.target_phase_noise,
                seed=seed,
            ),
        )
        target_only_phase.train(train_pairs, args.filler_steps)
        methods.append(("target_only_phase_binding_hebbian", target_only_phase, [], time.perf_counter() - start))

    if should_run("phase_binding_scrambled_control"):
        start = time.perf_counter()
        phase_scrambled = PhaseBindingHebbian(
            k,
            PhaseBindingConfig(args.phase_harmonics, args.phase_logit_scale, seed=seed),
            scramble_second_cue=True,
        )
        phase_scrambled.train(train_pairs, args.filler_steps)
        methods.append(("phase_binding_scrambled_control", phase_scrambled, [], time.perf_counter() - start))

    if should_run("reservoir_readout"):
        start = time.perf_counter()
        reservoir = ReservoirReadout(k, ReservoirConfig(args.reservoir_hidden_dim, args.reservoir_recurrent_scale, args.input_scale, args.ridge, seed))
        reservoir.train(train_pairs, args.filler_steps)
        methods.append(("reservoir_readout", reservoir, [], time.perf_counter() - start))

    if should_run("eprop_3factor"):
        start = time.perf_counter()
        eprop = ThreeFactorRNN(
            k,
            RNNConfig(
                args.hidden_dim,
                args.recurrent_scale,
                args.input_scale,
                args.output_scale,
                args.lr_hidden,
                args.lr_out,
                args.eligibility_decay,
                args.epochs,
                args.grad_clip,
                seed,
            ),
            resample_feedback=False,
        )
        eprop_history = eprop.train_eprop(train_pairs, args.filler_steps, args.epochs)
        methods.append(("eprop_3factor", eprop, eprop_history, time.perf_counter() - start))

    if should_run("eprop_resampled_feedback"):
        start = time.perf_counter()
        resampled = ThreeFactorRNN(
            k,
            RNNConfig(
                args.hidden_dim,
                args.recurrent_scale,
                args.input_scale,
                args.output_scale,
                args.lr_hidden,
                args.lr_out,
                args.eligibility_decay,
                args.epochs,
                args.grad_clip,
                seed,
            ),
            resample_feedback=True,
        )
        resampled_history = resampled.train_eprop(train_pairs, args.filler_steps, args.epochs)
        methods.append(("eprop_resampled_feedback", resampled, resampled_history, time.perf_counter() - start))

    if should_run("tuned_bptt"):
        start = time.perf_counter()
        bptt = ThreeFactorRNN(
            k,
            RNNConfig(
                args.hidden_dim,
                args.recurrent_scale,
                args.input_scale,
                args.output_scale,
                args.bptt_lr_hidden,
                args.bptt_lr_out,
                args.eligibility_decay,
                args.bptt_epochs,
                args.grad_clip,
                seed,
            ),
            resample_feedback=False,
        )
        bptt_history = bptt.train_bptt(train_pairs, args.filler_steps, args.bptt_epochs)
        methods.append(("tuned_bptt", bptt, bptt_history, time.perf_counter() - start))

    result_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for method, model, history, train_seconds in methods:
        seen = evaluate_model(model, train_pairs, k, args.filler_steps)
        heldout = evaluate_model(model, heldout_pairs, k, args.filler_steps)
        result_rows.append(
            {
                "k": k,
                "seed": seed,
                "method": method,
                "train_pairs": len(train_pairs),
                "heldout_pairs": len(heldout_pairs),
                "seen_target_acc": seen["accuracy"],
                "heldout_target_acc": heldout["accuracy"],
                "seen_loss": seen["loss"],
                "heldout_loss": heldout["loss"],
                "train_seconds": train_seconds,
            }
        )
        if history:
            stride = max(1, len(history) // 50)
            for epoch, loss in enumerate(history, start=1):
                if epoch == 1 or epoch == len(history) or epoch % stride == 0:
                    history_rows.append({"k": k, "seed": seed, "method": method, "epoch": epoch, "train_loss": loss})

    metadata = {
        "k": k,
        "seed": seed,
        "train_pairs": train_pairs,
        "heldout_pairs": heldout_pairs,
    }
    return result_rows, history_rows, metadata


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["k"]), str(row["method"])), []).append(row)
    summary = []
    for (k, method), group in sorted(groups.items()):
        summary.append(
            {
                "k": k,
                "method": method,
                "seeds": len(group),
                "seen_target_acc_mean": float(np.mean([row["seen_target_acc"] for row in group])),
                "seen_target_acc_std": float(np.std([row["seen_target_acc"] for row in group])),
                "heldout_target_acc_mean": float(np.mean([row["heldout_target_acc"] for row in group])),
                "heldout_target_acc_std": float(np.std([row["heldout_target_acc"] for row in group])),
                "seen_loss_mean": float(np.mean([row["seen_loss"] for row in group])),
                "heldout_loss_mean": float(np.mean([row["heldout_loss"] for row in group])),
                "train_seconds_mean": float(np.mean([row["train_seconds"] for row in group])),
            }
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "compositional_cue")
    parser.add_argument("--k-values", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--filler-steps", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--reservoir-hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=900)
    parser.add_argument("--bptt-epochs", type=int, default=1200)
    parser.add_argument("--input-scale", type=float, default=0.85)
    parser.add_argument("--output-scale", type=float, default=0.10)
    parser.add_argument("--recurrent-scale", type=float, default=0.55)
    parser.add_argument("--reservoir-recurrent-scale", type=float, default=0.95)
    parser.add_argument("--lr-hidden", type=float, default=0.010)
    parser.add_argument("--lr-out", type=float, default=0.045)
    parser.add_argument("--bptt-lr-hidden", type=float, default=0.020)
    parser.add_argument("--bptt-lr-out", type=float, default=0.070)
    parser.add_argument("--eligibility-decay", type=float, default=0.92)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--phase-harmonics", type=int, default=0)
    parser.add_argument("--phase-logit-scale", type=float, default=8.0)
    parser.add_argument("--phase-code-epochs", type=int, default=20)
    parser.add_argument("--phase-code-lr", type=float, default=0.35)
    parser.add_argument("--phase-code-noise", type=float, default=0.02)
    parser.add_argument("--target-phase-epochs", type=int, default=300)
    parser.add_argument("--target-phase-lr", type=float, default=0.20)
    parser.add_argument("--target-phase-noise", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    split_metadata: list[dict[str, Any]] = []

    for k in args.k_values:
        for seed in args.seeds:
            result_rows, history_rows, metadata = run_one(k, seed, args)
            all_results.extend(result_rows)
            all_history.extend(history_rows)
            split_metadata.append(metadata)

    summary_rows = aggregate(all_results)
    write_csv(args.out_dir / "results.csv", all_results)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "history.csv", all_history)
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "splits": split_metadata, "summary": summary_rows, "results": all_results}, f, indent=2, default=str)

    print("Summary:")
    for row in summary_rows:
        print(
            f"  K={row['k']} {row['method']}: "
            f"seen={row['seen_target_acc_mean']:.3f}+/-{row['seen_target_acc_std']:.3f} "
            f"heldout={row['heldout_target_acc_mean']:.3f}+/-{row['heldout_target_acc_std']:.3f} "
            f"loss={row['heldout_loss_mean']:.3f}"
        )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
