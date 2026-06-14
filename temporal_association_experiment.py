#!/usr/bin/env python3
"""
Minimal experiment for temporal association with NumPy LSTM, STDP, and BTSP.

The experiment isolates sequence learning from perception:
  - visible states: 00 -> 01 -> 10 -> 11 -> 00
  - latent states: one-hot concept neurons for each visible state
  - supervised baseline: a small LSTM trained with BPTT in NumPy
  - local baseline: asymmetric STDP on recurrent connections
  - local rule: BTSP-like plateau-gated recurrent association learning
  - long-association condition: 00 -> random insert -> 10 -> 11

Outputs:
  - printed transition matrices and prediction accuracies
  - a small summary figure saved to temporal_association_results.png
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


STATE_NAMES = ["00", "01", "10", "11"]
STATE_PATTERNS = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)
LATENT_CODES = np.eye(4, dtype=np.float32)
NUM_STATES = len(STATE_NAMES)
INPUT_DIM = STATE_PATTERNS.shape[1]
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class SequenceSTDPConfig:
    a_plus: float = 0.15
    a_minus: float = 0.12
    trace_decay: float = 0.70
    weight_decay: float = 0.995
    epochs: int = 10
    seed: int = 0


@dataclass
class SequenceBTSPConfig:
    potentiation: float = 0.32
    heterosynaptic_depression: float = 0.035
    eligibility_decay: float = 0.30
    weight_decay: float = 0.998
    max_weight: float = 2.5
    epochs: int = 12
    seed: int = 0


@dataclass
class NumpyLSTMConfig:
    hidden_dim: int = 12
    lr: float = 0.12
    epochs: int = 500
    eval_episodes: int = 128
    grad_clip: float = 2.0
    seed: int = 0


@dataclass
class BioGraphConfig:
    """Sparse directed E/I graph with compartmental recurrent plasticity."""

    exc_per_state: int = 10
    inh_per_state: int = 2
    epochs: int = 80
    seed: int = 0
    mean_exc_out: int = 10
    mean_inh_out: int = 10
    max_active_exc_inputs: int = 12
    length_scale: float = 1.35
    cue_gain: float = 1.25
    recurrent_gain: float = 1.05
    apical_gain: float = 0.55
    inhibition_gain: float = 0.90
    membrane_mix: float = 0.55
    threshold: float = 0.10
    trace_decay: float = 0.58
    eligibility_decay: float = 0.82
    stdp_plus: float = 0.010
    stdp_minus: float = 0.004
    btsp_potentiation: float = 0.026
    apical_btsp_scale: float = 0.45
    heterosynaptic_depression: float = 0.0015
    weight_decay: float = 0.999
    max_exc_weight: float = 0.85
    max_inh_weight: float = 0.70
    row_norm: float = 2.40
    noise_std: float = 0.002


def structured_cycle_sequence() -> list[int]:
    return [0, 1, 2, 3]


def random_order_sequence(rng: np.random.Generator) -> list[int]:
    return rng.permutation(NUM_STATES).tolist()


def delayed_00_to_10_sequence(rng: np.random.Generator) -> list[int]:
    inserted_state = int(rng.integers(NUM_STATES))
    return [0, inserted_state, 2, 3]


def sequence_from_fn(sequence_fn, rng: np.random.Generator) -> list[int]:
    if sequence_fn is structured_cycle_sequence:
        return sequence_fn()
    return sequence_fn(rng)


def train_recurrent_stdp(sequence_fn, cfg: SequenceSTDPConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Learn recurrent transition weights between latent state neurons with asymmetric STDP.

    Update rule:
        dR = A+ * post_t outer trace_{t-1} - A- * trace_{t-1} outer post_t

    This is the original local temporal-association baseline:
      - strengthen past -> current
      - weaken current -> past
    """
    rng = np.random.default_rng(cfg.seed)
    r = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float32)
    trace = np.zeros(NUM_STATES, dtype=np.float32)
    transition_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int32)

    for _ in range(cfg.epochs):
        seq = sequence_from_fn(sequence_fn, rng)
        for idx, curr_state in enumerate(seq):
            prev_state = seq[idx - 1]
            transition_counts[curr_state, prev_state] += 1

            h = LATENT_CODES[curr_state]
            delta = cfg.a_plus * np.outer(h, trace) - cfg.a_minus * np.outer(trace, h)
            r = cfg.weight_decay * r + delta.astype(np.float32)
            trace = cfg.trace_decay * trace + h

    return r, transition_counts


def train_recurrent_btsp(sequence_fn, cfg: SequenceBTSPConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Learn recurrent transition weights between latent state neurons with a BTSP-like rule.

    BTSP is modeled here as a broad eligibility trace gated by a plateau event in the
    currently active concept cell:

        eligibility_t = decay * eligibility_{t-1} + previous latent activity
        R[current, :] += potentiation * plateau_current * eligibility_t

    A small heterosynaptic depression term keeps the active row competitive instead
    of letting every recent state saturate equally.
    """
    rng = np.random.default_rng(cfg.seed)
    r = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float32)
    transition_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int32)

    for _ in range(cfg.epochs):
        seq = sequence_from_fn(sequence_fn, rng)
        trace = LATENT_CODES[seq[-1]].copy()

        for idx, curr_state in enumerate(seq):
            prev_state = seq[idx - 1]
            transition_counts[curr_state, prev_state] += 1

            plateau = LATENT_CODES[curr_state]
            r *= cfg.weight_decay
            r += cfg.potentiation * np.outer(plateau, trace)

            row = r[curr_state]
            inactive_trace = 1.0 - np.clip(trace, 0.0, 1.0)
            row -= cfg.heterosynaptic_depression * inactive_trace * row
            row[curr_state] = 0.0
            r[curr_state] = row

            np.clip(r, 0.0, cfg.max_weight, out=r)
            trace = cfg.eligibility_decay * trace + plateau

    return r, transition_counts


class BioGraphNetwork:
    """
    Compartmental sparse recurrent graph inspired by cortical microcircuits.

    The model is still deliberately small enough for this toy experiment, but it
    differs from the 4x4 transition matrices:
      - each visible state is represented by a population of excitatory cells
      - inhibitory cells provide local competition
      - excitatory cells have basal and apical recurrent compartments
      - all excitatory pairs can grow directed links, with distance/resource costs
      - basal links use short-window STDP; apical links use BTSP-like eligibility
    """

    def __init__(self, cfg: BioGraphConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.num_exc = NUM_STATES * cfg.exc_per_state
        self.num_inh = NUM_STATES * cfg.inh_per_state
        self.exc_state = np.repeat(np.arange(NUM_STATES), cfg.exc_per_state)
        self.inh_state = np.repeat(np.arange(NUM_STATES), cfg.inh_per_state)
        self.exc_positions = self._make_positions(self.exc_state)
        self.inh_positions = self._make_positions(self.inh_state)

        self.distance_gain = self._distance_gain(self.exc_positions, self.exc_positions)
        np.fill_diagonal(self.distance_gain, 0.0)

        self.w_basal = np.zeros((self.num_exc, self.num_exc), dtype=np.float32)
        self.w_apical = np.zeros((self.num_exc, self.num_exc), dtype=np.float32)
        self.w_ei = self._fixed_sparse_weights(
            target_pos=self.inh_positions,
            source_pos=self.exc_positions,
            target_count=self.num_inh,
            source_count=self.num_exc,
            mean_out=cfg.mean_exc_out,
            max_weight=cfg.max_inh_weight,
        )
        self.w_ie = self._fixed_sparse_weights(
            target_pos=self.exc_positions,
            source_pos=self.inh_positions,
            target_count=self.num_exc,
            source_count=self.num_inh,
            mean_out=cfg.mean_inh_out,
            max_weight=cfg.max_inh_weight,
        )

    def _make_positions(self, states: np.ndarray) -> np.ndarray:
        angles = 2.0 * math.pi * states / NUM_STATES
        centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        jitter = self.rng.normal(0.0, 0.12, centers.shape)
        return (centers + jitter).astype(np.float32)

    def _distance_gain(self, target_pos: np.ndarray, source_pos: np.ndarray) -> np.ndarray:
        diff = target_pos[:, None, :] - source_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        return np.exp(-dist / self.cfg.length_scale).astype(np.float32)

    def _fixed_sparse_weights(
        self,
        target_pos: np.ndarray,
        source_pos: np.ndarray,
        target_count: int,
        source_count: int,
        mean_out: int,
        max_weight: float,
    ) -> np.ndarray:
        gain = self._distance_gain(target_pos, source_pos)
        weights = np.zeros((target_count, source_count), dtype=np.float32)
        fanout = max(1, min(target_count, mean_out))
        for source in range(source_count):
            probs = gain[:, source].astype(np.float64)
            probs /= probs.sum()
            targets = self.rng.choice(target_count, size=fanout, replace=False, p=probs)
            weights[targets, source] = self.rng.uniform(0.05, max_weight, fanout)
        return weights

    def state_population(self, state_idx: int) -> np.ndarray:
        activity = np.zeros(self.num_exc, dtype=np.float32)
        activity[self.exc_state == state_idx] = 1.0
        return activity

    def step(
        self,
        cue_state: int,
        exc_prev: np.ndarray,
        inh_prev: np.ndarray,
        eligibility: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        cue = self.state_population(cue_state)
        inh_drive = self.w_ei @ exc_prev
        inh = (1.0 - cfg.membrane_mix) * np.tanh(np.maximum(inh_drive, 0.0))
        inh += cfg.membrane_mix * inh_prev

        basal = self.w_basal @ exc_prev
        apical = self.w_apical @ eligibility
        inhibition = self.w_ie @ inh
        noise = self.rng.normal(0.0, cfg.noise_std, self.num_exc)
        voltage = (
            cfg.cue_gain * cue
            + cfg.recurrent_gain * basal
            + cfg.apical_gain * apical
            - cfg.inhibition_gain * inhibition
            + noise
        )
        exc = np.tanh(np.maximum(voltage - cfg.threshold, 0.0)).astype(np.float32)
        exc = (1.0 - cfg.membrane_mix) * exc + cfg.membrane_mix * exc_prev
        return exc.astype(np.float32), inh.astype(np.float32)

    def train(self, sequence_fn) -> np.ndarray:
        cfg = self.cfg
        transition_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int32)

        for _ in range(cfg.epochs):
            seq = sequence_from_fn(sequence_fn, self.rng)
            exc_prev = self.state_population(seq[-1])
            inh_prev = np.zeros(self.num_inh, dtype=np.float32)
            short_trace = exc_prev.copy()
            eligibility = exc_prev.copy()

            for idx, curr_state in enumerate(seq):
                prev_state = seq[idx - 1]
                transition_counts[curr_state, prev_state] += 1
                plateau = self.state_population(curr_state)

                exc, inh = self.step(curr_state, exc_prev, inh_prev, eligibility)

                stdp_pot = np.outer(plateau, short_trace) * self.distance_gain
                stdp_dep = np.outer(short_trace, plateau) * self.distance_gain
                btsp_pot = np.outer(plateau, eligibility) * self.distance_gain
                self.w_basal *= cfg.weight_decay
                self.w_basal += cfg.stdp_plus * stdp_pot
                self.w_basal -= cfg.stdp_minus * stdp_dep
                self.w_apical *= cfg.weight_decay
                self.w_apical += cfg.btsp_potentiation * btsp_pot
                self.w_apical += cfg.apical_btsp_scale * cfg.btsp_potentiation * stdp_pot

                active_rows = plateau > 0.0
                inactive_pre = 1.0 - np.clip(eligibility, 0.0, 1.0)
                self.w_basal[active_rows] -= (
                    cfg.heterosynaptic_depression
                    * self.w_basal[active_rows]
                    * inactive_pre[None, :]
                )
                self.w_apical[active_rows] -= (
                    cfg.heterosynaptic_depression
                    * self.w_apical[active_rows]
                    * inactive_pre[None, :]
                )
                self._apply_weight_constraints()

                short_trace = cfg.trace_decay * short_trace + plateau
                eligibility = cfg.eligibility_decay * eligibility + plateau
                eligibility = np.clip(eligibility, 0.0, 1.0).astype(np.float32)
                exc_prev, inh_prev = exc, inh

        return transition_counts

    def _apply_weight_constraints(self) -> None:
        cfg = self.cfg
        np.fill_diagonal(self.w_basal, 0.0)
        np.fill_diagonal(self.w_apical, 0.0)
        np.clip(self.w_basal, 0.0, cfg.max_exc_weight, out=self.w_basal)
        np.clip(self.w_apical, 0.0, cfg.max_exc_weight, out=self.w_apical)
        self._normalize_rows(self.w_basal)
        self._normalize_rows(self.w_apical)

    def _normalize_rows(self, weights: np.ndarray) -> None:
        max_inputs = self.cfg.max_active_exc_inputs
        if max_inputs < weights.shape[1]:
            for row_idx in range(weights.shape[0]):
                row = weights[row_idx]
                active = np.flatnonzero(row > 0.0)
                if len(active) > max_inputs:
                    keep = np.argpartition(row, -max_inputs)[-max_inputs:]
                    mask = np.ones(row.shape[0], dtype=bool)
                    mask[keep] = False
                    row[mask] = 0.0
        row_sums = weights.sum(axis=1, keepdims=True)
        scale = np.minimum(1.0, self.cfg.row_norm / (row_sums + 1e-8))
        weights *= scale

    def predict_matrix(self) -> np.ndarray:
        matrix = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float32)
        weights = self.w_basal + self.w_apical
        for current_state in range(NUM_STATES):
            current = self.state_population(current_state)
            inh = np.tanh(np.maximum(self.w_ei @ current, 0.0))
            drive = weights @ current - self.cfg.inhibition_gain * (self.w_ie @ inh)
            scores = np.array(
                [float(drive[self.exc_state == target].mean()) for target in range(NUM_STATES)],
                dtype=np.float32,
            )
            matrix[:, current_state] = softmax(3.0 * scores)
        return matrix

    def delayed_probe(self) -> np.ndarray:
        probs_by_insert = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float32)
        for insert_state in range(NUM_STATES):
            exc = np.zeros(self.num_exc, dtype=np.float32)
            inh = np.zeros(self.num_inh, dtype=np.float32)
            eligibility = np.zeros(self.num_exc, dtype=np.float32)
            for state_idx in [0, insert_state]:
                exc, inh = self.step(state_idx, exc, inh, eligibility)
                eligibility = np.clip(
                    self.cfg.eligibility_decay * eligibility + self.state_population(state_idx),
                    0.0,
                    1.0,
                ).astype(np.float32)

            current = self.state_population(insert_state)
            drive = self.w_basal @ current + self.w_apical @ eligibility
            scores = np.array(
                [float(drive[self.exc_state == target].mean()) for target in range(NUM_STATES)],
                dtype=np.float32,
            )
            probs_by_insert[insert_state] = softmax(3.0 * scores)
        return probs_by_insert


def train_biograph_next_state(
    sequence_fn,
    cfg: BioGraphConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, BioGraphNetwork]:
    model = BioGraphNetwork(cfg)
    transition_counts = model.train(sequence_fn)
    matrix = model.predict_matrix()
    delayed_probe = model.delayed_probe() if sequence_fn is delayed_00_to_10_sequence else None
    return matrix, transition_counts, delayed_probe, model


class NumpyLSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        scale = 1.0 / math.sqrt(input_dim + hidden_dim)
        self.w = rng.normal(0.0, scale, (4 * hidden_dim, input_dim + hidden_dim)).astype(np.float32)
        self.b = np.zeros(4 * hidden_dim, dtype=np.float32)
        self.b[hidden_dim : 2 * hidden_dim] = 1.0
        self.w_out = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), (output_dim, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(output_dim, dtype=np.float32)

    def zero_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(self.hidden_dim, dtype=np.float32),
            np.zeros(self.hidden_dim, dtype=np.float32),
        )

    def step(
        self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        concat = np.concatenate([x, h_prev]).astype(np.float32)
        z = self.w @ concat + self.b
        i = sigmoid(z[: self.hidden_dim])
        f = sigmoid(z[self.hidden_dim : 2 * self.hidden_dim])
        o = sigmoid(z[2 * self.hidden_dim : 3 * self.hidden_dim])
        g = np.tanh(z[3 * self.hidden_dim :])
        c = f * c_prev + i * g
        tanh_c = np.tanh(c)
        h = o * tanh_c
        logits = self.w_out @ h + self.b_out
        cache = {
            "x": x,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "concat": concat,
            "i": i,
            "f": f,
            "o": o,
            "g": g,
            "c": c,
            "tanh_c": tanh_c,
            "h": h,
            "logits": logits,
        }
        return logits, h.astype(np.float32), c.astype(np.float32), cache

    def predict_matrix(
        self, sequence_fn, episodes: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        matrix = np.zeros((self.output_dim, NUM_STATES), dtype=np.float32)
        source_counts = np.zeros(NUM_STATES, dtype=np.float32)
        transition_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.int32)

        for _ in range(episodes):
            seq = sequence_from_fn(sequence_fn, rng)
            targets = [seq[(idx + 1) % len(seq)] for idx in range(len(seq))]
            h, c = self.zero_state()

            for state_idx, target_idx in zip(seq, targets):
                logits, h, c, _ = self.step(STATE_PATTERNS[state_idx], h, c)
                matrix[:, state_idx] += softmax(logits)
                source_counts[state_idx] += 1.0
                transition_counts[target_idx, state_idx] += 1

        matrix /= np.maximum(source_counts[None, :], 1.0)
        return matrix, transition_counts

    def delayed_probe(self) -> np.ndarray:
        probs_by_insert = np.zeros((NUM_STATES, self.output_dim), dtype=np.float32)
        for insert_state in range(NUM_STATES):
            h, c = self.zero_state()
            _, h, c, _ = self.step(STATE_PATTERNS[0], h, c)
            logits, _, _, _ = self.step(STATE_PATTERNS[insert_state], h, c)
            probs_by_insert[insert_state] = softmax(logits)
        return probs_by_insert


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    exp_logits = np.exp(shifted)
    return (exp_logits / np.sum(exp_logits)).astype(np.float32)


def train_lstm_next_state(
    sequence_fn, cfg: NumpyLSTMConfig
) -> tuple[np.ndarray, np.ndarray, list[float], np.ndarray | None]:
    rng = np.random.default_rng(cfg.seed + 101)
    model = NumpyLSTM(INPUT_DIM, cfg.hidden_dim, NUM_STATES, cfg.seed)
    losses: list[float] = []

    for _ in range(cfg.epochs):
        seq = sequence_from_fn(sequence_fn, rng)
        targets = [seq[(idx + 1) % len(seq)] for idx in range(len(seq))]
        h, c = model.zero_state()
        caches = []
        probs = []
        epoch_loss = 0.0

        for state_idx, target_idx in zip(seq, targets):
            x = STATE_PATTERNS[state_idx]
            logits, h, c, cache = model.step(x, h, c)
            p = softmax(logits)
            probs.append(p)
            caches.append(cache)
            epoch_loss += -math.log(float(p[target_idx]) + 1e-8)

        grads = {
            "w": np.zeros_like(model.w),
            "b": np.zeros_like(model.b),
            "w_out": np.zeros_like(model.w_out),
            "b_out": np.zeros_like(model.b_out),
        }
        dh_next = np.zeros(model.hidden_dim, dtype=np.float32)
        dc_next = np.zeros(model.hidden_dim, dtype=np.float32)

        for t in reversed(range(len(seq))):
            target_idx = targets[t]
            cache = caches[t]
            dy = probs[t].copy()
            dy[target_idx] -= 1.0

            grads["w_out"] += np.outer(dy, cache["h"])
            grads["b_out"] += dy

            dh = model.w_out.T @ dy + dh_next
            dc = dh * cache["o"] * (1.0 - cache["tanh_c"] ** 2) + dc_next

            di = dc * cache["g"]
            df = dc * cache["c_prev"]
            do = dh * cache["tanh_c"]
            dg = dc * cache["i"]

            dz_i = di * cache["i"] * (1.0 - cache["i"])
            dz_f = df * cache["f"] * (1.0 - cache["f"])
            dz_o = do * cache["o"] * (1.0 - cache["o"])
            dz_g = dg * (1.0 - cache["g"] ** 2)
            dz = np.concatenate([dz_i, dz_f, dz_o, dz_g]).astype(np.float32)

            grads["w"] += np.outer(dz, cache["concat"])
            grads["b"] += dz

            dconcat = model.w.T @ dz
            dh_next = dconcat[INPUT_DIM:].astype(np.float32)
            dc_next = (dc * cache["f"]).astype(np.float32)

        clip_grads(list(grads.values()), cfg.grad_clip)
        model.w -= cfg.lr * grads["w"]
        model.b -= cfg.lr * grads["b"]
        model.w_out -= cfg.lr * grads["w_out"]
        model.b_out -= cfg.lr * grads["b_out"]
        losses.append(epoch_loss / len(seq))

    matrix, transition_counts = model.predict_matrix(
        sequence_fn, cfg.eval_episodes, cfg.seed + 202
    )
    delayed_probe = model.delayed_probe() if sequence_fn is delayed_00_to_10_sequence else None
    return matrix, transition_counts, losses, delayed_probe


def clip_grads(grads: list[np.ndarray], max_norm: float) -> None:
    norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads))
    if norm <= max_norm or norm == 0.0:
        return
    scale = max_norm / (norm + 1e-8)
    for grad in grads:
        grad *= scale


def dominant_next_states(transition_counts: np.ndarray) -> np.ndarray:
    return np.argmax(transition_counts, axis=0)


def dominant_previous_states(transition_counts: np.ndarray) -> np.ndarray:
    return np.argmax(transition_counts, axis=1)


def predict_next_state(matrix: np.ndarray, current_state: int) -> int:
    return int(np.argmax(matrix[:, current_state]))


def predict_previous_state(matrix: np.ndarray, current_state: int) -> int:
    return int(np.argmax(matrix[current_state, :]))


def one_step_accuracy(matrix: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(matrix, axis=0)
    return float((preds == targets).mean())


def previous_state_accuracy(matrix: np.ndarray, targets: np.ndarray) -> float:
    preds = np.argmax(matrix, axis=1)
    return float((preds == targets).mean())


def rollout(matrix: np.ndarray, start_state: int, steps: int) -> list[int]:
    cur = start_state
    seq = [cur]
    for _ in range(steps):
        cur = predict_next_state(matrix, cur)
        seq.append(cur)
    return seq


def print_result_block(name: str, matrix: np.ndarray, transition_counts: np.ndarray) -> None:
    next_targets = dominant_next_states(transition_counts)
    prev_targets = dominant_previous_states(transition_counts)
    next_preds = np.argmax(matrix, axis=0)
    prev_preds = np.argmax(matrix, axis=1)
    next_acc = one_step_accuracy(matrix, next_targets)
    prev_acc = previous_state_accuracy(matrix, prev_targets)
    print(f"\n=== {name} ===")
    print("empirical dominant next state:")
    for state_idx, target in enumerate(next_targets):
        print(f"  {STATE_NAMES[state_idx]} -> {STATE_NAMES[target]}")
    print("learned next-state prediction:")
    for state_idx, pred in enumerate(next_preds):
        print(f"  {STATE_NAMES[state_idx]} -> {STATE_NAMES[pred]}")
    print(f"one-step next-state accuracy: {next_acc:.3f}")
    print("empirical dominant previous state:")
    for state_idx, target in enumerate(prev_targets):
        print(f"  prev({STATE_NAMES[state_idx]}) = {STATE_NAMES[target]}")
    print("learned previous-state retrieval:")
    for state_idx, pred in enumerate(prev_preds):
        print(f"  prev({STATE_NAMES[state_idx]}) = {STATE_NAMES[pred]}")
    print(f"previous-state retrieval accuracy: {prev_acc:.3f}")
    print("learned transition matrix M[next, current]:")
    print(np.round(matrix, 3))
    print("rollout from 00:", " -> ".join(STATE_NAMES[i] for i in rollout(matrix, 0, 5)))


def print_conflict_probe(matrix: np.ndarray, prev_state: int, forced_current_state: int) -> None:
    expected_next_from_prev = predict_next_state(matrix, prev_state)
    retrieved_prev_from_current = predict_previous_state(matrix, forced_current_state)
    next_from_forced_current = predict_next_state(matrix, forced_current_state)
    print("\n=== Conflict probe using BTSP matrix ===")
    print(
        f"previous latent state is {STATE_NAMES[prev_state]} "
        f"(so temporal prior expects next = {STATE_NAMES[expected_next_from_prev]})"
    )
    print(
        f"but we force the current sensory cue to be {STATE_NAMES[forced_current_state]}."
    )
    print(
        f"from the learned sequence graph, the most likely predecessor of "
        f"{STATE_NAMES[forced_current_state]} is {STATE_NAMES[retrieved_prev_from_current]}"
    )
    print(
        f"and if the system accepts {STATE_NAMES[forced_current_state]} as current state, "
        f"its most likely next state becomes {STATE_NAMES[next_from_forced_current]}"
    )
    print("Interpretation:")
    print(
        f"- temporal prior from {STATE_NAMES[prev_state]} alone pushes toward "
        f"{STATE_NAMES[expected_next_from_prev]}"
    )
    print(
        f"- current cue {STATE_NAMES[forced_current_state]} carries a backward association to "
        f"{STATE_NAMES[retrieved_prev_from_current]}"
    )
    print(
        f"- and a forward continuation toward {STATE_NAMES[next_from_forced_current]}"
    )


def print_long_association_probe(
    name: str,
    matrix: np.ndarray,
    delayed_lstm_probe: np.ndarray | None = None,
) -> None:
    print(f"\n=== Long association probe | {name} ===")
    print("task motif: 00 -> random insert -> 10 -> 11")
    print(
        f"direct learned association strength 00 -> 10: "
        f"{float(matrix[2, 0]):.3f}"
    )
    direct_rank = int(np.where(np.argsort(-matrix[:, 0]) == 2)[0][0]) + 1
    print(f"rank of 10 as successor of 00: {direct_rank}/4")

    if delayed_lstm_probe is None:
        print("conditioned delayed prediction is not stateful for this local matrix.")
        return

    correct = 0
    for insert_state in range(NUM_STATES):
        probs = delayed_lstm_probe[insert_state]
        pred = int(np.argmax(probs))
        correct += int(pred == 2)
        print(
            f"after 00 -> {STATE_NAMES[insert_state]}: "
            f"pred={STATE_NAMES[pred]}, p(10)={float(probs[2]):.3f}"
        )
    print(f"conditioned delayed 10 prediction accuracy: {correct / NUM_STATES:.3f}")


def print_biograph_summary(name: str, model: BioGraphNetwork) -> None:
    cfg = model.cfg
    basal_density = float(np.count_nonzero(model.w_basal) / model.w_basal.size)
    apical_density = float(np.count_nonzero(model.w_apical) / model.w_apical.size)
    print(f"\n=== BioGraph architecture | {name} ===")
    print(
        f"excitatory neurons={model.num_exc}, inhibitory neurons={model.num_inh}, "
        f"state code={cfg.exc_per_state}E+{cfg.inh_per_state}I per visible state"
    )
    print(
        "directed recurrent E->E links are compartmental: "
        f"basal density={basal_density:.3f}, apical density={apical_density:.3f}"
    )
    print(
        "constraints: Dale-style source sign, sparse distance-biased graph, "
        "row-wise synaptic resource normalization, local inhibition, no global BP."
    )


def plot_matrix(ax, matrix: np.ndarray, title: str) -> None:
    im = ax.imshow(matrix, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("current state")
    ax.set_ylabel("next state")
    ax.set_xticks(range(NUM_STATES), STATE_NAMES)
    ax.set_yticks(range(NUM_STATES), STATE_NAMES)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)


def plot_rollout_graph(ax, matrix: np.ndarray, title: str) -> None:
    angles = np.linspace(0, 2 * math.pi, NUM_STATES, endpoint=False)
    positions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    max_abs = np.max(np.abs(matrix)) + 1e-8
    node_radius = 0.22

    for src in range(NUM_STATES):
        dst = int(np.argmax(matrix[:, src]))
        weight = float(matrix[dst, src])
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        lw = 0.8 + 3.2 * abs(weight) / max_abs
        color = "firebrick" if weight >= 0.0 else "royalblue"
        alpha = 0.86 if weight >= 0.0 else 0.70

        if src == dst:
            start = (x0 + node_radius, y0 + node_radius * 0.15)
            end = (x0 + node_radius * 0.15, y0 + node_radius)
            connectionstyle = "arc3,rad=0.85"
        else:
            direction = positions[dst] - positions[src]
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            start = tuple(positions[src] + node_radius * direction)
            end = tuple(positions[dst] - node_radius * direction)
            connectionstyle = "arc3,rad=0.18"

        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13 + 4 * abs(weight) / max_abs,
            linewidth=lw,
            color=color,
            alpha=alpha,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
        ax.add_patch(arrow)

    for idx, (x, y) in enumerate(positions):
        ax.scatter([x], [y], s=850, c="lightgray", edgecolors="black", linewidths=1.2, zorder=3)
        ax.text(
            x,
            y,
            STATE_NAMES[idx],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            zorder=4,
        )

    ax.set_title(title)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")


def save_summary_figure(
    cycle_stdp: np.ndarray,
    random_stdp: np.ndarray,
    delayed_stdp: np.ndarray,
    cycle_btsp: np.ndarray,
    random_btsp: np.ndarray,
    delayed_btsp: np.ndarray,
    cycle_lstm: np.ndarray,
    random_lstm: np.ndarray,
    delayed_lstm: np.ndarray,
    cycle_biograph: np.ndarray,
    random_biograph: np.ndarray,
    delayed_biograph: np.ndarray,
    path: str,
) -> None:
    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    plot_matrix(axes[0, 0], cycle_stdp, "Structured cycle: STDP")
    plot_rollout_graph(axes[0, 1], cycle_stdp, "Structured cycle: STDP links")
    plot_matrix(axes[0, 2], random_stdp, "Random order control: STDP")
    plot_rollout_graph(axes[0, 3], random_stdp, "Random order control: STDP links")
    plot_matrix(axes[0, 4], delayed_stdp, "Delayed 00-to-10: STDP")
    plot_rollout_graph(axes[0, 5], delayed_stdp, "Delayed 00-to-10: STDP links")
    plot_matrix(axes[1, 0], cycle_btsp, "Structured cycle: BTSP")
    plot_rollout_graph(axes[1, 1], cycle_btsp, "Structured cycle: BTSP links")
    plot_matrix(axes[1, 2], random_btsp, "Random order control: BTSP")
    plot_rollout_graph(axes[1, 3], random_btsp, "Random order control: BTSP links")
    plot_matrix(axes[1, 4], delayed_btsp, "Delayed 00-to-10: BTSP")
    plot_rollout_graph(axes[1, 5], delayed_btsp, "Delayed 00-to-10: BTSP links")
    plot_matrix(axes[2, 0], cycle_lstm, "Structured cycle: LSTM")
    plot_rollout_graph(axes[2, 1], cycle_lstm, "Structured cycle: LSTM links")
    plot_matrix(axes[2, 2], random_lstm, "Random order control: LSTM")
    plot_rollout_graph(axes[2, 3], random_lstm, "Random order control: LSTM links")
    plot_matrix(axes[2, 4], delayed_lstm, "Delayed 00-to-10: LSTM")
    plot_rollout_graph(axes[2, 5], delayed_lstm, "Delayed 00-to-10: LSTM links")
    plot_matrix(axes[3, 0], cycle_biograph, "Structured cycle: BioGraph")
    plot_rollout_graph(axes[3, 1], cycle_biograph, "Structured cycle: BioGraph links")
    plot_matrix(axes[3, 2], random_biograph, "Random order control: BioGraph")
    plot_rollout_graph(axes[3, 3], random_biograph, "Random order control: BioGraph links")
    plot_matrix(axes[3, 4], delayed_biograph, "Delayed 00-to-10: BioGraph")
    plot_rollout_graph(axes[3, 5], delayed_biograph, "Delayed 00-to-10: BioGraph links")
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    stdp_cfg = SequenceSTDPConfig()
    btsp_cfg = SequenceBTSPConfig()
    lstm_cfg = NumpyLSTMConfig()
    biograph_cfg = BioGraphConfig()

    cycle_stdp, cycle_stdp_counts = train_recurrent_stdp(structured_cycle_sequence, stdp_cfg)
    random_stdp, random_stdp_counts = train_recurrent_stdp(random_order_sequence, stdp_cfg)
    delayed_stdp, delayed_stdp_counts = train_recurrent_stdp(delayed_00_to_10_sequence, stdp_cfg)
    cycle_btsp, cycle_btsp_counts = train_recurrent_btsp(structured_cycle_sequence, btsp_cfg)
    random_btsp, random_btsp_counts = train_recurrent_btsp(random_order_sequence, btsp_cfg)
    delayed_btsp, delayed_btsp_counts = train_recurrent_btsp(delayed_00_to_10_sequence, btsp_cfg)
    cycle_lstm, cycle_lstm_counts, cycle_losses, _ = train_lstm_next_state(
        structured_cycle_sequence, lstm_cfg
    )
    random_lstm, random_lstm_counts, random_losses, _ = train_lstm_next_state(
        random_order_sequence, lstm_cfg
    )
    delayed_lstm, delayed_lstm_counts, delayed_losses, delayed_lstm_probe = train_lstm_next_state(
        delayed_00_to_10_sequence, lstm_cfg
    )
    cycle_biograph, cycle_biograph_counts, _, cycle_biograph_model = train_biograph_next_state(
        structured_cycle_sequence, biograph_cfg
    )
    random_biograph, random_biograph_counts, _, random_biograph_model = train_biograph_next_state(
        random_order_sequence, biograph_cfg
    )
    delayed_biograph, delayed_biograph_counts, delayed_biograph_probe, delayed_biograph_model = (
        train_biograph_next_state(delayed_00_to_10_sequence, biograph_cfg)
    )

    print("Visible states:", ", ".join(STATE_NAMES))
    print("Visible patterns:")
    for name, pattern in zip(STATE_NAMES, STATE_PATTERNS):
        print(f"  {name}: {pattern.astype(int).tolist()}")

    print_result_block("Structured cycle 00->01->10->11->00 | STDP", cycle_stdp, cycle_stdp_counts)
    print_result_block("Random order control | STDP", random_stdp, random_stdp_counts)
    print_result_block("Delayed 00->random->10->11 | STDP", delayed_stdp, delayed_stdp_counts)
    print_result_block("Structured cycle 00->01->10->11->00 | BTSP", cycle_btsp, cycle_btsp_counts)
    print_result_block("Random order control | BTSP", random_btsp, random_btsp_counts)
    print_result_block("Delayed 00->random->10->11 | BTSP", delayed_btsp, delayed_btsp_counts)
    print_result_block("Structured cycle 00->01->10->11->00 | NumPy LSTM", cycle_lstm, cycle_lstm_counts)
    print_result_block("Random order control | NumPy LSTM", random_lstm, random_lstm_counts)
    print_result_block("Delayed 00->random->10->11 | NumPy LSTM", delayed_lstm, delayed_lstm_counts)
    print_result_block(
        "Structured cycle 00->01->10->11->00 | BioGraph local plasticity",
        cycle_biograph,
        cycle_biograph_counts,
    )
    print_result_block("Random order control | BioGraph local plasticity", random_biograph, random_biograph_counts)
    print_result_block(
        "Delayed 00->random->10->11 | BioGraph local plasticity",
        delayed_biograph,
        delayed_biograph_counts,
    )
    print(
        "\nLSTM final mean cross-entropy:"
        f" structured={cycle_losses[-1]:.4f},"
        f" random={random_losses[-1]:.4f},"
        f" delayed={delayed_losses[-1]:.4f}"
    )
    print_conflict_probe(cycle_btsp, prev_state=3, forced_current_state=2)
    print_long_association_probe("STDP", delayed_stdp)
    print_long_association_probe("BTSP", delayed_btsp)
    print_long_association_probe("NumPy LSTM", delayed_lstm, delayed_lstm_probe)
    print_long_association_probe("BioGraph local plasticity", delayed_biograph, delayed_biograph_probe)
    print_biograph_summary("structured", cycle_biograph_model)
    print_biograph_summary("random", random_biograph_model)
    print_biograph_summary("delayed", delayed_biograph_model)

    save_summary_figure(
        cycle_stdp,
        random_stdp,
        delayed_stdp,
        cycle_btsp,
        random_btsp,
        delayed_btsp,
        cycle_lstm,
        random_lstm,
        delayed_lstm,
        cycle_biograph,
        random_biograph,
        delayed_biograph,
        str(SCRIPT_DIR / "temporal_association_results.png"),
    )
    print(f"\nSaved figure: {SCRIPT_DIR / 'temporal_association_results.png'}")


if __name__ == "__main__":
    main()
