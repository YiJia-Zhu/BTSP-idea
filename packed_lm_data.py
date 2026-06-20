from __future__ import annotations

from typing import Iterable

import numpy as np


def pack_documents(docs: Iterable[np.ndarray], seq_len: int, eos_id: int) -> list[np.ndarray]:
    """Pack tokenized documents into fixed-length LM sequences.

    Each returned sequence has up to seq_len + 1 tokens so that it yields up to
    seq_len next-token targets.  Documents are separated by eos_id and no
    cross-document attention mask is created, matching common packed LM
    pretraining practice.
    """
    seq_len = max(int(seq_len), 1)
    stream: list[int] = []
    for doc in docs:
        if int(doc.size) <= 0:
            continue
        stream.extend(int(x) for x in doc.tolist())
        stream.append(int(eos_id))
    if len(stream) < 2:
        return []
    tokens = np.asarray(stream, dtype=np.int64)
    sequences: list[np.ndarray] = []
    step = seq_len
    for start in range(0, int(tokens.size) - 1, step):
        seq = tokens[start : min(start + seq_len + 1, int(tokens.size))]
        if int(seq.size) >= 2:
            sequences.append(seq.astype(np.int64, copy=False))
    return sequences


def count_sequence_targets(sequences: Iterable[np.ndarray]) -> int:
    return int(sum(max(int(seq.size) - 1, 0) for seq in sequences))


def sequence_context_targets(seq: np.ndarray, context_len: int) -> tuple[list[np.ndarray], np.ndarray]:
    contexts: list[np.ndarray] = []
    targets: list[int] = []
    context_len = max(int(context_len), 1)
    for pos in range(1, int(seq.size)):
        left = max(0, pos - context_len)
        contexts.append(seq[left:pos].astype(np.int64, copy=False))
        targets.append(int(seq[pos]))
    return contexts, np.asarray(targets, dtype=np.int64)


def flatten_sequence_batch(group: Iterable[np.ndarray], context_len: int) -> tuple[list[np.ndarray], np.ndarray]:
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for seq in group:
        seq_contexts, seq_targets = sequence_context_targets(seq, context_len)
        contexts.extend(seq_contexts)
        targets.append(seq_targets)
    if not targets:
        return contexts, np.asarray([], dtype=np.int64)
    return contexts, np.concatenate(targets).astype(np.int64, copy=False)
