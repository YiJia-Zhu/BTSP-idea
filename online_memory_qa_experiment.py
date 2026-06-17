#!/usr/bin/env python3
"""
API-compatible no-BP online memory prototype on synthetic personalization QA.

The goal is not to replace a language model.  It tests the adapter shape that a
frozen/API model could use:
  - base system has no access to newly introduced user facts
  - online memory stores compressed hashed question features and answer counts
  - no raw question/answer text is retained in the memory state
  - evaluation asks paraphrased questions after a small number of interactions
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


ATTRIBUTES = {
    "favorite color": [
        "blue",
        "green",
        "red",
        "yellow",
        "purple",
        "orange",
        "silver",
        "black",
    ],
    "favorite animal": [
        "cat",
        "dog",
        "rabbit",
        "horse",
        "fox",
        "owl",
        "panda",
        "turtle",
    ],
    "home city": [
        "paris",
        "tokyo",
        "seattle",
        "austin",
        "boston",
        "miami",
        "denver",
        "chicago",
    ],
    "favorite snack": [
        "cookies",
        "apples",
        "pretzels",
        "popcorn",
        "noodles",
        "berries",
        "cheese",
        "crackers",
    ],
    "pet name": [
        "milo",
        "luna",
        "coco",
        "buddy",
        "nori",
        "toby",
        "piper",
        "olive",
    ],
}


QUESTION_TEMPLATES = {
    "favorite color": [
        "What is {name}'s favorite color?",
        "Which color does {name} like best?",
        "Can you recall {name}'s preferred color?",
        "{name} likes which color most?",
    ],
    "favorite animal": [
        "What is {name}'s favorite animal?",
        "Which animal does {name} like best?",
        "Can you recall {name}'s preferred animal?",
        "{name} likes which animal most?",
    ],
    "home city": [
        "Where does {name} live?",
        "What is {name}'s home city?",
        "Which city is home for {name}?",
        "Can you recall {name}'s city?",
    ],
    "favorite snack": [
        "What is {name}'s favorite snack?",
        "Which snack does {name} like best?",
        "Can you recall {name}'s preferred snack?",
        "{name} likes which snack most?",
    ],
    "pet name": [
        "What is {name}'s pet called?",
        "Can you recall {name}'s pet name?",
        "What is the name of {name}'s pet?",
        "{name}'s pet has what name?",
    ],
}


LEARN_TEMPLATES = {
    "favorite color": "{name}'s favorite color is {value}.",
    "favorite animal": "{name}'s favorite animal is a {value}.",
    "home city": "{name} lives in {value}.",
    "favorite snack": "{name}'s favorite snack is {value}.",
    "pet name": "{name}'s pet is named {value}.",
}


NAMES = [
    "Ava",
    "Ben",
    "Cara",
    "Dion",
    "Eli",
    "Faye",
    "Gus",
    "Hana",
    "Iris",
    "Jude",
    "Kira",
    "Leo",
    "Mina",
    "Noah",
    "Omar",
    "Pia",
]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def token_set(text: str) -> set[str]:
    return set(normalize(text).split())


def attribute_aliases(attribute: str) -> list[str]:
    if attribute == "favorite color":
        return ["favorite color", "preferred color", "color like best", "likes which color"]
    if attribute == "favorite animal":
        return ["favorite animal", "preferred animal", "animal like best", "likes which animal"]
    if attribute == "home city":
        return ["home city", "lives in", "where live", "city home"]
    if attribute == "favorite snack":
        return ["favorite snack", "preferred snack", "snack like best", "likes which snack"]
    if attribute == "pet name":
        return ["pet name", "pet called", "pet named", "name of pet"]
    return [attribute]


def infer_attribute(text: str) -> str | None:
    norm = normalize(text)
    tokenized = set(norm.split())
    for attribute in ATTRIBUTES:
        for alias in attribute_aliases(attribute):
            alias_tokens = set(normalize(alias).split())
            if alias_tokens and alias_tokens <= tokenized:
                return attribute
    if "live" in tokenized or "lives" in tokenized:
        return "home city"
    if "city" in tokenized:
        return "home city"
    if "color" in tokenized:
        return "favorite color"
    if "animal" in tokenized:
        return "favorite animal"
    if "snack" in tokenized:
        return "favorite snack"
    if "pet" in tokenized:
        return "pet name"
    return None


def infer_name(text: str, names: set[str]) -> str | None:
    tokens = set(normalize(text).split())
    for name in names:
        if name.lower() in tokens:
            return name.lower()
    return None


def stable_hash_int(text: str, bits: int = 64) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=bits // 8).digest()
    return int.from_bytes(digest, "little")


def write_csv(path: Path, rows: list[dict]) -> None:
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


@dataclass
class Fact:
    name: str
    attribute: str
    value: str

    @property
    def answer_id(self) -> str:
        return f"{self.attribute}={self.value}"


def build_facts(num_people: int, seed: int) -> list[Fact]:
    rng = random.Random(seed)
    names = NAMES[:]
    rng.shuffle(names)
    facts: list[Fact] = []
    for idx in range(num_people):
        name = names[idx % len(names)]
        for attr, values in ATTRIBUTES.items():
            value = values[(idx + rng.randrange(len(values))) % len(values)]
            facts.append(Fact(name=name, attribute=attr, value=value))
    rng.shuffle(facts)
    return facts


def learn_sentence(fact: Fact) -> str:
    return LEARN_TEMPLATES[fact.attribute].format(name=fact.name, value=fact.value)


def question_for(fact: Fact, variant: int) -> str:
    templates = QUESTION_TEMPLATES[fact.attribute]
    return templates[variant % len(templates)].format(name=fact.name)


def answer_text(fact: Fact) -> str:
    return fact.value


class RandomProjectionMemory:
    """
    Hashed sparse feature -> answer-count memory.

    The memory stores only integer feature ids and integer answer ids.  The raw
    question, raw statement, and raw answer text are never stored after update.
    """

    def __init__(
        self,
        answer_ids: list[str],
        answer_attributes: dict[str, str],
        hash_bits: int,
        ngrams: int,
        names: set[str],
        seed: int,
        specific_feature_boost: float = 4.0,
    ) -> None:
        self.answer_ids = answer_ids
        self.answer_to_idx = {answer_id: idx for idx, answer_id in enumerate(answer_ids)}
        self.idx_to_answer = {idx: answer_id for idx, answer_id in enumerate(answer_ids)}
        self.answer_attributes = answer_attributes
        self.hash_bits = hash_bits
        self.ngrams = ngrams
        self.names = names
        self.seed = seed
        self.specific_feature_boost = specific_feature_boost
        self.tables: dict[int, dict[int, float]] = {}
        self.answer_counts = np.zeros(len(answer_ids), dtype=np.float32)

    def features(self, text: str) -> list[int]:
        tokens = normalize(text).split()
        feats: set[int] = set()
        for n in range(1, self.ngrams + 1):
            for idx in range(len(tokens) - n + 1):
                gram = " ".join(tokens[idx : idx + n])
                feats.add(stable_hash_int(f"{self.seed}:{n}:{gram}", bits=self.hash_bits))
        name = infer_name(text, self.names)
        attribute = infer_attribute(text)
        if name is not None:
            feats.add(stable_hash_int(f"{self.seed}:name:{name}", bits=self.hash_bits))
        if attribute is not None:
            feats.add(stable_hash_int(f"{self.seed}:attr:{attribute}", bits=self.hash_bits))
        if name is not None and attribute is not None:
            feats.add(stable_hash_int(f"{self.seed}:fact:{name}:{attribute}", bits=self.hash_bits))
        return sorted(feats)

    def update(self, text: str, answer_id: str, weight: float = 1.0) -> None:
        target = self.answer_to_idx[answer_id]
        self.answer_counts[target] += weight
        for feature in self.features(text):
            row = self.tables.setdefault(feature, {})
            row[target] = row.get(target, 0.0) + weight

    def forget(self, text: str, answer_id: str, weight: float = 1.0) -> None:
        target = self.answer_to_idx[answer_id]
        self.answer_counts[target] = max(float(self.answer_counts[target]) - weight, 0.0)
        for feature in self.features(text):
            row = self.tables.get(feature)
            if not row:
                continue
            new_value = float(row.get(target, 0.0)) - weight
            if new_value <= 0.0:
                row.pop(target, None)
            else:
                row[target] = new_value
            if not row:
                self.tables.pop(feature, None)

    def scores(self, text: str) -> np.ndarray:
        name = infer_name(text, self.names)
        attribute = infer_attribute(text)
        if name is not None and attribute is not None:
            feature = stable_hash_int(f"{self.seed}:fact:{name}:{attribute}", bits=self.hash_bits)
            row = self.tables.get(feature)
            if row:
                scores = 0.01 * self.answer_counts.astype(np.float32, copy=True)
                for target, value in row.items():
                    scores[int(target)] += self.specific_feature_boost * float(value)
                return scores

        scores = 0.05 * self.answer_counts.astype(np.float32, copy=True)
        for feature in self.features(text):
            row = self.tables.get(feature)
            if not row:
                continue
            for target, value in row.items():
                scores[int(target)] += float(value)
        return scores

    def predict(self, text: str) -> str:
        scores = self.scores(text)
        attribute = infer_attribute(text)
        if attribute is not None:
            mask = np.array(
                [self.answer_attributes[answer_id] == attribute for answer_id in self.answer_ids],
                dtype=bool,
            )
            if np.any(mask):
                scores = scores.copy()
                scores[~mask] = -1e9
        return self.idx_to_answer[int(np.argmax(scores))]

    def active_features(self) -> int:
        return len(self.tables)

    def state_bytes(self) -> int:
        learned_state = {
            "tables": self.tables,
            "answer_counts": self.answer_counts,
            "hash_bits": self.hash_bits,
            "ngrams": self.ngrams,
            "specific_feature_boost": self.specific_feature_boost,
        }
        return len(pickle.dumps(learned_state, protocol=pickle.HIGHEST_PROTOCOL))


class LexicalRetrievalBaseline:
    """Raw-example retrieval baseline. This is strong but violates no-raw-data."""

    def __init__(self) -> None:
        self.examples: list[tuple[str, str, set[str]]] = []

    def update(self, text: str, answer_id: str) -> None:
        self.examples.append((text, answer_id, token_set(text)))

    def predict(self, text: str) -> str:
        query = token_set(text)
        best_score = -1.0
        best_answer = ""
        for _, answer_id, tokens in self.examples:
            overlap = len(query & tokens)
            denom = len(query | tokens) or 1
            score = overlap / denom
            if score > best_score:
                best_score = score
                best_answer = answer_id
        return best_answer

    def state_bytes(self) -> int:
        return len(pickle.dumps(self.examples, protocol=pickle.HIGHEST_PROTOCOL))


def base_predict(answer_ids: list[str]) -> str:
    return answer_ids[0]


def evaluate(
    facts: list[Fact],
    predictor,
    answer_ids: list[str],
    variant: int,
) -> dict:
    correct = 0
    total = 0
    rows: list[dict] = []
    for fact in facts:
        question = question_for(fact, variant)
        target = fact.answer_id
        if predictor is None:
            pred = base_predict(answer_ids)
        else:
            pred = predictor.predict(question)
        ok = int(pred == target)
        rows.append(
            {
                "question": question,
                "target": target,
                "prediction": pred,
                "correct": ok,
            }
        )
        correct += ok
        total += 1
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total, "rows": rows}


def run_prequential(
    facts: list[Fact],
    memory: RandomProjectionMemory,
    retrieval: LexicalRetrievalBaseline,
    answer_ids: list[str],
    eval_every: int,
) -> tuple[list[dict], list[dict]]:
    stream_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for idx, fact in enumerate(facts):
        before_question = question_for(fact, 1)
        memory_pred_before = memory.predict(before_question)
        retrieval_pred_before = retrieval.predict(before_question) if retrieval.examples else ""
        target = fact.answer_id
        memory.update(learn_sentence(fact), target)
        retrieval.update(learn_sentence(fact), target)
        after_question = question_for(fact, 2)
        memory_pred_after = memory.predict(after_question)
        retrieval_pred_after = retrieval.predict(after_question)
        stream_rows.append(
            {
                "step": idx + 1,
                "name": fact.name,
                "attribute": fact.attribute,
                "target": target,
                "memory_before_correct": int(memory_pred_before == target),
                "memory_after_correct": int(memory_pred_after == target),
                "retrieval_before_correct": int(retrieval_pred_before == target),
                "retrieval_after_correct": int(retrieval_pred_after == target),
                "active_features": memory.active_features(),
                "memory_state_bytes": memory.state_bytes(),
                "retrieval_state_bytes": retrieval.state_bytes(),
            }
        )
        if (idx + 1) % eval_every == 0 or idx + 1 == len(facts):
            seen = facts[: idx + 1]
            memory_eval = evaluate(seen, memory, answer_ids, variant=3)
            retrieval_eval = evaluate(seen, retrieval, answer_ids, variant=3)
            prediction_rows.append(
                {
                    "step": idx + 1,
                    "method": "hashed_memory",
                    "accuracy": memory_eval["accuracy"],
                    "state_bytes": memory.state_bytes(),
                    "active_features": memory.active_features(),
                    "seen_facts": len(seen),
                    "stores_raw_text": False,
                }
            )
            prediction_rows.append(
                {
                    "step": idx + 1,
                    "method": "raw_retrieval",
                    "accuracy": retrieval_eval["accuracy"],
                    "state_bytes": retrieval.state_bytes(),
                    "active_features": len(retrieval.examples),
                    "seen_facts": len(seen),
                    "stores_raw_text": True,
                }
            )
    return stream_rows, prediction_rows


def plot_accuracy(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    methods = list(dict.fromkeys(row["method"] for row in rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        ax.plot(
            [int(row["step"]) for row in method_rows],
            [float(row["accuracy"]) for row in method_rows],
            marker="o",
            label=method,
        )
    ax.set_xlabel("online facts observed")
    ax.set_ylabel("paraphrased QA accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_deletion_audit(
    facts: list[Fact],
    answer_ids: list[str],
    answer_attributes: dict[str, str],
    names: set[str],
    args: argparse.Namespace,
) -> list[dict]:
    if not facts:
        return []
    memory = RandomProjectionMemory(
        answer_ids,
        answer_attributes,
        args.hash_bits,
        args.ngrams,
        names,
        args.seed + 11,
        specific_feature_boost=args.specific_feature_boost,
    )
    for fact in facts:
        memory.update(learn_sentence(fact), fact.answer_id)

    delete_count = max(1, len(facts) // 4)
    deleted = facts[:delete_count]
    retained = facts[delete_count:]
    before_deleted = evaluate(deleted, memory, answer_ids, variant=3)
    before_retained = evaluate(retained, memory, answer_ids, variant=3)
    for fact in deleted:
        memory.forget(learn_sentence(fact), fact.answer_id)
    after_deleted = evaluate(deleted, memory, answer_ids, variant=3)
    after_retained = evaluate(retained, memory, answer_ids, variant=3)
    return [
        {
            "split": "deleted",
            "before_forget_accuracy": before_deleted["accuracy"],
            "after_forget_accuracy": after_deleted["accuracy"],
            "facts": len(deleted),
            "state_bytes_after": memory.state_bytes(),
        },
        {
            "split": "retained",
            "before_forget_accuracy": before_retained["accuracy"],
            "after_forget_accuracy": after_retained["accuracy"],
            "facts": len(retained),
            "state_bytes_after": memory.state_bytes(),
        },
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "online_memory_qa")
    parser.add_argument("--num-people", type=int, default=8)
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--ngrams", type=int, default=2)
    parser.add_argument("--specific-feature-boost", type=float, default=4.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    facts = build_facts(args.num_people, args.seed)
    answer_ids = sorted({fact.answer_id for fact in facts})
    answer_attributes = {answer_id: answer_id.split("=", 1)[0] for answer_id in answer_ids}
    names = {fact.name for fact in facts}
    memory = RandomProjectionMemory(
        answer_ids,
        answer_attributes,
        args.hash_bits,
        args.ngrams,
        names,
        args.seed + 11,
        specific_feature_boost=args.specific_feature_boost,
    )
    retrieval = LexicalRetrievalBaseline()

    base_eval = evaluate(facts, None, answer_ids, variant=3)
    stream_rows, eval_rows = run_prequential(facts, memory, retrieval, answer_ids, args.eval_every)
    final_memory_eval = evaluate(facts, memory, answer_ids, variant=3)
    final_retrieval_eval = evaluate(facts, retrieval, answer_ids, variant=3)
    deletion_rows = run_deletion_audit(facts, answer_ids, answer_attributes, names, args)

    summary_rows = [
        {
            "method": "base_no_memory",
            "accuracy": base_eval["accuracy"],
            "state_bytes": 0,
            "active_features": 0,
            "stores_raw_text": False,
        },
        {
            "method": "hashed_memory",
            "accuracy": final_memory_eval["accuracy"],
            "state_bytes": memory.state_bytes(),
            "active_features": memory.active_features(),
            "stores_raw_text": False,
        },
        {
            "method": "raw_retrieval",
            "accuracy": final_retrieval_eval["accuracy"],
            "state_bytes": retrieval.state_bytes(),
            "active_features": len(retrieval.examples),
            "stores_raw_text": True,
        },
    ]

    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "stream_metrics.csv", stream_rows)
    write_csv(args.out_dir / "eval_curve.csv", eval_rows)
    write_csv(args.out_dir / "deletion_audit.csv", deletion_rows)
    plot_accuracy(eval_rows, args.out_dir / "eval_curve.png")
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    examples = []
    for row in final_memory_eval["rows"][:10]:
        examples.append(
            {
                "question": row["question"],
                "target": row["target"],
                "prediction": row["prediction"],
                "correct": row["correct"],
            }
        )
    with (args.out_dir / "examples.json").open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    print("Summary:")
    for row in summary_rows:
        print(
            f"  {row['method']}: acc={row['accuracy']:.3f} "
            f"bytes={row['state_bytes']:,} raw={row['stores_raw_text']}"
        )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")
    print(f"wrote eval curve: {args.out_dir / 'eval_curve.csv'}")
    print(f"wrote deletion audit: {args.out_dir / 'deletion_audit.csv'}")
    print(f"wrote examples: {args.out_dir / 'examples.json'}")


if __name__ == "__main__":
    main()
