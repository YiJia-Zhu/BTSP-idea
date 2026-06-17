#!/usr/bin/env python3
"""
OpenAI-compatible API demo for the no-raw-data QA adapter.

This script keeps the learned state local and uses the API only as a frozen
reasoning layer. It compares:
  - local hashed memory
  - local raw retrieval upper bound
  - optional OpenAI-compatible API with and without memory hints

By default it performs a dry run and writes request payloads. Set --run-api to
send requests to the configured endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import requests

import online_memory_qa_experiment as base


SCRIPT_DIR = Path(__file__).resolve().parent


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


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def candidate_values(attribute: str) -> list[str]:
    return list(base.ATTRIBUTES[attribute])


def memory_hint(memory: base.RandomProjectionMemory, question: str, attribute: str, top_k: int) -> list[dict[str, Any]]:
    scores = memory.scores(question)
    rows = []
    for value in candidate_values(attribute):
        answer_id = f"{attribute}={value}"
        idx = memory.answer_to_idx.get(answer_id)
        if idx is None:
            continue
        rows.append({"value": value, "score": float(scores[idx]), "answer_id": answer_id})
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:top_k]


def build_messages(question: str, attribute: str, candidates: list[str], hint: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    system = (
        "You answer a personalization question using the provided candidates. "
        "Reply with JSON only: {\"answer\":\"...\"}. "
        "Choose exactly one candidate value."
    )
    user_lines = [
        f"Question: {question}",
        f"Attribute: {attribute}",
        f"Candidate values: {', '.join(candidates)}",
    ]
    if hint:
        hint_text = ", ".join(f"{row['value']}={row['score']:.3f}" for row in hint)
        user_lines.append(f"Memory hint: {hint_text}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def parse_answer_text(text: str, attribute: str) -> str:
    norm = base.normalize(text)
    for value in candidate_values(attribute):
        if base.normalize(value) and base.normalize(value) in norm:
            return value
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
        for value in candidate_values(attribute):
            if base.normalize(value) == base.normalize(raw):
                return value
    if candidate_values(attribute):
        return candidate_values(attribute)[0]
    return ""


def api_predict(
    client: OpenAICompatibleClient | None,
    question: str,
    attribute: str,
    hint: list[dict[str, Any]] | None,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    messages = build_messages(question, attribute, candidate_values(attribute), hint)
    request_payload = {
        "model": client.model if client is not None else None,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if client is None:
        return "", request_payload
    response = client.chat(messages, max_tokens=max_tokens, temperature=temperature)
    raw_text = response["choices"][0]["message"]["content"]
    return parse_answer_text(raw_text, attribute), {
        "request": request_payload,
        "response": response,
        "raw_text": raw_text,
    }


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "online_memory_qa_api")
    parser.add_argument("--num-people", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--ngrams", type=int, default=2)
    parser.add_argument("--specific-feature-boost", type=float, default=4.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--api-limit", type=int, default=6)
    parser.add_argument("--api-hint-top-k", type=int, default=3)
    parser.add_argument("--api-base-url", type=str, default="https://yzhanghmeng.com/v1")
    parser.add_argument("--api-model", type=str, default="gpt-5.5")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--api-max-tokens", type=int, default=24)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-key-env", type=str, default="API_KEY")
    parser.add_argument("--run-api", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    facts = base.build_facts(args.num_people, args.seed)
    answer_ids = sorted({fact.answer_id for fact in facts})
    answer_attributes = {answer_id: answer_id.split("=", 1)[0] for answer_id in answer_ids}
    names = {fact.name for fact in facts}

    memory = base.RandomProjectionMemory(
        answer_ids,
        answer_attributes,
        args.hash_bits,
        args.ngrams,
        names,
        args.seed + 11,
        specific_feature_boost=args.specific_feature_boost,
    )
    retrieval = base.LexicalRetrievalBaseline()
    for fact in facts:
        memory.update(base.learn_sentence(fact), fact.answer_id)
        retrieval.update(base.learn_sentence(fact), fact.answer_id)

    local_base_eval = base.evaluate(facts, None, answer_ids, variant=3)
    local_memory_eval = base.evaluate(facts, memory, answer_ids, variant=3)
    local_retrieval_eval = base.evaluate(facts, retrieval, answer_ids, variant=3)

    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    client = None
    if args.run_api:
        if not api_key:
            raise RuntimeError(f"API key not found in ${args.api_key_env} or $OPENAI_API_KEY")
        client = OpenAICompatibleClient(args.api_base_url, api_key, args.api_model, args.api_timeout)

    api_rows: list[dict] = []
    api_requests: list[dict] = []
    api_responses: list[dict] = []
    api_eval_limit = min(args.api_limit, len(facts))
    for idx, fact in enumerate(facts[:api_eval_limit]):
        question = base.question_for(fact, 3)
        target_value = fact.value
        target_answer_id = fact.answer_id
        local_memory_answer_id = memory.predict(question)
        local_retrieval_answer_id = retrieval.predict(question)
        hint = memory_hint(memory, question, fact.attribute, args.api_hint_top_k)
        if client is None:
            api_no_memory_pred, api_no_memory_payload = api_predict(None, question, fact.attribute, None, args.api_max_tokens, args.api_temperature)
            api_memory_pred, api_memory_payload = api_predict(None, question, fact.attribute, hint, args.api_max_tokens, args.api_temperature)
            api_requests.append(
                {
                    "step": idx + 1,
                    "mode": "api_no_memory",
                    "payload": api_no_memory_payload,
                }
            )
            api_requests.append(
                {
                    "step": idx + 1,
                    "mode": "api_memory_hint",
                    "payload": api_memory_payload,
                }
            )
            api_rows.append(
                {
                    "step": idx + 1,
                    "question": question,
                    "target": target_value,
                    "local_memory_prediction": local_memory_answer_id,
                    "local_memory_correct": int(local_memory_answer_id == target_answer_id),
                    "local_retrieval_prediction": local_retrieval_answer_id,
                    "local_retrieval_correct": int(local_retrieval_answer_id == target_answer_id),
                    "api_no_memory_prediction": "",
                    "api_no_memory_correct": "",
                    "api_memory_prediction": "",
                    "api_memory_correct": "",
                    "memory_hint_top": "; ".join(f"{row['value']}={row['score']:.3f}" for row in hint),
                    "stores_raw_text": False,
                }
            )
            continue

        api_no_memory_pred, api_no_memory_meta = api_predict(client, question, fact.attribute, None, args.api_max_tokens, args.api_temperature)
        api_memory_pred, api_memory_meta = api_predict(client, question, fact.attribute, hint, args.api_max_tokens, args.api_temperature)
        api_requests.append({"step": idx + 1, "mode": "api_no_memory", "payload": api_no_memory_meta["request"]})
        api_requests.append({"step": idx + 1, "mode": "api_memory_hint", "payload": api_memory_meta["request"]})
        api_responses.append({"step": idx + 1, "mode": "api_no_memory", **api_no_memory_meta})
        api_responses.append({"step": idx + 1, "mode": "api_memory_hint", **api_memory_meta})
        api_rows.append(
            {
                "step": idx + 1,
                "question": question,
                "target": target_value,
                "local_memory_prediction": local_memory_answer_id,
                "local_memory_correct": int(local_memory_answer_id == target_answer_id),
                "local_retrieval_prediction": local_retrieval_answer_id,
                "local_retrieval_correct": int(local_retrieval_answer_id == target_answer_id),
                "api_no_memory_prediction": api_no_memory_pred,
                "api_no_memory_correct": int(api_no_memory_pred == target_value),
                "api_memory_prediction": api_memory_pred,
                "api_memory_correct": int(api_memory_pred == target_value),
                "memory_hint_top": "; ".join(f"{row['value']}={row['score']:.3f}" for row in hint),
                "stores_raw_text": False,
            }
        )

    summary_rows = [
        {
            "method": "local_base_no_memory",
            "accuracy": local_base_eval["accuracy"],
            "state_bytes": 0,
            "stores_raw_text": False,
        },
        {
            "method": "local_hashed_memory",
            "accuracy": local_memory_eval["accuracy"],
            "state_bytes": memory.state_bytes(),
            "stores_raw_text": False,
        },
        {
            "method": "local_raw_retrieval",
            "accuracy": local_retrieval_eval["accuracy"],
            "state_bytes": retrieval.state_bytes(),
            "stores_raw_text": True,
        },
    ]

    if client is not None and api_rows and "api_memory_correct" in api_rows[0]:
        api_memory_acc = sum(int(row["api_memory_correct"]) for row in api_rows) / max(len(api_rows), 1)
        api_no_memory_acc = sum(int(row["api_no_memory_correct"]) for row in api_rows) / max(len(api_rows), 1)
        summary_rows.extend(
            [
                {
                    "method": "api_no_memory",
                    "accuracy": api_no_memory_acc,
                    "state_bytes": 0,
                    "stores_raw_text": False,
                },
                {
                    "method": "api_memory_hint",
                    "accuracy": api_memory_acc,
                    "state_bytes": memory.state_bytes(),
                    "stores_raw_text": False,
                },
            ]
        )

    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(args.out_dir / "api_eval.csv", api_rows)
    with (args.out_dir / "api_requests.jsonl").open("w", encoding="utf-8") as f:
        for row in api_requests:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if api_responses:
        with (args.out_dir / "api_responses.jsonl").open("w", encoding="utf-8") as f:
            for row in api_responses:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if client is not None:
        running_rows: list[dict] = []
        no_mem_correct = 0
        mem_correct = 0
        for i, row in enumerate(api_rows):
            no_mem_correct += int(row["api_no_memory_correct"])
            mem_correct += int(row["api_memory_correct"])
            running_rows.append({"step": i + 1, "method": "api_no_memory", "accuracy": no_mem_correct / (i + 1)})
            running_rows.append({"step": i + 1, "method": "api_memory_hint", "accuracy": mem_correct / (i + 1)})
        plot_accuracy(running_rows, args.out_dir / "api_eval_curve.png")
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("Summary:")
    for row in summary_rows:
        print(
            f"  {row['method']}: acc={row['accuracy']:.3f} "
            f"bytes={row['state_bytes']:,} raw={row['stores_raw_text']}"
        )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")
    print(f"wrote api eval: {args.out_dir / 'api_eval.csv'}")
    print(f"wrote api requests: {args.out_dir / 'api_requests.jsonl'}")
    if api_responses:
        print(f"wrote api responses: {args.out_dir / 'api_responses.jsonl'}")


if __name__ == "__main__":
    main()
