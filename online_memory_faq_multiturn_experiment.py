#!/usr/bin/env python3
"""
Multi-turn FAQ session benchmark for no-raw-example online memory.

The session interleaves learn, query, revise, delete, and later query turns.
It compares:

- API no-memory baseline.
- Raw retrieval baseline that keeps the current answer text.
- Semantic sketch memory that stores hashed routing state plus structured values.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from online_memory_faq_api_experiment import (
    FaqFact,
    FaqMemory,
    OpenAICompatibleClient,
    answer_is_correct,
    api_answer,
    build_generated_facts,
    eval_questions,
    revised_fact,
    revision_text_for_fact,
    train_text_for_fact,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class SessionTurn:
    phase: str
    action: str
    fact_idx: int
    fact: FaqFact
    question: str = ""
    expected_unknown: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "online_memory_faq_multiturn")
    parser.add_argument("--fact-limit", type=int, default=16)
    parser.add_argument("--learn-count", type=int, default=8)
    parser.add_argument("--revision-count", type=int, default=4)
    parser.add_argument("--delete-count", type=int, default=2)
    parser.add_argument("--train-style", choices=["canonical", "dialogue"], default="dialogue")
    parser.add_argument("--eval-style", choices=["default", "paraphrase"], default="paraphrase")
    parser.add_argument("--router", choices=["semantic", "hybrid"], default="semantic")
    parser.add_argument("--semantic-dim", type=int, default=0)
    parser.add_argument("--semantic-feature-cap", type=int, default=12)
    parser.add_argument("--answer-store", choices=["sketch", "full"], default="sketch")
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--ngrams", type=int, default=2)
    parser.add_argument("--intent-boost", type=float, default=4.0)
    parser.add_argument("--api-hint-top-k", type=int, default=1)
    parser.add_argument("--session-api-limit", type=int, default=0)
    parser.add_argument("--api-base-url", type=str, default="https://yzhanghmeng.com/v1")
    parser.add_argument("--api-model", type=str, default="gpt-5.5")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--api-max-tokens", type=int, default=56)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-key-env", type=str, default="API_KEY")
    parser.add_argument("--run-api", action="store_true")
    return parser


def session_semantic_texts(fact: FaqFact, train_style: str) -> list[str]:
    texts = [train_text_for_fact(fact, train_style)]
    texts.extend(fact.questions)
    return list(dict.fromkeys(texts))


def remember_fact(memory: FaqMemory, fact: FaqFact, train_style: str) -> None:
    memory.update(train_text_for_fact(fact, train_style), fact.intent)
    for semantic_text in session_semantic_texts(fact, train_style):
        memory.observe_semantic(semantic_text, fact.intent)


def overwrite_fact(memory: FaqMemory, fact: FaqFact, train_style: str) -> None:
    memory.overwrite(revision_text_for_fact(fact, train_style), fact)
    for semantic_text in session_semantic_texts(fact, train_style):
        memory.observe_semantic(semantic_text, fact.intent)


def build_session(initial_facts: list[FaqFact], revised_facts: dict[int, FaqFact], args) -> list[SessionTurn]:
    learn_count = min(args.learn_count, len(initial_facts))
    revision_count = min(args.revision_count, learn_count)
    delete_count = min(args.delete_count, revision_count)
    turns: list[SessionTurn] = []

    for idx in range(learn_count):
        turns.append(SessionTurn("learn", "learn", idx, initial_facts[idx]))

    for idx in range(revision_count):
        question = eval_questions(initial_facts[idx], args.eval_style)[0]
        turns.append(SessionTurn("pre_revision_query", "query", idx, initial_facts[idx], question))

    for idx in range(revision_count):
        turns.append(SessionTurn("revise", "revise", idx, revised_facts[idx]))

    for idx in range(revision_count):
        question = eval_questions(revised_facts[idx], args.eval_style)[1]
        turns.append(SessionTurn("post_revision_query", "query", idx, revised_facts[idx], question))

    for idx in range(delete_count):
        turns.append(SessionTurn("delete", "delete", idx, revised_facts[idx]))

    for idx in range(delete_count):
        question = eval_questions(revised_facts[idx], args.eval_style)[0]
        turns.append(SessionTurn("post_delete_query", "query", idx, revised_facts[idx], question, expected_unknown=True))

    for idx in range(revision_count, learn_count):
        question = eval_questions(initial_facts[idx], args.eval_style)[idx % 2]
        turns.append(SessionTurn("retained_query", "query", idx, initial_facts[idx], question))

    return turns


def raw_state_bytes(active_raw_records: dict[str, dict[str, str]]) -> int:
    return len(pickle.dumps(active_raw_records, protocol=pickle.HIGHEST_PROTOCOL))


def raw_hint(active_raw_records: dict[str, dict[str, str]], fact: FaqFact) -> list[dict[str, Any]]:
    record = active_raw_records.get(fact.intent)
    answer = None if record is None else record["answer"]
    if answer is None:
        return []
    return [{"intent": fact.intent, "score": 1.0, "answer": answer}]


def first_hint_answer(hint: list[dict[str, Any]]) -> str:
    if not hint:
        return ""
    return str(hint[0]["answer"])


def is_query_correct(answer: str, fact: FaqFact, expected_unknown: bool) -> bool:
    if expected_unknown:
        return not answer_is_correct(answer, fact)
    return answer_is_correct(answer, fact)


def summarize(
    rows: list[dict[str, Any]],
    memory: FaqMemory,
    api_limit: int,
    raw_bytes: int,
) -> list[dict[str, Any]]:
    query_rows = [row for row in rows if row["action"] == "query"]

    def mean(key: str, subset: list[dict[str, Any]]) -> float:
        values = [float(row[key]) for row in subset if row.get(key) != ""]
        return sum(values) / max(len(values), 1)

    summary = [
        {
            "method": "local_raw_retrieval",
            "accuracy": mean("raw_local_correct", query_rows),
            "state_bytes": raw_bytes,
            "stores_raw_examples": True,
            "stores_answer_values": True,
            "stores_answer_text": True,
            "query_count": len(query_rows),
        },
        {
            "method": "local_semantic_sketch_memory",
            "accuracy": mean("memory_local_correct", query_rows),
            "state_bytes": memory.state_bytes(),
            "stores_raw_examples": False,
            "stores_answer_values": True,
            "stores_answer_text": memory.answer_store == "full",
            "query_count": len(query_rows),
        },
    ]

    api_rows = [row for row in query_rows if row.get("api_called")]
    if api_rows:
        summary.extend(
            [
                {
                    "method": "api_no_memory",
                    "accuracy": mean("api_no_memory_correct", api_rows),
                    "state_bytes": 0,
                    "stores_raw_examples": False,
                    "stores_answer_values": False,
                    "stores_answer_text": False,
                    "query_count": len(api_rows),
                },
                {
                    "method": "api_raw_retrieval",
                    "accuracy": mean("api_raw_retrieval_correct", api_rows),
                    "state_bytes": raw_bytes,
                    "stores_raw_examples": True,
                    "stores_answer_values": True,
                    "stores_answer_text": True,
                    "query_count": len(api_rows),
                },
                {
                    "method": "api_semantic_sketch_memory",
                    "accuracy": mean("api_memory_correct", api_rows),
                    "state_bytes": memory.state_bytes(),
                    "stores_raw_examples": False,
                    "stores_answer_values": True,
                    "stores_answer_text": memory.answer_store == "full",
                    "query_count": len(api_rows),
                },
            ]
        )
    else:
        summary.append(
            {
                "method": "api_not_run",
                "accuracy": "",
                "state_bytes": memory.state_bytes(),
                "stores_raw_examples": False,
                "stores_answer_values": True,
                "stores_answer_text": memory.answer_store == "full",
                "query_count": api_limit,
            }
        )
    return summary


def write_transcript(path: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    query_rows = [row for row in rows if row["action"] == "query"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Multi-turn FAQ Session Transcript\n\n")
        f.write("## Summary\n\n")
        f.write("| method | accuracy | query_count | state_bytes | raw_examples | answer_text |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['method']} | {row['accuracy']} | {row['query_count']} | "
                f"{row['state_bytes']} | {row['stores_raw_examples']} | {row['stores_answer_text']} |\n"
            )
        f.write("\n## Query Turns\n\n")
        for row in query_rows:
            expected = "UNKNOWN" if row["expected_unknown"] else row["expected_answer"]
            f.write(f"### Turn {row['turn']} - {row['phase']} - {row['intent']}\n\n")
            f.write(f"Question: {row['question']}\n\n")
            f.write(f"Expected: {expected}\n\n")
            f.write(f"Raw retrieval hint: {row['raw_hint_answer'] or '<none>'}\n\n")
            f.write(f"Semantic sketch hint: {row['memory_hint_answer'] or '<none>'}\n\n")
            if row.get("api_called"):
                f.write(f"No-memory API: {row['api_no_memory_answer']}\n\n")
                f.write(f"Raw-retrieval API: {row['api_raw_retrieval_answer']}\n\n")
                f.write(f"Semantic-sketch API: {row['api_memory_answer']}\n\n")
            f.write(
                "Correct: "
                f"raw_local={row['raw_local_correct']}, "
                f"memory_local={row['memory_local_correct']}, "
                f"api_no_memory={row['api_no_memory_correct']}, "
                f"api_raw={row['api_raw_retrieval_correct']}, "
                f"api_memory={row['api_memory_correct']}\n\n"
            )


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    facts = build_generated_facts(args.fact_limit)
    learn_count = min(args.learn_count, len(facts))
    revised_facts = {idx: revised_fact(facts[idx], idx) for idx in range(min(args.revision_count, learn_count))}
    memory = FaqMemory(
        facts,
        args.hash_bits,
        args.ngrams,
        args.intent_boost,
        args.router,
        args.semantic_dim,
        args.semantic_feature_cap,
        args.answer_store,
    )
    active_raw_records: dict[str, dict[str, str]] = {}
    turns = build_session(facts, revised_facts, args)

    client = None
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if args.run_api:
        if not api_key:
            raise RuntimeError(f"API key not found in ${args.api_key_env} or $OPENAI_API_KEY")
        client = OpenAICompatibleClient(args.api_base_url, api_key, args.api_model, args.api_timeout)

    api_args = SimpleNamespace(
        api_model=args.api_model,
        api_temperature=args.api_temperature,
        api_max_tokens=args.api_max_tokens,
    )
    rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    api_query_count = 0

    for turn_idx, turn in enumerate(turns, start=1):
        if turn.action == "learn":
            event_text = train_text_for_fact(turn.fact, args.train_style)
            remember_fact(memory, turn.fact, args.train_style)
            active_raw_records[turn.fact.intent] = {"event_text": event_text, "answer": turn.fact.answer}
            rows.append(
                {
                    "turn": turn_idx,
                    "phase": turn.phase,
                    "action": turn.action,
                    "intent": turn.fact.intent,
                    "event_text": event_text,
                }
            )
            continue

        if turn.action == "revise":
            event_text = revision_text_for_fact(turn.fact, args.train_style)
            overwrite_fact(memory, turn.fact, args.train_style)
            active_raw_records[turn.fact.intent] = {"event_text": event_text, "answer": turn.fact.answer}
            rows.append(
                {
                    "turn": turn_idx,
                    "phase": turn.phase,
                    "action": turn.action,
                    "intent": turn.fact.intent,
                    "event_text": event_text,
                }
            )
            continue

        if turn.action == "delete":
            memory.forget(turn.fact.intent)
            active_raw_records.pop(turn.fact.intent, None)
            rows.append(
                {
                    "turn": turn_idx,
                    "phase": turn.phase,
                    "action": turn.action,
                    "intent": turn.fact.intent,
                    "event_text": f"forget {turn.fact.intent}",
                }
            )
            continue

        memory_hint = memory.hint(turn.question, args.api_hint_top_k)
        raw_retrieval_hint = raw_hint(active_raw_records, turn.fact)
        memory_answer = first_hint_answer(memory_hint)
        raw_answer = first_hint_answer(raw_retrieval_hint)
        raw_local_correct = int(is_query_correct(raw_answer, turn.fact, turn.expected_unknown))
        memory_local_correct = int(is_query_correct(memory_answer, turn.fact, turn.expected_unknown))

        api_called = args.run_api and api_query_count < args.session_api_limit
        api_no_memory_answer = ""
        api_raw_answer = ""
        api_memory_answer = ""
        api_no_memory_correct: int | str = ""
        api_raw_correct: int | str = ""
        api_memory_correct: int | str = ""

        if api_called:
            api_query_count += 1
            no_memory_text, no_memory_meta = api_answer(client, turn.question, None, api_args)
            raw_text, raw_meta = api_answer(client, turn.question, raw_retrieval_hint, api_args)
            memory_text, memory_meta = api_answer(client, turn.question, memory_hint, api_args)
            api_no_memory_answer = no_memory_text
            api_raw_answer = raw_text
            api_memory_answer = memory_text
            api_no_memory_correct = int(is_query_correct(no_memory_text, turn.fact, turn.expected_unknown))
            api_raw_correct = int(is_query_correct(raw_text, turn.fact, turn.expected_unknown))
            api_memory_correct = int(is_query_correct(memory_text, turn.fact, turn.expected_unknown))
            request_rows.extend(
                [
                    {"turn": turn_idx, "mode": "api_no_memory", "payload": no_memory_meta["request"]},
                    {"turn": turn_idx, "mode": "api_raw_retrieval", "payload": raw_meta["request"]},
                    {"turn": turn_idx, "mode": "api_semantic_sketch_memory", "payload": memory_meta["request"]},
                ]
            )
            response_rows.extend(
                [
                    {"turn": turn_idx, "mode": "api_no_memory", **no_memory_meta},
                    {"turn": turn_idx, "mode": "api_raw_retrieval", **raw_meta},
                    {"turn": turn_idx, "mode": "api_semantic_sketch_memory", **memory_meta},
                ]
            )

        rows.append(
            {
                "turn": turn_idx,
                "phase": turn.phase,
                "action": turn.action,
                "intent": turn.fact.intent,
                "question": turn.question,
                "expected_answer": turn.fact.answer,
                "expected_unknown": int(turn.expected_unknown),
                "raw_hint_answer": raw_answer,
                "memory_hint_answer": memory_answer,
                "memory_hint": " | ".join(f"{row['intent']}={row['score']:.3f}" for row in memory_hint),
                "raw_local_correct": raw_local_correct,
                "memory_local_correct": memory_local_correct,
                "api_called": int(api_called),
                "api_no_memory_answer": api_no_memory_answer,
                "api_no_memory_correct": api_no_memory_correct,
                "api_raw_retrieval_answer": api_raw_answer,
                "api_raw_retrieval_correct": api_raw_correct,
                "api_memory_answer": api_memory_answer,
                "api_memory_correct": api_memory_correct,
            }
        )

    summary_rows = summarize(rows, memory, args.session_api_limit, raw_state_bytes(active_raw_records))
    write_csv(args.out_dir / "session_turns.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    with (args.out_dir / "api_requests.jsonl").open("w", encoding="utf-8") as f:
        for row in request_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if response_rows:
        with (args.out_dir / "api_responses.jsonl").open("w", encoding="utf-8") as f:
            for row in response_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_transcript(args.out_dir / "session_transcript.md", rows, summary_rows)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("Summary:")
    for row in summary_rows:
        acc = row["accuracy"]
        acc_text = f"{acc:.3f}" if isinstance(acc, float) else str(acc)
        print(
            f"  {row['method']}: acc={acc_text} queries={row['query_count']} "
            f"bytes={row['state_bytes']} raw_examples={row['stores_raw_examples']} "
            f"answer_text={row['stores_answer_text']}"
        )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")
    print(f"wrote transcript: {args.out_dir / 'session_transcript.md'}")


if __name__ == "__main__":
    main()
