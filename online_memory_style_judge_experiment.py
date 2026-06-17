#!/usr/bin/env python3
"""
Pairwise judge for the personalized style API benchmark.

This reuses previously generated responses and asks the API to rank the three
variants for each prompt:

- no-memory API
- raw profile baseline
- no-raw style sketch memory

The goal is not another generation benchmark. It is a compact proxy for the
"naturalness / usefulness" part of the GPT-like final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from online_memory_faq_api_experiment import OpenAICompatibleClient


SCRIPT_DIR = Path(__file__).resolve().parent


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


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match is None:
            raise
        return json.loads(match.group(0))


@dataclass(frozen=True)
class JudgeCase:
    turn: int
    profile_id: str
    prompt: str
    raw_hint: str
    memory_hint: str
    no_memory: str
    raw_profile: str
    style_sketch: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SCRIPT_DIR / "output" / "online_memory_style_delete_api")
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "online_memory_style_judge_api")
    parser.add_argument("--reuse-dir", type=Path, default=None)
    parser.add_argument("--case-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--api-base-url", type=str, default="https://yzhanghmeng.com/v1")
    parser.add_argument("--api-model", type=str, default="gpt-5.5")
    parser.add_argument("--api-timeout", type=float, default=90.0)
    parser.add_argument("--api-max-tokens", type=int, default=160)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-key-env", type=str, default="API_KEY")
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--api-retry-sleep", type=float, default=2.0)
    parser.add_argument("--judge-context", choices=["request_only", "style_memory", "raw_profile"], default="request_only")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--run-api", action="store_true")
    return parser


def load_cases(source_dir: Path, case_limit: int) -> list[JudgeCase]:
    csv_path = source_dir / "session_turns.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cases: list[JudgeCase] = []
    for row in rows:
        if row.get("score_phase") != "active":
            continue
        if row.get("api_called") not in {"1", "true", "True"}:
            continue
        cases.append(
            JudgeCase(
                turn=int(row["turn"]),
                profile_id=row["profile_id"],
                prompt=row["prompt"],
                raw_hint=row.get("raw_hint", ""),
                memory_hint=row.get("memory_hint", ""),
                no_memory=row["api_no_memory_answer"],
                raw_profile=row["api_raw_profile_answer"],
                style_sketch=row["api_memory_answer"],
            )
        )
    if case_limit > 0:
        cases = cases[:case_limit]
    return cases


def build_messages(case: JudgeCase, candidates: dict[str, str], judge_context: str) -> list[dict[str, str]]:
    system = (
        "You are a strict but practical customer-support writing judge. "
        "Rank the candidates by overall quality for the given request. "
        "If a learned style preference is provided, treat it as part of the user's request. "
        "Consider naturalness, usefulness, and how well the reply follows the applicable style constraints. "
        "Do not reward verbosity. Return only valid JSON with keys best, second, third, rationale."
    )
    context = ""
    if judge_context == "style_memory" and case.memory_hint:
        context = f"\nLearned style preference:\n{case.memory_hint}\n"
    elif judge_context == "raw_profile" and case.raw_hint:
        context = f"\nLearned style preference:\n{case.raw_hint}\n"
    user = (
        f"Request:\n{case.prompt}\n"
        f"{context}\n"
        f"Candidate A:\n{candidates['A']}\n\n"
        f"Candidate B:\n{candidates['B']}\n\n"
        f"Candidate C:\n{candidates['C']}\n\n"
        "Return JSON of the form:\n"
        '{"best":"A","second":"B","third":"C","rationale":"..."}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_case(client: OpenAICompatibleClient | None, case: JudgeCase, args, rng: random.Random) -> dict[str, Any]:
    labels = ["A", "B", "C"]
    mapping = [
        ("no_memory", case.no_memory),
        ("raw_profile", case.raw_profile),
        ("style_sketch", case.style_sketch),
    ]
    rng.shuffle(mapping)
    labeled = {label: text for label, (_, text) in zip(labels, mapping)}
    label_to_method = {label: method for label, (method, _) in zip(labels, mapping)}
    messages = build_messages(case, labeled, args.judge_context)
    payload = {"model": args.api_model, "messages": messages, "temperature": args.api_temperature, "max_tokens": args.api_max_tokens}
    if client is None:
        return {
            "turn": case.turn,
            "profile_id": case.profile_id,
            "judge_context": args.judge_context,
            "candidate_order": json.dumps({label: method for label, (method, _) in zip(labels, mapping)}),
            "request": payload,
            "response": "",
            "best": "",
            "second": "",
            "third": "",
            "rationale": "",
        }
    last_error: Exception | None = None
    for attempt in range(args.api_retries + 1):
        try:
            response = client.chat(messages, args.api_max_tokens, args.api_temperature)
            break
        except Exception as exc:
            last_error = exc
            if attempt >= args.api_retries:
                raise
            time.sleep(args.api_retry_sleep * (attempt + 1))
    else:
        raise RuntimeError("judge request failed") from last_error
    text = response["choices"][0]["message"]["content"]
    parsed = extract_json(text)
    best = str(parsed.get("best", ""))
    second = str(parsed.get("second", ""))
    third = str(parsed.get("third", ""))
    return {
        "turn": case.turn,
        "profile_id": case.profile_id,
        "judge_context": args.judge_context,
        "candidate_order": json.dumps({label: method for label, (method, _) in zip(labels, mapping)}),
        "request": payload,
        "response": response,
        "raw_text": text,
        "best": label_to_method.get(best, ""),
        "second": label_to_method.get(second, ""),
        "third": label_to_method.get(third, ""),
        "rationale": str(parsed.get("rationale", "")),
    }


def parse_saved_judges(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("raw_text"):
            continue
        parsed = extract_json(row["raw_text"])
        candidate_order = json.loads(row["candidate_order"])
        parsed_rows.append(
            {
                **row,
                "judge_context": row.get("judge_context", ""),
                "best": candidate_order.get(str(parsed.get("best", "")), ""),
                "second": candidate_order.get(str(parsed.get("second", "")), ""),
                "third": candidate_order.get(str(parsed.get("third", "")), ""),
                "rationale": str(parsed.get("rationale", "")),
            }
        )
    return parsed_rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if row.get(key) != ""]
        return sum(values) / max(len(values), 1)

    return [
        {"metric": "cases", "value": len(rows)},
        {"metric": "no_memory_best_rate", "value": mean("no_memory_best")},
        {"metric": "raw_profile_best_rate", "value": mean("raw_profile_best")},
        {"metric": "style_sketch_best_rate", "value": mean("style_sketch_best")},
        {"metric": "style_sketch_beats_no_memory_rate", "value": mean("style_sketch_beats_no_memory")},
        {"metric": "style_sketch_beats_raw_profile_rate", "value": mean("style_sketch_beats_raw_profile")},
        {"metric": "raw_profile_beats_no_memory_rate", "value": mean("raw_profile_beats_no_memory")},
        {"metric": "judge_context", "value": rows[0].get("judge_context", "")},
        {"metric": "judge_model", "value": rows[0].get("judge_model", "")},
    ]


def write_transcript(path: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Personalized Style Judge\n\n")
        f.write("## Summary\n\n")
        f.write("| metric | value |\n")
        f.write("|---|---:|\n")
        for row in summary_rows:
            f.write(f"| {row['metric']} | {row['value']} |\n")
        f.write("\n## Cases\n\n")
        for row in rows:
            f.write(f"### Turn {row['turn']} - {row['profile_id']}\n\n")
            f.write(f"Order: {row['candidate_order']}\n\n")
            f.write(f"Best: {row['best']}\n\n")
            f.write(f"Second: {row['second']}\n\n")
            f.write(f"Third: {row['third']}\n\n")
            f.write(f"Rationale: {row['rationale']}\n\n")


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.source_dir, args.case_limit)

    client = None
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if args.run_api:
        if not api_key:
            raise RuntimeError(f"API key not found in ${args.api_key_env} or $OPENAI_API_KEY")
        client = OpenAICompatibleClient(args.api_base_url, api_key, args.api_model, args.api_timeout)

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []

    if args.reuse_existing:
        reuse_dir = args.reuse_dir or args.out_dir
        rows = parse_saved_judges(reuse_dir / "judge_results.csv")
    else:
        for case in cases:
            row = judge_case(client, case, args, rng)
            if client is not None:
                request_rows.append({"turn": case.turn, "profile_id": case.profile_id, "request": row["request"]})
                response_rows.append({"turn": case.turn, "profile_id": case.profile_id, "response": row["response"], "raw_text": row["raw_text"]})
            rows.append(row)

    for row in rows:
        row["no_memory_best"] = int(row.get("best") == "no_memory")
        row["raw_profile_best"] = int(row.get("best") == "raw_profile")
        row["style_sketch_best"] = int(row.get("best") == "style_sketch")
        rank = {row.get("best"): 1, row.get("second"): 2, row.get("third"): 3}
        row["no_memory_rank"] = rank.get("no_memory", "")
        row["raw_profile_rank"] = rank.get("raw_profile", "")
        row["style_sketch_rank"] = rank.get("style_sketch", "")
        row["style_sketch_beats_no_memory"] = int(row["style_sketch_rank"] < row["no_memory_rank"])
        row["style_sketch_beats_raw_profile"] = int(row["style_sketch_rank"] < row["raw_profile_rank"])
        row["raw_profile_beats_no_memory"] = int(row["raw_profile_rank"] < row["no_memory_rank"])
        row["judge_model"] = args.api_model

    summary_rows = summarize(rows)
    write_csv(args.out_dir / "judge_results.csv", rows)
    write_csv(args.out_dir / "judge_summary.csv", summary_rows)
    with (args.out_dir / "judge_requests.jsonl").open("w", encoding="utf-8") as f:
        for row in request_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if response_rows:
        with (args.out_dir / "judge_responses.jsonl").open("w", encoding="utf-8") as f:
            for row in response_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_transcript(args.out_dir / "judge_transcript.md", rows, summary_rows)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("Summary:")
    for row in summary_rows:
        print(f"  {row['metric']}: {row['value']}")
    print(f"wrote summary: {args.out_dir / 'judge_summary.csv'}")
    print(f"wrote transcript: {args.out_dir / 'judge_transcript.md'}")


if __name__ == "__main__":
    main()
