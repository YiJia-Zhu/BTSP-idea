#!/usr/bin/env python3
"""
Personalized writing-style API benchmark for no-raw-example online memory.

FAQ experiments test exact fact recall. This benchmark tests a more GPT-like
surface: the API must generate natural short writing while obeying user-specific
style/preferences learned earlier in the session. The sketch memory stores
hashed routing statistics plus compact preference codes, not raw instructions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from online_memory_faq_api_experiment import OpenAICompatibleClient


SCRIPT_DIR = Path(__file__).resolve().parent


FORMAT_TEXT = {
    "three_bullets": "Use exactly three bullet points.",
    "two_bullets": "Use exactly two bullet points.",
    "subject_body": "Include one line starting with 'Subject:' and one line starting with 'Body:'.",
}

SIGNOFF_TEXT = {
    "mira": "Thanks, Mira.",
    "jules": "Regards, Jules.",
    "noor": "Best, Noor.",
    "ren": "Warmly, Ren.",
    "sol": "Cheers, Sol.",
    "talia": "Thank you, Talia.",
}

REQUIRED_TEXT = {
    "next_step": "Next step:",
    "owner": "Owner:",
    "eta": "ETA:",
    "why": "Why:",
    "action": "Action:",
    "note": "Note:",
}

AVOID_TEXT = {
    "urgent": "urgent",
    "guarantee": "guarantee",
    "cheap": "cheap",
    "sorry": "sorry",
    "asap": "asap",
    "perfect": "perfect",
}


@dataclass(frozen=True)
class StyleProfile:
    profile_id: str
    label: str
    aliases: list[str]
    format_code: str
    signoff_code: str
    required_code: str
    avoid_code: str
    max_words: int
    topic: str


@dataclass(frozen=True)
class StylePayload:
    format_code: str
    signoff_code: str
    required_code: str
    avoid_code: str
    max_words: int


@dataclass(frozen=True)
class StyleTurn:
    phase: str
    action: str
    profile_id: str
    prompt: str = ""


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def stable_hash_int(text: str, bits: int = 64) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=bits // 8).digest()
    return int.from_bytes(digest, "little")


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


def profile_payload(profile: StyleProfile) -> StylePayload:
    return StylePayload(
        profile.format_code,
        profile.signoff_code,
        profile.required_code,
        profile.avoid_code,
        profile.max_words,
    )


def payload_to_hint(payload: StylePayload, hint_style: str = "strict") -> str:
    if hint_style == "soft":
        format_text = {
            "three_bullets": "Write exactly three short bullet points.",
            "two_bullets": "Write exactly two short bullet points.",
            "subject_body": "Write a short email with a Subject line and a Body section.",
        }[payload.format_code]
        return " ".join(
            [
                "Write a short, polished reply.",
                format_text,
                f"Work in '{REQUIRED_TEXT[payload.required_code]}' naturally.",
                f"Close with '{SIGNOFF_TEXT[payload.signoff_code]}'.",
                f"Please avoid the word '{AVOID_TEXT[payload.avoid_code]}'.",
                f"Keep it under {payload.max_words} words.",
            ]
        )
    return " ".join(
        [
            FORMAT_TEXT[payload.format_code],
            f"Include the phrase '{REQUIRED_TEXT[payload.required_code]}'.",
            f"End with '{SIGNOFF_TEXT[payload.signoff_code]}'.",
            f"Do not use the word '{AVOID_TEXT[payload.avoid_code]}'.",
            f"Keep it under {payload.max_words} words.",
        ]
    )


def learning_text(profile: StyleProfile) -> str:
    return (
        f"Style update for {profile.label}: "
        f"{payload_to_hint(profile_payload(profile))}"
    )


def revision_text(profile: StyleProfile) -> str:
    return (
        f"Style correction for {profile.label}: "
        f"{payload_to_hint(profile_payload(profile))}"
    )


def raw_state_bytes(raw_records: dict[str, str]) -> int:
    return len(pickle.dumps(raw_records, protocol=pickle.HIGHEST_PROTOCOL))


class StyleMemory:
    def __init__(self, profiles: list[StyleProfile], hash_bits: int, ngrams: int, hint_style: str = "strict") -> None:
        self.profiles = {profile.profile_id: profile for profile in profiles}
        self.hash_bits = hash_bits
        self.ngrams = ngrams
        self.hint_style = hint_style
        self.payloads: dict[str, StylePayload | None] = {profile.profile_id: None for profile in profiles}
        self.tables: dict[int, dict[str, float]] = {}
        self.counts: dict[str, float] = {profile.profile_id: 0.0 for profile in profiles}

    def profile_for_text(self, text: str) -> str | None:
        tokens = set(normalize(text).split())
        for profile in self.profiles.values():
            for alias in profile.aliases:
                alias_tokens = set(normalize(alias).split())
                if alias_tokens and alias_tokens <= tokens:
                    return profile.profile_id
        return None

    def features(self, text: str) -> list[int]:
        tokens = normalize(text).split()
        feats: set[int] = set()
        for n in range(1, self.ngrams + 1):
            for idx in range(len(tokens) - n + 1):
                feats.add(stable_hash_int(f"{n}:{' '.join(tokens[idx:idx+n])}", self.hash_bits))
        profile_id = self.profile_for_text(text)
        if profile_id is not None:
            feats.add(stable_hash_int(f"profile:{profile_id}", self.hash_bits))
        return sorted(feats)

    def update(self, profile_id: str, payload: StylePayload, text: str) -> None:
        self.payloads[profile_id] = payload
        self.counts[profile_id] += 1.0
        features = set(self.features(text))
        features.add(stable_hash_int(f"profile:{profile_id}", self.hash_bits))
        for feature in features:
            row = self.tables.setdefault(feature, {})
            row[profile_id] = row.get(profile_id, 0.0) + 1.0

    def forget(self, profile_id: str) -> None:
        self.payloads[profile_id] = None
        self.counts[profile_id] = 0.0
        empty_features = []
        for feature, row in self.tables.items():
            row.pop(profile_id, None)
            if not row:
                empty_features.append(feature)
        for feature in empty_features:
            del self.tables[feature]

    def hint(self, prompt: str) -> list[dict[str, Any]]:
        profile_id = self.profile_for_text(prompt)
        if profile_id is None or self.payloads.get(profile_id) is None:
            return []
        score = self.counts[profile_id]
        feature = stable_hash_int(f"profile:{profile_id}", self.hash_bits)
        row = self.tables.get(feature, {})
        score += row.get(profile_id, 0.0)
        return [
            {
                "profile_id": profile_id,
                "score": float(score),
                "style": payload_to_hint(self.payloads[profile_id], self.hint_style),
            }
        ]

    def state_bytes(self) -> int:
        state = {
            "payloads": self.payloads,
            "tables": self.tables,
            "counts": self.counts,
            "hash_bits": self.hash_bits,
            "ngrams": self.ngrams,
            "hint_style": self.hint_style,
        }
        return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))


def build_profiles(limit: int) -> list[StyleProfile]:
    seeds = [
        ("mira", "Mira account", "three_bullets", "mira", "next_step", "urgent", 75, "shipping delay"),
        ("jules", "Jules desk", "subject_body", "jules", "owner", "guarantee", 85, "feature request"),
        ("noor", "Noor workspace", "two_bullets", "noor", "eta", "cheap", 70, "pricing clarification"),
        ("ren", "Ren project", "three_bullets", "ren", "why", "sorry", 75, "missed appointment"),
        ("sol", "Sol studio", "subject_body", "sol", "action", "asap", 85, "inventory change"),
        ("talia", "Talia queue", "two_bullets", "talia", "note", "perfect", 70, "warranty question"),
    ]
    profiles = []
    for idx, row in enumerate(seeds[:limit]):
        key, label, format_code, signoff_code, required_code, avoid_code, max_words, topic = row
        profile_id = f"profile_{idx:03d}_{key}"
        profiles.append(
            StyleProfile(
                profile_id=profile_id,
                label=label,
                aliases=[label, key, f"{key} customer", f"{key} account"],
                format_code=format_code,
                signoff_code=signoff_code,
                required_code=required_code,
                avoid_code=avoid_code,
                max_words=max_words,
                topic=topic,
            )
        )
    return profiles


def revised_profile(profile: StyleProfile, idx: int) -> StyleProfile:
    format_cycle = ["two_bullets", "three_bullets", "subject_body"]
    signoff_cycle = ["ren", "sol", "talia", "mira"]
    required_cycle = ["action", "note", "owner", "eta"]
    return StyleProfile(
        profile_id=profile.profile_id,
        label=profile.label,
        aliases=profile.aliases,
        format_code=format_cycle[idx % len(format_cycle)],
        signoff_code=signoff_cycle[idx % len(signoff_cycle)],
        required_code=required_cycle[idx % len(required_cycle)],
        avoid_code=profile.avoid_code,
        max_words=max(profile.max_words - 10, 55),
        topic=profile.topic,
    )


def writing_prompt(profile: StyleProfile, variant: int) -> str:
    prompts = [
        (
            f"Draft a customer reply for the {profile.label} about a {profile.topic}. "
            f"The customer called it {AVOID_TEXT[profile.avoid_code]} and wants a clear update."
        ),
        (
            f"Write a short customer-facing note for {profile.aliases[1]} about a {profile.topic}. "
            f"Keep it practical and useful."
        ),
    ]
    return prompts[variant % len(prompts)]


def build_session(profiles: list[StyleProfile], revised: dict[str, StyleProfile]) -> list[StyleTurn]:
    turns: list[StyleTurn] = []
    for profile in profiles:
        turns.append(StyleTurn("learn", "learn", profile.profile_id))
    for idx, profile in enumerate(profiles):
        turns.append(StyleTurn("initial_query", "query", profile.profile_id, writing_prompt(profile, idx)))
    for profile_id in list(revised)[:2]:
        turns.append(StyleTurn("revise", "revise", profile_id))
    for idx, profile_id in enumerate(list(revised)[:2]):
        turns.append(StyleTurn("revision_query", "query", profile_id, writing_prompt(revised[profile_id], idx + 1)))
    if profiles:
        turns.append(StyleTurn("delete", "delete", profiles[0].profile_id))
        turns.append(StyleTurn("deleted_query", "query", profiles[0].profile_id, writing_prompt(profiles[0], 0)))
    for idx, profile in enumerate(profiles[-2:]):
        turns.append(StyleTurn("retained_query", "query", profile.profile_id, writing_prompt(profile, idx + 1)))
    return turns


def build_messages(prompt: str, hint: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    system = (
        "You write concise customer-facing drafts. If a style hint is provided, follow every "
        "formatting, phrase, signoff, word-avoidance, and length instruction. If no style hint "
        "is provided, write a normal helpful draft."
    )
    lines = [f"Writing request: {prompt}"]
    if hint:
        style_lines = [f"{row['profile_id']}: {row['style']} (score {row['score']:.3f})" for row in hint]
        lines.append("Style hint:\n" + "\n".join(style_lines))
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


def api_style_answer(
    client: OpenAICompatibleClient | None,
    prompt: str,
    hint: list[dict[str, Any]] | None,
    args,
) -> tuple[str, dict[str, Any]]:
    messages = build_messages(prompt, hint)
    payload = {"model": args.api_model, "messages": messages, "temperature": args.api_temperature, "max_tokens": args.api_max_tokens}
    if client is None:
        return "", {"request": payload}
    response = client.chat(messages, args.api_max_tokens, args.api_temperature)
    text = response["choices"][0]["message"]["content"]
    return text, {"request": payload, "response": response, "raw_text": text}


def raw_hint(raw_records: dict[str, str], profile_id: str) -> list[dict[str, Any]]:
    text = raw_records.get(profile_id)
    if text is None:
        return []
    return [{"profile_id": profile_id, "score": 1.0, "style": text}]


def bullet_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*[-*]\s+\S", line))


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text))


def score_text(text: str, profile: StyleProfile) -> dict[str, int]:
    norm = normalize(text)
    payload = profile_payload(profile)
    if payload.format_code == "three_bullets":
        format_pass = int(bullet_count(text) == 3)
    elif payload.format_code == "two_bullets":
        format_pass = int(bullet_count(text) == 2)
    else:
        format_pass = int("subject:" in text.lower() and "body:" in text.lower())
    signoff_pass = int(normalize(SIGNOFF_TEXT[payload.signoff_code]) in norm)
    required_pass = int(normalize(REQUIRED_TEXT[payload.required_code]) in norm)
    avoid_pass = int(AVOID_TEXT[payload.avoid_code] not in norm.split())
    length_pass = int(word_count(text) <= payload.max_words)
    all_pass = int(format_pass and signoff_pass and required_pass and avoid_pass and length_pass)
    return {
        "format_pass": format_pass,
        "signoff_pass": signoff_pass,
        "required_pass": required_pass,
        "avoid_pass": avoid_pass,
        "length_pass": length_pass,
        "all_pass": all_pass,
        "word_count": word_count(text),
    }


def score_hint(hint: list[dict[str, Any]], profile: StyleProfile) -> dict[str, int]:
    if not hint:
        return {
            "format_pass": 0,
            "signoff_pass": 0,
            "required_pass": 0,
            "avoid_pass": 0,
            "length_pass": 0,
            "all_pass": 0,
            "word_count": 0,
        }
    style = hint[0]["style"]
    payload = profile_payload(profile)
    norm = normalize(style)
    tokens = set(norm.split())
    if payload.format_code == "three_bullets":
        format_pass = int(
            normalize(FORMAT_TEXT[payload.format_code]) in norm
            or ("three" in tokens and ("bullet" in tokens or "bullets" in tokens))
        )
    elif payload.format_code == "two_bullets":
        format_pass = int(
            normalize(FORMAT_TEXT[payload.format_code]) in norm
            or ("two" in tokens and ("bullet" in tokens or "bullets" in tokens))
        )
    else:
        format_pass = int(
            normalize(FORMAT_TEXT[payload.format_code]) in norm
            or ("subject" in tokens and "body" in tokens)
        )
    signoff_pass = int(normalize(SIGNOFF_TEXT[payload.signoff_code]) in norm)
    required_pass = int(normalize(REQUIRED_TEXT[payload.required_code]) in norm)
    style_lower = style.lower()
    avoid_pass = int(
        re.search(rf"\b{re.escape(AVOID_TEXT[payload.avoid_code])}\b", style_lower) is not None
        and any(
            cue in style_lower
            for cue in [
                "avoid",
                "do not",
                "dont",
                "don't",
                "not use",
                "leave out",
                "omit",
                "skip",
            ]
        )
    )
    length_pass = int(str(payload.max_words) in style)
    all_pass = int(format_pass and signoff_pass and required_pass and avoid_pass and length_pass)
    return {
        "format_pass": format_pass,
        "signoff_pass": signoff_pass,
        "required_pass": required_pass,
        "avoid_pass": avoid_pass,
        "length_pass": length_pass,
        "all_pass": all_pass,
        "word_count": word_count(style),
    }


def summarize(rows: list[dict[str, Any]], memory: StyleMemory, raw_bytes: int) -> list[dict[str, Any]]:
    active_rows = [row for row in rows if row.get("score_phase") == "active"]
    deleted_rows = [row for row in rows if row.get("score_phase") == "deleted"]

    def mean(key: str, subset: list[dict[str, Any]]) -> float:
        values = [float(row[key]) for row in subset if row.get(key) != ""]
        return sum(values) / max(len(values), 1)

    summary = [
        {
            "method": "local_raw_profile",
            "accuracy": mean("raw_local_all_pass", active_rows),
            "format_pass": mean("raw_local_format_pass", active_rows),
            "signoff_pass": mean("raw_local_signoff_pass", active_rows),
            "required_pass": mean("raw_local_required_pass", active_rows),
            "avoid_pass": mean("raw_local_avoid_pass", active_rows),
            "length_pass": mean("raw_local_length_pass", active_rows),
            "state_bytes": raw_bytes,
            "stores_raw_examples": True,
            "stores_preference_text": True,
            "query_count": len(active_rows),
        },
        {
            "method": "local_style_sketch_memory",
            "accuracy": mean("memory_local_all_pass", active_rows),
            "format_pass": mean("memory_local_format_pass", active_rows),
            "signoff_pass": mean("memory_local_signoff_pass", active_rows),
            "required_pass": mean("memory_local_required_pass", active_rows),
            "avoid_pass": mean("memory_local_avoid_pass", active_rows),
            "length_pass": mean("memory_local_length_pass", active_rows),
            "state_bytes": memory.state_bytes(),
            "stores_raw_examples": False,
            "stores_preference_text": False,
            "query_count": len(active_rows),
        },
    ]
    api_rows = [row for row in active_rows if row.get("api_called")]
    if api_rows:
        summary.extend(
            [
                {
                    "method": "api_no_memory",
                    "accuracy": mean("api_no_memory_all_pass", api_rows),
                    "format_pass": mean("api_no_memory_format_pass", api_rows),
                    "signoff_pass": mean("api_no_memory_signoff_pass", api_rows),
                    "required_pass": mean("api_no_memory_required_pass", api_rows),
                    "avoid_pass": mean("api_no_memory_avoid_pass", api_rows),
                    "length_pass": mean("api_no_memory_length_pass", api_rows),
                    "state_bytes": 0,
                    "stores_raw_examples": False,
                    "stores_preference_text": False,
                    "query_count": len(api_rows),
                },
                {
                    "method": "api_raw_profile",
                    "accuracy": mean("api_raw_profile_all_pass", api_rows),
                    "format_pass": mean("api_raw_profile_format_pass", api_rows),
                    "signoff_pass": mean("api_raw_profile_signoff_pass", api_rows),
                    "required_pass": mean("api_raw_profile_required_pass", api_rows),
                    "avoid_pass": mean("api_raw_profile_avoid_pass", api_rows),
                    "length_pass": mean("api_raw_profile_length_pass", api_rows),
                    "state_bytes": raw_bytes,
                    "stores_raw_examples": True,
                    "stores_preference_text": True,
                    "query_count": len(api_rows),
                },
                {
                    "method": "api_style_sketch_memory",
                    "accuracy": mean("api_memory_all_pass", api_rows),
                    "format_pass": mean("api_memory_format_pass", api_rows),
                    "signoff_pass": mean("api_memory_signoff_pass", api_rows),
                    "required_pass": mean("api_memory_required_pass", api_rows),
                    "avoid_pass": mean("api_memory_avoid_pass", api_rows),
                    "length_pass": mean("api_memory_length_pass", api_rows),
                    "state_bytes": memory.state_bytes(),
                    "stores_raw_examples": False,
                    "stores_preference_text": False,
                    "query_count": len(api_rows),
                },
            ]
        )
    if deleted_rows:
        summary.extend(
            [
                {
                    "method": "local_deleted_suppression",
                    "accuracy": mean("memory_deleted_suppressed", deleted_rows),
                    "format_pass": "",
                    "signoff_pass": "",
                    "required_pass": "",
                    "avoid_pass": "",
                    "length_pass": "",
                    "state_bytes": memory.state_bytes(),
                    "stores_raw_examples": False,
                    "stores_preference_text": False,
                    "query_count": len(deleted_rows),
                }
            ]
        )
        api_deleted_rows = [row for row in deleted_rows if row.get("api_called")]
        if api_deleted_rows:
            summary.append(
                {
                    "method": "api_deleted_suppression",
                    "accuracy": mean("api_memory_deleted_suppressed", api_deleted_rows),
                    "format_pass": "",
                    "signoff_pass": "",
                    "required_pass": "",
                    "avoid_pass": "",
                    "length_pass": "",
                    "state_bytes": memory.state_bytes(),
                    "stores_raw_examples": False,
                    "stores_preference_text": False,
                    "query_count": len(api_deleted_rows),
                }
            )
    return summary


def add_scores(prefix: str, row: dict[str, Any], scores: dict[str, int]) -> None:
    for key, value in scores.items():
        row[f"{prefix}_{key}"] = value


def write_transcript(path: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    query_rows = [row for row in rows if row.get("action") == "query"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Personalized Style API Transcript\n\n")
        f.write("## Summary\n\n")
        f.write("| method | accuracy | format | signoff | required | avoid | length | state_bytes |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['method']} | {row['accuracy']} | {row['format_pass']} | {row['signoff_pass']} | "
                f"{row['required_pass']} | {row['avoid_pass']} | {row['length_pass']} | {row['state_bytes']} |\n"
            )
        f.write("\n## Query Turns\n\n")
        for row in query_rows:
            f.write(f"### Turn {row['turn']} - {row['phase']} - {row['profile_id']}\n\n")
            f.write(f"Prompt: {row['prompt']}\n\n")
            f.write(f"Sketch hint: {row['memory_hint'] or '<none>'}\n\n")
            if row.get("api_called"):
                f.write(f"No-memory API:\n{row['api_no_memory_answer']}\n\n")
                f.write(f"Raw-profile API:\n{row['api_raw_profile_answer']}\n\n")
                f.write(f"Style-sketch API:\n{row['api_memory_answer']}\n\n")
            f.write(
                f"All-pass: local_raw={row.get('raw_local_all_pass')}, "
                f"local_memory={row.get('memory_local_all_pass')}, "
                f"api_no_memory={row.get('api_no_memory_all_pass')}, "
                f"api_raw={row.get('api_raw_profile_all_pass')}, "
                f"api_memory={row.get('api_memory_all_pass')}\n\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "online_memory_style_api")
    parser.add_argument("--profile-limit", type=int, default=6)
    parser.add_argument("--hash-bits", type=int, default=64)
    parser.add_argument("--ngrams", type=int, default=2)
    parser.add_argument("--hint-style", choices=["strict", "soft"], default="strict")
    parser.add_argument("--api-limit", type=int, default=0)
    parser.add_argument("--api-base-url", type=str, default="https://yzhanghmeng.com/v1")
    parser.add_argument("--api-model", type=str, default="gpt-5.5")
    parser.add_argument("--api-timeout", type=float, default=90.0)
    parser.add_argument("--api-max-tokens", type=int, default=160)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-key-env", type=str, default="API_KEY")
    parser.add_argument("--run-api", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    profiles = build_profiles(args.profile_limit)
    profiles_by_id = {profile.profile_id: profile for profile in profiles}
    revised = {profile.profile_id: revised_profile(profile, idx) for idx, profile in enumerate(profiles[:2])}
    memory = StyleMemory(profiles, args.hash_bits, args.ngrams, args.hint_style)
    raw_records: dict[str, str] = {}
    turns = build_session(profiles, revised)

    client = None
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if args.run_api:
        if not api_key:
            raise RuntimeError(f"API key not found in ${args.api_key_env} or $OPENAI_API_KEY")
        client = OpenAICompatibleClient(args.api_base_url, api_key, args.api_model, args.api_timeout)

    rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    active_payload_profile: dict[str, StyleProfile] = dict(profiles_by_id)
    api_count = 0

    for turn_idx, turn in enumerate(turns, start=1):
        profile = active_payload_profile.get(turn.profile_id, profiles_by_id[turn.profile_id])
        if turn.action == "learn":
            text = learning_text(profile)
            memory.update(turn.profile_id, profile_payload(profile), text)
            raw_records[turn.profile_id] = text
            rows.append({"turn": turn_idx, "phase": turn.phase, "action": turn.action, "profile_id": turn.profile_id, "event_text": text})
            continue

        if turn.action == "revise":
            profile = revised[turn.profile_id]
            active_payload_profile[turn.profile_id] = profile
            text = revision_text(profile)
            memory.update(turn.profile_id, profile_payload(profile), text)
            raw_records[turn.profile_id] = text
            rows.append({"turn": turn_idx, "phase": turn.phase, "action": turn.action, "profile_id": turn.profile_id, "event_text": text})
            continue

        if turn.action == "delete":
            memory.forget(turn.profile_id)
            raw_records.pop(turn.profile_id, None)
            rows.append({"turn": turn_idx, "phase": turn.phase, "action": turn.action, "profile_id": turn.profile_id, "event_text": "forget profile"})
            continue

        memory_hint = memory.hint(turn.prompt)
        raw_profile_hint = raw_hint(raw_records, turn.profile_id)
        raw_scores = score_hint(raw_profile_hint, profile)
        memory_scores = score_hint(memory_hint, profile)
        score_phase = "deleted" if turn.phase == "deleted_query" else "active"

        row: dict[str, Any] = {
            "turn": turn_idx,
            "phase": turn.phase,
            "action": turn.action,
            "score_phase": score_phase,
            "profile_id": turn.profile_id,
            "prompt": turn.prompt,
            "memory_hint": memory_hint[0]["style"] if memory_hint else "",
            "raw_hint": raw_profile_hint[0]["style"] if raw_profile_hint else "",
            "api_called": 0,
            "api_no_memory_answer": "",
            "api_raw_profile_answer": "",
            "api_memory_answer": "",
        }
        add_scores("raw_local", row, raw_scores)
        add_scores("memory_local", row, memory_scores)
        row["raw_deleted_suppressed"] = int(score_phase == "deleted" and not raw_profile_hint)
        row["memory_deleted_suppressed"] = int(score_phase == "deleted" and not memory_hint)

        api_called = args.run_api and api_count < args.api_limit
        if api_called:
            api_count += 1
            no_memory_text, no_memory_meta = api_style_answer(client, turn.prompt, None, args)
            raw_text, raw_meta = api_style_answer(client, turn.prompt, raw_profile_hint, args)
            memory_text, memory_meta = api_style_answer(client, turn.prompt, memory_hint, args)
            row["api_called"] = 1
            row["api_no_memory_answer"] = no_memory_text
            row["api_raw_profile_answer"] = raw_text
            row["api_memory_answer"] = memory_text
            add_scores("api_no_memory", row, score_text(no_memory_text, profile))
            add_scores("api_raw_profile", row, score_text(raw_text, profile))
            add_scores("api_memory", row, score_text(memory_text, profile))
            row["api_raw_deleted_suppressed"] = int(score_phase == "deleted" and not raw_profile_hint)
            row["api_memory_deleted_suppressed"] = int(score_phase == "deleted" and not memory_hint)
            request_rows.extend(
                [
                    {"turn": turn_idx, "mode": "api_no_memory", "payload": no_memory_meta["request"]},
                    {"turn": turn_idx, "mode": "api_raw_profile", "payload": raw_meta["request"]},
                    {"turn": turn_idx, "mode": "api_style_sketch_memory", "payload": memory_meta["request"]},
                ]
            )
            response_rows.extend(
                [
                    {"turn": turn_idx, "mode": "api_no_memory", **no_memory_meta},
                    {"turn": turn_idx, "mode": "api_raw_profile", **raw_meta},
                    {"turn": turn_idx, "mode": "api_style_sketch_memory", **memory_meta},
                ]
            )
        else:
            for prefix in ["api_no_memory", "api_raw_profile", "api_memory"]:
                for key in ["format_pass", "signoff_pass", "required_pass", "avoid_pass", "length_pass", "all_pass", "word_count"]:
                    row[f"{prefix}_{key}"] = ""
            row["api_raw_deleted_suppressed"] = ""
            row["api_memory_deleted_suppressed"] = ""

        rows.append(row)

    summary_rows = summarize(rows, memory, raw_state_bytes(raw_records))
    write_csv(args.out_dir / "session_turns.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    with (args.out_dir / "api_requests.jsonl").open("w", encoding="utf-8") as f:
        for row in request_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if response_rows:
        with (args.out_dir / "api_responses.jsonl").open("w", encoding="utf-8") as f:
            for row in response_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_transcript(args.out_dir / "style_transcript.md", rows, summary_rows)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("Summary:")
    for row in summary_rows:
        def fmt(value: Any) -> str:
            if value == "":
                return ""
            return f"{float(value):.3f}"

        print(
            f"  {row['method']}: acc={fmt(row['accuracy'])} format={fmt(row['format_pass'])} "
            f"signoff={fmt(row['signoff_pass'])} required={fmt(row['required_pass'])} "
            f"avoid={fmt(row['avoid_pass'])} length={fmt(row['length_pass'])} "
            f"bytes={row['state_bytes']} raw_examples={row['stores_raw_examples']}"
        )
    print(f"wrote summary: {args.out_dir / 'summary.csv'}")
    print(f"wrote transcript: {args.out_dir / 'style_transcript.md'}")


if __name__ == "__main__":
    main()
