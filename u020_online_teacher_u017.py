#!/usr/bin/env python3
"""
U020 online teacher loop for the U017 no-BP learner.

DeepSeek or a mock teacher generates short lessons, U017 learns them as normal
next-token text, and a lightweight teacher/judge loop decides the next focus.
The API is only a curriculum agent: it does not provide gradients, hidden
states, logits, or model-side modules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch

import u012_u009_torch_fast_variants as u012
from u017_error_population_llm import TorchU017ErrorPopulationModel


SCRIPT_DIR = Path(__file__).resolve().parent


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Lesson:
    lesson_id: str
    skill: str
    difficulty: int
    prompt: str
    target: str
    probe: str
    expected: str
    notes: str = ""


@dataclass
class ProbeSample:
    lesson_id: str
    question: str
    expected: str
    student: str


@dataclass
class JudgeResult:
    lesson_id: str
    correct: bool
    score: float
    problem: str
    correction: str = ""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def max_repeat_run(text: str) -> int:
    tokens = re.findall(r"[a-z0-9]+|[^\s]", text.lower())
    if not tokens:
        return 0
    best = 1
    run = 1
    prev = tokens[0]
    for token in tokens[1:]:
        if token == prev:
            run += 1
            best = max(best, run)
        else:
            run = 1
            prev = token
    return best


def bounded(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip()


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)

    def chat(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
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


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    if "content" in message:
        return str(message["content"])
    return str(choices[0].get("text", ""))


def extract_json_payload(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    starts = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx >= 0]
    if not starts:
        raise ValueError("no JSON object or array found in API response")
    start = min(starts)
    end_obj = stripped.rfind("}")
    end_arr = stripped.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        raise ValueError("truncated JSON in API response")
    return json.loads(stripped[start : end + 1])


def coerce_lessons(payload: Any, round_idx: int, max_field_chars: int) -> list[Lesson]:
    raw_lessons = payload.get("lessons") if isinstance(payload, dict) else payload
    if not isinstance(raw_lessons, list):
        raise ValueError("teacher response must contain a lessons list")
    lessons: list[Lesson] = []
    for idx, item in enumerate(raw_lessons):
        if not isinstance(item, dict):
            continue
        prompt = bounded(item.get("prompt"), max_field_chars)
        target = bounded(item.get("target"), max_field_chars)
        probe = bounded(item.get("probe"), max_field_chars)
        expected = bounded(item.get("expected"), max_field_chars)
        if not prompt or not target or not probe or not expected:
            continue
        lesson_id = bounded(item.get("id") or item.get("lesson_id") or f"r{round_idx:03d}_{idx:03d}", 80)
        skill = bounded(item.get("skill") or "basic_language", 80)
        try:
            difficulty = int(item.get("difficulty", 1))
        except (TypeError, ValueError):
            difficulty = 1
        lessons.append(
            Lesson(
                lesson_id=lesson_id,
                skill=skill,
                difficulty=max(difficulty, 1),
                prompt=prompt,
                target=target,
                probe=probe,
                expected=expected,
                notes=bounded(item.get("notes"), max_field_chars),
            )
        )
    return lessons


def build_teacher_messages(
    round_idx: int,
    lesson_count: int,
    difficulty: int,
    focus: str,
    last_metrics: dict[str, Any],
    recent_errors: str,
) -> list[dict[str, str]]:
    system = (
        "You are a patient preschool teacher for a tiny student language model. "
        "Teach through plain text examples only. Use short concrete English. "
        "Return JSON only. Do not use markdown."
    )
    schema = {
        "lessons": [
            {
                "id": "short_id",
                "skill": "object_memory",
                "difficulty": difficulty,
                "prompt": "Tell a very short story about ...",
                "target": "Two or three simple sentences.",
                "probe": "A short question about the story.",
                "expected": "A short answer.",
                "notes": "optional",
            }
        ]
    }
    metrics_text = json.dumps(last_metrics, ensure_ascii=True, sort_keys=True)
    user = (
        f"Round: {round_idx}\n"
        f"Need {lesson_count} lessons.\n"
        f"Difficulty: {difficulty} on a 1-6 scale.\n"
        f"Current focus: {focus}\n"
        f"Recent metrics: {metrics_text}\n"
        f"Recent student errors: {recent_errors or 'none'}\n\n"
        "Make the curriculum toddler-like: one idea per lesson, concrete nouns, "
        "short sentences, mild repetition with varied wording, and a direct probe. "
        "Do not include hidden reasoning. Keep each target under 45 words and each "
        "expected answer under 12 words.\n\n"
        f"JSON schema example:\n{json.dumps(schema, ensure_ascii=True)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def mock_lessons(round_idx: int, count: int, difficulty: int, focus: str, seed: int) -> list[Lesson]:
    rng = random.Random(seed + 1009 * round_idx + 17 * difficulty)
    names = ["Lily", "Tom", "Mia", "Ben", "Nora", "Sam", "Ava", "Leo"]
    colors = ["red", "blue", "green", "yellow", "white", "black"]
    objects = ["ball", "cup", "hat", "box", "kite", "book"]
    places = ["bed", "chair", "tree", "door", "table", "bench"]
    actions = ["held", "washed", "opened", "carried", "found", "shared"]
    lessons: list[Lesson] = []
    for idx in range(count):
        name = rng.choice(names)
        color = rng.choice(colors)
        obj = rng.choice(objects)
        place = rng.choice(places)
        action = rng.choice(actions)
        if difficulty <= 2:
            target = f"{name} found a {color} {obj} by the {place}. {name} smiled."
            probe = f"What did {name} find?"
            expected = f"a {color} {obj}"
            skill = "object_memory"
        elif difficulty <= 4:
            target = (
                f"{name} found a {color} {obj} by the {place}. "
                f"{name} {action} the {obj} because it was useful."
            )
            probe = f"Why did {name} {action} the {obj}?"
            expected = "it was useful"
            skill = "simple_cause"
        else:
            target = (
                f"{name} looked by the {place}. First {name} found a {color} {obj}. "
                f"Then {name} {action} it and went home."
            )
            probe = f"What happened first?"
            expected = f"{name} found a {color} {obj}"
            skill = "event_order"
        lessons.append(
            Lesson(
                lesson_id=f"mock_r{round_idx:03d}_{idx:03d}",
                skill=skill if focus == "balanced" else focus,
                difficulty=difficulty,
                prompt=f"Tell a very short story about {name} and a {color} {obj}.",
                target=target,
                probe=probe,
                expected=expected,
                notes="mock",
            )
        )
    return lessons


def lesson_to_training_text(lesson: Lesson) -> str:
    return (
        "Story request:\n"
        f"{lesson.prompt.strip()}\n\n"
        "Story:\n"
        f"{lesson.target.strip()}\n\n"
        "Question:\n"
        f"{lesson.probe.strip()}\n\n"
        "Answer:\n"
        f"{lesson.expected.strip()}\n"
    )


def lesson_to_probe_prompt(lesson: Lesson) -> str:
    return (
        "Story:\n"
        f"{lesson.target.strip()}\n\n"
        "Question:\n"
        f"{lesson.probe.strip()}\n\n"
        "Answer:\n"
    )


def encode_text(tokenizer: Any, text: str) -> np.ndarray:
    return np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int64)


# ── cold-start warmup ────────────────────────────────────────────────────────

def _fetch_url_text(url: str, max_bytes: int = 500_000) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read(max_bytes)
    text = raw.decode("utf-8", errors="replace")
    # strip HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _split_sentences(text: str, min_len: int = 40, max_len: int = 400) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        buf = (buf + " " + p).strip() if buf else p
        if len(buf) >= min_len:
            chunks.append(buf[:max_len])
            buf = ""
    if buf and len(buf) >= min_len:
        chunks.append(buf[:max_len])
    return chunks


def fetch_warmup_docs_url(urls: list[str], tokenizer: Any, max_docs: int) -> list[np.ndarray]:
    docs: list[np.ndarray] = []
    for url in urls:
        try:
            text = _fetch_url_text(url)
            sentences = _split_sentences(text)
            random.shuffle(sentences)
            for sent in sentences[:max_docs]:
                ids = encode_text(tokenizer, sent)
                if ids.size > 4:
                    docs.append(ids)
                if len(docs) >= max_docs:
                    break
        except Exception as exc:
            print(f"[warmup] failed to fetch {url}: {exc}", flush=True)
        if len(docs) >= max_docs:
            break
    return docs[:max_docs]


def fetch_warmup_docs_wikitext(tokenizer: Any, max_docs: int, split: str = "train") -> list[np.ndarray]:
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=False)
        texts: list[str] = [row["text"] for row in ds if row["text"].strip()]  # type: ignore
    except Exception as exc:
        print(f"[warmup] datasets load failed ({exc}), falling back to URL", flush=True)
        # fallback: grab wikitext via raw GitHub mirror
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        texts = _fetch_url_text(url).splitlines()
    docs: list[np.ndarray] = []
    random.shuffle(texts)
    for line in texts:
        line = line.strip()
        if len(line) < 40:
            continue
        ids = encode_text(tokenizer, line[:400])
        if ids.size > 4:
            docs.append(ids)
        if len(docs) >= max_docs:
            break
    return docs[:max_docs]


def run_warmup(
    model: Any,
    tokenizer: Any,
    source: str,
    warmup_docs: int,
    warmup_passes: int,
    warmup_urls: list[str],
    chunk_size: int,
    chunk_tokens: int,
    out_dir: Path,
) -> None:
    """Pretrain model on generic text before teacher loop starts."""
    print(f"[warmup] source={source} docs={warmup_docs} passes={warmup_passes}", flush=True)
    t0 = time.perf_counter()

    if source == "wikitext":
        docs = fetch_warmup_docs_wikitext(tokenizer, warmup_docs)
    elif source == "url":
        docs = fetch_warmup_docs_url(warmup_urls, tokenizer, warmup_docs)
    else:
        raise ValueError(f"unknown warmup source {source!r}")

    if not docs:
        print("[warmup] no docs fetched, skipping warmup", flush=True)
        return

    print(f"[warmup] fetched {len(docs)} docs, starting {warmup_passes} pass(es)", flush=True)
    for p in range(warmup_passes):
        random.shuffle(docs)
        summary, _ = u012.run_documents(
            model, docs, update=True, mode="chunk",
            chunk_size=chunk_size, chunk_tokens=chunk_tokens, eos_id=None,
        )
        print(
            f"[warmup] pass={p} loss={summary['loss']:.4f} "
            f"tokens={summary['tokens']} elapsed={time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "warmup_summary.json").open("w") as f:
            json.dump({"source": source, "docs": len(docs), "passes": warmup_passes,
                       "final_loss": float(summary["loss"]),
                       "elapsed": round(time.perf_counter() - t0, 2)}, f)
    print(f"[warmup] done in {time.perf_counter()-t0:.1f}s", flush=True)


# ── end cold-start warmup ────────────────────────────────────────────────────


def tokenize_lessons(tokenizer: Any, lessons: list[Lesson]) -> list[np.ndarray]:
    docs = []
    for lesson in lessons:
        ids = encode_text(tokenizer, lesson_to_training_text(lesson))
        if ids.size > 1:
            docs.append(ids)
    return docs


@torch.no_grad()
def evaluate_answer_ce(
    model: TorchU017ErrorPopulationModel,
    tokenizer: Any,
    probes: list[Lesson],
    batch_size: int,
) -> dict[str, float | int]:
    contexts: list[np.ndarray] = []
    targets: list[int] = []
    for lesson in probes:
        prefix = encode_text(tokenizer, lesson_to_probe_prompt(lesson))
        answer = encode_text(tokenizer, lesson.expected.strip() + "\n")
        if prefix.size == 0 or answer.size == 0:
            continue
        history = [int(x) for x in prefix.tolist()]
        for token in answer.tolist():
            contexts.append(np.asarray(history[-model.context_len :], dtype=np.int64))
            targets.append(int(token))
            history.append(int(token))
    loss_sum = 0.0
    correct = 0
    total = 0
    for start in range(0, len(targets), max(int(batch_size), 1)):
        chunk_contexts = contexts[start : start + max(int(batch_size), 1)]
        chunk_targets = np.asarray(targets[start : start + max(int(batch_size), 1)], dtype=np.int64)
        batch_loss, batch_correct, batch_total = u012.predict_chunk(model, chunk_contexts, chunk_targets)
        loss_sum += batch_loss
        correct += batch_correct
        total += batch_total
    return u012.summarize(loss_sum, correct, total)


@torch.no_grad()
def generate_probe_samples(
    model: TorchU017ErrorPopulationModel,
    tokenizer: Any,
    probes: list[Lesson],
    count: int,
    max_new_tokens: int,
    temperature: float,
) -> list[ProbeSample]:
    samples: list[ProbeSample] = []
    for lesson in probes[: max(int(count), 0)]:
        prompt = encode_text(tokenizer, lesson_to_probe_prompt(lesson))
        if prompt.size == 0:
            continue
        generated = model.generate(prompt, max_new_tokens, sample=False, temperature=temperature)
        completion_ids = generated[int(prompt.size) :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        samples.append(
            ProbeSample(
                lesson_id=lesson.lesson_id,
                question=lesson.probe,
                expected=lesson.expected,
                student=completion.strip(),
            )
        )
    return samples


def mock_judge(samples: list[ProbeSample]) -> list[JudgeResult]:
    results: list[JudgeResult] = []
    for sample in samples:
        expected = normalize_text(sample.expected)
        student = normalize_text(sample.student)
        repeated = max_repeat_run(sample.student)
        correct = bool(expected and expected in student)
        if repeated >= 3:
            problem = "repetition"
            score = 1.0 if not correct else 2.0
        elif correct:
            problem = "none"
            score = 5.0
        elif not student:
            problem = "empty"
            score = 0.0
        else:
            problem = "wrong_answer"
            score = 1.0
        results.append(
            JudgeResult(
                lesson_id=sample.lesson_id,
                correct=correct,
                score=score,
                problem=problem,
                correction=sample.expected,
            )
        )
    return results


def problem_to_focus(problem: str) -> str:
    mapping = {
        "none": "balanced",
        "unknown": "balanced",
        "empty": "answer_start_practice",
        "repetition": "clean_single_answer",
        "wrong_answer": "object_answer_contrast",
        "grammar": "simple_sentence_practice",
        "off_topic": "stay_on_question",
    }
    return mapping.get(problem, "balanced")


def coerce_judge_results(payload: Any, samples: list[ProbeSample]) -> list[JudgeResult]:
    raw_items = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise ValueError("judge response must contain a results list")
    by_id = {sample.lesson_id: sample for sample in samples}
    results: list[JudgeResult] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        lesson_id = bounded(item.get("id") or item.get("lesson_id"), 80)
        if not lesson_id or lesson_id not in by_id:
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        results.append(
            JudgeResult(
                lesson_id=lesson_id,
                correct=bool(item.get("correct", False)),
                score=max(0.0, min(5.0, score)),
                problem=bounded(item.get("problem") or "unknown", 80),
                correction=bounded(item.get("correction"), 240),
            )
        )
    return results


def build_judge_messages(samples: list[ProbeSample]) -> list[dict[str, str]]:
    system = (
        "You judge a tiny student language model. Compare the student's answer "
        "with the expected answer. Return JSON only, no markdown."
    )
    items = [
        {
            "id": sample.lesson_id,
            "question": sample.question,
            "expected": sample.expected,
            "student": sample.student[:500],
        }
        for sample in samples
    ]
    schema = {
        "results": [
            {
                "id": "same_id",
                "correct": False,
                "score": 0,
                "problem": "wrong_answer | repetition | grammar | off_topic | empty | none",
                "correction": "short corrected answer",
            }
        ]
    }
    user = (
        "Score each item from 0 to 5. Mark correct only if the core meaning is right. "
        "If the output repeats words, use problem='repetition'.\n\n"
        f"Items:\n{json.dumps(items, ensure_ascii=True)}\n\n"
        f"Return this JSON shape:\n{json.dumps(schema, ensure_ascii=True)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_summary(results: list[JudgeResult]) -> dict[str, Any]:
    if not results:
        return {"judge_count": 0, "judge_correct": 0.0, "judge_score": float("nan"), "top_problem": "none"}
    counts: dict[str, int] = {}
    for item in results:
        counts[item.problem] = counts.get(item.problem, 0) + 1
    top_problem = max(counts.items(), key=lambda kv: kv[1])[0]
    return {
        "judge_count": int(len(results)),
        "judge_correct": float(sum(1 for item in results if item.correct) / len(results)),
        "judge_score": float(sum(item.score for item in results) / len(results)),
        "top_problem": top_problem,
    }


def lesson_summary(round_idx: int, split: str, lesson: Lesson) -> dict[str, Any]:
    text = lesson_to_training_text(lesson)
    return {
        "round": int(round_idx),
        "split": split,
        "id": lesson.lesson_id,
        "skill": lesson.skill,
        "difficulty": int(lesson.difficulty),
        "prompt_chars": int(len(lesson.prompt)),
        "target_chars": int(len(lesson.target)),
        "probe_chars": int(len(lesson.probe)),
        "expected_chars": int(len(lesson.expected)),
        "text_sha256_16": stable_hash(text),
    }


def save_checkpoint(path: Path, model: TorchU017ErrorPopulationModel, cfg: u012.U012Config, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "input_codes": model.input_codes.detach().cpu(),
            "target_codes": model.target_codes.detach().cpu(),
            "position_codes": model.position_codes.detach().cpu(),
            "ff_weights": [tensor.detach().cpu() for tensor in model.ff_weights],
            "ff_biases": [tensor.detach().cpu() for tensor in model.ff_biases],
            "attn_q": [tensor.detach().cpu() for tensor in model.attn_q],
            "attn_k": [tensor.detach().cpu() for tensor in model.attn_k],
            "attn_v": [tensor.detach().cpu() for tensor in model.attn_v],
            "attn_o": [tensor.detach().cpu() for tensor in model.attn_o],
            "output_weights": model.output_weights.detach().cpu(),
            "output_counts": model.output_counts.detach().cpu(),
            "output_total": model.output_total.detach().cpu(),
            "output_bias": model.output_bias.detach().cpu(),
            "error_weights": [tensor.detach().cpu() for tensor in model.error_weights],
            "step": int(model.step),
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    load_dotenv_file(SCRIPT_DIR / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "u020_online_teacher_u017")
    parser.add_argument("--tokenizer", type=Path, default=u012.DEFAULT_TOKENIZER)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--teacher-mode", choices=["mock", "deepseek"], default="mock")
    parser.add_argument("--judge-mode", choices=["auto", "none", "mock", "deepseek"], default="auto")
    parser.add_argument("--api-base-url", type=str, default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-model", type=str, default="deepseek-chat")
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--api-temperature", type=float, default=0.7)
    parser.add_argument("--api-max-tokens", type=int, default=2500)
    parser.add_argument("--judge-max-tokens", type=int, default=1200)
    parser.add_argument("--max-api-requests", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--lessons-per-round", type=int, default=12)
    parser.add_argument("--holdout-per-round", type=int, default=3)
    parser.add_argument("--probe-bank-size", type=int, default=64)
    parser.add_argument("--probe-eval-count", type=int, default=24)
    parser.add_argument("--eval-every-rounds", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--sample-new-tokens", type=int, default=32)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--start-difficulty", type=int, default=1)
    parser.add_argument("--max-difficulty", type=int, default=6)
    parser.add_argument("--initial-focus", type=str, default="balanced")
    parser.add_argument("--max-field-chars", type=int, default=700)
    parser.add_argument("--save-lesson-text", action="store_true")
    parser.add_argument("--save-api-payloads", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--train-mode", choices=["packed", "chunk", "exact"], default="packed")
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--attn-rank", type=int, default=128)
    parser.add_argument("--output-lr", type=float, default=0.030)
    parser.add_argument("--hidden-lr", type=float, default=0.001)
    parser.add_argument("--hidden-bias-lr", type=float, default=0.0002)
    parser.add_argument("--embedding-lr", type=float, default=0.0005)
    parser.add_argument("--logit-scale", type=float, default=2.0)
    parser.add_argument("--attention-scale", type=float, default=0.70)
    parser.add_argument("--ff-scale", type=float, default=0.35)
    parser.add_argument("--position-scale", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--row-norm-interval", type=int, default=512)
    parser.add_argument("--bias-alpha", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--chunk-tokens", type=int, default=1000)
    parser.add_argument("--pack-eos-id", type=int, default=-1)
    parser.add_argument("--error-mode", choices=["transpose", "fixed"], default="transpose")
    parser.add_argument("--error-noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    # cold-start warmup
    parser.add_argument("--warmup-source", choices=["none", "wikitext", "url"], default="none",
                        help="pretrain on generic text before teacher loop")
    parser.add_argument("--warmup-docs", type=int, default=500,
                        help="number of text segments to use for warmup")
    parser.add_argument("--warmup-passes", type=int, default=1,
                        help="how many passes over warmup docs")
    parser.add_argument("--warmup-urls", type=str, default="",
                        help="comma-separated URLs to fetch for warmup (used when --warmup-source=url)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    tokenizer = u012.AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    if int(args.pack_eos_id) < 0:
        args.pack_eos_id = int(tokenizer.eos_token_id if tokenizer.eos_token_id is not None else len(tokenizer) - 1)

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

    if args.warmup_source != "none":
        warmup_urls = [u.strip() for u in args.warmup_urls.split(",") if u.strip()]
        run_warmup(
            model=model,
            tokenizer=tokenizer,
            source=args.warmup_source,
            warmup_docs=args.warmup_docs,
            warmup_passes=args.warmup_passes,
            warmup_urls=warmup_urls,
            chunk_size=args.chunk_size,
            chunk_tokens=args.chunk_tokens,
            out_dir=args.out_dir,
        )

    judge_mode = args.judge_mode
    if judge_mode == "auto":
        judge_mode = "deepseek" if args.teacher_mode == "deepseek" else "mock"

    client: OpenAICompatibleClient | None = None
    if args.teacher_mode == "deepseek" or judge_mode == "deepseek":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"API key not found in ${args.api_key_env}")
        client = OpenAICompatibleClient(args.api_base_url, api_key, args.api_model, args.api_timeout)

    api_requests = 0
    difficulty = max(int(args.start_difficulty), 1)
    focus = str(args.initial_focus)
    recent_errors = ""
    last_metrics: dict[str, Any] = {"probe_loss": None, "judge_score": None}
    probe_bank: list[Lesson] = []
    metrics_rows: list[dict[str, Any]] = []
    total_train_tokens = 0

    start_time = time.perf_counter()
    for round_idx in range(max(int(args.rounds), 0)):
        needed = max(int(args.lessons_per_round), 1) + max(int(args.holdout_per_round), 0)
        if args.teacher_mode == "mock":
            lessons = mock_lessons(round_idx, needed, difficulty, focus, args.seed)
        else:
            if client is None:
                raise RuntimeError("DeepSeek client was not initialized")
            if api_requests >= int(args.max_api_requests):
                raise RuntimeError("max API request budget reached before teacher call")
            messages = build_teacher_messages(round_idx, needed, difficulty, focus, last_metrics, recent_errors)
            response = client.chat(messages, max_tokens=args.api_max_tokens, temperature=args.api_temperature)
            api_requests += 1
            text = response_text(response)
            if args.save_api_payloads:
                append_jsonl(
                    args.out_dir / "api_payloads.jsonl",
                    {"round": round_idx, "kind": "teacher", "messages": messages, "response_text": text},
                )
            lessons = coerce_lessons(extract_json_payload(text), round_idx, args.max_field_chars)
        if len(lessons) < 1:
            raise RuntimeError(f"teacher returned no usable lessons in round {round_idx}")

        train_lessons = lessons[: max(int(args.lessons_per_round), 1)]
        holdout_lessons = lessons[max(int(args.lessons_per_round), 1) :]
        for lesson in train_lessons:
            append_jsonl(args.out_dir / "lesson_summaries.jsonl", lesson_summary(round_idx, "train", lesson))
            if args.save_lesson_text:
                append_jsonl(args.out_dir / "lesson_text.jsonl", {"round": round_idx, "split": "train", **asdict(lesson)})
        for lesson in holdout_lessons:
            append_jsonl(args.out_dir / "lesson_summaries.jsonl", lesson_summary(round_idx, "holdout", lesson))
            if args.save_lesson_text:
                append_jsonl(args.out_dir / "lesson_text.jsonl", {"round": round_idx, "split": "holdout", **asdict(lesson)})
        probe_bank.extend(holdout_lessons)
        if len(probe_bank) > max(int(args.probe_bank_size), 1):
            probe_bank = probe_bank[-max(int(args.probe_bank_size), 1) :]

        train_docs = tokenize_lessons(tokenizer, train_lessons)
        train_t0 = time.perf_counter()
        train_summary, _ = u012.run_documents(
            model,
            train_docs,
            update=True,
            mode=args.train_mode,
            chunk_size=args.chunk_size,
            chunk_tokens=args.chunk_tokens,
            eos_id=args.pack_eos_id,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_t0
        total_train_tokens += int(train_summary["tokens"])

        do_eval = (round_idx + 1) % max(int(args.eval_every_rounds), 1) == 0
        probe_eval = {"loss": float("nan"), "ppl": float("nan"), "accuracy": 0.0, "tokens": 0}
        sample_rows: list[ProbeSample] = []
        judge_results: list[JudgeResult] = []
        if do_eval and probe_bank:
            probes = probe_bank[-max(int(args.probe_eval_count), 1) :]
            probe_eval = evaluate_answer_ce(model, tokenizer, probes, batch_size=max(int(args.chunk_size), 1) * 16)
            sample_rows = generate_probe_samples(
                model,
                tokenizer,
                probes,
                args.sample_count,
                args.sample_new_tokens,
                args.sample_temperature,
            )
            if judge_mode == "mock":
                judge_results = mock_judge(sample_rows)
            elif judge_mode == "deepseek" and sample_rows:
                if client is None:
                    raise RuntimeError("DeepSeek client was not initialized")
                if api_requests >= int(args.max_api_requests):
                    raise RuntimeError("max API request budget reached before judge call")
                messages = build_judge_messages(sample_rows)
                response = client.chat(messages, max_tokens=args.judge_max_tokens, temperature=0.0)
                api_requests += 1
                text = response_text(response)
                if args.save_api_payloads:
                    append_jsonl(
                        args.out_dir / "api_payloads.jsonl",
                        {"round": round_idx, "kind": "judge", "messages": messages, "response_text": text},
                    )
                judge_results = coerce_judge_results(extract_json_payload(text), sample_rows)

            for sample in sample_rows:
                append_jsonl(args.out_dir / "probe_samples.jsonl", {"round": round_idx, **asdict(sample)})
            for result in judge_results:
                append_jsonl(args.out_dir / "judge_results.jsonl", {"round": round_idx, **asdict(result)})

        judged = judge_summary(judge_results)
        if do_eval:
            top_problem = str(judged["top_problem"])
            if top_problem not in ("none", "unknown") and float(judged.get("judge_correct", 0.0)) < 0.75:
                focus = problem_to_focus(top_problem)
            elif not math.isnan(float(probe_eval["loss"])) and float(probe_eval["loss"]) < 5.0:
                focus = "balanced"
                difficulty = min(difficulty + 1, max(int(args.max_difficulty), 1))
            if judge_results:
                recent_errors = "; ".join(
                    f"{item.problem}:{item.correction or item.lesson_id}"
                    for item in judge_results
                    if not item.correct
                )[:500]
            last_metrics = {
                "train_loss": train_summary["loss"],
                "probe_loss": probe_eval["loss"],
                "probe_accuracy": probe_eval["accuracy"],
                "judge_correct": judged["judge_correct"],
                "judge_score": judged["judge_score"],
                "top_problem": judged["top_problem"],
            }

        row = {
            "round": int(round_idx),
            "difficulty": int(difficulty),
            "focus": focus,
            "train_lessons": int(len(train_lessons)),
            "holdout_lessons": int(len(holdout_lessons)),
            "probe_bank": int(len(probe_bank)),
            "train_loss": train_summary["loss"],
            "train_accuracy": train_summary["accuracy"],
            "train_tokens": train_summary["tokens"],
            "probe_loss": probe_eval["loss"],
            "probe_accuracy": probe_eval["accuracy"],
            "probe_tokens": probe_eval["tokens"],
            "judge_count": judged["judge_count"],
            "judge_correct": judged["judge_correct"],
            "judge_score": judged["judge_score"],
            "top_problem": judged["top_problem"],
            "train_seconds": float(train_seconds),
            "train_tokens_per_second": float(train_summary["tokens"] / max(train_seconds, 1e-9)),
            "api_requests": int(api_requests),
            "total_train_tokens": int(total_train_tokens),
        }
        metrics_rows.append(row)
        write_csv(args.out_dir / "metrics.csv", metrics_rows)
        print(
            f"round={round_idx} train_loss={float(train_summary['loss']):.4f} "
            f"probe_loss={float(probe_eval['loss']):.4f} "
            f"judge={float(judged['judge_correct']):.3f} focus={focus} "
            f"tokens={int(train_summary['tokens'])} api={api_requests}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    max_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    summary = {
        "teacher_mode": args.teacher_mode,
        "judge_mode": judge_mode,
        "rounds": int(args.rounds),
        "tokenizer_len": int(len(tokenizer)),
        "context_len": int(args.context_len),
        "d_model": int(args.d_model),
        "blocks": int(args.blocks),
        "attn_rank": int(args.attn_rank),
        "parameters": int(model.parameter_count()),
        "state_bytes": int(model.state_bytes()),
        "total_train_tokens": int(total_train_tokens),
        "elapsed_seconds": float(elapsed),
        "api_requests": int(api_requests),
        "max_cuda_mem_bytes": int(max_mem),
        "final_focus": focus,
        "final_difficulty": int(difficulty),
    }
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "config": asdict(cfg),
                "summary": summary,
                "metrics": metrics_rows,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )
    if args.save_checkpoint:
        save_checkpoint(args.out_dir / "u017_online_checkpoint.pt", model, cfg, args)

    print("Summary:")
    print(
        f"  teacher={args.teacher_mode} judge={judge_mode} params={model.parameter_count():,} "
        f"state={model.state_bytes() / (1024 ** 2):.1f} MiB"
    )
    print(
        f"  tokens={total_train_tokens} elapsed={elapsed:.2f}s "
        f"api_requests={api_requests} max_mem={max_mem / (1024 ** 3):.2f} GiB"
    )
    print(f"  metrics: {args.out_dir / 'metrics.csv'}")
    print(f"  samples: {args.out_dir / 'probe_samples.jsonl'}")


if __name__ == "__main__":
    main()
