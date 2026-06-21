#!/usr/bin/env python3
"""
U021 kindergarten teacher agent for the U017 no-BP learner.

Three-level curriculum: vocabulary -> sentence -> story.
DeepSeek (or mock) sees the student's raw outputs and per-level losses,
writes a diagnosis, decides the lesson mix, and generates lessons.
The student (U017) learns every lesson as plain next-token text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from pathlib import Path
from typing import Any

import numpy as np
import torch

import u012_u009_torch_fast_variants as u012
from u017_error_population_llm import TorchU017ErrorPopulationModel
from u020_online_teacher_u017 import (
    OpenAICompatibleClient,
    JudgeResult,
    ProbeSample,
    mock_judge,
    judge_summary,
    build_judge_messages,
    coerce_judge_results,
    append_jsonl,
    write_csv,
    bounded,
    run_warmup,
    fetch_warmup_docs_wikitext,
    fetch_warmup_docs_url,
    encode_text,
    response_text,
    extract_json_payload,
)

SCRIPT_DIR = Path(__file__).resolve().parent
LEVEL_ORDER = ["vocabulary", "sentence", "story"]


# ── dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class Lesson:
    lesson_id: str
    lesson_type: str   # "vocabulary" | "sentence" | "story"
    skill: str
    difficulty: int
    content: str       # full teaching text (replaces prompt+target from u020)
    probe: str
    expected: str
    notes: str = ""


@dataclass
class StudentState:
    current_level: str = "vocabulary"
    vocab_loss: float = float("nan")
    sentence_loss: float = float("nan")
    story_loss: float = float("nan")
    recent_raw_outputs: list = field(default_factory=list)
    judge_correct_by_type: dict = field(default_factory=dict)
    diagnosis: str = ""


# ── text formatting ───────────────────────────────────────────────────────────

def lesson_to_training_text(lesson: Lesson) -> str:
    prefix = "Story:\n" if lesson.lesson_type == "story" else ""
    return (
        f"{prefix}{lesson.content.strip()}\n\n"
        f"Question:\n{lesson.probe.strip()}\n\n"
        f"Answer:\n{lesson.expected.strip()}\n"
    )


def lesson_to_probe_prompt(lesson: Lesson) -> str:
    prefix = "Story:\n" if lesson.lesson_type == "story" else ""
    return (
        f"{prefix}{lesson.content.strip()}\n\n"
        f"Question:\n{lesson.probe.strip()}\n\n"
        f"Answer:\n"
    )


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
) -> dict[str, float]:
    contexts: list[np.ndarray] = []
    targets: list[int] = []
    for lesson in probes:
        prefix = encode_text(tokenizer, lesson_to_probe_prompt(lesson))
        answer = encode_text(tokenizer, lesson.expected.strip() + "\n")
        if prefix.size == 0 or answer.size == 0:
            continue
        history = [int(x) for x in prefix.tolist()]
        for token in answer.tolist():
            contexts.append(np.asarray(history[-model.context_len:], dtype=np.int64))
            targets.append(int(token))
            history.append(int(token))
    if not targets:
        return {"loss": float("nan"), "accuracy": 0.0, "tokens": 0}
    loss_sum = 0.0
    correct = 0
    total = 0
    for start in range(0, len(targets), max(int(batch_size), 1)):
        chunk_ctx = contexts[start: start + max(int(batch_size), 1)]
        chunk_tgt = np.asarray(targets[start: start + max(int(batch_size), 1)], dtype=np.int64)
        bl, bc, bt = u012.predict_chunk(model, chunk_ctx, chunk_tgt)
        loss_sum += bl; correct += bc; total += bt
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
    for lesson in probes[:max(int(count), 0)]:
        prompt = encode_text(tokenizer, lesson_to_probe_prompt(lesson))
        if prompt.size == 0:
            continue
        generated = model.generate(prompt, max_new_tokens, sample=False, temperature=temperature)
        completion = tokenizer.decode(generated[int(prompt.size):], skip_special_tokens=False)
        samples.append(ProbeSample(
            lesson_id=lesson.lesson_id,
            question=lesson.probe,
            expected=lesson.expected,
            student=completion.strip(),
        ))
    return samples


# ── per-type evaluation ───────────────────────────────────────────────────────

def evaluate_by_type(
    model: TorchU017ErrorPopulationModel,
    tokenizer: Any,
    probe_bank: list[Lesson],
    batch_size: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for lt in LEVEL_ORDER:
        bucket = [l for l in probe_bank if l.lesson_type == lt]
        if not bucket:
            result[lt] = float("nan")
            continue
        ev = evaluate_answer_ce(model, tokenizer, bucket, batch_size)
        result[lt] = float(ev["loss"])
    return result


def judge_correct_by_type(
    judge_results: list[JudgeResult],
    lesson_by_id: dict[str, Lesson],
) -> dict[str, float]:
    counts: dict[str, list[float]] = {}
    for r in judge_results:
        lt = lesson_by_id[r.lesson_id].lesson_type if r.lesson_id in lesson_by_id else "story"
        counts.setdefault(lt, []).append(1.0 if r.correct else 0.0)
    return {lt: sum(v) / len(v) for lt, v in counts.items()}


# ── mock curriculum ───────────────────────────────────────────────────────────

def mock_lessons(
    round_idx: int,
    count: int,
    current_level: str,
    difficulty: int,
    focus: str,
    seed: int,
) -> list[Lesson]:
    rng = random.Random(seed + 1009 * round_idx + 17 * difficulty)
    names   = ["Lily", "Tom", "Mia", "Ben", "Nora", "Sam", "Ava", "Leo"]
    colors  = ["red", "blue", "green", "yellow", "white", "black"]
    objects = ["ball", "cup", "hat", "box", "kite", "book"]
    places  = ["bed", "chair", "tree", "door", "table", "bench"]
    actions = ["held", "washed", "opened", "carried", "found", "shared"]
    words   = objects + colors  # vocabulary pool

    lessons: list[Lesson] = []
    for idx in range(count):
        lid = f"mock_r{round_idx:03d}_{idx:03d}"
        if current_level == "vocabulary":
            word = rng.choice(words)
            content  = f"{word} = {word}. This is a {word}."
            probe    = "What is this?"
            expected = word
            skill    = "vocabulary"
            lt       = "vocabulary"
        elif current_level == "sentence":
            name  = rng.choice(names)
            color = rng.choice(colors)
            obj   = rng.choice(objects)
            content  = f"{name} has a {color} {obj}."
            probe    = f"What does {name} have?"
            expected = f"a {color} {obj}"
            skill    = "sentence"
            lt       = "sentence"
        else:  # story — mirror u020 difficulty tiers
            name   = rng.choice(names)
            color  = rng.choice(colors)
            obj    = rng.choice(objects)
            place  = rng.choice(places)
            action = rng.choice(actions)
            if difficulty <= 2:
                content  = f"{name} found a {color} {obj} by the {place}. {name} smiled."
                probe    = f"What did {name} find?"
                expected = f"a {color} {obj}"
                skill    = "object_memory"
            elif difficulty <= 4:
                content  = (f"{name} found a {color} {obj} by the {place}. "
                            f"{name} {action} the {obj} because it was useful.")
                probe    = f"Why did {name} {action} the {obj}?"
                expected = "it was useful"
                skill    = "simple_cause"
            else:
                content  = (f"{name} looked by the {place}. First {name} found a {color} {obj}. "
                            f"Then {name} {action} it and went home.")
                probe    = "What happened first?"
                expected = f"{name} found a {color} {obj}"
                skill    = "event_order"
            lt = "story"

        lessons.append(Lesson(
            lesson_id=lid, lesson_type=lt, skill=skill,
            difficulty=difficulty, content=content,
            probe=probe, expected=expected, notes="mock",
        ))
    return lessons


# ── teacher prompt builder ────────────────────────────────────────────────────

def build_teacher_messages(
    round_idx: int,
    lesson_count: int,
    state: StudentState,
    recent_errors: str,
) -> list[dict[str, str]]:
    system = (
        "You are a kindergarten teacher for a student model that has NEVER seen language before. "
        "Your job each round: (1) diagnose what the student knows from the evidence, "
        "(2) decide how many lessons of each type to write, (3) write the lessons. "
        "Return JSON only. No markdown."
    )

    def fmt_loss(v: float) -> str:
        return "n/a" if math.isnan(v) else f"{v:.2f}"

    raw_str = json.dumps(state.recent_raw_outputs[-4:], ensure_ascii=True)
    correct_str = json.dumps(state.judge_correct_by_type, ensure_ascii=True)

    schema = {
        "diagnosis": "1-2 sentences about what the student knows/struggles with",
        "lessons": [
            {
                "id": "r0_0",
                "lesson_type": "vocabulary | sentence | story",
                "skill": "short tag",
                "difficulty": 1,
                "content": "teaching text here",
                "probe": "question here",
                "expected": "answer here",
                "notes": "",
            }
        ],
    }

    user = (
        f"Round: {round_idx}\n"
        f"Need {lesson_count} lessons total.\n"
        f"Current student level: {state.current_level}\n\n"
        f"Loss by lesson type (lower = better learned):\n"
        f"  vocabulary : {fmt_loss(state.vocab_loss)}\n"
        f"  sentence   : {fmt_loss(state.sentence_loss)}\n"
        f"  story      : {fmt_loss(state.story_loss)}\n\n"
        f"Judge correct rate by type: {correct_str}\n\n"
        f"Promotion criteria (code enforces these as a floor — you decide the pace):\n"
        f"  vocabulary -> sentence : vocab_loss < 3.5\n"
        f"  sentence   -> story    : sentence_correct >= 0.60\n\n"
        f"Student's last raw outputs (question / expected / what student said):\n"
        f"{raw_str}\n\n"
        f"Your last diagnosis: {state.diagnosis or 'none'}\n"
        f"Recent errors: {recent_errors or 'none'}\n\n"
        "=== Lesson writing rules ===\n"
        "vocabulary lesson:\n"
        "  content: 'WORD = WORD. This is a WORD.'  (max 8 words)\n"
        "  probe: 'What is this?' or 'Say: WORD'\n"
        "  expected: the word itself\n\n"
        "sentence lesson:\n"
        "  content: 1-2 SVO sentences, max 10 words each, use concrete nouns\n"
        "  probe: 'What does NAME have/do/find?'\n"
        "  expected: 3-6 words\n\n"
        "story lesson:\n"
        "  content: 2-3 sentences, max 30 words total\n"
        "  probe: who/what/why question\n"
        "  expected: max 8 words\n\n"
        "Start with vocabulary unless the student has already mastered it. "
        "Mix types freely to reinforce weak areas. "
        "Toddler-simple language only.\n\n"
        f"JSON schema:\n{json.dumps(schema, ensure_ascii=True)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def coerce_lessons(
    payload: Any,
    round_idx: int,
    max_field_chars: int,
) -> tuple[list[Lesson], str]:
    """Parse API/mock JSON into (lessons, diagnosis)."""
    diagnosis = ""
    if isinstance(payload, dict):
        diagnosis = bounded(payload.get("diagnosis", ""), 400)
        raw_list = payload.get("lessons", [])
    elif isinstance(payload, list):
        raw_list = payload
    else:
        raise ValueError("teacher response must contain a lessons list")

    valid_types = {"vocabulary", "sentence", "story"}
    lessons: list[Lesson] = []
    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            continue
        lt = str(item.get("lesson_type") or "vocabulary").strip().lower()
        if lt not in valid_types:
            lt = "vocabulary"
        lid = bounded(item.get("id") or item.get("lesson_id") or f"r{round_idx:03d}_{i:03d}", 80)
        lessons.append(Lesson(
            lesson_id=lid,
            lesson_type=lt,
            skill=bounded(item.get("skill", "general"), 60),
            difficulty=max(1, min(int(item.get("difficulty", 1)), 6)),
            content=bounded(item.get("content", ""), max_field_chars),
            probe=bounded(item.get("probe", ""), max_field_chars),
            expected=bounded(item.get("expected", ""), 120),
            notes=bounded(item.get("notes", ""), 200),
        ))
    return lessons, diagnosis


# ── level promotion logic ─────────────────────────────────────────────────────

def maybe_promote(
    state: StudentState,
    difficulty: int,
    max_difficulty: int,
) -> tuple[str, int]:
    """Code-side floor for level promotion; DeepSeek decides the mix, this enforces minimums."""
    level = state.current_level
    if level == "vocabulary" and not math.isnan(state.vocab_loss) and state.vocab_loss < 3.5:
        level = "sentence"
    elif level == "sentence":
        sc = state.judge_correct_by_type.get("sentence", 0.0)
        if sc >= 0.60:
            level = "story"
    elif level == "story":
        sc = state.judge_correct_by_type.get("story", 0.0)
        sl = state.story_loss
        if sc >= 0.75 and not math.isnan(sl) and sl < 5.0:
            difficulty = min(difficulty + 1, max(max_difficulty, 1))
    return level, difficulty


# ── verbose per-round printer ─────────────────────────────────────────────────

def print_round(
    round_idx: int,
    state: StudentState,
    train_loss: float,
    type_losses: dict[str, float],
    samples: list[ProbeSample],
    judge_results: list[JudgeResult],
    lesson_by_id: dict[str, Lesson],
    diagnosis: str,
) -> None:
    def fl(v: float) -> str:
        return "n/a" if math.isnan(v) else f"{v:.3f}"

    correct_by_type = judge_correct_by_type(judge_results, lesson_by_id)
    correct_str = "  ".join(
        f"{lt}={correct_by_type.get(lt, float('nan')):.0%}" if not math.isnan(correct_by_type.get(lt, float('nan'))) else f"{lt}=n/a"
        for lt in LEVEL_ORDER
    )
    SEP = "=" * 72
    print(f"\n{SEP}")
    print(f"  ROUND {round_idx}  level={state.current_level}  train_loss={fl(train_loss)}")
    print(f"  losses  vocab={fl(type_losses.get('vocabulary', float('nan')))}  "
          f"sentence={fl(type_losses.get('sentence', float('nan')))}  "
          f"story={fl(type_losses.get('story', float('nan')))}")
    print(f"  judge   {correct_str}")
    print(f"  diagnosis: {diagnosis or 'none'}")
    print(SEP)
    for s, r in zip(samples, judge_results):
        lt = lesson_by_id[r.lesson_id].lesson_type if r.lesson_id in lesson_by_id else "?"
        tick = "✓" if r.correct else "✗"
        print(f"  {tick} [{lt}] q={s.question!r}")
        print(f"       expected : {s.expected!r}")
        print(f"       student  : {s.student!r}")
        print(f"       judge    : score={r.score}  problem={r.problem}")
    print()


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="U021 kindergarten teacher loop")
    parser.add_argument("--teacher-mode", choices=["mock", "deepseek"], default="mock")
    parser.add_argument("--judge-mode", choices=["auto", "none", "mock", "deepseek"], default="auto")
    parser.add_argument("--api-base-url", type=str,
                        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-model", type=str, default="deepseek-chat")
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--api-temperature", type=float, default=0.7)
    parser.add_argument("--api-max-tokens", type=int, default=3000)
    parser.add_argument("--judge-max-tokens", type=int, default=1200)
    parser.add_argument("--max-api-requests", type=int, default=60)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--lessons-per-round", type=int, default=6)
    parser.add_argument("--holdout-per-round", type=int, default=2)
    parser.add_argument("--probe-bank-size", type=int, default=64)
    parser.add_argument("--probe-eval-count", type=int, default=12)
    parser.add_argument("--eval-every-rounds", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--sample-new-tokens", type=int, default=24)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--start-difficulty", type=int, default=1)
    parser.add_argument("--max-difficulty", type=int, default=6)
    parser.add_argument("--max-field-chars", type=int, default=700)
    parser.add_argument("--save-api-payloads", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--out-dir", type=Path,
                        default=SCRIPT_DIR / "output" / "u021_kindergarten")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--device", type=str, default="cpu")
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
    parser.add_argument("--warmup-source", choices=["none", "wikitext", "url"], default="none")
    parser.add_argument("--warmup-docs", type=int, default=300)
    parser.add_argument("--warmup-passes", type=int, default=3)
    parser.add_argument("--warmup-urls", type=str, default="")
    return parser


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
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
        int(len(tokenizer)), cfg, device,
        error_mode=args.error_mode,
        error_noise=args.error_noise,
    )

    if args.warmup_source != "none":
        warmup_urls = [u.strip() for u in args.warmup_urls.split(",") if u.strip()]
        run_warmup(
            model=model, tokenizer=tokenizer,
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

    state = StudentState()
    difficulty = max(int(args.start_difficulty), 1)
    probe_bank: list[Lesson] = []
    lesson_by_id: dict[str, Lesson] = {}
    metrics_rows: list[dict[str, Any]] = []
    recent_errors = ""
    api_requests = 0
    total_train_tokens = 0
    start_time = time.perf_counter()

    for round_idx in range(max(int(args.rounds), 0)):
        needed = max(int(args.lessons_per_round), 1) + max(int(args.holdout_per_round), 0)

        # ── generate lessons ──────────────────────────────────────────────
        diagnosis = ""
        if args.teacher_mode == "mock":
            lessons = mock_lessons(round_idx, needed, state.current_level, difficulty, "balanced", args.seed)
        else:
            if client is None:
                raise RuntimeError("DeepSeek client not initialized")
            if api_requests >= int(args.max_api_requests):
                raise RuntimeError("max API request budget reached")
            messages = build_teacher_messages(round_idx, needed, state, recent_errors)
            resp = client.chat(messages, max_tokens=args.api_max_tokens, temperature=args.api_temperature)
            api_requests += 1
            text = response_text(resp)
            if args.save_api_payloads:
                append_jsonl(args.out_dir / "api_payloads.jsonl",
                             {"round": round_idx, "kind": "teacher", "messages": messages, "response_text": text})
            lessons, diagnosis = coerce_lessons(extract_json_payload(text), round_idx, args.max_field_chars)

        if not lessons:
            raise RuntimeError(f"teacher returned no usable lessons in round {round_idx}")

        train_lessons = lessons[:max(int(args.lessons_per_round), 1)]
        holdout_lessons = lessons[max(int(args.lessons_per_round), 1):]

        for l in train_lessons + holdout_lessons:
            lesson_by_id[l.lesson_id] = l
            append_jsonl(args.out_dir / "lesson_summaries.jsonl", {
                "round": round_idx, "split": "train" if l in train_lessons else "holdout",
                **asdict(l),
            })

        probe_bank.extend(holdout_lessons)
        if len(probe_bank) > max(int(args.probe_bank_size), 1):
            probe_bank = probe_bank[-max(int(args.probe_bank_size), 1):]

        # ── train ─────────────────────────────────────────────────────────
        docs = tokenize_lessons(tokenizer, train_lessons)
        train_t0 = time.perf_counter()
        train_summary, _ = u012.run_documents(
            model, docs, update=True, mode="packed",
            chunk_size=args.chunk_size, chunk_tokens=args.chunk_tokens,
            eos_id=args.pack_eos_id,
        )
        train_seconds = time.perf_counter() - train_t0
        total_train_tokens += int(train_summary["tokens"])

        # ── eval ──────────────────────────────────────────────────────────
        do_eval = (round_idx + 1) % max(int(args.eval_every_rounds), 1) == 0
        type_losses: dict[str, float] = {lt: float("nan") for lt in LEVEL_ORDER}
        samples: list[ProbeSample] = []
        judge_results: list[JudgeResult] = []

        if do_eval and probe_bank:
            probes = probe_bank[-max(int(args.probe_eval_count), 1):]
            batch_sz = max(int(args.chunk_size), 1) * 16
            type_losses = evaluate_by_type(model, tokenizer, probes, batch_sz)

            samples = generate_probe_samples(
                model, tokenizer, probes,
                args.sample_count, args.sample_new_tokens, args.sample_temperature,
            )

            if judge_mode == "mock":
                judge_results = mock_judge(samples)
            elif judge_mode == "deepseek" and samples:
                if client is None:
                    raise RuntimeError("DeepSeek client not initialized")
                if api_requests >= int(args.max_api_requests):
                    raise RuntimeError("max API request budget reached")
                messages = build_judge_messages(samples)
                resp = client.chat(messages, max_tokens=args.judge_max_tokens, temperature=0.0)
                api_requests += 1
                text = response_text(resp)
                if args.save_api_payloads:
                    append_jsonl(args.out_dir / "api_payloads.jsonl",
                                 {"round": round_idx, "kind": "judge", "messages": messages, "response_text": text})
                judge_results = coerce_judge_results(extract_json_payload(text), samples)

            for s in samples:
                append_jsonl(args.out_dir / "probe_samples.jsonl", {"round": round_idx, **asdict(s)})
            for r in judge_results:
                append_jsonl(args.out_dir / "judge_results.jsonl", {"round": round_idx, **asdict(r)})

            # update state
            state.vocab_loss = type_losses.get("vocabulary", float("nan"))
            state.sentence_loss = type_losses.get("sentence", float("nan"))
            state.story_loss = type_losses.get("story", float("nan"))
            state.judge_correct_by_type = judge_correct_by_type(judge_results, lesson_by_id)
            state.recent_raw_outputs = [
                {"type": lesson_by_id[s.lesson_id].lesson_type if s.lesson_id in lesson_by_id else "?",
                 "probe": s.question, "expected": s.expected, "student": s.student[:120]}
                for s in samples
            ]
            if diagnosis:
                state.diagnosis = diagnosis

            recent_errors = "; ".join(
                f"{r.problem}:{r.correction or r.lesson_id}"
                for r in judge_results if not r.correct
            )[:500]

            # level promotion (code floor)
            new_level, difficulty = maybe_promote(state, difficulty, int(args.max_difficulty))
            if new_level != state.current_level:
                print(f"[level up] {state.current_level} -> {new_level}", flush=True)
                state.current_level = new_level

            print_round(round_idx, state, float(train_summary["loss"]),
                        type_losses, samples, judge_results, lesson_by_id, diagnosis)

        judged = judge_summary(judge_results)
        row = {
            "round": round_idx,
            "current_level": state.current_level,
            "difficulty": difficulty,
            "train_loss": train_summary["loss"],
            "vocab_loss": type_losses.get("vocabulary", float("nan")),
            "sentence_loss": type_losses.get("sentence", float("nan")),
            "story_loss": type_losses.get("story", float("nan")),
            "judge_correct": judged["judge_correct"],
            "judge_score": judged["judge_score"],
            "top_problem": judged["top_problem"],
            "train_tokens": train_summary["tokens"],
            "total_train_tokens": total_train_tokens,
            "train_seconds": float(train_seconds),
            "api_requests": api_requests,
            "diagnosis": diagnosis,
        }
        metrics_rows.append(row)
        write_csv(args.out_dir / "metrics.csv", metrics_rows)

        if not do_eval:
            print(f"round={round_idx} train_loss={float(train_summary['loss']):.4f} "
                  f"level={state.current_level} tokens={train_summary['tokens']}", flush=True)

    elapsed = time.perf_counter() - start_time
    print(f"\nDone. {args.rounds} rounds in {elapsed:.1f}s. "
          f"final_level={state.current_level} total_tokens={total_train_tokens}", flush=True)

    if args.save_checkpoint:
        torch.save(model.state_dict(), args.out_dir / "model.pt")
        print(f"checkpoint saved to {args.out_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
