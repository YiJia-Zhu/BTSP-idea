#!/usr/bin/env python3
"""
U022 caregiver imitation teacher for the U017 no-BP learner.

The teacher first provides child-directed language exposure.  The student then
tries to imitate or continue a short prompt.  A judge decides whether the next
teacher turn should repeat, simplify, contrast, or advance.  The U017 core is
unchanged and learns every token in teacher_text with its local no-BP rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

import u012_u009_torch_fast_variants as u012
from u017_error_population_llm import TorchU017ErrorPopulationModel
from u020_online_teacher_u017 import (
    OpenAICompatibleClient,
    append_jsonl,
    bounded,
    encode_text,
    extract_json_payload,
    max_repeat_run,
    normalize_text,
    response_text,
    run_warmup,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PHASES = ["word", "phrase", "sentence", "cloze", "dialogue"]


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
class TeachingTurn:
    turn_id: str
    phase: str
    focus: str
    difficulty: int
    teacher_text: str
    imitation_prompt: str
    target_output: str
    notes: str = ""


@dataclass
class ImitationSample:
    turn_id: str
    phase: str
    focus: str
    prompt: str
    target: str
    student: str
    student_repr: str
    token_ids: list[int]


@dataclass
class JudgeResult:
    turn_id: str
    correct: bool
    score: float
    problem: str
    decision: str
    next_focus: str = ""
    correction: str = ""


@dataclass
class TeacherState:
    phase: str = "word"
    difficulty: int = 1
    focus: str = "basic words"
    mastery_streak: int = 0
    recent_errors: list[str] = field(default_factory=list)
    diagnosis: str = ""


def phase_index(phase: str) -> int:
    if phase in PHASES:
        return PHASES.index(phase)
    return 0


def clamp_phase(phase: str) -> str:
    phase = str(phase or "word").strip().lower()
    return phase if phase in PHASES else "word"


def next_phase(phase: str) -> str:
    idx = min(phase_index(phase) + 1, len(PHASES) - 1)
    return PHASES[idx]


def clean_training_text(text: str) -> str:
    lines = [" ".join(line.strip().split()) for line in str(text or "").splitlines()]
    joined = " ".join(line for line in lines if line)
    while "  " in joined:
        joined = joined.replace("  ", " ")
    return joined.strip()


def ensure_period(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    return text + "."


def mock_turns(round_idx: int, count: int, state: TeacherState, seed: int) -> tuple[list[TeachingTurn], str]:
    rng = random.Random(seed + 7919 * round_idx + 31 * state.difficulty + 3 * phase_index(state.phase))
    colors = ["red", "blue", "green", "yellow", "black", "white"]
    objects = ["ball", "cup", "hat", "box", "book", "kite"]
    names = ["Lily", "Tom", "Mia", "Ben", "Nora", "Sam"]
    actions = ["rolls", "falls", "jumps", "sits", "runs", "opens"]
    digits = ["one", "two", "three", "four"]
    letters = ["a", "b", "c", "d"]
    turns: list[TeachingTurn] = []
    for idx in range(max(int(count), 1)):
        phase = clamp_phase(state.phase)
        color = rng.choice(colors)
        obj = rng.choice(objects)
        name = rng.choice(names)
        action = rng.choice(actions)
        number = rng.choice(digits)
        letter = rng.choice(letters)

        if phase == "word":
            target = rng.choice([color, obj, number, letter])
            teacher_text = (
                f"{target} {target} hear {target} see {target} {target} again"
            )
            prompt = target
            expected = " " + target
            focus = target
        elif phase == "phrase":
            target = rng.choice([f"{color} {obj}", f"{number} {obj}", f"{name} {action}"])
            teacher_text = (
                f"{target} {target} hear {target} see {target} {target} again"
            )
            prompt = target
            expected = " " + target
            focus = target
        elif phase == "sentence":
            target = rng.choice(
                [
                    f"The {color} {obj} {action}.",
                    f"{name} has a {color} {obj}.",
                    f"I see {number} {obj}s.",
                ]
            )
            target = ensure_period(target)
            teacher_text = (
                f"{target} Teacher says {target} Child says {target} "
                f"Listen again. {target}"
            )
            prompt = "Teacher says " + target + " Child says"
            expected = " " + target
            focus = target
        elif phase == "cloze":
            teacher_text = (
                f"The {obj} is {color}. The {color} {obj} is here. "
                f"I see the {color} {obj}. The {obj} is {color}."
            )
            prompt = f"The {obj} is"
            expected = " " + ensure_period(color)
            focus = f"{obj} is {color}"
        else:
            teacher_text = (
                f"Teacher sees a {color} {obj}. The {obj} is {color}. "
                f"Teacher asks about the {obj}. Child says {color}."
            )
            prompt = f"The {obj} is"
            expected = " " + ensure_period(color)
            focus = f"{obj} color"

        turns.append(
            TeachingTurn(
                turn_id=f"mock_r{round_idx:03d}_{idx:03d}",
                phase=phase,
                focus=focus,
                difficulty=max(int(state.difficulty), 1),
                teacher_text=clean_training_text(teacher_text),
                imitation_prompt=prompt,
                target_output=expected,
                notes="mock",
            )
        )
    diagnosis = f"Mock caregiver phase={state.phase} difficulty={state.difficulty} focus={state.focus}."
    return turns, diagnosis


def build_teacher_messages(
    round_idx: int,
    turn_count: int,
    state: TeacherState,
    last_metrics: dict[str, Any],
    recent_samples: list[dict[str, Any]],
) -> list[dict[str, str]]:
    system = (
        "You are a caregiver teaching a text-only infant language model. "
        "First give simple child-directed speech. Then give a tiny imitation "
        "or continuation prompt. Return JSON only, no markdown."
    )
    schema = {
        "diagnosis": "short diagnosis",
        "turns": [
            {
                "id": "r0_0",
                "phase": "word | phrase | sentence | cloze | dialogue",
                "focus": "red ball",
                "difficulty": 1,
                "teacher_text": "Natural caregiver speech. Single paragraph. No blank lines.",
                "imitation_prompt": "Teacher says red ball. Child says",
                "target_output": " red ball.",
                "notes": "",
            }
        ],
    }
    user = (
        f"Round: {round_idx}\n"
        f"Need {turn_count} teaching turns.\n"
        f"Current phase: {state.phase}\n"
        f"Difficulty: {state.difficulty}\n"
        f"Current focus: {state.focus}\n"
        f"Mastery streak: {state.mastery_streak}\n"
        f"Recent errors: {'; '.join(state.recent_errors[-5:]) or 'none'}\n"
        f"Last metrics: {json.dumps(last_metrics, ensure_ascii=True)}\n"
        f"Recent student samples: {json.dumps(recent_samples[-4:], ensure_ascii=True)}\n\n"
        "Teaching rules:\n"
        "- Do not start with QA unless phase is dialogue.\n"
        "- The student learns every token in teacher_text, so avoid blank lines and repeated labels like Question/Answer.\n"
        "- teacher_text should sound like a caregiver speaking: simple, concrete, varied, and short.\n"
        "- Use single spaces and normal periods. No markdown.\n"
        "- The imitation prompt asks the student to repeat or continue one short phrase.\n"
        "- In word/phrase phases, the prompt can simply be the word or phrase, and target_output can be that word or phrase with a leading space.\n"
        "- Avoid overusing template words such as teacher, child, says, question, or answer.\n"
        "- In word and phrase phases, use very little punctuation; periods can dominate weak students.\n"
        "- If the student loops or outputs whitespace, repeat or simplify.\n"
        "- If the student succeeds for several turns, raise difficulty or move to the next phase.\n\n"
        f"JSON schema:\n{json.dumps(schema, ensure_ascii=True)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def coerce_turns(payload: Any, round_idx: int, max_field_chars: int) -> tuple[list[TeachingTurn], str]:
    diagnosis = ""
    raw_turns = payload
    if isinstance(payload, dict):
        diagnosis = bounded(payload.get("diagnosis", ""), 500)
        raw_turns = payload.get("turns", [])
    if not isinstance(raw_turns, list):
        raise ValueError("teacher response must contain a turns list")

    turns: list[TeachingTurn] = []
    for idx, item in enumerate(raw_turns):
        if not isinstance(item, dict):
            continue
        teacher_text = clean_training_text(bounded(item.get("teacher_text"), max_field_chars))
        prompt = bounded(item.get("imitation_prompt"), max_field_chars)
        target = bounded(item.get("target_output"), max_field_chars)
        if not teacher_text or not prompt or not target:
            continue
        try:
            difficulty = int(item.get("difficulty", 1))
        except (TypeError, ValueError):
            difficulty = 1
        turns.append(
            TeachingTurn(
                turn_id=bounded(item.get("id") or item.get("turn_id") or f"r{round_idx:03d}_{idx:03d}", 80),
                phase=clamp_phase(str(item.get("phase") or "word")),
                focus=bounded(item.get("focus") or "basic language", 120),
                difficulty=max(1, min(difficulty, 6)),
                teacher_text=teacher_text,
                imitation_prompt=prompt,
                target_output=target,
                notes=bounded(item.get("notes"), 240),
            )
        )
    return turns, diagnosis


def tokenize_teacher_texts(tokenizer: Any, turns: list[TeachingTurn]) -> list[np.ndarray]:
    texts: list[str] = []
    for turn in turns:
        text = clean_training_text(turn.teacher_text)
        if text:
            texts.append(text)
    if not texts:
        return []
    # Treat one caregiver round as one continuous listening stream.  With
    # packed training, separate tiny docs would insert EOS after every turn and
    # make EOS an artificial high-frequency target.
    ids = encode_text(tokenizer, " ".join(texts))
    return [ids] if ids.size > 1 else []


@torch.no_grad()
def evaluate_imitation_ce(
    model: TorchU017ErrorPopulationModel,
    tokenizer: Any,
    turns: list[TeachingTurn],
    batch_size: int,
) -> dict[str, float | int]:
    contexts: list[np.ndarray] = []
    targets: list[int] = []
    for turn in turns:
        prefix = encode_text(tokenizer, turn.imitation_prompt)
        target = encode_text(tokenizer, turn.target_output)
        if prefix.size == 0 or target.size == 0:
            continue
        history = [int(x) for x in prefix.tolist()]
        for token in target.tolist():
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
def generate_imitation_samples(
    model: TorchU017ErrorPopulationModel,
    tokenizer: Any,
    turns: list[TeachingTurn],
    count: int,
    max_new_tokens: int,
    extra_tokens: int,
    temperature: float,
) -> list[ImitationSample]:
    samples: list[ImitationSample] = []
    for turn in turns[: max(int(count), 0)]:
        prompt_ids = encode_text(tokenizer, turn.imitation_prompt)
        if prompt_ids.size == 0:
            continue
        target_ids = encode_text(tokenizer, turn.target_output)
        new_tokens = min(max(int(max_new_tokens), 1), max(int(target_ids.size) + max(int(extra_tokens), 0), 1))
        generated = model.generate(prompt_ids, new_tokens, sample=False, temperature=temperature)
        completion_ids = [int(x) for x in generated[int(prompt_ids.size) :]]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        samples.append(
            ImitationSample(
                turn_id=turn.turn_id,
                phase=turn.phase,
                focus=turn.focus,
                prompt=turn.imitation_prompt,
                target=turn.target_output,
                student=completion,
                student_repr=repr(completion),
                token_ids=completion_ids,
            )
        )
    return samples


def overlap_score(target: str, student: str) -> float:
    target_words = normalize_text(target).split()
    student_words = normalize_text(student).split()
    if not target_words:
        return 0.0
    if normalize_text(target) and normalize_text(target) in normalize_text(student):
        return 1.0
    hits = sum(1 for word in target_words if word in student_words)
    return hits / max(len(target_words), 1)


def mock_judge(samples: list[ImitationSample]) -> list[JudgeResult]:
    results: list[JudgeResult] = []
    for sample in samples:
        raw = sample.student
        raw_stripped = raw.strip()
        overlap = overlap_score(sample.target, raw)
        repeated = max_repeat_run(raw)
        if raw and not raw_stripped:
            problem = "whitespace_loop"
            score = 0.0
            correct = False
            decision = "simplify"
        elif repeated >= 3:
            problem = "repetition_loop"
            score = 1.0
            correct = False
            decision = "repeat_variant"
        elif overlap >= 0.99:
            problem = "none"
            score = 5.0
            correct = True
            decision = "advance"
        elif overlap >= 0.5:
            problem = "partial_imitation"
            score = 3.0
            correct = False
            decision = "repeat_variant"
        elif not raw_stripped:
            problem = "empty"
            score = 0.0
            correct = False
            decision = "simplify"
        else:
            problem = "wrong_words"
            score = 1.0
            correct = False
            decision = "repeat_same"
        results.append(
            JudgeResult(
                turn_id=sample.turn_id,
                correct=correct,
                score=score,
                problem=problem,
                decision=decision,
                next_focus=sample.focus,
                correction=sample.target,
            )
        )
    return results


def build_judge_messages(samples: list[ImitationSample]) -> list[dict[str, str]]:
    system = (
        "You judge a text-only infant language model trying to imitate or "
        "continue caregiver speech. Return JSON only, no markdown."
    )
    items = [
        {
            "id": sample.turn_id,
            "phase": sample.phase,
            "focus": sample.focus,
            "prompt": sample.prompt,
            "target": sample.target,
            "student": sample.student[:500],
            "student_repr": sample.student_repr,
            "token_ids": sample.token_ids[:64],
        }
        for sample in samples
    ]
    schema = {
        "results": [
            {
                "id": "same id",
                "correct": False,
                "score": 0,
                "problem": "whitespace_loop | repetition_loop | empty | wrong_words | partial_imitation | none",
                "decision": "repeat_same | repeat_variant | simplify | contrast | advance | switch_topic",
                "next_focus": "short phrase",
                "correction": "target-like correction",
            }
        ]
    }
    user = (
        "Judge whether the student reproduced the target meaning and form. "
        "Whitespace-only output is not correct. Repetition loops are not correct. "
        "Use score 0-5. Choose a decision for the next caregiver utterances.\n\n"
        f"Items:\n{json.dumps(items, ensure_ascii=True)}\n\n"
        f"Return this JSON shape:\n{json.dumps(schema, ensure_ascii=True)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def coerce_judge_results(payload: Any, samples: list[ImitationSample]) -> list[JudgeResult]:
    raw_items = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise ValueError("judge response must contain a results list")
    sample_ids = {sample.turn_id for sample in samples}
    results: list[JudgeResult] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        turn_id = bounded(item.get("id") or item.get("turn_id"), 80)
        if turn_id not in sample_ids:
            continue
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        results.append(
            JudgeResult(
                turn_id=turn_id,
                correct=bool(item.get("correct", False)),
                score=max(0.0, min(5.0, score)),
                problem=bounded(item.get("problem") or "unknown", 80),
                decision=bounded(item.get("decision") or "repeat_variant", 80),
                next_focus=bounded(item.get("next_focus"), 120),
                correction=bounded(item.get("correction"), 240),
            )
        )
    return results


def summarize_judge(results: list[JudgeResult]) -> dict[str, Any]:
    if not results:
        return {"judge_count": 0, "judge_correct": 0.0, "judge_score": float("nan"), "top_problem": "none"}
    counts: dict[str, int] = {}
    decisions: dict[str, int] = {}
    for item in results:
        counts[item.problem] = counts.get(item.problem, 0) + 1
        decisions[item.decision] = decisions.get(item.decision, 0) + 1
    top_problem = max(counts.items(), key=lambda kv: kv[1])[0]
    top_decision = max(decisions.items(), key=lambda kv: kv[1])[0]
    return {
        "judge_count": int(len(results)),
        "judge_correct": float(sum(1 for item in results if item.correct) / len(results)),
        "judge_score": float(sum(item.score for item in results) / len(results)),
        "top_problem": top_problem,
        "top_decision": top_decision,
    }


def save_model_checkpoint(
    path: Path,
    model: TorchU017ErrorPopulationModel,
    cfg: u012.U012Config,
    args: argparse.Namespace,
    state: TeacherState,
    round_idx: int,
    total_train_tokens: int,
    api_requests: int,
    probe_bank: list[TeachingTurn],
    metrics_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "round": int(round_idx),
            "total_train_tokens": int(total_train_tokens),
            "api_requests": int(api_requests),
            "teacher_state": asdict(state),
            "probe_bank": [asdict(turn) for turn in probe_bank[-256:]],
            "config": asdict(cfg),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "metrics_tail": metrics_rows[-20:],
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


def load_model_checkpoint(model: TorchU017ErrorPopulationModel, checkpoint: dict[str, Any], device: torch.device) -> None:
    model.input_codes = checkpoint["input_codes"].to(device)
    model.target_codes = checkpoint["target_codes"].to(device)
    model.position_codes = checkpoint["position_codes"].to(device)
    model.ff_weights = [tensor.to(device) for tensor in checkpoint["ff_weights"]]
    model.ff_biases = [tensor.to(device) for tensor in checkpoint["ff_biases"]]
    model.attn_q = [tensor.to(device) for tensor in checkpoint["attn_q"]]
    model.attn_k = [tensor.to(device) for tensor in checkpoint["attn_k"]]
    model.attn_v = [tensor.to(device) for tensor in checkpoint["attn_v"]]
    model.attn_o = [tensor.to(device) for tensor in checkpoint["attn_o"]]
    model.output_weights = checkpoint["output_weights"].to(device)
    model.output_counts = checkpoint["output_counts"].to(device)
    model.output_total = checkpoint["output_total"].to(device)
    model.output_bias = checkpoint["output_bias"].to(device)
    model.error_weights = [tensor.to(device) for tensor in checkpoint["error_weights"]]
    model.step = int(checkpoint.get("step", 0))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prune_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    keep_last = max(int(keep_last), 0)
    if keep_last <= 0 or not checkpoint_dir.exists():
        return
    checkpoints = sorted(checkpoint_dir.glob("round_*.pt"))
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink(missing_ok=True)


def update_teacher_state(
    state: TeacherState,
    judged: dict[str, Any],
    results: list[JudgeResult],
    advance_score: float,
    advance_streak: int,
    max_difficulty: int,
) -> None:
    score = float(judged.get("judge_score", float("nan")))
    correct = float(judged.get("judge_correct", 0.0))
    decision = str(judged.get("top_decision", "repeat_variant"))
    if not math.isnan(score) and score >= float(advance_score) and correct >= 0.50:
        state.mastery_streak += 1
    else:
        state.mastery_streak = 0

    if results:
        state.focus = results[0].next_focus or results[0].correction or results[0].next_focus or state.focus
        state.recent_errors = [
            f"{item.problem}:{item.correction or item.next_focus or item.turn_id}"
            for item in results
            if not item.correct
        ][-8:]

    if decision in ("simplify", "repeat_same", "repeat_variant", "contrast"):
        return
    if state.mastery_streak >= max(int(advance_streak), 1):
        state.mastery_streak = 0
        old_phase = state.phase
        state.phase = next_phase(state.phase)
        if state.phase == old_phase:
            state.difficulty = min(int(state.difficulty) + 1, max(int(max_difficulty), 1))


def build_parser() -> argparse.ArgumentParser:
    load_dotenv_file(SCRIPT_DIR / ".env")
    parser = argparse.ArgumentParser(description="U022 caregiver imitation loop for U017")
    parser.add_argument("--teacher-mode", choices=["mock", "deepseek"], default="deepseek")
    parser.add_argument("--judge-mode", choices=["auto", "none", "mock", "deepseek"], default="auto")
    parser.add_argument("--api-base-url", type=str, default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-model", type=str, default="deepseek-chat")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--api-temperature", type=float, default=0)
    parser.add_argument("--api-max-tokens", type=int, default=3000)
    parser.add_argument("--judge-max-tokens", type=int, default=1200)
    parser.add_argument("--max-api-requests", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=100000)
    parser.add_argument("--turns-per-round", type=int, default=8)
    parser.add_argument("--probe-bank-size", type=int, default=64)
    parser.add_argument("--probe-eval-count", type=int, default=16)
    parser.add_argument("--eval-every-rounds", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--sample-new-tokens", type=int, default=64)
    parser.add_argument("--sample-extra-tokens", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--start-phase", choices=PHASES, default="word")
    parser.add_argument("--start-difficulty", type=int, default=1)
    parser.add_argument("--max-difficulty", type=int, default=6)
    parser.add_argument("--advance-score", type=float, default=4.0)
    parser.add_argument("--advance-streak", type=int, default=2)
    parser.add_argument("--max-field-chars", type=int, default=1200)
    parser.add_argument("--save-api-payloads", action="store_true")
    parser.add_argument("--save-teacher-text", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "output" / "u022_deepseek_smoke")
    parser.add_argument("--tokenizer", type=Path, default=u012.DEFAULT_TOKENIZER)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--train-mode", choices=["packed", "chunk", "exact"], default="packed")
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=2048)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--attn-rank", type=int, default=512)
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
    parser.add_argument("--warmup-source", choices=["none", "wikitext", "url"], default="none")
    parser.add_argument("--warmup-docs", type=int, default=200)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--warmup-urls", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
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

    resume_blob: dict[str, Any] | None = None
    if args.resume_checkpoint is not None:
        resume_blob = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        ckpt_vocab = int(resume_blob["input_codes"].shape[0])
        if ckpt_vocab != int(len(tokenizer)):
            raise RuntimeError(f"checkpoint vocab={ckpt_vocab} but tokenizer len={len(tokenizer)}")
        cfg = u012.U012Config(**resume_blob["config"])
        for key, value in resume_blob["config"].items():
            if hasattr(args, key):
                setattr(args, key, value)
        print(f"[resume] loading {args.resume_checkpoint}", flush=True)
    else:
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
    if resume_blob is not None:
        load_model_checkpoint(model, resume_blob, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    if args.warmup_source != "none":
        warmup_urls = [part.strip() for part in str(args.warmup_urls).split(",") if part.strip()]
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

    state = (
        TeacherState(**resume_blob["teacher_state"])
        if resume_blob is not None and "teacher_state" in resume_blob
        else TeacherState(phase=args.start_phase, difficulty=max(int(args.start_difficulty), 1))
    )
    probe_bank: list[TeachingTurn] = (
        [TeachingTurn(**item) for item in resume_blob.get("probe_bank", [])]
        if resume_blob is not None
        else []
    )
    metrics_rows: list[dict[str, Any]] = read_csv_rows(args.out_dir / "metrics.csv") if resume_blob is not None else []
    recent_sample_dicts: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] = {}
    api_requests = int(resume_blob.get("api_requests", 0)) if resume_blob is not None else 0
    total_train_tokens = int(resume_blob.get("total_train_tokens", 0)) if resume_blob is not None else 0
    start_round = int(resume_blob.get("round", -1)) + 1 if resume_blob is not None else 0
    stop_reason = "completed_rounds"
    start_time = time.perf_counter()

    for round_idx in range(start_round, max(int(args.rounds), 0)):
        stop_after_round = False
        if args.teacher_mode == "mock":
            turns, diagnosis = mock_turns(round_idx, args.turns_per_round, state, args.seed)
        else:
            if client is None:
                raise RuntimeError("DeepSeek client was not initialized")
            if api_requests >= int(args.max_api_requests):
                stop_reason = "api_budget_before_teacher"
                print(f"[stop] max API request budget reached before round {round_idx}", flush=True)
                break
            messages = build_teacher_messages(round_idx, args.turns_per_round, state, last_metrics, recent_sample_dicts)
            response = client.chat(messages, max_tokens=args.api_max_tokens, temperature=args.api_temperature)
            api_requests += 1
            text = response_text(response)
            if args.save_api_payloads:
                append_jsonl(args.out_dir / "api_payloads.jsonl", {"round": round_idx, "kind": "teacher", "messages": messages, "response_text": text})
            turns, diagnosis = coerce_turns(extract_json_payload(text), round_idx, args.max_field_chars)
        if not turns:
            raise RuntimeError(f"teacher returned no usable turns in round {round_idx}")
        state.diagnosis = diagnosis or state.diagnosis

        for turn in turns:
            summary_row = {
                "round": round_idx,
                "turn_id": turn.turn_id,
                "phase": turn.phase,
                "focus": turn.focus,
                "difficulty": turn.difficulty,
                "teacher_chars": len(turn.teacher_text),
                "prompt_chars": len(turn.imitation_prompt),
                "target_chars": len(turn.target_output),
                "notes": turn.notes,
            }
            append_jsonl(args.out_dir / "turn_summaries.jsonl", summary_row)
            if args.save_teacher_text:
                append_jsonl(args.out_dir / "teacher_text.jsonl", {"round": round_idx, **asdict(turn)})

        docs = tokenize_teacher_texts(tokenizer, turns)
        train_t0 = time.perf_counter()
        train_summary, _ = u012.run_documents(
            model,
            docs,
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

        probe_bank.extend(turns)
        if len(probe_bank) > max(int(args.probe_bank_size), 1):
            probe_bank = probe_bank[-max(int(args.probe_bank_size), 1) :]

        do_eval = (round_idx + 1) % max(int(args.eval_every_rounds), 1) == 0
        eval_summary = {"loss": float("nan"), "accuracy": 0.0, "tokens": 0}
        samples: list[ImitationSample] = []
        judge_results: list[JudgeResult] = []
        if do_eval and probe_bank:
            probes = probe_bank[-max(int(args.probe_eval_count), 1) :]
            eval_summary = evaluate_imitation_ce(model, tokenizer, probes, batch_size=max(int(args.chunk_size), 1) * 16)
            samples = generate_imitation_samples(
                model,
                tokenizer,
                probes,
                args.sample_count,
                args.sample_new_tokens,
                args.sample_extra_tokens,
                args.sample_temperature,
            )
            if judge_mode == "mock":
                judge_results = mock_judge(samples)
            elif judge_mode == "deepseek" and samples:
                if client is None:
                    raise RuntimeError("DeepSeek client was not initialized")
                if api_requests >= int(args.max_api_requests):
                    stop_reason = "api_budget_before_judge"
                    stop_after_round = True
                    print(f"[stop] max API request budget reached before judge in round {round_idx}", flush=True)
                else:
                    messages = build_judge_messages(samples)
                    response = client.chat(messages, max_tokens=args.judge_max_tokens, temperature=0.0)
                    api_requests += 1
                    text = response_text(response)
                    if args.save_api_payloads:
                        append_jsonl(args.out_dir / "api_payloads.jsonl", {"round": round_idx, "kind": "judge", "messages": messages, "response_text": text})
                    judge_results = coerce_judge_results(extract_json_payload(text), samples)

            for sample in samples:
                append_jsonl(args.out_dir / "imitation_samples.jsonl", {"round": round_idx, **asdict(sample)})
            for result in judge_results:
                append_jsonl(args.out_dir / "judge_results.jsonl", {"round": round_idx, **asdict(result)})
            recent_sample_dicts = [
                {
                    "phase": sample.phase,
                    "prompt": sample.prompt,
                    "target": sample.target,
                    "student_repr": sample.student_repr,
                    "token_ids": sample.token_ids[:24],
                }
                for sample in samples
            ]

        judged = summarize_judge(judge_results)
        if do_eval:
            update_teacher_state(
                state,
                judged,
                judge_results,
                args.advance_score,
                args.advance_streak,
                args.max_difficulty,
            )
        last_metrics = {
            "train_loss": train_summary["loss"],
            "imitation_loss": eval_summary["loss"],
            "imitation_accuracy": eval_summary["accuracy"],
            "judge_correct": judged["judge_correct"],
            "judge_score": judged["judge_score"],
            "top_problem": judged["top_problem"],
            "top_decision": judged.get("top_decision", "none"),
        }
        row = {
            "round": round_idx,
            "phase": state.phase,
            "difficulty": state.difficulty,
            "focus": state.focus,
            "turns": len(turns),
            "train_loss": train_summary["loss"],
            "train_accuracy": train_summary["accuracy"],
            "train_tokens": train_summary["tokens"],
            "imitation_loss": eval_summary["loss"],
            "imitation_accuracy": eval_summary["accuracy"],
            "imitation_tokens": eval_summary["tokens"],
            "judge_count": judged["judge_count"],
            "judge_correct": judged["judge_correct"],
            "judge_score": judged["judge_score"],
            "top_problem": judged["top_problem"],
            "top_decision": judged.get("top_decision", "none"),
            "mastery_streak": state.mastery_streak,
            "train_seconds": train_seconds,
            "train_tokens_per_second": float(train_summary["tokens"] / max(train_seconds, 1e-9)),
            "api_requests": api_requests,
            "total_train_tokens": total_train_tokens,
            "diagnosis": state.diagnosis,
        }
        metrics_rows.append(row)
        write_csv(args.out_dir / "metrics.csv", metrics_rows)
        if int(args.checkpoint_every) > 0 and (round_idx + 1) % int(args.checkpoint_every) == 0:
            checkpoint_dir = args.out_dir / "checkpoints"
            checkpoint_path = checkpoint_dir / f"round_{round_idx:06d}.pt"
            save_model_checkpoint(
                checkpoint_path,
                model,
                cfg,
                args,
                state,
                round_idx,
                total_train_tokens,
                api_requests,
                probe_bank,
                metrics_rows,
            )
            prune_checkpoints(checkpoint_dir, args.keep_last_checkpoints)
            with (args.out_dir / "checkpoint_latest.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "round": int(round_idx),
                        "path": str(checkpoint_path),
                        "total_train_tokens": int(total_train_tokens),
                    },
                    f,
                    indent=2,
                    ensure_ascii=True,
                )
        print(
            f"round={round_idx} phase={state.phase} train_loss={float(train_summary['loss']):.4f} "
            f"imit_loss={float(eval_summary['loss']):.4f} judge={float(judged['judge_correct']):.3f} "
            f"decision={judged.get('top_decision', 'none')} tokens={int(train_summary['tokens'])} api={api_requests}",
            flush=True,
        )
        if stop_after_round:
            break

    elapsed = time.perf_counter() - start_time
    max_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    results = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "config": asdict(cfg),
        "summary": {
            "teacher_mode": args.teacher_mode,
            "judge_mode": judge_mode,
            "tokenizer_len": int(len(tokenizer)),
            "parameters": int(model.parameter_count()),
            "state_bytes": int(model.state_bytes()),
            "total_train_tokens": int(total_train_tokens),
            "elapsed_seconds": float(elapsed),
            "api_requests": int(api_requests),
            "stop_reason": stop_reason,
            "completed_rounds": int(len(metrics_rows)),
            "max_cuda_mem_bytes": int(max_mem),
            "final_phase": state.phase,
            "final_difficulty": int(state.difficulty),
        },
        "metrics": metrics_rows,
    }
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)
    print("Summary:")
    print(
        f"  teacher={args.teacher_mode} judge={judge_mode} phase={state.phase} "
        f"params={model.parameter_count():,} state={model.state_bytes() / (1024 ** 2):.1f} MiB"
    )
    print(
        f"  tokens={total_train_tokens} elapsed={elapsed:.2f}s api_requests={api_requests} "
        f"max_mem={max_mem / (1024 ** 3):.2f} GiB"
    )
    print(f"  metrics: {args.out_dir / 'metrics.csv'}")
    print(f"  samples: {args.out_dir / 'imitation_samples.jsonl'}")


if __name__ == "__main__":
    main()
