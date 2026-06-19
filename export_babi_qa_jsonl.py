#!/usr/bin/env python3
"""
Materialize local bAbI QA loader configs into project-local JSONL files.

The repository currently stores the Hugging Face bAbI loader metadata under
data/babi_qa/. Recent `datasets` versions no longer execute local dataset
loading scripts, so this exporter reads the loader's URL/path table, downloads
the original bAbI tarball, parses the official txt files, and writes compact
JSONL under data/babi_qa_processed/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

from datasets import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "data" / "babi_qa"
DEFAULT_ARCHIVE_DIR = SCRIPT_DIR / "data" / "babi_qa_raw"
DEFAULT_OUT_DIR = SCRIPT_DIR / "data" / "babi_qa_processed"
DEFAULT_HF_FALLBACK = "Muennighoff/babi"


def load_loader_metadata(dataset_dir: Path) -> tuple[str, dict[str, dict[str, dict[str, str]]]]:
    script_path = dataset_dir / "babi_qa.py"
    if not script_path.exists():
        raise FileNotFoundError(f"missing bAbI loader script: {script_path}")
    spec = importlib.util.spec_from_file_location("local_babi_qa_loader", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import loader metadata from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.ZIP_URL), module.paths


def resolve_config(config: str, paths: dict[str, dict[str, dict[str, str]]]) -> tuple[str, str]:
    for dataset_type, tasks in paths.items():
        for task_no in tasks:
            if config == f"{dataset_type}-{task_no}":
                return dataset_type, task_no
    available = sorted(f"{dataset_type}-{task_no}" for dataset_type, tasks in paths.items() for task_no in tasks)
    raise ValueError(f"unknown config {config!r}; available examples: {available[:10]}")


def archive_path_for(archive_dir: Path, url: str) -> Path:
    name = url.rstrip("/").rsplit("/", 1)[-1]
    return archive_dir / name


def ensure_archive(archive_dir: Path, url: str, explicit_archive: Path | None = None) -> Path:
    if explicit_archive is not None:
        if not explicit_archive.exists():
            raise FileNotFoundError(f"explicit archive does not exist: {explicit_archive}")
        return explicit_archive
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_path_for(archive_dir, url)
    if path.exists() and path.stat().st_size > 0:
        return path
    candidates = [url]
    if url.startswith("http://"):
        candidates.append("https://" + url[len("http://") :])
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            print(f"downloading {candidate} -> {path}")
            urllib.request.urlretrieve(candidate, path)
            if path.stat().st_size <= 0:
                raise RuntimeError(f"downloaded empty archive from {candidate}")
            return path
        except Exception as exc:  # noqa: BLE001 - preserve all download failures for reporting.
            last_error = exc
            if path.exists() and path.stat().st_size == 0:
                path.unlink()
    raise RuntimeError(f"could not download bAbI archive from {candidates}: {last_error}")


def parse_line(line: str) -> dict[str, Any]:
    stripped = line.strip()
    if not stripped:
        raise ValueError("empty line")
    line_no, rest = stripped.split(" ", 1)
    parts = rest.split("\t")
    if len(parts) > 1:
        return {
            "id": line_no,
            "type": "question",
            "text": parts[0].strip(),
            "answer": parts[1].strip(),
            "supporting_ids": parts[-1].strip().split(),
        }
    return {
        "id": line_no,
        "type": "context",
        "text": parts[0].strip(),
        "answer": "",
        "supporting_ids": [],
    }


def read_stories_from_archive(archive_path: Path, member_path: str) -> list[list[dict[str, Any]]]:
    stories: list[list[dict[str, Any]]] = []
    story: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        member = tar.getmember(member_path)
        f = tar.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"could not read {member_path} from {archive_path}")
        for raw_line in f:
            line = raw_line.decode("utf-8")
            if not line.strip():
                if story:
                    stories.append(story)
                    story = []
                continue
            item = parse_line(line)
            if item["id"] == "1" and story:
                stories.append(story)
                story = []
            story.append(item)
    if story:
        stories.append(story)
    return stories


def normalize_story_item(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("type"))
    return {
        "id": str(item.get("id", "")),
        "type": kind,
        "text": str(item.get("text", "")),
        "answer": str(item.get("answer", "")),
        "supporting_ids": [str(x) for x in item.get("supporting_ids", [])],
    }


def question_records(config: str, split: str, example_idx: int, story: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    context: list[dict[str, str]] = []
    question_index = 0
    for raw_item in story:
        item = normalize_story_item(raw_item)
        if item["type"] == "context":
            context.append({"id": item["id"], "text": item["text"]})
            continue
        if item["type"] != "question":
            continue
        question_index += 1
        supporting = set(item["supporting_ids"])
        supporting_context = [row for row in context if row["id"] in supporting]
        records.append(
            {
                "config": config,
                "split": split,
                "example_id": example_idx,
                "question_index": question_index,
                "context": context.copy(),
                "question": item["text"],
                "answer": item["answer"],
                "supporting_ids": item["supporting_ids"],
                "supporting_context": supporting_context,
            }
        )
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def export_config(
    archive_path: Path,
    out_dir: Path,
    config: str,
    paths: dict[str, dict[str, dict[str, str]]],
) -> dict[str, Any]:
    dataset_type, task_no = resolve_config(config, paths)
    split_paths = paths[dataset_type][task_no]
    summary: dict[str, Any] = {
        "config": config,
        "splits": {},
        "answer_vocab": [],
        "output_files": {},
    }
    answers: set[str] = set()
    for split, member_path in split_paths.items():
        stories = read_stories_from_archive(archive_path, member_path)
        rows: list[dict[str, Any]] = []
        for example_idx, story in enumerate(stories):
            rows.extend(question_records(config, split, example_idx, story))
        for row in rows:
            answers.add(row["answer"])
        out_path = out_dir / config / f"{split}.jsonl"
        write_jsonl(out_path, rows)
        summary["splits"][split] = {
            "stories": len(stories),
            "questions": len(rows),
        }
        summary["output_files"][split] = str(out_path.relative_to(SCRIPT_DIR))
    summary["answer_vocab"] = sorted(answers)
    summary["answer_count"] = len(answers)
    return summary


def task_number_from_config(config: str) -> int:
    task_name = config.rsplit("-", 1)[-1]
    if not task_name.startswith("qa"):
        raise ValueError(f"could not infer task number from config {config!r}")
    return int(task_name[2:])


def export_config_from_hf_fallback(out_dir: Path, hf_dataset: str, config: str) -> dict[str, Any]:
    task_no = task_number_from_config(config)
    dataset = load_dataset(hf_dataset)
    summary: dict[str, Any] = {
        "config": config,
        "source": hf_dataset,
        "splits": {},
        "answer_vocab": [],
        "output_files": {},
    }
    answers: set[str] = set()
    for split, data in dataset.items():
        rows: list[dict[str, Any]] = []
        split_rows = data.filter(lambda row, task_no=task_no: int(row["task"]) == task_no)
        for example_idx, example in enumerate(split_rows):
            context = [
                {"id": str(idx + 1), "text": text}
                for idx, text in enumerate(str(example["passage"]).splitlines())
                if text.strip()
            ]
            answer = str(example["answer"])
            rows.append(
                {
                    "config": config,
                    "split": split,
                    "example_id": example_idx,
                    "question_index": 1,
                    "context": context,
                    "question": str(example["question"]),
                    "answer": answer,
                    "supporting_ids": [],
                    "supporting_context": [],
                    "source_task": int(example["task"]),
                }
            )
            answers.add(answer)
        out_path = out_dir / config / f"{split}.jsonl"
        write_jsonl(out_path, rows)
        summary["splits"][split] = {
            "stories": len(rows),
            "questions": len(rows),
        }
        summary["output_files"][split] = str(out_path.relative_to(SCRIPT_DIR))
    summary["answer_vocab"] = sorted(answers)
    summary["answer_count"] = len(answers)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hf-fallback", default=DEFAULT_HF_FALLBACK)
    parser.add_argument(
        "--source",
        choices=["auto", "official-archive", "hf-fallback"],
        default="auto",
        help="Data source. auto tries official archive first, then HF fallback.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["en-qa1", "en-qa2", "en-qa3", "en-qa15", "en-qa16"],
        help="bAbI config names to export, for example en-qa1 en-qa2.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    url, paths = load_loader_metadata(args.dataset_dir)
    archive_path: Path | None = None
    archive_error: Exception | None = None
    if args.source in {"auto", "official-archive"}:
        try:
            archive_path = ensure_archive(args.archive_dir, url, args.archive)
        except Exception as exc:  # noqa: BLE001 - report and use fallback if requested.
            archive_error = exc
            if args.source == "official-archive":
                raise
            print(f"official archive unavailable; falling back to {args.hf_fallback}: {exc}")
    summaries = []
    for config in args.configs:
        if archive_path is not None and args.source != "hf-fallback":
            summary = export_config(archive_path, args.out_dir, config, paths)
            summary["source"] = str(archive_path)
        else:
            summary = export_config_from_hf_fallback(args.out_dir, args.hf_fallback, config)
            if archive_error is not None:
                summary["official_archive_error"] = str(archive_error)
        summaries.append(summary)
        split_summary = ", ".join(
            f"{split}: {values['questions']} questions" for split, values in summary["splits"].items()
        )
        print(f"{config}: {split_summary}; answers={summary['answer_count']}")
    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
