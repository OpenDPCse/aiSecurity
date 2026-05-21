#!/usr/bin/env python3
"""Build the prompt/question Cartesian-product dataset used in the experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "new" / "my_data"


def read_prompt_file(path: Path, fallback_role: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "prompt" not in df.columns:
        df = df.rename(columns={df.columns[0]: "prompt"})
    if "role" not in df.columns:
        df.insert(0, "role", fallback_role)
    return df[["role", "prompt"]]


def read_question_file(path: Path, fallback_category: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "question" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "question"})
    if "question_ctgr" not in df.columns:
        df.insert(0, "question_ctgr", fallback_category)
    return df[["question_ctgr", "question"]]


def build_dataset(data_dir: Path) -> pd.DataFrame:
    prompt_frames = [
        read_prompt_file(data_dir / "hacker_prompt.csv", "hacker"),
        read_prompt_file(data_dir / "non_hacker_prompt.csv", "non_hacker"),
    ]
    question_frames = [
        read_question_file(data_dir / "hacking_question.csv", "hacking"),
        read_question_file(data_dir / "non_hacking_question.csv", "non_hacking"),
    ]

    prompts = pd.concat(prompt_frames, ignore_index=True)
    questions = pd.concat(question_frames, ignore_index=True)
    dataset = prompts.merge(questions, how="cross")
    dataset["concat_prompt_question"] = (
        dataset["prompt"].astype(str).str.rstrip()
        + "\n\nUser question: "
        + dataset["question"].astype(str).str.strip()
    )
    return dataset[
        ["role", "prompt", "question_ctgr", "question", "concat_prompt_question"]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the jailbreak evaluation dataset from prompt/question CSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing hacker_prompt.csv, non_hacker_prompt.csv, hacking_question.csv, and non_hacking_question.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_DIR / "concat_prompt.csv",
        help="Path for the combined CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()
