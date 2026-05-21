#!/usr/bin/env python3
"""Summarize jailbreak harmfulness scores by role and question category."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "new" / "last_concat.csv"


def summarize(path: Path, score_column: str) -> dict[str, float]:
    df = pd.read_csv(path)
    if score_column not in df.columns and "New_Column" in df.columns:
        score_column = "New_Column"
    required = {"role", "question_ctgr", score_column}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    return {
        "overall_asr": df[score_column].mean(),
        "hacker_asr": df.loc[df["role"] == "hacker", score_column].mean(),
        "non_hacker_asr": df.loc[df["role"] == "non_hacker", score_column].mean(),
        "hacking_question_asr": df.loc[
            df["question_ctgr"] == "hacking", score_column
        ].mean(),
        "non_hacking_question_asr": df.loc[
            df["question_ctgr"] == "non_hacking", score_column
        ].mean(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print ASR-style score summaries.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--score-column", default="score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = summarize(args.input, args.score_column)
    print("ASR summary")
    print("=" * 32)
    for name, value in results.items():
        rendered = "nan" if pd.isna(value) else f"{value:.4f}"
        print(f"{name:28s} {rendered}")


if __name__ == "__main__":
    main()
