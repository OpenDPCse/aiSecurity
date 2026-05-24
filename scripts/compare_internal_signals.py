#!/usr/bin/env python3
"""Compare internal-signal features across prompt groups."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "analysis" / "internal_signals.csv"
DEFAULT_SUMMARY = ROOT / "analysis" / "internal_signal_group_summary.csv"
DEFAULT_PAIRS = ROOT / "analysis" / "internal_signal_pairwise_delta.csv"


METRIC_COLUMNS = [
    "hidden_last_norm",
    "hidden_mean_norm",
    "hidden_prompt_question_cos",
    "hidden_last_question_cos",
    "attn_last_entropy",
    "attn_last_to_prompt",
    "attn_last_to_question",
]


def cohen_d(left: pd.Series, right: pd.Series) -> float:
    left = pd.to_numeric(left, errors="coerce").dropna()
    right = pd.to_numeric(right, errors="coerce").dropna()
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = (((len(left) - 1) * left.var()) + ((len(right) - 1) * right.var())) / (
        len(left) + len(right) - 2
    )
    if pooled <= 0:
        return 0.0
    return float((left.mean() - right.mean()) / (pooled ** 0.5))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [col for col in METRIC_COLUMNS if col in df.columns]
    grouped = df.groupby(["prompt_type", "layer"], dropna=False)[metrics]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join([str(part) for part in col if part])
        for col in summary.columns.to_flat_index()
    ]
    return summary


def pairwise(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [col for col in METRIC_COLUMNS if col in df.columns]
    rows = []
    for layer, layer_df in df.groupby("layer"):
        groups = sorted(layer_df["prompt_type"].dropna().unique())
        for left_name, right_name in combinations(groups, 2):
            left = layer_df[layer_df["prompt_type"] == left_name]
            right = layer_df[layer_df["prompt_type"] == right_name]
            for metric in metrics:
                left_mean = pd.to_numeric(left[metric], errors="coerce").mean()
                right_mean = pd.to_numeric(right[metric], errors="coerce").mean()
                rows.append(
                    {
                        "layer": layer,
                        "group_a": left_name,
                        "group_b": right_name,
                        "metric": metric,
                        "group_a_mean": left_mean,
                        "group_b_mean": right_mean,
                        "mean_delta_a_minus_b": left_mean - right_mean,
                        "cohen_d_a_minus_b": cohen_d(left[metric], right[metric]),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create group-level and pairwise summaries for model internals."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pairwise-output", type=Path, default=DEFAULT_PAIRS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    required = {"prompt_type", "layer"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    summary_df = summarize(df)
    pairwise_df = pairwise(df)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_output, index=False, encoding="utf-8")
    pairwise_df.to_csv(args.pairwise_output, index=False, encoding="utf-8")

    print(f"Wrote group summary to {args.summary_output}")
    print(f"Wrote pairwise deltas to {args.pairwise_output}")
    if not pairwise_df.empty:
        top = pairwise_df.reindex(
            pairwise_df["cohen_d_a_minus_b"].abs().sort_values(ascending=False).index
        ).head(10)
        print("\nLargest absolute effect sizes:")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
