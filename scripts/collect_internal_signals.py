#!/usr/bin/env python3
"""Collect hidden-state and attention features for grouped jailbreak prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "new" / "last_concat.csv"
DEFAULT_OUTPUT = ROOT / "analysis" / "internal_signals.csv"


def parse_group_rules(raw: str | None) -> dict[str, dict[str, str]]:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def infer_prompt_type(row: pd.Series, rules: dict[str, dict[str, str]]) -> str:
    for column, patterns in rules.items():
        value = str(row.get(column, "")).lower()
        if not isinstance(patterns, dict):
            continue
        for pattern, group in patterns.items():
            if pattern.lower() in value:
                return group

    if "prompt_type" in row and pd.notna(row["prompt_type"]):
        return str(row["prompt_type"])

    role = str(row.get("role", "")).lower()
    prompt = str(row.get("prompt", "")).lower()
    if role in {"non_hacker", "non-hacker"}:
        return "non_hacker_roleplay"
    if role == "hacker" or "hacker" in role:
        return "hacker_roleplay"
    if role in {"other", "baseline", "dan", "ucar"}:
        return "other"
    if "hacker" in prompt or "malware" in prompt or "cyber" in prompt:
        return "hacker_roleplay"
    if "roleplay" in prompt or "act as" in prompt or "stay in character" in prompt:
        return "other"
    return role or "unknown"


def entropy(prob: torch.Tensor) -> torch.Tensor:
    return -(prob.clamp_min(1e-12) * prob.clamp_min(1e-12).log()).sum(dim=-1)


def token_span(tokenizer: Any, text: str, prefix: str) -> tuple[int, int]:
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_ids = tokenizer(text, add_special_tokens=False).input_ids
    return len(prefix_ids), len(full_ids)


def load_model(model_id: str, dtype: str, device_map: str, attn_implementation: str):
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = {
        "torch_dtype": dtype_map[dtype],
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


def sample_frame(df: pd.DataFrame, max_per_group: int | None, seed: int) -> pd.DataFrame:
    if max_per_group is None:
        return df
    frames = []
    for _, group_df in df.groupby("prompt_type", dropna=False):
        frames.append(
            group_df.sample(
                n=min(max_per_group, len(group_df)),
                random_state=seed,
            )
        )
    return pd.concat(frames, ignore_index=True)


def collect_for_row(
    row: pd.Series,
    row_id: int,
    tokenizer: Any,
    model: Any,
    prompt_column: str,
    question_column: str,
    max_length: int,
) -> list[dict[str, Any]]:
    text = str(row[prompt_column])
    question = str(row.get(question_column, ""))
    prompt_prefix = text.rsplit(question, 1)[0] if question and question in text else ""
    prompt_start, prompt_end = token_span(tokenizer, text, prompt_prefix) if prompt_prefix else (0, 0)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    device = model.get_input_embeddings().weight.device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    seq_len = int(encoded["input_ids"].shape[-1])
    prompt_end = min(prompt_end, seq_len)
    question_start = min(prompt_end, seq_len - 1)

    with torch.no_grad():
        output = model(
            **encoded,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = output.hidden_states[1:]
    attentions = output.attentions or [None] * len(hidden_states)
    rows = []
    for layer_idx, hidden in enumerate(hidden_states):
        h = hidden[0].float()
        last = h[-1]
        mean = h.mean(dim=0)
        prompt_mean = h[:prompt_end].mean(dim=0) if prompt_end > 0 else mean
        question_mean = h[question_start:].mean(dim=0)
        record = {
            "row_id": row_id,
            "prompt_type": row["prompt_type"],
            "role": row.get("role", ""),
            "question_ctgr": row.get("question_ctgr", ""),
            "layer": layer_idx,
            "seq_len": seq_len,
            "hidden_last_norm": float(last.norm().cpu()),
            "hidden_mean_norm": float(mean.norm().cpu()),
            "hidden_prompt_question_cos": float(
                F.cosine_similarity(prompt_mean, question_mean, dim=0).cpu()
            ),
            "hidden_last_question_cos": float(
                F.cosine_similarity(last, question_mean, dim=0).cpu()
            ),
        }
        attention = attentions[layer_idx]
        if attention is not None:
            attn = attention[0].float()
            last_attn = attn[:, -1, :]
            record["attn_last_entropy"] = float(entropy(last_attn).mean().cpu())
            record["attn_last_to_prompt"] = (
                float(last_attn[:, :prompt_end].sum(dim=-1).mean().cpu())
                if prompt_end > 0
                else 0.0
            )
            record["attn_last_to_question"] = float(
                last_attn[:, question_start:].sum(dim=-1).mean().cpu()
            )
        rows.append(record)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-layer hidden-state and attention summaries for prompt groups."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        required=True,
        help="Local model path or HuggingFace model id for the safety-domain model.",
    )
    parser.add_argument("--prompt-column", default="concat_prompt_question")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-per-group", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Use eager attention when attention weights are needed.",
    )
    parser.add_argument(
        "--group-rules",
        default=None,
        help='Optional JSON mapping, e.g. {"role":{"hacker":"hacker_roleplay","other":"other"}}.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = parse_group_rules(args.group_rules)
    df = pd.read_csv(args.input)
    if args.prompt_column not in df.columns:
        raise KeyError(f"Missing prompt column: {args.prompt_column}")
    df["prompt_type"] = df.apply(lambda row: infer_prompt_type(row, rules), axis=1)
    df = sample_frame(df, args.max_per_group, args.seed)

    tokenizer, model = load_model(
        args.model,
        args.dtype,
        args.device_map,
        args.attn_implementation,
    )

    records = []
    for row_id, row in df.reset_index(drop=True).iterrows():
        records.extend(
            collect_for_row(
                row=row,
                row_id=row_id,
                tokenizer=tokenizer,
                model=model,
                prompt_column=args.prompt_column,
                question_column=args.question_column,
                max_length=args.max_length,
            )
        )
        if (row_id + 1) % 10 == 0:
            print(f"Processed {row_id + 1}/{len(df)} prompts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote internal-signal features to {args.output}")


if __name__ == "__main__":
    main()
