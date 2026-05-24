#!/usr/bin/env python3
"""Generate model answers for a CSV of jailbreak prompts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import transformers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "new" / "my_data" / "concat_prompt.csv"
DEFAULT_OUTPUT = ROOT / "new" / "newMethodOutput.csv"


def load_pipeline(model_id: str, dtype: str, device_map: str) -> Any:
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    return transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": dtype_map[dtype]},
        device_map=device_map,
    )


def render_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate_one(
    pipe: Any,
    prompt: str,
    system_prompt: str | None,
    max_new_tokens: int,
    temperature: float,
) -> str:
    output = pipe(
        render_messages(prompt, system_prompt),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_full_text=False,
    )
    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        return generated[-1].get("content", "")
    return str(generated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a local HuggingFace chat model for every prompt in a CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct",
        help="Local model path or HuggingFace model id.",
    )
    parser.add_argument(
        "--prompt-column",
        default="concat_prompt_question",
        help="Column containing the final prompt sent to the model.",
    )
    parser.add_argument(
        "--answer-column",
        default="model_answer",
        help="Column used to store model responses.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt for the target model. Use an empty string to omit it.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for a smoke test before a full run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing output rows and skip rows that already have answers.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Write progress every N generated rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.output.exists():
        df = pd.read_csv(args.output)
    else:
        df = pd.read_csv(args.input)

    if args.limit is not None:
        df = df.head(args.limit).copy()
    if args.prompt_column not in df.columns:
        raise KeyError(f"Missing prompt column: {args.prompt_column}")
    if args.answer_column not in df.columns:
        df[args.answer_column] = ""

    pipe = load_pipeline(args.model, args.dtype, args.device_map)
    system_prompt = args.system_prompt or None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pending = df[args.answer_column].isna() | (df[args.answer_column].astype(str) == "")
    total = int(pending.sum())
    done = 0
    for index, row in df[pending].iterrows():
        answer = generate_one(
            pipe=pipe,
            prompt=str(row[args.prompt_column]),
            system_prompt=system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        df.at[index, args.answer_column] = answer
        done += 1
        if done % args.save_every == 0:
            df.to_csv(args.output, index=False, encoding="utf-8")
            print(f"Saved progress: {done}/{total}")

    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows with answers to {args.output}")


if __name__ == "__main__":
    main()
