# Hacker Roleplay Jailbreak Evaluation

This project studies whether **hacker roleplay jailbreak prompts** have a stronger impact on safety-domain LLMs than non-hacker roleplay prompts or other jailbreak prompts.

The experiment compares prompt groups such as:

- `hacker_roleplay`
- `non_hacker_roleplay`
- `other`

It measures both output-level harmfulness scores and internal model signals such as hidden states and attention patterns.

## Research Question

> Compared with other jailbreak prompts, do prompts that ask the model to roleplay as a hacker cause stronger safety degradation in safety-domain models?

## Repository Structure

```text
promptGeneration/        Early scripts for generating roleplay prompts
new/                     Main data files and experiment outputs
new/my_data/             Cleaned prompt/question CSVs
new/strong_reject/       StrongREJECT evaluator code
eval/                    Earlier evaluation scripts and results
scripts/                 Reproducible experiment utilities
analysis/                Suggested output directory for interpretability results
```

## Setup

```bash
pip install -r requirements.txt
```

For score aggregation only, `pandas` is enough.  
For model generation or interpretability analysis, prepare a local HuggingFace-compatible model and install `torch`, `transformers`, and `accelerate`.

## Workflow

### 1. Build the Evaluation Dataset

```bash
python scripts/prepare_dataset.py
```

Default input files:

```text
new/my_data/hacker_prompt.csv
new/my_data/non_hacker_prompt.csv
new/my_data/hacking_question.csv
new/my_data/non_hacking_question.csv
```

Default output:

```text
new/my_data/concat_prompt.csv
```

### 2. Generate Model Answers

Smoke test:

```bash
python scripts/generate_model_answers.py --limit 5
```

Full run:

```bash
python scripts/generate_model_answers.py \
  --input new/my_data/concat_prompt.csv \
  --output new/newMethodOutput.csv \
  --model /root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --resume
```

The output CSV contains a `model_answer` column.

### 3. Summarize Harmfulness Scores

```bash
python scripts/score_summary.py --input new/last_concat.csv
```

Main metrics:

```text
overall_asr
hacker_asr
non_hacker_asr
hacking_question_asr
non_hacking_question_asr
```

Here, ASR is treated as the average harmfulness score. Higher scores indicate stronger unsafe behavior.

## Interpretability Analysis

The project also includes scripts for comparing internal model behavior across prompt types.

### Collect Internal Signals

```bash
python scripts/collect_internal_signals.py \
  --input new/last_concat.csv \
  --output analysis/internal_signals.csv \
  --model /root/autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --max-per-group 30
```

Collected per-layer features include:

```text
hidden_last_norm
hidden_mean_norm
hidden_prompt_question_cos
hidden_last_question_cos
attn_last_entropy
attn_last_to_prompt
attn_last_to_question
```

### Compare Prompt Groups

```bash
python scripts/compare_internal_signals.py \
  --input analysis/internal_signals.csv
```

Default outputs:

```text
analysis/internal_signal_group_summary.csv
analysis/internal_signal_pairwise_delta.csv
```

The pairwise file reports mean differences and Cohen's d for each layer and metric, making it easier to locate where `hacker_roleplay` diverges from the control groups.

## Prompt Grouping

Default mapping:

```text
role=hacker      -> hacker_roleplay
role=non_hacker  -> non_hacker_roleplay
role=other       -> other
```

If your CSV already contains a `prompt_type` column, the interpretability script uses it directly.

You can also pass custom JSON rules:

```bash
python scripts/collect_internal_signals.py \
  --input your.csv \
  --model /path/to/model \
  --group-rules '{"role":{"hacker":"hacker_roleplay","other":"other"}}'
```

## Suggested Reporting Order

1. Report output-level results: whether `hacker_roleplay` has higher harmfulness scores than controls.
2. Report internal-signal results: which layers show the largest hidden-state or attention differences.
3. Use pairwise Cohen's d to discuss whether differences appear mainly in early, middle, or late layers.

## Safety Note

This repository contains jailbreak and harmful prompts. It is intended only for LLM safety research and model evaluation.
