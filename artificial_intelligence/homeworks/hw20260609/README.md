# Homework 4 - D3-Agentic Prompting

This folder contains the Homework 4 solution scaffold for the CLARITY/QEvasion response-clarity task.

## Core Files

- `sdi2200160.py`: command-line experiment runner for D3-Agentic Prompting.
- `sdi2200160 homework-4-d3-agentic.ipynb`: self-contained Kaggle notebook.
- `specification.pdf`: assignment statement.

The shared D3 implementation lives in:

- `../agentic.py`

## Required Method

The required system is `d3-agentic`, implemented with `Qwen/Qwen3.5-0.8B` and four agents:

1. Question Intent Agent
2. Answer Content Agent
3. Gap and Evasion Agent
4. Decision Agent

The script/notebook also runs `single-agent`, a same-model direct prompting comparator. This isolates the effect of D3 decomposition without changing model size.

## Local Smoke Test

The smoke preset uses synthetic data and a static generator. It checks artifact creation without loading Hugging Face models:

```bash
python -m artificial_intelligence.homeworks.hw20260609.sdi2200160 --preset smoke
```

## Full Run

The intended full diagnostic run is:

```bash
python -m artificial_intelligence.homeworks.hw20260609.sdi2200160 --preset full --eval-per-label 10
```

This writes outputs under `runs_hw4_full/`, including:

- `experiment_summary.csv`
- `previous_assignment_baselines.csv`
- `baseline_comparison.csv`
- per-run generations, agent outputs, confusion matrices, reports, and errors
- `submissions/submission_best_d3_agentic_system.csv`

## Kaggle

Submit and run:

```text
sdi2200160 homework-4-d3-agentic.ipynb
```

The notebook writes the required submission file in `/kaggle/working`:

```text
submission_best_d3_agentic_system.csv
```

The first Kaggle cell upgrades the Hugging Face runtime stack and installs a current Transformers source build when needed for the `qwen3_5` architecture. Keep Internet enabled for the first run.

## Report

The final PDF report should be filled after the real run completes, using:

- `runs_hw4_full/experiment_summary.csv`
- `runs_hw4_full/baseline_comparison.csv`
- `runs_hw4_full/errors_all.csv`
- `runs_hw4_full/runs/qwen-0.8b_d3-agentic.agent_outputs.csv`

Selection is by validation macro F1, with accuracy as tie-breaker.
