# Homework 4 - D3-Agentic Prompting

This folder contains the Homework 4 solution for the CLARITY/QEvasion response-clarity task.

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

After the first full run, the final-decision step was hardened to avoid invalid labels from truncated JSON rationales. The parser now recovers explicit `"label": "..."` fields from incomplete JSON-like generations, and the final decision prompts request label-only JSON. The existing `runs_hw4_full/` artifacts have been reparsed with this hardened parser, so no model rerun is required for the corrected metrics. A fresh run is only needed to measure whether the shorter label-only prompt changes generations:

```bash
python -m artificial_intelligence.homeworks.hw20260609.sdi2200160 --preset full --eval-per-label 10 --output-dir runs_hw4_hardened
```

The original full run writes outputs under `runs_hw4_full/`; the optional hardened rerun command above writes the same artifact set under `runs_hw4_hardened/`, including:

- `experiment_summary.csv`
- `previous_assignment_baselines.csv`
- `baseline_comparison.csv`
- per-run generations, agent outputs, confusion matrices, reports, and errors
- `submissions/submission_best_d3_agentic_system.csv`

The completed local full run selected the same-model `single-agent` comparator by validation macro F1:

| System | Accuracy | Macro F1 | Invalid rate |
|---|---:|---:|---:|
| `qwen-0.8b_single-agent` | 0.3667 | 0.3062 | 0.0000 |
| `qwen-0.8b_d3-agentic` | 0.3000 | 0.2206 | 0.0000 |

The required D3 system is still reported and analyzed. Its interesting failure mode is the reverse of the earlier Qwen prompting runs: it detects `Ambivalent`, but too broadly. It recovers 7/10 true `Ambivalent` examples while predicting `Ambivalent` 24/30 times and missing every `Clear Non-Reply`, so it is not selected for the final CSV.

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

The PDF report uses:

- `runs_hw4_full/experiment_summary.csv`
- `runs_hw4_full/baseline_comparison.csv`
- `runs_hw4_full/errors_all.csv`
- `runs_hw4_full/runs/qwen-0.8b_d3-agentic.agent_outputs.csv`

Selection is by validation macro F1, with accuracy as tie-breaker.
