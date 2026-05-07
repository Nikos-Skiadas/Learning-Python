# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This repo bundles coursework from three separate university courses, each with its own conventions:

- [classes/](classes/) — OOP exercises. Polygon hierarchy (`Scalene` → `Regular` → `Triangle` → `Right`/`Isosceles`/`Equilateral`, plus `Quadrilateral`). Uses `__init_subclass__` to propagate the `order` class attribute. Scripts are standalone (flat imports like `from polygon import ...`), so run them from inside [classes/](classes/).
- [artificial_intelligence/homeworks/](artificial_intelligence/homeworks/) — ML homeworks organized around a **Protocol-based pipeline framework**. Has its own [CLAUDE.md](artificial_intelligence/homeworks/CLAUDE.md) with pipeline details — read it before editing anything under this tree.
- [data-mining/](data-mining/) — Music analysis project. Raw datasets live in `data-mining/project/data/` and `data-mining/Εργασίες/…` (both gitignored via `**/data`). The `project/src/` sources were cleaned up in commit `aa5dc97` and have not been re-added yet.

There is no top-level package — `artificial_intelligence/homeworks` and `classes` are unrelated and should not import from each other.

## Environment

- **Python 3.14.3**, single shared venv at repo-root [.venv/](.venv/). Activate it before running anything: `source .venv/bin/activate`.
- Dependencies pinned in [requirements.txt](requirements.txt) (torch, transformers, sentence-transformers, sklearn, nltk, pandas, datasets, matplotlib/seaborn, dotenv, rich). Install with `pip install -r requirements.txt`.
- [.env](.env) holds `HF_TOKEN` for HuggingFace dataset/model downloads. Loaded via `dotenv.load_dotenv(override=True)` in [artificial_intelligence/homeworks/data.py](artificial_intelligence/homeworks/data.py).
- `pytest` is the configured test runner with `-s` passed by default ([.vscode/settings.json](.vscode/settings.json)). `unittest` is disabled.

## Running Code

```bash
# AI homework (run from repo root — uses relative imports like `from ..protocols import ...`):
python -m artificial_intelligence.homeworks.hw20260424.sdi2200160 --models bert distilbert deberta

# Standalone OOP scripts:
cd classes && python triangle.py
```

The AI homeworks take CLI flags (`--models`, `--epochs`, `--batch-size`, `--learning-rate`, `--max-length`) — see [hw20260424/sdi2200160.py](artificial_intelligence/homeworks/hw20260424/sdi2200160.py) for the current set.

## Conventions

These are non-obvious and enforced across the whole repo — don't reformat:

- **Indentation is TABS.** `.vscode/settings.json` sets `editor.insertSpaces: false` for Python; pylint is configured with `--indent-string='\t'` and `--max-line-length=132`.
- **`from __future__ import annotations`** at the top of every Python file.
- **PEP 695 generic syntax**: `class Classifier[DecodedSource, EncodedSource, ...]`, not `Generic[T]`. Protocols use `@typing.runtime_checkable`.
- **Student ID filename convention**: `sdi2200160.py` / `sdi2200160.bib` / `sdi2200160.ipynb` in every homework directory. Do not rename.
- **Reproducibility**: `RANDOM_STATE = 42` is a hard contract in the AI assignments — pass it to every splitter, shuffler, and torch seeder.
- pylint disables a long list of checks ([.vscode/settings.json](.vscode/settings.json#L17-L49)) — e.g. `missing-*-docstring`, `multiple-statements`, `unused-argument`. The codebase relies on those exceptions, so don't add docstrings or split one-liners just to satisfy a stock pylint config.

## Security note

The `HF_TOKEN` currently committed into [.env](.env) is a live HuggingFace token. `.env` is `.gitignore`d (line 125), but rotate the token if you suspect it was ever pushed.
