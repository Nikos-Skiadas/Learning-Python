# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Python Version**: 3.14.3

**Virtual Environment**: Project uses `.venv/` at repository root. Activate with:

```bash
source ../../.venv/bin/activate
```

**Install Dependencies**:

```bash
pip install -r ../../requirements.txt
```

**Key Dependencies**: torch, scikit-learn, transformers, sentence-transformers, datasets, pandas, numpy, nltk, matplotlib, seaborn

## Running Code

**Run a homework assignment**:

```bash
cd hw20260320
python sdi2200160.py
```

**Run with pytest** (if tests exist):

```bash
pytest -s
```

## Code Architecture

### Pipeline-Based ML Framework

This codebase uses a **Protocol-based generic architecture** for machine learning pipelines. The design separates concerns through composable components:

**Core Protocols** (in `protocols.py`):

- `Preprocessor[Decoded]` - Transforms raw data collections
- `Encoder[Decoded, Encoded]` - Fits and transforms data (e.g., TF-IDF vectorizer)
- `Bicoder[Decoded, Encoded]` - Encoder with inverse transform (e.g., label encoder)
- `Model[Source, Target]` - ML model with fit/predict interface
- `Scorer[Target, Result]` - Evaluation metric

**Pipeline Orchestration** (in `pipelines.py`):

- `Classifier[DecodedSource, EncodedSource, EncodedTarget, DecodedTarget]` - Orchestrates the full pipeline:
  1. Preprocessing (multiple preprocessors chained)
  2. Source encoding (features)
  3. Target encoding (labels)
  4. Model fitting/prediction
  5. Evaluation with metrics

**Key Design Patterns**:

- **Protocol-based composition**: Components are duck-typed via `typing.Protocol`, allowing any implementation that matches the interface
- **Generic type parameters**: Uses PEP 695 syntax `[TypeVar]` for type safety across transformations
- **Fluent API**: `compile()` method returns `Self` for method chaining
- **Separation of encoding pipelines**: Source and target have independent encoders

**Data Loading** (in `data.py`):

- Uses HuggingFace `datasets` library
- Loads from `.env` configuration (via `dotenv`)
- Example: QEvasion dataset for response clarity classification

### Homework Structure

Each homework is in a dated subdirectory (e.g., `hw20260320/`):

- Main implementation file: `sdi{student_id}.py`
- Bibliography file: `sdi{student_id}.bib`
- Assignment specification: `specification.pdf`
- Output artifacts: JSON metrics, plots, logs (gitignored)

## Coding Conventions

**Indentation**: Use **TABS**, not spaces (configured in `.vscode/settings.json`)

**Modern Python Features**:

- Always use `from __future__ import annotations` at top of file
- Use PEP 695 generic syntax: `class Foo[T]` instead of `class Foo(Generic[T])`
- Type hints are expected on function signatures

**Import Style**:

- Relative imports within the homeworks package: `from .protocols import ...`
- Absolute imports for external libraries

**Naming**:

- Student ID convention: `sdi2200160` (appears in filenames and code)
- Constants in UPPERCASE: `RANDOM_STATE = 42`

**ML Reproducibility**:

- Always set `random_state=42` for reproducible results
- Use stratified splits when appropriate (`stratify=y` in `train_test_split`)

**Error Handling**:

- Explicit encoding handling: `encoding="utf-8"`, `decode_error="replace"`
- Zero division protection: `zero_division=0` in sklearn metrics

## Common Workflows

**Creating a new homework assignment**:

1. Create directory: `hwYYYYMMDD/`
2. Add `__init__.py` (empty or with exports)
3. Create main file: `sdi{student_id}.py`
4. Import shared utilities from parent: `from ..protocols import ...`, `from ..data import ...`

**Implementing a new Encoder**:
Must provide `fit(source, signal=None)` and `transform(source)` methods that match the `Encoder` protocol. For reversible encodings, implement `Bicoder` with `inverse_transform()`.

**Extending the Classifier**:
The `Classifier` accepts multiple preprocessors via `*preprocessors` and chains them in order. Add custom preprocessing by implementing the `Preprocessor` protocol.

**Model Evaluation**:
Use `classifier.score(source, target, **metrics)` where metrics are `Scorer` implementations. Returns `dict[str, float]` of metric names to values.
