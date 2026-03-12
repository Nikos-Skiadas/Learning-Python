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
  1. Preprocessing (single preprocessor, use `ChainPreprocessor` for multiple steps)
  2. Source encoding (features)
  3. Target encoding (labels)
  4. Model fitting/prediction
  5. Evaluation with metrics
  - Inherits from `sklearn.base.BaseEstimator` and `ClassifierMixin` for full sklearn compatibility

**Key Design Patterns**:

- **Protocol-based composition**: Components are duck-typed via `typing.Protocol`, allowing any implementation that matches the interface
- **Generic type parameters**: Uses PEP 695 syntax `[TypeVar]` for type safety across transformations
- **Fluent API**: `compile()` method returns `Self` for method chaining
- **Separation of encoding pipelines**: Source and target have independent encoders

**Data Loading** (in `data.py`):

- Uses HuggingFace `datasets` library
- Loads from `.env` configuration (via `dotenv`)
- Example: QEvasion dataset for response clarity classification

**Preprocessing Utilities** (in `preprocessing.py`):

- `ChainPreprocessor[Decoded]` - Chains multiple preprocessors sequentially
- `IdentityPreprocessor[Decoded]` - No-op preprocessor (returns input unchanged)

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
The `Classifier` accepts a single `preprocessor` argument. For multiple preprocessing steps, use `ChainPreprocessor` from `preprocessing.py`:

```python
from ..preprocessing import ChainPreprocessor

preprocessor = ChainPreprocessor(CleanText(), Lemmatize())
classifier = Classifier(preprocessor=preprocessor, ...)
```

Add custom preprocessing by implementing the `Preprocessor` protocol.

**Model Evaluation**:
Use `classifier.score(source, target, **metrics)` where metrics are `Scorer` implementations. Returns `dict[str, float]` of metric names to values.

## Hyperparameter Optimization

The `Classifier` class inherits from `sklearn.base.BaseEstimator` and `ClassifierMixin`, making it fully compatible with sklearn's hyperparameter optimization tools like `GridSearchCV` and `RandomizedSearchCV`.

**Usage with GridSearchCV**:

```python
from sklearn.model_selection import GridSearchCV
from ..preprocessing import ChainPreprocessor

preprocessor = ChainPreprocessor(CleanText(), Lemmatize())

classifier = Classifier(
    preprocessor=preprocessor,
    model=sklearn.linear_model.LogisticRegression(random_state=42),
    source_encoder=sklearn.feature_extraction.text.TfidfVectorizer(),
    target_bicoder=sklearn.preprocessing.LabelEncoder(),
)

classifier.compile(f1=macro_averaged(sklearn.metrics.f1_score))

param_grid = {
    'source_encoder__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'source_encoder__max_df': [0.85, 0.90, 0.95],
    'model__C': [0.1, 1.0, 10.0],
    'model__solver': ['lbfgs', 'liblinear'],
}

grid_search = GridSearchCV(
    classifier,
    param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)
best_classifier = grid_search.best_estimator_
```

**Parameter Naming Convention**:

- Use double-underscore notation: `component__parameter`
- `source_encoder__ngram_range` tunes the TfidfVectorizer's ngram_range
- `model__C` tunes the LogisticRegression's C parameter
- `target_bicoder__` prefix for label encoder parameters (if any)
- `preprocessor__` prefix for preprocessor parameters (if preprocessor has tunable params)

**Key Benefits**:

- Inherits from `BaseEstimator` - automatic `get_params()`/`set_params()` implementation
- Inherits from `ClassifierMixin` - provides default `score()` method (though we override it)
- Works seamlessly with all sklearn tools: `GridSearchCV`, `RandomizedSearchCV`, `cross_val_score`, etc.
- Framework-agnostic: sklearn models, PyTorch models (with protocol-compliant wrappers), or custom implementations

**See**: `hw20260320/sdi2200160_optimized.py` for complete GridSearchCV example.
