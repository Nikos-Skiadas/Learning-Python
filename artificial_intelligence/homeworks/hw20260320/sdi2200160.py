from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from data import data


# Reproducibility seed required by the assignment
RANDOM_STATE = 42


def build_question_answer_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single input text from the segmented question and the interview answer.

    We use the segmented 'question' field (single question), not the original
    multi-question journalist turn. This follows the assignment specification.
    """
    df = df.copy()
    df["question_answer"] = (
        df["question"].fillna("").astype(str).str.strip()
        + " | "
        + df["interview_answer"].fillna("").astype(str).str.strip()
    )
    return df


def main() -> None:
    # Load the TRAIN split for model development
    train_df = data["train"].to_pandas()
    assert isinstance(train_df, pd.DataFrame)

    # Keep only the columns we need and build the combined QA input
    train_df = build_question_answer_column(train_df)

    # Features (X) and labels (y)
    X = train_df["question_answer"]
    y = train_df["clarity_label"]

    # Stratified split so label proportions are preserved in train/validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training examples   : {len(X_train)}")
    print(f"Validation examples : {len(X_valid)}")
    print()

    # TF-IDF baseline representation
    # - lowercase=True for normalization
    # - stop_words='english' to remove common English function words
    # - token_pattern includes contractions like don't
    # - ngram_range=(1, 2) often helps in QA/NLP classification tasks
    # - max_df removes extremely frequent terms
    # - sublinear_tf often improves linear models with sparse features
    vectorizer = TfidfVectorizer(
        encoding="utf-8",
        decode_error="replace",
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b\w[\w']*\b",
        ngram_range=(1, 2),
        max_df=0.95,
        sublinear_tf=True,
    )

    # Fit ONLY on training data, then transform validation data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    print(f"TF-IDF vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print()

    # Logistic Regression baseline model required by the assignment
    # class_weight='balanced' can help if the classes are imbalanced
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_valid_tfidf)

    accuracy = accuracy_score(y_valid, y_pred)
    precision_macro = precision_score(y_valid, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_valid, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_valid, y_pred, average="macro", zero_division=0)

    print("=== TF-IDF + Logistic Regression Baseline ===")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {precision_macro:.4f}")
    print(f"Macro Recall    : {recall_macro:.4f}")
    print(f"Macro F1        : {f1_macro:.4f}")
    print()

    print("=== Classification Report ===")
    print(classification_report(y_valid, y_pred, digits=4, zero_division=0))
    print()

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_valid, y_pred))
    print()

    # Save metrics for later use in the report
    results = {
        "model": "TF-IDF + Logistic Regression",
        "random_state": RANDOM_STATE,
        "train_size": len(X_train),
        "validation_size": len(X_valid),
        "vocabulary_size": int(len(vectorizer.get_feature_names_out())),
        "accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "labels": sorted(pd.Series(y).dropna().unique().tolist()),
    }

    output_path = Path("baseline_metrics.json")
    output_path.write_text(
        json.dumps(results, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved metrics to: {output_path.resolve()}")


if __name__ == "__main__":
    main()