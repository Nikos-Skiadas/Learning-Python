from __future__ import annotations


import functools
import typing

import pandas
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics

from ..protocols import Scorer
from ..pipelines import Classifier
from ..data import data


# Reproducibility seed required by the assignment
RANDOM_STATE = 42


# ============================================================================
# Preprocessors
# ============================================================================

class CombineQA:
	"""Combines question and answer fields into a single text input."""

	def __call__(self, sources: pandas.DataFrame) -> list[str]:
		# TODO: Implement combining 'question' and 'interview_answer' fields
		# Hint: You decided to use " | " as separator in your previous implementation
		# Example: sources["question"].fillna("").str.strip() + " | " + sources["interview_answer"].fillna("").str.strip()
		# Return: list of combined question+answer strings
		raise NotImplementedError


# ============================================================================
# Metric Helpers
# ============================================================================

def macro_averaged(
	metric_fn: Scorer,
**kwargs) -> Scorer:
	"""Wrap sklearn metric with macro averaging and zero_division handling.

	Returns a partial function that can be used as a Scorer.
	"""
	return functools.partial(metric_fn,
		average = 'macro',
		zero_division = 0,
	**kwargs)


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
	"""Run TF-IDF + Logistic Regression baseline."""

	# Load training data
	train_df = data["train"].to_pandas()
	assert isinstance(train_df, pandas.DataFrame)

	# TODO: Extract features (X) and labels (y) from dataframe
	# Hint: Combine 'question' and 'interview_answer' fields with " | " separator
	# Example: combined = train_df["question"].fillna("").str.strip() + " | " + train_df["interview_answer"].fillna("").str.strip()
	X = pandas.Series(dtype = str)  # TODO: Fill with combined text strings
	y = pandas.Series(dtype = str)  # TODO: Fill with clarity_label values

	# Stratified train/validation split
	X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y,
		test_size = 0.2,
		random_state = RANDOM_STATE,
		stratify = y,
	)

	print(f"Training examples   : {len(X_train)}")
	print(f"Validation examples : {len(X_valid)}")
	print()

	# Build the pipeline components
	tfidf_encoder = sklearn.feature_extraction.text.TfidfVectorizer(
		encoding = "utf-8",
		decode_error = "replace",
		strip_accents = "unicode",
		lowercase = True,
		stop_words = "english",
		token_pattern = r"(?u)\b\w[\w']*\b",
		ngram_range = (1, 2),
		max_df = 0.95,
		sublinear_tf = True,
	)

	label_bicoder = sklearn.preprocessing.LabelEncoder()

	model = sklearn.linear_model.LogisticRegression(
		max_iter = 1000,
		class_weight = "balanced",
		random_state = RANDOM_STATE,
	)

	# Create the classifier pipeline
	classifier = Classifier(
		# preprocessors go here if you need any beyond CombineQA
		model = model,
		source_encoder = tfidf_encoder,
		target_bicoder = label_bicoder,  # type: ignore[arg-type]
	)

	# Compile with metrics
	# Note: type: ignore needed due to sklearn returning numpy.Float vs Python float
	classifier.compile(
		accuracy = sklearn.metrics.accuracy_score,
		precision = macro_averaged(sklearn.metrics.precision_score),
		recall = macro_averaged(sklearn.metrics.recall_score),
		f1 = macro_averaged(sklearn.metrics.f1_score),
	)

	# Fit the pipeline
	print("Training pipeline...")
	classifier.fit(X_train, y_train)
	print("Training complete.")
	print()

	# Evaluate on validation set
	print("=== TF-IDF + Logistic Regression Baseline ===")
	scores = classifier.score(
		X_valid,
		y_valid,
	)

	for metric_name, score_value in scores.items():
		print(f"{metric_name:12s}: {score_value:.4f}")
	print()

	# TODO: Generate predictions for analysis
	# y_pred = classifier.predict(X_valid)

	# TODO: Save results, generate plots, confusion matrix, etc.


if __name__ == "__main__":
	main()
