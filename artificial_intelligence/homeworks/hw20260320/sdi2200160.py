from __future__ import annotations


import typing

import numpy
import pandas
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics

from ..protocols import Preprocessor, Encoder, Bicoder, Model, Scorer
from ..pipelines import Classifier
from ..data import data


# Reproducibility seed required by the assignment
RANDOM_STATE = 42


# ============================================================================
# Preprocessors
# ============================================================================

class CombineQA(Preprocessor[dict]):
	"""Combines question and answer fields into a single text input."""

	def __call__(self, sources: typing.Collection[dict]) -> list[str]:
		# TODO: Implement combining 'question' and 'interview_answer' fields
		# Hint: You decided to use " | " as separator in your previous implementation
		# Return: list of combined question+answer strings
		raise NotImplementedError


# ============================================================================
# Encoders
# ============================================================================

class TfidfEncoder(Encoder[str, typing.Any]):
	"""Wrapper for sklearn's TfidfVectorizer to match Encoder protocol."""

	def __init__(self, **kwargs):
		self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(**kwargs)

	def fit(self, source: typing.Collection[str], signal: typing.Any | None = None) -> typing.Self:
		self.vectorizer.fit(list(source))
		return self

	def transform(self, source: typing.Collection[str]) -> typing.Any:
		# Returns scipy sparse matrix
		return self.vectorizer.transform(list(source))


class LabelBicoder(Bicoder[str, typing.Any]):
	"""Wrapper for sklearn's LabelEncoder to match Bicoder protocol."""

	def __init__(self):
		self.encoder = sklearn.preprocessing.LabelEncoder()

	def fit(self, source: typing.Collection[str], signal: typing.Any | None = None) -> typing.Self:
		self.encoder.fit(list(source))
		return self

	def transform(self, source: typing.Collection[str]) -> typing.Any:
		return self.encoder.transform(list(source))

	def inverse_transform(self, target: typing.Collection[typing.Any]) -> typing.Any:
		return self.encoder.inverse_transform(numpy.array(list(target)))


# ============================================================================
# Models
# ============================================================================

class LogisticModel(Model[typing.Any, typing.Any]):
	"""Wrapper for sklearn's LogisticRegression to match Model protocol."""

	def __init__(self, **kwargs):
		self.model = sklearn.linear_model.LogisticRegression(**kwargs)

	def fit(self, source: typing.Any, target: typing.Any) -> typing.Self:
		self.model.fit(source, target)
		return self

	def predict(self, source: typing.Any) -> typing.Any:
		return self.model.predict(source)


# ============================================================================
# Scorers
# ============================================================================

class AccuracyScorer(Scorer[typing.Any, float]):
	"""Accuracy metric scorer."""

	def __call__(self, true: typing.Any, pred: typing.Any) -> float:
		return float(sklearn.metrics.accuracy_score(true, pred))


class MacroPrecisionScorer(Scorer[typing.Any, float]):
	"""Macro-averaged precision scorer."""

	def __call__(self, true: typing.Any, pred: typing.Any) -> float:
		return float(sklearn.metrics.precision_score(true, pred, average='macro', zero_division=0))


class MacroRecallScorer(Scorer[typing.Any, float]):
	"""Macro-averaged recall scorer."""

	def __call__(self, true: typing.Any, pred: typing.Any) -> float:
		return float(sklearn.metrics.recall_score(true, pred, average='macro', zero_division=0))


class MacroF1Scorer(Scorer[typing.Any, float]):
	"""Macro-averaged F1 scorer."""

	def __call__(self, true: typing.Any, pred: typing.Any) -> float:
		return float(sklearn.metrics.f1_score(true, pred, average='macro', zero_division=0))


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
	# Example: train_df["combined"] = train_df["question"].str.strip() + " | " + train_df["interview_answer"].str.strip()
	X: list[str] = []  # TODO: Fill with combined text strings
	y: list[str] = []  # TODO: Fill with clarity_label values

	# Stratified train/validation split
	X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=RANDOM_STATE,
		stratify=y,
	)

	print(f"Training examples   : {len(X_train)}")
	print(f"Validation examples : {len(X_valid)}")
	print()

	# Build the pipeline
	# TODO: Configure TF-IDF parameters based on your previous experiments
	tfidf_encoder = TfidfEncoder(
		# encoding="utf-8",
		# lowercase=True,
		# stop_words="english",
		# ngram_range=(1, 2),
		# ...
	)

	label_bicoder = LabelBicoder()

	# TODO: Configure LogisticRegression parameters
	model = LogisticModel(
		max_iter=1000,
		class_weight="balanced",
		random_state=RANDOM_STATE,
		# multi_class='multinomial',  # or 'ovr'
	)

	# Create the classifier pipeline
	classifier = Classifier(
		# preprocessors go here if you need any beyond CombineQA
		model=model,
		source_encoder=tfidf_encoder,
		target_bicoder=label_bicoder,
	)

	# Compile with metrics
	classifier.compile(
		accuracy=AccuracyScorer(),
		precision=MacroPrecisionScorer(),
		recall=MacroRecallScorer(),
		f1=MacroF1Scorer(),
	)

	# Fit the pipeline
	print("Training pipeline...")
	classifier.fit(X_train, y_train)
	print("Training complete.")
	print()

	# Evaluate on validation set
	print("=== TF-IDF + Logistic Regression Baseline ===")
	scores = classifier.score(X_valid, y_valid,
		accuracy=AccuracyScorer(),
		precision=MacroPrecisionScorer(),
		recall=MacroRecallScorer(),
		f1=MacroF1Scorer(),
	)

	for metric_name, score_value in scores.items():
		print(f"{metric_name:12s}: {score_value:.4f}")
	print()

	# TODO: Generate predictions for analysis
	# y_pred = classifier.predict(X_valid)

	# TODO: Save results, generate plots, confusion matrix, etc.


if __name__ == "__main__":
	main()
