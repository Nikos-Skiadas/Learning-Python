"""Run TF-IDF + Logistic Regression baseline."""


from __future__ import annotations


import functools
import re

import pandas
import nltk
import nltk.stem
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics

from ..protocols import Scorer
from ..pipelines import Classifier
from ..preprocessing import ChainPreprocessor
from ..data import data


# Reproducibility seed required by the assignment
RANDOM_STATE = 42


# ============================================================================
# Preprocessors
# ============================================================================

class CleanText:
	"""Remove noise: URLs, emails, extra whitespace."""

	def __call__(self, source: pandas.Series) -> pandas.Series:
		cleaned = source.str.replace(r'http\S+|www\.\S+', ' ', regex = True)  # Remove URLs
		cleaned = cleaned.str.replace(r'\S+@\S+', ' ', regex = True)  # Remove email addresses
		cleaned = cleaned.str.replace(r'\s+', ' ', regex = True).str.strip()  # Remove extra whitespace

		return cleaned


class Lemmatize:
	"""Reduce words to base forms (running → run, better → good)."""

	def __init__(self):
		nltk.download('wordnet', quiet = True)
		nltk.download('omw-1.4', quiet = True)

		self.lemmatizer = nltk.stem.WordNetLemmatizer()

	def __call__(self, source: pandas.Series) -> pandas.Series:
		return source.apply(
			lambda text: ' '.join(
				self.lemmatizer.lemmatize(word) for word in text.split()
			)
		)


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


def data_prep(frac: float = .0) -> tuple[
	pandas.Series,
	pandas.Series,
	pandas.Series,
	pandas.Series,
	pandas.Series,
	pandas.Series,
]:
	# Load training data
	train_data = data["train"].to_pandas(); assert isinstance(train_data, pandas.DataFrame)
	valid_data = data["test" ].to_pandas(); assert isinstance(valid_data, pandas.DataFrame)

	# Create combined text features and labels (handling missing values)
	separator = " | "
	X = train_data.question.fillna("").str.strip() + separator + train_data.interview_answer.fillna("").str.strip()
	y = train_data.clarity_label.fillna("").str.strip()

	# Stratified train/validation split
	if frac:
		X_train, X_devel, y_train, y_devel = sklearn.model_selection.train_test_split(X, y,
			test_size = frac,
			random_state = RANDOM_STATE,
			stratify = y,
		)

	else:
		X_train, y_train = X, y
		X_devel, y_devel = pandas.Series(dtype = str), pandas.Series(dtype = str)

	X_valid = valid_data.question.fillna("").str.strip() + separator + valid_data.interview_answer.fillna("").str.strip()
	y_valid = valid_data.clarity_label.fillna("").str.strip()

	print(f"Training    examples: {len(X_train)}")
	print(f"Development examples: {len(X_devel)}")
	print(f"Validation  examples: {len(X_valid)}")
	print()

	return X_train, y_train, X_devel, y_devel, X_valid, y_valid


def optimize_with_grid_search():
	"""Optimize TF-IDF + Logistic Regression using GridSearchCV."""
	# Prepare data
	print("Preparing data...")

	# For grid search, we will use the entire training set and let it handle the splitting internally for cross-validation.
	# The development set is not needed for this process, but we will prepare it anyway for potential future use.
	X_train, y_train, X_devel, y_devel, X_valid, y_valid = data_prep(frac=.0)  # grid search does its own splitting

	# Create preprocessor chain
	preprocessor = ChainPreprocessor(
		CleanText(),
		Lemmatize(),
	)

	X_encoder = sklearn.feature_extraction.text.TfidfVectorizer(
		encoding = "utf-8",
		decode_error = "replace",
		strip_accents = "unicode",
		lowercase = True,
		stop_words = "english",
		token_pattern = r"(?u)\b\w[\w']*\b",
		sublinear_tf = True,
	)

	y_bicoder = sklearn.preprocessing.LabelEncoder()

	model = sklearn.linear_model.LogisticRegression(
		random_state = RANDOM_STATE,
		max_iter = 1000,  # allow room for convergence
		class_weight = "balanced",  # infer class weights from data to handle imbalance
	)

	classifier = Classifier(preprocessor, model, X_encoder, y_bicoder)  # type: ignore[arg-type]

	# Define parameter grid
	param_grid = dict(
		source_encoder__ngram_range = [(1, 2), (1, 3), (2, 3)],  # explore all possible ngram combinations from 1 to 3
		source_encoder__max_df = [.95, 1.],  # allow up to 95% document frequency to filter out very common terms or not
		source_encoder__min_df = [1, 2],  # minimum document frequency to include a term or not
	#	source_encoder__sublinear_tf = [True, False],  # whether to apply sublinear scaling to term frequencies or not
		model__C = [1e-1, 1, 1e+1],  # inverse of regularization strength (smaller values specify stronger regularization)
	)

	# Create GridSearchCV
	grid_search = sklearn.model_selection.GridSearchCV(
		estimator = classifier,
		param_grid = param_grid,
		scoring = 'f1_macro',
		cv = 3,  # 3-fold cross-validation
		n_jobs = -1,  # Use all CPU cores
		verbose = 2,
		return_train_score = True,
	)

	# Run optimization
	print("\nStarting hyperparameter optimization...")
	print(f"Total combinations: {len(sklearn.model_selection.ParameterGrid(param_grid))}")
	print(f"Total fits: {len(sklearn.model_selection.ParameterGrid(param_grid)) * 5} (5-fold CV)")
	print()

	grid_search.fit(X_train, y_train)  # type: ignore[no-untyped-call]

	# Print results
	print("\n" + "="*80)
	print("OPTIMIZATION RESULTS")
	print("="*80)
	print(f"\nBest F1 (macro, cross-validation): {grid_search.best_score_:.4f}")
	print("\nBest parameters:")

	for param, value in grid_search.best_params_.items():
		print(f"  {param:35s} = {value}")

	# Evaluate on validation set
	print("\n" + "="*80)
	print("VALIDATION SET EVALUATION")
	print("="*80)

	best_classifier = grid_search.best_estimator_
	best_classifier.compile(
		accuracy = sklearn.metrics.accuracy_score,
		precision = macro_averaged(sklearn.metrics.precision_score),
		recall = macro_averaged(sklearn.metrics.recall_score),
		f1 = macro_averaged(sklearn.metrics.f1_score),
	)
	val_scores = best_classifier.score(X_valid, y_valid)

	print()

	for name, score in val_scores.items():
		print(f"{name:12s} {score:.4f}")

	print()

	y_pred = pandas.Series(best_classifier.predict(X_valid), name = "Predicted")
	y_pred.index.name = "Id"

	print(y_pred.sample(10))
	print()

	y_pred.to_csv("submission.csv")

#	return grid_search


if __name__ == "__main__":
	optimize_with_grid_search()
