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


def data_prep(frac: float = .0):
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
			test_size = .2,
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

def tfidf_logistic_regression(X_train, y_train, X_devel, y_devel, X_valid, y_valid):
	# Build the pipeline components
	X_encoder = sklearn.feature_extraction.text.TfidfVectorizer(
		encoding = "utf-8",
		decode_error = "replace",
		strip_accents = "unicode",
		lowercase = True,
	#	preprocessor = None,
	#	tokenizer = None,
	#	analyzer = "word",
		stop_words = "english",
		token_pattern = r"(?u)\b\w[\w']*\b",  # include contractions  # r"(?u)\b\w\w+\b"
		ngram_range = (1, 3),
		max_df = .95,
	#	min_df = .01,
	#	max_features = None,
	#	norm = "l2",
	#	use_idf = True,
	#	smooth_idf = True,
		sublinear_tf = True,
	)
	y_bicoder = sklearn.preprocessing.LabelEncoder()

	model = sklearn.linear_model.LogisticRegression(
		max_iter = 1000,
		class_weight = "balanced",
		random_state = RANDOM_STATE,
	)

	# Create preprocessors
	text_cleaner = CleanText()
	lemmatizer = Lemmatize()

	# Create the classifier pipeline
	classifier = Classifier(
		text_cleaner,
		lemmatizer,
		model = model,
		source_encoder = X_encoder,
		target_bicoder = y_bicoder,  # type: ignore[arg-type]
	)

	# Compile with metrics
	# Note: type: ignore needed due to sklearn returning numpy.Float vs Python float
	classifier.compile(
		accuracy = sklearn.metrics.accuracy_score,
		precision = macro_averaged(sklearn.metrics.precision_score),
		recall = macro_averaged(sklearn.metrics.recall_score),
		f1 = macro_averaged(sklearn.metrics.f1_score),
	)

	# Fit the pipeline~
	print("Training pipeline...")
	classifier.fit(
		X_train,
		y_train,
	)
	print("Training complete.")
	print()

	# Evaluate on validation set
	print("TF-IDF + Logistic Regression Baseline:")
	scores = classifier.score(
		X_valid,
		y_valid,
	)

	for metric_name, score_value in scores.items():
		print(f"{metric_name:12s} {score_value:.4f}")

	print()

#	y_pred = classifier.predict(X_valid)


if __name__ == "__main__":
	splits = data_prep()
	tfidf_logistic_regression(*splits)
