"""Run TF-IDF + Logistic Regression baseline."""


from __future__ import annotations


import functools
import re
import zipfile
import urllib.request

from pathlib import Path

import pandas
import numpy as np
import nltk
import nltk.stem
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.base

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
# Encoders
# ============================================================================

class Word2VecEncoder(
	sklearn.base.BaseEstimator,
	sklearn.base.TransformerMixin,
):
	"""Average Word2Vec embeddings for document representation.

	Loads pre-trained Word2Vec vectors and represents documents as
	the average of their word vectors.

	Inherits from BaseEstimator and TransformerMixin for sklearn compatibility:
	- Automatic get_params() and set_params() from BaseEstimator
	- Automatic fit_transform() from TransformerMixin
	"""

	def __init__(self, embeddings_path: str = "glove-wiki-gigaword-50", cache_dir: str = ".cache"):
		"""Initialize Word2Vec encoder.

		Args:
			embeddings_path: Path to embeddings file or preset name:
				Wiki-Gigaword (Wikipedia + Gigaword):
				- 'glove-wiki-gigaword-50' (50d, 400K words, ~70MB)
				- 'glove-wiki-gigaword-100' (100d, 400K words, ~140MB)
				- 'glove-wiki-gigaword-200' (200d, 400K words, ~280MB)
				- 'glove-wiki-gigaword-300' (300d, 400K words, ~420MB)

				Twitter (2B tweets, more informal):
				- 'glove-twitter-25' (25d, 1.2M words, ~100MB)
				- 'glove-twitter-50' (50d, 1.2M words, ~200MB)
				- 'glove-twitter-100' (100d, 1.2M words, ~400MB)
				- 'glove-twitter-200' (200d, 1.2M words, ~800MB)

				- Or provide your own path to .txt file
			cache_dir: Directory to cache downloaded embeddings
		"""
		self.embeddings_path = embeddings_path
		self.cache_dir = cache_dir
		self.word_vectors = None
		self.vector_size = None

	def download_glove(self, embeddings_name: str, dim: int) -> str:
		"""Download GloVe embeddings if not cached.

		Args:
			embeddings_name: 'wiki-gigaword' or 'twitter'
			dim: Embedding dimension
		"""
		cache_path = Path(self.cache_dir)
		cache_path.mkdir(exist_ok = True)

		# Determine file names and URLs based on embeddings type
		if embeddings_name == 'wiki-gigaword':
			embeddings_file = cache_path / f"glove.6B.{dim}d.txt"
			url = "https://nlp.stanford.edu/data/glove.6B.zip"
			zip_name = "glove.6B.zip"
		elif embeddings_name == 'twitter':
			embeddings_file = cache_path / f"glove.twitter.27B.{dim}d.txt"
			url = "https://nlp.stanford.edu/data/glove.twitter.27B.zip"
			zip_name = "glove.twitter.27B.zip"
		else:
			raise ValueError(f"Unknown embeddings type: {embeddings_name}")

		if embeddings_file.exists():
			print(f"Using cached embeddings: {embeddings_file}")
			return str(embeddings_file)

		# Download GloVe embeddings
		print(f"Downloading GloVe {embeddings_name} {dim}d embeddings...")
		zip_path = cache_path / zip_name

		# Download
		urllib.request.urlretrieve(url, zip_path)
		print(f"Downloaded to {zip_path}")

		# Extract the specific dimension file
		print(f"Extracting {embeddings_file.name}...")

		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extract(embeddings_file.name, cache_path)

		# Clean up zip
		zip_path.unlink()
		print(f"Embeddings ready at {embeddings_file}")

		return str(embeddings_file)

	def fit(self, source: pandas.Series, signal = None):
		"""Load pre-trained Word2Vec embeddings.

		Args:
			source: Series of text documents (not used, kept for API compatibility)
			signal: Optional target signal (unused)
		"""
		# Handle preset names
		if self.embeddings_path.startswith('glove-wiki-gigaword-'):
			dim = int(self.embeddings_path.split('-')[-1])
			embeddings_file = self.download_glove('wiki-gigaword', dim)

		elif self.embeddings_path.startswith('glove-twitter-'):
			dim = int(self.embeddings_path.split('-')[-1])
			embeddings_file = self.download_glove('twitter', dim)

		else:
			embeddings_file = self.embeddings_path

		# Load embeddings from text file
		print(f"Loading word vectors from {embeddings_file}...")
		self.word_vectors = {}

		with open(embeddings_file, 'r', encoding='utf-8') as f:
			for line_num, line in enumerate(f, 1):
				parts = line.rstrip().split(' ')
				word = parts[0]
				vector = np.array([float(x) for x in parts[1:]], dtype = np.float32)

				if self.vector_size is None:
					self.vector_size = len(vector)

				self.word_vectors[word] = vector

		print(f"Loaded {len(self.word_vectors)} word vectors of dimension {self.vector_size}")

		return self

	def transform(self, source: pandas.Series):
		"""Convert documents to averaged word vectors.

		Args:
			source: Series of text documents

		Returns:
			numpy array of shape (n_documents, vector_size)
		"""
		if self.word_vectors is None or self.vector_size is None:
			raise ValueError("Model must be fitted before transform")

		embeddings = []

		# Tokenize (matching TF-IDF's token_pattern to strip punctuation):
		for doc in source:
			words = re.findall(r"(?u)\b\w[\w']*\b", doc.lower())
			word_vectors = [self.word_vectors[word] for word in words if word in self.word_vectors]

			if word_vectors: embeddings.append(np.mean(word_vectors, axis = 0))  # Average word vectors
			else: embeddings.append(np.zeros(self.vector_size, dtype = np.float32))  # Zero vector for documents with no known words

		return np.array(embeddings)


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


def tfidf_encoder_factory() -> tuple[sklearn.feature_extraction.text.TfidfVectorizer, dict]:
	"""Create TF-IDF encoder with default params and search grid."""
	X_encoder = sklearn.feature_extraction.text.TfidfVectorizer(
		encoding = "utf-8",
		decode_error = "replace",
		strip_accents = "unicode",
		lowercase = True,
		stop_words = "english",
		token_pattern = r"(?u)\b\w[\w']*\b",
		sublinear_tf = True,
	)

	param_grid = dict(
		source_encoder__ngram_range = [(1, 2), (1, 3), (2, 3)],  # explore all possible ngram combinations from 1 to 3
		source_encoder__max_df = [.95, 1.],  # allow up to 95% document frequency to filter out very common terms or not
		source_encoder__min_df = [1, 2],  # minimum document frequency to include a term or not
	#	source_encoder__sublinear_tf = [True, False],  # whether to apply sublinear scaling to term frequencies or not
		model__C = [1e-1, 1, 1e+1],  # inverse of regularization strength (smaller values specify stronger regularization)
	)

	return X_encoder, param_grid


def word2vec_encoder_factory(embeddings_path: str = "glove-wiki-gigaword-50") -> tuple[Word2VecEncoder, dict]:
	"""Create Word2Vec encoder with default params and search grid.

	Args:
		embeddings_path: Pre-trained embeddings to use.
			Wiki-Gigaword options (Wikipedia + news):
			- 'glove-wiki-gigaword-50' (default, 50d, 400K words, ~70MB)
			- 'glove-wiki-gigaword-100' (100d, 400K words, ~140MB)
			- 'glove-wiki-gigaword-200' (200d, 400K words, ~280MB)
			- 'glove-wiki-gigaword-300' (300d, 400K words, ~420MB)

			Twitter options (2B tweets, informal language):
			- 'glove-twitter-25' (25d, 1.2M words, ~100MB)
			- 'glove-twitter-50' (50d, 1.2M words, ~200MB)
			- 'glove-twitter-100' (100d, 1.2M words, ~400MB)
			- 'glove-twitter-200' (200d, 1.2M words, ~800MB)

			- Or provide your own path to embeddings .txt file

	Returns:
		Tuple of (encoder, param_grid)
	"""
	X_encoder = Word2VecEncoder(embeddings_path = embeddings_path)

	# Pre-download embeddings to avoid parallel download conflicts in GridSearchCV
	# This ensures the embeddings are cached before parallel jobs start
	if embeddings_path.startswith('glove-wiki-gigaword-'):
		dim = int(embeddings_path.split('-')[-1])
		print(f"Pre-downloading GloVe wiki-gigaword {dim}d embeddings for caching...")
		X_encoder.download_glove('wiki-gigaword', dim)
		print("Embeddings cached and ready for grid search.\n")

	elif embeddings_path.startswith('glove-twitter-'):
		dim = int(embeddings_path.split('-')[-1])
		print(f"Pre-downloading GloVe twitter {dim}d embeddings for caching...")
		X_encoder.download_glove('twitter', dim)
		print("Embeddings cached and ready for grid search.\n")

	# For pre-trained word embeddings, only tune the classifier
	# (the encoder itself has no hyperparameters to tune)
	param_grid = dict(
		model__C = [
			1e-1,
			1e+0,
			1e+1,
		],  # inverse of regularization strength
		model__solver = ["saga"],  # saga solver supports L1 and L2 regularization and is efficient for large datasets
		model__l1_ratio = [
			0,
		#	.1,
			.2,
		#	.3,
			.4,
		#	.5,
			.6,
		#	.7,
			.8,
		#	.9,
			1.,
		],  # explore L2 (0), elastic net (0.5), and L1 (1) regularization
	)

	return X_encoder, param_grid


def optimize_with_grid_search(source_encoder, param_grid: dict,
	preprocessor = None,
):
	"""Optimize encoder + Logistic Regression using GridSearchCV.

	Args:
		source_encoder: Feature encoder (e.g., TfidfVectorizer, Word2VecEncoder)
		param_grid: Hyperparameter search space
		preprocessor: Optional preprocessor (defaults to CleanText + Lemmatize)
	"""
	# Prepare data
	print("Preparing data...")

	# For grid search, we will use the entire training set and let it handle the splitting internally for cross-validation.
	# The development set is not needed for this process, but we will prepare it anyway for potential future use.
	X_train, y_train, X_devel, y_devel, X_valid, y_valid = data_prep(frac=.0)  # grid search does its own splitting

	# Create preprocessor chain
	if preprocessor is None:
		preprocessor = ChainPreprocessor(
			CleanText(),
			Lemmatize(),
		)

	y_bicoder = sklearn.preprocessing.LabelEncoder()

	model = sklearn.linear_model.LogisticRegression(
		random_state = RANDOM_STATE,
		max_iter = 1000,  # allow room for convergence
		class_weight = "balanced",  # infer class weights from data to handle imbalance
	)

	classifier = Classifier(preprocessor, model, source_encoder, y_bicoder)  # type: ignore[arg-type]

	# Create GridSearchCV
	cv = 3  # 3-fold cross-validation
	grid_search = sklearn.model_selection.GridSearchCV(
		estimator = classifier,
		param_grid = param_grid,
		scoring = 'f1_macro',
		cv = cv,
		n_jobs = -1,  # Use all CPU cores
		verbose = 2,
		return_train_score = True,
	)

	# Run optimization
	print("\nStarting hyperparameter optimization...")
	print(f"Total combinations: {len(sklearn.model_selection.ParameterGrid(param_grid))}")
	print(f"Total fits: {len(sklearn.model_selection.ParameterGrid(param_grid)) * cv} ({cv}-fold CV)")
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
	import argparse

	parser = argparse.ArgumentParser(description = "Run TF-IDF or Word2Vec encoder with Logistic Regression baseline.")
	parser.add_argument("--encoders", nargs = "+", required = True,
		help = "Encoder type to run, for example: tfidf word2vec"
	)
	parser.add_argument("--embeddings",
		default = "glove-wiki-gigaword-50",
		help = """Pre-trained word embeddings (default: glove-wiki-gigaword-50).
		Options: glove-wiki-gigaword-{50,100,200,300}, glove-twitter-{25,50,100,200}."""
	)

	args = parser.parse_args()

	for encoder_type in args.encoders:
		print("=" * 80)
		print(f"{encoder_type.upper()} BASELINE")
		print("=" * 80)
		print()

		if encoder_type == "tfidf": encoder, search = tfidf_encoder_factory()
		elif encoder_type == "word2vec": encoder, search = word2vec_encoder_factory(embeddings_path = args.embeddings)
		else:
			print(f"Unknown encoder type: {encoder_type}")
			print("Valid options: tfidf, word2vec")

			continue

		optimize_with_grid_search(encoder, search)
