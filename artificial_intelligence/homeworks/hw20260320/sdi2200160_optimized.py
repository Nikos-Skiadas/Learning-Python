"""Hyperparameter optimization using GridSearchCV."""


from __future__ import annotations


import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics

from ..pipelines import Classifier
from ..preprocessing import ChainPreprocessor
from .sdi2200160 import CleanText, Lemmatize, macro_averaged, data_prep, RANDOM_STATE


def optimize_with_grid_search():
	"""Optimize TF-IDF + Logistic Regression using GridSearchCV."""
	# Prepare data
	print("Preparing data...")
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
	)

	y_bicoder = sklearn.preprocessing.LabelEncoder()

	model = sklearn.linear_model.LogisticRegression(
		random_state = RANDOM_STATE,
	)

	classifier = Classifier(
		preprocessor = preprocessor,
		model = model,
		source_encoder = X_encoder,
		target_bicoder = y_bicoder,  # type: ignore[arg-type]
	)

	classifier.compile(
		accuracy = sklearn.metrics.accuracy_score,
		precision = macro_averaged(sklearn.metrics.precision_score),
		recall = macro_averaged(sklearn.metrics.recall_score),
		f1 = macro_averaged(sklearn.metrics.f1_score),
	)

	# Define parameter grid
	param_grid = {
		'source_encoder__ngram_range': [(1, 1), (1, 2), (1, 3)],
		'source_encoder__max_df': [.85, .90, .95],
		'source_encoder__min_df': [1, 2, 5],
		'source_encoder__sublinear_tf': [True, False],
		'model__C': [1e-1, 1, 1e+1],
		'model__class_weight': ['balanced', None],
		'model__solver': ['lbfgs', 'liblinear'],
		'model__max_iter': [500, 1000],
	}

	# Create GridSearchCV
	grid_search = sklearn.model_selection.GridSearchCV(
		estimator = classifier,
		param_grid = param_grid,
		scoring = 'f1_macro',
		cv = 5,  # 5-fold cross-validation
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
	val_scores = best_classifier.score(X_valid, y_valid)
	print()
	for name, score in val_scores.items():
		print(f"{name:12s} {score:.4f}")

	print()

	return grid_search


if __name__ == "__main__":
	grid_search_run = optimize_with_grid_search()
