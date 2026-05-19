from __future__ import annotations


import argparse
import importlib
import pathlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy
import pandas

from joblib import Parallel, delayed, parallel_backend
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	adjusted_rand_score,
	f1_score,
	hamming_loss,
	jaccard_score,
	multilabel_confusion_matrix,
	precision_score,
	recall_score,
	silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_TEXT_C = 1.
DEFAULT_AUDIO_C = 10.
DEFAULT_EARLY_C = 1.
DEFAULT_BILINEAR_C = 0.1

DEFAULT_CV_N_JOBS = -1
DEFAULT_HEAVY_CV_N_JOBS = 4


if TYPE_CHECKING:
	import matplotlib.axes
	import matplotlib.figure


class SupportsPredictProba(Protocol):
	def predict_proba(self, x: numpy.ndarray) -> numpy.ndarray | list[numpy.ndarray]:
		...


class BilinearPooling(TransformerMixin, BaseEstimator):
	def __init__(self, n_text_features: int) -> None:
		self.n_text_features = n_text_features
		self.text_scaler_: StandardScaler | None = None
		self.audio_scaler_: StandardScaler | None = None

	def fit(self, x: numpy.ndarray,
		y: numpy.ndarray | None = None,
	) -> BilinearPooling:
		del y

		text, audio = self.split(x)
		self.text_scaler_ = StandardScaler().fit(text)
		self.audio_scaler_ = StandardScaler().fit(audio)

		return self

	def transform(self, x: numpy.ndarray) -> numpy.ndarray:
		text_scaler = self.text_scaler_
		audio_scaler = self.audio_scaler_

		if text_scaler is None or audio_scaler is None:
			raise RuntimeError("BilinearPooling must be fitted before transform.")

		text, audio = self.split(x)
		text = numpy.asarray(text_scaler.transform(text), dtype = float)
		audio = numpy.asarray(audio_scaler.transform(audio), dtype = float)

		return (text[:, :, numpy.newaxis] * audio[:, numpy.newaxis, :]).reshape(len(x), -1)

	def split(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
		values = numpy.asarray(x, dtype = float)

		return (
			values[:, :self.n_text_features],
			values[:, self.n_text_features:],
		)


class LateFusionClassifier(ClassifierMixin, BaseEstimator):
	def __init__(self,
		text_estimator: BaseEstimator,
		audio_estimator: BaseEstimator,
		n_text_features: int,
	) -> None:
		self.text_estimator = text_estimator
		self.audio_estimator = audio_estimator
		self.n_text_features = n_text_features
		self.text_model_: BaseEstimator | None = None
		self.audio_model_: BaseEstimator | None = None
		self.n_labels_: int | None = None
		self.classes_: numpy.ndarray | None = None

	def fit(self, x: numpy.ndarray,
		y: numpy.ndarray,
	) -> LateFusionClassifier:
		text, audio = self.split(x)
		self.text_model_ = cast(BaseEstimator, cast(Any, clone(self.text_estimator)).fit(text, y))
		self.audio_model_ = cast(BaseEstimator, cast(Any, clone(self.audio_estimator)).fit(audio, y))
		self.n_labels_ = y.shape[1]
		self.classes_ = numpy.arange(y.shape[1])

		return self

	def predict_proba(self, x: numpy.ndarray) -> numpy.ndarray:
		text_model = self.text_model_
		audio_model = self.audio_model_
		n_labels = self.n_labels_

		if text_model is None or audio_model is None or n_labels is None:
			raise RuntimeError("LateFusionClassifier must be fitted before predict_proba.")

		text, audio = self.split(x)

		text_proba = multilabel_proba(text_model, text, n_labels)
		audio_proba = multilabel_proba(audio_model, audio, n_labels)

		return (text_proba + audio_proba) / 2

	def predict(self, x: numpy.ndarray) -> numpy.ndarray:
		return probabilities_to_labels(self.predict_proba(x))

	def split(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
		values = numpy.asarray(x, dtype = float)

		return (
			values[:, :self.n_text_features],
			values[:, self.n_text_features:],
		)


@dataclass
class MultilabelData:
	dataset: pandas.DataFrame
	labels: list[str]
	y: pandas.DataFrame
	text: pandas.DataFrame
	audio: pandas.DataFrame
	fused: pandas.DataFrame


@dataclass
class PredictionResult:
	name: str
	y_pred: pandas.DataFrame
	y_proba: pandas.DataFrame
	metrics: pandas.Series


@dataclass
class ExperimentResults:
	data: MultilabelData
	predictions: dict[str, PredictionResult]
	metrics: pandas.DataFrame
	cv_scores: pandas.DataFrame
	cv_summary: pandas.DataFrame
	confusion_matrices: dict[str, dict[str, pandas.DataFrame]]
	clustering: pandas.DataFrame | None = None


@contextmanager
def optional_progress(enabled: bool,
	total_steps: int,
) -> Iterator[tuple[Any, Any]]:
	if not enabled:
		yield None, None
		return

	try:
		rich_progress = importlib.import_module("rich.progress")
	except ModuleNotFoundError:
		print("rich is not installed; continuing without progress bars.", file = sys.stderr)
		yield None, None
		return

	bar_column = getattr(rich_progress, "BarColumn")
	progress_type = getattr(rich_progress, "Progress")
	spinner_column = getattr(rich_progress, "SpinnerColumn")
	text_column = getattr(rich_progress, "TextColumn")
	time_elapsed_column = getattr(rich_progress, "TimeElapsedColumn")

	with progress_type(
		spinner_column(),
		text_column("[progress.description]{task.description}"),
		bar_column(),
		time_elapsed_column(),
	) as progress:
		overall_task = progress.add_task("Overall classification pipeline", total = total_steps)
		yield progress, overall_task


@contextmanager
def progress_step(progress: Any,
	overall_task: Any,
	description: str,
) -> Iterator[None]:
	if progress is None or overall_task is None:
		yield
		return

	task = progress.add_task(description, total = None)
	completed = False

	try:
		yield
		completed = True
	finally:
		progress.remove_task(task)

		if completed:
			progress.advance(overall_task)


def load_multilabel_data(data_dir: str | pathlib.Path,
	k: int = 5,
	label_count: int | None = None,
) -> MultilabelData:
	data_dir = pathlib.Path(data_dir)
	label_count = label_count or k

	dataset = pandas.read_csv(data_dir / f"dataset.{k}.csv", index_col = 0)
	genres = pandas.read_parquet(data_dir / f"dataset.{k}.genres.parquet")
	audio = pandas.read_parquet(data_dir / f"dataset.{k}.audio.parquet")
	text = pandas.read_parquet(data_dir / f"dataset.{k}.lyrics.parquet")

	index = dataset.index.intersection(genres.index).intersection(audio.index).intersection(text.index)
	labels = genres.loc[index].sum().sort_values(ascending = False).head(label_count).index.tolist()
	y = genres.loc[index, labels].astype(int)
	mask = y.sum(axis = "columns") > 0

	dataset = dataset.loc[index].loc[mask]
	y = y.loc[mask]
	text = text.loc[index].loc[mask]
	audio = audio.loc[index].loc[mask]

	fused = pandas.concat(
		[
			text.add_prefix("text__"),
			audio.add_prefix("audio__"),
		],
		axis = "columns",
	)

	return MultilabelData(
		dataset = dataset,
		labels = labels,
		y = y,
		text = text,
		audio = audio,
		fused = fused,
	)


def sample_data(data: MultilabelData,
	max_samples: int | None,
	random_state: int = 42,
) -> MultilabelData:
	if max_samples is None or max_samples >= len(data.y):
		return data

	index = data.y.sample(n = max_samples, random_state = random_state).index

	return MultilabelData(
		dataset = data.dataset.loc[index],
		labels = data.labels,
		y = data.y.loc[index],
		text = data.text.loc[index],
		audio = data.audio.loc[index],
		fused = data.fused.loc[index],
	)


def build_classifier(kind: str = "logistic",
	random_state: int = 42,
	regularization_c: float = 1.,
) -> BaseEstimator:
	if kind == "logistic":
		base = LogisticRegression(
			C = regularization_c,
			max_iter = 1000,
			class_weight = "balanced",
			solver = "liblinear",
			random_state = random_state,
		)

		return make_pipeline(
			StandardScaler(),
			OneVsRestClassifier(base, n_jobs = 1),
		)

	if kind == "random_forest":
		base = RandomForestClassifier(
			n_estimators = 300,
			class_weight = "balanced_subsample",
			n_jobs = 1,
			random_state = random_state,
		)

		return OneVsRestClassifier(base, n_jobs = 1)

	raise ValueError(f"Unsupported classifier kind: {kind}")


def build_bilinear_classifier(kind: str,
	n_text_features: int,
	random_state: int = 42,
	regularization_c: float = 1.,
) -> BaseEstimator:
	if kind == "logistic":
		base = LogisticRegression(
			C = regularization_c,
			max_iter = 1000,
			class_weight = "balanced",
			solver = "liblinear",
			random_state = random_state,
		)

		return make_pipeline(
			BilinearPooling(n_text_features),
			OneVsRestClassifier(base, n_jobs = 1),
		)

	if kind == "random_forest":
		base = RandomForestClassifier(
			n_estimators = 300,
			class_weight = "balanced_subsample",
			n_jobs = 1,
			random_state = random_state,
		)

		return make_pipeline(
			BilinearPooling(n_text_features),
			OneVsRestClassifier(base, n_jobs = 1),
		)

	raise ValueError(f"Unsupported classifier kind: {kind}")


def build_late_fusion_classifier(kind: str,
	n_text_features: int,
	random_state: int = 42,
	text_regularization_c: float = DEFAULT_TEXT_C,
	audio_regularization_c: float = DEFAULT_AUDIO_C,
) -> BaseEstimator:
	return LateFusionClassifier(
		text_estimator = build_classifier(
			kind,
			random_state = random_state,
			regularization_c = text_regularization_c,
		),
		audio_estimator = build_classifier(
			kind,
			random_state = random_state,
			regularization_c = audio_regularization_c,
		),
		n_text_features = n_text_features,
	)


def labelset_codes(y: pandas.DataFrame | numpy.ndarray) -> numpy.ndarray:
	if isinstance(y, pandas.DataFrame):
		values = y.to_numpy(dtype = int)
	else:
		values = numpy.asarray(y, dtype = int)

	return numpy.array(["".join(row.astype(str)) for row in values])


def make_cv_splits(y: pandas.DataFrame,
	n_splits: int = 10,
	random_state: int = 42,
) -> list[tuple[numpy.ndarray, numpy.ndarray]]:
	codes = pandas.Series(labelset_codes(y), index = y.index)

	if codes.value_counts().min() >= n_splits:
		splitter = StratifiedKFold(
			n_splits = n_splits,
			shuffle = True,
			random_state = random_state,
		)

		return list(splitter.split(numpy.zeros(len(y)), codes))

	splitter = KFold(
		n_splits = n_splits,
		shuffle = True,
		random_state = random_state,
	)

	return list(splitter.split(numpy.zeros(len(y))))


def multilabel_proba(estimator: BaseEstimator,
	x: numpy.ndarray,
	n_labels: int,
) -> numpy.ndarray:
	raw = cast(SupportsPredictProba, estimator).predict_proba(x)

	if isinstance(raw, list):
		classes = getattr(estimator, "classes_", [None] * len(raw))
		columns = []

		for probabilities, label_classes in zip(raw, classes):
			if label_classes is None:
				columns.append(probabilities[:, -1])
				continue

			positive = numpy.flatnonzero(numpy.asarray(label_classes) == 1)
			if len(positive) == 0:
				columns.append(numpy.zeros(len(x)))
			else:
				columns.append(probabilities[:, positive[0]])

		return numpy.column_stack(columns)

	probabilities = numpy.asarray(raw)

	if probabilities.ndim == 3:
		return probabilities[:, :, -1]

	if probabilities.shape[1] != n_labels:
		raise ValueError(
			f"Expected {n_labels} probability columns, got {probabilities.shape[1]}."
		)

	return probabilities


def probabilities_to_labels(probabilities: numpy.ndarray,
	threshold: float = 0.5,
	ensure_one: bool = True,
) -> numpy.ndarray:
	y_pred = (probabilities >= threshold).astype(int)

	if ensure_one:
		empty = y_pred.sum(axis = 1) == 0
		y_pred[empty, probabilities[empty].argmax(axis = 1)] = 1

	return y_pred


def score_multilabel(y_true: pandas.DataFrame | numpy.ndarray,
	y_pred: pandas.DataFrame | numpy.ndarray,
) -> pandas.Series:
	zero_division = cast(Any, 0)

	return pandas.Series({
		"subset_accuracy": accuracy_score(y_true, y_pred),
		"hamming_loss": hamming_loss(y_true, y_pred),
		"precision_macro": precision_score(y_true, y_pred, average = "macro", zero_division = zero_division),
		"recall_macro": recall_score(y_true, y_pred, average = "macro", zero_division = zero_division),
		"f1_macro": f1_score(y_true, y_pred, average = "macro", zero_division = zero_division),
		"precision_micro": precision_score(y_true, y_pred, average = "micro", zero_division = zero_division),
		"recall_micro": recall_score(y_true, y_pred, average = "micro", zero_division = zero_division),
		"f1_micro": f1_score(y_true, y_pred, average = "micro", zero_division = zero_division),
		"f1_samples": f1_score(y_true, y_pred, average = "samples", zero_division = zero_division),
		"jaccard_samples": jaccard_score(y_true, y_pred, average = "samples", zero_division = zero_division),
	})


def make_prediction_result(name: str,
	y_true: pandas.DataFrame,
	y_pred: numpy.ndarray,
	y_proba: numpy.ndarray,
) -> PredictionResult:
	y_pred_frame = pandas.DataFrame(
		y_pred,
		index = y_true.index,
		columns = y_true.columns,
	)
	y_proba_frame = pandas.DataFrame(
		y_proba,
		index = y_true.index,
		columns = y_true.columns,
	)

	return PredictionResult(
		name = name,
		y_pred = y_pred_frame,
		y_proba = y_proba_frame,
		metrics = score_multilabel(y_true, y_pred_frame),
	)


def cross_validated_predictions(name: str,
	estimator: BaseEstimator,
	x: pandas.DataFrame,
	y: pandas.DataFrame,
	splits: Iterable[tuple[numpy.ndarray, numpy.ndarray]],
	threshold: float = 0.5,
	n_jobs: int = DEFAULT_CV_N_JOBS,
) -> tuple[PredictionResult, pandas.DataFrame]:
	x_values = x.to_numpy(dtype = float)
	y_values = y.to_numpy(dtype = int)
	split_list = list(splits)
	pre_dispatch = n_jobs if n_jobs not in (None, -1) else "2*n_jobs"
	cv_result = cross_validate(
		estimator,
		x_values,
		y_values,
		cv = split_list,
		n_jobs = n_jobs,
		pre_dispatch = cast(Any, pre_dispatch),
		return_estimator = True,
		return_indices = True,  # type: ignore
	)

	probabilities = numpy.zeros(y_values.shape, dtype = float)
	fold_rows = []

	for fold, (model, test_idx) in enumerate(
		zip(cv_result["estimator"], cv_result["indices"]["test"]),
		start = 1,
	):
		fold_probabilities = multilabel_proba(model, x_values[test_idx], y.shape[1])
		fold_predictions = probabilities_to_labels(fold_probabilities, threshold = threshold)
		fold_scores = score_multilabel(y_values[test_idx], fold_predictions)

		probabilities[test_idx] = fold_probabilities

		fold_rows.append({
			"model": name,
			"fold": fold,
			"fit_time": cv_result["fit_time"][fold - 1],
			"score_time": cv_result["score_time"][fold - 1],
			**fold_scores.to_dict(),
		})

	predictions = probabilities_to_labels(probabilities, threshold = threshold)
	fold_frame = pandas.DataFrame(fold_rows).set_index(["model", "fold"])

	return make_prediction_result(name, y, predictions, probabilities), fold_frame


def confusion_by_label(y_true: pandas.DataFrame,
	y_pred: pandas.DataFrame,
) -> dict[str, pandas.DataFrame]:
	matrices = multilabel_confusion_matrix(y_true, y_pred)

	return {
		label: pandas.DataFrame(
			matrix,
			index = ["actual_0", "actual_1"],
			columns = ["pred_0", "pred_1"],
		)
		for label, matrix in zip(y_true.columns, matrices)
	}


def evaluate_kmeans_count(n_clusters: int,
	x: numpy.ndarray,
	y_values: numpy.ndarray,
	labelsets: numpy.ndarray,
	sample_size: int | None,
	random_state: int,
) -> dict[str, float | int]:
	clusters = KMeans(
		n_clusters = n_clusters,
		random_state = random_state,
		n_init = cast(Any, 10),
	).fit_predict(x)

	per_label_ari = [
		adjusted_rand_score(y_values[:, label], clusters)
		for label in range(y_values.shape[1])
	]

	if sample_size is not None and sample_size < len(x):
		silhouette = silhouette_score(
			x,
			clusters,
			sample_size = sample_size,
			random_state = random_state,
		)
	else:
		silhouette = silhouette_score(x, clusters)

	return {
		"n_clusters": n_clusters,
		"silhouette": silhouette,  # type: ignore
		"ari_labelset": adjusted_rand_score(labelsets, clusters),
		"ari_per_label_macro": float(numpy.mean(per_label_ari)),
	}


def evaluate_kmeans(data: MultilabelData,
	cluster_counts: Iterable[int] = range(2, 16),
	sample_size: int | None = 10000,
	random_state: int = 8312,
	n_jobs: int = DEFAULT_HEAVY_CV_N_JOBS,
) -> pandas.DataFrame:
	x = StandardScaler().fit_transform(data.fused.to_numpy(dtype = float))
	y_values = data.y.to_numpy(dtype = int)
	labelsets = labelset_codes(y_values)
	cluster_count_list = list(cluster_counts)
	pre_dispatch = n_jobs if n_jobs not in (None, -1) else "2*n_jobs"

	with parallel_backend("loky", inner_max_num_threads = 1):
		rows = Parallel(n_jobs = n_jobs, pre_dispatch = cast(Any, pre_dispatch))(
			delayed(evaluate_kmeans_count)(
				n_clusters,
				x,
				y_values,
				labelsets,
				sample_size,
				random_state,
			)
			for n_clusters in cluster_count_list
		)

	return pandas.DataFrame(rows).set_index("n_clusters")  # type: ignore


def run_experiments(data_dir: str | pathlib.Path,
	k: int = 5,
	label_count: int | None = None,
	n_splits: int = 10,
	classifier: str = "logistic",
	regularization_c: float | None = None,
	text_regularization_c: float = DEFAULT_TEXT_C,
	audio_regularization_c: float = DEFAULT_AUDIO_C,
	early_regularization_c: float = DEFAULT_EARLY_C,
	bilinear_regularization_c: float = DEFAULT_BILINEAR_C,
	threshold: float = 0.5,
	max_samples: int | None = None,
	include_bilinear: bool = True,
	include_clustering: bool = True,
	random_state: int = 42,
	show_progress: bool = False,
	heavy_n_jobs: int = DEFAULT_HEAVY_CV_N_JOBS,
) -> ExperimentResults:
	if regularization_c is not None:
		text_regularization_c = regularization_c
		audio_regularization_c = regularization_c
		early_regularization_c = regularization_c
		bilinear_regularization_c = regularization_c

	progress_steps = 6 + int(include_bilinear) + int(include_clustering)

	with optional_progress(show_progress, progress_steps) as (progress, overall_task):
		with progress_step(progress, overall_task, "Loading cached Part B data"):
			data = load_multilabel_data(data_dir, k = k, label_count = label_count)

		with progress_step(progress, overall_task, f"Preparing {n_splits}-fold CV splits"):
			data = sample_data(data, max_samples = max_samples, random_state = random_state)
			splits = make_cv_splits(data.y, n_splits = n_splits, random_state = random_state)

		with progress_step(progress, overall_task, f"Text-only CV ({n_splits} folds, n_jobs={DEFAULT_CV_N_JOBS})"):
			text, text_cv = cross_validated_predictions(
				"Text-only",
				build_classifier(classifier, random_state = random_state, regularization_c = text_regularization_c),
				data.text,
				data.y,
				splits,
				threshold = threshold,
				n_jobs = DEFAULT_CV_N_JOBS,
			)

		with progress_step(progress, overall_task, f"Audio-only CV ({n_splits} folds, n_jobs={DEFAULT_CV_N_JOBS})"):
			audio, audio_cv = cross_validated_predictions(
				"Audio-only",
				build_classifier(classifier, random_state = random_state, regularization_c = audio_regularization_c),
				data.audio,
				data.y,
				splits,
				threshold = threshold,
				n_jobs = DEFAULT_CV_N_JOBS,
			)

		with progress_step(
			progress,
			overall_task,
			f"Early-fusion CV ({n_splits} folds, n_jobs={DEFAULT_CV_N_JOBS})",
		):
			early, early_cv = cross_validated_predictions(
				"Early fusion",
				build_classifier(classifier, random_state = random_state, regularization_c = early_regularization_c),
				data.fused,
				data.y,
				splits,
				threshold = threshold,
				n_jobs = DEFAULT_CV_N_JOBS,
			)

		with progress_step(
			progress,
			overall_task,
			f"Late-fusion CV ({n_splits} folds, n_jobs={DEFAULT_CV_N_JOBS})",
		):
			late, late_cv = cross_validated_predictions(
				"Late fusion",
				build_late_fusion_classifier(
					classifier,
					n_text_features = data.text.shape[1],
					random_state = random_state,
					text_regularization_c = text_regularization_c,
					audio_regularization_c = audio_regularization_c,
				),
				data.fused,
				data.y,
				splits,
				threshold = threshold,
				n_jobs = DEFAULT_CV_N_JOBS,
			)

		bilinear = bilinear_cv = None

		if include_bilinear:
			with progress_step(
				progress,
				overall_task,
				f"Bilinear-pooling CV ({n_splits} folds, n_jobs={heavy_n_jobs})",
			):
				bilinear, bilinear_cv = cross_validated_predictions(
					"Bilinear pooling",
					build_bilinear_classifier(
						classifier,
						n_text_features = data.text.shape[1],
						random_state = random_state,
						regularization_c = bilinear_regularization_c,
					),
					data.fused,
					data.y,
					splits,
					threshold = threshold,
					n_jobs = heavy_n_jobs,
				)

		clustering = None

		if include_clustering:
			with progress_step(progress, overall_task, f"K-Means clustering evaluation (n_jobs={heavy_n_jobs})"):
				clustering = evaluate_kmeans(data, n_jobs = heavy_n_jobs)

	predictions = {
		result.name: result
		for result in (text, audio, early, late, bilinear)
		if result is not None
	}
	metrics = pandas.DataFrame(
		{
			name: result.metrics
			for name, result in predictions.items()
		}
	).T
	cv_scores = pandas.concat(
		[
			frame
			for frame in (text_cv, audio_cv, early_cv, late_cv, bilinear_cv)
			if frame is not None
		],
		axis = "index",
	)
	cv_summary = cast(pandas.DataFrame, cv_scores.groupby(level = "model").agg(["mean", "std"]))
	confusions = {
		name: confusion_by_label(data.y, result.y_pred)
		for name, result in predictions.items()
	}
	return ExperimentResults(
		data = data,
		predictions = predictions,
		metrics = metrics,
		cv_scores = cv_scores,
		cv_summary = cv_summary,
		confusion_matrices = confusions,
		clustering = clustering,
	)


def plot_f1_comparison(metrics: pandas.DataFrame,
	ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
	import matplotlib.pyplot

	if ax is None:
		fig, ax = matplotlib.pyplot.subplots(figsize = (8, 5))
	else:
		fig = ax.figure

	metrics["f1_macro"].plot(kind = "bar", ax = ax, color = "#3b6ea8")
	ax.set_ylabel("Macro F1")
	ax.set_xlabel("")
	ax.set_ylim(0, 1)
	ax.set_title("Multi-label Genre Classification")
	ax.tick_params(axis = "x", rotation = 20)

	return fig  # type: ignore


def plot_label_confusions(confusions: dict[str, pandas.DataFrame],
	title: str,
) -> matplotlib.figure.Figure:
	import matplotlib.pyplot

	n_labels = len(confusions)
	fig, axes = matplotlib.pyplot.subplots(
		1,
		n_labels,
		figsize = (3.2 * n_labels, 3.2),
	)
	axes = numpy.atleast_1d(axes)

	for ax, (label, matrix) in zip(axes, confusions.items()):
		ax.imshow(matrix.values, cmap = "Blues")
		ax.set_title(label)
		ax.set_xticks([0, 1], ["pred 0", "pred 1"])
		ax.set_yticks([0, 1], ["actual 0", "actual 1"])

		for row in range(2):
			for column in range(2):
				ax.text(
					column,
					row,
					f"{matrix.iloc[row, column]:,}",
					ha = "center",
					va = "center",
					color = "black",
				)

	fig.suptitle(title)
	fig.tight_layout()

	return fig


def plot_clustering_metrics(clustering: pandas.DataFrame,
	ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
	import matplotlib.pyplot

	if ax is None:
		fig, ax = matplotlib.pyplot.subplots(figsize = (8, 5))
	else:
		fig = ax.figure

	clustering[["silhouette", "ari_labelset", "ari_per_label_macro"]].plot(
		ax = ax,
		marker = "o",
	)
	ax.set_xlabel("Number of clusters")
	ax.set_ylabel("Score")
	ax.set_title("K-Means Clustering Evaluation")
	ax.grid(alpha = 0.25)

	return fig  # type: ignore


def safe_filename(name: str) -> str:
	slug = "".join(
		char.lower() if char.isalnum() else "_"
		for char in name
	).strip("_")

	return "_".join(part for part in slug.split("_") if part)


def save_evaluation_outputs(results: ExperimentResults,
	output: pathlib.Path,
) -> None:
	import os

	output.mkdir(parents = True, exist_ok = True)
	os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))

	import matplotlib.pyplot

	results.metrics.to_csv(output / "classification_metrics.csv")
	results.cv_scores.to_csv(output / "classification_cv_folds.csv")
	results.cv_summary.to_csv(output / "classification_cv_summary.csv")

	fig = plot_f1_comparison(results.metrics)
	fig.savefig(output / "f1_macro_comparison.png", dpi = 150, bbox_inches = "tight")
	matplotlib.pyplot.close(fig)

	for model_name, confusions in results.confusion_matrices.items():
		fig = plot_label_confusions(confusions, f"{model_name} Confusion Matrices")
		fig.savefig(
			output / f"confusion_{safe_filename(model_name)}.png",
			dpi = 150,
			bbox_inches = "tight",
		)
		matplotlib.pyplot.close(fig)

	if results.clustering is not None:
		results.clustering.to_csv(output / "clustering_metrics.csv")

		fig = plot_clustering_metrics(results.clustering)
		fig.savefig(output / "clustering_metrics.png", dpi = 150, bbox_inches = "tight")
		matplotlib.pyplot.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description = "Multi-label Part B experiments.")
	parser.add_argument("data", type = str, help = "Path to the cached data directory.")
	parser.add_argument("-k", type = int, default = 5, help = "Dataset top-k cache to load.")
	parser.add_argument("--labels", type = int, default = None, help = "Number of top labels to predict.")
	parser.add_argument("--folds", type = int, default = 10, help = "Number of cross-validation folds.")
	parser.add_argument("--classifier", choices = ["logistic", "random_forest"], default = "logistic")
	parser.add_argument("--regularization-c", type = float, default = None, help = "Override all logistic C values.")
	parser.add_argument("--text-c", type = float, default = DEFAULT_TEXT_C, help = "Logistic C for text-only features.")
	parser.add_argument("--audio-c", type = float, default = DEFAULT_AUDIO_C, help = "Logistic C for audio-only features.")
	parser.add_argument("--early-c", type = float, default = DEFAULT_EARLY_C, help = "Logistic C for concatenated early-fusion features.")
	parser.add_argument("--bilinear-c", type = float, default = DEFAULT_BILINEAR_C, help = "Logistic C for bilinear pooling features.")
	parser.add_argument("--threshold", type = float, default = 0.5)
	parser.add_argument("--max-samples", type = int, default = None, help = "Optional sample size for quick checks.")
	parser.add_argument("--skip-bilinear", action = "store_true")
	parser.add_argument("--skip-clustering", action = "store_true")
	parser.add_argument("--no-progress", action = "store_true", help = "Disable rich progress bars.")
	parser.add_argument("--output", type = str, default = None, help = "Optional directory for CSV and PNG outputs.")

	args = parser.parse_args()

	results = run_experiments(
		args.data,
		k = args.k,
		label_count = args.labels,
		n_splits = args.folds,
		classifier = args.classifier,
		regularization_c = args.regularization_c,
		text_regularization_c = args.text_c,
		audio_regularization_c = args.audio_c,
		early_regularization_c = args.early_c,
		bilinear_regularization_c = args.bilinear_c,
		threshold = args.threshold,
		max_samples = args.max_samples,
		include_bilinear = not args.skip_bilinear,
		include_clustering = not args.skip_clustering,
		show_progress = not args.no_progress,
	)

	print("Labels:", ", ".join(results.data.labels))
	print()
	print(results.metrics.round(4).to_string())
	print()
	print("Fold-wise CV summary:")
	print(
		results.cv_summary.loc[
			:,
			[
				("precision_macro", "mean"),
				("recall_macro", "mean"),
				("f1_macro", "mean"),
				("f1_macro", "std"),
			],
		].round(4).to_string()
	)

	if results.clustering is not None:
		print()
		print(results.clustering.round(4).to_string())

	if args.output:
		save_evaluation_outputs(results, pathlib.Path(args.output))


if __name__ == "__main__":
	main()
