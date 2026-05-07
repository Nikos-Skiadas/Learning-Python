from __future__ import annotations


import argparse
import pathlib
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Protocol, cast

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot
import numpy
import pandas

from sklearn.base import BaseEstimator, clone
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SupportsPredictProba(Protocol):
	def predict_proba(self, x: numpy.ndarray) -> numpy.ndarray | list[numpy.ndarray]:
		...


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
	confusion_matrices: dict[str, dict[str, pandas.DataFrame]]
	clustering: pandas.DataFrame | None = None


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
) -> BaseEstimator:
	if kind == "logistic":
		base = LogisticRegression(
			max_iter = 1000,
			class_weight = "balanced",
			solver = "liblinear",
			random_state = random_state,
		)

		return make_pipeline(
			StandardScaler(),
			OneVsRestClassifier(base),
		)

	if kind == "random_forest":
		return RandomForestClassifier(
			n_estimators = 300,
			class_weight = "balanced_subsample",
			n_jobs = -1,
			random_state = random_state,
		)

	raise ValueError(f"Unsupported classifier kind: {kind}")


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
	return pandas.Series({
		"subset_accuracy": accuracy_score(y_true, y_pred),
		"hamming_loss": hamming_loss(y_true, y_pred),
		"precision_macro": precision_score(y_true, y_pred, average = "macro", zero_division = 0),
		"recall_macro": recall_score(y_true, y_pred, average = "macro", zero_division = 0),
		"f1_macro": f1_score(y_true, y_pred, average = "macro", zero_division = 0),
		"precision_micro": precision_score(y_true, y_pred, average = "micro", zero_division = 0),
		"recall_micro": recall_score(y_true, y_pred, average = "micro", zero_division = 0),
		"f1_micro": f1_score(y_true, y_pred, average = "micro", zero_division = 0),
		"f1_samples": f1_score(y_true, y_pred, average = "samples", zero_division = 0),
		"jaccard_samples": jaccard_score(y_true, y_pred, average = "samples", zero_division = 0),
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
) -> PredictionResult:
	x_values = x.to_numpy(dtype = float)
	y_values = y.to_numpy(dtype = int)
	probabilities = numpy.zeros(y_values.shape, dtype = float)

	for train_idx, test_idx in splits:
		model = clone(estimator)
		model.fit(x_values[train_idx], y_values[train_idx])
		probabilities[test_idx] = multilabel_proba(model, x_values[test_idx], y.shape[1])

	predictions = probabilities_to_labels(probabilities, threshold = threshold)

	return make_prediction_result(name, y, predictions, probabilities)


def late_fusion(text: PredictionResult,
	audio: PredictionResult,
	y_true: pandas.DataFrame,
	threshold: float = 0.5,
) -> PredictionResult:
	probabilities = (text.y_proba.to_numpy() + audio.y_proba.to_numpy()) / 2
	predictions = probabilities_to_labels(probabilities, threshold = threshold)

	return make_prediction_result(
		"Late fusion",
		y_true,
		predictions,
		probabilities,
	)


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


def evaluate_kmeans(data: MultilabelData,
	cluster_counts: Iterable[int] = range(2, 16),
	sample_size: int | None = 10000,
	random_state: int = 8312,
) -> pandas.DataFrame:
	x = StandardScaler().fit_transform(data.fused.to_numpy(dtype = float))
	labelsets = labelset_codes(data.y)
	rows = []

	for n_clusters in cluster_counts:
		clusters = KMeans(
			n_clusters = n_clusters,
			random_state = random_state,
			n_init = 10,
		).fit_predict(x)

		per_label_ari = [
			adjusted_rand_score(data.y[label].to_numpy(), clusters)
			for label in data.labels
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

		rows.append({
			"n_clusters": n_clusters,
			"silhouette": silhouette,
			"ari_labelset": adjusted_rand_score(labelsets, clusters),
			"ari_per_label_macro": float(numpy.mean(per_label_ari)),
		})

	return pandas.DataFrame(rows).set_index("n_clusters")


def run_experiments(data_dir: str | pathlib.Path,
	k: int = 5,
	label_count: int | None = None,
	n_splits: int = 10,
	classifier: str = "logistic",
	threshold: float = 0.5,
	max_samples: int | None = None,
	include_clustering: bool = True,
	random_state: int = 42,
) -> ExperimentResults:
	data = load_multilabel_data(data_dir, k = k, label_count = label_count)
	data = sample_data(data, max_samples = max_samples, random_state = random_state)
	splits = make_cv_splits(data.y, n_splits = n_splits, random_state = random_state)

	text = cross_validated_predictions(
		"Text-only",
		build_classifier(classifier, random_state = random_state),
		data.text,
		data.y,
		splits,
		threshold = threshold,
	)
	audio = cross_validated_predictions(
		"Audio-only",
		build_classifier(classifier, random_state = random_state),
		data.audio,
		data.y,
		splits,
		threshold = threshold,
	)
	early = cross_validated_predictions(
		"Early fusion",
		build_classifier(classifier, random_state = random_state),
		data.fused,
		data.y,
		splits,
		threshold = threshold,
	)
	late = late_fusion(text, audio, data.y, threshold = threshold)

	predictions = {
		result.name: result
		for result in (text, audio, early, late)
	}
	metrics = pandas.DataFrame(
		{
			name: result.metrics
			for name, result in predictions.items()
		}
	).T
	confusions = {
		name: confusion_by_label(data.y, result.y_pred)
		for name, result in predictions.items()
	}
	clustering = evaluate_kmeans(data) if include_clustering else None

	return ExperimentResults(
		data = data,
		predictions = predictions,
		metrics = metrics,
		confusion_matrices = confusions,
		clustering = clustering,
	)


def plot_f1_comparison(metrics: pandas.DataFrame,
	ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.figure.Figure:
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


def main() -> None:
	parser = argparse.ArgumentParser(description = "Multi-label Part B experiments.")
	parser.add_argument("data", type = str, help = "Path to the cached data directory.")
	parser.add_argument("-k", type = int, default = 5, help = "Dataset top-k cache to load.")
	parser.add_argument("--labels", type = int, default = None, help = "Number of top labels to predict.")
	parser.add_argument("--folds", type = int, default = 10, help = "Number of cross-validation folds.")
	parser.add_argument("--classifier", choices = ["logistic", "random_forest"], default = "logistic")
	parser.add_argument("--threshold", type = float, default = 0.5)
	parser.add_argument("--max-samples", type = int, default = None, help = "Optional sample size for quick checks.")
	parser.add_argument("--skip-clustering", action = "store_true")
	parser.add_argument("--output", type = str, default = None, help = "Optional directory for CSV outputs.")

	args = parser.parse_args()

	results = run_experiments(
		args.data,
		k = args.k,
		label_count = args.labels,
		n_splits = args.folds,
		classifier = args.classifier,
		threshold = args.threshold,
		max_samples = args.max_samples,
		include_clustering = not args.skip_clustering,
	)

	print("Labels:", ", ".join(results.data.labels))
	print()
	print(results.metrics.round(4).to_string())

	if results.clustering is not None:
		print()
		print(results.clustering.round(4).to_string())

	if args.output:
		output = pathlib.Path(args.output)
		output.mkdir(parents = True, exist_ok = True)
		results.metrics.to_csv(output / "classification_metrics.csv")

		if results.clustering is not None:
			results.clustering.to_csv(output / "clustering_metrics.csv")


if __name__ == "__main__":
	main()
