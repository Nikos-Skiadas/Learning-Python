"""Evaluation, comparison, and reporting helpers for clarity experiments."""


from __future__ import annotations


import pathlib
import typing

import numpy
import pandas
import sklearn.metrics

from .prompting import CLARITY_LABELS


MetricDict = dict[str, float]


def classification_scores(
	true: typing.Iterable[str],
	pred: typing.Iterable[str | None],
	/,
	labels: tuple[str, ...] = CLARITY_LABELS,
) -> MetricDict:
	"""Return report metrics required by the assignments."""
	y_true = pandas.Series(list(true), dtype = "object")
	y_pred = pandas.Series(list(pred), dtype = "object")
	valid = y_pred.isin(labels)
	y_eval = y_pred.where(valid, "__invalid__")

	return {
		"accuracy": float(sklearn.metrics.accuracy_score(y_true, y_eval)),
		"precision_macro": float(sklearn.metrics.precision_score(y_true, y_eval, labels = labels, average = "macro", zero_division = 0)),
		"recall_macro": float(sklearn.metrics.recall_score(y_true, y_eval, labels = labels, average = "macro", zero_division = 0)),
		"f1_macro": float(sklearn.metrics.f1_score(y_true, y_eval, labels = labels, average = "macro", zero_division = 0)),
		"f1_weighted": float(sklearn.metrics.f1_score(y_true, y_eval, labels = labels, average = "weighted", zero_division = 0)),
		"f1_micro": float(sklearn.metrics.f1_score(y_true, y_eval, labels = labels, average = "micro", zero_division = 0)),
		"invalid_rate": float((~valid).mean()) if len(valid) else 0.0,
	}


def classification_report_frame(
	true: typing.Iterable[str],
	pred: typing.Iterable[str | None],
	/,
	labels: tuple[str, ...] = CLARITY_LABELS,
) -> pandas.DataFrame:
	y_true = pandas.Series(list(true), dtype = "object")
	y_pred = pandas.Series(list(pred), dtype = "object")
	y_eval = y_pred.where(y_pred.isin(labels), "__invalid__")
	report = sklearn.metrics.classification_report(
		y_true,
		y_eval,
		labels = list(labels),
		output_dict = True,
		zero_division = 0,
	)

	return pandas.DataFrame(report).transpose()


def confusion_matrix_frame(
	true: typing.Iterable[str],
	pred: typing.Iterable[str | None],
	/,
	labels: tuple[str, ...] = CLARITY_LABELS,
	include_invalid: bool = True,
) -> pandas.DataFrame:
	y_true = pandas.Series(list(true), dtype = "object")
	y_pred = pandas.Series(list(pred), dtype = "object")
	matrix_labels = list(labels)
	if include_invalid and (~y_pred.isin(labels)).any():
		y_pred = y_pred.where(y_pred.isin(labels), "__invalid__")
		matrix_labels.append("__invalid__")

	matrix = sklearn.metrics.confusion_matrix(
		y_true,
		y_pred,
		labels = matrix_labels,
	)

	return pandas.DataFrame(matrix, index = matrix_labels, columns = matrix_labels)


def experiment_row(
	model: str,
	strategy: str,
	true: typing.Iterable[str],
	pred: typing.Iterable[str | None],
	**metadata: typing.Any,
) -> dict[str, typing.Any]:
	row: dict[str, typing.Any] = {
		"model": model,
		"strategy": strategy,
	}
	row.update(metadata)
	row.update(classification_scores(true, pred))

	return row


def summarize_experiments(rows: typing.Iterable[typing.Mapping[str, typing.Any]], /) -> pandas.DataFrame:
	frame = pandas.DataFrame(rows)  # type: ignore
	if "f1_macro" in frame:
		frame = frame.sort_values(
			by = ["f1_macro", "accuracy"],
			ascending = [False, False],
		)

	return frame.reset_index(drop = True)


def add_length_features(
	frame: pandas.DataFrame,
	/,
	question_key: str = "question",
	answer_key: str = "interview_answer",
	q: int = 3,
) -> pandas.DataFrame:
	"""Add word-count features and quantile bins for subgroup analysis."""
	result = frame.copy()
	result["question_words"] = result[question_key].fillna("").astype(str).str.split().str.len()
	result["answer_words"] = result[answer_key].fillna("").astype(str).str.split().str.len()
	result["combined_words"] = result["question_words"] + result["answer_words"]

	for key in ("question_words", "answer_words", "combined_words"):
		result[f"{key}_bin"] = _qcut_labels(result[key], q = q)

	return result


def subgroup_scores(
	frame: pandas.DataFrame,
	true_col: str,
	pred_col: str,
	group_col: str,
	/,
	labels: tuple[str, ...] = CLARITY_LABELS,
) -> pandas.DataFrame:
	rows: list[dict[str, typing.Any]] = []
	for group, group_frame in frame.groupby(group_col, dropna = False, observed = True):
		row: dict[str, typing.Any] = {
			"group": group,
			"group_col": group_col,
			"n": len(group_frame),
		}
		row.update(classification_scores(group_frame[true_col], group_frame[pred_col], labels = labels))
		rows.append(row)

	return pandas.DataFrame(rows).sort_values(["group_col", "group"]).reset_index(drop = True)


def length_subgroup_scores(
	frame: pandas.DataFrame,
	true_col: str,
	pred_col: str,
	question_key: str = "question",
	answer_key: str = "interview_answer",
	labels: tuple[str, ...] = CLARITY_LABELS,
	q: int = 3,
) -> pandas.DataFrame:
	length_frame = add_length_features(frame, question_key = question_key, answer_key = answer_key, q = q)
	parts = [
		subgroup_scores(length_frame, true_col, pred_col, f"{key}_bin", labels = labels)
		for key in ("question_words", "answer_words", "combined_words")
	]

	return pandas.concat(parts, ignore_index = True)


def error_cases_frame(
	frame: pandas.DataFrame,
	true_col: str,
	pred_col: str,
	/,
	question_key: str = "question",
	answer_key: str = "interview_answer",
	n: int | None = None,
	keep_correct: bool = False,
) -> pandas.DataFrame:
	"""Return concrete examples for qualitative error analysis."""
	result = add_length_features(frame, question_key = question_key, answer_key = answer_key)
	result["correct"] = result[true_col] == result[pred_col]
	if not keep_correct:
		result = result[~result["correct"]]

	columns = [
		question_key,
		answer_key,
		true_col,
		pred_col,
		"question_words",
		"answer_words",
		"combined_words",
		"correct",
	]
	available = [column for column in columns if column in result.columns]
	result = result[available]
	if n is not None:
		result = result.head(n)

	return result


def failure_overlap_frame(
	true: typing.Iterable[str],
	predictions: typing.Mapping[str, typing.Iterable[str | None]],
	/,
) -> pandas.DataFrame:
	"""Describe which systems fail on each instance."""
	true_series = pandas.Series(list(true), dtype = "object")
	frame = pandas.DataFrame(index = true_series.index)
	frame["true"] = true_series

	for name, pred in predictions.items():
		pred_series = pandas.Series(list(pred), dtype = "object")
		frame[f"{name}_pred"] = pred_series
		frame[f"{name}_wrong"] = pred_series != true_series

	wrong_cols = [column for column in frame.columns if column.endswith("_wrong")]
	frame["failure_count"] = frame[wrong_cols].sum(axis = 1)
	frame["failed_by"] = frame[wrong_cols].apply(
		lambda row: ", ".join(column.removesuffix("_wrong") for column, wrong in row.items() if wrong),
		axis = 1,
	)

	return frame


def failure_overlap_matrix(
	true: typing.Iterable[str],
	predictions: typing.Mapping[str, typing.Iterable[str | None]],
	/,
) -> pandas.DataFrame:
	"""Pairwise Jaccard overlap between systems' failure sets."""
	true_series = pandas.Series(list(true), dtype = "object")
	failure_sets: dict[str, set[int]] = {}
	for name, pred in predictions.items():
		pred_series = pandas.Series(list(pred), dtype = "object")
		failure_sets[name] = set(pred_series[pred_series != true_series].index)

	names = list(predictions)
	matrix = pandas.DataFrame(index = names, columns = names, dtype = float)
	for left in names:
		for right in names:
			union = failure_sets[left] | failure_sets[right]
			intersection = failure_sets[left] & failure_sets[right]
			matrix.loc[left, right] = len(intersection) / len(union) if union else 0.0

	return matrix


def save_submission(
	predictions: typing.Iterable[str | None],
	path: str | pathlib.Path,
	/,
	invalid_label: str = "Ambivalent",
) -> pandas.DataFrame:
	values = [
		prediction if prediction in CLARITY_LABELS else invalid_label
		for prediction in predictions
	]
	frame = pandas.DataFrame({"Predicted": values})
	frame.index.name = "Id"
	path = pathlib.Path(path)
	path.parent.mkdir(parents = True, exist_ok = True)
	frame.to_csv(path)

	return frame


def plot_metric_bars(
	results: pandas.DataFrame,
	path: str | pathlib.Path,
	/,
	metric: str = "f1_macro",
	hue: str = "strategy",
	x: str = "model",
) -> None:
	import matplotlib.pyplot as plt
	import seaborn

	figure, axis = plt.subplots(figsize = (10, 5))
	seaborn.barplot(data = results, x = x, y = metric, hue = hue, ax = axis)
	axis.set_ylim(0, 1)
	axis.set_title(metric.replace("_", " ").title())
	axis.set_ylabel(metric)
	axis.set_xlabel(x)
	figure.tight_layout()
	_path(path).parent.mkdir(parents = True, exist_ok = True)
	figure.savefig(path, dpi = 200)
	plt.close(figure)


def plot_confusion_matrix(
	matrix: pandas.DataFrame,
	path: str | pathlib.Path,
	/,
	title: str = "Confusion Matrix",
) -> None:
	import matplotlib.pyplot as plt
	import seaborn

	figure, axis = plt.subplots(figsize = (6, 5))
	seaborn.heatmap(matrix, annot = True, fmt = "d", cmap = "Blues", ax = axis)
	axis.set_title(title)
	axis.set_xlabel("Predicted")
	axis.set_ylabel("True")
	figure.tight_layout()
	_path(path).parent.mkdir(parents = True, exist_ok = True)
	figure.savefig(path, dpi = 200)
	plt.close(figure)


def _qcut_labels(values: pandas.Series, /, q: int) -> pandas.Series:
	try:
		binned = pandas.qcut(values, q = q, duplicates = "drop")
	except ValueError:
		return pandas.Series(["all"] * len(values), index = values.index, dtype = "object")

	return binned.astype(str)


def _path(path: str | pathlib.Path, /) -> pathlib.Path:
	return path if isinstance(path, pathlib.Path) else pathlib.Path(path)

