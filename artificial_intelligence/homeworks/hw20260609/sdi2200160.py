"""Run D3-Agentic Prompting experiments for response clarity classification."""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import pathlib
import typing

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas
import sklearn.model_selection

from ..agentic import (
	D3AgenticClassifier,
	D3PromptBuilder,
	D3PromptSettings,
	DirectAgenticComparator,
)
from ..evaluation import (
	classification_report_frame,
	confusion_matrix_frame,
	error_cases_frame,
	experiment_row,
	failure_overlap_frame,
	failure_overlap_matrix,
	length_subgroup_scores,
	plot_confusion_matrix,
	plot_metric_bars,
	save_submission,
	summarize_experiments,
)
from ..generation import (
	DecodingConfig,
	HuggingFaceGenerator,
	StaticGenerator,
	seed_everything,
)
from ..parsing import GenerationParser
from ..prompting import (
	CLARITY_LABELS,
	EvasionSpec,
	canonical_clarity_label,
	canonical_evasion_label,
	infer_evasion_specs,
)


RANDOM_STATE = 42
SYSTEM_MESSAGE = "You are a careful coordinator for response-clarity annotation agents."
SELECTION_METRIC = "f1_macro"
SELECTION_TIE_BREAKER = "accuracy"

MODELS: dict[str, str] = {
	"qwen-0.8b": "Qwen/Qwen3.5-0.8B",
	"qwen-2b": "Qwen/Qwen3.5-2B",
	"qwen-4b": "Qwen/Qwen3.5-4B",
}

EXPERIMENTS: dict[str, dict[str, typing.Any]] = {
	"d3-agentic": {
		"description": "Required four-agent D3 pipeline.",
		"calls_per_example": 4,
		"required_core": True,
	},
	"single-agent": {
		"description": "Single-prompt comparator with the same model and label guidance.",
		"calls_per_example": 1,
		"required_core": False,
	},
}

PREVIOUS_BASELINES: tuple[dict[str, typing.Any], ...] = (
	{
		"homework": "HW1",
		"system": "TF-IDF + Logistic Regression",
		"evaluation": "official test",
		"accuracy": 0.63,
		"f1_macro": 0.45,
		"note": "Best classical baseline; approximate values from HW1 report.",
	},
	{
		"homework": "HW1",
		"system": "GloVe Wiki + Logistic Regression",
		"evaluation": "official test",
		"accuracy": None,
		"f1_macro": 0.39,
		"note": "Mean-pooled embedding baseline from HW1.",
	},
	{
		"homework": "HW1",
		"system": "GloVe Twitter + Logistic Regression",
		"evaluation": "official test",
		"accuracy": None,
		"f1_macro": 0.38,
		"note": "Mean-pooled conversational embedding baseline from HW1.",
	},
	{
		"homework": "HW2",
		"system": "BERT-base fine-tuning",
		"evaluation": "official test",
		"accuracy": 0.6299,
		"f1_macro": 0.5561,
		"note": "Best encoder-only transformer from HW2.",
	},
	{
		"homework": "HW2",
		"system": "DistilBERT fine-tuning",
		"evaluation": "official test",
		"accuracy": 0.5779,
		"f1_macro": 0.5066,
		"note": "Smaller encoder-only transformer from HW2.",
	},
	{
		"homework": "HW2",
		"system": "DeBERTa-v3 fine-tuning",
		"evaluation": "official test",
		"accuracy": 0.6688,
		"f1_macro": 0.2672,
		"note": "Unstable run; collapsed to all Ambivalent predictions.",
	},
	{
		"homework": "HW3",
		"system": "Qwen 4B few-shot prompting",
		"evaluation": "balanced validation",
		"accuracy": 0.4667,
		"f1_macro": 0.4667,
		"note": "Best single-invocation prompting run by macro F1.",
	},
	{
		"homework": "HW3",
		"system": "Qwen 0.8B few-shot prompting",
		"evaluation": "balanced validation",
		"accuracy": 0.4667,
		"f1_macro": 0.4091,
		"note": "Same model scale as the required HW4 D3 system.",
	},
)

BASE_OPTIONS: dict[str, typing.Any] = {
	"model": "qwen-0.8b",
	"experiments": ["d3-agentic", "single-agent"],
	"output_dir": "runs_hw4",
	"train_csv": None,
	"test_csv": None,
	"synthetic_data": False,
	"limit": 0,
	"eval_size": 0,
	"eval_per_label": 0,
	"evaluation_split": "validation",
	"sample_seed": RANDOM_STATE,
	"validation_fraction": 0.2,
	"preview_prompts": 0,
	"batch_size": 1,
	"max_question_chars": 300,
	"max_answer_chars": 900,
	"max_intermediate_chars": 1200,
	"include_evasion_taxonomy": True,
	"max_new_tokens": 96,
	"decision_max_new_tokens": 48,
	"do_sample": False,
	"temperature": 0.0,
	"top_p": 1.0,
	"repetition_penalty": 1.0,
	"generator_backend": "auto",
	"device_map": "auto",
	"torch_dtype": "auto",
	"enable_thinking": False,
	"system_message": SYSTEM_MESSAGE,
	"use_chat_template": True,
	"save_prompts": True,
	"plots": True,
	"error_examples": 20,
	"smoke_test": False,
	"make_submission": False,
	"submission_source": "best",
	"submission_limit": 0,
	"final_experiment": "d3-agentic",
}

PRESETS: dict[str, dict[str, typing.Any]] = {
	"smoke": {
		"experiments": ["d3-agentic", "single-agent"],
		"output_dir": "runs_hw4_smoke",
		"synthetic_data": True,
		"validation_fraction": 0.5,
		"limit": 3,
		"smoke_test": True,
		"plots": False,
		"make_submission": True,
		"submission_limit": 3,
	},
	"check": {
		"experiments": ["d3-agentic"],
		"output_dir": "runs_hw4_check",
		"eval_per_label": 1,
		"make_submission": False,
		"plots": False,
	},
	"full": {
		"experiments": ["d3-agentic", "single-agent"],
		"output_dir": "runs_hw4_full",
		"eval_per_label": 10,
		"limit": 0,
		"make_submission": True,
	},
}


def run_id(model_key: str, experiment: str, /) -> str:
	return f"{model_key}_{experiment}"


def data_prep(
	validation_fraction: float = 0.2,
	train_csv: str | None = None,
	test_csv: str | None = None,
	synthetic_data: bool = False,
) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
	if synthetic_data:
		train, test = synthetic_clarity_data()
	elif train_csv is not None:
		train = pandas.read_csv(train_csv).fillna("")
		if test_csv is None:
			raise ValueError("--test-csv is required when --train-csv is used.")
		test = pandas.read_csv(test_csv).fillna("")
	else:
		from ..data import data

		train = data["train"].to_pandas(); assert isinstance(train, pandas.DataFrame)
		test = data["test"].to_pandas(); assert isinstance(test, pandas.DataFrame)

	train = normalize_clarity_frame(train, require_label = True)
	test = normalize_clarity_frame(test, require_label = False)

	train_fit, validation = sklearn.model_selection.train_test_split(
		train,
		test_size = validation_fraction,
		random_state = RANDOM_STATE,
		stratify = train["clarity_label"],
	)

	return train_fit.reset_index(drop = True), validation.reset_index(drop = True), test.reset_index(drop = True)


def normalize_clarity_frame(frame: pandas.DataFrame, /, require_label: bool) -> pandas.DataFrame:
	required = ["question", "interview_answer"]
	missing = [column for column in required if column not in frame]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")
	if require_label and "clarity_label" not in frame:
		raise ValueError("Training data must include clarity_label.")

	result = frame.copy()
	if "clarity_label" not in result:
		result["clarity_label"] = ""
	if "evasion_label" not in result:
		result["evasion_label"] = ""
	result["clarity_label"] = result["clarity_label"].apply(canonical_clarity_label)
	result["evasion_label"] = result["evasion_label"].apply(canonical_evasion_label)

	return result[["question", "interview_answer", "clarity_label", "evasion_label"]].fillna("")


def representative_sample(
	frame: pandas.DataFrame,
	size: int,
	/,
	label_key: str = "clarity_label",
	seed: int = RANDOM_STATE,
) -> pandas.DataFrame:
	if size <= 0 or size >= len(frame):
		return frame.reset_index(drop = True)

	labels = frame[label_key].fillna("").astype(str) if label_key in frame else pandas.Series(dtype = str)
	can_stratify = (
		label_key in frame
		and labels.ne("").all()
		and labels.nunique() > 1
		and size >= labels.nunique()
		and labels.value_counts().min() >= 2
	)
	if can_stratify:
		_, sample = sklearn.model_selection.train_test_split(
			frame,
			test_size = size,
			random_state = seed,
			stratify = labels,
		)
	else:
		sample = frame.sample(n = size, random_state = seed)

	return sample.sort_index().reset_index(drop = True)


def balanced_label_sample(
	frame: pandas.DataFrame,
	per_label: int,
	/,
	label_key: str = "clarity_label",
	seed: int = RANDOM_STATE,
	labels: tuple[str, ...] = CLARITY_LABELS,
) -> pandas.DataFrame:
	if per_label <= 0:
		return frame.reset_index(drop = True)
	if label_key not in frame:
		raise ValueError(f"Cannot sample per label because {label_key!r} is missing.")

	parts: list[pandas.DataFrame] = []
	for i, label in enumerate(labels):
		group = frame[frame[label_key] == label]
		if len(group) < per_label:
			raise ValueError(f"Cannot sample {per_label} examples for {label!r}; only {len(group)} available.")
		parts.append(group.sample(n = per_label, random_state = seed + i))

	return pandas.concat(parts, ignore_index = False).sort_index().reset_index(drop = True)


def choose_evaluation_frame(
	args: argparse.Namespace,
	validation: pandas.DataFrame,
	test: pandas.DataFrame,
) -> pandas.DataFrame:
	source = validation if args.evaluation_split == "validation" else test
	if args.evaluation_split == "test" and source["clarity_label"].fillna("").astype(str).eq("").all():
		raise ValueError("Cannot evaluate on test split because clarity_label is not available.")

	if args.eval_per_label:
		source = balanced_label_sample(source, args.eval_per_label, seed = args.sample_seed)
	else:
		source = representative_sample(source, args.eval_size, seed = args.sample_seed)
	if args.limit:
		source = source.head(args.limit).reset_index(drop = True)

	return source


def synthetic_clarity_data() -> tuple[pandas.DataFrame, pandas.DataFrame]:
	rows = [
		("Will you sign the bill?", "Yes, I will sign it tomorrow.", "Clear Reply", "Explicit"),
		("Did unemployment fall?", "The unemployment rate fell last quarter.", "Clear Reply", "Explicit"),
		("Why did prices rise?", "There are several complex factors we are studying.", "Ambivalent", "General"),
		("Will you support the amendment?", "I want to see the final language first.", "Ambivalent", "Implicit"),
		("Did you approve the plan?", "Let me talk instead about job creation.", "Clear Non-Reply", "Declining"),
		("Where did the funds go?", "The important issue is our future agenda.", "Clear Non-Reply", "Clarification"),
	]
	train = pandas.DataFrame(rows, columns = ["question", "interview_answer", "clarity_label", "evasion_label"])
	test = train.groupby("clarity_label", group_keys = False).head(1).reset_index(drop = True)

	return train, test


def decoding_from_args(args: argparse.Namespace, /, decision: bool = False) -> DecodingConfig:
	return DecodingConfig(
		max_new_tokens = args.decision_max_new_tokens if decision else args.max_new_tokens,
		do_sample = args.do_sample,
		temperature = args.temperature,
		top_p = args.top_p,
		repetition_penalty = args.repetition_penalty,
	)


def prompt_settings_from_args(
	args: argparse.Namespace,
	evasion_specs: tuple[EvasionSpec, ...],
) -> D3PromptBuilder:
	settings = D3PromptSettings(
		max_question_chars = _none_if_non_positive(args.max_question_chars),
		max_answer_chars = _none_if_non_positive(args.max_answer_chars),
		max_intermediate_chars = _none_if_non_positive(args.max_intermediate_chars),
		include_evasion_taxonomy = args.include_evasion_taxonomy,
	)
	return D3PromptBuilder(settings = settings, evasion_specs = evasion_specs)


def _none_if_non_positive(value: int | None, /) -> int | None:
	if value is None or value <= 0:
		return None

	return value


def make_generator(args: argparse.Namespace, /, decision: bool = False):
	if args.smoke_test:
		return StaticGenerator('{"label": "Ambivalent", "rationale": "static smoke-test output"}')

	return HuggingFaceGenerator(
		MODELS[args.model],
		decoding = decoding_from_args(args, decision = decision),
		batch_size = args.batch_size,
		device_map = args.device_map,
		torch_dtype = args.torch_dtype,
		use_chat_template = args.use_chat_template,
		backend = args.generator_backend,
		system_message = args.system_message,
		enable_thinking = args.enable_thinking,
	)


def make_classifier(
	experiment: str,
	args: argparse.Namespace,
	evasion_specs: tuple[EvasionSpec, ...],
):
	builder = prompt_settings_from_args(args, evasion_specs)
	if experiment == "d3-agentic":
		return D3AgenticClassifier(
			generator = make_generator(args),
			builder = builder,
			parser = GenerationParser(),
		)
	if experiment == "single-agent":
		return DirectAgenticComparator(
			generator = make_generator(args, decision = True),
			builder = builder,
			parser = GenerationParser(),
		)

	raise ValueError(f"Unknown experiment: {experiment}")


def run_experiment(
	experiment: str,
	args: argparse.Namespace,
	train: pandas.DataFrame,
	evaluation: pandas.DataFrame,
	evasion_specs: tuple[EvasionSpec, ...],
) -> tuple[dict[str, typing.Any], pandas.DataFrame]:
	classifier = make_classifier(experiment, args, evasion_specs)
	classifier.fit(train, train["clarity_label"])
	run_frame = classifier.generate_frame(evaluation, include_prompts = args.save_prompts)
	joined = evaluation.join(run_frame)
	row = experiment_row(
		model = MODELS[args.model],
		strategy = experiment,
		true = joined["clarity_label"],
		pred = joined["Predicted"],
		run_id = run_id(args.model, experiment),
		model_key = args.model,
		preset = args.preset,
		evaluation_split = args.evaluation_split,
		eval_size = len(evaluation),
		eval_per_label = args.eval_per_label,
		max_new_tokens = args.max_new_tokens,
		decision_max_new_tokens = args.decision_max_new_tokens,
		do_sample = args.do_sample,
		temperature = args.temperature,
		top_p = args.top_p,
		repetition_penalty = args.repetition_penalty,
		max_question_chars = args.max_question_chars,
		max_answer_chars = args.max_answer_chars,
		max_intermediate_chars = args.max_intermediate_chars,
		include_evasion_taxonomy = args.include_evasion_taxonomy,
		enable_thinking = args.enable_thinking,
		generator_backend = args.generator_backend,
		calls_per_example = EXPERIMENTS[experiment]["calls_per_example"],
		required_core = EXPERIMENTS[experiment]["required_core"],
	)
	update_run_diagnostics(row, joined)
	update_agent_diagnostics(row, joined)

	return row, joined


def _column_key(value: str, /) -> str:
	return value.casefold().replace("-", " ").replace("/", " ").replace(" ", "_")


def count_columns(prefix: str, values: pandas.Series, /) -> dict[str, int]:
	counts = values.fillna("__invalid__").astype(str).value_counts(dropna = False)
	result: dict[str, int] = {}
	for label in (*CLARITY_LABELS, "__invalid__"):
		result[f"{prefix}_{_column_key(label)}"] = int(counts.get(label, 0))

	return result


def update_run_diagnostics(row: dict[str, typing.Any], run_frame: pandas.DataFrame, /) -> None:
	true = run_frame["clarity_label"].fillna("").astype(str)
	pred = run_frame["Predicted"].fillna("__invalid__").astype(str)
	pred = pred.where(pred.isin(CLARITY_LABELS), "__invalid__")
	row.update(count_columns("truth", true))
	row.update(count_columns("pred", pred))

	majority_label = true.value_counts().idxmax()
	majority_accuracy = float((true == majority_label).mean())
	row.update({
		"majority_label": majority_label,
		"majority_accuracy": majority_accuracy,
		"accuracy_minus_majority": float(row.get("accuracy", 0.0) - majority_accuracy),
	})

	if "valid" in run_frame:
		valid = run_frame["valid"].fillna(False).astype(bool)
		row["valid_rate"] = float(valid.mean()) if len(valid) else 0.0
	if "parse_method" in run_frame:
		parse_counts = run_frame["parse_method"].fillna("__missing__").astype(str).value_counts()
		for method, count in parse_counts.items():
			row[f"parse_{_column_key(str(method))}"] = int(count)


def update_agent_diagnostics(row: dict[str, typing.Any], run_frame: pandas.DataFrame, /) -> None:
	for column in (
		"question_intent_generation",
		"answer_content_generation",
		"gap_evasion_generation",
		"decision_generation",
	):
		if column not in run_frame:
			continue
		lengths = run_frame[column].fillna("").astype(str).str.len()
		row[f"{column}_chars_mean"] = float(lengths.mean()) if len(lengths) else 0.0
		row[f"{column}_chars_max"] = int(lengths.max()) if len(lengths) else 0


def write_analysis_artifacts(
	run_frame: pandas.DataFrame,
	output_dir: pathlib.Path,
	model_key: str,
	experiment: str,
	error_examples: int,
	skip_plots: bool,
) -> dict[str, str]:
	files: dict[str, str] = {}
	run_name = run_id(model_key, experiment)
	prefix = output_dir / "runs" / run_name
	(output_dir / "runs").mkdir(parents = True, exist_ok = True)
	(output_dir / "plots").mkdir(parents = True, exist_ok = True)
	(output_dir / "tables").mkdir(parents = True, exist_ok = True)

	run_path = f"{prefix}.generations.csv"
	run_frame.to_csv(run_path, index = True)
	files["generations"] = run_path

	agent_cols = [
		column for column in (
			"question",
			"interview_answer",
			"clarity_label",
			"Predicted",
			"question_intent_generation",
			"answer_content_generation",
			"gap_evasion_generation",
			"decision_generation",
			"rationale",
		)
		if column in run_frame
	]
	agent_outputs_path = f"{prefix}.agent_outputs.csv"
	run_frame[agent_cols].to_csv(agent_outputs_path, index = True)
	files["agent_outputs"] = agent_outputs_path

	report_path = f"{prefix}.classification_report.csv"
	classification_report_frame(run_frame["clarity_label"], run_frame["Predicted"]).to_csv(report_path)
	files["classification_report"] = report_path

	matrix = confusion_matrix_frame(run_frame["clarity_label"], run_frame["Predicted"])
	matrix_path = f"{prefix}.confusion.csv"
	matrix.to_csv(matrix_path)
	files["confusion"] = matrix_path

	counts_path = f"{prefix}.prediction_counts.csv"
	run_frame["Predicted"].value_counts(dropna = False).rename_axis("Predicted").reset_index(name = "count").to_csv(counts_path, index = False)
	files["prediction_counts"] = counts_path

	if not skip_plots:
		plot_path = output_dir / "plots" / f"{run_name}.confusion.png"
		plot_confusion_matrix(matrix, plot_path, title = f"{model_key} {experiment}")
		files["confusion_plot"] = str(plot_path)

	subgroup_path = f"{prefix}.length_subgroups.csv"
	length_subgroup_scores(run_frame, true_col = "clarity_label", pred_col = "Predicted").to_csv(subgroup_path, index = False)
	files["length_subgroups"] = subgroup_path

	error_path = f"{prefix}.errors.csv"
	error_cases_frame(run_frame, "clarity_label", "Predicted", n = error_examples).to_csv(error_path, index = True)
	files["errors"] = error_path

	return files


def write_global_artifacts(
	results: pandas.DataFrame,
	run_frames: dict[str, pandas.DataFrame],
	output_dir: pathlib.Path,
	skip_plots: bool,
) -> dict[str, str]:
	files: dict[str, str] = {}
	(output_dir / "tables").mkdir(parents = True, exist_ok = True)
	(output_dir / "plots").mkdir(parents = True, exist_ok = True)

	summary_path = output_dir / "experiment_summary.csv"
	results.to_csv(summary_path, index = False)
	files["experiment_summary"] = str(summary_path)

	previous_path = output_dir / "previous_assignment_baselines.csv"
	previous = pandas.DataFrame(PREVIOUS_BASELINES)
	previous.to_csv(previous_path, index = False)
	files["previous_assignment_baselines"] = str(previous_path)

	comparison_path = output_dir / "baseline_comparison.csv"
	comparison = build_baseline_comparison(results, previous)
	comparison.to_csv(comparison_path, index = False)
	files["baseline_comparison"] = str(comparison_path)

	latex_path = output_dir / "tables" / "experiment_summary.tex"
	try:
		results.to_latex(latex_path, index = False, float_format = "%.4f")
	except ImportError as error:
		latex_path.write_text(f"% pandas.to_latex failed because an optional dependency is missing: {error}\n", encoding = "utf-8")
	files["experiment_summary_latex"] = str(latex_path)

	if not skip_plots and not results.empty:
		metric_path = output_dir / "plots" / "f1_macro_by_experiment.png"
		plot_metric_bars(results, metric_path, metric = "f1_macro", x = "strategy", hue = "model_key")
		files["f1_macro_plot"] = str(metric_path)

		invalid_path = output_dir / "plots" / "invalid_rate_by_experiment.png"
		plot_metric_bars(results, invalid_path, metric = "invalid_rate", x = "strategy", hue = "model_key")
		files["invalid_rate_plot"] = str(invalid_path)

	if run_frames:
		subgroups: list[pandas.DataFrame] = []
		errors: list[pandas.DataFrame] = []
		predictions: dict[str, pandas.Series] = {}
		first_frame = next(iter(run_frames.values()))
		true = first_frame["clarity_label"]

		for name, frame in run_frames.items():
			model_key, experiment = name.split("_", maxsplit = 1)
			subgroup = length_subgroup_scores(frame, true_col = "clarity_label", pred_col = "Predicted")
			subgroup.insert(0, "experiment", experiment)
			subgroup.insert(0, "model_key", model_key)
			subgroup.insert(0, "run_id", name)
			subgroups.append(subgroup)

			error_frame = error_cases_frame(frame, "clarity_label", "Predicted")
			error_frame.insert(0, "experiment", experiment)
			error_frame.insert(0, "model_key", model_key)
			error_frame.insert(0, "run_id", name)
			errors.append(error_frame)

			predictions[name] = frame["Predicted"]

		subgroups_path = output_dir / "length_subgroups_all.csv"
		pandas.concat(subgroups, ignore_index = True).to_csv(subgroups_path, index = False)
		files["length_subgroups_all"] = str(subgroups_path)

		errors_path = output_dir / "errors_all.csv"
		pandas.concat(errors, ignore_index = True).to_csv(errors_path, index = False)
		files["errors_all"] = str(errors_path)

		if len(predictions) > 1:
			overlap_path = output_dir / "failure_overlap.csv"
			failure_overlap_frame(true, predictions).to_csv(overlap_path, index = True)
			files["failure_overlap"] = str(overlap_path)

			overlap_matrix_path = output_dir / "failure_overlap_matrix.csv"
			failure_overlap_matrix(true, predictions).to_csv(overlap_matrix_path)
			files["failure_overlap_matrix"] = str(overlap_matrix_path)

	if not results.empty:
		best = json_ready(results.iloc[0].to_dict())
		best["selection_metric"] = SELECTION_METRIC
		best["selection_tie_breaker"] = SELECTION_TIE_BREAKER
		best_path = output_dir / "best_validation_system.json"
		best_path.write_text(json.dumps(best, indent = 2), encoding = "utf-8")
		files["best_validation_system"] = str(best_path)

	return files


def build_baseline_comparison(results: pandas.DataFrame, previous: pandas.DataFrame) -> pandas.DataFrame:
	current_rows = []
	for _, row in results.iterrows():
		current_rows.append({
			"homework": "HW4",
			"system": f"{row['model_key']} {row['strategy']}",
			"evaluation": "balanced validation" if row["evaluation_split"] == "validation" else row["evaluation_split"],
			"accuracy": row.get("accuracy"),
			"f1_macro": row.get("f1_macro"),
			"note": "Current D3-agentic assignment run.",
		})

	return pandas.concat([previous, pandas.DataFrame(current_rows)], ignore_index = True)


def run_final_submission(
	args: argparse.Namespace,
	train: pandas.DataFrame,
	test: pandas.DataFrame,
	output_dir: pathlib.Path,
	final_experiment: str,
	evasion_specs: tuple[EvasionSpec, ...],
) -> dict[str, str]:
	files: dict[str, str] = {}
	classifier = make_classifier(final_experiment, args, evasion_specs)
	classifier.fit(train, train["clarity_label"])

	source = test.head(args.submission_limit) if args.submission_limit else test
	submission_dir = output_dir / "submissions"
	submission_dir.mkdir(parents = True, exist_ok = True)

	run_frame = classifier.generate_frame(source, include_prompts = args.save_prompts)
	generations_path = submission_dir / "best_d3_agentic_system.generations.csv"
	run_frame.to_csv(generations_path, index = True)
	files["submission_generations"] = str(generations_path)

	submission_path = submission_dir / "submission_best_d3_agentic_system.csv"
	save_submission(run_frame["Predicted"], submission_path)
	files["submission"] = str(submission_path)

	system_path = submission_dir / "best_d3_agentic_system.json"
	system_path.write_text(
		json.dumps(
			json_ready({
				"model_key": args.model,
				"model": MODELS[args.model],
				"experiment": final_experiment,
				"selection": args.submission_source,
				"selection_metric": SELECTION_METRIC,
				"selection_tie_breaker": SELECTION_TIE_BREAKER,
				"decoding": decoding_from_args(args).asdict(),
				"decision_decoding": decoding_from_args(args, decision = True).asdict(),
				"max_question_chars": args.max_question_chars,
				"max_answer_chars": args.max_answer_chars,
				"max_intermediate_chars": args.max_intermediate_chars,
				"include_evasion_taxonomy": args.include_evasion_taxonomy,
				"enable_thinking": args.enable_thinking,
				"generator_backend": args.generator_backend,
				"preset": args.preset,
			}),
			indent = 2,
		),
		encoding = "utf-8",
	)
	files["submission_system"] = str(system_path)

	return files


def preview_prompts(
	args: argparse.Namespace,
	evaluation: pandas.DataFrame,
	output_dir: pathlib.Path,
	evasion_specs: tuple[EvasionSpec, ...],
) -> dict[str, str]:
	if args.preview_prompts <= 0:
		return {}

	files: dict[str, str] = {}
	preview_dir = output_dir / "prompts"
	preview_dir.mkdir(parents = True, exist_ok = True)
	builder = prompt_settings_from_args(args, evasion_specs)
	source = evaluation.head(args.preview_prompts)
	for index, record in source.iterrows():
		intent = builder.question_intent_prompt(record)
		content = builder.answer_content_prompt(record)
		gap = builder.gap_evasion_prompt(record, '{"asked_for": "..."}', '{"explicit_claims": "..."}')
		decision = builder.decision_prompt(record, '{"asked_for": "..."}', '{"explicit_claims": "..."}', '{"missing_requirements": "..."}')
		for agent_prompt in (intent, content, gap, decision):
			path = preview_dir / f"{index}.{agent_prompt.agent}.txt"
			path.write_text(agent_prompt.prompt, encoding = "utf-8")
			files[f"prompt_preview.{index}.{agent_prompt.agent}"] = str(path)

	return files


def write_manifest(
	args: argparse.Namespace,
	output_dir: pathlib.Path,
	train: pandas.DataFrame,
	validation: pandas.DataFrame,
	test: pandas.DataFrame,
	evaluation: pandas.DataFrame,
	evasion_specs: tuple[EvasionSpec, ...],
	files: dict[str, str],
) -> None:
	manifest = json_ready({
		"created_at": datetime.datetime.now(datetime.UTC).isoformat(),
		"random_state": RANDOM_STATE,
		"args": vars(args),
		"dataset": {
			"train_fit_examples": len(train),
			"validation_examples": len(validation),
			"test_examples": len(test),
			"evaluation_examples": len(evaluation),
			"validation_fraction": args.validation_fraction,
			"evaluation_split": args.evaluation_split,
			"eval_size": args.eval_size,
			"eval_per_label": args.eval_per_label,
			"sample_seed": args.sample_seed,
		},
		"models": MODELS,
		"experiments": EXPERIMENTS,
		"selection_metric": SELECTION_METRIC,
		"selection_tie_breaker": SELECTION_TIE_BREAKER,
		"previous_baselines": PREVIOUS_BASELINES,
		"evasion_taxonomy": [
			{
				"name": spec.name,
				"clarity_label": spec.clarity_label,
				"description": spec.description,
			}
			for spec in evasion_specs
		],
		"artifacts": files,
	})
	(output_dir / "manifest.json").write_text(json.dumps(manifest, indent = 2), encoding = "utf-8")


def json_ready(value: typing.Any, /) -> typing.Any:
	if isinstance(value, dict):
		return {str(key): json_ready(item) for key, item in value.items()}
	if isinstance(value, list | tuple):
		return [json_ready(item) for item in value]
	if hasattr(value, "item"):
		return value.item()

	return value


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description = "Run D3-Agentic Prompting experiments for HW4.",
		epilog = (
			"Examples:\n"
			"  %(prog)s --preset smoke\n"
			"  %(prog)s --preset check\n"
			"  %(prog)s --preset full --eval-per-label 10\n"
			"  %(prog)s --preset full --experiments d3-agentic --make-submission"
		),
		formatter_class = argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("--preset", choices = ["manual", *PRESETS], default = "manual")

	common = parser.add_argument_group("common")
	common.add_argument("--model", default = argparse.SUPPRESS, choices = list(MODELS))
	common.add_argument("--experiments", nargs = "+", default = argparse.SUPPRESS, choices = list(EXPERIMENTS))
	common.add_argument("--output-dir", default = argparse.SUPPRESS)
	common.add_argument("--limit", type = int, default = argparse.SUPPRESS)
	common.add_argument("--eval-size", type = int, default = argparse.SUPPRESS)
	common.add_argument("--eval-per-label", type = int, default = argparse.SUPPRESS)
	common.add_argument("--evaluation-split", default = argparse.SUPPRESS, choices = ["validation", "test"])
	common.add_argument("--sample-seed", type = int, default = argparse.SUPPRESS)

	data_group = parser.add_argument_group("data")
	data_group.add_argument("--train-csv", default = argparse.SUPPRESS)
	data_group.add_argument("--test-csv", default = argparse.SUPPRESS)
	data_group.add_argument("--synthetic-data", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	data_group.add_argument("--validation-fraction", type = float, default = argparse.SUPPRESS)
	data_group.add_argument("--preview-prompts", type = int, default = argparse.SUPPRESS)

	prompt_group = parser.add_argument_group("agent prompts and decoding")
	prompt_group.add_argument("--max-question-chars", type = int, default = argparse.SUPPRESS)
	prompt_group.add_argument("--max-answer-chars", type = int, default = argparse.SUPPRESS)
	prompt_group.add_argument("--max-intermediate-chars", type = int, default = argparse.SUPPRESS)
	prompt_group.add_argument("--evasion-taxonomy", dest = "include_evasion_taxonomy", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	prompt_group.add_argument("--max-new-tokens", type = int, default = argparse.SUPPRESS)
	prompt_group.add_argument("--decision-max-new-tokens", type = int, default = argparse.SUPPRESS)
	prompt_group.add_argument("--do-sample", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	prompt_group.add_argument("--temperature", type = float, default = argparse.SUPPRESS)
	prompt_group.add_argument("--top-p", type = float, default = argparse.SUPPRESS)
	prompt_group.add_argument("--repetition-penalty", type = float, default = argparse.SUPPRESS)
	prompt_group.add_argument("--system-message", default = argparse.SUPPRESS)

	runtime = parser.add_argument_group("runtime")
	runtime.add_argument("--batch-size", type = int, default = argparse.SUPPRESS)
	runtime.add_argument("--generator-backend", default = argparse.SUPPRESS, choices = ["auto", "causal-lm", "image-text-to-text"])
	runtime.add_argument("--device-map", default = argparse.SUPPRESS)
	runtime.add_argument("--torch-dtype", default = argparse.SUPPRESS)
	runtime.add_argument("--thinking", dest = "enable_thinking", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	runtime.add_argument("--chat-template", dest = "use_chat_template", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	runtime.add_argument("--save-prompts", dest = "save_prompts", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	runtime.add_argument("--plots", dest = "plots", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	runtime.add_argument("--skip-plots", dest = "plots", action = "store_false", default = argparse.SUPPRESS)
	runtime.add_argument("--smoke-test", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)

	artifacts = parser.add_argument_group("artifacts and submission")
	artifacts.add_argument("--error-examples", type = int, default = argparse.SUPPRESS)
	artifacts.add_argument("--make-submission", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	artifacts.add_argument("--submission-source", default = argparse.SUPPRESS, choices = ["best", "manual"])
	artifacts.add_argument("--submission-limit", type = int, default = argparse.SUPPRESS)
	artifacts.add_argument("--final-experiment", default = argparse.SUPPRESS, choices = list(EXPERIMENTS))

	return resolve_args(parser.parse_args())


def resolve_args(raw_args: argparse.Namespace, /) -> argparse.Namespace:
	preset = raw_args.preset
	values = dict(BASE_OPTIONS)
	values.update(PRESETS.get(preset, {}))

	overrides = vars(raw_args).copy()
	overrides.pop("preset", None)
	values.update(overrides)
	values["preset"] = preset

	return argparse.Namespace(**values)


def main() -> None:
	args = parse_args()
	seed_everything(RANDOM_STATE)
	output_dir = pathlib.Path(args.output_dir)
	output_dir.mkdir(parents = True, exist_ok = True)

	train, validation, test = data_prep(
		validation_fraction = args.validation_fraction,
		train_csv = args.train_csv,
		test_csv = args.test_csv,
		synthetic_data = args.synthetic_data,
	)
	evaluation = choose_evaluation_frame(args, validation, test)
	experiment_train = (
		pandas.concat([train, validation], ignore_index = True)
		if args.evaluation_split == "test"
		else train
	)
	taxonomy_source = pandas.concat([train, validation], ignore_index = True)
	evasion_specs = infer_evasion_specs(taxonomy_source)
	rows: list[dict[str, typing.Any]] = []
	run_frames: dict[str, pandas.DataFrame] = {}
	files: dict[str, str] = {}
	files.update(preview_prompts(args, evaluation, output_dir, evasion_specs))

	for experiment in args.experiments:
		print("=" * 80)
		print(f"MODEL: {MODELS[args.model]}")
		print(f"EXPERIMENT: {experiment}")
		row, run_frame = run_experiment(experiment, args, experiment_train, evaluation, evasion_specs)
		rows.append(row)
		name = run_id(args.model, experiment)
		run_frames[name] = run_frame
		files.update({
			f"{name}.{key}": value
			for key, value in write_analysis_artifacts(
				run_frame,
				output_dir,
				args.model,
				experiment,
				error_examples = args.error_examples,
				skip_plots = not args.plots,
			).items()
		})
		print(row)
		gc.collect()

	results = summarize_experiments(rows)
	if not results.empty:
		results = results.sort_values(by = [SELECTION_METRIC, SELECTION_TIE_BREAKER], ascending = [False, False]).reset_index(drop = True)
	files.update(write_global_artifacts(results, run_frames, output_dir, skip_plots = not args.plots))
	print(results.to_string(index = False))

	if args.make_submission:
		if args.submission_source == "best" and not results.empty:
			final_experiment = typing.cast(str, results.iloc[0]["strategy"])
		else:
			final_experiment = args.final_experiment

		full_train = pandas.concat([train, validation], ignore_index = True)
		files.update(run_final_submission(args, full_train, test, output_dir, final_experiment, evasion_specs))

	write_manifest(args, output_dir, train, validation, test, evaluation, evasion_specs, files)


if __name__ == "__main__":
	main()
