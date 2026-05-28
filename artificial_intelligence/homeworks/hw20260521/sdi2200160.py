"""Prompt Qwen-family LMs for response clarity classification."""


from __future__ import annotations


import argparse
import dataclasses
import datetime
import itertools
import json
import os
import pathlib
import typing

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas
import sklearn.model_selection

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
	PromptedGenerationClassifier,
	StaticGenerator,
	seed_everything,
)
from ..parsing import GenerationParser
from ..prompting import (
	CLARITY_LABELS,
	EvasionSpec,
	FewShotSampler,
	PromptBuilder,
	PromptConfig,
	PromptEncoder,
	canonical_clarity_label,
	canonical_evasion_label,
	infer_evasion_specs,
)


RANDOM_STATE = 42

MODELS: dict[str, str] = {
	"qwen-0.8b": "Qwen/Qwen3.5-0.8B",
	"qwen-2b": "Qwen/Qwen3.5-2B",
	"qwen-4b": "Qwen/Qwen3.5-4B",
}

PROMPT_CONFIGS: dict[str, PromptConfig] = {
	"zero-shot": PromptConfig(
		name = "zero-shot",
		reasoning = "none",
		use_json = False,
	),
	"few-shot": PromptConfig(
		name = "few-shot",
		reasoning = "none",
		use_json = False,
	),
	"cot": PromptConfig(
		name = "cot",
		reasoning = "step_by_step",
		use_json = False,
		include_example_rationales = False,
	),
	"self-check": PromptConfig(
		name = "self-check",
		reasoning = "self_check",
		use_json = False,
	),
}


SYSTEM_MESSAGE = "You are a careful annotator for political interview response clarity."

BASE_OPTIONS: dict[str, typing.Any] = {
	"models": list(MODELS),
	"strategies": list(PROMPT_CONFIGS),
	"output_dir": "runs_hw3",
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
	"k_shots": 3,
	"k_per_label": 1,
	"fewshot_strategy": "balanced",
	"max_question_chars": 500,
	"max_answer_chars": 2400,
	"max_context_chars": 500,
	"max_new_tokens": 48,
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
	"final_model": "qwen-4b",
	"final_strategy": "few-shot",
}

PRESETS: dict[str, dict[str, typing.Any]] = {
	"smoke": {
		"models": ["qwen-0.8b"],
		"strategies": ["zero-shot", "few-shot"],
		"output_dir": "runs_hw3_smoke",
		"synthetic_data": True,
		"limit": 3,
		"smoke_test": True,
		"plots": False,
		"make_submission": True,
		"submission_limit": 3,
	},
	"check": {
		"models": ["qwen-0.8b"],
		"strategies": ["zero-shot"],
		"output_dir": "runs_hw3_check",
		"limit": 5,
		"make_submission": False,
	},
	"full": {
		"models": list(MODELS),
		"strategies": list(PROMPT_CONFIGS),
		"output_dir": "runs_hw3_full",
		"eval_size": 90,
		"limit": 0,
		"make_submission": True,
	},
}


def run_id(model_key: str, strategy: str, /) -> str:
	return f"{model_key}_{strategy}"


def data_prep(
	validation_fraction: float = 0.2,
	train_csv: str | None = None,
	test_csv: str | None = None,
	synthetic_data: bool = False,
) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
	"""Load the shared CLARITY split and create a stratified validation split."""
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
	"""Return a deterministic stratified sample when labels are available."""
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
	"""Return exactly N examples for each clarity label."""
	if per_label <= 0:
		return frame.reset_index(drop = True)
	if label_key not in frame:
		raise ValueError(f"Cannot sample per label because {label_key!r} is missing.")

	parts: list[pandas.DataFrame] = []
	for i, label in enumerate(labels):
		group = frame[frame[label_key] == label]
		if len(group) < per_label:
			raise ValueError(
				f"Cannot sample {per_label} examples for {label!r}; only {len(group)} available."
			)
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


def preview_prompts(
	args: argparse.Namespace,
	train: pandas.DataFrame,
	evaluation: pandas.DataFrame,
	output_dir: pathlib.Path,
	evasion_specs: tuple[EvasionSpec, ...],
) -> dict[str, str]:
	if args.preview_prompts <= 0:
		return {}

	files: dict[str, str] = {}
	preview_dir = output_dir / "prompts"
	preview_dir.mkdir(parents = True, exist_ok = True)
	source = evaluation.head(args.preview_prompts)
	for strategy in args.strategies:
		encoder = make_prompt_encoder(
			strategy = strategy,
			k_shots = args.k_shots,
			k_per_label = args.k_per_label,
			fewshot_strategy = args.fewshot_strategy,
			max_question_chars = args.max_question_chars,
			max_answer_chars = args.max_answer_chars,
			max_context_chars = args.max_context_chars,
			evasion_specs = evasion_specs,
		)
		encoder.fit(train, train["clarity_label"])
		prompts = encoder.transform(source)
		for i, prompt in enumerate(prompts):
			path = preview_dir / f"{strategy}.{i}.txt"
			path.write_text(prompt, encoding = "utf-8")
			files[f"prompt_preview.{strategy}.{i}"] = str(path)

	return files


def synthetic_clarity_data() -> tuple[pandas.DataFrame, pandas.DataFrame]:
	"""Small local dataset for smoke-testing the artifact pipeline."""
	rows = [
		("Will you sign the bill?", "Yes, I will sign it tomorrow.", "Clear Reply", "Explicit"),
		("Did unemployment fall?", "The unemployment rate fell last quarter.", "Clear Reply", "Explicit"),
		("Are talks continuing?", "Yes, our teams are meeting this week.", "Clear Reply", "Explicit"),
		("Will taxes rise?", "No, this budget does not raise taxes.", "Clear Reply", "Explicit"),
		("Did you meet the minister?", "I met the minister on Monday.", "Clear Reply", "Explicit"),
		("Why did prices rise?", "There are several complex factors we are studying.", "Ambivalent", "General"),
		("Will you support the amendment?", "I want to see the final language first.", "Ambivalent", "Implicit"),
		("Was the policy a mistake?", "It is too early to draw a final conclusion.", "Ambivalent", "Dodging"),
		("Do you accept responsibility?", "Many people were involved in the process.", "Ambivalent", "Deflection"),
		("Is the agreement final?", "The broad direction is clear, but details remain.", "Ambivalent", "Partial"),
		("Did you approve the plan?", "Let me talk instead about job creation.", "Clear Non-Reply", "Declining"),
		("Where did the funds go?", "The important issue is our future agenda.", "Clear Non-Reply", "Clarification"),
		("Did you read the report?", "I reject the premise of that question.", "Clear Non-Reply", "Declining"),
		("When did you know?", "Families care about results, not political games.", "Clear Non-Reply", "Clarification"),
		("Was there a meeting?", "We should focus on what voters need next.", "Clear Non-Reply", "Ignorance"),
	]
	train = pandas.DataFrame(rows, columns = ["question", "interview_answer", "clarity_label", "evasion_label"])
	test = train.groupby("clarity_label", group_keys = False).head(1).reset_index(drop = True)

	return train, test


def decoding_from_args(args: argparse.Namespace, /) -> DecodingConfig:
	return DecodingConfig(
		max_new_tokens = args.max_new_tokens,
		do_sample = args.do_sample,
		temperature = args.temperature,
		top_p = args.top_p,
		repetition_penalty = args.repetition_penalty,
	)


def _none_if_non_positive(value: int | None, /) -> int | None:
	if value is None or value <= 0:
		return None

	return value


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


def make_prompt_encoder(
	strategy: str,
	k_shots: int = 0,
	k_per_label: int | None = 1,
	fewshot_strategy: typing.Literal["balanced", "random", "length_matched"] = "balanced",
	max_question_chars: int | None = None,
	max_answer_chars: int | None = None,
	max_context_chars: int | None = None,
	evasion_specs: tuple[EvasionSpec, ...] | None = None,
) -> PromptEncoder:
	"""Build the prompt encoder for one prompting strategy."""
	config = dataclasses.replace(
		PROMPT_CONFIGS[strategy],
		max_question_chars = _none_if_non_positive(max_question_chars),
		max_answer_chars = _none_if_non_positive(max_answer_chars),
		max_context_chars = _none_if_non_positive(max_context_chars),
	)
	sampler = None
	if strategy in {"few-shot", "cot", "self-check"} and k_shots:
		sampler = FewShotSampler(
			k = k_shots,
			k_per_label = k_per_label,
			strategy = fewshot_strategy,
			seed = RANDOM_STATE,
		)

	builder = (
		PromptBuilder(config = config)
		if evasion_specs is None
		else PromptBuilder(config = config, evasion_specs = evasion_specs)
	)

	return PromptEncoder(
		builder = builder,
		fewshot_sampler = sampler,
	)


def make_classifier(
	model_name: str,
	strategy: str,
	decoding: DecodingConfig,
	k_shots: int,
	k_per_label: int | None,
	fewshot_strategy: typing.Literal["balanced", "random", "length_matched"],
	batch_size: int,
	generator_backend: typing.Literal["auto", "causal-lm", "image-text-to-text"],
	device_map: str,
	torch_dtype: str,
	use_chat_template: bool,
	enable_thinking: bool | None,
	max_question_chars: int | None,
	max_answer_chars: int | None,
	max_context_chars: int | None,
	system_message: str,
	evasion_specs: tuple[EvasionSpec, ...],
	smoke_test: bool = False,
) -> PromptedGenerationClassifier:
	encoder = make_prompt_encoder(
		strategy = strategy,
		k_shots = k_shots,
		k_per_label = k_per_label,
		fewshot_strategy = fewshot_strategy,
		max_question_chars = max_question_chars,
		max_answer_chars = max_answer_chars,
		max_context_chars = max_context_chars,
		evasion_specs = evasion_specs,
	)
	generator = (
		StaticGenerator()
		if smoke_test
		else HuggingFaceGenerator(
			model_name,
			decoding = decoding,
			batch_size = batch_size,
			device_map = device_map,
			torch_dtype = torch_dtype,
			use_chat_template = use_chat_template,
			backend = generator_backend,
			system_message = system_message,
			enable_thinking = enable_thinking,
		)
	)

	return PromptedGenerationClassifier(
		prompt_encoder = encoder,
		generator = generator,
		parser = GenerationParser(),
	)


def run_experiment(
	model_key: str,
	strategy: str,
	args: argparse.Namespace,
	train: pandas.DataFrame,
	evaluation: pandas.DataFrame,
	evasion_specs: tuple[EvasionSpec, ...],
) -> tuple[dict[str, typing.Any], pandas.DataFrame]:
	"""Run one model/strategy combination on the validation split."""
	model_name = MODELS[model_key]
	decoding = decoding_from_args(args)
	classifier = make_classifier(
		model_name = model_name,
		strategy = strategy,
		decoding = decoding,
		k_shots = args.k_shots,
		k_per_label = args.k_per_label,
		fewshot_strategy = args.fewshot_strategy,
		batch_size = args.batch_size,
		generator_backend = args.generator_backend,
		device_map = args.device_map,
		torch_dtype = args.torch_dtype,
		use_chat_template = args.use_chat_template,
		enable_thinking = args.enable_thinking,
		max_question_chars = args.max_question_chars,
		max_answer_chars = args.max_answer_chars,
		max_context_chars = args.max_context_chars,
		system_message = args.system_message,
		evasion_specs = evasion_specs,
		smoke_test = args.smoke_test,
	)
	classifier.fit(train, train["clarity_label"])

	run_frame = classifier.generate_frame(evaluation, include_prompts = args.save_prompts)
	joined = evaluation.join(run_frame)
	row = experiment_row(
		model = model_name,
		strategy = strategy,
		true = joined["clarity_label"],
		pred = joined["Predicted"],
		run_id = run_id(model_key, strategy),
		model_key = model_key,
		preset = args.preset,
		evaluation_split = args.evaluation_split,
		eval_size = len(evaluation),
		eval_per_label = args.eval_per_label,
		max_new_tokens = args.max_new_tokens,
		do_sample = args.do_sample,
		temperature = args.temperature,
		top_p = args.top_p,
		repetition_penalty = args.repetition_penalty,
		k_shots = args.k_shots,
		k_per_label = args.k_per_label,
		fewshot_strategy = args.fewshot_strategy,
		max_question_chars = args.max_question_chars,
		max_answer_chars = args.max_answer_chars,
		max_context_chars = args.max_context_chars,
		enable_thinking = args.enable_thinking,
		generator_backend = args.generator_backend,
	)
	if "prompt" in joined:
		prompt_chars = joined["prompt"].fillna("").astype(str).str.len()
		row.update({
			"prompt_chars_mean": float(prompt_chars.mean()),
			"prompt_chars_max": int(prompt_chars.max()),
		})
	update_run_diagnostics(row, joined)

	return row, joined


def write_analysis_artifacts(
	run_frame: pandas.DataFrame,
	output_dir: pathlib.Path,
	model_key: str,
	strategy: str,
	error_examples: int,
	skip_plots: bool,
) -> dict[str, str]:
	files: dict[str, str] = {}
	run_name = run_id(model_key, strategy)
	prefix = output_dir / "runs" / run_name
	(output_dir / "runs").mkdir(parents = True, exist_ok = True)
	(output_dir / "plots").mkdir(parents = True, exist_ok = True)
	(output_dir / "tables").mkdir(parents = True, exist_ok = True)

	run_path = f"{prefix}.generations.csv"
	run_frame.to_csv(run_path, index = True)
	files["generations"] = run_path

	report_path = f"{prefix}.classification_report.csv"
	classification_report_frame(
		run_frame["clarity_label"],
		run_frame["Predicted"],
	).to_csv(report_path)
	files["classification_report"] = report_path

	matrix = confusion_matrix_frame(
		run_frame["clarity_label"],
		run_frame["Predicted"],
	)
	matrix_path = f"{prefix}.confusion.csv"
	matrix.to_csv(matrix_path)
	files["confusion"] = matrix_path

	counts_path = f"{prefix}.prediction_counts.csv"
	run_frame["Predicted"].value_counts(dropna = False).rename_axis("Predicted").reset_index(name = "count").to_csv(counts_path, index = False)
	files["prediction_counts"] = counts_path

	if not skip_plots:
		plot_path = output_dir / "plots" / f"{run_name}.confusion.png"
		plot_confusion_matrix(matrix, plot_path, title = f"{model_key} {strategy}")
		files["confusion_plot"] = str(plot_path)

	subgroup_path = f"{prefix}.length_subgroups.csv"
	length_subgroup_scores(
		run_frame,
		true_col = "clarity_label",
		pred_col = "Predicted",
	).to_csv(subgroup_path, index = False)
	files["length_subgroups"] = subgroup_path

	error_path = f"{prefix}.errors.csv"
	error_cases_frame(
		run_frame,
		"clarity_label",
		"Predicted",
		n = error_examples,
	).to_csv(error_path, index = True)
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

	latex_path = output_dir / "tables" / "experiment_summary.tex"
	try:
		results.to_latex(latex_path, index = False, float_format = "%.4f")
	except ImportError as error:
		latex_path.write_text(
			f"% pandas.to_latex failed because an optional dependency is missing: {error}\n",
			encoding = "utf-8",
		)
	files["experiment_summary_latex"] = str(latex_path)

	if not skip_plots and not results.empty:
		metric_path = output_dir / "plots" / "f1_macro_by_model_strategy.png"
		plot_metric_bars(results, metric_path, metric = "f1_macro", x = "model_key", hue = "strategy")
		files["f1_macro_plot"] = str(metric_path)

		invalid_path = output_dir / "plots" / "invalid_rate_by_model_strategy.png"
		plot_metric_bars(results, invalid_path, metric = "invalid_rate", x = "model_key", hue = "strategy")
		files["invalid_rate_plot"] = str(invalid_path)

	if run_frames:
		subgroups: list[pandas.DataFrame] = []
		errors: list[pandas.DataFrame] = []
		predictions: dict[str, pandas.Series] = {}
		first_frame = next(iter(run_frames.values()))
		true = first_frame["clarity_label"]

		for name, frame in run_frames.items():
			model_key, strategy = name.split("_", maxsplit = 1)
			subgroup = length_subgroup_scores(frame, true_col = "clarity_label", pred_col = "Predicted")
			subgroup.insert(0, "strategy", strategy)
			subgroup.insert(0, "model_key", model_key)
			subgroup.insert(0, "run_id", name)
			subgroups.append(subgroup)

			error_frame = error_cases_frame(frame, "clarity_label", "Predicted")
			error_frame.insert(0, "strategy", strategy)
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
		best_path = output_dir / "best_validation_system.json"
		best_path.write_text(json.dumps(best, indent = 2), encoding = "utf-8")
		files["best_validation_system"] = str(best_path)

	return files


def run_final_submission(
	args: argparse.Namespace,
	train: pandas.DataFrame,
	test: pandas.DataFrame,
	output_dir: pathlib.Path,
	final_model: str,
	final_strategy: str,
	evasion_specs: tuple[EvasionSpec, ...],
) -> dict[str, str]:
	files: dict[str, str] = {}
	model_name = MODELS[final_model]
	decoding = decoding_from_args(args)
	classifier = make_classifier(
		model_name = model_name,
		strategy = final_strategy,
		decoding = decoding,
		k_shots = args.k_shots,
		k_per_label = args.k_per_label,
		fewshot_strategy = args.fewshot_strategy,
		batch_size = args.batch_size,
		generator_backend = args.generator_backend,
		device_map = args.device_map,
		torch_dtype = args.torch_dtype,
		use_chat_template = args.use_chat_template,
		enable_thinking = args.enable_thinking,
		max_question_chars = args.max_question_chars,
		max_answer_chars = args.max_answer_chars,
		max_context_chars = args.max_context_chars,
		system_message = args.system_message,
		evasion_specs = evasion_specs,
		smoke_test = args.smoke_test,
	)
	classifier.fit(train, train["clarity_label"])

	source = test.head(args.submission_limit) if args.submission_limit else test
	submission_dir = output_dir / "submissions"
	submission_dir.mkdir(parents = True, exist_ok = True)

	run_frame = classifier.generate_frame(source, include_prompts = args.save_prompts)
	generations_path = submission_dir / "best_prompting_system.generations.csv"
	run_frame.to_csv(generations_path, index = True)
	files["submission_generations"] = str(generations_path)

	submission_path = submission_dir / "submission best prompting system.csv"
	save_submission(
		run_frame["Predicted"],
		submission_path,
	)
	files["submission"] = str(submission_path)

	system_path = submission_dir / "best_prompting_system.json"
	system_path.write_text(
		json.dumps(
			json_ready({
				"model_key": final_model,
				"model": model_name,
				"strategy": final_strategy,
				"selection": args.submission_source,
				"decoding": decoding.asdict(),
				"k_shots": args.k_shots,
				"k_per_label": args.k_per_label,
				"fewshot_strategy": args.fewshot_strategy,
				"max_question_chars": args.max_question_chars,
				"max_answer_chars": args.max_answer_chars,
				"max_context_chars": args.max_context_chars,
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
		"prompt_strategies": list(PROMPT_CONFIGS),
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
	(output_dir / "manifest.json").write_text(
		json.dumps(manifest, indent = 2),
		encoding = "utf-8",
	)


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
		description = "Run Qwen-family prompting experiments for HW3.",
		epilog = (
			"Examples:\n"
			"  %(prog)s --preset smoke\n"
			"  %(prog)s --preset check\n"
			"  %(prog)s --preset full --batch-size 1\n"
			"  %(prog)s --preset full --models qwen-0.8b qwen-2b --no-make-submission"
		),
		formatter_class = argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("--preset",
		choices = ["manual", *PRESETS],
		default = "manual",
		help = "Preset option bundle. Explicit flags override preset values.",
	)

	common = parser.add_argument_group("common")
	common.add_argument("--models", nargs = "+", default = argparse.SUPPRESS, choices = list(MODELS))
	common.add_argument("--strategies", nargs = "+", default = argparse.SUPPRESS, choices = list(PROMPT_CONFIGS))
	common.add_argument("--output-dir", default = argparse.SUPPRESS)
	common.add_argument("--limit", type = int, default = argparse.SUPPRESS, help = "Optional row limit for quick checks.")
	common.add_argument("--eval-size", type = int, default = argparse.SUPPRESS, help = "Representative evaluation sample size. Use 0 for the whole evaluation split.")
	common.add_argument("--eval-per-label", type = int, default = argparse.SUPPRESS, help = "Evaluate on exactly N examples from each clarity label. Overrides --eval-size when positive.")
	common.add_argument("--evaluation-split", default = argparse.SUPPRESS, choices = ["validation", "test"])
	common.add_argument("--sample-seed", type = int, default = argparse.SUPPRESS)

	data_group = parser.add_argument_group("data")
	data_group.add_argument("--train-csv", default = argparse.SUPPRESS)
	data_group.add_argument("--test-csv", default = argparse.SUPPRESS)
	data_group.add_argument("--synthetic-data", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS, help = "Use a tiny built-in dataset for local artifact smoke tests.")
	data_group.add_argument("--validation-fraction", type = float, default = argparse.SUPPRESS)
	data_group.add_argument("--preview-prompts", type = int, default = argparse.SUPPRESS, help = "Write N prompt previews per strategy before running.")

	prompt_group = parser.add_argument_group("prompting and decoding")
	prompt_group.add_argument("--k-shots", type = int, default = argparse.SUPPRESS)
	prompt_group.add_argument("--k-per-label", type = int, default = argparse.SUPPRESS)
	prompt_group.add_argument("--fewshot-strategy", default = argparse.SUPPRESS, choices = ["balanced", "random", "length_matched"])
	prompt_group.add_argument("--max-question-chars", type = int, default = argparse.SUPPRESS, help = "Truncate questions in prompts. Use 0 to disable truncation.")
	prompt_group.add_argument("--max-answer-chars", type = int, default = argparse.SUPPRESS, help = "Truncate answers in prompts using a head/tail excerpt. Use 0 to disable truncation.")
	prompt_group.add_argument("--max-context-chars", type = int, default = argparse.SUPPRESS, help = "Truncate auxiliary context fields. Use 0 to disable truncation.")
	prompt_group.add_argument("--max-new-tokens", type = int, default = argparse.SUPPRESS)
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
	runtime.add_argument("--thinking", dest = "enable_thinking", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS, help = "Enable Qwen thinking-mode chat templates when supported.")
	runtime.add_argument("--chat-template", dest = "use_chat_template", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	runtime.add_argument("--save-prompts", dest = "save_prompts", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	runtime.add_argument("--plots", dest = "plots", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	runtime.add_argument("--skip-plots", dest = "plots", action = "store_false", default = argparse.SUPPRESS, help = "Alias for --no-plots.")
	runtime.add_argument("--smoke-test", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS, help = "Use a static generator instead of loading an LLM.")

	artifacts = parser.add_argument_group("artifacts and submission")
	artifacts.add_argument("--error-examples", type = int, default = argparse.SUPPRESS)
	artifacts.add_argument("--make-submission", action = argparse.BooleanOptionalAction, default = argparse.SUPPRESS)
	artifacts.add_argument("--submission-source", default = argparse.SUPPRESS, choices = ["best", "manual"])
	artifacts.add_argument("--submission-limit", type = int, default = argparse.SUPPRESS)
	artifacts.add_argument("--final-model", default = argparse.SUPPRESS, choices = list(MODELS))
	artifacts.add_argument("--final-strategy", default = argparse.SUPPRESS, choices = list(PROMPT_CONFIGS))

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
	files.update(preview_prompts(args, experiment_train, evaluation, output_dir, evasion_specs))
	for model_key, strategy in itertools.product(args.models, args.strategies):
		print("=" * 80)
		print(f"MODEL: {MODELS[model_key]}")
		print(f"STRATEGY: {strategy}")
		row, run_frame = run_experiment(model_key, strategy, args, experiment_train, evaluation, evasion_specs)
		rows.append(row)
		name = run_id(model_key, strategy)
		run_frames[name] = run_frame
		files.update({
			f"{name}.{key}": value
			for key, value in write_analysis_artifacts(
				run_frame,
				output_dir,
				model_key,
				strategy,
				error_examples = args.error_examples,
				skip_plots = not args.plots,
			).items()
		})
		print(row)

	results = summarize_experiments(rows)
	files.update(write_global_artifacts(results, run_frames, output_dir, skip_plots = not args.plots))
	print(results.to_string(index = False))

	if args.make_submission:
		if args.submission_source == "best" and not results.empty:
			best = results.iloc[0]
			final_model = typing.cast(str, best["model_key"])
			final_strategy = typing.cast(str, best["strategy"])
		else:
			final_model = args.final_model
			final_strategy = args.final_strategy

		full_train = pandas.concat([train, validation], ignore_index = True)
		files.update(run_final_submission(args, full_train, test, output_dir, final_model, final_strategy, evasion_specs))

	write_manifest(args, output_dir, train, validation, test, evaluation, evasion_specs, files)


if __name__ == "__main__":
	main()
