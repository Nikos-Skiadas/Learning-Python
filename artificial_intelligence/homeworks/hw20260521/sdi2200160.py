"""Prompt Qwen-family LMs for response clarity classification."""


from __future__ import annotations


import argparse
import itertools
import pathlib
import typing

import pandas
import sklearn.model_selection

from ..evaluation import (
	classification_report_frame,
	confusion_matrix_frame,
	experiment_row,
	length_subgroup_scores,
	save_submission,
	summarize_experiments,
)
from ..generation import (
	DecodingConfig,
	HuggingFaceCausalLMGenerator,
	PromptedGenerationClassifier,
	StaticGenerator,
	seed_everything,
)
from ..parsing import GenerationParser
from ..prompting import (
	FewShotSampler,
	PromptBuilder,
	PromptConfig,
	PromptEncoder,
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
		use_json = True,
	),
	"few-shot": PromptConfig(
		name = "few-shot",
		reasoning = "none",
		use_json = True,
	),
	"cot": PromptConfig(
		name = "cot",
		reasoning = "step_by_step",
		use_json = True,
		include_example_rationales = False,
	),
	"self-check": PromptConfig(
		name = "self-check",
		reasoning = "self_check",
		use_json = True,
	),
}


def data_prep(
	validation_fraction: float = 0.2,
) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
	"""Load the shared CLARITY split and create a stratified validation split."""
	from ..data import data

	train = data["train"].to_pandas(); assert isinstance(train, pandas.DataFrame)
	test = data["test"].to_pandas(); assert isinstance(test, pandas.DataFrame)

	columns = ["question", "interview_answer", "clarity_label"]
	train = train[columns].fillna("")
	test = test[columns].fillna("")

	train_fit, validation = sklearn.model_selection.train_test_split(
		train,
		test_size = validation_fraction,
		random_state = RANDOM_STATE,
		stratify = train["clarity_label"],
	)

	return train_fit.reset_index(drop = True), validation.reset_index(drop = True), test.reset_index(drop = True)


def make_prompt_encoder(
	strategy: str,
	k_shots: int = 0,
	k_per_label: int | None = 1,
	fewshot_strategy: typing.Literal["balanced", "random", "length_matched"] = "balanced",
) -> PromptEncoder:
	"""Build the prompt encoder for one prompting strategy."""
	config = PROMPT_CONFIGS[strategy]
	sampler = None
	if strategy in {"few-shot", "cot", "self-check"} and k_shots:
		sampler = FewShotSampler(
			k = k_shots,
			k_per_label = k_per_label,
			strategy = fewshot_strategy,
			seed = RANDOM_STATE,
		)

	return PromptEncoder(
		builder = PromptBuilder(config = config),
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
	smoke_test: bool = False,
) -> PromptedGenerationClassifier:
	encoder = make_prompt_encoder(
		strategy = strategy,
		k_shots = k_shots,
		k_per_label = k_per_label,
		fewshot_strategy = fewshot_strategy,
	)
	generator = (
		StaticGenerator()
		if smoke_test
		else HuggingFaceCausalLMGenerator(
			model_name,
			decoding = decoding,
			batch_size = batch_size,
			system_message = "You are a careful annotator for political interview response clarity.",
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
	validation: pandas.DataFrame,
) -> tuple[dict[str, typing.Any], pandas.DataFrame]:
	"""Run one model/strategy combination on the validation split."""
	model_name = MODELS[model_key]
	decoding = DecodingConfig(
		max_new_tokens = args.max_new_tokens,
		do_sample = args.do_sample,
		temperature = args.temperature,
		top_p = args.top_p,
	)
	classifier = make_classifier(
		model_name = model_name,
		strategy = strategy,
		decoding = decoding,
		k_shots = args.k_shots,
		k_per_label = args.k_per_label,
		fewshot_strategy = args.fewshot_strategy,
		batch_size = args.batch_size,
		smoke_test = args.smoke_test,
	)
	classifier.fit(train, train["clarity_label"])

	source = validation.head(args.limit) if args.limit else validation
	run_frame = classifier.generate_frame(source)
	joined = source.join(run_frame)
	row = experiment_row(
		model = model_name,
		strategy = strategy,
		true = joined["clarity_label"],
		pred = joined["Predicted"],
		max_new_tokens = args.max_new_tokens,
		do_sample = args.do_sample,
		temperature = args.temperature,
		top_p = args.top_p,
		k_shots = args.k_shots,
		fewshot_strategy = args.fewshot_strategy,
	)

	return row, joined


def write_analysis_artifacts(
	run_frame: pandas.DataFrame,
	output_dir: pathlib.Path,
	model_key: str,
	strategy: str,
) -> None:
	prefix = output_dir / f"{model_key}_{strategy}"
	classification_report_frame(
		run_frame["clarity_label"],
		run_frame["Predicted"],
	).to_csv(f"{prefix}.classification_report.csv")
	confusion_matrix_frame(
		run_frame["clarity_label"],
		run_frame["Predicted"],
	).to_csv(f"{prefix}.confusion.csv")
	length_subgroup_scores(
		run_frame,
		true_col = "clarity_label",
		pred_col = "Predicted",
	).to_csv(f"{prefix}.length_subgroups.csv", index = False)
	run_frame.to_csv(f"{prefix}.generations.csv", index = True)


def run_final_submission(
	args: argparse.Namespace,
	train: pandas.DataFrame,
	test: pandas.DataFrame,
	output_dir: pathlib.Path,
) -> None:
	model_name = MODELS[args.final_model]
	decoding = DecodingConfig(
		max_new_tokens = args.max_new_tokens,
		do_sample = args.do_sample,
		temperature = args.temperature,
		top_p = args.top_p,
	)
	classifier = make_classifier(
		model_name = model_name,
		strategy = args.final_strategy,
		decoding = decoding,
		k_shots = args.k_shots,
		k_per_label = args.k_per_label,
		fewshot_strategy = args.fewshot_strategy,
		batch_size = args.batch_size,
		smoke_test = args.smoke_test,
	)
	classifier.fit(train, train["clarity_label"])

	source = test.head(args.limit) if args.limit else test
	predictions = classifier.predict(source)
	save_submission(
		predictions,
		output_dir / "submission best prompting system.csv",
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description = "Run Qwen-family prompting experiments for HW3.",
	)
	parser.add_argument("--models", nargs = "+", default = list(MODELS), choices = list(MODELS))
	parser.add_argument("--strategies", nargs = "+", default = list(PROMPT_CONFIGS), choices = list(PROMPT_CONFIGS))
	parser.add_argument("--output-dir", default = "runs_hw3")
	parser.add_argument("--limit", type = int, default = 0, help = "Optional row limit for quick checks.")
	parser.add_argument("--batch-size", type = int, default = 1)
	parser.add_argument("--k-shots", type = int, default = 3)
	parser.add_argument("--k-per-label", type = int, default = 1)
	parser.add_argument("--fewshot-strategy", default = "balanced", choices = ["balanced", "random", "length_matched"])
	parser.add_argument("--max-new-tokens", type = int, default = 48)
	parser.add_argument("--do-sample", action = "store_true")
	parser.add_argument("--temperature", type = float, default = 0.0)
	parser.add_argument("--top-p", type = float, default = 1.0)
	parser.add_argument("--smoke-test", action = "store_true", help = "Use a static generator instead of loading an LLM.")
	parser.add_argument("--make-submission", action = "store_true")
	parser.add_argument("--final-model", default = "qwen-4b", choices = list(MODELS))
	parser.add_argument("--final-strategy", default = "few-shot", choices = list(PROMPT_CONFIGS))

	return parser.parse_args()


def main() -> None:
	args = parse_args()
	seed_everything(RANDOM_STATE)
	output_dir = pathlib.Path(args.output_dir)
	output_dir.mkdir(parents = True, exist_ok = True)

	train, validation, test = data_prep()
	rows: list[dict[str, typing.Any]] = []
	for model_key, strategy in itertools.product(args.models, args.strategies):
		print("=" * 80)
		print(f"MODEL: {MODELS[model_key]}")
		print(f"STRATEGY: {strategy}")
		row, run_frame = run_experiment(model_key, strategy, args, train, validation)
		rows.append(row)
		write_analysis_artifacts(run_frame, output_dir, model_key, strategy)
		print(row)

	results = summarize_experiments(rows)
	results.to_csv(output_dir / "experiment_summary.csv", index = False)
	print(results.to_string(index = False))

	if args.make_submission:
		run_final_submission(args, train, test, output_dir)


if __name__ == "__main__":
	main()
