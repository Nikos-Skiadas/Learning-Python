"""Fine-tune transformer models for response clarity classification."""


from __future__ import annotations


import argparse
import functools
import itertools
import os
import random
import typing

# Required for DeBERTa v3 tokenizer (SentencePiece -> protobuf)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas
import sklearn.base
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.utils.data
import transformers

from ..pipelines import Classifier
from ..preprocessing import IdentityPreprocessor
from ..protocols import Scorer


RANDOM_STATE = 42
DEVICE = torch.device("cuda")

MODELS: dict[str, str] = {
	"bert": "bert-base-uncased",
	"distilbert": "distilbert-base-uncased",
	"deberta": "microsoft/deberta-v3-base",
}

DEFAULT_CONFIG: dict[str, int | float] = {
	"batch_size": 16,
	"learning_rate": 2e-5,
	"weight_decay": 0.01,
	"num_epochs": 4,
	"max_length": 256,
	"max_grad_norm": 1.0,
	"val_frac": 0.15,
}

SWEEP_GRID: dict[str, tuple[int, ...]] = {
	"num_epochs": (3, 4),
	"max_length": (128, 256, 384),
}


def require_cuda() -> torch.device:
	"""Return the CUDA device or fail fast with a clear message."""
	if not torch.cuda.is_available():
		raise RuntimeError(
			"HW2 training requires a CUDA GPU. "
			"Run this script in your local GPU environment or on Kaggle with GPU enabled."
		)

	return DEVICE


def seed_everything(seed: int = RANDOM_STATE) -> None:
	"""Set all random seeds for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


class ClarityDataset(torch.utils.data.Dataset):
	"""PyTorch dataset wrapping tokenized question-answer pairs."""

	def __init__(self,
		encodings: dict[str, torch.Tensor],
		labels: torch.Tensor | None = None,
	) -> None:
		self.encodings = encodings
		self.labels = labels

	def __len__(self) -> int:
		return len(self.encodings["input_ids"])

	def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
		item = {key: value[idx] for key, value in self.encodings.items()}
		if self.labels is not None:
			item["labels"] = self.labels[idx]

		return item


class TokenizerEncoder(
	sklearn.base.BaseEstimator,
	sklearn.base.TransformerMixin,
):
	"""Wrap AutoTokenizer as an Encoder[pandas.DataFrame, BatchEncoding]."""

	def __init__(self,
		model_name: str = "bert-base-uncased",
		max_length: int = 256,
	) -> None:
		self.model_name = model_name
		self.max_length = max_length

	def fit(self, source: pandas.DataFrame, signal = None) -> typing.Self:
		return self

	def transform(self, source: pandas.DataFrame) -> dict[str, torch.Tensor]:
		if not hasattr(self, "_tokenizer"):
			self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

		encoding: transformers.BatchEncoding = self._tokenizer(
			source["question"].tolist(),
			source["answer"].tolist(),
			padding = True,
			truncation = True,
			max_length = self.max_length,
			return_tensors = "pt",
		)

		return typing.cast(dict[str, torch.Tensor], dict(encoding))


class TransformerModel(sklearn.base.BaseEstimator):
	"""Wrap transformer fine-tuning in the shared Classifier pipeline."""

	def __init__(self,
		model_name: str = "bert-base-uncased",
		num_labels: int = 3,
		batch_size: int = 16,
		learning_rate: float = 2e-5,
		weight_decay: float = 0.01,
		num_epochs: int = 4,
		max_grad_norm: float = 1.0,
		class_weights: torch.Tensor | None = None,
		val_frac: float = 0.15,
		device: torch.device = DEVICE,
	) -> None:
		self.model_name = model_name
		self.num_labels = num_labels
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.num_epochs = num_epochs
		self.max_grad_norm = max_grad_norm
		self.class_weights = class_weights
		self.val_frac = val_frac
		self.device = device

		self._model: transformers.PreTrainedModel | None = None
		self.history: dict[str, list[float]] = {}
		self.best_val_f1_: float = 0.0
		self.best_epoch_: int = 0

	def _build(self) -> None:
		require_cuda()

		model = transformers.AutoModelForSequenceClassification.from_pretrained(
			self.model_name,
			num_labels = self.num_labels,
		)
		assert isinstance(model, transformers.PreTrainedModel)
		self._model = model.to(self.device)  # type: ignore

	def _make_loss_fn(self) -> torch.nn.CrossEntropyLoss:
		weights = None
		if self.class_weights is not None:
			assert self._model is not None
			model_dtype = next(self._model.parameters()).dtype
			weights = self.class_weights.to(device = self.device, dtype = model_dtype)

		return torch.nn.CrossEntropyLoss(weight = weights)

	def _split_train_validation(self,
		source: dict[str, torch.Tensor],
		labels: torch.Tensor,
	) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
		indices = np.arange(len(labels))
		train_idx, val_idx = sklearn.model_selection.train_test_split(
			indices,
			test_size = self.val_frac,
			random_state = RANDOM_STATE,
			stratify = labels.cpu().numpy(),
		)

		train_enc = {key: value[train_idx] for key, value in source.items()}
		val_enc = {key: value[val_idx] for key, value in source.items()}
		train_labels = labels[train_idx]
		val_labels = labels[val_idx]

		return train_enc, train_labels, val_enc, val_labels

	def fit(self,
		source: dict[str, torch.Tensor],
		target: np.ndarray | torch.Tensor, /,
	) -> typing.Self:
		if self._model is None:
			self._build()

		assert self._model is not None

		labels = target if isinstance(target, torch.Tensor) else torch.tensor(target, dtype = torch.long)
		train_enc, train_labels, val_enc, val_labels = self._split_train_validation(source, labels)

		train_loader = torch.utils.data.DataLoader(
			ClarityDataset(train_enc, train_labels),
			batch_size = self.batch_size,
			shuffle = True,
			pin_memory = True,
		)

		optimizer = torch.optim.AdamW(
			self._model.parameters(),
			lr = self.learning_rate,
			weight_decay = self.weight_decay,
		)
		loss_fn = self._make_loss_fn()

		self.history = {"train_loss": [], "val_loss": [], "val_f1": []}
		self.best_val_f1_ = 0.0
		self.best_epoch_ = 0
		best_state: dict[str, torch.Tensor] | None = None

		for epoch in range(self.num_epochs):
			self._model.train()
			total_loss = 0.0

			for batch in train_loader:
				optimizer.zero_grad()
				model_inputs = {
					key: value.to(self.device, non_blocking = True)
					for key, value in batch.items()
					if key != "labels"
				}
				labels_batch = batch["labels"].to(self.device, non_blocking = True)

				outputs = self._model(**model_inputs)
				loss = loss_fn(outputs.logits, labels_batch)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.max_grad_norm)
				optimizer.step()

				total_loss += float(loss.item())

			avg_train_loss = total_loss / len(train_loader)
			val_metrics = self._evaluate(val_enc, val_labels)

			self.history["train_loss"].append(avg_train_loss)
			self.history["val_loss"].append(val_metrics["loss"])
			self.history["val_f1"].append(val_metrics["f1"])

			improved = val_metrics["f1"] > self.best_val_f1_
			if improved:
				self.best_val_f1_ = val_metrics["f1"]
				self.best_epoch_ = epoch + 1
				best_state = {key: value.detach().cpu().clone() for key, value in self._model.state_dict().items()}

			print(
				f"Epoch {epoch + 1}/{self.num_epochs} | "
				f"Train Loss: {avg_train_loss:.4f} | "
				f"Val Loss: {val_metrics['loss']:.4f} | "
				f"Val F1: {val_metrics['f1']:.4f}"
				f"{'  *' if improved else ''}"
			)

		if best_state is not None:
			self._model.load_state_dict(best_state)
			self._model.to(self.device)  # type: ignore
			print(f"\nRestored best checkpoint from epoch {self.best_epoch_} (Val F1: {self.best_val_f1_:.4f})")

		return self

	@torch.no_grad()
	def _evaluate(self,
		encodings: dict[str, torch.Tensor],
		labels: torch.Tensor,
	) -> dict[str, float]:
		assert self._model is not None
		self._model.eval()

		loader = torch.utils.data.DataLoader(
			ClarityDataset(encodings, labels),
			batch_size = self.batch_size,
			shuffle = False,
			pin_memory = True,
		)
		loss_fn = self._make_loss_fn()
		total_loss = 0.0
		predictions: list[int] = []

		for batch in loader:
			model_inputs = {
				key: value.to(self.device, non_blocking = True)
				for key, value in batch.items()
				if key != "labels"
			}
			labels_batch = batch["labels"].to(self.device, non_blocking = True)

			outputs = self._model(**model_inputs)
			total_loss += float(loss_fn(outputs.logits, labels_batch).item())
			predictions.extend(outputs.logits.argmax(dim = -1).cpu().tolist())

		true = labels.cpu().numpy()
		pred = np.array(predictions)

		return {
			"loss": total_loss / len(loader),
			"f1": float(sklearn.metrics.f1_score(true, pred, average = "macro", zero_division = 0)),
		}

	@torch.no_grad()
	def predict(self,
		source: dict[str, torch.Tensor], /,
	) -> np.ndarray:
		assert self._model is not None
		self._model.eval()

		loader = torch.utils.data.DataLoader(
			ClarityDataset(source),
			batch_size = self.batch_size,
			shuffle = False,
			pin_memory = True,
		)
		predictions: list[int] = []

		for batch in loader:
			model_inputs = {
				key: value.to(self.device, non_blocking = True)
				for key, value in batch.items()
			}
			outputs = self._model(**model_inputs)
			predictions.extend(outputs.logits.argmax(dim = -1).cpu().tolist())

		return np.array(predictions)


def macro_averaged(
	metric_fn: Scorer,
	**kwargs,
) -> Scorer:
	"""Wrap an sklearn metric with macro averaging and zero_division=0."""
	return functools.partial(
		metric_fn,
		average = "macro",
		zero_division = 0,
		**kwargs,
	)


def data_prep() -> tuple[
	pandas.DataFrame,
	pandas.Series,
	pandas.DataFrame,
	pandas.Series,
]:
	"""Prepare question-answer pairs for transformer sentence-pair encoding."""
	from ..data import data

	train_df = data["train"].to_pandas(); assert isinstance(train_df, pandas.DataFrame)
	test_df = data["test"].to_pandas(); assert isinstance(test_df, pandas.DataFrame)

	X_train = pandas.DataFrame({
		"question": train_df.question.fillna("").str.strip(),
		"answer": train_df.interview_answer.fillna("").str.strip(),
	})
	y_train = train_df.clarity_label.fillna("").str.strip()

	X_test = pandas.DataFrame({
		"question": test_df.question.fillna("").str.strip(),
		"answer": test_df.interview_answer.fillna("").str.strip(),
	})
	y_test = test_df.clarity_label.fillna("").str.strip()

	return X_train, y_train, X_test, y_test


def compute_class_weights(
	labels: torch.Tensor,
	num_classes: int = 3,
) -> torch.Tensor:
	"""Compute balanced class weights using sklearn's formula."""
	counts = torch.bincount(labels, minlength = num_classes).float()

	return len(labels) / (num_classes * counts)


def make_label_encoder(y_train: pandas.Series) -> tuple[sklearn.preprocessing.LabelEncoder, torch.Tensor]:
	"""Fit the label encoder and compute training class weights."""
	label_encoder = sklearn.preprocessing.LabelEncoder()
	label_encoder.fit(y_train)

	encoded = torch.tensor(label_encoder.transform(y_train), dtype = torch.long)
	class_weights = compute_class_weights(encoded, num_classes = len(label_encoder.classes_))

	return label_encoder, class_weights


def make_classifier(
	model_key: str,
	label_encoder: sklearn.preprocessing.LabelEncoder,
	class_weights: torch.Tensor,
	**config: int | float,
) -> Classifier:
	"""Create the shared pipeline with transformer-specific components."""
	model_name = MODELS[model_key]
	max_length = typing.cast(int, config["max_length"])

	classifier = Classifier(
		preprocessor = IdentityPreprocessor(),
		model = TransformerModel(
			model_name = model_name,
			num_labels = 3,
			batch_size = typing.cast(int, config["batch_size"]),
			learning_rate = typing.cast(float, config["learning_rate"]),
			weight_decay = typing.cast(float, config["weight_decay"]),
			num_epochs = typing.cast(int, config["num_epochs"]),
			max_grad_norm = typing.cast(float, config["max_grad_norm"]),
			class_weights = class_weights,
			val_frac = typing.cast(float, config["val_frac"]),
			device = require_cuda(),
		),
		source_encoder = TokenizerEncoder(
			model_name = model_name,
			max_length = max_length,
		),
		target_bicoder = label_encoder,  # type: ignore[arg-type]
	)

	classifier.compile(
		accuracy = sklearn.metrics.accuracy_score,
		precision = macro_averaged(sklearn.metrics.precision_score),
		recall = macro_averaged(sklearn.metrics.recall_score),
		f1 = macro_averaged(sklearn.metrics.f1_score),
	)

	return classifier


def submission_filename(model_name: str) -> str:
	"""Return the Kaggle submission filename required by HW2."""
	return f"submission {model_name.replace('/', '-')}.csv"


def config_with_overrides(**overrides: int | float) -> dict[str, int | float]:
	"""Merge runtime overrides into the shared default configuration."""
	config = dict(DEFAULT_CONFIG)
	config.update(overrides)

	return config


def describe_run(
	model_name: str,
	config: dict[str, int | float],
	class_weights: torch.Tensor,
	label_encoder: sklearn.preprocessing.LabelEncoder,
) -> None:
	"""Print the fixed training recipe before running a model."""
	print("=" * 80)
	print(f"{model_name.upper()} FINE-TUNING")
	print("=" * 80)
	print(f"Device:        {require_cuda()}")
	print(f"Max length:    {config['max_length']}")
	print(f"Batch size:    {config['batch_size']}")
	print(f"Learning rate: {config['learning_rate']}")
	print(f"Weight decay:  {config['weight_decay']}")
	print(f"Epochs:        {config['num_epochs']}")
	print(f"Val fraction:  {config['val_frac']}")
	print(f"Labels:        {list(label_encoder.classes_)}")
	print(f"Class weights: {dict(zip(label_encoder.classes_, class_weights.tolist()))}")
	print()


def train_and_evaluate(
	model_key: str,
	**overrides: int | float,
) -> tuple[Classifier, dict[str, float], pandas.Series, dict[str, int | float]]:
	"""Train one required transformer model and evaluate it on the official test split."""
	seed_everything(RANDOM_STATE)

	model_name = MODELS[model_key]
	config = config_with_overrides(**overrides)
	X_train, y_train, X_test, y_test = data_prep()
	label_encoder, class_weights = make_label_encoder(y_train)

	describe_run(model_name, config, class_weights, label_encoder)

	classifier = make_classifier(model_key, label_encoder, class_weights, **config)
	classifier.fit(X_train, y_train)

	print("\n" + "=" * 80)
	print("TEST SET EVALUATION")
	print("=" * 80)

	test_scores = {
		name: float(score)
		for name, score in classifier.score(X_test, y_test).items()
	}
	for name, score in test_scores.items():
		print(f"{name:12s} {score:.4f}")

	predictions = pandas.Series(classifier.predict(X_test), name = "Predicted")
	predictions.index.name = "Id"
	output_path = submission_filename(model_name)
	predictions.to_csv(output_path)

	print()
	print(f"Saved submission to {output_path}")
	print(predictions.value_counts())
	print()

	return classifier, test_scores, predictions, config


def run_tiny_sweep(model_key: str) -> tuple[pandas.DataFrame, dict[str, int | float]]:
	"""Run the 2 x 3 fixed validation sweep for one selected model."""
	seed_everything(RANDOM_STATE)

	X_train, y_train, _, _ = data_prep()
	label_encoder, class_weights = make_label_encoder(y_train)
	results: list[dict[str, int | float | str]] = []
	model_name = MODELS[model_key]

	print("=" * 80)
	print(f"{model_name.upper()} SANITY SWEEP")
	print("=" * 80)

	for num_epochs, max_length in itertools.product(
		SWEEP_GRID["num_epochs"],
		SWEEP_GRID["max_length"],
	):
		config = config_with_overrides(
			num_epochs = num_epochs,
			max_length = max_length,
		)
		print(f"\nTrying epochs={num_epochs}, max_length={max_length}")

		classifier = make_classifier(model_key, label_encoder, class_weights, **config)
		classifier.fit(X_train, y_train)

		model = typing.cast(TransformerModel, classifier.model)
		results.append({
			"model": model_name,
			"num_epochs": num_epochs,
			"max_length": max_length,
			"best_epoch": model.best_epoch_,
			"val_f1": model.best_val_f1_,
		})

	results_df = pandas.DataFrame(results).sort_values(
		by = ["val_f1", "max_length", "num_epochs"],
		ascending = [False, False, False],
	).reset_index(drop = True)
	best_row = results_df.iloc[0]
	best_config = config_with_overrides(
		num_epochs = int(best_row["num_epochs"]),
		max_length = int(best_row["max_length"]),
	)

	sweep_path = f"{model_key}_sanity_sweep.csv"
	results_df.to_csv(sweep_path, index = False)
	print("\nSweep summary:")
	print(results_df.to_string(index = False))
	print(f"\nSaved sweep summary to {sweep_path}")
	print(
		f"Selected {model_name} recipe: epochs={best_config['num_epochs']}, "
		f"max_length={best_config['max_length']}"
	)
	print()

	return results_df, best_config


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments."""
	parser = argparse.ArgumentParser(
		description = "Fine-tune one required transformer model for response clarity classification.",
	)
	parser.add_argument(
		"--model",
		required = True,
		choices = list(MODELS.keys()),
		help = "Which required transformer to train.",
	)
	parser.add_argument(
		"--run-sweep",
		action = "store_true",
		help = "Run the 2 x 3 sanity sweep for the selected model before its final training run.",
	)

	return parser.parse_args()


def main() -> None:
	"""CLI entry point."""
	args = parse_args()

	config: dict[str, int | float] = {}
	if args.run_sweep:
		_, config = run_tiny_sweep(args.model)

	train_and_evaluate(args.model, **config)


if __name__ == "__main__":
	main()
