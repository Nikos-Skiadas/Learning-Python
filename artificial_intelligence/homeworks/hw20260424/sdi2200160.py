"""Fine-tune transformer models for response clarity classification."""


from __future__ import annotations


import functools
import os
import random
import typing

# Required for DeBERTa v3 tokenizer (SentencePiece → protobuf)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas
import torch
import torch.nn
import torch.utils.data
import transformers
import sklearn.base
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

from ..protocols import Scorer
from ..pipelines import Classifier
from ..preprocessing import IdentityPreprocessor
from ..data import data


# Reproducibility seed required by the assignment
RANDOM_STATE = 42

# Device selection (CUDA if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model registry: short name -> HuggingFace model ID
MODELS: dict[str, str] = {
	"bert": "bert-base-uncased",
	"distilbert": "distilbert-base-uncased",
	"deberta": "microsoft/deberta-v3-base",
}


# ============================================================================
# Reproducibility
# ============================================================================

def seed_everything(seed: int = RANDOM_STATE) -> None:
	"""Set all random seeds for full reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


# ============================================================================
# Dataset (internal, used by TransformerModel)
# ============================================================================

class ClarityDataset(torch.utils.data.Dataset):
	"""PyTorch Dataset wrapping tokenized text with optional labels."""

	def __init__(self,
		encodings: dict[str, torch.Tensor],
		labels: torch.Tensor | None = None,
	) -> None:
		self.encodings = encodings
		self.labels = labels

	def __len__(self) -> int:
		return len(self.encodings["input_ids"])

	def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
		item = {key: val[idx] for key, val in self.encodings.items()}

		if self.labels is not None:
			item["labels"] = self.labels[idx]

		return item


# ============================================================================
# Encoder: TokenizerEncoder (Encoder protocol)
# ============================================================================

class TokenizerEncoder(
	sklearn.base.BaseEstimator,
	sklearn.base.TransformerMixin,
):
	"""Wraps AutoTokenizer as a source Encoder.

	Satisfies the Encoder[pandas.DataFrame, BatchEncoding] protocol:
	- fit() is a no-op (pretrained tokenizer needs no fitting)
	- transform() tokenizes question-answer pairs into sentence pair format:
	    [CLS] question [SEP] answer [SEP]

	The tokenizer automatically includes/excludes token_type_ids
	based on the model architecture (BERT/DeBERTa use them,
	DistilBERT does not).

	Inherits from BaseEstimator/TransformerMixin for sklearn compatibility
	(automatic get_params/set_params and fit_transform).
	"""

	def __init__(self,
		model_name: str = "bert-base-uncased",
		max_length: int = 256,
	) -> None:
		self.model_name = model_name
		self.max_length = max_length

	def fit(self, source: pandas.DataFrame, signal = None):
		"""No-op: pretrained tokenizer needs no fitting."""
		return self

	def transform(self, source: pandas.DataFrame) -> dict[str, torch.Tensor]:
		"""Tokenize question-answer pairs into model inputs.

		Args:
			source: DataFrame with 'question' and 'answer' columns.

		Returns:
			Dict with input_ids, attention_mask, and
			(for BERT/DeBERTa) token_type_ids as PyTorch tensors.
		"""
		# Lazy-load tokenizer (avoids downloading at construction time,
		# critical for sklearn's clone() which re-constructs objects)
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


# ============================================================================
# Model: TransformerModel (Model protocol)
# ============================================================================

class TransformerModel(sklearn.base.BaseEstimator):
	"""Wraps transformer fine-tuning as a Model.

	Satisfies the Model[BatchEncoding, np.ndarray] protocol:
	- fit() runs a custom training loop (no HF Trainer API)
	- predict() runs batch inference

	Training loop features:
	- AdamW optimizer with decoupled weight decay
	- Linear warmup + linear decay LR schedule
	- Gradient clipping
	- Optional class-weighted cross-entropy loss
	- Internal train/val split with best-model checkpointing

	Inherits from BaseEstimator so Classifier's get_params(deep=True)
	can expose model__learning_rate, model__num_epochs, etc.
	"""

	def __init__(self,
		model_name: str = "bert-base-uncased",
		num_labels: int = 3,
		batch_size: int = 16,
		learning_rate: float = 2e-5,
		weight_decay: float = 0.01,
		num_epochs: int = 4,
		warmup_ratio: float = 0.1,
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
		self.warmup_ratio = warmup_ratio
		self.max_grad_norm = max_grad_norm
		self.class_weights = class_weights
		self.val_frac = val_frac
		self.device = device

		# Initialized lazily in _build()
		self._model: transformers.PreTrainedModel | None = None
		self.history: dict[str, list[float]] = {}

	def _build(self) -> None:
		"""Load pretrained model from checkpoint."""
		model = transformers.AutoModelForSequenceClassification.from_pretrained(
			self.model_name,
			num_labels = self.num_labels,
		)
		assert isinstance(model, transformers.PreTrainedModel)
		self._model = model.to(self.device)  # type: ignore[arg-type]

	def fit(self,
		source: dict[str, torch.Tensor],
		target: np.ndarray, /,
	) -> TransformerModel:
		"""Fine-tune the transformer model.

		Args:
			source: Tokenized dict from TokenizerEncoder.transform()
			target: Encoded integer labels from LabelEncoder.transform()
		"""
		if self._model is None:
			self._build()

		assert self._model is not None

		labels: torch.Tensor = target if isinstance(target, torch.Tensor) else torch.tensor(target, dtype = torch.long)

		# Internal train/val split (Classifier.fit doesn't pass validation data,
		# so the model handles it internally -- same idea as GridSearchCV's cv)
		train_enc: dict[str, torch.Tensor]
		val_enc: dict[str, torch.Tensor] | None
		val_labels: torch.Tensor | None

		if self.val_frac > 0:
			idx = np.arange(len(labels))
			train_idx, val_idx = sklearn.model_selection.train_test_split(
				idx,
				test_size = self.val_frac,
				random_state = RANDOM_STATE,
				stratify = target,
			)
			train_enc  = {k: v[train_idx] for k, v in source.items()}
			val_enc    = {k: v[val_idx]   for k, v in source.items()}
			train_labels = typing.cast(torch.Tensor, labels[train_idx])
			val_labels = typing.cast(torch.Tensor, labels[val_idx])
		else:
			train_enc    = dict(source)
			train_labels = labels
			val_enc      = None
			val_labels   = None

		# DataLoader
		train_ds = ClarityDataset(train_enc, train_labels)
		train_loader = torch.utils.data.DataLoader(
			train_ds,
			batch_size = self.batch_size,
			shuffle = True,
		)

		# Optimizer with separate weight decay groups
		# (no decay on bias and LayerNorm -- standard practice)
		no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
		param_groups = [
			{
				"params": [p for n, p in self._model.named_parameters()
					if not any(nd in n for nd in no_decay)],
				"weight_decay": self.weight_decay,
			},
			{
				"params": [p for n, p in self._model.named_parameters()
					if any(nd in n for nd in no_decay)],
				"weight_decay": 0.0,
			},
		]

		optimizer = torch.optim.AdamW(param_groups, lr = self.learning_rate)

		# Linear warmup then linear decay
		total_steps = len(train_loader) * self.num_epochs
		warmup_steps = int(total_steps * self.warmup_ratio)

		scheduler = transformers.get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps = warmup_steps,
			num_training_steps = total_steps,
		)

		# Loss with optional class weights for imbalance handling
		loss_fn = torch.nn.CrossEntropyLoss(
			weight = self.class_weights.to(self.device) if self.class_weights is not None else None,
		)

		# Training
		self.history = {"train_loss": [], "val_loss": [], "val_f1": []}
		best_val_f1 = 0.0
		best_state: dict[str, torch.Tensor] | None = None

		for epoch in range(self.num_epochs):
			# --- Train ---
			self._model.train()
			total_loss = 0.0

			for batch in train_loader:
				optimizer.zero_grad()

				model_inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
				outputs = self._model(**model_inputs)

				loss = loss_fn(outputs.logits, batch["labels"].to(self.device))

				loss.backward()
				torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.max_grad_norm)
				optimizer.step()
				scheduler.step()

				total_loss += loss.item()

			avg_loss = total_loss / len(train_loader)
			self.history["train_loss"].append(avg_loss)

			# --- Validate ---
			if val_enc is not None and val_labels is not None:
				val_metrics = self._evaluate(val_enc, val_labels)
				self.history["val_loss"].append(val_metrics["loss"])
				self.history["val_f1"].append(val_metrics["f1"])

				improved = val_metrics["f1"] > best_val_f1

				if improved:
					best_val_f1 = val_metrics["f1"]
					best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

				print(
					f"Epoch {epoch + 1}/{self.num_epochs} | "
					f"Train Loss: {avg_loss:.4f} | "
					f"Val Loss: {val_metrics['loss']:.4f} | "
					f"Val F1: {val_metrics['f1']:.4f}"
					f"{'  *' if improved else ''}"
				)
			else:
				print(f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {avg_loss:.4f}")

		# Restore best model checkpoint
		if best_state is not None:
			self._model.load_state_dict(best_state)
			self._model.to(self.device)  # type: ignore[arg-type]
			print(f"\nRestored best model (Val F1: {best_val_f1:.4f})")

		return self

	@torch.no_grad()
	def _evaluate(self,
		encodings: dict[str, torch.Tensor],
		labels: torch.Tensor,
	) -> dict[str, float]:
		"""Internal evaluation on pre-tokenized data."""
		assert self._model is not None

		self._model.eval()

		ds = ClarityDataset(encodings, labels)
		loader = torch.utils.data.DataLoader(ds, batch_size = self.batch_size)

		all_preds: list[int] = []
		total_loss = 0.0
		loss_fn = torch.nn.CrossEntropyLoss()

		for batch in loader:
			model_inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
			outputs = self._model(**model_inputs)

			total_loss += loss_fn(outputs.logits, batch["labels"].to(self.device)).item()
			all_preds.extend(outputs.logits.argmax(dim = -1).cpu().tolist())

		preds = np.array(all_preds)
		true = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

		return {
			"loss": total_loss / len(loader),
			"f1": float(sklearn.metrics.f1_score(true, preds, average = "macro", zero_division = 0)),  # type: ignore[call-overload]
		}

	@torch.no_grad()
	def predict(self,
		source: dict[str, torch.Tensor], /,
	) -> np.ndarray:
		"""Predict encoded labels from tokenized inputs.

		Args:
			source: BatchEncoding from TokenizerEncoder.transform()

		Returns:
			numpy array of integer predictions.
		"""
		assert self._model is not None

		self._model.eval()

		ds = ClarityDataset(source)
		loader = torch.utils.data.DataLoader(ds, batch_size = self.batch_size)

		all_preds: list[int] = []

		for batch in loader:
			model_inputs = {k: v.to(self.device) for k, v in batch.items()}
			outputs = self._model(**model_inputs)
			all_preds.extend(outputs.logits.argmax(dim = -1).cpu().tolist())

		return np.array(all_preds)


# ============================================================================
# Metric Helpers
# ============================================================================

def macro_averaged(metric_fn: typing.Any, **kwargs: typing.Any) -> typing.Any:
	"""Wrap sklearn metric with macro averaging and zero_division handling.

	Returns a partial function that can be used as a Scorer.
	"""
	return functools.partial(metric_fn,
		average = "macro",
		zero_division = 0,
		**kwargs,
	)


# ============================================================================
# Data Preparation
# ============================================================================

def data_prep() -> tuple[
	pandas.DataFrame,
	pandas.Series,
	pandas.DataFrame,
	pandas.Series,
]:
	"""Prepare data as DataFrames for transformer sentence pair encoding.

	Unlike HW1 which concatenated question + " | " + answer into a single
	Series for TF-IDF / Word2Vec, here we keep questions and answers as
	separate DataFrame columns. The TokenizerEncoder then produces proper
	sentence pair encoding: [CLS] question [SEP] answer [SEP]

	Returns:
		(X_train, y_train, X_test, y_test)
		where X is a DataFrame with 'question' and 'answer' columns,
		and y is a Series of string labels.
	"""
	train_df = data["train"].to_pandas(); assert isinstance(train_df, pandas.DataFrame)
	test_df  = data["test" ].to_pandas(); assert isinstance(test_df,  pandas.DataFrame)

	X_train = pandas.DataFrame({
		"question": train_df.question.fillna("").str.strip(),
		"answer":   train_df.interview_answer.fillna("").str.strip(),
	})
	y_train = train_df.clarity_label.fillna("").str.strip()

	X_test = pandas.DataFrame({
		"question": test_df.question.fillna("").str.strip(),
		"answer":   test_df.interview_answer.fillna("").str.strip(),
	})
	y_test = test_df.clarity_label.fillna("").str.strip()

	print(f"Training examples: {len(X_train)}")
	print(f"Test     examples: {len(X_test)}")
	print()

	return X_train, y_train, X_test, y_test


def compute_class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
	"""Compute balanced class weights (same formula as sklearn's 'balanced').

	w_c = n_samples / (n_classes * count_c)
	"""
	counts = torch.bincount(labels, minlength = num_classes).float()

	return len(labels) / (num_classes * counts)


# ============================================================================
# Model Factory
# ============================================================================

def model_factory(
	model_key: str,
	class_weights: torch.Tensor | None = None,
	**overrides,
) -> tuple[TokenizerEncoder, TransformerModel]:
	"""Create encoder + model pair for a given transformer.

	Mirrors tfidf_encoder_factory / word2vec_encoder_factory from HW1.

	Args:
		model_key: One of 'bert', 'distilbert', 'deberta'
		class_weights: Optional class weight tensor for loss function
		**overrides: Override any TransformerModel parameter

	Returns:
		(source_encoder, model) ready to plug into Classifier.
	"""
	model_name = MODELS[model_key]

	source_encoder = TokenizerEncoder(
		model_name = model_name,
		max_length = overrides.pop("max_length", 256),
	)

	defaults: dict[str, typing.Any] = dict(
		model_name = model_name,
		num_labels = 3,
		batch_size = 16,
		learning_rate = 2e-5,
		weight_decay = 0.01,
		num_epochs = 4,
		warmup_ratio = 0.1,
		max_grad_norm = 1.0,
		class_weights = class_weights,
		val_frac = 0.15,
	)
	defaults.update(overrides)

	model = TransformerModel(**defaults)

	return source_encoder, model


# ============================================================================
# Training Orchestration
# ============================================================================

def train_and_evaluate(
	model_key: str,
	**model_overrides: typing.Any,
) -> tuple[typing.Any, dict[str, float], pandas.Series]:
	"""Full training pipeline for a single transformer model.

	Mirrors optimize_with_grid_search from HW1 but adapted for
	transformer fine-tuning. Uses the same Classifier pipeline --
	only the source_encoder (TokenizerEncoder) and model
	(TransformerModel) are swapped in.

	Args:
		model_key: One of 'bert', 'distilbert', 'deberta'
		**model_overrides: Override default hyperparameters

	Returns:
		(classifier, test_scores, decoded_predictions)
	"""
	seed_everything(RANDOM_STATE)

	model_name = MODELS[model_key]

	print("=" * 80)
	print(f"{model_name.upper()} FINE-TUNING")
	print("=" * 80)
	print()

	# Prepare data
	print("Preparing data...")
	X_train, y_train, X_test, y_test = data_prep()

	# Compute class weights from training labels
	label_encoder = sklearn.preprocessing.LabelEncoder()
	label_encoder.fit(y_train)
	class_weights = compute_class_weights(
		torch.tensor(label_encoder.transform(y_train), dtype = torch.long),
	)
	classes: list[str] = list(typing.cast(np.ndarray, label_encoder.classes_))
	print(f"Class weights: {dict(zip(classes, class_weights.tolist()))}")
	print()

	# Build components using the same Classifier pipeline as HW1
	source_encoder, model = model_factory(model_key,
		class_weights = class_weights,
		**model_overrides,
	)

	classifier = Classifier(
		preprocessor = IdentityPreprocessor(),
		model = model,
		source_encoder = source_encoder,
		target_bicoder = label_encoder,  # type: ignore[arg-type]  # LabelEncoder.fit takes 1 positional arg, protocol expects 2
	)

	classifier.compile(
		accuracy  = sklearn.metrics.accuracy_score,
		precision = macro_averaged(sklearn.metrics.precision_score),
		recall    = macro_averaged(sklearn.metrics.recall_score),
		f1        = macro_averaged(sklearn.metrics.f1_score),
	)

	# Print configuration
	print(f"Training {model_name}...")
	print(f"  Max length:    {source_encoder.max_length}")
	print(f"  Batch size:    {model.batch_size}")
	print(f"  Learning rate: {model.learning_rate}")
	print(f"  Weight decay:  {model.weight_decay}")
	print(f"  Epochs:        {model.num_epochs}")
	print(f"  Warmup ratio:  {model.warmup_ratio}")
	print(f"  Val fraction:  {model.val_frac}")
	print(f"  Device:        {model.device}")
	print()

	# Train (Classifier.fit handles the full pipeline:
	#   IdentityPreprocessor → TokenizerEncoder → LabelEncoder → TransformerModel)
	classifier.fit(X_train, y_train)

	# Evaluate
	print("\n" + "=" * 80)
	print("TEST SET EVALUATION")
	print("=" * 80)

	test_scores: dict[str, float] = {
		k: float(v) for k, v in classifier.score(X_test, y_test).items()
	}

	print()
	for name, score in test_scores.items():
		print(f"{name:12s} {score:.4f}")
	print()

	# Generate submission (predict returns decoded string labels via LabelEncoder)
	predictions = pandas.Series(classifier.predict(X_test), name = "Predicted")
	predictions.index.name = "Id"

	model_slug = model_name.replace("/", "-")
	submission_path = f"submission_{model_slug}.csv"
	predictions.to_csv(submission_path)

	print(f"Submission saved to {submission_path}")
	print(predictions.value_counts())
	print()

	return classifier, test_scores, predictions


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description = "Fine-tune transformer models for response clarity classification.",
	)
	parser.add_argument("--models", nargs = "+", required = True,
		choices = list(MODELS.keys()),
		help = "Model(s) to train: bert, distilbert, deberta",
	)
	parser.add_argument("--epochs", type = int, default = 4)
	parser.add_argument("--batch-size", type = int, default = 16)
	parser.add_argument("--learning-rate", type = float, default = 2e-5)
	parser.add_argument("--max-length", type = int, default = 256)

	args = parser.parse_args()

	for model_name_key in args.models:
		train_and_evaluate(
			model_name_key,
			num_epochs = args.epochs,
			batch_size = args.batch_size,
			learning_rate = args.learning_rate,
			max_length = args.max_length,
		)
