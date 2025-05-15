from __future__ import annotations


import logging; logging.basicConfig(level = logging.INFO)
import os; os.environ["PYTORCHINDUCTOR_LOGLEVEL"] = "ERROR"
from pathlib import Path
import random
from typing import cast
import warnings; warnings.simplefilter(action = "ignore", category = UserWarning)

from rich import print
from rich.progress import Progress, track

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch; torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
import datasets
import transformers


class BertSentenceEmbedder:

	def __init__(self, *,
		model_name = "bert-base-uncased",  # Default to the standard uncased BERT
	):
		self.tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
		self.model: transformers.BertModel = transformers.BertModel.from_pretrained(model_name,
			output_hidden_states = True,
		)
		self.model = self.model.to(self.model,
			device = torch.get_default_device())
		self.model.eval()  # Evaluation mode

	def __call__(self, texts: list[str] | str) -> torch.Tensor:
		return self.encode(texts)


	def encode(self, texts: list[str] | str) -> torch.Tensor:
		if isinstance(texts, str):
			texts = [texts]

		# Tokenize text input
		tokens = self.tokenizer(texts,
			padding = True,  # Pad sequences to the same length
			truncation = True,  # Truncate if too long for BERT
			return_tensors = "pt",  # Return as PyTorch tensors
		)

		# Inference without gradients
		with torch.no_grad():
			output = self.model(**tokens)
			hidden_states = output.hidden_states  # (layers, batch, tokens, hidden_dim)

		token_embeddings = torch.stack(hidden_states,
			dim = 0,  # Stack to shape: (layers, batch, tokens, dim)
		)

		# Return sentence embedding: (batch, dim) — one vector per input text
		return token_embeddings.sum(
			 dim = 0,  # Sum across layers
		).mean(
			 dim = 1,  # Mean across tokens
		)


def fix_seed(seed: int = 42):
	random.seed(seed)

	np.random.seed(seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # if using multi-GPU

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


fix_seed()


def twitter_dataset(
	model_name: str = "bert-base-uncased",
	root: Path = Path().cwd(),
**column_types: datasets.Value):
	if not column_types:
		column_types = dict(
			index  = datasets.Value(dtype = "int32" ),
			text   = datasets.Value(dtype = "string"),
			labels = datasets.Value(dtype = "int32" ),
		)

	features = datasets.Features(column_types)

	dataset = cast(datasets.DatasetDict,
		datasets.load_dataset("csv",
			name = "Twitter",
			data_files = dict(
				train = str(root / "train_dataset.csv"),
				val   = str(root / "val_dataset.csv"  ),
				test  = str(root / "test_dataset.csv" ),
			),
		)
	)

	for split in dataset:
		columns = dict(zip(dataset[split].column_names, column_types))
		dataset[split] = dataset[split].rename_columns(columns).cast(features)

#	tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

#	def tokenize(batch):
#		return tokenizer(batch["text"],
#			padding = "max_length",
#			truncation = True,
#		)

#	dataset = dataset.map(tokenize)
#	dataset.set_format(
#		type = "torch",
#		columns = [
#			"input_ids",
#			"attention_mask",
#			"labels",
#		],
#	)

	dataset["test"] = dataset["test"].remove_columns("labels")

	return dataset

model_name = "bert-base-uncased"
model = transformers.BertForSequenceClassification.from_pretrained(model_name,
	num_labels = 2,
)

training_args = transformers.TrainingArguments(
	output_dir = "./results",
	logging_dir = "./logs",

	evaluation_strategy = "epoch",
	save_strategy = "epoch",

#	per_device_train_batch_size = 16,
#	per_device_eval_batch_size = 64,

	num_train_epochs = 4,
	learning_rate = 1e-4,
	weight_decay = 1e-2,

	load_best_model_at_end = True,
#	metric_for_best_model = "accuracy",  # `eval_loss` by default
)


dataset = twitter_dataset()


def compute_metrics(eval_pred: transformers.EvalPrediction) -> dict[str, float]:
	metrics = evaluate.combine(
		[
			evaluate.load("accuracy"                     ),
			evaluate.load("precision", average = "binary"),
			evaluate.load("recall"   , average = "binary"),
			evaluate.load("f1"       , average = "binary"),
		]
	)

	y_pred, y_true = eval_pred
	y_pred = np.argmax(y_pred,
		axis = 1,
	)

	return metrics.compute(
		predictions = y_pred,
		references  = y_true,
	)


trainer = transformers.Trainer(
	model = model,
	args = training_args,
	train_dataset = dataset["train"],
	eval_dataset = dataset["val"],
	processing_class = transformers.BertTokenizer.from_pretrained(model_name),
	compute_metrics = compute_metrics,
)

trainer.train()
