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
import torch; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import datasets
import transformers


def fix_seed(seed: int = 42):
	random.seed(seed)

	np.random.seed(seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # if using multi-GPU

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	return seed


class TwitterDataset(datasets.DatasetDict):

	@classmethod
	def preprocessed(cls,
		model_name: str = "bert-base-uncased",
		root: Path = Path().cwd(),
		trim: int | None = None,
	**column_types: datasets.Value):

		if not column_types:
			column_types = dict(
				index  = datasets.Value(dtype = "int32" ),
				text   = datasets.Value(dtype = "string"),
				labels = datasets.Value(dtype = "int32" ),
			)

		features = datasets.Features(column_types)

		logging.info("Loading dataset...")

		dataset = cast(datasets.DatasetDict,
			datasets.load_dataset("csv",
				name = "Twitter",
				data_files = dict(
					train = str(root / "train_dataset.csv"),
					val   = str(root /   "val_dataset.csv"),
					test  = str(root /  "test_dataset.csv"),
				),
			)
		)

		logging.info("Dataset loaded.")
		logging.info("Renaming columns...")

		for split in dataset:
			columns = dict(zip(dataset[split].column_names, column_types))
			dataset[split] = dataset[split].rename_columns(columns).cast(features)

			if trim is not None:
				dataset[split] = dataset[split].select(range(trim))

		logging.info("Columns renamed.")
		logging.info("Processing dataset...")

		tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

		def tokenize(batch):
			return tokenizer(batch["text"],
				padding = "max_length",
				truncation = True,
			)

		dataset = dataset.map(tokenize)
		dataset.set_format(
			type = "torch",
			columns = [
				"input_ids",
				"attention_mask",
				"labels",
			],
		)
		dataset["test"] = dataset["test"].remove_columns("labels")

		logging.info("Dataset processed.")

		return cls(dataset)


class TwitterClassifier:

	def __init__(self,
		model_name: str | Path = "bert-base-uncased",
		num_labels: int = 2,
	) -> None:
		self.model_name = model_name
		self.num_labels = num_labels

		self.tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
		self.model = transformers.BertForSequenceClassification.from_pretrained(model_name,
			num_labels = num_labels,
		)

		self.trained = False


	@classmethod
	def load(cls, model_name: str):
		classifier = cls(Path.cwd() / model_name)
		classifier.trained = True

		return classifier

	def save(self, path: Path):
		self.model.save_pretrained(path)
		self.tokenizer.save_pretrained(path)


	def compile(self, dataset: TwitterDataset,
		training_args: transformers.training_args.TrainingArguments = transformers.training_args.TrainingArguments(
			output_dir = "./results",
			logging_dir = "./logs",

			eval_strategy = "epoch",
			save_strategy = "epoch",

			per_device_train_batch_size = 32,
			per_device_eval_batch_size = 128,
			fp16 = True,
			gradient_accumulation_steps = 4,

			data_seed = fix_seed(),
			seed = fix_seed(),

			num_train_epochs = 1,
			learning_rate = 1e-4,
			weight_decay = 1e-2,

			load_best_model_at_end = True,
		#	metric_for_best_model = "accuracy",  # `eval_loss` by default
		)
	):
		self.trainer = transformers.trainer.Trainer(
			model = self.model,
			args = training_args,
			train_dataset = dataset["train"],
			eval_dataset = dataset["val"],
			processing_class = self.tokenizer,
			compute_metrics = self.compute_metrics,
		)

	def fit(self) -> dict[str, float]:
		if self.trained:
			return dict()

		self.model.train()
		output = self.trainer.train()
		self.save(Path.cwd() / "model")
		self.trained = True

		return output.metrics

	def evaluate(self) -> dict[str, float]:
		self.model.eval()

		return self.trainer.evaluate()


	def predict(self, texts: list[str] | str) -> list[int]:
		return torch.argmax(self.logits(texts),
			dim = 1,
		).tolist()

	def predict_proba(self, texts: list[str] | str) -> list[float]:
		return torch.softmax(self.logits(texts),
			dim = 1,
		)[:, 1].tolist()


	def logits(self, texts: list[str] | str) -> torch.Tensor:
		if isinstance(texts, str):
			texts = [texts]

		tokens = self.tokenizer(texts,
			return_tensors = "pt",
			padding = True,
			truncation = True,
		)
		tokens = {k: v.to(device) for k, v in tokens.items()}

		with torch.no_grad():
			logits = self.model(**tokens).logits

			return logits.cpu()

	@classmethod
	def compute_metrics(cls, eval_pred) -> dict[str, float]:
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

	def submit(self, dataset: TwitterDataset):
		submission = pd.DataFrame(
			data = {
				"index": dataset["test"]["index"],
				"labels": self.predict(dataset["test"]["text"]),
			}
		)

		submission.to_csv("submission.csv",
			index = False,
		)


if __name__ == "__main__":
	fix_seed()

	dataset = TwitterDataset.preprocessed()
	classifier = TwitterClassifier()
#	classifier = TwitterClassifier.load("model")
	classifier.compile(dataset)
	classifier.fit()
	classifier.evaluate()
	classifier.submit(dataset)
