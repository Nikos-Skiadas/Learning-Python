from __future__ import annotations


import logging; logging.basicConfig(level = logging.INFO)
import os; os.environ["PYTORCHINDUCTOR_LOGLEVEL"] = "ERROR"
from pathlib import Path
import random
from typing import cast
import warnings; warnings.simplefilter(action = "ignore", category = UserWarning)

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
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


root = Path.cwd(); root.mkdir(
	parents = True,
	exist_ok = True,
)
models_path = root / "models"; models_path.mkdir(
	parents = True,
	exist_ok = True,
)


class TwitterDataset(datasets.DatasetDict):

	@classmethod
	def preprocessed(cls,
		model_name: str = "bert-base-uncased",
		root: Path = root,
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

		tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)

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
		self.trained = (path := models_path / model_name).exists()

		self.model_name = model_name if not self.trained else path
		self.num_labels = num_labels

		self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
		self.model = transformers.DistilBertForSequenceClassification.from_pretrained(model_name,
			num_labels = num_labels,
		)


	def __enter__(self):
		logging.info(f"Loading model {self.model_name}...")

		return self

	def __exit__(self, *_):
		self.model.save_pretrained(models_path / self.model_name)
		self.tokenizer.save_pretrained(models_path / self.model_name)

		return True


	def compile(self, dataset: TwitterDataset,
		training_args: transformers.training_args.TrainingArguments = transformers.training_args.TrainingArguments(
			output_dir = "./results",
			logging_dir = "./logs",

			eval_strategy = "epoch",
			save_strategy = "epoch",

			per_device_train_batch_size = 32,
			per_device_eval_batch_size = 128,
			gradient_accumulation_steps = 4,

			fp16 = True,

			dataloader_num_workers = 20,
		#	dataloader_persistent_workers = True,
			dataloader_pin_memory = True,

			data_seed = fix_seed(),
			seed = fix_seed(),

			num_train_epochs = 1,
			learning_rate = 1e-4,
			weight_decay = 1e-2,

			load_best_model_at_end = True,
		#	metric_for_best_model = "accuracy",  # `eval_loss` by default
		)
	):
		logging.info("Compiling model and initializing its trainer...")

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
			logging.info("Model already trained. Skipping training.")

			return dict()

		logging.info("Training model...")

		self.model.train()
		output = self.trainer.train()
		self.trained = True

		return output.metrics

	def evaluate(self) -> dict[str, float]:
		if not self.trained:
			logging.error("Model not trained. Cannot evaluate.")

			return dict()

		logging.info("Evaluating model...")

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

		def tokenize(batch):
			return self.tokenizer(batch["text"],
				padding = "max_length",
				truncation = True,
			)

		dummy_dataset = datasets.Dataset.from_dict({"text": texts})
		dummy_dataset = dummy_dataset.map(tokenize,
			batched = True,
		)

		dummy_dataset.set_format(
			type = "torch",
			columns = [
				"input_ids",
				"attention_mask",
			],
		)

		return torch.tensor(self.trainer.predict(dummy_dataset).predictions)  # type: ignore[arg-type]


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

		logging.info("Computing metrics...")

		y_pred, y_true = eval_pred
		y_pred = np.argmax(y_pred,
			axis = 1,
		)

		return metrics.compute(
			predictions = y_pred,
			references  = y_true,
		)

	def plot(self, dataset: TwitterDataset,
		output_dir: Path = root / "plots",
	):
		output_dir.mkdir(
			parents = True,
			exist_ok = True,
		)

		logging.info("Plotting results...")

	#	Learning curves:
		if self.trainer.state.log_history:
			logs = pd.DataFrame(self.trainer.state.log_history)

		#	Filter out unnecessary entries
			train_logs = logs[logs["loss"].notna()]
			eval_logs = logs[logs["eval_loss"].notna()]

		#	Plot train vs eval loss:
			plt.figure()
			plt.plot(train_logs["step"], train_logs["loss"],
				label = "Train Loss",
			)
			plt.plot(eval_logs["step"], eval_logs["eval_loss"],
				label = "Eval Loss",
			)
			plt.xlabel("Step")
			plt.ylabel("Loss")
			plt.legend()
			plt.title("Training vs Evaluation Loss")
			plt.savefig(output_dir / "loss_curve.png")
			plt.close()

		#	Plot evaluation metrics:
			metrics = ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]
			for metric in metrics:
				if metric in eval_logs:
					plt.figure()
					plt.plot(eval_logs["step"], eval_logs[metric],
			  			label = metric,
					)
					plt.xlabel("Step")
					plt.ylabel(metric.split("_")[-1].capitalize())
					plt.title(metric.replace("_", " ").title())
					plt.savefig(output_dir / f"{metric}_curve.png")
					plt.close()

	#	AUC and Precision-Recall Curve on validation set:
		validation = self.trainer.predict(dataset["val"])  # type: ignore
		y_true = validation.label_ids
		y_prob = torch.softmax(torch.tensor(validation.predictions),
			dim = 1,
		)[:, 1].numpy()
		y_pred = np.argmax(validation.predictions,
			axis = 1,
		)

	#	ROC Curve:
		fpr, tpr, _ = sklearn.metrics.roc_curve(
			y_true,  # type: ignore
			y_prob,
		)
		y_true = validation.label_ids
		roc_auc = sklearn.metrics.auc(
			fpr,
			tpr,
		)
		plt.figure()
		plt.plot(fpr, tpr,
		   	label = f"ROC AUC = {roc_auc:.2f}",
		)
		plt.plot([0, 1], [0, 1],
			linestyle = "--",
			color = "gray",
		)
		plt.xlabel("False Positive Rate")
		plt.ylabel("True Positive Rate")
		plt.title("ROC Curve")
		plt.legend()
		plt.savefig(output_dir / "roc_curve.png")
		plt.close()

	#	Precision-Recall Curve:
		precision, recall, _ = sklearn.metrics.precision_recall_curve(
			y_true,  # type: ignore
			y_prob,
		)
		y_true = validation.label_ids
		pr_auc = sklearn.metrics.auc(recall, precision)
		plt.figure()
		plt.plot(recall, precision,
			label = f"PR AUC = {pr_auc:.2f}",
		)
		plt.xlabel("Recall")
		plt.ylabel("Precision")
		plt.title("Precision-Recall Curve")
		plt.legend()
		plt.savefig(output_dir / "pr_curve.png")
		plt.close()

	def submit(self, dataset: TwitterDataset):
		logging.info("Submitting predictions...")

		pd.DataFrame(
			data = {
				"index": dataset["test"]["index"],
				"labels": self.predict(dataset["test"]["text"]),
			}
		).to_csv("submission.csv",
			index = False,
		)


if __name__ == "__main__":
	fix_seed()

	dataset = TwitterDataset.preprocessed(trim = 256)

	with TwitterClassifier("bert-base-uncased") as classifier:
		classifier.compile(dataset)
		classifier.fit()
		classifier.evaluate()
		classifier.plot(dataset)
		classifier.submit(dataset)
