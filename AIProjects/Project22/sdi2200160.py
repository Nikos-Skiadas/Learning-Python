from __future__ import annotations

import os; os.environ["PYTORCHINDUCTOR_LOGLEVEL"] = "ERROR"
import warnings; warnings.simplefilter(action = "ignore", category = UserWarning)

import argparse
from collections import Counter
from functools import wraps
import json
import math
from pathlib import Path
import random
import re
import string
from typing import Callable, Iterable, Literal, Self

from rich import print
from rich.progress import Progress, track

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.utils
import torch; torch.set_default_device("cuda")

import nltk
import nltk.stem
import nltk.corpus


nltk.download('wordnet')  # for lemmatization
nltk.download('stopwords')  # for removing stopwords

print()

cache_path = Path.cwd()


def fix_seed(seed: int = 42):
	random.seed(seed)

	np.random.seed(seed)
	sklearn.utils.check_random_state(seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # if using multi-GPU

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


class Preprocessor:

	def __call__(self, text: str) -> str:
		text = re.sub(r"@\w+"   , "", text)  # remove mentions
		text = re.sub(r"#\w+"   , "", text)  # remove hashtags
		text = re.sub(r"\S+@\S+", "", text)  # remove emails

		return text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation


class Tokenizer:

	def __init__(self):
		self.stopwords = set(nltk.corpus.stopwords.words("english"))
		self.lemmatizer = nltk.WordNetLemmatizer()
		self.stemmer = nltk.stem.PorterStemmer()
		self.tokenizer = nltk.tokenize.TweetTokenizer(
			preserve_case = False,
			reduce_len = True,
			strip_handles = True,
		)

	def __call__(self, text: str):
		tokens = []

		for token in self.tokenizer.tokenize(text):
			if token and not token.isdigit() and token not in self.stopwords:
				token = self.lemmatizer.lemmatize(token)
				token = self.stemmer.stem(token)

				tokens.append(token)

		return tokens


def preprocess_and_tokenize(text: str) -> list[str]:
	text = Preprocessor()(text)
	tokens = Tokenizer()(text)

	return tokens


class Vocabulary(dict[str, int]):

	def __init__(self, word2idx: dict[str, int], *,
		pad_token = "<pad>",
		unk_token = "<unk>",
	):
		super().__init__(word2idx)

		self.pad_token = pad_token
		self.unk_token = unk_token

		self.pad_idx = self.get(pad_token, 0)
		self.unk_idx = self.get(unk_token, 1)

	def __call__(self, tokens: list[str]) -> list[int]:
		return [self.get(token, self.unk_idx) for token in tokens]


class TextTransform:

	def __init__(self,
		vocabulary: Vocabulary,
		preprocessor: Callable | None = None,
		tokenizer: Callable | None = None,
		max_len: int | None = None,
	):
		self.vocabulary = vocabulary
		self.preprocessor = preprocessor
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __call__(self, text: str) -> torch.Tensor:
		if self.preprocessor: text = self.preprocessor(text)
		if self.tokenizer: tokens = self.tokenizer(text)
		else: tokens = text.split()

		indices = self.vocabulary(tokens)

		if self.max_len is not None:
			if len(indices) < self.max_len: indices += [self.vocabulary.pad_idx] * (self.max_len - len(indices))
			else: indices = indices[:self.max_len]

		return torch.tensor(indices,
			dtype = torch.long,
		)


class Embedding(torch.nn.Embedding):

	word2idx: Vocabulary  # word to index mapping


	@classmethod
	def from_glove(cls,
		embedding_dim: int = 100,
		num_of_tokens: int = 0,  # HACK: 0 for Twitter GloVe hardcoded
		freeze: bool = False,
		pad_token: str = "<pad>",
		unk_token: str = "<unk>",
	**kwargs) -> Self:
		source_path: Path = \
			Path("embeddings") / f"glove.{num_of_tokens}B.{embedding_dim}d.txt" if num_of_tokens else \
			Path("embeddings") / f"glove.twitter.27B.{embedding_dim}d.txt"
		target_path: Path = source_path.with_suffix(".word2vec.txt")

		cls.convert_glove_to_word2vec(source_path, target_path)

		word2idx, tensor = cls.load_word2vec_format(target_path)

		# Insert special tokens:
		pad_vector = torch.zeros(tensor.shape[1], device = tensor.device)
		unk_vector = torch.randn(tensor.shape[1], device = tensor.device) * 0.1  # smaller variance

		# Rebuild mapping with special tokens:
		word2idx = {
			pad_token: 0,
			unk_token: 1,
		**{word: idx + 2 for word, idx in word2idx.items()}}

		# Stack special vectors:
		tensor = torch.vstack(
			[
				pad_vector,
				unk_vector,
			tensor]
		)

		self = cls.from_pretrained(tensor,
			freeze = freeze,
		**kwargs)
		self.word2idx = Vocabulary(word2idx,
			pad_token = pad_token,
			unk_token = unk_token,
		)

		return self

	@classmethod
	def convert_glove_to_word2vec(cls,
		source_path: Path,
		target_path: Path | None = None,
	):
		with open(source_path, "r+", encoding="utf-8") as source_file:
			lines = source_file.readlines()  # read all lines

		num_tokens, embedding_dim = len(lines), len(lines[0].strip().split()) - 1  # count vocabulary size and embeddings dimension

		with open(target_path if target_path is not None else source_path, "w+", encoding="utf-8") as target_file:
			target_file.write(f"{num_tokens} {embedding_dim}\n")  # write the `word2vec` header

			for line in track(lines, "converting glove to word2vec".ljust(32), num_tokens):
				target_file.write(line)  # copy the rest of the lines

	@classmethod
	def load_word2vec_format(cls, word2vec_path: Path) -> tuple[Vocabulary, torch.Tensor]:
		with open(word2vec_path, "r+", encoding="utf-8") as word2vec_file:
			num_tokens, embedding_dim = map(int, word2vec_file.readline().strip().split())

			word2idx = {}  # word to index mapping
			vectors = torch.zeros(num_tokens, embedding_dim)  # preallocate tensor for word vectors

			for index, line in track(enumerate(word2vec_file), "get embeddings from word2vec".ljust(32), num_tokens):
				word, *vec = line.strip().split()
				vec = list(map(float, vec))

				if len(vec) < embedding_dim: vec += [0.0] * (embedding_dim - len(vec))
				if len(vec) > embedding_dim: vec = vec[:embedding_dim]

				vec_tensor = torch.tensor(vec)

				word2idx[word] = index       # map word to index
				vectors[index] = vec_tensor  # assign vector to the corresponding index

		return Vocabulary(word2idx), vectors


	def index(self, key: str | Iterable[str]) -> int | list[int]:
		return [self.word2idx[item] for item in key] if isinstance(key, Iterable) else self.word2idx[key]

	def get(self, key: str | Iterable[str]) -> torch.Tensor:
		return self.weight[self.index(key)]

	def prune_with_frequencies(self,
		frequencies: Counter[str],
		min_frequency: int = 1,
		max_vocab_size: int | None = None,
		pad_token: str = "<pad>",
		unk_token: str = "<unk>",
	) -> Self:
		# Filter tokens based on frequency and availability in current vocabulary:
		tokens = [token for token, frequency in frequencies.items() if frequency >= min_frequency and token in self.word2idx]

		# Sort tokens by frequency (desc), then alphabetically (asc):
		tokens.sort(
			key = lambda token: (-frequencies[token], token)
		)

		# Apply vocab size cap:
		tokens = tokens[:max_vocab_size]

		# Build pruned token list:
		all_tokens = [
			pad_token,
			unk_token,
		] + tokens

		word2idx = {token: idx for idx, token in enumerate(all_tokens)}

		# Build new weight matrix:
		embedding_dim = self.embedding_dim
		new_weights = torch.zeros(len(all_tokens), embedding_dim)

		for token, new_idx in word2idx.items():
			if   token == pad_token: continue
			elif token == unk_token: new_weights[new_idx] = torch.randn(embedding_dim) * 0.1
			else:
				old_idx = self.word2idx[token]
				new_weights[new_idx] = self.weight[old_idx]

		# Create new embedding layer with new weights:
		new_embedding = self.from_pretrained(new_weights)
		new_embedding.word2idx = Vocabulary(word2idx,
			pad_token = pad_token,
			unk_token = unk_token,
		)

		return new_embedding


class TwitterDataset(torch.utils.data.Dataset):

	def __init__(self, split: Literal["train", "val", "test"], *,
		transform: Callable,
	):
		self.data = self.load_data(split).reset_index()
		self.transform = transform

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx: int | slice):
		text = self.data.Text[idx]
		tokens = self.transform(text)

		try: label = torch.tensor(self.data.Label[idx], dtype = torch.float)
		except AttributeError: label = torch.tensor(-1, dtype = torch.float)

		return tokens, label


	@classmethod
	def load_data(cls, split: Literal["train", "val", "test"],
		root: str = "",
		index: str = "ID",
	):
		return pd.read_csv(os.path.join(f"{root}", f"{split}_dataset.csv"),
			index_col = index,  # Set ID column as index
			encoding = "utf-8",
		)


class TwitterLayer(torch.nn.Sequential):

	def __init__(self,
		inputs_dim: int,
		output_dim: int | None = None,
		dropout: float = 0.5,
	):
		super().__init__(
			torch.nn.SiLU(),
			torch.nn.Dropout(
				p = dropout,
			),
			torch.nn.Linear(inputs_dim, output_dim or inputs_dim),
		)


class TwitterModel(torch.nn.Module):

	def __init__(self, embedding: Embedding,
		hidden_dim: int | list[int] = 100,
		num_layers: int = 2,  # ignored if hidden_dim is a list
		dropout: float = 0.5,
	):
		super().__init__()

		self.embedding = embedding

		self.input_dim = self.embedding.embedding_dim
		self.output_dim = 1  # binary classification (positive/negative)

		if isinstance(hidden_dim, int):
			hidden_dim = [hidden_dim] * num_layers

		layer_dims = [self.input_dim] + hidden_dim + [self.output_dim]  # input and output dimensions along with hidden dimensions

		self.model = torch.nn.Sequential(
			*(
				TwitterLayer(
					inputs_dim = inputs_dim,
					output_dim = output_dim, dropout = dropout
				) for inputs_dim, output_dim in zip(
					layer_dims[ :-1],
					layer_dims[1:  ],
				)
			)
		)

		print()
		print(f"Model summary:")
		print()
		print(self)
		print()

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		embeddings = self.embedding.forward(input)
		mask = (input != self.embedding.word2idx.pad_idx).unsqueeze(-1).float()
		embeddings *= mask
		pooled = embeddings.sum(1) / mask.sum(1).clamp(
			min = 1e-6,
		)

		return self.model.forward(pooled).squeeze(-1)


def preload(method):
	cache_file = cache_path / method.__name__; cache_file = cache_file.with_suffix(".pt")

	@wraps(method)
	def wrapper(self, *args, **kwargs):
		if cache_file.exists() and cache_file.is_file():
			with cache_file.open("r+b") as f:
				return torch.load(f)

		result = method(self, *args, **kwargs)

		with cache_file.open("w+b") as f:
			torch.save(result, f)

		return result

	return wrapper


class TwitterClassifier:

	def __init__(self, model: TwitterModel,
		max_len: int = 32,
		path: Path = cache_path / "model.pt",
	):
		self.model = model
		self.max_len = max_len
		self.path = path

	def __enter__(self) -> Self:
		if self.path is not None and self.path.is_file():
			with self.path.open("rb") as file:
				self.model.load_state_dict(torch.load(file))

		return self

	def __exit__(self, *_):
		if self.path is not None:
			with self.path.open("wb+") as file:
				torch.save(self.model.state_dict(), file)


	def compile(self,
		learning_rate: float = 1e-3,
		weight_decay : float = 0,
	):
		def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
			return (y_pred * y_true).mean()

		def precision(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
			true_positive = (y_pred * y_true).sum()
			predicted_positive = y_pred.sum()

			return true_positive / predicted_positive.clamp(1e-6)

		def recall(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
			true_positive = (y_pred * y_true).sum()
			actual_positive = y_true.sum()

			return true_positive / actual_positive.clamp(1e-6)

		def f1(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
			p = precision(y_pred, y_true)
			r = recall(y_pred, y_true)

			return 2 * p * r / (p + r).clamp(1e-6)

		self.model.to(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

		self.optimizer = torch.optim.AdamW(self.model.parameters(),
			lr = learning_rate,
			weight_decay = weight_decay,
		)
		self.loss_fn = torch.nn.BCEWithLogitsLoss()
		self.metrics = {
			"loss"     : self.loss_fn,
			"accuracy" : accuracy,
			"precision": precision,
			"recall"   : recall,
			"f1"       : f1,
		}

		return self

	def compute(self,
		y_pred: torch.Tensor,
		y_true: torch.Tensor,
	) -> dict[str, float]:
		y_prob = torch.sigmoid(y_pred)

		return {name: metric(y_pred if name == "loss" else y_prob, y_true).item() for name, metric in self.metrics.items()}

	def prune(self, train_dataset: TwitterDataset,
		min_frequency: int = 1,
		max_vocab_size: int | None = None,
		pad_token: str = "<pad>",
		unk_token: str = "<unk>",
	):
		if min_frequency > 1 or max_vocab_size is not None:
			frequencies = Counter()

			for text in track(train_dataset.data.Text, "counting frequencies".ljust(32), len(train_dataset.data)):
				frequencies.update(preprocess_and_tokenize(text))

			self.model.embedding = self.model.embedding.prune_with_frequencies( frequencies,
				min_frequency = min_frequency,
				max_vocab_size = max_vocab_size,
				pad_token = pad_token,
				unk_token = unk_token,
			)
			train_dataset.transform.vocabulary = self.model.embedding.word2idx

	def fit(self,
		train_dataset: TwitterDataset,
		val_dataset  : TwitterDataset,
		epochs: int = 1,
		min_frequency: int = 1,
		max_vocab_size: int | None = None,
	**kwargs) -> dict[str, list[float]]:
		self.prune(train_dataset,
			min_frequency = min_frequency,
			max_vocab_size = max_vocab_size,
			pad_token = self.model.embedding.word2idx.pad_token,
			unk_token = self.model.embedding.word2idx.unk_token,
		)
		val_dataset.transform.vocabulary = self.model.embedding.word2idx

		metrics = Counter(
			loss      = [], val_loss      = [],  # type: ignore
			accuracy  = [], val_accuracy  = [],  # type: ignore
			precision = [], val_precision = [],  # type: ignore
			recall    = [], val_recall    = [],  # type: ignore
			f1        = [], val_f1        = [],  # type: ignore
		)
		total_loss = 0.

		print()
		print(f"Training for {epochs} epochs with:")
		print(
			json.dumps(kwargs,
				indent = 4,
			)
		)

		train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
		assert train_loader.batch_size is not None
		batches = len(train_dataset) // train_loader.batch_size

		self.model.train()

		print()

		with Progress() as progress:
			train_task = progress.add_task(description = "finished epoch ---/---".ljust(32), total = epochs )
			batch_task = progress.add_task(description = "training loss -.------".ljust(32), total = batches)

			for epoch in range(epochs):
				progress.reset(batch_task)

				self.model.train()

				for batch_index, batch in enumerate(train_loader,
					start = epoch * batches + 1,
				):
					x, y_true = batch

					self.optimizer.zero_grad()
					y_pred = self.model.forward(x)
					loss = self.loss_fn.forward(
						y_pred,
						y_true,
					)
					loss.backward()
					self.optimizer.step()

					total_loss += loss.item()

					progress.update(batch_task,
						description = f"training loss {total_loss/batch_index:.6f}".ljust(32),
						total = batches,
						advance = 1,
					)

				metrics.update({       name  : [metric] for name, metric in self.evaluate(train_dataset).items()})
				metrics.update({f"val_{name}": [metric] for name, metric in self.evaluate(  val_dataset).items()})

				progress.update(train_task,
					description = f"finished epoch {epoch+1:3d}/{epochs:3d}".ljust(32),
					total = epochs,
					advance = 1,
				)

		return dict(metrics)  # type: ignore

	@torch.no_grad
	def evaluate(self, test_dataset: TwitterDataset, **kwargs) -> dict[str, float]:
		batch_size = kwargs.pop("batch_size", len(test_dataset))
		loader = torch.utils.data.DataLoader(test_dataset,
			batch_size = batch_size,
		**kwargs)

		self.model.eval()

		y_preds = []
		y_trues = []

		for x, y_true in loader:
			y_pred = self.model(x)

			y_preds.append(y_pred)
			y_trues.append(y_true)

		y_pred = torch.cat(y_preds)
		y_true = torch.cat(y_trues)

		return self.compute(
			y_pred,
			y_true,
		)

	@torch.no_grad
	def predict(self, dataset: TwitterDataset, **kwargs) -> torch.Tensor:
		self.model.eval()

		y_pred = []

		for i in range(len(dataset)):
			x, _ = dataset[i]
			x = x.unsqueeze(0)
			logits = self.model(x)
			preds = (torch.sigmoid(logits) > 0.5).long()
			y_pred.append(preds)

		return torch.cat(y_pred)

	@torch.no_grad
	def predict_proba(self, dataset: TwitterDataset, **kwargs) -> torch.Tensor:
		self.model.eval()

		y_probs = []

		for i in range(len(dataset)):
			x, _ = dataset[i]
			x = x.unsqueeze(0)
			probs = torch.sigmoid(self.model(x))
			y_probs.append(probs)

		return torch.cat(y_probs)


	def classification_report_str(self, dataset: TwitterDataset, **kwargs) -> str:
		y_true = torch.cat([y for _, y in torch.utils.data.DataLoader(dataset, **kwargs)])
		y_pred = self.predict(dataset, **kwargs)

		report = sklearn.metrics.classification_report(
			y_true.numpy(force = True),
			y_pred.numpy(force = True), digits = 6
		)

		return str(report)

	def roc_auc(self, dataset: TwitterDataset, **kwargs) -> float:
		y_true = torch.cat([y for _, y in torch.utils.data.DataLoader(dataset, **kwargs)])
		y_prob = self.predict_proba(dataset, **kwargs)

		score = sklearn.metrics.roc_auc_score(
			y_true.numpy(force = True),
			y_prob.numpy(force = True),
		)

		return float(score)

	def roc_curve(self, dataset: TwitterDataset, **kwargs) -> tuple[
		np.ndarray,
		np.ndarray,
		np.ndarray,
	]:
		y_true = torch.cat([y for _, y in torch.utils.data.DataLoader(dataset, **kwargs)])
		y_prob = self.predict_proba(dataset, **kwargs)

		return sklearn.metrics.roc_curve(
			y_true.numpy(force = True),
			y_prob.numpy(force = True),
		)

	def plot_roc_curve(self, dataset: TwitterDataset, **kwargs):
		fpr, tpr, _ = self.roc_curve(dataset, **kwargs)
		auc = self.roc_auc(dataset, **kwargs)

		plt.figure(
			figsize = (
				9,
				9,
			)
		)
		plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}")
		plt.plot(
			[0, 1],
			[0, 1], linestyle = "--", color = "gray"
		)
		plt.xlabel("False Positive Rate")
		plt.ylabel("True Positive Rate")
		plt.title("ROC Curve")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		plt.savefig("roc_curve.pdf")

	def plot_learning_curve(self, metrics: dict[str, list[float]],
		keys: set[str] = {
			"loss",
			"accuracy",
			"precision",
			"recall",
			"f1",
		},
	):
		plt.figure(
			figsize = (
				18,
				6,
			)
		)

		for i, key in enumerate(keys, 1):
			plt.subplot(1, len(keys), i)
			plt.plot(metrics[label :=        key  ], label = label)
			plt.plot(metrics[label := f"val_{key}"], label = label)
			plt.xlabel("epoch")
			plt.ylabel(key)
			plt.title(f"{key} vs. epoch")
			plt.legend()
			plt.grid(True)

		plt.tight_layout()
		plt.savefig("learning_curve.pdf")

	def submit(self, dataset: TwitterDataset, *,
		submission_path: Path = Path("submission.csv"),
	):
		pd.DataFrame(
			dict(
				ID = dataset.data.index,
				Label = self.predict(dataset).tolist(),
			)
		).to_csv(submission_path,
			index = False,
			encoding = "utf-8",
		)


def round_metrics(metrics: dict[str, list[float]],
	digits: int = 6,
) -> dict[str, list[float]]:
	return {k: [round(v, digits) for v in values] for k, values in metrics.items()}


if __name__ == "__main__":
	fix_seed(42)

#	Read arguments:
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed",
		type = int,
		default = 42,
		help = "Random seed for reproducibility",
	)
	parser.add_argument("--layer-dims",
		type = int,
		nargs = "+",
		default = None,
		help = "Lengths of hidden layers",
	)
	parser.add_argument("--hidden-dim",
		type = int,
		default = 100,
		help = "Length of hidden layers",
	)
	parser.add_argument("--num-layers",
		type = int,
		default = 0,
		help = "Number of hidden layers",
	)
	parser.add_argument("--dropout",
		type = float,
		default = .5,
		help = "Dropout probability",
	)
	parser.add_argument("--epochs",
		type = int,
		default = 1,
		help = "Number of epochs to train",
	)
	parser.add_argument("--learning-rate",
		type = float,
		default = 1e-3,
		help = "Learning rate",
	)
	parser.add_argument("--weight-decay",
		type = float,
		default = 1e-1,
		help = "Weight decay",
	)
	parser.add_argument("--glove-dim",
		type = int,
		default = 100,
		help = "GloVe embedding dimension",
	)
	parser.add_argument("--glove-tokens",
		type = int,
		default = 0,
		help = "GloVe number of tokens (0 for Twitter GloVe)",
	)
	parser.add_argument("--freeze",
		action = "store_true",
		help = "Freeze GloVe embedding layer",
	)
	parser.add_argument("--min-frequency",
		type = int,
		default = None,
		help = "Minimum token frequency to include in vocabulary",
	)
	parser.add_argument("--max-vocab-size",
		type = int,
		default = None,
		help = "Maximum vocabulary size",
	)
	parser.add_argument("--max-len",
		type = int,
		default = 32,
		help = "Token sequence length",
	)
	args = parser.parse_args()

#	Fix seed:
	fix_seed(args.seed)

#	Initialize embedding layer and model::
	embedding = Embedding.from_glove(args.glove_dim,
		freeze = args.freeze,
	)
	model = TwitterModel(embedding,
		hidden_dim = args.layer_dims or args.hidden_dim,
		num_layers = args.num_layers,
		dropout    = args.dropout   ,
	)
	model.compile()

	transform = TextTransform(embedding.word2idx,
		preprocessor = Preprocessor(),
		tokenizer = Tokenizer(),
		max_len = args.max_len,
	)
	train_data = TwitterDataset("train", transform = transform)
	val_data   = TwitterDataset("val"  , transform = transform)
	test_data  = TwitterDataset("test" , transform = transform)

#	Initialize classifier:
	with TwitterClassifier(model) as classifier:
		classifier.compile(
			learning_rate = args.learning_rate,
			weight_decay  = args.weight_decay ,
		)

	#	Train the model:
		metrics = classifier.fit(
			train_data,
			val_data,
			epochs = args.epochs,
			batch_size = int(math.log10(len(train_data) + len(val_data))) + 1,
		)

	#	Generate report:
		print()
		print(classifier.classification_report_str(val_data))
		print("ROC AUC:", classifier.roc_auc(val_data))
		print()

		classifier.plot_roc_curve(val_data)
		classifier.plot_learning_curve(metrics)

	#	Submit predictions:
		classifier.submit(test_data,
			submission_path = Path("submission.csv"),
		)


#	python -m sdi2200160 --glove-dim 100 --glove-tokens 0 --max-len 256 --min-frequency 2 --layer-dim 100 50 25 20 10 5 4 2 1 --dropout 1e-1 --weight-decay 1e-2 --learning-rate 1e-4 --epochs 1
