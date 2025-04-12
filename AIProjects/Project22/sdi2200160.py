from __future__ import annotations

import warnings; warnings.simplefilter(action = "ignore", category = UserWarning)

import math
from collections import Counter
import os; os.environ["TORCH_INDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"  # disable autotuning
from pathlib import Path
import re
import string
from typing import Callable, Iterable, Literal, Self

from rich.progress import Progress, track
import pandas
import torch; torch.set_default_device("cuda")

# Importing NLTK for Natural Language Processing
import nltk
import nltk.stem
import nltk.corpus

# Download required NLP datasets
nltk.download('wordnet')  # for lemmatization
nltk.download('stopwords')  # for removing stopwords


class Preprocessor:
    def __call__(self, text: str) -> str:
        text = re.sub(r"@\w+"   , "", text)  # remove mentions
        text = re.sub(r"#\w+"   , "", text)  # remove hashtags
        text = re.sub(r"\S+@\S+", "", text)  # remove emails

        return text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation


class Tokenizer:

    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.stem.PorterStemmer()
        self.tokenizer = nltk.tokenize.TweetTokenizer(
            preserve_case = False,
            reduce_len = True,
            strip_handles = True,
        )

    def __call__(self, text: str):
        return [self.stemmer.stem(self.lemmatizer.lemmatize(token))
            for token in self.tokenizer.tokenize(text) if token and not token.isdigit()]


class Vocabulary(dict[str, int]):

	def __init__(self, word2idx: dict[str, int], *,
		pad_token = "<pad>",
		unk_token = "<unk>",
	):
		super().__init__(word2idx)

		self.pad_token = pad_token
		self.unk_token = unk_token

		self.pad_idx = self.get(pad_token, 0)
		self.unk_idx = self.get(unk_token, 0)

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
		else: tokens = text.lower().split()

		indices = self.vocabulary(tokens)

		if self.max_len is not None:
			if len(indices) < self.max_len: indices += [self.vocabulary.pad_idx] * (self.max_len - len(indices))
			else: indices = indices[:self.max_len]

		return torch.tensor(indices,
			dtype = torch.long,
		)


class TwitterDataset(torch.utils.data.Dataset):

	# Class method to load dataset (train, val, test)
	@classmethod
	def load_data(cls, split: Literal["train", "val", "test"],
		root: str = "",
		index: str = "ID",
	):
		return pandas.read_csv(os.path.join(f"{root}", f"{split}_dataset.csv"),
			index_col = index,  # Set ID column as index
			encoding = "utf-8",
		)


	def __init__(self, split: Literal["train", "val", "test"], *,
		transform: Callable,
	):
		self.data = self.load_data(split).reset_index()
		self.transform = transform

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx: int | slice):
		x = self.transform(self.data.Text[idx])  # apply TextTransform -> tensor of token indices
		y = float(self.data.Label[idx].item())  # get label

		return x, y  # return tensor of token indices and label

	@property
	def x(self):
		...


class Embedding(torch.nn.Embedding):

	word2idx: Vocabulary  # word to index mapping


	@classmethod
	def from_glove(cls,
		embedding_dim: Literal[50, 100, 200, 300] = 50,
		pad_token: str = "<pad>",
		unk_token: str = "<unk>",
		**kwargs,
	) -> Self:
		source_path: Path = Path("embeddings") / f"glove.6B.{embedding_dim}d.txt"
		target_path: Path = source_path.with_suffix(".word2vec.txt")

		cls.convert_glove_to_word2vec(source_path, target_path)

		word2idx, tensor = cls.load_word2vec_format(target_path)

		# Insert special tokens:
		pad_vector = torch.zeros(tensor.shape[1], device=tensor.device)
	#	unk_vector = torch.randn(tensor.shape[1], device=tensor.device) * 0.1  # smaller variance

		# Rebuild mapping with special tokens:
		word2idx = {
			pad_token: 0,
		#	unk_token: 1,
		**{word: idx + 2 for word, idx in word2idx.items()}}

		# Stack special vectors:
		tensor = torch.vstack(
			[
				pad_vector,
			#	unk_vector,
			tensor]
		)

		self = cls.from_pretrained(tensor, **kwargs)
		self.word2idx = Vocabulary(word2idx,
			pad_token = pad_token,
		#	unk_token = unk_token,
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
				word, *vec = line.strip().split()  # split word and vector
				vec = torch.tensor(list(map(float, vec)))  # convert vector to tensor

				word2idx[word] = index  # map word to index
				vectors[index] = vec  # assign vector to the corresponding index

		return Vocabulary(word2idx), vectors


	def index(self, key: str | Iterable[str]) -> int | list[int]:
		return [self.word2idx[item] for item in key] if isinstance(key, Iterable) else self.word2idx[key]

	def get(self, key: str | Iterable[str]) -> torch.Tensor:
		return self.weight[self.index(key)]


class TwitterModel(torch.nn.Module):

	def __init__(self, embedding: Embedding,
		hidden_dim: int = 128,
	):
		super().__init__()

		self.embedding = embedding

		self.input_dim = self.embedding.embedding_dim
		self.output_dim = 1  # binary classification (positive/negative)

		self.model = torch.nn.Sequential(
			torch.nn.Linear(self.input_dim, hidden_dim),
			torch.nn.SiLU(),
			torch.nn.Dropout(),  # TODO: add dropout
			torch.nn.Linear(hidden_dim, self.output_dim),  # single output neuron
		#	torch.nn.Sigmoid(),  # output activation function  # FIXME: return logits instead of probabilities
		)


	def forward(self, input: torch.Tensor) -> torch.Tensor:
		"""
		x: LongTensor of shape [batch_size, seq_len] (token indices)
		"""
		embeddings = self.embedding(input)  # [batch_size, seq_len, embedding_dim]

		# Mask out padded positions
		mask = (input != self.embedding.word2idx.get("<pad>", 0)).unsqueeze(-1)  # [batch_size, seq_len, 1]
		embeddings = embeddings * mask  # zero out padded embeddings
		pooled = embeddings.sum(dim = 1) / mask.sum(dim = 1).clamp(min = 1)  # [batch_size, embedding_dim]
		logits = self.model(pooled).squeeze(-1)  # [batch_size]

		return logits


class TwitterClassifier:

	def __init__(self, model: TwitterModel,
		max_len: int = 32,
	):
		self.model = model
		self.max_len = max_len


	def compile(self,
		learning_rate: float = 1e-3,
		weight_decay: float = 0.,
	):
		def accuracy (x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: return (x * y).float().mean()
		def precision(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: return (x * y).float().sum() / x.sum()
		def recall   (x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: return (x * y).float().sum() / y.sum()

		def f1       (x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: return 2 / (1 / precision(x, y) + 1 / recall(x, y))

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


	def fit(self,
		train_dataset: TwitterDataset,
		val_dataset  : TwitterDataset,
		epochs: int = 1,
	**kwargs) -> dict[str, list[float]]:
		train_loader = torch.utils.data.DataLoader(train_dataset, drop_last = True, **kwargs)
		batches = len(train_dataset) // train_loader.batch_size if train_loader.batch_size is not None else None

		metrics = Counter(
			loss      = [], val_loss      = [],  # type: ignore
			accuracy  = [], val_accuracy  = [],  # type: ignore
			precision = [], val_precision = [],  # type: ignore
			recall    = [], val_recall    = [],  # type: ignore
			f1        = [], val_f1        = [],  # type: ignore
		)

		with Progress() as progress:
			train_task = progress.add_task(
				description = "training epoch --/--".ljust(32),
				total = epochs
			)
			batch_task = progress.add_task(
				description = "training loss -.----".ljust(32),
				total = batches,
			)

			for epoch in range(1, epochs + 1):
				progress.reset(batch_task)

				self.model.train()

				for batch in train_loader:
					x, y_true = batch

					self.optimizer.zero_grad()
					y_pred = self.model.forward(x)
					loss = self.loss_fn.forward(
						y_pred,
						y_true,
					)
					loss.backward()
					self.optimizer.step()

					progress.update(batch_task,
						description = f"training loss {loss.item():.4f}".ljust(32),
						total = batches,
						advance = 1,
					)

				metrics.update({       name  : [metric] for name, metric in self.evaluate(train_dataset).items()})
				metrics.update({f"val_{name}": [metric] for name, metric in self.evaluate(  val_dataset).items()})

				progress.update(train_task,
					description = f"training epoch {epoch:02d}/{epochs:02d}".ljust(32),
					total = epochs,
					advance = 1,
				)

		return dict(metrics)  # type: ignore

	@torch.no_grad
	def evaluate(self, test_dataset: TwitterDataset, **kwargs) -> dict[str, float]:
		batch_size = kwargs.pop("batch_size", len(test_dataset))
		loader = torch.utils.data.DataLoader(test_dataset,
			batch_size = batch_size,  # evaluate on the whole dataset
			drop_last = True,
		**kwargs)

		self.model.eval()

		for batch in loader:
			x, y_true = batch

			y_pred = self.model(x)
			metrics = self.compute(
				y_pred,
				y_true,
			)

		return metrics

	@torch.no_grad
	def predict(self, dataset: TwitterDataset, **kwargs) -> torch.Tensor:
		loader = torch.utils.data.DataLoader(dataset,
			drop_last = True,
		**kwargs)

		self.model.eval()

		y_pred = []

		for batch in loader:
			x, _ = batch  # ignore labels
			y_pred.append((torch.sigmoid(self.model(x)) > 0.5).long())

		return torch.cat(y_pred)


if __name__ == "__main__":
	embedding = Embedding.from_glove(50)
	model = TwitterModel(embedding)
	model.compile()

	classifier = TwitterClassifier(model)
	classifier.compile()

	transform = TextTransform(embedding.word2idx,
		preprocessor = Preprocessor(),
		tokenizer = Tokenizer(),
		max_len = 32,
	)

	train_data = TwitterDataset("train",
		transform = transform
	)
	val_data = TwitterDataset("val",
		transform = transform
	)

	history = classifier.fit(
		train_data,
		val_data,
		epochs = 10,
		batch_size = math.isqrt(len(train_data)) + 1,
	)
