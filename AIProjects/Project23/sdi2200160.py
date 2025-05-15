from __future__ import annotations


import logging; logging.basicConfig(level = logging.INFO)
import os; os.environ["PYTORCHINDUCTOR_LOGLEVEL"] = "ERROR"
from pathlib import Path
import random
from typing import cast
import warnings; warnings.simplefilter(action = "ignore", category = UserWarning)

from rich import print
from rich.progress import Progress, track

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

		if split == "test":
			dataset[split] = dataset[split].remove_columns("labels")

	return dataset
