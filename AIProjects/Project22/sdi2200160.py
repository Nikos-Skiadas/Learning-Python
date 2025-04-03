from __future__ import annotations

import warnings; warnings.simplefilter(action = "ignore", category = UserWarning)
from typing import Iterable, Literal, Self

from pathlib import Path

from rich.progress import track
import torch; torch.set_default_device("cuda")


class Embedding(torch.nn.Embedding):

	word2idx: dict[str, int]  # word to index mapping


	@classmethod
	def from_glove(cls,
		embedding_dim: Literal[50, 100, 200, 300] = 50,
	**kwargs) -> Self:
		source_path: Path = Path("embeddings") / f"glove.6B.{embedding_dim}d.txt"
		target_path: Path = source_path.with_suffix(".word2vec.txt")

		cls.convert_glove_to_word2vec(
			source_path,
			target_path,
		)
		word2idx, tensor = cls.load_word2vec_format(target_path)

		self = cls.from_pretrained(tensor, **kwargs)
		self.word2idx = word2idx

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

			for line in track(lines, "converting glove to word2vec", num_tokens):
				target_file.write(line)  # copy the rest of the lines

	@classmethod
	def load_word2vec_format(cls, word2vec_path: Path) -> tuple[dict[str, int], torch.Tensor]:
		with open(word2vec_path, "r+", encoding="utf-8") as word2vec_file:
			num_tokens, embedding_dim = map(int, word2vec_file.readline().strip().split())

			word2idx = {}  # word to index mapping
			vectors = torch.zeros(num_tokens, embedding_dim)  # preallocate tensor for word vectors

			for index, line in track(enumerate(word2vec_file), "get embeddings from word2vec", num_tokens):
				word, *vec = line.strip().split()  # split word and vector
				vec = torch.tensor(list(map(float, vec)))  # convert vector to tensor

				word2idx[word] = index  # map word to index
				vectors[index] = vec  # assign vector to the corresponding index

		return word2idx, torch.tensor(vectors)


	def index(self, key: str | Iterable[str]) -> int | list[int]:
		return [self.word2idx[item] for item in key] if isinstance(key, Iterable) else self.word2idx[key]

	def get(self, key: str | Iterable[str]) -> torch.Tensor:
		return self.weight[self.index(key)]
