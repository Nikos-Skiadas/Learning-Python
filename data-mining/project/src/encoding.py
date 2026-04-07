from __future__ import annotations


import math

import pandas
import sentence_transformers
import torch


if torch.cuda.is_available():
	torch.set_default_device("cuda")


class AudioAutoencoder(torch.nn.Module):

	def __init__(self, input_dim: int, bottleneck: int | None = None) -> None:
		super().__init__()

		self.bottleneck = bottleneck or math.isqrt(input_dim) + 1

		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(input_dim, self.bottleneck),
			torch.nn.SiLU(),
		)
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(self.bottleneck, input_dim),
		)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.decoder(self.encoder(x))

	@staticmethod
	@torch.no_grad
	def normalize(data: torch.Tensor) -> torch.Tensor:
		return (data - data.mean(dim = 0)) / data.std(dim = 0)

	def compile(self,
		optimizer: torch.optim.Optimizer | None = None,
		loss_fn: torch.nn.MSELoss | None = None,
		lr: float = 1e-3,
	) -> None:
		self.optimizer = optimizer or torch.optim.Adam(self.parameters(), lr = lr)
		self.loss_fn = loss_fn or torch.nn.MSELoss()

	def fit(self, data: torch.Tensor,
		epochs: int = 200,
		batch_size: int = 1,
	) -> None:
		dataset = torch.utils.data.TensorDataset(self.normalize(data))
		loader = torch.utils.data.DataLoader(dataset,
			batch_size = batch_size,
			shuffle = True,
		)

		self.train()

		for epoch in range(epochs):
			total_loss = 0.0

			for batch, in loader:
				loss = self.loss_fn(self(batch), batch)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				total_loss += loss.item() * batch.size(0)

			print(f"  Autoencoder epoch {epoch + 1}/{epochs} — loss: {total_loss / len(data):.6f}")

		self.eval()

	@torch.no_grad
	def evaluate(self, data: torch.Tensor) -> float:
		normalized = self.normalize(data)

		return self.loss_fn(self(normalized), normalized).item()

	@torch.no_grad
	def encode(self, data: torch.Tensor) -> torch.Tensor:
		normalized = self.normalize(data)

		return self.encoder(normalized)


def encode_genres(genres: pandas.Series) -> pandas.DataFrame:
	return genres.str.get_dummies(sep = ",")


def embed_audio(features: pandas.DataFrame,
	bottleneck: int | None = None,
	epochs: int = 10,
	lr: float = 1e-3,
	batch_size: int = 256,
) -> pandas.DataFrame:
	tensor = torch.tensor(features.values)

	model = AudioAutoencoder(features.shape[1], bottleneck)
	model.compile(lr = lr)
	model.fit(tensor, epochs, batch_size)
	embeddings = model.encode(tensor).numpy(force = True)

	return pandas.DataFrame(embeddings,
		index = features.index,
		columns = [f"audio_{i:03d}" for i in range(model.bottleneck)],
	)


def embed_lyrics(
	lyrics: pandas.Series,
	model_name: str = "all-MiniLM-L6-v2",
	batch_size: int = 256,
) -> pandas.DataFrame:
	model = sentence_transformers.SentenceTransformer(model_name)
	embeddings = model.encode(
		lyrics.tolist(),
		batch_size = batch_size,
		show_progress_bar = True,
	)

	return pandas.DataFrame(
		embeddings,
		index = lyrics.index,
		columns = [f"lyric_{i:03d}" for i in range(embeddings.shape[1])],
	)
