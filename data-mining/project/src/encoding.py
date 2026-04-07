from __future__ import annotations


import math

import pandas
import rich.progress
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
		epochs: int = 8,
		batch_size: int = 1,
	) -> None:
		dataset = torch.utils.data.TensorDataset(self.normalize(data))
		loader = torch.utils.data.DataLoader(dataset,
			batch_size = batch_size,
			shuffle = True,
			generator = torch.Generator(
				device = data.device,
			),
		)

		self.train()

		with rich.progress.Progress(
			rich.progress.TextColumn("[bold blue]{task.description}"),
			rich.progress.BarColumn(),
			rich.progress.MofNCompleteColumn(),
			rich.progress.TextColumn("loss: {task.fields[loss]:.4f}"),
			rich.progress.TimeRemainingColumn(),
			rich.progress.TimeElapsedColumn(),
		) as progress:
			epoch_task = progress.add_task("epoch", total = epochs, loss = 0.)
			batch_task = progress.add_task("batch", total = len(loader), loss = 0.)

			cumulative_loss = 0.

			for epoch in range(epochs):
				total_loss = 0.
				samples = 0

				progress.reset(batch_task, total = len(loader))

				for batch, in loader:
					loss = self.loss_fn(self(batch), batch); self.optimizer.zero_grad()
					loss.backward(); self.optimizer.step()

					total_loss += loss.item() * batch.size(0); samples += batch.size(0)
					progress.update(batch_task, advance = 1, loss = total_loss / samples)

				cumulative_loss += total_loss / len(data)
				progress.update(epoch_task, advance = 1, loss = cumulative_loss / (epoch + 1))

			progress.remove_task(batch_task)

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


def embed_audio(features: pandas.DataFrame) -> pandas.DataFrame:
	tensor = torch.tensor(features.values, dtype = torch.float32)

	model = AudioAutoencoder(len(features.columns))
	model.compile()
	model.fit(tensor)
	embeddings = model.encode(tensor).numpy(force = True)

	return pandas.DataFrame(embeddings,
		index = features.index,
		columns = [f"audio_{i:03d}" for i in range(model.bottleneck)],
	)


def embed_lyrics(lyrics: pandas.Series,
	model_name: str = "all-MiniLM-L6-v2",
) -> pandas.DataFrame:
	model = sentence_transformers.SentenceTransformer(model_name)
	embeddings = model.encode(lyrics.tolist())

	return pandas.DataFrame(
		embeddings,
		index = lyrics.index,
		columns = [f"lyric_{i:03d}" for i in range(embeddings.shape[1])],
	)
