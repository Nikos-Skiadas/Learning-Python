from __future__ import annotations


import argparse
import pathlib

import pandas

from . data import MusicSeries, MusicDataFrame
from . encoding import encode_genres, embed_audio, embed_lyrics


def load_dataset(path: str | pathlib.Path, k: int, epochs: int,
	force: bool = False,
) -> pandas.DataFrame:
	path = pathlib.Path(path)

	dataset_path = path / f"dataset.{k}.csv"
	genres_path = path / f"dataset.{k}.genres.parquet"
	audio_path = path / f"dataset.{k}.audio.parquet"
	lyrics_path = path / f"dataset.{k}.lyrics.parquet"
	tags_path = path / f"dataset.{k}.tags.parquet"


	if dataset_path.exists() and all(path.exists() for path in (genres_path, audio_path, lyrics_path)) and not force:
		return pandas.read_csv(dataset_path, index_col = 0)

	lyrics = MusicSeries.from_tar(path / "processed_lyrics.tar.gz")
	genres = MusicSeries.from_csv(path / "id_genres.csv")
	tags = MusicSeries.from_csv(path / "id_tags.csv")
	info = MusicDataFrame.from_csv(path / "id_information.csv")

	audio_stats = MusicDataFrame.from_csv(path / "id_mfcc_stats.tsv.bz2")

	dataset = audio_stats.intersection(genres.top(k), tags, *[info[column] for column in info.columns], lyrics)

	genre_embeddings = encode_genres(dataset["genres"])
	audio_embeddings = embed_audio(dataset[[column for column in dataset.columns if column.startswith(("MFCC", "cov_"))]], epochs)
	lyric_embeddings = embed_lyrics(dataset["lyrics"])
	tags_embeddings = encode_genres(dataset["tags"])

	dataset.to_csv(dataset_path)

	genre_embeddings.to_parquet(genres_path)
	audio_embeddings.to_parquet(audio_path)
	lyric_embeddings.to_parquet(lyrics_path)
	tags_embeddings.to_parquet(tags_path)

	return dataset


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("data", type = str, help = "Path to the data archive.")
	parser.add_argument("-k", type = int, help = "Number of top genres to consider.", default = 5)
	parser.add_argument("--epochs", type = int, help = "Number of epochs to train the audio autoencoder for.", default = 1)
	parser.add_argument("--force", action = "store_true", help = "Whether to ignore cached datasets and embeddings.")

	args = parser.parse_args()

	print(load_dataset(args.data, args.k, args.epochs, args.force))
