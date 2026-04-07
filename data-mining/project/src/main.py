from __future__ import annotations


import argparse
import pathlib

import pandas

from . data import MusicSeries, MusicDataFrame
from . encoding import encode_genres, embed_audio, embed_lyrics


def load_dataset(path: str | pathlib.Path, k: int) -> pandas.DataFrame:
	path = pathlib.Path(path)

	dataset_path = path / f"dataset.{k}.csv"
	genres_path = path / f"dataset.{k}.genres.parquet"
	audio_path = path / f"dataset.{k}.audio.parquet"
	lyrics_path = path / f"dataset.{k}.lyrics.parquet"

	if dataset_path.exists() and all(p.exists() for p in (genres_path, audio_path, lyrics_path)):
		return pandas.read_csv(dataset_path, index_col = 0)

	lyrics = MusicSeries.from_tar(path / "processed_lyrics.tar.gz")
	genres = MusicSeries.from_csv(path / "id_genres.csv")

	audio_stats = MusicDataFrame.from_csv(path / "id_mfcc_stats.tsv.bz2")

	dataset = audio_stats.intersection(genres.top(k), lyrics)

	mfcc_columns = [c for c in dataset.columns if c.startswith(("MFCC", "cov_"))]

	genre_embeddings = encode_genres(dataset["genres"])
	audio_embeddings = embed_audio(dataset[mfcc_columns])
	lyric_embeddings = embed_lyrics(dataset["lyrics"])

	dataset.to_csv(dataset_path)

	genre_embeddings.to_parquet(genres_path)
	audio_embeddings.to_parquet(audio_path)
	lyric_embeddings.to_parquet(lyrics_path)

	return dataset


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("data", type = str, help = "Path to the data archive.")
	parser.add_argument("-k", type = int, help = "Number of top genres to consider.", default = 5)

	args = parser.parse_args()

	print(load_dataset(args.data, args.k))
