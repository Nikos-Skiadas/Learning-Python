from __future__ import annotations


import functools
import pathlib
import tarfile

import pandas
import typing

from . encoding import encode_genres, embed_audio, embed_lyrics


type BinaryDataFrame = pandas.DataFrame


class MusicSeries(pandas.Series):

	@property
	def _constructor(self) -> type[typing.Self]:
		return self.__class__


	@classmethod
	def from_csv(cls, path: str | pathlib.Path) -> typing.Self:
		return cls(
			pandas.read_csv(path,
				sep = "\t",
				index_col = 0,
				low_memory = False,
			).squeeze()
		)


	@classmethod
	def from_tar(cls, path: str | pathlib.Path) -> typing.Self:
		records: dict[str, str] = {}

		with tarfile.open(path, "r:gz") as archive:
			for member in archive.getmembers():
				if not member.isfile():
					continue

				song_id = pathlib.Path(member.name).stem
				file = archive.extractfile(member)

				if file is not None:
					records[song_id] = file.read().decode("utf-8",
						errors = "replace",
					)

		return cls(
			pandas.Series(records,
				name = "lyrics",
			)
		)


	@functools.cached_property
	def encoding(self) -> BinaryDataFrame:
		return self.str.get_dummies(
			sep = ",",
		)


	def mask(self, genres: typing.Iterable[str], multi: bool = True) -> pandas.Series:
		if multi: return self.encoding[genres].any(axis = "columns")
		else: return self.isin(genres)

	def distribution(self, multi: bool = True) -> pandas.Series:
		if multi: return self.encoding.sum(axis = "index")
		else: return self.value_counts()

	def top_labels(self, k: int, multi: bool = True) -> pandas.Index:
		if multi: return self.distribution(multi = True).sort_values(ascending = False).head(k).index
		else: return self.distribution(multi = False).head(k).index

	def top(self, k: int, multi: bool = True) -> pandas.Series:
		return self[self.mask(self.top_labels(k, multi = multi), multi = multi)]


class MusicDataFrame(pandas.DataFrame):

	@property
	def _constructor(self) -> type[typing.Self]:
		return self.__class__


	@classmethod
	def from_csv(cls, path: str | pathlib.Path) -> typing.Self:
		return cls(
			pandas.read_csv(path,
				sep = "\t",
				index_col = 0,
				low_memory = False,
			)
		)

	def intersection(self, *attributes: pandas.Series) -> pandas.DataFrame:
		indices = self.index.intersection(pandas.Index(set.intersection(*(set(attribute.index) for attribute in attributes))))
		combined = pandas.concat([attribute.loc[indices] for attribute in attributes], axis = "columns")
		mask = combined.notna().all(axis = "columns") \
			& combined.astype(str).apply(lambda column: column.str.strip().ne("")).all(axis = "columns")

		return pandas.concat([combined.loc[mask], self.loc[indices].loc[mask]], axis = "columns")


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

	genre_encoding = encode_genres(dataset["genres"])
	audio_embeddings = embed_audio(dataset[mfcc_columns])
	lyric_embeddings = embed_lyrics(dataset["lyrics"])

	dataset.to_csv(dataset_path)
	genre_encoding.to_parquet(genres_path)
	audio_embeddings.to_parquet(audio_path)
	lyric_embeddings.to_parquet(lyrics_path)

	return dataset


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("data", type = str, help = "Path to the data archive.")
	parser.add_argument("-k", type = int, help = "Number of top genres to consider.", default = 5)

	args = parser.parse_args()

	print(load_dataset(args.data, args.k))
