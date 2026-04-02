from __future__ import annotations


import functools
import pathlib
import tarfile

import pandas
import typing


type BinaryDataFrame = pandas.DataFrame


class MusicSeries(pandas.Series):

	@property
	def _constructor(self) -> type[typing.Self]:
		return self.__class__


	@classmethod
	def from_csv(cls, path: str) -> typing.Self:
		return cls(
			pandas.read_csv(path,
				sep = "\t",
				index_col = 0,
				low_memory = False,
			).squeeze()
		)


	@classmethod
	def from_tar(cls, path: str) -> typing.Self:
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


	def mask(self, genres: typing.Iterable[str]) -> pandas.Series:
		return self.encoding[genres].any(
			axis = "columns",
		)

	def distribution(self) -> pandas.Series:
		return self.encoding.sum(
			axis = "index",
		)

	def top_labels(self, k: int) -> pandas.Index:
		return self.distribution().sort_values(
			ascending = False,
		).head(k).index

	def top(self, k: int) -> pandas.Series:
		return self[self.mask(self.top_labels(k))]


class MusicDataFrame(pandas.DataFrame):

	@property
	def _constructor(self) -> type[typing.Self]:
		return self.__class__


	@classmethod
	def from_csv(cls, path: str) -> typing.Self:
		return cls(
			pandas.read_csv(path,
				sep = "\t",
				index_col = 0,
				low_memory = False,
			)
		)


def intersection(
	audio_stats: MusicDataFrame,

	lyrics: MusicSeries,
	genres: MusicSeries,
) -> MusicDataFrame:
	ids = audio_stats.index.intersection(lyrics.index).intersection(genres.index)

	audio_stats = audio_stats.loc[ids]

	lyrics = lyrics.loc[ids]  # type: ignore
	genres = genres.loc[ids]  # type: ignore

	mask = genres.notna() & lyrics.notna() & lyrics.astype(str).str.strip().ne("")

	audio_stats = audio_stats.loc[mask]

	lyrics = lyrics.loc[mask]  # type: ignore
	genres = genres.loc[mask]  # type: ignore

	return MusicDataFrame(
		pandas.concat(
			[
				genres.rename("genre"),
				lyrics.rename("lyrics"),
				audio_stats
			],
			axis = "columns",
		)
	)
