from __future__ import annotations


import functools

import pandas
import typing


type BinaryDataFrame = pandas.DataFrame


class MusicSeries(pandas.Series):

	@classmethod
	def from_csv(cls, path: str) -> typing.Self:
		return cls(
			pandas.read_csv(path,
				sep = "\t",
				index_col = 0,
				low_memory = False,
			).squeeze()
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
		return self[self.top_labels(k)]


class MusicDataFrame(pandas.DataFrame):

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
	ids = audio_stats.index \
		.intersection(lyrics.index, sort = False) \
		.intersection(genres.index, sort = False)

	audio_stats = audio_stats.loc[ids]
	lyrics = lyrics.loc[ids]
	genres = genres.loc[ids]

	mask = (
		genres.notna()
		& lyrics.notna()
		& lyrics.astype(str).str.strip().ne("")
		& ~audio_stats.isna().any(axis = "columns")
	)

	audio_stats = audio_stats.loc[mask]
	lyrics = lyrics.loc[mask]
	genres = genres.loc[mask]

	return MusicDataFrame(
		{
			"id": audio_stats.index,
			"genre": genres.to_numpy(),
			"lyrics": lyrics.to_numpy(),
			"mfcc_stats": audio_stats.to_numpy().tolist(),
		}
	).drop_duplicates(subset = "id").reset_index(drop = True)
