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

	return MusicDataFrame(
		{
			"id": ids,
			"genre": genres.loc[ids].to_numpy(),
			"lyrics": lyrics.loc[ids].to_numpy(),
			"mfcc_stats": audio_stats.loc[ids].to_numpy().tolist(),
		}
	)
