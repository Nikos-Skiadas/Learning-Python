from __future__ import annotations


import functools

import pandas
import typing


type BinaryDataFrame = pandas.DataFrame


def normalize_ids(index: pandas.Index) -> pandas.Index:
	return pandas.Index(index.map(
		lambda song_id: str(song_id).strip(),
	))


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
		return self[self.mask(self.top_labels(k))]


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
	# Keep the join key consistent even if one file loads ids as ints and another as strings.
	audio_stats.index = normalize_ids(audio_stats.index)
	lyrics.index = normalize_ids(lyrics.index)
	genres.index = normalize_ids(genres.index)

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

	# If audio stats are wide, store each row as one MFCC vector. If they are already one column, keep that value.
	mfcc_stats = (
		audio_stats.iloc[:, 0].to_numpy()
		if audio_stats.shape[1] == 1
		else audio_stats.to_numpy().tolist()
	)

	return MusicDataFrame(
		{
			"id": audio_stats.index,
			"genre": genres.to_numpy(),
			"lyrics": lyrics.to_numpy(),
			"mfcc_stats": mfcc_stats,
		}
	).drop_duplicates(subset = "id").reset_index(drop = True)
