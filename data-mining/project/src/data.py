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
	def from_csv(cls, path: str) -> typing.Self:
		return cls(
			pandas.read_csv(path,
				sep = "\t",
				index_col = 0,
				low_memory = False,
			)
		)

	def intersection(self, *attributes: MusicSeries) -> MusicDataFrame:
		indices = self.index.intersection(pandas.Index(set.intersection(*(set(attribute.index) for attribute in attributes))))
		combined = pandas.concat([attribute.loc[indices] for attribute in attributes], axis = "columns")
		mask = combined.notna().all(axis = "columns") \
			& combined.astype(str).apply(lambda column: column.str.strip().ne("")).all(axis = "columns")

		return MusicDataFrame(
			pandas.concat([combined.loc[mask], self.loc[indices].loc[mask]], axis = "columns")
		)
