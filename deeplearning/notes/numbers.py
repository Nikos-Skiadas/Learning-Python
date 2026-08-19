from __future__ import annotations


from typing import Self


class N(frozenset):

	# https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset

	def __add__(self, other: Self) -> Self:
		...  # +

	def __mul__(self, other: Self) -> Self:
		...  # *

	def __repr__(self) -> str:
		...

	@property
	def next(self) -> Self:
		...
