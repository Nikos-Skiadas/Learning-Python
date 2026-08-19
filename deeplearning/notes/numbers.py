from __future__ import annotations


from typing import Self


class N(set):

	# https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset

	def __add__(self, other: Self) -> Self:
		...  # +

	def __mul__(self, other: Self) -> Self:
		...  # *

	def __le__(self, other: Self) -> bool:
		...  # <= (Do I really need to define this?)

	def __repr__(self) -> str:
		...

	@property
	def next(self) -> Self:
		...
