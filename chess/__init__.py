from __future__ import annotations


import typing


class Game:

	...


class Square(tuple[int, int]):

	@classmethod
	def from_notation(cls, notation: str) -> Square:
		file, rank = notation

		#TODO: Limit squares that can be made.

		return Square([int(rank) - 1, ord(file) - ord("a")])


	def __repr__(self) -> str:
		return self.file + str(self.rank)


	@property
	def rank(self) -> int:
		return self[0] + 1

	@property
	def file(self) -> str:
		return chr(self[1] + ord("a"))

	@property
	def black(self) -> bool:
		return (self[0] + self[1]) % 2 == 0


class Board(list[Square]):

	...


class Player:

	...


class Piece:

	...


class History:

	...


class Move:

	...
