from __future__ import annotations


import typing


class Game:

	...


class Square(tuple[int, int]):

	def __repr__(self) -> str:
		...


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
