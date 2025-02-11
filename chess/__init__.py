from __future__ import annotations


import typing


class Square(tuple[int, int]):

	def __new__(cls, notation: str) -> Square:
		file, rank = notation

		if not 'a' <= file <= 'h' or not '1' <= rank <= '8':
			raise IndexError(f"Invalid square notation: {notation}")

		return super().__new__(cls, (int(rank) - 1, ord(file) - ord("a")))


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


	def __sub__(self, other: Square) -> tuple[int, int]:
		return (self[0] - other[0], self[1] - other[1])


























































class Game:

	def __init__(self):
		self.board = Board()

		self.history = History()

		self.white = Player()
		self.black = Player()


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
