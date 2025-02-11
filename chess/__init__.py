from __future__ import annotations


import typing


class Square(tuple[int, int]):

	def __new__(cls,
		file: int,
		rank: int,
	) -> Square:
		square = super().__new__(cls, (file, rank))

		# Check if tuple of integers is within chess bounds:
		if not "a" <= square.file <= "h" or not "1" <= square.rank <= "8":
			raise IndexError(f"Invalid square {square}")

		return square



	def __repr__(self) -> str:
		return self.file + str(self.rank)


	@classmethod
	def fromnotation(cls, notation: str) -> Square:
		file, rank = notation

		if not "a" <= file <= "h" or not "1" <= rank <= "8":
			raise IndexError(f"Invalid square notation: {notation}")

		return super().__new__(cls, (int(rank) - 1, ord(file) - ord("a")))


	@property
	def rank(self) -> str:
		return str(self[0] + 1)

	@property
	def file(self) -> str:
		return chr(self[1] + ord("a"))

	@property
	def black(self) -> bool:
		return (self[0] + self[1]) % 2 == 0


	def __add__(self, other: Vector) -> Square:
		return Square(
			self[0] + other[0],
			self[1] + other[1],
		)

	def __sub__(self, other: Square) -> Vector:
		return Vector(
			self[0] - other[0],
			self[1] - other[1],
		)


class Vector(tuple[int, int]):

	def __new__(cls,
		file_diff: int,
		rank_diff: int,
	) -> Vector:
		return super().__new__(cls, (file_diff, rank_diff))























































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
