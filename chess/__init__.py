from __future__ import annotations


import typing


class Square(tuple[int, int]):

	def __new__(cls,
		rank: int,
		file: int,
	) -> Square:
		square = super().__new__(cls, (rank, file))

		# Check if tuple of integers is within chess bounds:
		if not "a" <= square.file <= "h" or not "1" <= square.rank <= "8":
			raise IndexError(f"invalid square {square}")

		return square

	def __repr__(self) -> str:
		return self.file + str(self.rank)


	@classmethod
	def fromnotation(cls, notation: str) -> Square:
		file, rank = notation

		return cls(
			int(rank) - 1,
			ord(file) - ord("a"),
		)


	@property
	def rank(self) -> str:
		return str(self[0] + 1)

	@property
	def file(self) -> str:
		return chr(self[1] + ord("a"))

	@property
	def is_black(self) -> bool:
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


class Piece:

	value: int
	legal_moves: set[Vector]


	def __init__(self, color: str, position: str):
		color = color.lower()

		if color not in {
			"white",
			"black",
		}:
			raise ValueError("Piece must be either white or black")

		self.is_black = color == "black"
		self.position = Square.fromnotation(position)


	def legal_positions(self) -> set[Square]:
		positions = set()

		for legal_move in self.legal_moves:
			try:
				positions.add(self.position + legal_move)

			except IndexError:
				continue

		return positions


"""HOMEWORK FOR NEXT WEEK:

EASY:
-	Fill values for all other piece classes apart from `King`.
-	Fill in the `legal_moves` of `King`.

HARD:
-	How do you handle the value of `King`? Mathematically speaking, the king's value is infinite.
-	What is the major difference between `{Bishop, Rook, Queen}` and `{Knight, King}` in how they move?
	NOTE: `Ranged` vs `Melee` pieces
-	What could be the `legal_moves` of other pieces?
-	What could be the `legal_positions` of other pieces? The `Ranged` ones in particular?

HINTS:
-	Use inheritance to its fullest!
-	Feel free to redisgin the classes, and possibly add new ones!
"""



class Pawn(Piece):

	...


class Rook(Piece):

	...


class Bishop(Piece):

	...


class Knight(Piece):

	value = 3
	legal_moves = {
		Vector(+1, +2),
		Vector(+2, +1),
		Vector(+2, -1),
		Vector(+1, -2),
		Vector(-1, -2),
		Vector(-2, -1),
		Vector(-2, +1),
		Vector(-1, +2),
	}


class Queen(Piece):

	...


class King(Piece):


#	TODO: What shall we use for value of king?

	...



















































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


class History:

	...


class Move:

	...
