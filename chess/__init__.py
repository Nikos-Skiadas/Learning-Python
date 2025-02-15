from __future__ import annotations


import enum


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


	def __add__(self, other: Vector) -> Vector:
		return Vector(
			self[0] + other[0],
			self[1] + other[1],
		)

	def __mul__(self, other: int) -> Vector:
		return Vector(
			self[0] * other,
			self[1] * other,
		)


class Vectors(enum.Enum, Vector):

	N = Vector(+1,  0)  # king queen rook pawn(white)
	E = Vector(-1,  0)  # king queen rook
	S = Vector( 0, +1)  # king queen rook pawn(black)
	W = Vector( 0, -1)  # king queen rook

	N2 = N * 2  # pawn(white leap)
	S2 = S * 2  # pawn(black leap)
	E2 = E * 2  # king(castle)
	W2 = W * 2  # king(castle)
	E4 = E * 4  # rook(castle)
	W3 = W * 3  # rook(castle)

	NE = N + E  # queen bishop pawn(white capture)
	SE = S + E  # queen bishop pawn(black capture)
	SW = S + W  # queen bishop pawn(black capture)
	NW = N + W  # queen bishop pawn(white capture)

	N2E = N + NE  # knight
	NE2 = NE + E  # knight
	SE2 = SE + E  # knight
	S2E = S + SE  # knight
	S2W = S + SW  # knight
	SW2 = SW + W  # knight
	NW2 = NW + W  # knight
	N2W = N + NW  # knight


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

	value = 1

	legal_moves = {
		Vectors.N, Vectors.NE, Vectors.NW, Vectors.N2,
		Vectors.S, Vectors.SE, Vectors.SW, Vectors.S2,
	}

class Rook(Piece):

	value = 5

	legal_moves = {
		*{Vectors.N * step for step in range(1, 8)},
		*{Vectors.S * step for step in range(1, 8)},
		*{Vectors.E * step for step in range(1, 8)},
		*{Vectors.W * step for step in range(1, 8)},
	}

class Bishop(Piece):

	value = 3

	legal_moves = {
		*{Vectors.NE * step for step in range(1, 8)},
		*{Vectors.SE * step for step in range(1, 8)},
		*{Vectors.SW * step for step in range(1, 8)},
		*{Vectors.NW * step for step in range(1, 8)},
	}

class Knight(Piece):

	value = 3

	legal_moves = {
		Vectors.N2E,
		Vectors.NE2,
		Vectors.SE2,
		Vectors.S2E,
		Vectors.S2W,
		Vectors.SW2,
		Vectors.NW2,
		Vectors.N2W,
	}


class Queen(Piece):

	value = 9

	legal_moves = Rook.legal_moves | Bishop.legal_moves


class King(Piece):

	value = 0  # TODO: figure out what to do with this value

	legal_moves = {
		Vectors.N, Vectors.NE,
		Vectors.S, Vectors.SE,
		Vectors.E, Vectors.SW,
		Vectors.W, Vectors.NW,

		Vectors.E2,
		Vectors.W2,
	}

















































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
