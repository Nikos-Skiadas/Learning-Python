from __future__ import annotations


import itertools
import enum
import re


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


class Vectors(Vector, enum.Enum):

	O = Vector( 0,  0)  # type: ignore
	N = Vector(+1,  0)  # type: ignore  # king queen rook pawn(white)
	S = Vector(-1,  0)  # type: ignore  # king queen rook
	E = Vector( 0, +1)  # type: ignore  # king queen rook pawn(black)
	W = Vector( 0, -1)  # type: ignore  # king queen rook

	N2 = N * 2  # type: ignore  # pawn(white leap)
	S2 = S * 2  # type: ignore  # pawn(black leap)
	E2 = E * 2  # type: ignore  # king(castle)
	W2 = W * 2  # type: ignore  # king(castle)
	E4 = E * 4  # type: ignore  # rook(castle)
	W3 = W * 3  # type: ignore  # rook(castle)

	NE = N + E  # type: ignore  # queen bishop pawn(white capture)
	SE = S + E  # type: ignore  # queen bishop pawn(black capture)
	SW = S + W  # type: ignore  # queen bishop pawn(black capture)
	NW = N + W  # type: ignore  # queen bishop pawn(white capture)

	N2E = N + NE  # type: ignore  # knight
	NE2 = NE + E  # type: ignore  # knight
	SE2 = SE + E  # type: ignore  # knight
	S2E = S + SE  # type: ignore  # knight
	S2W = S + SW  # type: ignore  # knight
	SW2 = SW + W  # type: ignore  # knight
	NW2 = NW + W  # type: ignore  # knight
	N2W = N + NW  # type: ignore  # knight


class Piece:

	value: int
	legal_steps: set[Vector]


	def __init__(self, color: str, position: str):
		color = color.lower()

		if color not in {
			"white",
			"black",
		}:
			raise ValueError("Piece must be either white or black")

		self.color = color
		self.position = Square.fromnotation(position)


class Melee(Piece):

	def legal_positions(self, friends: set[Square], foes: set[Square]) -> set[Square]:
		positions = set()

		for legal_step in self.legal_steps:
			try:
				next_position = self.position + legal_step

				if next_position in friends:
					continue

				positions.add(next_position)

			except IndexError:
				continue

		return positions


class Ranged(Piece):

	def legal_positions(self, friends: set[Square], foes: set[Square]) -> set[Square]:
		positions = set()

		for legal_step in self.legal_steps:
			for leap in range(1, 8):
				try:
					next_position = self.position + legal_step * leap

					if next_position in friends:
						break

					positions.add(next_position)

					if next_position in foes:
						break

				except IndexError:
					break

		return positions


class Pawn(Piece):

	value = 1

	legal_steps = {
		Vectors.N.value, Vectors.NE.value, Vectors.NW.value, Vectors.N2.value,
		Vectors.S.value, Vectors.SE.value, Vectors.SW.value, Vectors.S2.value,
	}


class Rook(Ranged):

	value = 5

	legal_steps = {
		Vectors.N.value,
		Vectors.S.value,
		Vectors.E.value,
		Vectors.W.value,
	}


class Bishop(Ranged):

	value = 3

	legal_steps = {
		Vectors.NE.value,
		Vectors.SE.value,
		Vectors.SW.value,
		Vectors.NW.value,
	}


class Knight(Melee):

	value = 3

	legal_steps = {
		Vectors.N2E.value,
		Vectors.NE2.value,
		Vectors.SE2.value,
		Vectors.S2E.value,
		Vectors.S2W.value,
		Vectors.SW2.value,
		Vectors.NW2.value,
		Vectors.N2W.value,
	}
#	legal_steps = {
#		straight + diagonal for straight, diagonal in itertools.product(
#			Rook.legal_steps,
#			Bishop.legal_steps,
#		)
#	} - Rook.legal_steps



class Queen(Ranged):

	value = 9  # Rook.value + Bishop.value + Pawn.value

	legal_steps = {
		Vectors.N.value, Vectors.NE.value,
		Vectors.S.value, Vectors.SE.value,
		Vectors.E.value, Vectors.SW.value,
		Vectors.W.value, Vectors.NW.value,
	}
#	legal_steps = Rook.legal_steps | Bishop.legal_steps


class King(Melee):

	value = 0  # TODO: figure out what to do with this value

	legal_steps = {
		Vectors.N.value, Vectors.NE.value,
		Vectors.S.value, Vectors.SE.value,
		Vectors.E.value, Vectors.SW.value,
		Vectors.W.value, Vectors.NW.value,

		Vectors.E2.value,
		Vectors.W2.value,
	}
#	legal_steps = Queen.legal_steps | {
#		Vectors.E2.value,
#		Vectors.W2.value,
#	}


"""HOMEWORK: CONTEXT

- We need a `Board` to connect all pieces and squares together. How?
- Perhaps a board is a list of lists of squares?
- Expand the `__init__` method below to perhaps initialize custom boards as well?
- Can we modiify the board? What is the easierst way? Lookup magic methods :
	`__setitem__`,
	`__getitem__`,
	`__delitem__`. (optionally)
- Feel free to implement how these methods work and are called.
- Lets not forget that we are here to play chess.
"""


class Board:

	def __init__(self):
		self.pieces: list[list[Piece | None]]
