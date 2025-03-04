from __future__ import annotations


import typing

from chess.algebra import Square, Vectors

if typing.TYPE_CHECKING: from chess.engine import Board


class Piece:

	black: str
	white: str

	value: int
	steps: set[int]


	def __init__(self, color: str,
		square: Square | None = None,
	):
		color = color.lower()

		if color not in {
			"white",
			"black",
		}:
			raise ValueError("Piece must be either 'white' or 'black'")

		self.color = color
		self.square = square

	def __repr__(self) -> str:
		return self.black if self.color == "black" else self.white


class Melee(Piece):

	def squares(self, board: Board) -> set[Square]:
		squares = set()

		if self.square is not None:
			for step in self.steps:
				try:
					...

				except IndexError:
					continue

		return squares


class Ranged(Piece):

	def squares(self, board: Board) -> set[Square]:
		squares = set()

		if self.square is not None:
			for step in self.steps:
				while True:
					try:
						...

					except IndexError:
						break

		return squares


class Pawn(Piece):

	black: str = "\u265f"
	white: str = "\u2659"

	value = 1

	steps = {
		Vectors.N.value, Vectors.NE.value, Vectors.NW.value, Vectors.N2.value,
		Vectors.S.value, Vectors.SE.value, Vectors.SW.value, Vectors.S2.value,
	}


class Rook(Ranged):

	black: str = "\u265c"
	white: str = "\u2656"

	value = 5

	steps = {
		Vectors.N.value,
		Vectors.S.value,
		Vectors.E.value,
		Vectors.W.value,
	}


class Bishop(Ranged):

	black: str = "\u265d"
	white: str = "\u2657"

	value = 3

	steps = {
		Vectors.NE.value,
		Vectors.SE.value,
		Vectors.SW.value,
		Vectors.NW.value,
	}


class Knight(Melee):

	black: str = "\u265e"
	white: str = "\u2658"

	value = 3

	steps = {
		Vectors.N2E.value,
		Vectors.NE2.value,
		Vectors.SE2.value,
		Vectors.S2E.value,
		Vectors.S2W.value,
		Vectors.SW2.value,
		Vectors.NW2.value,
		Vectors.N2W.value,
	}
#	steps = {
#		straight + diagonal for straight, diagonal in itertools.product(
#			Rook.steps,
#			Bishop.steps,
#		)
#	} - Rook.steps



class Queen(Ranged):

	black: str = "\u265b"
	white: str = "\u2655"

	value = 9  # Rook.value + Bishop.value + Pawn.value

	steps = {
		Vectors.N.value, Vectors.NE.value,
		Vectors.S.value, Vectors.SE.value,
		Vectors.E.value, Vectors.SW.value,
		Vectors.W.value, Vectors.NW.value,
	}
#	steps = Rook.steps | Bishop.steps


class King(Melee):

	black: str = "\u265a"
	white: str = "\u2654"

	value = 0  # TODO: figure out what to do with this value

	steps = {
		Vectors.N.value, Vectors.NE.value,
		Vectors.S.value, Vectors.SE.value,
		Vectors.E.value, Vectors.SW.value,
		Vectors.W.value, Vectors.NW.value,

		Vectors.E2.value,
		Vectors.W2.value,
	}
#	steps = Queen.steps | {
#		Vectors.E2.value,
#		Vectors.W2.value,
#	}

