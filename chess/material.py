from __future__ import annotations


from chess.algebra import Square, Vectors


class Piece:

	black: str
	white: str

	value: int
	steps: set[int]


	def __init__(self, color: str, square: str):
		color = color.lower()

		if color not in {
			"white",
			"black",
		}:
			raise ValueError("Piece must be either white or black")

		self.color = color
		self.square = Square.fromnotation(square)

	def __repr__(self) -> str:
		return self.black if self.color == "black" else self.white


class Melee(Piece):

	def squares(self, friends: set[Square], foes: set[Square]) -> set[Square]:
		squares = set()

		for legal_step in self.steps:
			try:
				next_square = self.square + legal_step

				if next_square in friends:
					continue

				squares.add(next_square)

			except IndexError:
				continue

		return squares


class Ranged(Piece):

	def squares(self, friends: set[Square], foes: set[Square]) -> set[Square]:
		squares = set()

		for legal_step in self.steps:
			for leap in range(1, 8):
				try:
					next_square = self.square + legal_step * leap

					if next_square in friends:
						break

					squares.add(next_square)

					if next_square in foes:
						break

				except IndexError:
					break

		return squares


class Pawn(Piece):

	black = "p"
	white = "P"

	value = 1

	legal_steps = {
		Vectors.N.value, Vectors.NE.value, Vectors.NW.value, Vectors.N2.value,
		Vectors.S.value, Vectors.SE.value, Vectors.SW.value, Vectors.S2.value,
	}


class Rook(Ranged):

	black = "r"
	white = "R"

	value = 5

	legal_steps = {
		Vectors.N.value,
		Vectors.S.value,
		Vectors.E.value,
		Vectors.W.value,
	}


class Bishop(Ranged):

	black = "b"
	white = "B"

	value = 3

	legal_steps = {
		Vectors.NE.value,
		Vectors.SE.value,
		Vectors.SW.value,
		Vectors.NW.value,
	}


class Knight(Melee):

	black = "n"
	white = "N"

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

	black = "q"
	white = "Q"

	value = 9  # Rook.value + Bishop.value + Pawn.value

	legal_steps = {
		Vectors.N.value, Vectors.NE.value,
		Vectors.S.value, Vectors.SE.value,
		Vectors.E.value, Vectors.SW.value,
		Vectors.W.value, Vectors.NW.value,
	}
#	legal_steps = Rook.legal_steps | Bishop.legal_steps


class King(Melee):

	black = "k"
	white = "K"

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

