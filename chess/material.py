from __future__ import annotations


from chess.algebra import Square, Vectors


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

	def squares(self, friends: set[Square], foes: set[Square]) -> set[Square]:
		squares = set()

		if self.square is not None:
			for step in self.steps:
				try:
					next_square = self.square + step

					if next_square in friends:
						continue

					squares.add(next_square)

				except IndexError:
					continue

		return squares


class Ranged(Piece):

	def squares(self, friends: set[Square], foes: set[Square]) -> set[Square]:
		squares = set()

		if self.square is not None:
			for step in self.steps:
				for leap in range(1, 8):
					try:
						next_square = self.square + step * leap

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

	steps = {
		Vectors.N.value, Vectors.NE.value, Vectors.NW.value, Vectors.N2.value,
		Vectors.S.value, Vectors.SE.value, Vectors.SW.value, Vectors.S2.value,
	}


class Rook(Ranged):

	black = "r"
	white = "R"

	value = 5

	steps = {
		Vectors.N.value,
		Vectors.S.value,
		Vectors.E.value,
		Vectors.W.value,
	}


class Bishop(Ranged):

	black = "b"
	white = "B"

	value = 3

	steps = {
		Vectors.NE.value,
		Vectors.SE.value,
		Vectors.SW.value,
		Vectors.NW.value,
	}


class Knight(Melee):

	black = "n"
	white = "N"

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

	black = "q"
	white = "Q"

	value = 9  # Rook.value + Bishop.value + Pawn.value

	steps = {
		Vectors.N.value, Vectors.NE.value,
		Vectors.S.value, Vectors.SE.value,
		Vectors.E.value, Vectors.SW.value,
		Vectors.W.value, Vectors.NW.value,
	}
#	steps = Rook.steps | Bishop.steps


class King(Melee):

	black = "k"
	white = "K"

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

