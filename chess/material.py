from __future__ import annotations


from chess.algebra import Vector, Square, Vectors


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

