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

