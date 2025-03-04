from __future__ import annotations


import enum
import typing


SIZE = 8


class Index(int):

	bound: int


	def __init_subclass__(cls, *arg,
		bound: int,
	**kwargs):
		super().__init_subclass__(*arg, **kwargs)

		cls.bound = bound

	def __new__(cls, index: int) -> typing.Self:
		if not 0 <= (index := super().__new__(cls, index)) < cls.bound:
			raise IndexError(f"invalid {cls.__name__.lower()} '{index}'")

		return index


class Rank(Index,
	bound = SIZE,
):

	def __repr__(self) -> str:
		return str(8 - self)


	@classmethod
	def from_notation(cls, rank: str) -> typing.Self:
		return cls(8 - int(rank))


class File(Index,
	bound = SIZE,
):

	def __repr__(self) -> str:
		return chr(self + ord("a"))


	@classmethod
	def from_notation(cls, file: str) -> typing.Self:
		return cls(ord(file) - ord("a"))


class Square(Index,
	bound = SIZE * SIZE,
):

	def __repr__(self) -> str:
		return repr(self.file) + repr(self.rank)

	def __add__(self, other: int) -> Square: return Square(super().__add__(other))
	def __sub__(self, other: Square) -> int: return        super().__sub__(other)


	@classmethod
	def from_rank_and_file(cls,
		rank: int,
		file: int,
	) -> typing.Self:
		return cls(rank * SIZE + file)

	@classmethod
	def from_notation(cls, square: str) -> typing.Self:
		file, rank = square

		return cls.from_rank_and_file(
			Rank.from_notation(rank),
			File.from_notation(file),
		)


	@property
	def rank(self) -> Rank:
		return Rank(self // 8)

	@property
	def file(self) -> File:
		return File(self % 8)

	@property
	def is_black(self) -> bool:
		return (self.rank + self.file) % 2 == 0


class Vectors(int, enum.Enum):

	O =  0
	N = +1 * SIZE  # king queen rook pawn(white)
	S = -1 * SIZE  # king queen rook
	E = +1  # king queen rook pawn(black)
	W = -1  # king queen rook

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

