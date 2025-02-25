from __future__ import annotations


import os

from chess.algebra import Rank, File, Square, SIZE
from chess.material import Piece


class Board(list[Piece | None]):

	def __init__(self):
		super().__init__(None for _ in range(64))

	def __repr__(self) -> str:
		representation  = os.linesep
		representation += "   A B C D E F G H   " + os.linesep

		for rank in range(SIZE):
			representation += f" {Rank(rank)}"

			for file in range(SIZE):
				square = Square(rank * 8 + file)
				piece = self[square]

				representation += " "
				representation += repr(piece) if piece is not None else "-"

			representation += f" {Rank(rank)} " + os.linesep

		representation += "   A B C D E F G H   " + os.linesep

		return representation

	def __getitem__(self, square: str | Square) -> Piece | None:
		if isinstance(square, str):
			square = Square.fromnotation(square)

		return super().__getitem__(square)

	def __setitem__(self, square: str | Square, piece: Piece | None):
		if isinstance(square, str):
			square = Square.fromnotation(square)

		if piece is not None:
			piece.square = square

		del self[square]

		return super().__setitem__(square, piece)

	def __delitem__(self, square: str | Square):
		if isinstance(square, str):
			square = Square.fromnotation(square)

		if (piece := self[square]) is not None:
			piece.square = None

		self[square] = None
