from __future__ import annotations


import os
import typing

from chess.algebra import Rank, File, Square, SIZE
from chess.material import Piece, Pawn, Rook, Bishop, Knight, Queen, King


class Board(list[Piece | None]):

	def __init__(self):
		super().__init__(None for _ in range(64))

	def __repr__(self) -> str:
		representation  = os.linesep
		representation += "   A B C D E F G H   " + os.linesep

		for rank in range(SIZE):
			representation += f" {Rank(rank)}"

			for file in range(SIZE):
				square = Square.from_rank_and_file(
					rank,
					file,
				)
				piece = self[square]

				representation += " "
				representation += repr(piece) if piece is not None else "-"

			representation += f" {Rank(rank)} " + os.linesep

		representation += "   A B C D E F G H   " + os.linesep

		return representation

	def __getitem__(self, square: str | Square) -> Piece | None:
		if isinstance(square, str):
			square = Square.from_notation(square)

		return super().__getitem__(square)

	def __setitem__(self, square: str | Square, piece: Piece | None):
		if isinstance(square, str):
			square = Square.from_notation(square)

		if piece is not None:
			piece.square = square

		del self[square]

		return super().__setitem__(square, piece)

	def __delitem__(self, square: str | Square):
		if isinstance(square, str):
			square = Square.from_notation(square)

		if (piece := self[square]) is not None:
			piece.square = None

		super().__setitem__(square, None)


	@classmethod
	def new_game(cls) -> typing.Self:
		board = cls()

		board["a8"] = Rook  ("black")
		board["b8"] = Knight("black")
		board["c8"] = Bishop("black")
		board["d8"] = Queen ("black")
		board["e8"] = King  ("black")
		board["f8"] = Bishop("black")
		board["g8"] = Knight("black")
		board["h8"] = Rook  ("black")

		for file in "ABCDEFGH".lower():
			board[Square.from_notation(file + "7")] = Pawn("black")
			board[Square.from_notation(file + "2")] = Pawn("white")

		board["a1"] = Rook  ("white")
		board["b1"] = Knight("white")
		board["c1"] = Bishop("white")
		board["d1"] = Queen ("white")
		board["e1"] = King  ("white")
		board["f1"] = Bishop("white")
		board["g1"] = Knight("white")
		board["h1"] = Rook  ("white")

		return board

	def move(self,
		source: Square,
		target: Square,
	):
		...
