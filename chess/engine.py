from __future__ import annotations


from chess.material import Piece


class Board(list[Piece | None]):

	def __init__(self):
		super().__init__([None] * 64)
