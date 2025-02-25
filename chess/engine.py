from __future__ import annotations


from chess.material import Piece


class Board(list[list[Piece | None]]):

	def __init__(self):
		super().__init__([None] * 8 for _ in range(8))
