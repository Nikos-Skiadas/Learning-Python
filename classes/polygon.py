from __future__ import annotations

import math


class Scalene:

	def __init__(self,
		sides: list[float | int],

		angles: list[float | int],
	):
		self.sides = list(map(float, sides))

		self.angles = list(map(float, angles))

	def __repr__(self) -> str:
		return ":".join(vars(self).values())


	@property
	def order(self) -> int:
		return len(self.sides) + 1

	@property
	def side_last(self) -> float:
		...

	@property
	def angle_last(self) -> float:
		...

	@property
	def angle_remaining(self) -> float:
		return len(self.angles) * math.pi - sum(self.angles)

	@property
	def perimeter(self) -> float:
		return sum(self.sides) + self.side_last

	@property
	def area(self) -> float:
		...


class Regular(Scalene):

	def __init__(self,
		order: int,
		side_0: float | int = 0,
	):
		super().__init__(
			[side_0] * (order - 1),

			[(order - 2) / order * math.pi] * (order - 2),
		)

	@property
	def side_last(self) -> float:
		return self.sides[0]

	@property
	def angle_last(self) -> float:
		return self.angles[0]

	@property
	def angle_remaining(self) -> float:
		return self.angles[0]

	@property
	def perimeter(self) -> float:
		return self.sides[0] * self.order

	@property
	def area(self) -> float:
		...
