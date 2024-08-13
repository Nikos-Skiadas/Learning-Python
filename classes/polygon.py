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
	def side_0(self) -> float:
		...

	@property
	def angle_1(self) -> float:
		...

	@property
	def angle_0(self) -> float:
		return (self.order - 2) * math.pi - sum(self.angles)

	@property
	def perimeter(self) -> float:
		return sum(self.sides) + self.side_0

	@property
	def area(self) -> float:
		...
