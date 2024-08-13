from __future__ import annotations

import math

from polygon import Scalene


class Triangle(Scalene):

	def __init__(self,
		side_0: float | int = 0,
		side_1: float | int = 0,

		angle_0: float | int = 0,
	):
		super().__init__(
			[
				side_0,
				side_1,
			],

			[
				angle_0,
			],
		)


	@property
	def angle_last(self) -> float:
		...

	@property
	def side_last(self) -> float:
		return math.sqrt(
			self.sides[0] * self.sides[0] +
			self.sides[1] * self.sides[1] -
			self.sides[0] * self.sides[1] * math.cos(self.angles[0]) * 2
		)

	@property
	def area(self) -> float:
		return self.sides[0] * self.sides[1] * math.cos(self.angles[0]) / 2


class Right(Triangle):

	def __init__(self,
		side_0: float | int = 0,
		side_1: float | int = 0,
	):
		super().__init__(
			side_0,
			side_1,

			math.pi / 2,
		)


	@property
	def side_last(self) -> float:
		return math.sqrt(
			self.sides[0] * self.sides[0] +
			self.sides[1] * self.sides[1]
		)

	@property
	def area(self) -> float:
		return self.sides[0] * self.sides[1] / 2


class Isosceles(Triangle):

	def __init__(self,
		side_0: float | int = 0,

		angle_0: float | int = 0,
	):
		super().__init__(
			side_0,
			side_0,

			angle_0,
		)


	@property
	def side_last(self) -> float:
		return self.sides[0] * math.sin(self.angles[0] / 2) * 2

	@property
	def area(self) -> float:
		return self.sides[0] * self.sides[0] * math.sin(self.angles[0]) / 2


class Equilateral(Isosceles):

	def __init__(self,
		side_0: float | int = 0,
	):
		super().__init__(
			side_0,

			math.pi / 3,
		)


	@property
	def area(self) -> float:
		return self.sides[0] * self.sides[0] * math.sqrt(3) / 4


class IsoscelesRight(Right):

	def __init__(self,
		side_0: float | int,
	):
		super().__init__(
			side_0,
			side_0,
		)
