from __future__ import annotations

import math

from polygon import Scalene


class Quadrilateral(Scalene):

	def __init__(self,
		side_0: float | int = 0.,
		side_1: float | int = 0.,
		side_2: float | int = 0.,

		angle_0: float | int = 0.,
		angle_1: float | int = 0.,
	):
		super().__init__(
			[
				side_0,
				side_1,
				side_2,
			],

			[
				angle_0,
				angle_1,
			],
		)


class Trapezoid(Quadrilateral):

	def __init__(self,
		side_0: float | int = 0.,
		side_1: float | int = 0.,
		side_2: float | int = 0.,

		angle_0: float | int = 0.,
	):
		super().__init__(
			side_0 = side_0,
			side_1 = side_1,
			side_2 = side_2,

			angle_0 = angle_0,
			angle_1 = math.atan(side_0 * math.sin(angle_0) / (side_2 - side_0 * math.cos(angle_0))),
		)


	@property
	def side_last(self) -> float:
		return self.sides[1] - self.sides[1] * math.sin(self.angles[0]) - self.sides[2] * math.sin(self.angles[1])

	@property
	def angle_last(self) -> float:
		return math.pi - self.angle_last

	@property
	def area(self) -> float:
		return (self.sides[1] + self.side_last) * self.sides[0] * math.cos(self.angles[0]) / 2


class Parallelogram(Trapezoid):

	def __init__(self,
		side_0: float | int = 0.,
		side_1: float | int = 0.,

		angle_0: float | int = 0.,
	):
		super().__init__(
			side_0 = side_0,
			side_1 = side_1,
			side_2 = side_0,

			angle_0 = angle_0,
		)


	@property
	def area(self) -> float:
		return self.sides[1] * self.side_last * math.sin(self.angle_remaining)


class Rombus(Parallelogram):

	def __init__(self,
		side_0: float | int = 0.,

		angle_0: float | int = 0.,
	):
		super().__init__(
			side_0 = side_0,
			side_1 = side_0,

			angle_0 = angle_0,
		)


class Rectangle(Parallelogram):

	def __init__(self,
		side_0: float | int = 0.,
		side_1: float | int = 0.,
	):
		super().__init__(
			side_0 = side_0,
			side_1 = side_1,

			angle_0 = math.pi / 2,
		)

	@property
	def area(self) -> float:
		return self.side_last * self.sides[1]


class Square(Rectangle):  # Rombus

	def __init__(self,
		side_0: float | int = 0.,
	):
		super().__init__(
			side_0 = side_0,
			side_1 = side_0,
		)
