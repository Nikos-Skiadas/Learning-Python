from __future__ import annotations

import math


class Triangle:

	def __init__(self,
		side_0: float | int = 0,
		side_1: float | int = 0,

		angle_0: float | int = 0,
	):
		self.side_0 = float(side_0)
		self.side_1 = float(side_1)

		self.angle_0 = float(angle_0)

	def __repr__(self) -> str:
		return ":".join(vars(self).values())


	@property
	def angle_1(self) -> float:
		...

	@property
	def angle_2(self) -> float:
		return math.pi - self.angle_0 + self.angle_1


	@property
	def side_2(self) -> float:
		return math.sqrt(
			self.side_0 * self.side_0 +
			self.side_1 * self.side_1 -
			self.side_0 * self.side_1 * math.cos(self.angle_0) * 2
		)

	@property
	def perimeter(self) -> float:
		return self.side_0 + self.side_1 + self.side_2

	@property
	def area(self) -> float:
		return self.side_0 * self.side_1 * math.cos(self.angle_0) / 2


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
	def side_2(self) -> float:
		return math.sqrt(
			self.side_0 * self.side_0 +
			self.side_1 * self.side_1
		)

	@property
	def area(self) -> float:
		return self.side_0 * self.side_1 / 2


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
	def side_2(self) -> float:
		return self.side_0 * math.sin(self.angle_0 / 2) * 2

	@property
	def area(self) -> float:
		return self.side_0 * self.side_0 * math.sin(self.angle_0) / 2


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
		return self.side_0 * self.side_0 * math.sqrt(3) / 4


class IsoscelesRight(Right):

	def __init__(self,
		side_0: float | int,
	):
		super().__init__(
			side_0,
			side_0,
		)
