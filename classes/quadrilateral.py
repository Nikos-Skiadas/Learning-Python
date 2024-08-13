from __future__ import annotations


import math


class Quadrilateral:

	def __init__(self,
		side_0: float | int = 0.,
		side_1: float | int = 0.,
		side_2: float | int = 0.,

		angle_0: float | int = 0.,
		angle_1: float | int = 0.,
	):
		self.side_0 = float(side_0)
		self.side_1 = float(side_1)
		self.side_2 = float(side_2)

		self.angle_0 = float(angle_0)
		self.angle_1 = float(angle_1)

	def __repr__(self) -> str:
		return ":".join(vars(self).values())


	@property
	def side_3(self) -> float:
		trigonomatric = self.side_0 * math.cos(self.angle_1)

		return trigonomatric + math.sqrt((trigonomatric * trigonomatric - (
					self.side_1 * self.side_1 +
					self.side_2 * self.side_2 -
					self.side_0 * self.side_0 -
					self.side_1 * self.side_2 * math.cos(self.angle_1) * 2
				)
			)
		)

	@property
	def angle_2(self) -> float:
		return math.acos((
				self.side_0 * self.side_0 +
				self.side_1 * self.side_1 -
				self.side_2 * self.side_2 -
				self.side_3 * self.side_3 -
				self.side_0 * self.side_1 * 2 * math.cos(self.angle_0)
			) / self.side_2 / self.side_3 / 2
		)

	@property
	def angle_3(self) -> float:
		return 2 * math.pi - (self.angle_0 + self.angle_1 + self.angle_2)

	@property
	def perimeter(self) -> float:
		return self.side_0 + self.side_1 + self.side_2 + self.side_3

	@property
	def area(self) -> float:
		semiperimeter = self.perimeter / 2

		return math.sqrt(
			(semiperimeter / 2 - self.side_0) *
			(semiperimeter / 2 - self.side_1) *
			(semiperimeter / 2 - self.side_2) *
			(semiperimeter / 2 - self.side_3) +

			self.side_0 *
			self.side_1 *
			self.side_2 *
			self.side_3 * math.cos((self.angle_0 + self.angle_1) / 2)
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
	def angle_2(self) -> float:
		return math.pi - self.angle_1

	@property
	def side_3(self) -> float:
		return self.side_1 - self.side_0 * math.sin(self.angle_0) - self.side_2 * math.sin(self.angle_1)

	@property
	def area(self) -> float:
		return (self.side_1 + self.side_3) * self.side_0 * math.cos(self.angle_0) / 2


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
		return self.side_1 * self.side_0 * math.sin(self.angle_0)


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
		return self.side_0 * self.side_1


class Square(Rectangle):  # Rombus

	def __init__(self,
		side_0: float | int = 0.,
	):
		super().__init__(
			side_0 = side_0,
			side_1 = side_0,
		)


if __name__ == "__main__":
	x = Square(7)
