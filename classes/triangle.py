from __future__ import annotations


"""EXERCISE:

Fill the classes below with
-	proper initializations,
-	perimeter and area methods
-	any other helping methods like the renaining sides or angles
-	whatever other method you think may better stand alone

Aims:
-	Make it functional, which means to give correct results.
-	Make it optimal, which means to truncate any trigonometric computations where applicable.

NOTE: The `super()` function in Python is used to give access to methods and properties of a parent or sibling class.
It is especially useful in a scenario involving class inheritance, where a class inherits from one or more base classes.
"""


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
