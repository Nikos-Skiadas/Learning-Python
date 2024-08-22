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
		side_0: float,
		side_1: float,
		side_2: float,
	):
		self.side_0 = side_0
		self.side_1 = side_1
		self.side_2 = side_2

	@property
	def perimeter(self) -> float:
		return self.side_0 + self.side_1 + self.side_2

	@property
	def area(self) -> float:
		return math.sqrt(self.perimeter
			* (self.perimeter - self.side_0)
			* (self.perimeter - self.side_1)
			* (self.perimeter - self.side_2)
		)


class Right(Triangle):
	def __init__(self, base: float, height: float):
		hypotenuse = math.sqrt(base ** 2 + height ** 2)
		super().__init__(base, height, hypotenuse)

	@property
	def area(self) -> float:
		return (self.side_0 * self.side_1) / 2


class Isosceles(Triangle):
	def __init__(self, equal_side: float, base: float):
		super().__init__(equal_side, equal_side, base)

	@property
	def height(self) -> float:
		return math.sqrt(self.side_0 ** 2 - (self.side_2 / 2) ** 2)  # from Pythagorean theorem

	@property
	def area(self) -> float:
		return (self.side_2 * self.height) / 2


class Equilateral(Isosceles):
	def __init__(self, side: float):
		super().__init__(side, side)

	@property
	def height(self) -> float:
		return (math.sqrt(3) / 2) * self.side_0

	@property
	def area(self) -> float:
		return (self.side_0 ** 2 * math.sqrt(3)) / 4


class IsoscelesRight(Isosceles, Right):
	def __init__(self, equal_side: float):
		base = equal_side * math.sqrt(2)
		super().__init__(equal_side, base)

	@property
	def area(self) -> float:
		return (self.side_0 ** 2) / 2
