"""EXERCISE:

Fill the classes below with
-	proper initializations,
-	perimeter and area methods
-	any other helping methods like the renaining sides or angles
-	whatever other method you think may better stand alone

Aims:
-	Make it functional, which means to give correct results.
-	Make it optimal, which means to truncate any trigonometric computations where applicable.
"""


from __future__ import annotations


import math


class Triangle:

	@property
	def perimeter(self) -> float:
		...

	@property
	def area(self) -> float:
		...


class Right(Triangle):

	...


class Isosceles(Triangle):

	...


class Equilateral(Isosceles):

	...


class IsoscelesRight(Isosceles, Right):

	...
