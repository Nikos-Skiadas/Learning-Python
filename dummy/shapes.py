from __future__ import annotations


import math
import typing


# DO NOT EDIT --------------------------------------------------------------------------------------------------------------------

class Shape(typing.Protocol):

	def __add__(self, other: typing.Self) -> typing.Self: ...
	def __mul__(self, other: typing.Self) -> typing.Self: ...


	@property
	def perimeter(self):
		...

	@property
	def area(self):
		...


# DO EDIT ------------------------------------------------------------------------------------------------------------------------

"""Solve each bullet point and commit it before moving on the next:

Basic:
-	Implement an `__init__` for each of the following shape types, by adding as many attributes needed to describe it. (HAS)
	-	For examples length of sides, angles or whatever is needed.
-	Implement the two properties `perimeter` and `area` for as many shapes below as you can.
	-	Some more generic shapes may have a very difficult formula, do not get stuck to it.
-	Think about how would `+` or `*` work for shapes. Then implement it for every shape.

Advanced:
-	Are any of the classes related by inheritance? Which is which? Redesign your solution by exploiting inheritance. (IS)
	-	Make sure you fully exploit inheritance.
		When inherting from another class, you get everything from that other class. Use it!
-	Can you think of new classes that may extend what we got here?
	-	For example a shape with 5 sides? What if I want 6 sides? Or more...
	-	How are more generic classes affecting your current implementation this far? Do you have to change a lot or nothing?
-	If you think of a shape not included, feel free to add it!
"""


class Triangle:

	...


class Isosceles:

	...


class Equilateral:

	...


class Orthogonal:

	...


class Quadrangle:

	...


class Trapezoid:

	...


class Parallelogram:

	...


class Rombus:

	...


class Rectangle:

	...


class Square:

	...
