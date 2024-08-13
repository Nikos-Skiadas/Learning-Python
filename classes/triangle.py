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



"""
The super() function in Python is used to give access to methods and properties of a parent or sibling class. 
It is especially useful in a scenario involving class inheritance, where a class inherits from one or more base classes
"""

from __future__ import annotations


import math


class Triangle:
    def __init__(self, side_a: float, side_b: float, side_c: float):
        self.side_a = side_a
        self.side_b = side_b
        self.side_c = side_c

    @property
    def perimeter(self) -> float:
        return self.side_a + self.side_b + self.side_c

    @property
    def area(self) -> float:
        s = self.perimeter / 2
        return math.sqrt(s * (s - self.side_a) * (s - self.side_b) * (s - self.side_c))


class Right(Triangle):
    def __init__(self, base: float, height: float):
        hypotenuse = math.sqrt(base**2 + height**2)
        super().__init__(base, height, hypotenuse)

    @property
    def area(self) -> float:
        return (self.side_a * self.side_b) / 2


class Isosceles(Triangle):
    def __init__(self, equal_side: float, base: float):
        super().__init__(equal_side, equal_side, base)

    @property
    def height(self) -> float:
        return math.sqrt(self.side_a**2 - (self.side_c / 2)**2) # From Pythagorean theorem

    @property
    def area(self) -> float:
        return (self.side_c * self.height) / 2


class Equilateral(Isosceles):
    def __init__(self, side: float):
        super().__init__(side, side)

    @property
    def height(self) -> float:
        return (math.sqrt(3) / 2) * self.side_a

    @property
    def area(self) -> float:
        return (self.side_a**2 * math.sqrt(3)) / 4


class IsoscelesRight(Isosceles, Right):
    def __init__(self, equal_side: float):
        base = equal_side * math.sqrt(2)   ????
        super().__init__(equal_side, base)

    @property
    def area(self) -> float:
        return (self.side_a**2)
