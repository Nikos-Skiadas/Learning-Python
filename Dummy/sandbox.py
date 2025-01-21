from __future__ import annotations

from copy import copy
from math import sqrt

class complex:
	def __init__(self,
		real: float = 0
		imag: float = 0
	):
		self.real = real
		self.imag = imag

    def __repr__(self) -> str:
        return f"({self.real}, {self.imag})"

    def __add__(self, other: complex) -> complex:
        return complex(
            self.real + other.real,
            self.imag + other.imag,
        )

    def __sub__(self, other: complex) -> complex:
        return complex(
            self.real - other.real,
            self.imag - other.imag,
        )

    def __neg__(self) -> complex:
        return complex(
            -self.real,
            -self.imag,
        )

    def __pos__(self) -> complex:
        return copy(self)

    def __mul__(self, other: complex) -> complex:
        # Correct formula for complex multiplication: (a + ib) (c + id) = (ac - bd) + i(ad + bc)
        return complex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def __truediv__(self, other: complex) -> complex:
        # Correct formula for complex division
		# https://courses.lumenlearning.com/waymakercollegealgebra/chapter/divide-complex-numbers/
        denom = other.real ** 2 + other.imag ** 2
        if denom == 0:
            raise ZeroDivisionError("Cannot divide by zero in complex division.")
        return complex(
            (self.real * other.real + self.imag * other.imag) / denom,
            (self.imag * other.real - self.real * other.imag) / denom,
        )

    def __conj__(self) -> complex:
        # Fix for the conjugate method
        return complex(
            self.real,
            -self.imag,
        )

    def abs(self) -> float:
        return sqrt(self.real ** 2 + self.imag ** 2)