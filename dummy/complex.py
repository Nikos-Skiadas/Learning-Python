from __future__ import annotations

from copy import copy
from math import sqrt

class Complex:

    zero: Complex


    def __init__(self,
        real: float = 0,
        imag: float = 0,
    ):
        self.real = real
        self.imag = imag

    def __repr__(self) -> str:
        return f"({self.real}, {self.imag})"


    def __add__(self, other: Complex) -> Complex:
        return Complex(
            self.real + other.real,
            self.imag + other.imag,
        )

    def __sub__(self, other: Complex) -> Complex:
        return Complex(
            self.real - other.real,
            self.imag - other.imag,
        )

    def __neg__(self) -> Complex:
        return Complex.zero - self

    def __pos__(self) -> Complex:
        return copy(self)

    def __mul__(self, other: Complex) -> Complex:
        # Correct formula for complex multiplication: (a + ib) (c + id) = (ac - bd) + i(ad + bc)
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def __truediv__(self, other: Complex) -> Complex:
        numer = other @ self
        denom = self.norm

        return Complex(
            numer.real / denom,
            numer.imag / denom,
        )

    def __matmul__(self, other: Complex) -> Complex:
        return self.conjugate() * other

    def __abs__(self) -> float:
        return sqrt(self.norm)


    def conjugate(self) -> Complex:
        # Fix for the conjugate method
        return Complex(
            +self.real,
            -self.imag,
        )
    @property
    def norm(self) -> float:
        return (self @ self).real  # self.real ** 2 + self.imag ** 2


Complex.zero = Complex()


# να φτιαξω μια αντικειμενο που να ειναι vector απο n πραγματικουσ αριθμους, προσθεση απο vectors, αφαιρεση προσημο, εσωτερικο γινομενο,
# και μια δευτερη κλαση που λεγεται matrix που να δεχεται πραξεις για πινακες, προσθεση, trace, πολλαπλασιαμο κτλ