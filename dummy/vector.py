from __future__ import annotations


import math
import typing


class Vector(tuple[float, ...]):

    def __add__(self, other: Vector) -> Vector:
        return Vector([x + y for x, y in zip(self, other)])

    def __sub__(self, other: Vector) -> Vector:
        return Vector([x - y for x, y in zip(self, other)])

    def __mul__(self, times: float) -> Vector:
        return Vector([x * times for x in self])

    def __rmul__(self, times: float) -> Vector:
        return self * times

    def __truediv__(self, times: float) -> Vector:
        return Vector([x / times for x in self])

    def __matmul__(self, other: Vector) -> float:  # corresponds to "@"
        return sum(x * y for x, y in zip(self, other))

    def __abs__(self) -> float:
        return math.sqrt(self @ self)


    @property
    def dim(self) -> int:
        return len(self)

    @property
    def norm(self) -> float:
        return abs(self)


if __name__ == "__main__":
    x = Vector(
        [
            2.,
            3.,
            5.,
            7.,
        ]
    )
    y = Vector(
        [
            4.,
            6.,
            8.,
            9.,
        ]
    )
    a = 10.

    print(f"x    : {x    }")
    print(f"    y: {    y}")
    print(f"x + y: {x + y}")
    print(f"x - y: {x - y}")
    print(f"x @ y: {x @ y}")
    print(f"a * y: {a * y}")
    print(f"x * a: {x * a}")
    print()
    print(f"||x||: {abs(x)} == {x.norm}")
    print(f"||y||: {abs(y)} == {y.norm}")
    print()
    print(f"dim x: {len(x)} == {x.dim}")
    print(f"dim y: {len(y)} == {y.dim}")
