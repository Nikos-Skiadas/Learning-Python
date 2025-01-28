from __future__ import annotations


import math


class Vector:

    # MAGIC METHODS

    def __init__(self, components: list[float]):
        for x in components:
            if not isinstance(x, float):
                raise TypeError("Not all given components are floats")

        self.components = components

    def __repr__(self) -> str:
        return str(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __add__(self, other: Vector) -> Vector:
        return Vector([x + y for x, y in zip(self.components, other.components)])

    def __sub__(self, other: Vector) -> Vector:
        return Vector([x - y for x, y in zip(self.components, other.components)])

    def __mul__(self, times: float) -> Vector:
        return Vector([x * times for x in self.components])

    def __rmul__(self, times: float) -> Vector:
        return self * times

    def __truediv__(self, times: float) -> Vector:
        return Vector([x / times for x in self.components])

    def __matmul__(self, other: Vector) -> float:  # corresponds to "@"
        return sum(x * y for x, y in zip(self.components, other.components))

    def __abs__(self) -> float:
        return math.sqrt(self @ self)


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
    print(f"||x||: {abs(x)}")
    print(f"||y||: {abs(y)}")
    print()
    print(f"dim x: {len(x)}")
    print(f"dim y: {len(y)}")
