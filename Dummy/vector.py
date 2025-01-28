from __future__ import annotations

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
        return self.dim

    def __add__(self, other: Vector) -> Vector:
        return self.add(other)

    def __sub__(self, other: Vector) -> Vector:
        return self.sub(other)

    def __mul__(self, times: float) -> Vector:
        return self.mul(times)

    def __rmul__(self, times: float) -> Vector:
        return self.__mul__(times)

    def __truediv__(self, times: float) -> Vector:
        if times == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return Vector([x / times for x in self.components])

    def __matmul__(self, other: Vector) -> float:  # corresponds to "@"
        return NotImplemented

    def __abs__(self) -> float:
        return NotImplemented

    @property
    def dim(self) -> int:
        return len(self.components)

    def add(self, other: Vector) -> Vector:
        return Vector([x + y for x, y in zip(self.components, other.components)])

    def sub(self, other: Vector) -> Vector:
        return Vector([x - y for x, y in zip(self.components, other.components)])

    def mul(self, times: float) -> Vector:
        return Vector([x * times for x in self.components])

    def div(self, times: float) -> Vector:
        return self.__truediv__(times)

    def dot(self, other: Vector) -> float:
        return NotImplemented

    def norm(self) -> float:
        return NotImplemented

if __name__ == "__main__":
    x = Vector([
        1.,
        2.,
        3.,
        4.,
        5.,
    ])
    y = Vector([
        1.,
        2.,
        3.,
        4.,
        5.,
    ])
