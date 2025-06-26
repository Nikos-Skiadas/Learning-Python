from __future__ import annotations


import typing


@typing.runtime_checkable
class AlgebraProtocol(typing.Protocol):

	def __add__    (self, other): ...  # a + b
	def __sub__    (self, other): ...  # a - b
	def __mul__    (self, other): ...  # a * b
	def __truediv__(self, other): ...  # a / b

	def __pos__(self): ...  # +a
	def __neg__(self): ...  # -a


type Scalar = float


class Vector2(tuple[float, float]):

	def __add__(self, other: Vector2) -> Vector2:
		return Vector2(
			[
				self[0] + other[0],
				self[1] + other[1],
			]
		)


class Vector(tuple[Scalar, ...]):

	@property
	def dimension(self: Vector) -> int:
		return len(self)


	def __add__(self: Vector, other: Vector) -> Vector:
		assert self.dimension == other.dimension
		return Vector([left + right for left, right in zip(self, other)])

	def __sub__(self: Vector, other: Vector) -> Vector:
		return self + (-other)

	def __mul__(self: Vector, times: Scalar) -> Vector:
		return Vector([left * times for left in self])

	def __rmul__(self: Vector, times: Scalar) -> Vector:
		return self * times

	def __truediv__(self: Vector, times: Scalar) -> Vector:
		return self * (1 / times)

	def __pos__(self: Vector) -> Vector:
		return self

	def __neg__(self: Vector) -> Vector:
		return self * -1


if __name__ == "__main__":
	x = Vector([2,3,5])
	y = Vector([3,5,7])
