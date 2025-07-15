from __future__ import annotations


from math import sqrt
from typing import Literal, Protocol, Self, Sequence, runtime_checkable


type Number = int | float | complex


@runtime_checkable
class Ring(Protocol):
	def  __add__    (self, other, /) -> Self: ...
	def __radd__    (self, other, /) -> Self: ...
	def  __sub__    (self, other, /) -> Self: ...
	def  __mul__    (self, other, /) -> Self: ...
	def __rmul__    (self, other, /) -> Self: ...
	def  __truediv__(self, other, /) -> Self: ...

	def __pos__(self) -> Self: ...
	def __neg__(self) -> Self: ...

	def __abs__(self) -> float: ...


class Scalar(float):

	def  __add__    (self, other: Number, /) -> Self: return self.__class__(self + other)
	def  __sub__    (self, other: Number, /) -> Self: return self.__class__(self - other)
	def  __mul__    (self, other: Number, /) -> Self: return self.__class__(self * other)
	def  __truediv__(self, other: Number, /) -> Self: return self.__class__(self / other)

	def __radd__    (self, other: Number, /) -> Self: return self.__class__(self + other)
	def __rsub__    (self, other: Number, /) -> Self: return self.__class__(self - other)
	def __rmul__    (self, other: Number, /) -> Self: return self.__class__(self * other)
	def __rtruediv__(self, other: Number, /) -> Self: return self.__class__(self / other)

	def __pos__(self) -> Self: return self.__class__(+self)
	def __neg__(self) -> Self: return self.__class__(-self)


class Vector[F: Ring](tuple[F, ...]):

	@property
	def dimension(self) -> int:
		return len(self)


	def __add__(self, other: Self | Literal[0], /) -> Self:
		if isinstance(other, int) and other == 0:
			return self

		assert self.dimension == other.dimension
		return self.__class__([left + right for left, right in zip(self, other)])

	def __radd__(self, other: Self | Literal[0], /) -> Self:
		return self + other

	def __sub__(self, other: Self | Literal[0], /) -> Self:
		if isinstance(other, int) and other == 0:
			return +self

		return self + (-other)

	def __rsub__(self, other: Self | Literal[0], /) -> Self:
		return (-self) + other

	def __mul__(self, times: Number, /) -> Self:
		return self.__class__([left * times for left in self])

	def __rmul__(self, times: Number, /) -> Self:
		return self * times

	def __truediv__(self, times: Number, /) -> Self:
		return (1 / times) * self

	def __pos__(self) -> Self:
		return self

	def __neg__(self) -> Self:
		return self * -1

	def __matmul__(self, other: Self, /) -> F:
		assert self.dimension == other.dimension
		return sum(left * right for left, right in zip(self, other))  # type: ignore[return-value]

	def __abs__(self) -> float:
		return sqrt(sum(abs(left) ** 2 for left in self))


class Matrix[F: Ring](Vector[Vector[F]]):

	def __new__(cls,
		data: Sequence[Sequence[F]] | None = None,
	) -> Self:
		return super().__new__(cls, [Vector(row) for row in data] if data is not None else [])

	def __matmul__(self, other: Self | Literal[1], /) -> Self:
		if isinstance(other, int) and other == 1:
			return self

		other = other.transpose

		assert self.dimension == other.dimension
		return self.__class__([[left @ right for right in other] for left in self])

	def __rmatmul__(self, other: Self | Literal[1], /) -> Self:
		return self @ other


	@property
	def transpose(self) -> Self:
		return self.__class__(tuple(zip(*self)))

	@property
	def trace(self) -> F:
		assert self.dimension == self[0].dimension
		return sum(self[i][i] for i in range(self.dimension))  # type: ignore[return-value]
		return sum(row[i] for i, row in enumerate(self))  # type: ignore[return-value]


class Tensor[F: Ring](Vector[Matrix[F]]):

	...
