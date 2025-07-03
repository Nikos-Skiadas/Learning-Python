from __future__ import annotations


from typing import Literal, Protocol, Self, Sequence, runtime_checkable


type Number = int | float


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


x = Vector[float]([1., 2., 3.])


class Matrix[F: Ring](Vector[Vector[F]]):

	def __new__(cls, data: Sequence[Sequence[F]]) -> Self:
		return super().__new__(cls, [Vector(row) for row in data])
