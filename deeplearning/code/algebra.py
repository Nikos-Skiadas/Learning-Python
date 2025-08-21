from __future__ import annotations


from copy import deepcopy
from math import sqrt
from typing import Literal, Protocol, Self, Iterable, runtime_checkable


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


	def __add__(self, other: Vector[F] | Literal[0], /) -> Self:
		if isinstance(other, int) and other == 0:
			return self

		assert self.dimension == other.dimension
		return self.__class__([left + right for left, right in zip(self, other)])

	def __radd__(self, other: Vector[F] | Literal[0], /) -> Self:
		return self + other

	def __sub__(self, other: Vector[F] | Literal[0], /) -> Self:
		if isinstance(other, int) and other == 0:
			return +self

		return self + (-other)

	def __rsub__(self, other: Vector[F] | Literal[0], /) -> Self:
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

	def __matmul__(self, other: Vector[F] | Matrix[F], /) -> F | Self:
		if isinstance(other, Matrix): return self.mat(other)
		else                        : return self.dot(other)

	def __abs__(self) -> float:
		return sqrt(sum(abs(left) ** 2 for left in self))


	def dot(self, other: Vector[F], /) -> F:
		assert self.dimension == other.dimension
		return sum(left * right for left, right in zip(self, other))  # type: ignore[return-value]

	def mat(self, other: Matrix[F], /) -> Self:
		return self.__class__(other.transpose.dot(self))

	def is_linear_combination_of(self, *others: Vector[F]) -> bool:
		return Matrix([self, *others]).rank <= len(others)

	def change_basis(self, *others: Vector[F]) -> Self:
		assert len(others) == self.dimension
		return self.mat(Matrix([*others]).inverse)


class Matrix[F: Ring](Vector[Vector[F]]):

	def __new__(cls,
		data: Iterable[Iterable[F]] | None = None,
	) -> Self:
		return super().__new__(cls, [Vector(row) for row in data] if data is not None else [])

	def __matmul__(self, other: Matrix[F] | Vector[F] | Literal[1], /) -> Self | Vector[F]:
		if isinstance(other, int) and other == 1:
			return self

		if isinstance(other, Matrix): return self.mat(other)
		else                        : return self.dot(other)


	def dot(self, other: Vector[F], /) -> Vector[F]:
		assert self.transpose.dimension == other.dimension
		return Vector(left.dot(other) for left in self)

	def mat(self, other: Matrix[F], /) -> Self:
		return self.__class__(left.mat(other) for left in self)


	@property
	def columns(self) -> int:
		return self.transpose.dimension

	@property
	def rank(self) -> int:
		A = [list(row) for row in self]  # make mutable copy
		num_rows, num_cols = len(A), len(A[0])

		rank = 0
		current_row = 0

		for current_col in range(current_row, num_cols):
			pivot = None

		#	Find pivot row
			for row in range(current_row, num_rows):
				if A[row][current_col] != 0:
					pivot = row

					break

			if pivot is None:
				continue  # no pivot in this column

		#	Swap to current row:
			A[current_row], A[pivot] = A[pivot], A[current_row]

		#	Normalize pivot row:
			pivot_normalizer = A[current_row][current_col]
			A[current_row] = [value / pivot_normalizer for value in A[current_row]]

		#	Eliminate below:
			for row in range(current_row + 1, num_rows):
				if A[row][current_col] != 0:
					factor = A[row][current_col]
					A[row] = [rv - factor * lv for rv, lv in zip(A[row], A[current_row])]

			rank += 1
			current_row += 1

			if current_row == num_rows:
				break

		return rank

	@property
	def transpose(self) -> Self:
		return self.__class__(zip(*self))

	@property
	def trace(self) -> F:
		assert self.dimension == self[0].dimension
		return sum(left[i] for i, left in enumerate(self))  # type: ignore[return-value]

	@property
	def inverse(self) -> Self:
		assert self.dimension == self[0].dimension
		assert self.rank == self.dimension

		return NotImplemented


class Tensor[F: Ring](Vector[Matrix[F]]):

	...
