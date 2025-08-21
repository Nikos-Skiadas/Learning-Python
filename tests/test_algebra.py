from __future__ import annotations


from deeplearning.code.algebra import *

import numpy


_A = numpy.random.rand(2, 3); A = Matrix(_A)
_B = numpy.random.rand(3, 2); B = Matrix(_B)

_x = numpy.random.rand(2); x = Vector(_x)
_y = numpy.random.rand(3); y = Vector(_y)

_C = numpy.random.rand(2, 2); C = SquareMatrix(_C)
_D = numpy.random.rand(3, 3); D = SquareMatrix(_D)


def equal(a, b):
	return numpy.allclose(
		numpy.array(a),
		numpy.array(b),
	)


class TestVector:

	def test_add(self):
		assert equal(x + x, _x + _x)
		assert equal(y + y, _y + _y)

	def test_sub(self):
		assert equal(x - x, _x - _x)
		assert equal(y - y, _y - _y)

	def test_mul(self):
		assert equal(x * 2, _x *  2)
		assert equal(2 * x,  2 * _x)
		assert equal(y * 3, _y *  3)
		assert equal(3 * y,  3 * _y)

	def test_div(self):
		assert equal(x / 2, _x /  2)
		assert equal(y / 3, _y /  3)

	def test_abs(self):
		assert abs(x) == numpy.linalg.norm(_x)
		assert abs(y) == numpy.linalg.norm(_y)

	def	test_matmul(self):
		assert equal(x @ x, _x @ _x)
		assert equal(y @ y, _y @ _y)

		assert equal(x @ A, _x @ _A)
		assert equal(y @ B, _y @ _B)
		assert equal(x @ C, _x @ _C)
		assert equal(y @ D, _y @ _D)

class TestMatrix:

	def test_add(self):
		assert equal(A + A, _A + _A)
		assert equal(B + B, _B + _B)
		assert equal(C + C, _C + _C)
		assert equal(D + D, _D + _D)

	def test_sub(self):
		assert equal(A - A, _A - _A)
		assert equal(B - B, _B - _B)
		assert equal(C - C, _C - _C)
		assert equal(D - D, _D - _D)

	def test_mul(self):
		assert equal(A * 2, _A *  2)
		assert equal(2 * A,  2 * _A)
		assert equal(B * 3, _B *  3)
		assert equal(3 * B,  3 * _B)
		assert equal(C * 2, _C *  2)
		assert equal(2 * C,  2 * _C)
		assert equal(D * 3, _D *  3)
		assert equal(3 * D,  3 * _D)

	def test_div(self):
		assert equal(A / 2, _A / 2)
		assert equal(B / 3, _B / 3)
		assert equal(C / 2, _C / 2)
		assert equal(D / 3, _D / 3)

	def test_abs(self):
		assert abs(A) == numpy.linalg.norm(_A)
		assert abs(B) == numpy.linalg.norm(_B)
		assert abs(C) == numpy.linalg.norm(_C)
		assert abs(D) == numpy.linalg.norm(_D)

	def test_matmul(self):
		assert equal(A @ y, _A @ _y)
		assert equal(B @ x, _B @ _x)
		assert equal(C @ x, _C @ _x)
		assert equal(D @ y, _D @ _y)

		assert equal(A @ B, _A @ _B)
		assert equal(B @ A, _B @ _A)


class TestSquareMatrix:

	def test_add(self):
		assert equal(C + C, _C + _C)
		assert equal(D + D, _D + _D)

	def test_sub(self):
		assert equal(C - C, _C - _C)
		assert equal(D - D, _D - _D)

	def test_mul(self):
		assert equal(C * 2, _C *  2)
		assert equal(2 * C,  2 * _C)
		assert equal(D * 3, _D *  3)
		assert equal(3 * D,  3 * _D)

	def test_div(self):
		assert equal(C / 2, _C /  2)
		assert equal(D / 3, _D /  3)

	def test_abs(self):
		assert abs(C) == numpy.linalg.norm(_C)
		assert abs(D) == numpy.linalg.norm(_D)

	def test_matmul(self):
		assert equal(C @ x, _C @ _x)
		assert equal(D @ y, _D @ _y)

		assert equal(C @ C, _C @ _C)
		assert equal(D @ D, _D @ _D)

	def test_trace(self):
		assert C.trace == numpy.trace(_C)
		assert D.trace == numpy.trace(_D)

#	def test_inverse(self):
#		assert equal(C.inverse, numpy.linalg.inv(_C))
#		assert equal(D.inverse, numpy.linalg.inv(_D))

#	def test_determinant(self):
#		assert C.determinant == numpy.linalg.det(_C)
#		assert D.determinant == numpy.linalg.det(_D)
