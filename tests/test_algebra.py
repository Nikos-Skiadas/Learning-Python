from __future__ import annotations


from deeplearning.code.algebra import *


class TestMatrix:

	x = Matrix[float](
		[
			[1., 2.],
			[3., 4.],
		]
	)
	y = Matrix[float](
		[
			[5., 6.],
			[7., 8.],
		]
	)

	def test_add(self):
		assert self.x + self.y == Matrix[float](
			[
				[ 6.,  8.],
				[10., 12.],
			]
		)

	def test_sub(self):
		assert self.x - self.y == Matrix[float](
			[
				[-4., -4.],
				[-4., -4.],
			]
		)

	def test_mul(self):
		assert self.x * 2 == Matrix[float](
			[
				[ 2.,  4.],
				[ 6.,  8.],
			]
		)

	def test_rmul(self):
		assert 2 * self.x == Matrix[float](
			[
				[ 2.,  4.],
				[ 6.,  8.],
			]
		)

	def test_truediv(self):
		assert self.x / 2 == Matrix[float](
			[
				[0.5, 1.],
				[1.5, 2.],
			]
		)

	def test_pos(self):
		assert +self.x == self.x

	def test_neg(self):
		assert -self.x == Matrix[float](
			[
				[-1., -2.],
				[-3., -4.],
			]
		)
