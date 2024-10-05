"""An implementation of 3-dimensioan vectors:

Lets have some details about it!

Classes:
	Vector3: A 3-dimensional vector supporting basic operations.

Functions:
	add: Add two 3-dimensional vectors.
	dot: Get the dot product of two vectors.
"""


import math


class Vector3:

	"""A 3-dimensional vector supporting basic operations.

	Attributes:
		x: ...
		y: ...
		z: ...

	Methods:
		norm: Get the norm of a vector.
	"""

	def __init__(self,
		x: float,
		y: float,
		z: float,
	):
		self.x = x
		self.y = y
		self.z = z

	def norm(self) -> float:
		"""Get the norm of a vector.

		The norm is the square root of the dor product of the vector with itself

		Returns:
			A `float` value as the norm of this vector.
		"""
		return math.sqrt(dot(self, self))


def add(
	a: Vector3,
	b: Vector3,
) -> Vector3:
	"""Add two 3-dimensional vectors.

	The components of the sum of two vectors is the vector of the components summed.
	This is a more detailed comment about summation.

	Arguments:
		a: The first vector to add upon.
		b: The other vector to add to the first.

	Returns:
		A new `Vector3` as the sum of `a` and `b`.
	"""
	return Vector3(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z,
	)


def dot(
	a: Vector3,
	b: Vector3,
) -> float:
	"""Get the dot product of two vectors.

	The dot product is the sum  of the product of the components of each vector.
	This is a more detailed comment about summation.

	Arguments:
		a: The first vector to add upon.
		b: The other vector to add to the first.

	Returns:
		A `float` value as the dot product of `a` and `b`.
	"""
	return a.x * b.x + a.y * b.y + a.z * b.z
