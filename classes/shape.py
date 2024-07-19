from __future__ import annotations


class Rectangle:

	def __init__(self,
		length: float | int = 0.,
		height: float | int = 0.,
	):
		self.length = float(length)
		self.height = float(height)

	def __repr__(self) -> str:
		return f"{self.length} × {self.height}"

	def __add__(self, other: Rectangle) -> Rectangle:
		return Rectangle(
			self.length + other.length,
			self.height + other.height,
		)

	def __mul__(self, factor: float | int) -> Rectangle:
		return Rectangle(
			self.length * factor,
			self.height * factor,
		)

	def __rmul__(self, factor: float | int) -> Rectangle:
		return self.__mul__(factor)

	@property
	def perimeter(self) -> float:
		return 2 * (self.length + self.height)

	@property
	def area(self) -> float:
		return self.length * self.height
