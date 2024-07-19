class Rectangle:

	def __init__(self,
		length: float | int = 0.,
		height: float | int = 0.,
	):
		self.length = float(length)
		self.height = float(height)

	@property
	def perimeter(self) -> float:
		return 2 * (self.length + self.height)

	@property
	def area(self) -> float:
		return self.length * self.height

	def scale(self, factor: float | int):
		self.length *= factor
		self.height *= factor
