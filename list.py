A = [
	[2, 3, 5, 7],
	[3, 5, 7, 2],
	[5, 7, 2, 3],
	[7, 2, 3, 5],
]
B = [
	[2, 3],
	[5, 7],
]
C = [
	[2, 5],
	[3, 7],
]

def copy(matrix: list[list[int]]) -> list[list[int]]:
	return [[item for item in row] for row in matrix]

	matrix_copy = []

	for row in matrix:
		row_copy = []

		for item in row:
			row_copy.append(item)

		matrix_copy.append(row_copy)

	return matrix_copy


def diagonal(matrix: list[list[int]]) -> list[int]:
	return [item for i, row in enumerate(matrix) for j, item in enumerate(row) if i == j]

	items = []

	for i, row in enumerate(matrix):
		for j, item in enumerate(row):
			if i == j:
				diagonal_items.append(item)

	return items


def trace(matrix: list[list[int]]) -> int:
	return sum(diagonal(matrix))

	sum = 0
	diagonal_items = diagonal(matrix)

	for item in diagonal_items:
		sum += item

	return sum


def transpose(matrix: list[list[int]]) -> list[list[int]]:
	...


def mul(left: list[list], right: list[list[int]]) -> list[list[int]]:
	product = []

	for i, left_row in enumerate(left):
		...

		for j, right_column in enumerate(...):
			...

			for k, item in enumerate(...):
				...

			...

		...

	return product
