"""`list`

sized: they have length (supports `len`)
iterable: you can iterate through their items
container: you can ask if an item is in a list
sequence: means all of the above plus supports getting an item at index
mutable: means all of the above supports support setting an item at index

summable: supports extending list via addition
"""


from typing import Sized, Iterable,	Container, Sequence, MutableSequence


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


if __name__ == "__main__":
	x = [2, 3, 5]
	y = [3, 5, 7]

	x.append(11)  # append to end of list
	x_last_item = x.pop()  # pop item from end of list
	x.clear()  # empty the list
	z = x.extend(y)  # extent list with another
	x_item_index = x.index(x[1])  # x_item_index == 1
	x.insert(1, 11)  # insert item at index
	x.remove(11)  # remove item in list (has to be in the list)
	x.reverse()  # turn the list upside down
	x.sort()  # sort list
	z = x.copy()  # copy of list
	x_item_count = x.count(3)  # count how many times item is in list

	z = x + y  # extent list with another
	z = x * 3  # extend list by two more copies of list
