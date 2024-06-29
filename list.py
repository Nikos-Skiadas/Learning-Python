# An example of list and some stuff needed:
numbers = [2, 3, 5, 7]; print(numbers)
number = 11
index = len(numbers) // 2

# List methods:
numbers.append(number); print(numbers)  # append 11 to numbers
numbers.extend(range(12))  # extend number
_ = numbers.pop(index); print(_)  # pop element with index from numbers
numbers.insert(index, number); print(numbers)  # insert number at index in numbers
numbers.remove(number); print(numbers)  # remove first occurance of number from numbers if number is there
_ = numbers.index(number, 0, len(numbers)); print(_)  # search for index of first occurance of number in numbers
_ = numbers.count(number); print(_)  # count occurances of number in numbers
numbers.sort(reverse = True); print(numbers)  # clear the list
numbers.reverse(); print(numbers)  # reverse numbers
_ = numbers.copy(); print(_)  # copy numbers
numbers.clear(); print(numbers)  # clear numbers

# List comprehensions

# 1:
squares = []

for number in range(10):
	squares.append(number * number)

# 2a:
squares = list(range(10))

for index in range(10):
	squares[index] *= squares[index]

# 2b:
squares = list(range(10))

for index, number in enumerate(squares):
	squares[index] *= number

# 3a:
squares = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 3b:
squares = [number * number for number in squares]


# Nested list comprehension:
matrix = [
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9],
]

# 1:
temp = matrix.copy()

for i, row in enumerate(matrix):
	for j, _ in enumerate(row):
		if i != j:
			matrix[i][j] = temp[j][i]

# 2:
matrix = [[matrix[j][i] for j, _ in enumerate(row)] for i, row in enumerate(matrix)]
