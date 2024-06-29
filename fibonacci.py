def fibonacci(n: int, *, a: int = 0, b: int = 1) -> list[int]:
	"""Calculate the fibonnaci sequence up to a number.

	It stores all results in a list.
	It also prints results are they are evaluated.

	Arguments:
		n: The maximum number to reach when calculating fibonacci terms.

	Keyword arguments:
		a: F_0
		b: F_1

	Returns:
		The list of fibonacci terms up to given number.
	"""
	f = []

	# Go on until number is reached:
	while a < n:
		f.append(a)
		a, b = b, a + b  # advance by one term

	# Return all terms up to number
	else:
		return f
