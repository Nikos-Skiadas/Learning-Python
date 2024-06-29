def fibonacci(upper_limit):
	a = 0
	b = 1

	while a < upper_limit:
		print(a)
		a, b = b, a + b

	else:
		print("Reached the upper limit")

	return a
