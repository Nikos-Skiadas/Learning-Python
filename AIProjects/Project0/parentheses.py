def complete_parentheses(expression: list[str]) -> list[str]:
	operators = {
		"+",
		"-",
		"*",
		"/",
		"%",
	}

	balanced = []

	for index, token in enumerate(expression[::-1]):
		if token == ")":
			return complete_parentheses(expression[:index])

		balanced.append(token)

	balanced.append("(")

	return balanced[::-1]


if __name__ == "__main__":
	bad_expression = ['1', '+', '2', ')', '*', '3', '-', '4', ')', '*', '5', '-', '6', ')', ')', ')']

	print(*bad_expression)
	print(*complete_parentheses(bad_expression))
