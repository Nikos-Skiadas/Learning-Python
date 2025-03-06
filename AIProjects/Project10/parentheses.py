def complete_parentheses(expression):
    """This function takes a list representing a numerical expression without left parentheses
    and returns the expression with the correct left parentheses inserted.
    """
    stack = []

    for token in expression:
        if token == ')':
            operand2 = stack.pop()
            operator = stack.pop()
            operand1 = stack.pop()
            stack.append(['(', operand1, operator, operand2, ')'])
        else:
            stack.append(token)

    return stack[0]


def flatten_expression(expr):

    flat = []

    for item in expr:
        if isinstance(item, list):
            flat.extend(flatten_expression(item))  # Recursively flatten nested lists

        else:
            flat.append(item)

    return flat


if __name__ == "__main__":

    expression = [1, '-', 2, ')', '*', 3, '-', 4, ')', '*', 5, '-', 6, ')', ')', ')', ')']
    result = complete_parentheses(expression)

    flattened_result = flatten_expression(result)

    print(*flattened_result)
