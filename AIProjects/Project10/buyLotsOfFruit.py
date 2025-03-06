# buyLotsOfFruit.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""To run this script, type:

>>> python buyLotsOfFruit.py

Once you have correctly implemented the buyLotsOfFruit function, the script should produce the output:

>>> Cost of [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)] is 12.25
"""


from __future__ import print_function


fruitPrices = {
    'apples': 2.00,
    'oranges': 1.50,
    'pears': 1.75,
    'limes': 0.75,
    'strawberries': 1.00,
}


def buyLotsOfFruit(orderList: list[tuple[str, float]]) -> float | None:
    """Calculate the cost of an order list according to the weight of the fruit and its price.

    Based on `shop.FruitShop.getCostPerPound` modified to handle missing fruit.

    Args:
        orderList: A list of fruits and corresponding weight ordered.

    Returns:
        The total cost of the order list.
        If some fruit in the order list is missing, return `None` with a message.
    """
    totalCost = 0.0

    try:
        for fruit, numPounds in orderList:
            costPerPound = fruitPrices[fruit]
            totalCost += numPounds * costPerPound

    # If there is a fruit in the list that does not have a price, print an error message and return `None`:
    except KeyError:
        print(f"Some fruit are out of stock or unavailble.")

        return

    return totalCost


def buyLotsOfFruitBetter(orderDict: dict[str, float]) -> float | None:
    """Calculate the cost of an order list according to the weight of the fruit and its price.

    I believe using a dictionary for a shopping list is more approriate as items in such a list are unique.
    That way we can more easily find the missing fruit.

    Args:
        orderDict: A dictionary of fruits and corresponding weight ordered.

    Returns:
        The total cost of the order list.
        If some fruit in the order list is missing, return `None` with a message.
    """
    try:
        costDict = {fruit: orderDict[fruit] * fruitPrices[fruit] for fruit in orderDict}

        return sum(costDict.values())

    # If there is a fruit in the list that does not have a price, print an error message with the missing fruit and return `None`:
    except KeyError:
        missingFruit = orderDict.keys() - fruitPrices.keys()  # `dict.keys()` is set-like so I can take the set-difference
        print("The", *missingFruit, "are out of stock or unavailble.")  # print missing fruit in sequence

        return


# Main Method
if __name__ == '__main__':
    """This code runs when you invoke the script from the command line."""

    orderList = [
        ('apples', 2.0),
        ('pears', 3.0),
        ('limes', 4.0),
    ]
    print('Cost of', orderList, 'is', buyLotsOfFruit(orderList))

    orderDict = {
        'apples': 2.0,
        'pears': 3.0,
        'limes': 4.0,
    }
    print('Cost of', orderDict, 'is', buyLotsOfFruitBetter(orderDict))


    orderDictBad = {
        'apples': 2.0,
        'pears': 3.0,
        "tomato": 10.0,
        'limes': 4.0,
        "coconut": 10.0,
    }
    print('Cost of', orderDictBad, 'is', buyLotsOfFruitBetter(orderDictBad))
