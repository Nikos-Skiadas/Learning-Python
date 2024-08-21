"""`set`
sized: they have length (supports `len`)
iterable: you can iterate through their items
container: you can ask if an item is in a list
collection: means all of the above
set: supports set operations like union, intersection, difference
mutable: can `add` and `discard` items
"""


from typing import Sized, Iterable, Container, Collection, Set, MutableSet


if __name__ == "__main__":
	x = {2, 3, 5}
	y = {3, 5, 7}

	x.add(11)  # add item to set if not there
	x.remove(11)  # remove item from set if there
	x.discard(11)  # remove item from set
	x.update(y)  # update items of x with items from y (mutate x with y)
	x.clear()  # empty the set
	z = x.copy()  # copy of set
	b = x.issuperset(y)  # x is superset of y
	b = x.issubset(y)  # x is subset of y
	z = x.union(y)
	z = x.intersection(y)
	z = x.difference(y)
	z = x.symmetric_difference(y)

	b = x >= y  # x is superset of y
	b = x > y  # x is genuine superset of y
	b = x <= y  # x is subset of y
	b = x < y  # x is genuine subset of y
	b = x == y  # x is equal to y
	b = x != y  # x is not equal to y

	z = x | y  # x.union(y)
	z = x & y  # x.intersection(y)
	z = x - y  # x.difference(y)
	z = x ^ y  # x.symmetric_difference(y)
