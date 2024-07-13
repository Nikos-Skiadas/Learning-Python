"""`dict`
sized: they have length (supports `len`)
iterable: you can iterate through their items
container: you can ask if an item is in a list
collection: means all of the above
mapping: contains logic of mapping keys to values and getting an item at key
mutable: means all of the above supports support setting an item at key
"""


from typing import Sized, Iterable, Container, Collection, Mapping, MutableMapping
from numpy import inf


if __name__ == "__main__":
	x = {
		"c": 2,
		"a": 3,
		"b": 5,
	}
	y = {
		"c": 3,
		"d": 5,
		"e": 7,
	}

	x_item = x["a"]  # get value at key
	x["a"] = 11  # set value at key (if it exists ovewrite it else add the key!)
	del x["a"]  # remove key (and value) from dict if there

	x_item = x.get("a", 0)  # get value at key if there else return default value
	x.setdefault("a", 11)  # set value at key (if it exists do nothing it else add the key!)
	x_item = x.pop("a", 0)  # remove key and return value from dict if there else return default value

	x.update(y)  # update values of x at keys of y with values from y (mutate x with y)
	x.clear()  # empty the dict
	z = x.copy()  # copy of dict

	b = x == y  # x is equal to y
	b = x != y  # x is not equal to y

	z = x | y  # z = x.copy(); z.update(y)


def dijkstra(graph: dict[str, dict[str, float]]):
	dist = dict.fromkeys(graph.keys(), inf)
	prev = {}
	vertices = set(graph.keys())

	while vertices:
		u = min(vertices)
		vertices.remove(u)

		for v, weight in graph[u].items():
			if v in vertices:
				alt = dist[v] + weight

				if alt < dist[v]:
					dist[v] = alt
					prev[v] = u

	return dist, prev
