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


def dijkstra(graph: dict[str, dict[str, float]], source: str, target: str | None = None):
	dist = dict.fromkeys(graph.keys(), inf)
	prev = {}
	vertices = set(graph.keys())
	dist[source] = 0.

	while vertices:
		u = min(vertices, key = dist.get)  # type: ignore
		vertices.remove(u)

		for v, weight in graph[u].items():
			if v in vertices:
				alt = dist[u] + weight

				if alt < dist[v]:
					dist[v] = alt
					prev[v] = u

	return dist, prev


if __name__ == "__main__":
	graph = {
		"A": {
			"B": 2.,
			"C": 3.,
			"D": 5.,
		},
		"B": {
			"C": 7.,
			"D": 2.,
		},
		"C": {
			"D": 3.,
		},
		"D": {
		},
	}

	print(*dijkstra(graph, "A"))
