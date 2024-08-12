import json

from data_structures.network import *


class TestGraph:

	def test_representations(self):
		unweighted_edges = {
			(1, 2),
			(1, 3),
			(1, 4),
			(1, 5),
			(2, 3),
			(2, 4),
			(2, 5),
			(3, 4),
			(3, 5),
			(4, 5),
		}
		graph = UndirectedUnweightedGraph.from_edges(*unweighted_edges)

		print()
		print()
		print(graph)

		...
