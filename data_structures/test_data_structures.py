import json

from data_structures.network import *


class TestGraph:

	def test_representations(self):
		edges = {
			(1, 2, 3),
			(1, 3, 4),
			(1, 4, 5),
			(1, 5, 6),
			(2, 3, 5),
			(2, 4, 6),
			(2, 5, 7),
			(3, 4, 7),
			(3, 5, 8),
			(4, 5, 9),
		}
		graph = Graph.from_edges(edges)

		print(json.dumps(graph))
