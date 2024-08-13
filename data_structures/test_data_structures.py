import json

from data_structures.network import *


class TestGraph:

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
	edges = {(*edge, sum(edge)) for edge in unweighted_edges}

	@staticmethod
	def empty(lines: int):
		for line in range(lines):
			print()

	def test_graph(self):
		self.empty(2)
		print(Graph.from_edges(*self.edges))

	def test_unweighted_graph(self):
		self.empty(2)
		print(UnweightedGraph.from_edges(*self.unweighted_edges))

	def test_undirected_graph(self):
		self.empty(2)
		print(UndirectedGraph.from_edges(*self.edges))

	def test_undirected_unweighted_graph(self):
		self.empty(2)
		print(UndirectedUnweightedGraph.from_edges(*self.unweighted_edges))
