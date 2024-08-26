from __future__ import annotations


class Graph(dict[int, dict[int, float]]):

	def get_edge(self,
		tail: int,
		head: int,
	) -> float:
		...

	def set_edge(self,
		tail: int,
		head: int,
		edge: float,
	):
		...

	def del_edge(self,
		tail: int,
		head: int,
	):
		...


class UnweightedGraph(dict[int, set[int]]):

	def get_edge(self,
		tail: int,
		head: int,
	) -> bool:
		...

	def set_edge(self,
		tail: int,
		head: int,
	):
		...

	def del_edge(self,
		tail: int,
		head: int,
	):
		...


class UndirectedGraph(Graph):

	def get_edge(self,
		tail: int,
		head: int,
	) -> float:
		...

	def set_edge(self,
		tail: int,
		head: int,
		edge: float,
	):
		...

	def del_edge(self,
		tail: int,
		head: int,
	):
		...


class UndirectedUnweightedGraph(UnweightedGraph):

	def get_edge(self,
		tail: int,
		head: int,
	) -> bool:
		...

	def set_edge(self,
		tail: int,
		head: int,
	):
		...

	def del_edge(self,
		tail: int,
		head: int,
	):
		...
