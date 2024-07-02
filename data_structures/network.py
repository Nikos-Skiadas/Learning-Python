from __future__ import annotations

import collections
import typing


class Collection(typing.Collection):

	@property
	def degree(self) -> int:
		return len(self)


class Set[Key](set[Key], Collection):
	...


class Dict[Key, Value](dict[Key, Value], Collection):
	...


class Base[Vert: typing.Hashable, Neighborhood: Collection](collections.defaultdict[Vert, Neighborhood]):

	@classmethod
	def fromIterable(cls, *edges: tuple):
		graph = cls()

		for edge in edges:
			graph.setEdge(*edge)

		return graph


	@property
	def vertices(self) -> set[Vert]:
		return set(self.keys())

	@property
	def neighborhoods(self) -> list[Neighborhood]:
		return list(self.values())

	@property
	def order(self) -> int:
		return len(self)

	@property
	def size(self) -> int:
		return sum([neighborhood.degree for neighborhood in self.neighborhoods])


	def neighbors(self,
		tail: Vert,
	) -> Neighborhood:
		return self[tail]

	def adjacent(self,
		tail: Vert,
		head: Vert,
	) -> bool:
		return head in self.neighbors(tail)


	def setVert(self,
		tail: Vert,
	) -> None:
		self.neighbors(tail)

	def delVert(self,
		tail: Vert,
	) -> None:
		self.pop(tail, None)

		for head in self:
			self.delEdge(head, tail)

	def setEdge(self,
		tail: Vert,
		head: Vert,
	*args) -> None:
		raise NotImplementedError

	def getEgde(self,
		tail: Vert,
		head: Vert,
	):
		raise NotImplementedError

	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		raise NotImplementedError

class DictGraph[Vert: typing.Hashable, Edge: typing.Any](Base[Vert, Dict[Vert, Edge]]):

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(Dict, *args, **kwargs)


	def setEdge(self,
		tail: Vert,
		head: Vert,
		edge: Edge,
	) -> None:
		self[tail][head] = edge

	def getEgde(self,
		tail: Vert,
		head: Vert,
	) -> Edge:
		return self[tail][head]

	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		self[tail].pop(head, None)


class SetGraph[Vert](Base[Vert, Set[Vert]]):

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(Set, *args, **kwargs)


	def setEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		self.neighbors(tail).add(head)

	def getEgde(self,
		tail: Vert,
		head: Vert,
	) -> bool:
		return self.adjacent(
			tail,
			head,
		)

	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		self.neighbors(tail).discard(head)


class Undirected[Vert: typing.Hashable, Neighborhood: Collection](Base[Vert, Neighborhood]):

	def setEdge(self,
		tail: Vert,
		head: Vert,
	*args) -> None:
		super().setEdge(tail, head, *args)
		super().setEdge(head, tail, *args)

	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		super().delEdge(tail, head)
		super().delEdge(head, tail)



class Graph(DictGraph):
	...


class UnweightedGraph(SetGraph):
	...


class UndirectedGraph(Undirected, Graph):
	...


class UndirectedUnweightedGraph(Undirected, UnweightedGraph):
	...
