from __future__ import annotations

import abc
import collections
import json
import typing


class NeighborhoodBase(typing.Collection):

	def __repr__(self):
		return json.dumps(self if isinstance(self, typing.Mapping) else list(self),
			indent = 4,
		).replace('"', '')


	@property
	def degree(self) -> int:
		return len(self)


class Neighborhood[
	Vertex,
	Weight,
](
	NeighborhoodBase,
	dict[
		Vertex,
		Weight,
	],
):

	...

class UnweightedNeighborhood[
	Vertex,
](
	NeighborhoodBase,
	set[
		Vertex,
	],
):

	...


class GraphBase[
	Vertex: typing.Hashable,
	Neighborhood: NeighborhoodBase,
](
	collections.defaultdict[
		Vertex,
		Neighborhood
	],
	metaclass = abc.ABCMeta,
):

	@classmethod
	def from_edges(cls, *edges: tuple):
		graph = cls()

		for edge in edges:
			graph.setEdge(*edge)

		return graph


	def __init__(self, *args, **kwargs):
		super().__init__(self.default_factory, *args, **kwargs)

	def __repr__(self):
		return json.dumps(self,
			indent = 4,
		).replace('"', '')

	def __delitem__(self, key: Vertex) -> None:
		self.delVert(key)


	@property
	@abc.abstractmethod
	def default_factory(self) -> type:
		...

	@property
	def vertices(self) -> typing.Collection[Vertex]:
		return self.keys()

	@property
	def neighborhoods(self) -> typing.Collection[Neighborhood]:
		return self.values()

	@property
	def order(self) -> int:
		return len(self)

	@property
	def size(self) -> int:
		return sum(neighborhood.degree for neighborhood in self.neighborhoods)


	def adjacent(self,
		tail: Vertex,
		head: Vertex,
	) -> bool:
		return head in self[tail]


	def addVert(self,vert: Vertex) -> None:
		self[vert]

	def setVert(self, vert: Vertex, neighborhood: Neighborhood) -> None:
		self[vert] = neighborhood

	def getVert(self, vert: Vertex) -> Neighborhood:
		return self[vert]

	def delVert(self, vert: Vertex) -> None:
		self.pop(vert, None)

		for head in self:
			self.delEdge(head, vert)

	def setEdge(self,
		tail: Vertex,
		head: Vertex,
	*args) -> None:
		raise NotImplementedError

	def getEgde(self,
		tail: Vertex,
		head: Vertex,
	):
		raise NotImplementedError

	def delEdge(self,
		tail: Vertex,
		head: Vertex,
	) -> None:
		raise NotImplementedError


class Graph[Vert: typing.Hashable, Edge](GraphBase[Vert, Neighborhood[Vert, Edge]]):

	@property
	def default_factory(self) -> type:
		return Neighborhood


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


class UnweightedGraph[Vert: typing.Hashable](GraphBase[Vert, UnweightedNeighborhood[Vert]]):

	@property
	def default_factory(self) -> type:
		return UnweightedNeighborhood


	def setEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		self[tail].add(head)

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
		self[tail].discard(head)


class Undirected[Vert: typing.Hashable, Collection: NeighborhoodBase](GraphBase[Vert, Collection]):

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


class UndirectedGraph(Undirected, Graph):

	...


class UndirectedUnweightedGraph(Undirected, UnweightedGraph):

	...
