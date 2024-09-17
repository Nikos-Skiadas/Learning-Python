#	1) import collections: This imports the collections module, which contains specialized data structures like deque, Counter, OrderedDict, etc.
#	2) import json: This imports Python’s json module, which provides methods for parsing JSON strings and serializing Python objects to JSON.
#	3) import typing: This imports the typing module, which provides tools for type hints in Python, allowing for better type-checking and clarity in function signatures, class methods, etc.


from __future__ import annotations

import collections
import json
import typing


class NeighborhoodBase(typing.Collection): # It is a class that inherits from "typing.Collection"

	# The method returns a JSON-like string representation of the object.
	def __repr__(self):
		return json.dumps(self,
			indent = 4,
			default = self.serialize,
		).replace('"', '')

	# This method calculates the degree of the neighborhood, in other words it shows how many nodes are adjacent to a particular node in a graph.
	@property
	def degree(self) -> int:
		return len(self)

	# This serialize method defines a contract for child classes like UnweightedNeighborhood to implement a way to convert their instances into a serializable format.
	@classmethod
	def serialize(cls, neighborhood: typing.Self):
		raise NotImplementedError


class UnweightedNeighborhood[
	VertexType,
](
	set[
		VertexType,
	],
	NeighborhoodBase,
):

	# This method implements the serialize method from the parent class and converts the neighborhood into a list.
	@classmethod
	def serialize(cls, neighborhood: typing.Self) -> list[
		VertexType,
	]:
		return list(neighborhood)

	# This checks whether the specified key (a vertex) is present in the set (self) and ti returns True if the key exists, otherwise False
	def __getitem__(self, key: VertexType) -> bool:
		return key in self


class Neighborhood[
	VertexType,
	WeightType,
](
	dict[
		VertexType,
		WeightType,
	],
	NeighborhoodBase,
):

	# This method converts the Neighborhood object into a serializable format (from a), which in this case is simply the dictionary itself.

	@classmethod
	def serialize(cls, neighborhood: typing.Self) -> dict[
		VertexType,
		WeightType,
	]:
		return neighborhood

# 1) The graph's vertices are stored as keys in the dictionary.
# 2) Each vertex maps to a neighborhood, which could either be weighted (Neighborhood) or unweighted (UnweightedNeighborhood).
# collections.defaultdict is used to simplify graph creation, ensuring that when a vertex is referenced that doesn't yet have a neighborhood, a new neighborhood is automatically created.

class GraphBase[
	VertexType: typing.Hashable,
	NeighborhoodType: NeighborhoodBase,
](
	collections.defaultdict[
		VertexType,
		NeighborhoodType
	],
):

	@classmethod
	def serialize(cls, neighborhood: NeighborhoodType):
		return neighborhood.serialize(neighborhood)

	@classmethod
	def from_edges(cls, *edges: tuple):
		graph = cls()

		for edge in edges:
			graph.add(*edge)

		return graph


	def __init__(self, *args, **kwargs):
		super().__init__(self.default_factory, *args, **kwargs)

	def __call__(self, *args):
		return self.weight(*args)

	def __repr__(self):
		return json.dumps(self,
			indent = 4,
			default = self.serialize
		).replace('"', '')

	def __delitem__(self, key: VertexType):
		self.pop(key, None)

		for vert in self:
			self.discard(vert, key)


	@property
	def default_factory(self) -> type:
		raise NotImplementedError

	@property
	def vertices(self) -> typing.Collection[VertexType]:
		return self.keys()

	@property
	def neighborhoods(self) -> typing.Collection[NeighborhoodType]:
		return self.values()

	@property
	def order(self) -> int:
		return len(self)

	@property
	def size(self) -> int:
		return sum(neighborhood.degree for neighborhood in self.neighborhoods)


	def adjacent(self,
		tail: VertexType,
		head: VertexType,
	) -> bool:
		return head in self[tail]

d
	def add(self, *args):
		raise NotImplementedError

	def weight(self, *args):
		raise NotImplementedError

	def discard(self, *args):
		raise NotImplementedError


class Graph[
	VertexType: typing.Hashable,
	WeightType,
](
	GraphBase[
		VertexType,
		Neighborhood[
			VertexType,
			WeightType,
		]
	]
):

	@property
	def default_factory(self) -> type:
		return Neighborhood


	def add(self,
		tail: VertexType,
		head: VertexType,
		edge: WeightType,
	):
		self[tail][head] = edge

	def weight(self,
		tail: VertexType,
		head: VertexType,
	) -> WeightType:
		return self[tail][head]

	def discard(self,
		tail: VertexType,
		head: VertexType,
	):
		self[tail].pop(head, None)


class UnweightedGraph[Vert: typing.Hashable](GraphBase[Vert, UnweightedNeighborhood[Vert]]):

	@property
	def default_factory(self) -> type:
		return UnweightedNeighborhood


	def add(self,
		tail: Vert,
		head: Vert,
	):
		self[tail].add(head)

	def weight(self,
		tail: Vert,
		head: Vert,
	) -> bool:
		return self[tail][head]

	def discard(self,
		tail: Vert,
		head: Vert,
	):
		self[tail].discard(head)


class Undirected[Vert: typing.Hashable, Collection: NeighborhoodBase](GraphBase[Vert, Collection]):

	def add(self,
		tail: Vert,
		head: Vert,
	*args):
		super().add(tail, head, *args)
		super().add(head, tail, *args)

	def discard(self,
		tail: Vert,
		head: Vert,
	):
		super().discard(tail, head)
		super().discard(head, tail)


class UndirectedGraph(Undirected, Graph):

	...


class UndirectedUnweightedGraph(Undirected, UnweightedGraph):

	...
