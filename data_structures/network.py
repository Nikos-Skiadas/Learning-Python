from __future__ import annotations

import typing


class Node[
	Vert: typing.Hashable,
](
	typing.Collection[
		Vert,
	]
):

	@property
	def degree(self) -> int:
		return len(self)


class Base[
	Vert: typing.Hashable,
	Node: typing.Collection,
](
	dict[
		Vert,
		Node,
	],
):

	@classmethod
	def from_edges(cls, edges: typing.Iterable):
		raise NotImplementedError


	def __init__(self, default_factory: type[Node], *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.default_factory = default_factory


	@property
	def order(self) -> int:
		return len(self)

	@property
	def size(self) -> int:
		return sum([len(node) for node in self.values()])


	def adjacent(self,
		tail: Vert,
		head: Vert,
	) -> bool:
		return tail in self[head]

	def neighbors(self,
		tail: Vert,
	) -> Node:
		return self[tail]

	def setVert(self,
		tail: Vert,
	) -> None:
		self.setdefault(tail, self.default_factory())

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
		self.setVert(tail)
		self.setVert(head)

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

class Weighted[
	Vert: typing.Hashable,
	Edge: typing.Any,
](
	Base[
		Vert,
		dict[
			Vert,
			Edge,
		]
	]
):

	def __init__(self, *args, **kwargs):
		super().__init__(dict, *args, **kwargs)


	def setEdge(self,
		tail: Vert,
		head: Vert,
		edge: Edge,
	) -> None:
		super().setEdge(tail, head)

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


class Unweighted[
	Vert,
](
	Base[
		Vert,
		set[
			Vert,
		],
	]
):

	def __init__(self, *args, **kwargs):
		super().__init__(set, *args, **kwargs)


	def setEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		super().setEdge(tail, head)

		self[tail].add(head)

	def getEgde(self,
		tail: Vert,
		head: Vert,
	) -> bool:
		return head in self[tail]

	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		self[tail].discard(head)


class Undirected[
	Vert: typing.Hashable,
	Node: typing.Collection,
](
	Base[
		Vert,
		Node,
	],
):

	def setEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		super().setEdge(tail, head)
		super().setEdge(head, tail)

	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		super().delEdge(tail, head)
		super().delEdge(head, tail)



class Graph(Weighted):
	...


class UnweightedGraph(Unweighted):
	...


class UndirectedGraph(Undirected, Weighted):
	...


class UndirectedUnweightedGraph(Undirected, Unweighted):
	...
