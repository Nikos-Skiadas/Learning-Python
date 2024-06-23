from __future__ import annotations

import functools
import typing


Vert = typing.TypeVar("Vert", bound = typing.Hashable)
Data = typing.TypeVar("Data", bound = typing.Any)


@functools.total_ordering
class Node[
	Vert,
	Data,
](
	dict[
		Vert,
		Data,
	]
):

	def __le__(self, node: Node) -> bool:
		return self.keys() <= node.keys() and all(weight == node[vert] for vert, weight in self.items())


	@property
	def degree(self) -> int:
		return len(self)


Edge: typing.TypeAlias = tuple[
	Vert,
	Vert,
	Data,
]


@functools.total_ordering
class Graph[
	Vert,
	Data,
](
	Node[
		Vert,
		Node[
			Vert,
			Data,
		],
	]
):

	@classmethod
	def from_edges(cls,
		edges: set[Edge]
	) -> Graph:
		graph = Graph()

		for tail, head, edge in edges:
			graph.addEdge(
				tail,
				head,
				edge,
			)

		return graph


	def __le__(self, graph: Graph) -> bool:
		return self.keys() <= graph.keys() and all(data <= graph[vert] for vert, data in self.items())


	@property
	def order(self) -> int:
		return len(self)

	@property
	def size(self) -> int:
		return sum(bool(edge) for node in self.values() for edge in node.values())

	@property
	def edges(self) -> set[Edge]:
		return set((tail, head, edge) for tail, node in self.items() for head, edge in node.items())


	def adjacent(self,
		tail: Vert,
		head: Vert,
	) -> bool:
		return tail in self[head]

	def neighbors(self,
		tail: Vert,
	) -> Node:
		return self[tail]

	def addNode(self,
		tail: Vert,
	) -> None:
		self.setdefault(tail, Node())

	def addEdge(self,
		tail: Vert,
		head: Vert,
		edge: Data,
	) -> None:
		self.addNode(tail)
		self.addNode(head)

		self[tail][head] = edge

	def delNode(self,
		tail: Vert,
	) -> None:
		if not self[tail]:
			del self[tail]

	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		del self[tail][head]


class BiGraph[
	Vert,
	Data,
](
	Graph[
		Vert,
		Data,
	]
):

	def addEdge(self,
		tail: Vert,
		head: Vert,
		edge: Data,
	) -> None:
		super().addEdge(
			tail,
			head,
			edge,
		)
		super().addEdge(
			head,
			tail,
			edge,
		)

	def delNode(self,
		tail: Vert,
	) -> None:
		for head in self[tail]:
			super().delEdge(
				head,
				tail,
			)

		super().delNode(tail)

	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		super().delEdge(
			tail,
			head,
		)
		super().delEdge(
			head,
			tail,
		)
