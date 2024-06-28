from __future__ import annotations

import abc
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


class Graph[
	Vert: typing.Hashable,
	Node: typing.Collection,
](
	dict[
		Vert,
		Node,
	],
	metaclass = abc.ABCMeta,
):

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

	@abc.abstractmethod
	def setEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		self.setVert(tail)
		self.setVert(head)

	@abc.abstractmethod
	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		...

class DictGraph[
	Vert: typing.Hashable,
	Edge: typing.Any,
](
	Graph[
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

	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		self[tail].pop(head, None)


class SetGraph[
	Vert,
](
	Graph[
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

	def delEdge(self,
		tail: Vert,
		head: Vert,
	) -> None:
		self[tail].discard(head)


