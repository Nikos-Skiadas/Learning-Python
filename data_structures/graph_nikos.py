from __future__ import annotations


import json
import typing


typing.Hashable


class Serializable:

	def __repr__(self):
		return json.dumps(self,
			indent = 4,
			default = self.serialize,
		).replace('"', '').replace("[", "{").replace("]", "}")

	@classmethod
	def serialize(cls, object: dict | set):
		return list(object) if isinstance(object, set) else object


class GraphProtocol(typing.Protocol):

	def get_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable,
	) -> typing.Any | None:
		...

	def set_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable, *args: typing.Any,
	):
		...

	def add_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable, *args: typing.Any,
	):
		...

	def del_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable,
	):
		...


class Graph(Serializable, dict[typing.Hashable, dict[typing.Hashable, typing.Any]]):

	def get_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable,
	) -> typing.Any | None:
		return self.get(tail, dict()).get(head)

	def set_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable, weight: typing.Any,
	):
		if tail not in self:
			self[tail] = dict()

		self[tail][head] = weight

	def add_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable, weight: typing.Any,
	):
		if tail not in self:
			self[tail] = dict()

		self[tail].setdefault(head, weight)

	def del_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable,
	):
		self[tail].pop(head, None)

		if not self[tail]:
			self.pop(tail)


class UnweightedGraph(Serializable, dict[typing.Hashable, set[typing.Hashable]]):

	def get_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable,
	) -> bool:
		return head in self.get(tail, set())

	def set_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable,
	):
		if tail not in self:
			self[tail] = set()

		self[tail].add(head)

	def add_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable,
	):
		self.set_edge(
			tail,
			head,
		)

	def del_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable,
	):
		self[tail].discard(head)

		if not self[tail]:
			self.pop(tail)


class UndirectedProtocol(GraphProtocol):

	def set_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable, *args: typing.Any,
	):
		self.set_edge(tail, head, *args)
		self.set_edge(head, tail, *args)

	def add_edge(self,
		tail: typing.Hashable,
		head: typing.Hashable, *args: typing.Any,
	):
		self.add_edge(tail, head, *args)
		self.add_edge(head, tail, *args)


class UndirectedGraph(UndirectedProtocol, Graph):

	...


class UndirectedUnweightedGraph(UndirectedProtocol, UnweightedGraph):

	...
