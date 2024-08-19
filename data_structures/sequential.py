from __future__ import annotations


import typing


class SingleNode[Data]:

	@classmethod
	def insert(cls,
		node: SingleNode[Data] | None,
		data:            Data        ,
	):
		temp = SingleNode(
			data = data,
			next = node.next if node is not None else node
		)

		if node is not None:
			node.next = temp

	@classmethod
	def delete(cls, node: SingleNode[Data]):
		node.next = node.next.next if node.next is not None else node.next


	def __init__(self,
		data:            Data               ,
		next: SingleNode[Data] | None = None,
	):
		self.data = data
		self.next = next

	def __next__(self):
		if self.next is None:
			raise StopIteration

		return self.next


	@property
	def next(self) -> SingleNode[Data] | None:
		return self._next

	@next.setter
	def next(self, node: SingleNode[Data] | None):
		self._next = node

	@next.deleter
	def next(self):
		self.next = None


class List[Data]:

	def __init__(self):
		self.head: SingleNode[Data] | None = None
		self.node: SingleNode[Data] | None = self.head

	def __bool__(self) -> bool:
		return self.head is not None

	def __iter__(self):
		self.node = self.head

		return self

	def __next__(self):
		if self.node is None:
			raise StopIteration

		data      = self.node.data
		self.node = self.node.next

		return data

	def pop(self) -> Data:
		if self.head is None:
			raise IndexError(f"{self.pop.__name__}-ing from empty {self.__class__.__name__.lower()}")

		data      = self.head.data
		self.head = self.head.next

		return data


class Stack[Data](List[Data]):

	def push(self, data: Data):
		self.head = SingleNode(
			data =      data,
			next = self.head,
		)


class Queue[Data](List[Data]):

	def enqueue(self, data: Data):
		SingleNode.insert(
			data = data,
			node = self.head,
		)

	def dequeue(self) -> Data:
		if self.head is None:
			raise IndexError(f"{self.pop.__name__}-ing from empty {self.__class__.__name__.lower()}")

		data = self.head.data
		SingleNode.delete(self.head)

		return data
