from __future__ import annotations


import typing


class SingleNode[Data]:

	@classmethod
	def insert(cls,
		node: SingleNode[Data],
		data:            Data ,
	):
		node.next = SingleNode(
			data =      data,
			next = node.next,
		)

	@classmethod
	def pop(cls, node: SingleNode[Data]) -> Data:
		temp = node.next if node.next is not None else node

		data = temp.data
		node.next = temp.next

		return data


	def __init__(self,
		data:            Data               ,
		next: SingleNode[Data] | None = None,
	):
		self.data = data
		self.next = next if next is not None else self

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

	def append(self, data: Data):
		self.head = SingleNode(
			data =      data,
			next = self.head,
		)

	def pop(self) -> Data:
		if self.head is None:
			raise IndexError(f"{self.pop.__name__}-ing from empty {self.__class__.__name__.lower()}")

		data      = self.head.data
		self.head = self.head.next

		return data


class Stack[Data](List[Data]):

	def push(self, data: Data):
		super().append(data)


class Queue[Data](List[Data]):

	def enqueue(self, data: Data):
		if self.head is None:
			super().append(data)

		else:
			SingleNode.insert(
				data = data,
				node = self.head,
			)

	def dequeue(self) -> Data:
		if self.head is None:
			raise IndexError(f"{self.pop.__name__}-ing from empty {self.__class__.__name__.lower()}")

		return SingleNode.pop(self.head)
