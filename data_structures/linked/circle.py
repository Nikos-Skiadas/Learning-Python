from __future__ import annotations


class Node[Data]:

	def __init__(self, data: Data,
		next: Node[Data] | None = None,
	):
		self.data = data
		self.next = next if next is not None else self


	@property
	def next(self) -> Node[Data]:
		return self._next

	@next.setter
	def next(self, node: Node[Data]):
		self._next = node

	@next.deleter
	def next(self):
		self.next = self


class List[Data]:

	def __init__(self):
		self.tail: Node[Data] | None = None

	def __iter__(self) -> List[Data]:
		self.node = self.tail

		return self

	def __next__(self) -> Data:
		if self.node is None:
			raise StopIteration

		self.node = self.node.next
		data      = self.node.data

		if self.node is self.tail:
			self.node = None

		return data


class Queue[Data](List[Data]):

	def enqueue(self, data: Data):
		if self.tail is None:
			self.tail = Node(data)

		else:
			self.tail.next = Node(data,
				next = self.tail.next,
			)

		self.tail = self.tail.next

	def dequeue(self) -> Data:
		if self.tail is None:
			raise IndexError(f"`{self.dequeue.__name__}` from an empty `{self.__class__.__name__}`")

		if self.tail is self:
			data      = self.tail.data
			self.tail = None

		else:
			data           = self.tail.next.data
			self.tail.next = self.tail.next.next

		return data
