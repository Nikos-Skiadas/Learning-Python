from __future__ import annotations


class Node[Data]:

	def __init__(self,data: Data,
		next: Node[Data] | None = None,
	):
		self.data = data
		self.next = next

	def __bool__(self) -> bool:
		return self.next is not None


	@property
	def next(self) -> Node[Data] | None:
		return self._next

	@next.setter
	def next(self, node: Node[Data] | None):
		self._next = node

	@next.deleter
	def next(self):
		self.next = None


class List[Data]:

	def __init__(self):
		self.head: Node[Data] | None = None

	def __iter__(self) -> List[Data]:
		self.node = self.head

		return self

	def __next__(self) -> Data:
		if self.node is None:
			raise StopIteration

		data      = self.node.data
		self.node = self.node.next

		return data


class Stack[Data](List[Data]):

	def push(self, data: Data):
		self.head = Node(data,
			next = self.head,
		)

	def pop(self) -> Data:
		if self.head is None:
			raise IndexError(f"`{self.pop.__name__}` from an empty `{self.__class__.__name__}`")

		data      = self.head.data
		self.head = self.head.next

		return data
