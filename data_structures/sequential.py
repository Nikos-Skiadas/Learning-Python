from __future__ import annotations


class Node[Data]:

	def __init__(self, data: Data,
		next: Node[Data] | None = None,
		prev: Node[Data] | None = None,
	):
		self.data = data
		self.next = next
		self.prev = prev


	@property
	def next(self) -> Node[Data] | None:
		return self._next

	@next.setter
	def next(self, node: Node[Data] | None):
		self._next = node

		if node is not None:
			node._prev = self

	@next.deleter
	def next(self):
		self.next = None


	@property
	def prev(self) -> Node[Data] | None:
		return self._prev

	@prev.setter
	def prev(self, node: Node[Data] | None):
		self._prev = node

		if node is not None:
			node._next = self

	@prev.deleter
	def prev(self):
		self.prev = None


class List[Data]:

	def __init__(self):
		self.head: Node[Data] | None = None
		self.tail: Node[Data] | None = None

	def __bool__(self) -> bool:
		return \
			self.head is not None or \
			self.tail is not None

	def __iter__(self):
		node = self.head

		while node is not None:
			yield node.data
			node = node.next

	def __reversed__(self):
		node = self.tail

		while node is not None:
			yield node.data
			node = node.prev


class Deque[Data](List[Data]):

	def prepend(self, data: Data):
		self.head = Node(data,
			next = self.head,
		)

		if self.tail is None:
			self.tail = self.head

	def append(self, data: Data):
		self.tail = Node(data,
			prev = self.tail,
		)

		if self.head is None:
			self.head = self.tail

	def pull(self) -> Data:
		if self.head is None:
			raise IndexError(f"`{self.pull.__name__}` from an empty `{self.__class__.__name__}`")

		data      = self.head.data
		self.head = self.head.next

		return data

	def pop(self) -> Data:
		if self.tail is None:
			raise IndexError(f"`{self.pop.__name__}` from an empty `{self.__class__.__name__}`")

		data      = self.tail.data
		self.tail = self.tail.prev

		return data


class Stack[Data](Deque[Data]):

	def push(self, data: Data):
		super().append(data)


class Queue[Data](Deque[Data]):

	def enqueue(self, data: Data):
		super().append(data)

	def dequeue(self) -> Data:
		return super().pull()
