from __future__ import annotations


class Node[Data]:

	def __init__(self,
		data: Data,
		next: Node | None = None,
		prev: Node | None = None,
	):
		self._data = data
		self._next = next
		self._prev = prev

		if next is not None:
			next.prev = self

		if prev is not None:
			prev.next = self

	def __next__(self):
		if self.next is None:
			raise StopIteration

		return self.next


	@property
	def data(self) -> Data:
		return self._data

	@property
	def next(self) -> Node | None:
		return self._next

	@property
	def prev(self) -> Node | None:
		return self._prev


	@next.setter
	def next(self, node: Node | None):
		self._next = node

		if node is not None:
			node._prev = self

	@prev.setter
	def prev(self, node: Node | None):
		self._prev = node

		if node is not None:
			node._next = self


	@next.deleter
	def next(self):
		self.next = None

	@prev.deleter
	def prev(self):
		self.prev = None


class Deque[Data]:

	def __init__(self):
		self.tail: Node | None = None
		self.head: Node | None = None
		self.node: Node | None = None

	def __bool__(self) -> bool:
		return self.tail is not None or self.head is not None

	def __iter__(self):
		self.node = self.head

		return self

	def __next__(self):
		if self.node is None:
			raise StopIteration

		data      = self.node.data
		self.node = self.node.next

		return data


	def append_tail(self, data: Data):
		self.tail = Node(
			data =      data,
			prev = self.tail,
		)

		if self.head is None:
			self.head = self.tail

	def append_head(self, data: Data):
		self.head = Node(
			data =      data,
			next = self.head,
		)

		if self.tail is None:
			self.tail = self.head

	def remove_tail(self) -> Data:
		if self.tail is not None:
			data      = self.tail.data
			self.tail = self.tail.prev

			if self.tail is None:
				self.head = self.tail

			return data

		raise IndexError(f"pop from empty {self.__class__.__name__.lower()}")

	def remove_head(self) -> Data:
		if self.head is not None:
			data      = self.head.data
			self.head = self.head.next

			if self.head is None:
				self.tail = self.head

			return data

		raise IndexError(f"pop from empty {self.__class__.__name__.lower()}")


class Stack[Data](Deque[Data]):

	def push(self, data: Data):
		return self.append_tail(data)

	def pop(self) -> Data:
		return self.remove_tail()


class Queue[Data](Deque[Data]):

	def enqueue(self, data: Data):
		return self.append_tail(data)

	def dequeue(self) -> Data:
		return self.remove_tail()
