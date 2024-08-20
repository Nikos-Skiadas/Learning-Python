from __future__ import annotations


class Node[Data]:


	def __init__(self,
		data:      Data               ,
		next: Node[Data] | None = None,
	):
		self.data = data
		self.next = next if next is not None else self

	def __bool__(self) -> bool:
		return self.next is not self


	@property
	def next(self) -> Node[Data]:
		return self._next

	@next.setter
	def next(self, node: Node[Data]):
		self._next = node

	@next.deleter
	def next(self):
		self.next = self


	def append(self, data: Data):
		self.next = Node(
			data =      data,
			next = self.next,
		)

	def remove(self) -> Data:
		data      = self.next.data
		self.next = self.next.next

		return data


class List[Data]:

	def __init__(self):
		self.head: Node[Data] | None = None

	def __bool__(self) -> bool:
		return self.head is not None

	def __iter__(self):
		self.tail = self.head

		return self

	def __next__(self):
		if self.tail is None:
			raise StopIteration

		self.tail = self.tail.next
		data      = self.tail.data

		if self.tail is self.head:
			self.tail = None

		return data


	def wind(self):
		if self.head is not None:
			self.head = self.head.next

	def append(self, data: Data):
		if self.head is None:
			self.head = Node(data)

		else:
			self.head.append(data)

	def remove(self) -> Data:
		if self.head is None:
			raise IndexError(f"Removing from an empty {self.__class__.__name__.lower()}")

		if not self.head:
			data = self.head.data
			self.head = None

		else:
			data = self.head.remove()

		return data


class Stack[Data](List[Data]):

	def push(self, data: Data):
		super().append(data)

	def pop(self) -> Data:
		return super().remove()


class Queue[Data](List[Data]):

	def enqueue(self, data: Data):
		super().append(data)
		self.wind()

	def dequeue(self) -> Data:
		return super().remove()
