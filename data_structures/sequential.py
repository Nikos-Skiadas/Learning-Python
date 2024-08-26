from __future__ import annotations


class Node[Data]:

	def __init__(self, data: Data,
		next: Node[Data] | None = None,
		prev: Node[Data] | None = None,
		index: int = 0): # Added index parameter with default value and later i have to add the attribute

		"""We would like a node to hold an index in case it belongs to a list.

		How is `self.index` related to the `self.prev.index` and its `self.next.index`?
		How is `self.index` updated when either `self.prev` or `self.next` is set?
		How is `self.prev.index` or `self.next.index` affected when `self.prev` or `self.next` is set?
		Which methods of `Node` require updating the `index`?
		Note that all methods mutate the node so possibly all of them need to affect some index.
		"""
		self.data = data
		self.next = next
		self.prev = prev
        self.index = index  # Added index attribute




	@property
	def next(self) -> Node[Data] | None:
		return self._next

	@next.setter
	def next(self, node: Node[Data] | None):
		self._next = node

		if node is not None:
			node._prev = self
			node.index = self.index + 1 # Updated index

	@next.deleter
	def next(self):
		self.next = None

	def insert_next(self, data: Data):
		self.next = Node(data,
			next = self.next,
			prev = self
			index = self.index + 1 # Updated index
		)

	def remove_next(self) -> Data:
		if self.next is None:
			raise IndexError("Remove from orhpan node")

		data      = self.next.data
		self.next = self.next.next

		if self.next:
					current = self.next
					while current:
						current.index = current.prev.index + 1  # Recalculate subsequent indexes
						current = current.next

		return data


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

	def insert_prev(self, data: Data):
		self.prev = Node(data,
			next = self,
			prev = self.prev,
			index=self.index - 1
		)

	def remove_prev(self) -> Data:
		if self.prev is None:
			raise IndexError("Remove from orhpan node")

		data      = self.prev.data
		self.prev = self.prev.prev

		return data


class List[Data]:

	def __init__(self):
		del self.head
		del self.tail

	@property
	def size(self) -> int:
		"""Is there a way to count how many elements the list has?

		First find a wat to do this by simply iterating all elements in the list:
		This can be done either with a while loop,
		or using the `Iterable` nature of `List` implemented in `List.__iter__`.

		Next see if you can provide a faster way, utilizing the `index` attribute of nodes.
		"""
		...


	def __bool__(self) -> bool:
		return self.head is not self.tail is not None

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


	@property
	def head(self) -> Node[Data] | None:
		return self._head

	@head.setter
	def head(self, node: Node[Data] | None):
		self._head = node

		if self._tail is None:
			self._tail = self._head

	@head.deleter
	def head(self):
		self._head = None
		self._tail = None


	@property
	def tail(self) -> Node[Data] | None:
		return self._tail

	@tail.setter
	def tail(self, node: Node[Data] | None):
		self._tail = node

		if self._head is None:
			self._head = self._tail

	@tail.deleter
	def tail(self):
		self._tail = None
		self._head = None


class Deque[Data](List[Data]):

	def append(self, data: Data):
		"""Add data to the end of the deque."""
		self.tail = Node(data,
			prev = self.tail
		)

	def pop(self) -> Data:
		"""Pop data from the end of the deque."""
		if self.tail is None:
			raise IndexError(f"Pop from empty {self.__class__.__name__.lower()}")

		data      = self.tail.data
		self.tail = self.tail.prev

		if self.tail is None:
			del self.tail

		else:
			del self.tail.next

		return data


	def prepend(self, data: Data):
		"""Add data to the beginning of the deque."""
		self.head = Node(data,
        	next=self.head)
		

        if self.head.next:  !!!!!!!
            self.head.index = 0
            current = self.head.next
            while current:
                current.index = current.prev.index + 1
                current = current.next

	def pull(self) -> Data:
		"""Pop data from the beginning of the deque."""
		 if self.head is None:
            raise IndexError("...")


class Stack[Data](Deque[Data]):

	def push(self, data: Data):
		super().append(data)


class Queue[Data](Deque[Data]):

	def enqueue(self, data: Data):
		super().append(data)

	def dequeue(self) -> Data:
		return super().pull()
