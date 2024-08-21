from __future__ import annotations


class Node[Data]:

	def __init__(self,
		data:      Data,
		next: Node[Data] | None = None,
	):
		self.data = data
		self.next = next


	def insert(self, data: Data):
		"""Implement function `insertAfter` as a `Node` method but with `data` instead of a `newNode`.

		https://en.wikipedia.org/wiki/Linked_list#Singly_linked_lists
		"""
		self.next = Node(
			data = data,
			next = self.next,
		)

	def remove(self) -> Data:
		"""Implement function `removeAfter` as a `Node` method.

		https://en.wikipedia.org/wiki/Linked_list#Singly_linked_lists
		"""
		...


class List[Data]:

	def __init__(self):
		self.head: Node[Data] | None = None

	def __bool__(self) -> bool:
		return self.head is None


	def append(self, data: Data):
		"""Implement function `insertBeginning` as a `List` method but with `data` instead of a `newNode`.

		https://en.wikipedia.org/wiki/Linked_list#Singly_linked_lists
		"""
		self.head = Node(
			data = data,
			next = self.head,
		)

	def pop(self) -> Data:
		"""Implement function `removeBeginning` as a `List` method.

		https://en.wikipedia.org/wiki/Linked_list#Singly_linked_lists
		"""
		...
