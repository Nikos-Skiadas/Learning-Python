from __future__ import annotations


class Node[Data]:

	def __init__(self,
		data:      Data,
		next: Node[Data] | None = None,
	):
		self.data = data
		self.next = next


	def insert(self, data: Data):
		self.next = Node(
			data = data,
			next = self.next,
		)



class List[Data]:

	def __init__(self):
		self.head: Node[Data] | None = None

	def __bool__(self) -> bool:
		return self.head is None


	def append(self, data: Data):
		self.head = Node(
			data = data,
			next = self.head,
		)
