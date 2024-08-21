from __future__ import annotations


class Node[Data]:

	def __init__(self,
		data:      Data,
		next: Node[Data] | None = None,
	):
		self.data = data
		self.next = next


class List[Data]:

	def __init__(self):
		self.head: Node[Data] | None = None
