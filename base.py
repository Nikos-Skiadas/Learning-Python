from __future__ import annotations


from collections import deque
from heapq import heapify, heappop, heappush, heappushpop, merge


class stack[ItemType](deque[ItemType]):

	def push(self, item: ItemType) -> None:
		super().append(item)

	def pop(self) -> ItemType:
		return super().pop()


class queue[ItemType](deque[ItemType]):

	def enqueue(self, item: ItemType) -> None:
		super().append(item)

	def dequeue(self) -> ItemType:
		return super().popleft()


class heap[ItemType](list[ItemType]):

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		heapify(self)

	def insert(self, item: ItemType) -> None:
		heappush(self, item)

	def extract(self) -> ItemType:
		return heappop(self)

	def replace(self, item: ItemType) -> ItemType:
		return heappushpop(self, item)

	def merge(self, other: heap[ItemType]) -> None:
		for item in other:
			self.insert(item)

	def __radd__(self, other: heap[ItemType]) -> heap[ItemType]:
		return heap[ItemType](merge(self, other)) if other else self

	def __add__(self, other: heap[ItemType]) -> heap[ItemType]:
		return other + self

	def __iadd__(self, other: heap[ItemType]) -> heap[ItemType]:
		self.merge(other)

		return self
