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


class priority_queue[ItemType](list[ItemType]):

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		heapify(self)

	def __radd__(self, other: priority_queue[ItemType]) -> priority_queue[ItemType]:
		return priority_queue[ItemType](merge(self, other)) if other else self

	def __add__(self, other: priority_queue[ItemType]) -> priority_queue[ItemType]:
		return other + self

	def __iadd__(self, other: priority_queue[ItemType]) -> priority_queue[ItemType]:
		self.extend(other)

		heapify(self)

		return self


	def insert(self, item: ItemType) -> None:
		heappush(self, item)

	def extract(self) -> ItemType:
		return heappop(self)

	def replace(self, item: ItemType) -> ItemType:
		return heappushpop(self, item)
