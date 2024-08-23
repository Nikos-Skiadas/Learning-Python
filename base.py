from __future__ import annotations


import collections
import heapq


class stack[ItemType](collections.deque[ItemType]):

	def push(self, item: ItemType) -> None:
		super().append(item)

	def pop(self) -> ItemType:
		return super().pop()


class queue[ItemType](collections.deque[ItemType]):

	def enqueue(self, item: ItemType) -> None:
		super().append(item)

	def dequeue(self) -> ItemType:
		return super().popleft()


class priority_queue[ItemType](list[ItemType]):

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

		heapq.heapify(self)

	def __add__(self, other: priority_queue[ItemType]) -> priority_queue[ItemType]:
		return other + self

	def __radd__(self, other: priority_queue[ItemType]) -> priority_queue[ItemType]:
		return priority_queue[ItemType](heapq.merge(self, other)) if other else self

	def __iadd__(self, other: priority_queue[ItemType]) -> priority_queue[ItemType]:
		self.extend(other)

		return self


	def extend(self, other: priority_queue[ItemType]) -> None:
		super().extend(other)

		heapq.heapify(self)

	def insert(self, item: ItemType) -> None:
		heapq.heappush(self, item)

	def extract(self) -> ItemType:
		return heapq.heappop(self)

	def replace(self, item: ItemType) -> ItemType:
		return heapq.heappushpop(self, item)
