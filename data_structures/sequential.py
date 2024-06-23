from __future__ import annotations

import collections
import heapq
import typing


Data = typing.TypeVar("Data", bound = typing.Any)


class Stack[Data](collections.deque[Data]):

	def push(self, data: Data) -> None:
		self.append(data)

	def pop(self) -> Data:
		return self.pop()


class Queue[Data](collections.deque[Data]):

	def enqueue(self, data: Data) -> None:
		self.append(data)

	def dequeue(self) -> Data:
		return self.popleft()


class PriorityQueue[Data](list[Data]):

	def insert(self, data: Data) -> None:
		heapq.heappush(self, data)

	def pull(self) -> Data:
		return heapq.heappop(self)

	def replace(self, data: Data) -> Data:
		return heapq.heappushpop(self, data)
