from __future__ import annotations


from datetime import datetime

from typing import Self



class Person:

	count: int = 0


	def __init__(self, name: str, birthday: datetime = datetime.now(), friends: list[Self] | None = None) -> None:
		self.name = name
		self.birthday = birthday

		self.friends = friends if friends is not None else list()

		self.__class__.count += 1

	def __del__(self):
		self.__class__.count -= 1


	@classmethod
	def born(cls, name: str) -> Self:
		return cls(name)

	@property
	def age(self) -> int:
		return (datetime.now() - self.birthday).days // 365

	def greet(self) -> str:
		return f"Hello, my name is {self.name} and I am {self.age} years old."

	def add(self, other: Self) -> None:
		self.friends.append(other)
		other.friends.append(self)
