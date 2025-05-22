from __future__ import annotations


from datetime import datetime, timedelta
from typing import Self


class Person:

	count = 0


	def __init__(self, name: str, birthday: str):
		self.name = name
		self.birthday = datetime.fromisoformat(birthday)
		self.friends = set()

		self.__class__.count += 1

	def __del__(self):
		self.__class__.count -= 1

	def __repr__(self):
		return f"{self.__class__.__name__}(name={self.name}, age={self.age})"

	def __hash__(self):
		return hash(self.name)

	def __eq__(self, other: Self) -> bool:
		return self.name == other.name


	@classmethod
	def born(cls, name: str) -> Self:
		return cls(name, datetime.now().isoformat())

	@property
	def age(self) -> int:
		return (datetime.now() - self.birthday).days // 365

	def greet(self) -> str:
		return f"Hello, my name is {self.name} and I am {self.age} years old."

	def add(self, other: Person) -> None:
		self.friends.add(other)
		other.friends.add(self)
