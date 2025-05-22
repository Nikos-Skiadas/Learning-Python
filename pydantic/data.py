from __future__ import annotations


from datetime import datetime
from dataclasses import dataclass
from typing import ClassVar, Self


class Person:

	count: int = 0


	def __init__(self, name: str, birthday: datetime | str) -> None:
		if isinstance(birthday, str):
			birthday = datetime.fromisoformat(birthday)

		self.name = name
		self.birthday = birthday

		self.friends: set[Self] = set()

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
		return cls(name, datetime.now())

	@property
	def age(self) -> int:
		return (datetime.now() - self.birthday).days // 365

	def greet(self) -> str:
		return f"Hello, my name is {self.name} and I am {self.age} years old."

	def add(self, other: Self) -> None:
		self.friends.add(other)
		other.friends.add(self)


@dataclass
class PersonData:

	count: ClassVar[int] = 0

	name: str
	birthday: datetime


	def __post_init__(self):
		self.__class__.count += 1
		self.friends: set[Self]  = set()

		if isinstance(self.birthday, str):
			self.birthday = datetime.fromisoformat(self.birthday)

	def __del__(self):
		self.__class__.count -= 1

	def __hash__(self):
		return hash(self.name)

	def __eq__(self, other: Self) -> bool:
		return self.name == other.name


	@classmethod
	def born(cls, name: str) -> Self:
		return cls(name, datetime.now())

	@property
	def age(self) -> int:
		return (datetime.now() - self.birthday).days // 365

	def greet(self) -> str:
		return f"Hello, my name is {self.name} and I am {self.age} years old."

	def add(self, other: Self) -> None:
		self.friends.add(other)
		other.friends.add(self)
