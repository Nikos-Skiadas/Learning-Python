from __future__ import annotations


from datetime import datetime
from dataclasses import dataclass, field
from typing import ClassVar, Self


@dataclass
class Person:

	count: ClassVar[int] = 0

	name: str
	birthday: datetime = field(default_factory = datetime.now)

	friends: list[Self] = field(default_factory = list)


	def __post_init__(self):
		self.__class__.count += 1

	def __del__(self):
		self.__class__.count -= 1


	@classmethod
	def born(cls, name: str) -> Self:
		return cls(name, datetime.now())

	@property
	def age(self) -> int:
		return (datetime.now() - self.birthday).days // 365

	def greet(self) -> str:
		return f"Hello, my name is {self.name} and I am {self.age} years old."

	def add(self, other: Self) -> None:
		self.friends.append(other)
		other.friends.append(self)
