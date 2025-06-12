from __future__ import annotations


from datetime import datetime
from dataclasses import dataclass, field
from typing import ClassVar, Self



@dataclass
class Name:

	first: str
	last: str
	middle: str | None = None


	def __repr__(self) -> str:
		return f"{self.first} {self.middle} {self.last}" if self.middle is not None else f"{self.first} {self.last}"


@dataclass
class Email:

	user: str
	domain: str
	suffix: str = "com"


	def __repr__(self) -> str:
		return f"{self.user}@{self.domain}.{self.suffix}"


@dataclass
class City:

	name: str
	zip: int
	country: Country


	def __repr__(self) -> str:
		return f"{self.zip} {self.name}, {self.country.name}"

@dataclass
class Country:

	name: str
	code: int


	def __repr__(self) -> str:
		return f"{self.name} ({self.code})"


@dataclass
class Address:

	street: str
	number: int
	city: City


	def __repr__(self) -> str:
		return f"{self.street} {self.number}, {self.city}"


@dataclass
class Phone:

	number: int
	country: Country


	def __repr__(self) -> str:
		return f"{self.country.code:+} {self.number}"


@dataclass
class Person:

	count: ClassVar[int] = 0


	name: Name
	email: Email
	address: Address
	phone: Phone

	birthday: datetime = field(default_factory = datetime.now)
	friends: list[Self] = field(default_factory = list)


	def __post_init__(self):
		self.__class__.count += 1

	def __del__(self):
		self.__class__.count -= 1


	@property
	def age(self) -> int:
		return (datetime.now() - self.birthday).days // 365

	def greet(self) -> str:
		return f"Hello, my name is {self.name} and I am {self.age} years old."

	def add(self, other: Self) -> None:
		self.friends.append(other)
		other.friends.append(self)
