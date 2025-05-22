from __future__ import annotations


from datetime import datetime
from dataclasses import dataclass
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
	birthday: datetime
	address: Address
	phone: Phone


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
	def born(cls, *args, **kwargs) -> Self:
		return cls(*args,
			birthday = datetime.now(),
		**kwargs)

	@property
	def age(self) -> int:
		return (datetime.now() - self.birthday).days // 365

	def greet(self) -> str:
		return f"Hello, my name is {self.name} and I am {self.age} years old."

	def add(self, other: Self) -> None:
		self.friends.add(other)
		other.friends.add(self)
