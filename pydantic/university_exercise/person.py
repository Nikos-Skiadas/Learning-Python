'''HOMEWORK:

Below is a code snippet that defines several data classes representing a person, their name, email, address, phone number, and other related information.

Write a `secretary.py` module that contains a `Secretary` class that represents a university department. Implement additional classes as needed, for example a `University` class.

Write a `course.py` module that contains a `Course` class that represents a university course. Implement additional classes as needed, for example a `Student` class.

The `Person` class may need an update (for example an `id` attribute?).
Also mind the special type of `Person`, a `Student` or a `Teacher.
Also mind the interaction between `Person`, `Secretary` and `Course`.

Finally do not forget to implement methods. For example:
-	a course may have students and teachers
	-	for students hold grades as well?
-	a person may have courses with (or without) grades
	-	if a student, then courses may have grades
	-	if a teacher, then courses may not have grades
-	a secretary may have students and courses
-	a student may have an average grade (running average of all grades in all courses)
-	...

Look up the old `C++` homework about this project and fill in whatever else is necessary.
'''


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
