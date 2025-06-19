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


from datetime import datetime, timezone
from dataclasses import dataclass, field
from statistics import StatisticsError, mean
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

	count: ClassVar[int] = 1

	id: int = field(init = False, default_factory = lambda: Person.count)
	name: Name
	email: Email
	address: Address
	phone: Phone

	created: datetime = field(default_factory = lambda: datetime.now(timezone.utc))


	def __post_init__(self):
		self.__class__.count += 1

	def __del__(self):
		self.__class__.count -= 1

	def __hash__(self) -> int:
		return hash(self.id)


	@property
	def year(self) -> int:
		return (datetime.now() - self.created).days // 365


@dataclass
class Student(Person):

	courses: dict[Course, float | None] = field(default_factory = dict)


	@property
	def grade(self) -> float:
		try: return mean(grade for grade in self.courses.values() if grade is not None)
		except StatisticsError:	return 0.

	@property
	def graduable(self) -> bool:
		return self.grade >= .5 and self.year >= 5  # NOTE: 5 years is arbitrary, fetch from university policy instead.


	def register_to(self, course: Course,
		grade: float | None = None,
	):
		self.courses.setdefault(course, grade)
		course.students.setdefault(self, grade)



@dataclass
class Teacher(Person):

	courses: set[Course] = field(default_factory = set)


	def assign(self, course: Course) -> None:
		self.courses.add(course)
		course.teacher = self


@dataclass
class Course:

	code: str
	name: str
	year: int

	teacher: Teacher | None = None
	students: dict[Student, float | None] = field(default_factory = dict)

	optional: bool = False

	created: datetime = field(default_factory = lambda: datetime.now(timezone.utc))


	def __repr__(self) -> str:
		return f"{self.name} ({self.code})"

	def __hash__(self) -> int:
		return hash(self.code)


	def assign_to(self, teacher: Teacher):
		self.teacher = teacher
		teacher.courses.add(self)

	def register(self, student: Student,
		grade: float | None = None,
	):
		self.students.setdefault(student, grade)
		student.courses.setdefault(self, grade)


@dataclass
class Department:

	name: str
	domain: str
	address: Address
	phone: Phone

	teachers: set[Teacher] = field(default_factory = set)
	students: set[Student] = field(default_factory = set)

	courses: set[Course] = field(default_factory = set)
