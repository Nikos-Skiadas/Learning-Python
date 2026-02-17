'''HOMEWORK:

Below is a id snippet that defines several data classes representing a person, their name, email, address, phone number, and other related information.

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
	suffix: str = "edu"


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
	id: int


	def __repr__(self) -> str:
		return f"{self.name} ({self.id})"


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
		return f"{self.country.id:+} {self.number}"


@dataclass
class Person:

	count: ClassVar[int] = 1

	id: int = field(init = False, default_factory = lambda: Person.count)
	name: Name
	address: Address
	phone: Phone

	department: Department | None = None

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

	@property
	def email(self) -> Email | None:
		handle = f"{self.name}.{self.id}".lower().replace(" ",".")

		return Email(handle, self.department.domain) if self.department is not None else None


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


	def register(self, course: Course):
		if self.department is None: raise ValueError("Student and course must belong to the same department.")
		if course.department is None: raise ValueError("Course must belong to a department.")

		self.courses.setdefault(course)
		course.students.setdefault(self)


@dataclass
class Teacher(Person):

	courses: set[Course] = field(default_factory = set)


	def evaluate(self, student: Student, course: Course, grade: float):
		if student not in course.students: raise ValueError("Student is not enrolled in the course.")
		if course not in self.courses: raise ValueError("Teacher is not assigned to the course.")
		if course.department != self.department: raise ValueError("Wrong department for course.")

		# What if the student has already a grade for this course? Maybe update it only if greater than the existing one?

		student.courses[course] = grade
		course.students[student] = grade


@dataclass
class Course:

	id: str
	name: str
	year: int

	department: Department | None = None
	teacher: Teacher | None = None
	students: dict[Student, float | None] = field(default_factory = dict)

	optional: bool = False

	created: datetime = field(default_factory = lambda: datetime.now(timezone.utc))


	def __repr__(self) -> str:
		return f"{self.name} ({self.id})"

	def __hash__(self) -> int:
		return hash(self.id)


@dataclass
class Department:

	id: str
	name: str
	domain: str
	address: Address
	phone: Phone

	teachers: set[Teacher] = field(default_factory = set)
	students: set[Student] = field(default_factory = set)
	courses : set[Course ] = field(default_factory = set)


	def __hash__(self) -> int:
		return hash(self.id)


	def hire(self, teacher: Teacher):
		self.teachers.add(teacher)
		teacher.department = self

	def enroll(self, student: Student):
		self.students.add(student)
		student.department = self

	def assign(self, course: Course, teacher: Teacher):
		self.courses.add(course)
		course.department = self

		teacher.courses.add(course)
		course.teacher = teacher
