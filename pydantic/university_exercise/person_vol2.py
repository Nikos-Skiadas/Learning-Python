from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Literal, Optional
import itertools

@dataclass
class Name:
    first: str
    middle: Optional[str] = None
    last: str

    def __repr__(self) -> str:
        if self.middle:
            return f"{self.first} {self.middle} {self.last}"
        return f"{self.first} {self.last}"

@dataclass
class Email:
    user: str
    domain: str
    suffix: str = "com"

    def __repr__(self) -> str:
        return f"{self.user}@{self.domain}.{self.suffix}"

@dataclass
class Phone:
    country_code: int
    number: str

    def __repr__(self) -> str:
        return f"+{self.country_code} {self.number}"

@dataclass
class Person:

    """
    Person class initialization for university system.

    Proposition:
        id: UUnique identifier assigned automatically
        name: Name object (first, optional middle, last)
        role: either 'student' or 'teacher'
        email: Email object
        phone: Phone object
        created_at: Exact time of creation
    """
    # auto-id counter
    _id_iter: ClassVar[itertools.count] = itertools.count(1)

    id: int = field(init=False)

	"""
	There is an id attribute on each instance of type int.
	But it should not be included as a parameter to __init__.
	"""

    name: Name
    role: Literal['student', 'teacher']
    email: Email
    phone: Phone
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # assign unique id on creation
        self.id = next(self._id_iter)

    @property
    def full_name(self) -> str:
        return repr(self.name)

    def __repr__(self) -> str:
        return f"<Person id={self.id} name={self.full_name} role={self.role}>"

@dataclass
class Student(Person):
    """
    Student subclass: tracks enrolled courses and grades (must include somehow teachers or add the to the courses and make it a dictionary).
    """
    # maps course identifiers to an optional grade
    courses: dict[str, Optional[float]] = field(default_factory=dict)
	"""
	Whenever you create a new Student, call dict() (i.e. make a fresh, empty dict) and assign that to self.courses
	"""

    def enroll(self, course: str) -> None:
        """Enroll the student in a new course (no grade yet)."""
        if course in self.courses:
            raise ValueError(f"Already enrolled in course {course!r}")
        self.courses[course] = None

    """
	Add this as a teacher method


	def add_grade(self, course: str, grade: float) -> None:
        if course not in self.courses:
            raise ValueError(f"Not enrolled in course {course!r}")
        if not (0.0 <= grade <= 100.0):
            raise ValueError("Grade must be between 0 and 100")
        self.courses[course] = grade
	"""

    def average_grade(self) -> float:
        grades = [g for g in self.courses.values() if g is not None]
        return sum(grades) / len(grades) if grades else 0.0

