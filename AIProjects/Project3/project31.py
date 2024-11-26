from __future__ import annotations


import collections
import typing

import pandas

import csp


type Names = list[typing.Hashable]
type Domains = dict[typing.Hashable, list]
type Constraint = typing.Callable


def day(numerator: int,
	denominator: int = 3,
) -> int:
	return numerator // denominator


def slot(numerator: int,
	denominator: int = 3,
) -> int:
	return numerator % denominator


class UniformDict(collections.defaultdict):

	"""A better implementation against `CPS.UniversalDict`.

	This one outright utilizes `collections.defaultdict` for `__missing__`.

	This however is unsafe as values may be modified nonetheless.
	"""

	def __init__(self, value):
		super().__init__(lambda: value)


def collective(*constraints: Constraint) -> Constraint:
	"""Collect all constraints into one constraint.

	Return a collective constraint.
	"""
	def collective_constraint(*args) -> bool:
		return all(constraint(*args) for constraint in constraints)

	return collective_constraint


def binary(constraint: Constraint) -> Constraint:
	"""Provide the base for binary constraints.

	Binary constraints should only focus on non trivial constraints.
	This decorator takes care of the trivial condition A == B.
	"""
	def binary_constraint(
		A,
		a,
		B,
		b,
	) -> bool:
		return A == B or constraint(
			A,
			a,
			B,
			b,
		)

	return binary_constraint


class ExamTimetabling(csp.CSP):


	@binary
	def course_constraint(self,
		A: str,
		a: int,
		B: str,
		b: int,
	) -> bool:
		"""No exams may overlap.

		Each exams shall ahve its own hour.
		"""
		return a != b

	@binary
	def semester_constraint(self,
		A: str,
		a: int,
		B: str,
		b: int,
	) -> bool:
		"""No same semester exams in one day.

		Either the days are different, or
		the exam semesters are different.
		"""
		return day(a) != day(b) or self.exams[A].semester != self.exams[B].semester

	@binary
	def has_lab_constraint(self,
		A: str,
		a: int,
		B: str,
		b: int,
	) -> bool:
		"""Courses with labs are all examined in one day with 2 slots..

		Either one of the exams has a lab and an approriate slot
		"""
		return (not self.exams[A].has_lab or (slot(a) != 2 and (day(a) != day(b) or slot(b) == (slot(a) + 2) % 3))) \
			or (not self.exams[B].has_lab or (slot(b) != 2 and (day(b) != day(a) or slot(a) == (slot(b) + 2) % 3)))

	@binary
	def is_hard_constraint(self,
		A: str,
		a: int,
		B: str,
		b: int,
	) -> bool:
		"""No hard exams closer than 2 days.

		Either at least one of the exams is easy, or
		the exams are at least 2 days apart.
		"""
		return not self.exams[A].is_hard \
			or not self.exams[B].is_hard or abs(day(a) - day(b)) >= 2


	def __init__(self, *,
		exams_file: str,
		num_days: int = 0,
		num_slots: int = 0,
	):
		self.exams = pandas.read_csv(exams_file).rename(
			columns = {
				"Εξάμηνο": "semester",
				"Μάθημα": "course",
				"Καθηγητής": "teacher",
				"Δύσκολο (TRUE/FALSE)": "is_hard",
				"Εργαστήριο (TRUE/FALSE)": "has_lab",
			}  # rename silly greek keys (columns) to ASCII ones
		).set_index("course")  # index exam table by course name

		hours = list(range(num_days * num_slots))

		super().__init__(
			variables = self.exams.course.to_list(),
			domains = UniformDict(hours),
			neighbors = UniformDict(hours),
			constraints = collective(
				self.semester_constraint,
				self.has_lab_constraint,
				self.is_hard_constraint,
			),
		)
