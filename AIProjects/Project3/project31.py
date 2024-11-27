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


def conflict(constraint: Constraint) -> Constraint:
	"""Provide the base for binary constraints.

	Binary constraints should only focus on non trivial constraints.
	This decorator takes care of the trivial condition A == B.
	"""
	def conflict_constraint(
		A,
		a,
		B,
		b,
	) -> bool:
		return A == B or (
			a != b and constraint(
				A,
				a,
				B,
				b,
			)
		)

	return conflict_constraint


class ExamTimetabling(csp.CSP):

	@conflict
	def has_lab_constraint(self,
		A: str,
		a: int,
		B: str,
		b: int,
	) -> bool:
		"""Courses with labs are all examined in one day with 2 slots
		"""
		return (not self.exams.has_lab[A] or (slot(a) != 2 and (day(a) != day(b) or slot(b) == (slot(a) + 2) % 3))) \
			or (not self.exams.has_lab[B] or (slot(b) != 2 and (day(b) != day(a) or slot(a) == (slot(b) + 2) % 3)))

	@staticmethod
	def day_attribute(attribute: str) -> Constraint:
		@conflict
		def day_attribute_constraint(self,
			A,
			a,
			B,
			b,
		) -> bool:
			return day(a) != day(b) or self.exams[attribute][A] != self.exams[attribute][B]

		return day_attribute_constraint


	@conflict
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
		return not self.exams.is_hard[A] \
			or not self.exams.is_hard[B] or abs(day(a) - day(b)) >= 2


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
			neighbors = {},
			constraints = collective(
				ExamTimetabling.day_attribute("semester"),
				ExamTimetabling.day_attribute("teacher"),
				self.has_lab_constraint,
				self.is_hard_constraint,
			),
		)
