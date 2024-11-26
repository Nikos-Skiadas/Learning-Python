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


class CSP(csp.CSP):

	"""Proxy class for type-hinting parameters required to define a CSP.

	Lock parameters to keyword-only, for a more readable code.
	"""

	def __init__(self, *,
		variables: Names | None = None,
		domains: Domains,
		neighbors: Domains,
		constraints: Constraint,
	):
		super().__init__(
			variables,
			domains,
			neighbors,
			constraints,
		)


def collect(*constraints: Constraint) -> Constraint:

	"""Collect all constraints into one constraint.

	Return a collective constraint.
	"""
	def collective(*args) -> bool:
		return all(constraint(*args) for constraint in constraints)

	return collective


class ExamTimetabling(CSP):

	exams = pandas.read_csv("h3-data.csv").rename(
		columns = {
			"Εξάμηνο": "semester",
			"Μάθημα": "course",
			"Καθηγητής": "teacher",
			"Δύσκολο (TRUE/FALSE)": "is_hard",
			"Εργαστήριο (TRUE/FALSE)": "has_lab",
		}
	).set_index("semester")


	@staticmethod
	def semester_constraint(
		A: str,
		a: int,
		B: str,
		b: int,
	) -> bool:
		return A == B or day(a) != day(b) or ExamTimetabling.exams[A].semester != ExamTimetabling.exams[B].semester

	@staticmethod
	def has_lab_constraint(
		A: str,
		a: int,
	) -> bool:
		return not ExamTimetabling.exams[A].has_lab or slot(a) < 2

	@staticmethod
	def is_hard_constraint(
		A: str,
		a: int,
		B: str,
		b: int,
	) -> bool:
		return A == B or not ExamTimetabling.exams[A].is_hard or not ExamTimetabling.exams[B].is_hard or abs(day(a) - day(b)) >= 2



	def __init__(self, *,
		exams_file: str,
		num_days: int = 0,
		num_slots: int = 0,
	):
		slots = list(range(num_days * num_slots))

		super().__init__(
			variables = self.exams.course.to_list(),
			domains = UniformDict(slots),
			neighbors = UniformDict(slots),
			constraints = collect(
				ExamTimetabling.semester_constraint,
				ExamTimetabling.has_lab_constraint,
				ExamTimetabling.is_hard_constraint,
			),
		)
