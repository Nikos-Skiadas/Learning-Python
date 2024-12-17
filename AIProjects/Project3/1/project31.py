from __future__ import annotations


import collections
import os
import typing

import pandas

import csp


type Constraint = typing.Callable[..., bool]


def read(exams_file: str) -> pandas.DataFrame:
	return pandas.read_csv(exams_file).rename(
			columns = {
				"Εξάμηνο": "semester",
				"Μάθημα": "course",
				"Καθηγητής": "teacher",
				"Δύσκολο (TRUE/FALSE)": "is_hard",
				"Εργαστήριο (TRUE/FALSE)": "has_lab",
			}  # rename silly greek keys (columns) to ASCII ones
		).set_index("course")  # index exam table by course name


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

	exams: pandas.DataFrame


	@staticmethod
	@conflict
	def has_lab_constraint(
		A: str,
		a: int,
		B: str,
		b: int,
	) -> bool:
		"""Courses with labs are all examined in one day with 2 slots
		"""
		return (not ExamTimetabling.exams.has_lab[A] or (slot(a) != 2 and (day(a) != day(b) or slot(b) == (slot(a) + 2) % 3))) \
		   and (not ExamTimetabling.exams.has_lab[B] or (slot(b) != 2 and (day(b) != day(a) or slot(a) == (slot(b) + 2) % 3)))

	@staticmethod
	def different_day(attribute: str) -> Constraint:
		@conflict
		def day_attribute_constraint(
			A,
			a,
			B,
			b,
		) -> bool:
			return day(a) != day(b) or ExamTimetabling.exams[attribute][A] != ExamTimetabling.exams[attribute][B]  # type: ignore

		return day_attribute_constraint


	@staticmethod
	@conflict
	def is_hard_constraint(
		A: str,
		a: int,
		B: str,
		b: int,
	) -> bool:
		"""No hard exams closer than 2 days.

		Either at least one of the exams is easy, or
		the exams are at least 2 days apart.
		"""
		return not ExamTimetabling.exams.is_hard[A] \
			or not ExamTimetabling.exams.is_hard[B] or abs(day(a) - day(b)) >= 2


	def __init__(self, *,
		exams_file: str,
		num_days: int = 0,
		num_slots: int = 0,
	):
		ExamTimetabling.exams = read(exams_file)

		hours = list(range(num_days * num_slots))
		courses = self.exams.index.to_list()

		super().__init__(
			variables = courses,
			domains = UniformDict(hours),
			neighbors = UniformDict(courses),
			constraints = collective(
				ExamTimetabling.different_day("semester"),
				ExamTimetabling.different_day("teacher"),
				ExamTimetabling.has_lab_constraint,
				ExamTimetabling.is_hard_constraint,
			),
		)


	def to_pandas(self, assignment) -> pandas.DataFrame:
		return ExamTimetabling.exams.assign(
			day_and_slot = assignment,
		).sort_values("day_and_slot")


	def display(self, assignment):
		print(self.to_pandas(assignment))


if __name__ == "__main__":
	filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "h3-data.csv")

	problem = ExamTimetabling(
		exams_file = filepath,
		num_days = 21,
		num_slots = 3,
	)

	for backtracking_method in [
		csp.forward_checking,
		csp.mac,
	]:
		problem.to_pandas(
			csp.backtracking_search(
				csp = problem,
				select_unassigned_variable = csp.mrv,
				order_domain_values = csp.lcv,
				inference = backtracking_method,
			)
		).to_csv(filepath.replace(".csv", f".{backtracking_method.__name__}.csv"))

	problem.to_pandas(
		csp.min_conflicts(
			csp = problem,
		)
	).to_csv(filepath.replace(".csv", f".{csp.min_conflicts.__name__}.csv"))
