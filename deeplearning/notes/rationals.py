"""Set-constructive definition of the rational numbers.

The sequel to `integers.py`, for §1.5.3 ("ℚ: Rational numbers"): represent ℚ on top of the ℤ
built there, and let arithmetic fall out of §1.5.3's definitions instead of out of python's
own arithmetic.

The construction is a quotient set again, in the sense of §1.4, but of a *restricted* product:

	ℚ = (ℤ × ℤ*)/∼         where (m, n) ∼ (m', n') <=> m·n' = m'·n,  and ℤ* is ℤ without 0

The intuition is that the pair (m, n) stands for m/n, which is why the relation is stated
with two multiplications rather than a division that ℤ does not have. So one rational is
*many* pairs, exactly as one integer was many pairs:

	1/2 = (1, 2) = (2, 4) = (3, 6) = ...   and also (-1, -2) = (-2, -4) = ...
	  3 = (3, 1) = (6, 2) = (9, 3) = ...
	  0 = (0, 1) = (0, 2) = (0, 3) = ...

Everything `integers.py` taught still applies: a bare class, two members, equality that is
not structural, `__hash__` repaired by hand, a canonical representative that is the handle on
the class. Read that file first; this one assumes it. What follows is only what is *new*.

**1. The underlying set is restricted, so the constructor can fail.** ℕ and ℤ were built on
ℕ and ℕ × ℕ — every pair you could write down named something. Here the second member must
be non-zero, and `Q(1, 0)` is not a rational number that happens to be awkward, it is not a
rational number at all. This is the first constructor in the chapter that rejects an argument
for a *mathematical* reason rather than a plumbing one, and `ZeroDivisionError` is what python
calls that.

**2. The canonical form needs an algorithm.** §1.5.3 says the canonical representative is the
unique (m, n) with gcd(m, n) = 1 and n > 0 — lowest terms. ℤ's canonical form was a loop that
peeled one off both members; this one needs a greatest common divisor and an exact division,
and ℤ has neither. They have to be built first, out of what ℤ does have, which is why this
file opens with arithmetic on ℤ before it defines anything about ℚ.

**3. Representatives grow while values shrink.** This one is not in the notes, and it is the
sharpest thing in the exercise. `integers.py` deliberately did *not* reduce its pairs — an
instance holds a representative, and that is the whole point of a quotient. Harmless there,
because nothing iterated. Here the gcd iterates, and the arithmetic compounds: `Z(8) - Z(6)`
is the pair (8, 6), not (2, 0), and after three more subtractions the value 2 is being carried
as (18, 16). The numbers in the *members* grow without bound while the number they denote
shrinks to nothing — and since ℕ's equality is exponential in the size of its operands, the
whole thing seizes up. `reduced` below exists for exactly this, and every loop in this file
calls it. The lesson is worth more than the function: a representation you never normalise is
free until something iterates over it.

**4. The order is sign-sensitive.** §1.5.3's ordering rule has two branches, and the second is
easy to miss:

	[(m, n)] ≤ [(m', n')]  <=>  (m·n' ≤ m'·n  ∧  n·n' > 0)  ∨  (m·n' ≥ m'·n  ∧  n·n' < 0)

Cross-multiplying reverses the inequality when the denominators have opposite signs. `Q(1, -2)`
is −1/2 and is below zero, but the naive single-branch test says otherwise.

**5. ℚ is a field.** ℤ gave every number an additive inverse; ℚ gives every *non-zero* number
a multiplicative one. `reciprocal` is this chapter's `__neg__` — the reason the construction
exists — and it swaps the two members just as negation did, which is a pleasing rhyme and not
a coincidence: both undo an operation by exchanging the roles of the pair.

A standing cost, and it sets the size of everything below. ℚ sits on ℤ sits on ℕ, and ℕ's
equality is exponential between two numbers built independently — `naturals.py` says so, and
three constructions up it is the binding constraint. Measured here, on *equal* operands, which
is the expensive case: a gcd at magnitude 16 costs 0.002s, at 20 costs 0.021s, at 24 costs
0.35s, at 26 costs 1.3s, doubling every two from there. Unequal operands stay cheap. So keep
numerators and denominators to single digits, and watch the products: `Q(1,2) + Q(1,3)` is
5/6 and instant, while `Q(1,2) + Q(1,3) + Q(1,6)` reaches a denominator of 36 and will not
finish. This is the price of standing on three constructions, not a defect in any of them.

Working through this: the file opens with four functions on ℤ, then the class, laid out in the
same three groups as `integers.py` — representations, operations, relations — each leading
with what is given, then what is to implement. Eleven members carry the exercise and fourteen
are given; each states what it must return, which definition from §1.5.3 it discharges, the
errors it owes the caller, and hints, plus doctests that serve as its specification. Run

	python3 -m doctest deeplearning/notes/rationals.py

from the repository root as the progress bar: silence means done.

References:

* https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
* https://docs.python.org/3/library/exceptions.html#ZeroDivisionError
* https://docs.python.org/3/reference/datamodel.html#object.__hash__
"""


from __future__ import annotations


from typing import Self

from deeplearning.notes.integers import Z
from deeplearning.notes.naturals import N


# ---------------------------------------------------------------------------------------
# ARITHMETIC ON ℤ — three things the integers' own exercise never needed, and the one the
#                   rationals cannot be built without.
# ---------------------------------------------------------------------------------------

# ............................................................................ given


def reduced(value: Z) -> Z:
	"""Return `value` as its canonical representative — the same integer, smaller members.

	Given; not part of the exercise.

	`integers.py` was right not to reduce: an instance holds a representative, and refusing
	to normalise is what makes the class a quotient rather than a pair of numbers. The cost
	of that decision only appears when something iterates, and this file iterates constantly.

	Watch what repeated subtraction does to the members, as against the value:

		a = 6 (6, 0)   b = 8 (8, 0)
		a = 6 (6, 0)   b = 2 (8, 6)        b - a
		a = 4 (12, 8)  b = 2 (8, 6)        a - b
		a = 2 (18, 16) b = 2 (8, 6)        a - b

	The value falls from 6 to 2 while its members climb to 18. ℕ's equality is exponential in
	the size of its operands, so a few more steps of that and nothing returns. Every loop
	below passes its result through here, which keeps the members at the size of the value
	they denote.

	>>> reduced(Z(8) - Z(6))
	2
	>>> tuple(sorted(vars(Z(8) - Z(6)).items()))              # the unreduced pair
	(('a', 8), ('b', 6))
	>>> tuple(sorted(vars(reduced(Z(8) - Z(6))).items()))     # the same integer, reduced
	(('a', 2), ('b', 0))
	"""
	return Z(*value.canonical)


def magnitude(value: Z) -> Z:
	"""Return `value` without its sign, i.e. |value|.

	Given; not part of the exercise.

	`integers.py` never defined `__abs__`, because nothing there needed one. The Euclidean
	algorithm does: it is stated for non-negative integers, and the sign of a gcd is a
	convention rather than a fact. One line, using only the comparison and the negation that
	ℤ already has.

	>>> magnitude(Z(5)), magnitude(Z(-5)), magnitude(Z())
	(5, 5, 0)
	"""
	return value if value >= Z() else -value


def divide(dividend: Z, divisor: Z) -> Z:
	"""Return the exact quotient `dividend / divisor`, which must divide without remainder.

	Given; not part of the exercise.

	ℤ has no division — §1.5.2 defines addition, multiplication, negation and subtraction, and
	stops, because an integer divided by an integer is in general not an integer. That is the
	gap ℚ is built to fill, so this function is deliberately narrow: it is the division that
	*is* exact, which is the only kind `canonical` ever asks for, having just divided out a
	common divisor.

	The method is repeated subtraction, counting. It is the same shape as `gcd` below, and for
	the same reason — subtraction and comparison are all ℤ offers. Each difference goes through
	`reduced`, without which the members grow as described there.

	Raises ZeroDivisionError on a zero divisor, and ValueError when the division leaves a
	remainder — the second being a loud failure rather than a silent wrong answer, since every
	caller in this file believes the division is exact.

	>>> divide(Z(12), Z(4)), divide(Z(12), Z(3)), divide(Z(0), Z(5))
	(3, 4, 0)
	>>> divide(Z(-12), Z(4)), divide(Z(12), Z(-4)), divide(Z(-12), Z(-4))
	(-3, -3, 3)
	>>> divide(Z(7), Z(2))
	Traceback (most recent call last):
		...
	ValueError: 2 does not divide 7 exactly
	>>> divide(Z(7), Z())
	Traceback (most recent call last):
		...
	ZeroDivisionError: division by zero
	"""
	if not divisor:
		raise ZeroDivisionError("division by zero")

	negative = (dividend < Z()) != (divisor < Z())
	remainder, step = magnitude(dividend), magnitude(divisor)
	quotient = Z()

	while step <= remainder:
		remainder = reduced(remainder - step)
		quotient = quotient + 1

	if remainder:
		raise ValueError(f"{divisor} does not divide {dividend} exactly")

	return -quotient if negative else quotient


# ..................................................................... to implement


def gcd(left: Z, right: Z) -> Z:
	"""Return the greatest common divisor of two integers, as a non-negative integer.

	§1.5.3 defines the canonical representative of a rational as the pair in lowest terms —
	gcd(m, n) = 1 with n > 0 — so `canonical` cannot be written until this exists. It is this
	chapter's brain-stretcher, the counterpart of ℕ's `prev` and ℤ's `canonical`.

	Scaffolding, in the order worth thinking about:

	* The Euclidean algorithm you may know divides and takes remainders. You do not have
	  division: `divide` above is exact only, and it is not what this needs. Look instead for
	  the *older* form of the algorithm, the one Euclid actually wrote, which uses nothing but
	  comparison and subtraction — both of which ℤ has.
	* The idea in one line: gcd(a, b) is unchanged if you replace the larger by the difference
	  of the two, because any number dividing both divides their difference as well. Convince
	  yourself of that before coding — it is the whole algorithm.
	* So the loop repeatedly replaces the larger of the two by the difference, and stops when
	  one of them reaches zero. The other is then the answer. Zero in ℤ is falsy, so the loop
	  condition needs no comparison.
	* When one is zero, the answer is the other. There is a tidy way to write "whichever of
	  the two is not zero" that needs no branch at all, given that the other is zero.
	* Sign is a convention: gcd(-8, 12) is 4, not -4. Strip the signs once, at the top, and the
	  loop never has to think about them. `magnitude` is there for that.
	* **Reduce every difference.** This is the point the module docstring makes at length: a
	  difference of two integers is a pair whose members are as large as the operands, and
	  without `reduced` the loop grinds to a halt after a handful of steps. Write it without,
	  watch `gcd(Z(12), Z(18))` hang, then put it in — it is worth seeing once.
	* Check it against a case where the two are equal, one where they are coprime, and one
	  where one of them is zero.

	>>> gcd(Z(12), Z(18)), gcd(Z(4), Z(6)), gcd(Z(9), Z(3))
	(6, 2, 3)
	>>> gcd(Z(7), Z(5)), gcd(Z(6), Z(6))              # coprime, and equal
	(1, 6)
	>>> gcd(Z(-8), Z(12)), gcd(Z(8), Z(-12))          # sign-blind, and never negative
	(4, 4)
	>>> gcd(Z(0), Z(5)), gcd(Z(5), Z(0))              # everything divides zero
	(5, 5)
	"""
	...


class Q:
	"""A rational number, represented by two integers standing for their quotient.

	The members `m` and `n` mean m/n, with `n` never zero, and two rationals with the same
	quotient are the same rational: `Q(1, 2)` and `Q(2, 4)` are both a half. An instance holds
	one representative; `canonical` produces the one in lowest terms, which every class has
	exactly one of.

	The definitions being transcribed, all from §1.5.3:

		[(m, n)] = [(m', n')]  <=>  m·n' = m'·n
		[(m, n)] ≤ [(m', n')]  <=>  (m·n' ≤ m'·n ∧ n·n' > 0) ∨ (m·n' ≥ m'·n ∧ n·n' < 0)

		[(m, n)] + [(m', n')] = [(m·n' + m'·n, n·n')]
		[(m, n)] · [(m', n')] = [(m·m', n·n')]

	Eleven members to implement. The file groups them by what they are; this is the order to
	*write* them in, chosen so that each is solvable by the time you reach it:

		1. `gcd`         — above the class, and nothing here works without it.
		2. `canonical`   — lowest terms, with the sign moved to the numerator.
		3. `__repr__`    — the debugger for everything after.
		4. `__eq__`, then `__hash__`.
		5. `__mul__`, then `__add__`  — multiplication is the easier of the two here.
		6. `reciprocal`, then `__truediv__`  — what makes ℚ a field.
		7. `__bool__`, then `__le__`  — and read §1.5.3's ordering rule twice.

	>>> Q(1, 2), Q(2, 4), Q(3), Q()
	(1/2, 1/2, 3, 0)
	>>> Q(1, 2) == Q(2, 4)                    # one rational, two representatives
	True
	>>> Q(1, 2).canonical, Q(2, 4).canonical  # ... reducing to the same lowest terms
	((1, 2), (1, 2))
	>>> len({Q(1, 2), Q(2, 4), Q(3, 6)})      # ... so a set collapses them
	1
	>>> Q(1, 2) + Q(1, 3), Q(1, 2) * Q(2, 3), Q(1, 2) - Q(1, 4)
	(5/6, 1/3, 1/4)
	>>> Q(1, 2) / Q(3, 4), Q(2, 3).reciprocal
	(2/3, 3/2)
	>>> Q(1, 2) + 1, Q(1, 2) * 3              # ints, naturals and integers coerce on the right
	(3/2, 3/2)
	>>> Q(1, 3) < Q(1, 2), Q(1, -2) < Q()     # ... and a negative denominator still works
	(True, True)
	>>> Q(1, 3).mediant(Q(1, 2))              # strictly between them: ℚ is dense
	2/5
	>>> Q(1, 2).equivalence_class             # the object really is a class of pairs
	'{(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), ...}'
	"""

	# ---------------------------------------------------------------------------------------
	# REPRESENTATIONS — how a rational is built, reduced to lowest terms, and shown.
	# ---------------------------------------------------------------------------------------

	# ............................................................................ given

	def __init__(self,
		m: Z | N | int = 0,
		n: Z | N | int = 1,
	) -> None:
		"""Store the two integers `m` and `n` that represent the rational m/n.

		Given; not part of the exercise.

		Four decisions are settled here, and the rest of the class is written against them:

		* **Two members, and they are integers.** §1.5.3 says a rational is an ordered pair of
		  integers with the second non-zero; `self.m` and `self.n` are that pair. They are `Z`
		  instances, never python ints, which is what puts ℤ's `+`, `*`, `-`, `==` and `<=` at
		  the disposal of every method below.
		* **The denominator defaults to one.** `Q()` is zero and `Q(3)` is the class [(3, 1)],
		  which is how §1.5.3 embeds ℤ in ℚ: every integer m is the rational m/1.
		* **A zero denominator is refused.** The underlying set is ℤ × ℤ*, not ℤ × ℤ, so
		  `Q(1, 0)` names nothing. This is the first constructor in the chapter to reject an
		  argument on mathematical rather than plumbing grounds, and `ZeroDivisionError` is
		  python's name for it. Note the order: the members are coerced *first*, so that
		  `Q(1, Z())` and `Q(1, 0)` fail the same way.
		* **The members are reduced, the fraction is not.** Each member is passed through
		  `reduced`, which replaces the integer's representative by a tidier one denoting the
		  same integer — see the module docstring on why that matters. This is *not* reducing
		  the fraction: `Q(2, 4)` still stores 2 and 4, and finding that it is a half is
		  `canonical`'s job. Normalising the fraction here would collapse the quotient and
		  throw away the exercise.

		>>> Q(1, 2), Q(3), Q(), Q(2, 4)
		(1/2, 3, 0, 1/2)
		>>> Q(3, -4), Q(-3, 4)                    # the sign may sit on either member
		(-3/4, -3/4)
		>>> Q(Z(1), Z(2)), Q(N(1), N(2))          # integers and naturals pass through
		(1/2, 1/2)
		>>> sorted(vars(Q(1, 2)))                 # the two members, before __repr__ exists
		['m', 'n']
		>>> Q(1, 0)
		Traceback (most recent call last):
			...
		ZeroDivisionError: a rational number has no zero denominator
		"""
		self.m = reduced(m if isinstance(m, Z) else Z(m))
		self.n = reduced(n if isinstance(n, Z) else Z(n))

		if not self.n:
			raise ZeroDivisionError("a rational number has no zero denominator")

	@property
	def equivalence_class(self) -> str:
		"""The first few members of this rational's class, e.g. `'{(1, 2), (2, 4), ...}'`.

		Given; not part of the exercise.

		`__repr__` shows a rational the way people write rationals, which is what makes this
		module readable — and also what hides the object being studied. This property does the
		opposite: it displays the rational as what §1.5.3 says it is, a class of pairs, by
		listing the first five multiples of the canonical one.

		The class is larger than what is shown in two ways. It is infinite upward, hence the
		ellipsis; and every pair here has a negated twin — (-1, -2) is a half as surely as
		(1, 2) is — which is why the canonical form has to pin the sign of the denominator as
		well as the common divisor.

		The counterpart of `naturals.py`'s `set` and `integers.py`'s `equivalence_class`, and
		pointing at the same thing: the notation on the page is a summary, and underneath it
		there is a construction.

		>>> Q(2, 4).equivalence_class
		'{(1, 2), (2, 4), (3, 6), (4, 8), (5, 10), ...}'
		>>> Q(3).equivalence_class
		'{(3, 1), (6, 2), (9, 3), (12, 4), (15, 5), ...}'
		>>> Q(-1, 2).equivalence_class
		'{(-1, 2), (-2, 4), (-3, 6), (-4, 8), (-5, 10), ...}'
		>>> Q(1, 2).equivalence_class == Q(5, 10).equivalence_class    # one rational, one class
		True
		"""
		m, n = self.canonical
		members = ", ".join(f"({m * k}, {n * k})" for k in range(1, 6))

		return "{" + members + ", ...}"

	# ..................................................................... to implement

	@property
	def canonical(self) -> tuple[Z, Z]:
		"""Return this rational in lowest terms: gcd(m, n) = 1 with n > 0.

		§1.5.3 asserts that every class contains exactly one such pair, and calls it the
		representation in lowest terms. It is the handle on the class as a whole — `__hash__`,
		`__repr__` and `equivalence_class` all need something that comes out the same for every
		representative, and this is the only thing on offer.

		Returns a plain pair of integers rather than a `Q`, for the same reason `integers.py`
		returned a plain pair: a `Q` would compare equal to every other representative, which
		makes it useless for the jobs above.

		Hints:

		* Two things have to be true of the answer, and they are independent. One is about the
		  common divisor; the other is about the sign of the denominator. Do them in whichever
		  order you like, but do not forget the second — §1.5.3 requires n > 0, and without it
		  the class has two candidates rather than one.
		* The sign is the easier half. If the denominator is negative, negate *both* members.
		  Check against the defining relation that this stays inside the class: is (m, n) ∼
		  (−m, −n)? Substitute into m·n' = m'·n and see.
		* The divisor is `gcd` above, and dividing it out is `divide` above — which is exact
		  precisely because what you are dividing by is a common divisor. That is the whole
		  reason `divide` may refuse an inexact division: here it never has to.
		* Zero needs no special case, but check it anyway. What is gcd(0, n)? And does the
		  answer it gives you come out as (0, 1)?
		* Sanity checks: the result must still be equal, as a rational, to what you started
		  with; applying it twice must change nothing; and `Q(2, 4)` and `Q(3, 6)` must produce
		  the *same* pair, since that is the property every other member leans on.

		>>> Q(6, 8).canonical, Q(2, 4).canonical
		((3, 4), (1, 2))
		>>> Q(-9, 12).canonical, Q(9, -12).canonical      # the sign moves to the numerator
		((-3, 4), (-3, 4))
		>>> Q(0, 5).canonical, Q(7, 1).canonical
		((0, 1), (7, 1))
		>>> Q(2, 4).canonical == Q(3, 6).canonical        # one class, one canonical form
		True
		"""
		...

	def __repr__(self) -> str:
		"""Return the familiar fraction, e.g. `'3/4'`, or just `'3'` when the rational is whole.

		Without this there is nothing to look at: `object`'s repr prints a class name and an
		address. Write it as soon as `canonical` works — it is the debugger for everything
		after.

		Hints:

		* The pair to print is the canonical one, not the stored one: `Q(2, 4)` should read as
		  a half, not as two-quarters. Everything else follows from `canonical`'s answer.
		* Both members already carry ℤ's own signed decimal repr from `integers.py`, so there
		  is nothing to compute — only to arrange, with a slash between them.
		* One case deserves its own branch, and §1.5.3 names it: every integer m is the
		  rational m/1. Printing `3/1` where `3` would do is noise, and the canonical form
		  hands you the test for it directly.
		* Zero, and negatives, should fall out of that without a second branch. Check `Q()`
		  prints `'0'` and `Q(3, -4)` prints `'-3/4'` rather than `'3/-4'` — the second is
		  `canonical`'s doing, not this method's.

		>>> repr(Q(1, 2)), repr(Q(2, 4)), repr(Q(3, -4))
		('1/2', '1/2', '-3/4')
		>>> Q(6, 3), Q(0, 7), Q()                 # whole numbers lose the denominator
		(2, 0, 0)
		>>> print(Q(5, 10))
		1/2
		"""
		...

	def __hash__(self) -> int:
		"""Hash the rational, consistently with `__eq__`.

		Equal objects must hash equally, or a `set` or `dict` holding them quietly misbehaves.
		`Q(1, 2)` and `Q(2, 4)` are equal, so whatever this method hashes has to come out the
		same for both — which rules out hashing `self.m` and `self.n` as they are.

		Hints:

		* The same answer as in `integers.py`, for the same reason: hash the one representative
		  every member of the class agrees on. `canonical` hands it to you as a plain pair.
		* Note that hashing a `Q` instead would call this method again.
		* This method exists at all because `__eq__` set `__hash__` to `None` — python refusing
		  to let a changed equality keep an inherited hash, and right to.

		>>> hash(Q(1, 2)) == hash(Q(2, 4)) == hash(Q(3, 6))
		True
		>>> len({Q(1, 2), Q(2, 4), Q(3, 6), Q(4, 8)})     # one rational, four representatives
		1
		>>> {Q(1, 2): "a half"}[Q(3, 6)]                  # usable as a dict key
		'a half'
		"""
		...

	# ---------------------------------------------------------------------------------------
	# OPERATIONS — the arithmetic of §1.5.3, plus the inverse that makes ℚ a field.
	# ---------------------------------------------------------------------------------------

	# ............................................................................ given

	def __neg__(self) -> Self:
		"""Return the additive inverse, −m/n.

		Given; not part of the exercise.

		In `integers.py` this was the star — the reason ℤ exists — and it swapped the two
		members. Here it is a one-liner that negates the numerator and changes nothing else,
		and it carries no new idea: ℤ already supplied additive inverses, and ℚ simply inherits
		them. The member that matters at this level is the *multiplicative* inverse, which is
		`reciprocal`, and that one is assigned.

		>>> -Q(1, 2), -Q(-1, 2), -Q()
		(-1/2, 1/2, 0)
		>>> -(-Q(2, 3)) == Q(2, 3)
		True
		"""
		return type(self)(-self.m, self.n)

	def __sub__(self, other: Self | Z | N | int) -> Self:
		"""Return `self - other`, as the addition of the additive inverse.

		Given; not part of the exercise. §1.5.3 lists no subtraction rule, because by this
		point none is needed: ℚ inherits ℤ's additive inverses, and `a - b = a + (-b)` is the
		same one-liner that closed out `integers.py`.

		>>> Q(1, 2) - Q(1, 3), Q(1, 2) - Q(1, 2), Q(1, 2) - Q(3, 4)
		(1/6, 0, -1/4)
		>>> Q(1, 2) - 1, Q(1, 2) - Z(1)
		(-1/2, -1/2)
		"""
		if not isinstance(other, (Q, Z, N, int)):
			return NotImplemented

		return self + -(other if isinstance(other, Q) else type(self)(other))

	def __radd__(self, other: Z | N | int) -> Self:
		"""Return `other + self`, so that an int, natural or integer works on the left of `+`.

		Given; not part of the exercise.

		As in the two exercises before it, the one line pays for its convenience by computing
		`other + self` as `self + other`, which is legitimate only because addition commutes —
		a theorem in this development rather than one of §1.5.3's definitions.

		>>> 1 + Q(1, 2), Z(1) + Q(1, 2), N(1) + Q(1, 2)
		(3/2, 3/2, 3/2)
		"""
		return self + other

	def __rsub__(self, other: Z | N | int) -> Self:
		"""Return `other - self`, so that an int, natural or integer works on the left of `-`.

		Given; not part of the exercise. As in `integers.py`, `other - self` is computed as
		`(-self) + other`, which reorders an addition rather than claiming that subtraction
		commutes.

		>>> 1 - Q(1, 2), 2 - Q(1, 2), Z(1) - Q(3, 4)
		(1/2, 3/2, 1/4)
		"""
		return -self + other

	def __rmul__(self, other: Z | N | int) -> Self:
		"""Return `other * self`, so that an int, natural or integer works on the left of `*`.

		Given; not part of the exercise. The mirror of `__radd__`, assuming that multiplication
		commutes.

		>>> 3 * Q(1, 2), Z(2) * Q(1, 3)
		(3/2, 2/3)
		"""
		return self * other

	def __rtruediv__(self, other: Z | N | int) -> Self:
		"""Return `other / self`, so that an int, natural or integer works on the left of `/`.

		Given; not part of the exercise.

		Worth reading beside `__rsub__`, because division commutes no more than subtraction
		does and the line is correspondingly different: `other / self` is computed as
		`self.reciprocal * other`, which reorders a *multiplication*. So the debt is again
		commutativity of `·`, and not a false claim about `/`.

		>>> 1 / Q(1, 2), 3 / Q(3, 4), Z(2) / Q(1, 2)
		(2, 4, 4)
		"""
		return self.reciprocal * other

	# ..................................................................... to implement

	def __mul__(self, other: Self | Z | N | int) -> Self:
		"""Return `self * other`, per §1.5.3:

			[(m, n)] · [(m', n')] = [(m·m', n·n')]

		Memberwise multiplication, using ℤ's `*` on each — the easiest definition in the file,
		and the one to write first, because `reciprocal` and `__truediv__` both lean on it.

		`other` may be a `Q`, a `Z`, an `N`, or a python int, the last three read through the
		constructor on the way in. Any other type returns `NotImplemented`, which lets python
		raise its ordinary `TypeError`. Only the right operand is this method's business —
		`3 * Q(1, 2)` is routed to the given `__rmul__`.

		Hints:

		* Screen the type before touching either operand, exactly as in `integers.py`, and
		  coerce afterwards. A `Q` must not be rebuilt; anything else has to be.
		* The result cannot have a zero denominator: neither operand's does, and ℤ has no zero
		  divisors, so n·n' is never zero. That is a small thing worth noticing — the
		  restricted set ℤ* is closed under multiplication, which is exactly why this
		  definition is allowed to be this simple.
		* Do not reduce the result. `Q(2, 3) * Q(3, 4)` is (6, 12), and that it reads as a half
		  is `__repr__`'s business, by way of `canonical`.

		>>> Q(1, 2) * Q(2, 3), Q(2, 3) * Q(3, 4)
		(1/3, 1/2)
		>>> Q(1, 2) * Q(), Q(-1, 2) * Q(2, 3)
		(0, -1/3)
		>>> Q(1, 2) * 3, Q(1, 2) * Z(3), Q(1, 2) * N(3)
		(3/2, 3/2, 3/2)
		>>> Q(1, 2) * None
		Traceback (most recent call last):
			...
		TypeError: unsupported operand type(s) for *: 'Q' and 'NoneType'
		"""
		...

	def __add__(self, other: Self | Z | N | int) -> Self:
		"""Return `self + other`, per §1.5.3:

			[(m, n)] + [(m', n')] = [(m·n' + m'·n, n·n')]

		Not memberwise, unlike every addition so far. Putting two fractions over a common
		denominator is what the rule says, and the common denominator it picks is the product
		— not the least common multiple, which would need a gcd and would make the definition
		depend on an algorithm rather than on arithmetic.

		Same operand handling as `__mul__`.

		Hints:

		* Transcribe the rule. Three products and one sum, all of them ℤ's, all already
		  working.
		* It is worth seeing why the rule is the right one before writing it. Over a common
		  denominator n·n', the first fraction is m·n' parts and the second is m'·n parts; add
		  the parts. That is the ordinary schoolroom method, and this is its definition.
		* The numerator can come out zero — `Q(1, 2) + Q(-1, 2)` is (0, 4) — and that is fine.
		  It is the *denominator* that may never be zero, and it cannot be.
		* Watch the size of what you produce. Denominators multiply, so chains of additions
		  grow fast, and the module docstring explains why that is not merely untidy here.

		>>> Q(1, 2) + Q(1, 3), Q(1, 2) + Q(1, 4)
		(5/6, 3/4)
		>>> Q(1, 2) + Q(-1, 2), Q(2, 3) + Q(1, 3)
		(0, 1)
		>>> Q(1, 2) + 1, Q(1, 2) + Z(1), Q(1, 2) + N(1)
		(3/2, 3/2, 3/2)
		>>> Q(1, 2) + "x"
		Traceback (most recent call last):
			...
		TypeError: unsupported operand type(s) for +: 'Q' and 'str'
		"""
		...

	@property
	def reciprocal(self) -> Self:
		"""Return the multiplicative inverse, n/m. Not defined at zero.

		This is what ℚ is for. ℤ gave every number an additive inverse and stopped; ℚ gives
		every *non-zero* number a multiplicative one, and that is the difference between a ring
		and a field. §1.5.3 does not spell the rule out, because it does not have to — read the
		equivalence relation and ask which pair, multiplied by (m, n), lands in the class of
		(1, 1).

		Raises ZeroDivisionError at zero, which is the one place the restriction on ℤ* bites
		from the inside: the reciprocal of (0, n) would be (n, 0), and that is not a pair this
		construction admits.

		Hints:

		* The answer swaps the two members, exactly as `integers.py`'s `__neg__` did. Both
		  invert an operation by exchanging the roles of the pair, which is the rhyme the
		  module docstring mentions — negation swaps a difference, reciprocal swaps a quotient.
		* Guard zero *before* swapping, not after. Building the swapped pair first would hand
		  a zero denominator to the constructor, which raises the right error for the wrong
		  reason and with a message about the constructor rather than about reciprocals.
		* A negative rational has a negative reciprocal, and you get that for free — check
		  `Q(-2, 3).reciprocal` and satisfy yourself that no sign handling was needed.
		* The property to check is the defining one: a number times its reciprocal is one.

		>>> Q(2, 3).reciprocal, Q(1, 2).reciprocal
		(3/2, 2)
		>>> Q(-2, 3).reciprocal
		-3/2
		>>> Q(2, 3) * Q(2, 3).reciprocal == Q(1)
		True
		>>> Q().reciprocal
		Traceback (most recent call last):
			...
		ZeroDivisionError: zero has no reciprocal
		"""
		...

	def __truediv__(self, other: Self | Z | N | int) -> Self:
		"""Return `self / other`, as multiplication by the reciprocal.

		The operation ℤ could not have: there, `a / b` is an integer only by accident, which is
		exactly the gap this whole construction was built to close. Note which of `divide` and
		this one is the real division — `divide` above is a crutch for `canonical`, restricted
		to the exact case, and this is the operation ℚ exists to provide.

		Same operand handling as `__mul__`. Raises ZeroDivisionError on division by zero.

		Hints:

		* One line, in terms of two members this class already has. `__sub__` stands to
		  `__neg__` exactly as this stands to `reciprocal`, and it is given just above as a
		  model.
		* Division by zero needs no guard of its own. Work out which member raises it, and
		  satisfy yourself that the message it produces is the right one to show a caller who
		  wrote `x / 0`.
		* Coerce first all the same, or an int on the right will not have a reciprocal to take.

		>>> Q(1, 2) / Q(3, 4), Q(2, 3) / Q(2, 3)
		(2/3, 1)
		>>> Q(1, 2) / 2, Q(1, 2) / Z(2), Q(3, 4) / Q(1, 2)
		(1/4, 1/4, 3/2)
		>>> Q(1, 2) / Q()
		Traceback (most recent call last):
			...
		ZeroDivisionError: zero has no reciprocal
		"""
		...

	# ---------------------------------------------------------------------------------------
	# RELATIONS — when two rationals are the same, when one is below another, and when one
	#             is zero.
	# ---------------------------------------------------------------------------------------

	# ............................................................................ given

	def mediant(self, other: Self) -> Self:
		"""Return a rational strictly between `self` and `other`. Witnesses that ℚ is dense.

		Given; not part of the exercise.

		§1.5.3's Properties close by observing that between any two different rationals there
		sits another, and therefore infinitely many: for m/n < m'/n' with positive denominators,

			m/n  <  (m + m')/(n + n')  <  m'/n'

		The middle term is called the *mediant*, and it is the one place in this file where
		adding numerators to numerators and denominators to denominators — the classic wrong
		way to add fractions — is the right thing to do. It is not addition; it is a witness.

		One wrinkle makes it worth reading rather than skipping: the mediant is **not well
		defined on classes**. Taken on representatives, (1,3) and (1,2) give 2/5, while (2,6)
		and (1,2) give 3/8 — both strictly between a third and a half, but different numbers.
		So this method reduces both sides to their canonical form first, which is what makes
		it a function of the two rationals rather than of the two pairs. Every other member of
		this class is well defined on classes without having to arrange it; this one is the
		exception that shows what the others are quietly getting right.

		>>> Q(1, 3).mediant(Q(1, 2))
		2/5
		>>> Q(1, 3) < Q(1, 3).mediant(Q(1, 2)) < Q(1, 2)
		True
		>>> Q().mediant(Q(1)), Q(2, 6).mediant(Q(1, 2))    # the second reduces first
		(1/2, 2/5)
		"""
		m, n = self.canonical
		p, q = other.canonical

		return type(self)(m + p, n + q)

	def __lt__(self, other: Self) -> bool:
		"""Return whether `self` < `other`, as `≤` and not `=`.

		Given; not part of the exercise. Derivable boilerplate around `__le__` and `__eq__`,
		both of which carry the actual definitions.

		>>> Q(1, 3) < Q(1, 2), Q(1, 2) < Q(1, 2), Q(1, 2) < Q(1, 3)
		(True, False, False)
		>>> Q(1, 2) < Q(2, 4)                     # equal, so not below
		False
		"""
		below = self.__le__(other)

		return below if below is NotImplemented else below and self != other

	def __ge__(self, other: Self) -> bool:
		"""Return whether `self` ≥ `other`, by asking `other` whether it is ≤ `self`.

		Given; not part of the exercise.

		>>> Q(1, 2) >= Q(1, 3), Q(1, 2) >= Q(1, 2), Q(1, 3) >= Q(1, 2)
		(True, True, False)
		"""
		if not isinstance(other, Q):
			return NotImplemented

		return other.__le__(self)

	def __gt__(self, other: Self) -> bool:
		"""Return whether `self` > `other`, by asking `other` whether it is < `self`.

		Given; not part of the exercise.

		>>> Q(1, 2) > Q(1, 3), Q(1, 2) > Q(1, 2), Q(1, 3) > Q(1, 2)
		(True, False, False)
		"""
		if not isinstance(other, Q):
			return NotImplemented

		return other.__lt__(self)

	# ..................................................................... to implement

	def __eq__(self, other: object) -> bool:
		"""Return whether `self` and `other` are the same rational, per §1.5.3:

			[(m, n)] = [(m', n')]  <=>  m·n' = m'·n

		The method that makes the class a quotient rather than a pair of numbers, and the same
		lesson as `integers.py`'s `__eq__` one construction up: `object`'s inherited `==`
		compares identity, so two separately built representatives of a half would be different
		rationals.

		Returns `NotImplemented` for anything that is not a `Q`, which lets python fall back to
		identity and answer `False` rather than raise — the same deliberate asymmetry with the
		arithmetic operators, which do coerce.

		Hints:

		* Transcribe the relation. Two products and ℤ's `==`, and nothing else — in particular
		  no division, which is the reason §1.5.3 states the rule this way.
		* Unlike the ordering below, this one needs no attention to signs. Convince yourself
		  why: what happens to `m·n' = m'·n` when both members of one pair are negated?
		* There is a second, equally correct route: compare `canonical` forms. It agrees, but
		  it is slower, and transcribing the relation is the definition.
		* You do **not** need `__ne__`. `object` derives `!=` by negating this method.

		>>> Q(1, 2) == Q(2, 4), Q(1, 2) == Q(1, 3)
		(True, False)
		>>> Q(1, 2) == Q(-1, -2), Q(1, 2) == Q(-1, 2)     # the sign belongs to the class
		(True, False)
		>>> Q(1, 2) != Q(2, 4)                            # free, from object
		False
		>>> Q(1, 2) == 0.5, Q(1, 2) == "x"                # not a Q: NotImplemented, hence False
		(False, False)
		"""
		...

	def __bool__(self) -> bool:
		"""Return whether this rational is non-zero.

		`object` calls every instance true, zero included. That is wrong on its own, and
		quietly wrong in any code that writes `if q:` — including `reciprocal`, which has to
		ask exactly this question.

		Hint: zero is the class [(0, 1)], and §1.5.3's equality says which other pairs are in
		it. Read that rule with m' = 0 and see what it collapses to. The denominator cannot be
		zero, so it has no say in the matter, and the answer is a one-member test.

		>>> bool(Q(1, 2)), bool(Q(-1, 2))
		(True, True)
		>>> bool(Q()), bool(Q(0, 7)), bool(Q(1, 2) - Q(1, 2))
		(False, False, False)
		"""
		...

	def __le__(self, other: Self) -> bool:
		"""Return whether `self` ≤ `other`, per §1.5.3:

			[(m, n)] ≤ [(m', n')]  <=>  (m·n' ≤ m'·n ∧ n·n' > 0) ∨ (m·n' ≥ m'·n ∧ n·n' < 0)

		The trap of this exercise, and the one place where ℚ's rule is genuinely harder than
		ℤ's rather than merely different. Cross-multiplying compares the two fractions only
		when the denominators agree in sign; when they do not, multiplying through by a
		negative reverses the inequality, and the rule's second branch is that reversal.

		Takes a `Q` only and returns `NotImplemented` otherwise, matching `__eq__` rather than
		the arithmetic operators. `__lt__`, `__gt__` and `__ge__` are derived from this one
		and given.

		Hints:

		* Compute the two cross products once, then choose which way to compare them. Written
		  that way the whole method is three lines and reads like the rule.
		* The condition that selects the branch is the sign of n·n'. Neither denominator can be
		  zero, so that product cannot be either, and the two branches are exhaustive — there
		  is no third case to worry about and no need for `elif`.
		* `>` and `<` on ℤ come from `integers.py` and already work; so does `≤`. Nothing here
		  needs ℕ directly.
		* The case to test first is the one a single-branch version gets wrong: `Q(1, -2)` is
		  −1/2 and must come out below zero. If your version says otherwise, you have written
		  the first branch only — which is the mistake this rule exists to prevent.
		* It would be tempting to sidestep all of this by reducing both sides to canonical form
		  first, where denominators are positive by construction. That works, and it is slower
		  for the obvious reason, but the deeper objection is that it answers a different
		  question: the rule in §1.5.3 is stated about arbitrary representatives, and a
		  definition that only holds for canonical ones is not the definition.

		>>> Q(1, 3) <= Q(1, 2), Q(1, 2) <= Q(1, 3)
		(True, False)
		>>> Q(1, -2) <= Q(), Q() <= Q(1, -2)              # a negative denominator flips it
		(True, False)
		>>> Q(1, 2) <= Q(2, 4), Q(2, 4) <= Q(1, 2)        # ≤ is reflexive, up to ∼
		(True, True)
		>>> sorted([Q(1, 2), Q(-1, 3), Q(2, 3), Q()])
		[-1/3, 0, 1/2, 2/3]
		"""
		...
