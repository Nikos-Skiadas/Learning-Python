"""Set-constructive definition of the natural numbers.

Exercise 8 of the notes (§1.4.1, "ℕ: the natural numbers"): represent ℕ with python's
built-in set type, and let arithmetic fall out of the Peano axioms instead of out of
python's own int arithmetic.

The construction being modelled is von Neumann's:

	0    = {}
	σ(n) = n ∪ {n}

so that every natural number *is* the set of all naturals below it, and nothing else is
smuggled in:

	0 = {}
	1 = {0}       = {{}}
	2 = {0, 1}    = {{}, {{}}}
	3 = {0, 1, 2} = {{}, {{}}, {{}, {{}}}}

Two consequences of that table are worth holding in mind before starting, because this
exercise leans on both:

* n has exactly n elements. This is where cardinality comes from (§1.3.2).
* The elements of n are nested: 0 ⊂ 1 ⊂ ... ⊂ (n-1). A natural number is not merely a
  set of sets, it is a chain of them.

Why `frozenset` and not `set`: σ(n) = n ∪ {n} requires n to be an *element* of another
number, and python's `set` is unhashable, so `{set()}` raises TypeError. `frozenset` is
hashable, so `{frozenset()}` is fine and the construction can actually be built.

Subclassing `frozenset` also means three things are already correct and should not be
reimplemented:

* `==` is extensionality (§1.1.1) — two numbers are equal iff they have the same
  elements. That is Peano axiom 4, enforced by the runtime rather than by you.
* `<=` and `<` are ⊆ and ⊂, and by the order definition in §1.4.1 (n ≥ m iff n ⊇ m)
  that *is* the order on ℕ. Comparison works before a single line is written.
* `__hash__` exists, which is exactly what keeps σ legal.

Two gotchas:

* `frozenset`'s operators return the base type, not the subclass, so `type(N() | {N()})`
  is `frozenset`, and a plain `frozenset` has no `.next`. Wrap anything you assemble out
  of set operations back into `N(...)`.
* The inherited `==` and `<=` are correct but *expensive* between two numbers that were
  built independently of one another: every membership test recurses into members that
  are structurally equal without being identical, and the cost roughly doubles per
  increment (measured: 0.02s at 20, 0.33s at 24, hopeless by 50). The `is`-identical case
  short-circuits, and *unequal* numbers are answered instantly from hash and length, so
  it is specifically the "equal, but constructed twice" case that bites. To check
  arithmetic over a range of values, compare `len` or `repr` rather than the numbers
  themselves. This is the standing cost of the construction, not a defect in it.

Working through this: five members carry the exercise, and each one states what it must
return, which axioms it discharges, the errors it owes the caller, and hints — plus
doctests that serve as its specification. `__new__` and `set` are provided rather than
assigned, and say so in their own docstrings: the first because plumbing a python int into
this construction is a chore rather than a lesson, the second because it is a viewer, not
part of the mathematics. Implement in the order the class docstring suggests, and treat
`python3 -m doctest numbers.py` as the progress bar: silence means done.

References:

* https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset
* https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
* https://docs.python.org/3/reference/datamodel.html#object.__new__
"""


from __future__ import annotations


from collections.abc import Iterable
from typing import Self


class N(frozenset):
	"""A natural number, represented as the set of all natural numbers below it.

	An instance is built either from a python int — `N(3)` is {0, 1, 2} — or from the
	ground up, `N()` being zero and `.next` the successor. Both arithmetic operators accept
	an N *or* a python int on the right, and both are defined by recursion on that second
	operand, exactly as in §1.4.1:

		a + 0 = a                a · 0 = 0
		a + σ(b) = σ(a + b)      a · σ(b) = a + a · b

	Five members to implement, in the order they are easiest to get right: `__repr__`
	first, so that you can see anything at all, then `next`, `prev`, `__add__`, and finally
	`__mul__`, which cannot work before `__add__` does. `__new__` and `set` are provided.

	>>> N()
	0
	>>> N().next
	1
	>>> N(3)
	3
	>>> len(N(3))                    # n has exactly n elements
	3
	>>> N(3).set                     # the same number, unrolled
	'{{}, {{}}, {{}, {{}}}}'
	>>> N(2) + N(3)
	5
	>>> N(2) * N(3)
	6
	>>> N(2) + 3, N(2) * 3           # ints are coerced on the right
	(5, 6)
	>>> N(2) < N(3)                  # inherited from frozenset; nothing to implement
	True
	"""

	# https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset

	def __new__(cls, n: Iterable[Self] | int = ()) -> Self:
		"""Construct a natural number, from either a python int or its own elements.

		Provided; not part of the exercise.

		Two paths, chosen by the type of `n`:

		* an **int** builds the natural number n from the ground up, by starting at zero
		  and applying the successor n times, so `N(3)` is {0, 1, 2}. This is the single
		  bridge between python's ints and this construction, and it exists so that tests
		  and experiments need not spell out `N().next.next.next`. It is also what lets the
		  operators accept `N(2) + 3`: they hand the right operand straight to here.
		* an **iterable** is passed through to `frozenset`, its elements taken to be
		  already-correct naturals. Every internal rewrap travels this path, because `next`
		  and `prev` both assemble their result out of set operations and those return
		  plain `frozenset`s that need the class put back on them. The path is deliberately
		  literal: the elements are right already, they only need reclassifying.

		`N()` is zero either way, since the default `n` is empty and `N(0)` applies the
		successor zero times.

		Three details are worth knowing, being the reasons this is written as it is rather
		than more simply:

		* `frozenset` is immutable, so contents are fixed at the moment the object exists.
		  There is no "make it, then fill it in", which is why the int path counts
		  successors: it must have a *finished* number in hand before it returns.
		  `super().__new__(cls)` with nothing else is zero, the seed it counts up from.
		* Dispatch is on the *type* of `n`, never on its truthiness, since `N(0)` and `N()`
		  must both be zero while the int 0 and an empty iterable need different routes to
		  get there. `bool` subclasses `int`, so `N(True)` is 1 — harmless, and arguably
		  right.
		* A negative int denotes no natural number at all, so it raises rather than
		  clamping to zero: `N(-3)` is a caller's bug.

		Accepting an iterable does mean this constructor will take a set that is *not* a
		natural number: `N({N(4)})` builds a one-element set that no amount of counting
		from zero would ever produce. Screening for that costs as much as building the
		number, so it is left to the caller — and nothing inside this class ever produces
		one. Closing the door instead, by taking ints only, was tried and turned out to
		cost more than it saved: `next` and `prev` then need a second, private constructor
		for their rewraps, the operators need a third for their coercion, and `frozenset`'s
		own pickling protocol breaks, since it rebuilds by calling `cls(elements)`. One
		permissive constructor keeps `copy` and `pickle` working for free.

		Cost: counting to n applies the successor n times and each application copies a set
		of up to n elements, so `N(n)` is quadratic in n. Fine for numbers this file can
		legibly print; do not reach for `N(10_000)`.

		>>> N(), N(0), N(3)
		(0, 0, 3)
		>>> N(frozenset({N(), N(1)}))    # the iterable path: elements of 2, rewrapped
		2
		>>> N(-1)
		Traceback (most recent call last):
		    ...
		ValueError: no natural number corresponds to a negative integer
		"""
		if isinstance(n, int):
			if n < 0:
				raise ValueError("no natural number corresponds to a negative integer")

			result = super().__new__(cls)

			for _ in range(n):
				result = result.next

			return result

		return super().__new__(cls, n)

	def __repr__(self) -> str:
		"""Return the familiar decimal spelling of this number, e.g. `'3'` for {0, 1, 2}.

		Without this, printing 3 shows the raw nested-braces form from the module
		docstring, which stops being readable at about 4. Implement it first — it is the
		debugger for everything else here. When the nesting is what you actually want to
		look at, `set` renders it deliberately.

		Hint: re-read the first of the two consequences in the module docstring. How many
		elements does n have? A builtin answers exactly that question, and its answer
		*is* the number. No recursion, no inspecting the members.

		>>> repr(N(3)), repr(N())
		('3', '0')
		>>> print(N(7))
		7
		"""
		...

	def __add__(self, other: Self | int) -> Self:
		"""Return `self + other`, per the addition axioms of §1.4.1.

		1. a + 0 = a
		2. a + σ(b) = σ(a + b)

		Axiom 1 is the base case, axiom 2 the recursive step, so the recursion runs on
		`other` and never on `self`: each step peels one successor off `other` and puts
		it back on the outside of the result.

		`other` may be an N or a python int, the latter read through the constructor on its
		way into the recursion. Any other type returns `NotImplemented`, which is what lets
		python raise its ordinary TypeError instead of something baffling from inside the
		recursion. That check has to come *before* the base case, not after. Coercion is a
		convenience at the boundary only: past it, nothing knows that ints exist.

		Hints:

		* Axiom 2 is stated in the direction that *builds* a sum out of a smaller one.
		  Code has to read it backwards: given a non-zero `other`, what is the b for
		  which `other == σ(b)`? That question is the entire reason `prev` exists below.
		* The base case needs no comparison at all. Zero is {}, and the empty set is
		  falsy, so `if not other` already reads "if other is zero".
		* Reject the wrong *types* before reaching the base case. `if not other` is true
		  of every falsy object, so a base case that runs first will answer `N(2) + None`
		  with 2 and `N(3) * []` with 0 — wrong, and silently so, which is the worst way
		  to be wrong.
		* Coerce once, at the top, rather than at each use of `other`. The recursive step
		  needs `other.prev` and an int has no predecessor property, so coercion has to
		  happen before it either way; doing it once saves rewrapping the same operand on
		  every step of the recursion.
		* Neither axiom mentions int, len, or python's own `+`. Past the coercion, if int
		  arithmetic shows up in the body, the exercise has been short-circuited.
		* Only the right operand is coerced, so `N(2) + 3` works while `3 + N(2)` raises
		  TypeError. Fixing that needs `__radd__`, which for this operator would have to
		  lean on commutativity of + — a theorem here, not an axiom — so it is left out on
		  purpose. Adding it is a fair extension, provided you notice the debt.
		* Depth: this recurses as deep as `other` is large, so it will hit python's
		  recursion limit somewhere in the low thousands. That is a property of writing
		  down a definition rather than an implementation, not a bug to fix.

		>>> N(2) + N(3), N(2) + N()
		(5, 2)
		>>> N(2) + 3, N(2) + 0
		(5, 2)
		>>> N(2) + "x"
		Traceback (most recent call last):
		    ...
		TypeError: unsupported operand type(s) for +: 'N' and 'str'
		"""
		if not isinstance(other, (N, int)):
			return NotImplemented

		cls = type(self)

		...

	def __mul__(self, other: Self | int) -> Self:
		"""Return `self * other`, per the multiplication axioms of §1.4.1.

		1. a · 0 = 0
		2. a · σ(b) = a + a · b

		The same shape as `__add__` — recursion on `other`, base case at zero, and the
		same `Self | int` screening and coercion of the right operand — but one rung up the
		ladder: where addition's step applies the successor, multiplication's step applies
		addition. So this one cannot be finished until `__add__` works.

		Hints:

		* The base case returns zero rather than `self`. Which N is zero?
		* Lay these two axioms beside the two for `+` and notice how close the recursive
		  step is to a transcription; most of the work was already done next door.
		* The operand handling is the same as in `__add__`, and for the same reason:
		  `other.prev` in the recursive step needs an N. `1 * N(3)` still raises TypeError,
		  as only the right operand is coerced.
		* The operand order in axiom 2 is deliberate, not cosmetic. With the recursive
		  call to the *right* of the `+`, the notes' derivation of the multiplicative
		  identity stays self-contained: a · 1 = a · σ(0) = a + a · 0 = a + 0 = a, which
		  leans only on a + 0 = a — addition's own axiom 1. Writing the step as a · b + a
		  instead ends that same chain at 0 + a, which is no axiom and needs an induction
		  of its own. Both forms define the same function; only one is provable this early.
		* Note also that this order mirrors addition's: there, axiom 2 puts the new σ on
		  the outside and the recursive call within; here, the new copy of `a` goes on the
		  outside and the recursive call within.
		* What that derivation buys is a · 1 = a, a *right* identity. That 1 · a = a holds
		  too is true, but it is a theorem rather than a two-line unfolding — the doctests
		  exercise it, an induction on `a` would prove it.

		>>> N(2) * N(3), N(3) * N(2)
		(6, 6)
		>>> N(3) * 1, N(3) * 0, N(1) * 3
		(3, 0, 3)
		"""
		if not isinstance(other, (N, int)):
			return NotImplemented

		cls = type(self)

		...

	def __radd__(self, other: int) -> Self:
		"""Return `other + self`, per the addition axioms of §1.4.1.

		Coerce the left operand, then lean on `__add__` to do the work. This is a fair
		extension of the exercise, since it is not an axiom but a theorem that + is
		commutative.

		Hint: `__add__` already coerces its right operand, so this one does not need to
		reimplement that logic.

		>>> 3 + N(2)
		5
		"""
		cls = type(self)

		return self.__add__(cls(other))

	def __rmul__(self, other: int) -> Self:
		"""Return `other * self`, per the multiplication axioms of §1.4.1.

		Coerce the left operand, then lean on `__mul__` to do the work. This is a fair
		extension of the exercise, since it is not an axiom but a theorem that * is
		commutative.

		Hint: `__mul__` already coerces its right operand, so this one does not need to
		reimplement that logic.

		>>> 3 * N(2)
		6
		"""
		cls = type(self)

		return self.__mul__(cls(other))

	@property
	def next(self) -> Self:
		"""Return the successor σ(self) = self ∪ {self}, i.e. this number's next number.

		Peano axioms 2 to 4 in a single line, and the only place where the construction
		actually grows. It is a property rather than a method so that counting reads like
		counting: `three = N().next.next.next`.

		Hints:

		* Transcribe σ literally. The singleton containing this number is written
		  `{self}`, which is legal *only* because the class inherits from `frozenset` —
		  see the module docstring on why `set` cannot work here.
		* Combining a `frozenset` subclass with a set yields a plain `frozenset`, which
		  has no `.next`, so `.next.next` will fail on the second hop unless the result is
		  wrapped back into this class before returning. That is the constructor's
		  iterable path, and passing an already-correct family of elements to it is exactly
		  what it is for.

		>>> N().next.next.next
		3
		>>> len(N(4).next)
		5
		"""
		cls = type(self)

		...

	@property
	def prev(self) -> Self:
		"""Return the predecessor: the unique p for which `p.next == self`. Not defined at zero.

		Peano axiom 3 says zero is not the successor of anything, so zero has no
		predecessor; axiom 4 (σ is injective) is what makes every other number's
		predecessor *unique*, hence well defined at all. `__add__` and `__mul__` both
		depend on this, since their recursive steps are phrased in terms of σ(b) and have
		to recover that b.

		Raises ValueError at zero.

		This is the brain-stretcher. Some scaffolding, in the order worth thinking about:

		* Write out 3 = {0, 1, 2} and ask what the answer must be. It is 2 — and 2 is
		  already sitting there as one of the three elements. So `prev` does not build
		  anything new; it recovers something out of `self`.
		* Which element? The greatest one. And "greatest" here is not about `len`, it is
		  the order of §1.4.1, which for these sets is ⊆.
		* Now bring in the second consequence from the module docstring: the elements of
		  n form a chain, 0 ⊂ 1 ⊂ ... ⊂ (n-1). A chain has a greatest member, and that
		  member absorbs every other. Which single set operation from §1.2.2, applied
		  across a whole family of nested sets, hands you that greatest member? Verify
		  your answer by hand on 3 = {{}, {{}}, {{}, {{}}}} before writing any code.
		* `frozenset`'s method for that operation takes the sets to combine as separate
		  arguments, so unpacking with `*` is how a family gets handed over in one go. Its
		  result is a plain `frozenset`, so wrap it as `next` does.
		* Check your answer with `n.prev.next == n` for every n. The operation you land on
		  is harmlessly total at zero and would quietly hand back `N()`, which axiom 3
		  forbids — so guard zero explicitly and raise instead. A correct `__add__` never
		  trips that guard, since it tests for zero before recursing, which is precisely
		  what makes the guard worth having: it turns a *wrong* base case into a loud
		  failure rather than a silent 0.

		>>> N(3).prev, N(1).prev
		(2, 0)
		>>> all(N(k).prev.next == N(k) for k in range(1, 12))
		True
		>>> N().prev
		Traceback (most recent call last):
		    ...
		ValueError: 0 has no predecessor
		"""
		if not self:
			raise ValueError("0 has no predecessor")

		cls = type(self)

		...

	@property
	def set(self) -> str:
		"""The number written out as nested braces, e.g. `'{{}, {{}}}'` for 2.

		Provided; not part of the exercise.

		`__repr__` shows a natural number the way people write numbers, which is what makes
		the rest of this module readable — and also what hides the object being studied.
		This property does the opposite: it unrolls the construction the whole way down to
		the empty set, so that what a number actually *is* becomes visible. It reproduces
		the "unrolled" column of Table 1 in the notes exactly:

			N(0).set == '{}'
			N(1).set == '{{}}'
			N(2).set == '{{}, {{}}}'
			N(3).set == '{{}, {{}}, {{}, {{}}}}'

		Three details of the rendering are deliberate:

		* **It recurses.** Each member is rendered by this same property rather than by
		  `__repr__`. That is the whole difference between Table 1's two right-hand
		  columns: stopping at `__repr__` would print 3 as the "inductive" `{0, 1, 2}`,
		  whereas recursing prints the "unrolled" `{{}, {{}}, {{}, {{}}}}`.
		* **Zero renders as `{}`**, not python's `set()`. That is what the notes write, and
		  it is what makes the nesting legible rather than noisy. It also costs nothing to
		  arrange: a number with no members has nothing to join, so the braces close on an
		  empty string by themselves.
		* **Members are sorted** before joining, so the output is deterministic and
		  ascending. Sets are unordered and python iterates a frozenset in hash order, so
		  without the sort `N(9)` would list its members as {3, 8, 4, 2, 5, 1, 7, 0, 6}.
		  Sorting by cardinality is the same as sorting by ⊆ here, since the members of a
		  natural number form a chain — which is the second consequence noted in the module
		  docstring, put to work.

		Note that the unrolled form roughly doubles in width with every successor: 22
		characters at 3, 3,070 at 10, and 3,145,726 at 20. This is a microscope, not a
		printer. Use it on the small numbers, where seeing the nesting is the point.

		>>> N(0).set, N(1).set, N(2).set
		('{}', '{{}}', '{{}, {{}}}')
		>>> N(3).set
		'{{}, {{}}, {{}, {{}}}}'
		>>> print(N(4).set)
		{{}, {{}}, {{}, {{}}}, {{}, {{}}, {{}, {{}}}}}
		>>> N(3).set == N(2).next.set              # a rendering, not a representation
		True
		"""
		return "{" + ", ".join(member.set for member in sorted(self, key=len)) + "}"
