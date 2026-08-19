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
  is `frozenset`. Wrap anything you build back into `N(...)`.
* The inherited `==` and `<=` are correct but *expensive* between two numbers that were
  built independently of one another: every membership test recurses into members that
  are structurally equal without being identical, and the cost roughly doubles per
  increment (measured: 0.02s at 20, 0.33s at 24, hopeless by 50). The `is`-identical case
  short-circuits, and *unequal* numbers are answered instantly from hash and length, so
  it is specifically the "equal, but constructed twice" case that bites. To check
  arithmetic over a range of values, compare `len` or `repr` rather than the numbers
  themselves. This is the standing cost of the construction, not a defect in it.

Working through this: every member of `N` below states what it must return, which axioms
it discharges, the errors it owes the caller, and hints — plus doctests that serve as its
specification. Implement them in the order the class docstring suggests, and treat
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

	An instance is built either from the ground up — `N()` is zero and `.next` is the
	successor — or by handing a python int to the constructor, so that `N(3)` is
	{0, 1, 2}. Both arithmetic operators accept an N *or* a python int on the right, and
	both are defined by recursion on that second operand, exactly as in §1.4.1:

		a + 0 = a                a · 0 = 0
		a + σ(b) = σ(a + b)      a · σ(b) = a + a · b

	Six members to implement, in the order they are easiest to get right: `__repr__`
	first, so that you can see anything at all, then `next`, `__new__`, `prev`, `__add__`,
	and finally `__mul__`, which cannot work before `__add__` does.

	>>> N()
	0
	>>> N().next
	1
	>>> N(3)
	3
	>>> len(N(3))                    # n has exactly n elements
	3
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

		Two paths, chosen by the type of `n`:

		* an **int** n builds the natural number n from the ground up, by starting at zero
		  and applying the successor n times, so `N(3)` is {0, 1, 2}. This is the single
		  bridge between python's ints and this construction, and it exists so that tests
		  and experiments need not spell out `N().next.next.next`.
		* an **iterable** is handed straight through to `frozenset`, whose elements are
		  taken to be already-correct naturals. Every internal rewrap in this class —
		  `next`, `prev`, and both operators — travels this path, so it must stay literal:
		  the elements are right already, they only need the class put back on them.

		`N()` is zero either way, since the default `n` is empty and `N(0)` applies
		the successor zero times.

		Raises ValueError on a negative int.

		Hints:

		* `frozenset` is immutable, so contents are fixed at the moment the object exists.
		  That rules out "make it, then fill it in": the int path must have a *finished*
		  number in hand before it returns, which is why counting successors works and
		  mutating does not.
		* `super().__new__(cls)` with nothing else is zero — the seed the int path counts
		  up from. Each `.next` calls this constructor again, which is harmless: those
		  calls arrive down the iterable path, never back into the int one.
		* Dispatch on the *type* of `n`, not on its truthiness. `N(0)` and `N()` must
		  both be zero, and the int 0 and an empty iterable are both falsy while needing
		  different routes to get there. Note `bool` subclasses `int`, so `N(True)` is 1 —
		  harmless, and arguably right.
		* A negative int denotes no natural number at all. Raise: `N(-3)` is a caller's
		  bug, not a zero.
		* Cost: building n applies the successor n times and each application copies a set
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
		...

	def __add__(self, other: Self | int) -> Self:
		"""Return `self + other`, per the addition axioms of §1.4.1.

		1. a + 0 = a
		2. a + σ(b) = σ(a + b)

		Axiom 1 is the base case, axiom 2 the recursive step, so the recursion runs on
		`other` and never on `self`: each step peels one successor off `other` and puts
		it back on the outside of the result.

		`other` may be an N or a python int, the latter coerced through the constructor
		before anything else happens. Any other type returns `NotImplemented`, which is
		what lets python raise its ordinary TypeError instead of something baffling from
		inside the recursion. Coercion is a convenience at the boundary only: past it,
		nothing knows that ints exist.

		Hints:

		* Axiom 2 is stated in the direction that *builds* a sum out of a smaller one.
		  Code has to read it backwards: given a non-zero `other`, what is the b for
		  which `other == σ(b)`? That question is the entire reason `prev` exists below.
		* The base case needs no comparison at all. Zero is {}, and the empty set is
		  falsy, so `if not other` already reads "if other is zero".
		* Coerce before doing anything else. The base case would survive a raw int (0 is
		  falsy either way), but the recursive step asks for `other.prev`, and an int has
		  no predecessor property — so coerce at the top and the recursion only ever sees
		  Ns.
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
		...

	def __mul__(self, other: Self | int) -> Self:
		"""Return `self * other`, per the multiplication axioms of §1.4.1.

		1. a · 0 = 0
		2. a · σ(b) = a + a · b

		The same shape as `__add__` — recursion on `other`, base case at zero, and the
		same `Self | int` coercion on the right operand — but one rung up the ladder:
		where addition's step applies the successor, multiplication's step applies
		addition. So this one cannot be finished until `__add__` works.

		Hints:

		* The base case returns zero rather than `self`. Which N is zero?
		* Lay these two axioms beside the two for `+` and notice how close the recursive
		  step is to a transcription; most of the work was already done next door.
		* The coercion is the same as in `__add__`, and for the same reason:
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
		...

	def __repr__(self) -> str:
		"""Return the familiar decimal spelling of this number, e.g. `'3'` for {0, 1, 2}.

		Without this, printing 3 shows the raw nested-braces form from the module
		docstring, which stops being readable at about 4. Implement it first — it is the
		debugger for everything else here.

		Hint: re-read the first of the two consequences in the module docstring. How many
		elements does n have? A builtin answers exactly that question, and its answer
		*is* the number. No recursion, no inspecting the members.

		>>> repr(N(3)), repr(N())
		('3', '0')
		>>> print(N(7))
		7
		"""
		...

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
		  has no `.next`, so `.next.next` will fail on the second hop unless the result
		  is wrapped back into this class before returning.

		>>> N().next.next.next
		3
		>>> len(N(4).next)
		5
		"""
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
		  arguments, so unpacking with `*` is how a family gets handed over in one go.
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
		...
