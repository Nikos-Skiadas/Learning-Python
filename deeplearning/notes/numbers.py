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

One gotcha: `frozenset`'s operators return the base type, not the subclass, so
`type(N() | {N()})` is `frozenset`. Wrap anything you build back into `N(...)`.

References:

* https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset
* https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
"""


from __future__ import annotations


from typing import Self


class N(frozenset):
	"""A natural number, represented as the set of all natural numbers below it.

	An instance is built either from the ground up — `N()` is zero, `.next` is the
	successor — or via the `fromint` convenience constructor. Both arithmetic operators
	are defined by recursion on their *second* operand, exactly as in §1.4.1:

		a + 0 = a                a · 0 = 0
		a + σ(b) = σ(a + b)      a · σ(b) = a · b + a

	Suggested order of attack: `__repr__` first (so you can see what you are doing), then
	`next`, `fromint`, `prev`, `__add__`, and finally `__mul__`.

	>>> N()
	0
	>>> N().next
	1
	>>> N.fromint(2) + N.fromint(3)
	5
	>>> N.fromint(2) * N.fromint(3)
	6
	>>> N.fromint(2) < N.fromint(3)  # inherited from frozenset; nothing to implement
	True
	"""

	# https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset

	def __add__(self, other: Self) -> Self:
		"""Return `self + other`, per the addition axioms of §1.4.1.

		1. a + 0 = a
		2. a + σ(b) = σ(a + b)

		Axiom 1 is the base case, axiom 2 the recursive step, so the recursion runs on
		`other` and never on `self`: each step peels one successor off `other` and puts
		it back on the outside of the result.

		Hints:

		* Axiom 2 is stated in the direction that *builds* a sum out of a smaller one.
		  Code has to read it backwards: given a non-zero `other`, what is the b for
		  which `other == σ(b)`? That question is the entire reason `prev` exists below.
		* The base case needs no comparison at all. Zero is {}, and the empty set is
		  falsy, so `if not other` already reads "if other is zero".
		* Neither axiom mentions int, len, or python's own `+`. If int arithmetic shows
		  up in the body, the exercise has been short-circuited.
		* Depth: this recurses as deep as `other` is large, so it will hit python's
		  recursion limit somewhere in the low thousands. That is a property of writing
		  down a definition rather than an implementation, not a bug to fix.
		"""
		...  # +

	def __mul__(self, other: Self) -> Self:
		"""Return `self * other`, per the multiplication axioms of §1.4.1.

		1. a · 0 = 0
		2. a · σ(b) = a · b + a

		The same shape as `__add__` — recursion on `other`, base case at zero — but one
		rung up the ladder: where addition's step applies the successor, multiplication's
		step applies addition. So this one cannot be finished until `__add__` works.

		Hints:

		* The base case returns zero rather than `self`. Which N is zero?
		* Lay these two axioms beside the two for `+` and notice how close the recursive
		  step is to a transcription; most of the work was already done next door.
		* Worth proving on paper afterwards (notes, multiplication item 3): a · 1 = a
		  follows from these axioms *plus* 0 + a = a. The axioms only ever handed you
		  a + 0 = a, so convincing yourself that 0 + a = a needs an induction of its own
		  is a good use of ten minutes.
		"""
		...  # *

	def __repr__(self) -> str:
		"""Return the familiar decimal spelling of this number, e.g. `'3'` for {0, 1, 2}.

		Without this, printing 3 shows the raw nested-braces form from the module
		docstring, which stops being readable at about 4. Implement it first — it is the
		debugger for everything else here.

		Hint: re-read the first of the two consequences in the module docstring. How many
		elements does n have? A builtin answers exactly that question, and its answer
		*is* the number. No recursion, no inspecting the members.
		"""
		...

	@classmethod
	def fromint(cls, n: int) -> Self:
		"""Build the N corresponding to the python int `n`: the set of the first n naturals.

		A bridge for tests and experiments, and deliberately the only place in the class
		where an int is allowed anywhere near the construction. `fromint(0)` is {} and
		`fromint(3)` is {0, 1, 2}.

		Hints:

		* This is the constructive reading of the notes' table: start at zero and apply
		  the successor n times. Which member below is "apply the successor"?
		* A plain loop is fine, and clearer than recursion, for this one.
		* A negative `n` denotes no natural number at all, so decide deliberately what
		  should happen: raise, or quietly return zero. Either is defensible; picking by
		  accident is not.
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
		* Checks: `n.prev.next == n` for every n. The operation you land on also turns
		  out to be harmlessly total at zero, giving `N().prev == N()` — mathematically
		  wrong by axiom 3, so decide whether to raise instead, and notice that a correct
		  `__add__` never calls `prev` on zero anyway.
		"""
		...
