"""Set-constructive definition of the integers.

The sequel to `naturals.py`, for §1.5.2 ("ℤ: the integers"): represent ℤ on top of the ℕ
built there, and let arithmetic fall out of §1.5.2's definitions instead of out of python's
own int arithmetic.

The construction being modelled is a quotient set in the sense of §1.4:

	ℤ = (ℕ × ℕ)/∼          where (a, b) ∼ (c, d) <=> a + d = b + c

The intuition is that the pair (a, b) stands for a − b, which is why the relation is stated
without a subtraction that ℕ does not have. So one integer is *many* pairs:

	 2 = (2, 0) = (3, 1) = (4, 2) = ...
	 0 = (0, 0) = (1, 1) = (2, 2) = ...
	-2 = (0, 2) = (1, 3) = (2, 4) = ...

Three consequences are worth holding in mind before starting, because this exercise leans on
all three:

* An instance is a *representative* (§1.4.1's word), not a class. `Z(1, 2)` and `Z(4, 5)` are
  different pairs that have to behave as the same integer.
* Equality is therefore **not** structural. This is the one deep difference from `naturals.py`,
  where extensionality made `==` free and correct.
* Every class has exactly one member of the form (n, 0) or (0, n) — §1.5.2's last paragraph.
  That unique member is the handle on the class, and several members below need it.

Why a bare class with two plain members, rather than a subclass of anything: because nothing
in python models a quotient. `naturals.py` could inherit from `frozenset` and get `==`, `<=` and
`__hash__` already *correct*, since extensionality is Peano axiom 4 and ⊆ is the order on von
Neumann naturals. There is no such base here. A `tuple` would have supplied immutability and
unpacking, but its `==` is componentwise, its `<` lexicographic, its `+` concatenation and
its `*` repetition — every one of them wrong for ℤ, and wrong *silently*. Inheriting nothing
costs a few lines and buys the guarantee that every behaviour this class has is one it
claimed on purpose. That choice is the first lesson of the exercise: picking a base class is
picking which behaviours you are asserting.

What `object` still hands you, and it is worth knowing which way each one cuts:

* `==` and `hash` by *identity*. Both wrong for a quotient — two representatives of one
  integer are two objects — and wrong in the way easiest to miss, since `z == z` looks fine.
  Both are assigned below.
* `!=` derived from whatever `__eq__` you write. Free and correct: `object` negates `__eq__`
  for you, which is why there is no `__ne__` below. (A `tuple` base would *not* have done
  this — it implements `!=` separately, so a correct `__eq__` would have left `x == y` and
  `x != y` both true. One more reason the bare class is the calmer starting point.)
* Truthiness: everything is true. Zero is [(a, a)], and must not be.
* No ordering whatsoever — `<` raises TypeError until you write it.
* `copy`, `deepcopy` and `pickle`, working out of the box, because a plain instance keeps its
  state in `__dict__` and the default protocols know how to move that around.

Three gotchas:

* Defining `__eq__` sets `__hash__` to `None` unless you define `__hash__` too — python
  assumes a changed equality invalidates the inherited hash, and here it is right to. An
  integer that cannot go in a set is not much of an integer, so both are assigned.
* Immutability is a *convention* here, not a guarantee. `frozenset` enforced it in
  `naturals.py`; nothing stops `self.a` being reassigned now, and doing that after the object
  has gone into a set or a dict corrupts the container. Treat the two members as write-once.
  (`__slots__` narrows the door without closing it: it forbids *new* attributes, not
  assignment to the declared ones.)
* No python int arithmetic anywhere. The members are naturals, and ℕ already has `+`, `*`,
  `==`, `<=`, `.next` and `.prev` from `naturals.py`; every definition in §1.5.2 is written in
  terms of exactly those. If `len`, `-`, or an int literal turns up inside a method body, the
  construction has been short-circuited. The one exception is `__init__`, which is given, and
  which is the bridge from python's ints *into* the construction — bridges are allowed to see
  both sides.

On negative arguments: `Z(-3)` works, and the way it works is worth a look rather than a
shrug. ℕ holds no negative number, so the constructor cannot simply hand −3 to it. What it
does instead is put the 3 in the *other* member: −3 is (0, 3). That is not a workaround, it
is the construction's whole idea — in ℤ, negativity is positional rather than a sign, and the
constructor is the first place you see it. `N(-1)` raises, as it must; `Z(-1)` does not, and
the difference between those two facts is the difference between ℕ and ℤ.

Working through this: the class is laid out in three groups — representations, operations,
relations — and each group leads with the members that are given, then the ones to implement.
Ten members carry the exercise, and each states what it must return, which definition from
§1.5.2 it discharges, the errors it owes the caller, and hints, plus doctests that serve as
its specification. Eight more are given, and say so in their own docstrings: `__init__`,
because plumbing python ints into a pair of naturals is a chore rather than a lesson;
`equivalence_class`, because it is a viewer rather than part of the mathematics; `__radd__`,
`__rsub__` and `__rmul__`, because they buy their convenience by assuming commutativity,
which is a theorem here; and `__lt__`, `__ge__` and `__gt__`, because they are derivable
boilerplate around the two comparisons that carry the mathematics.

The file's grouping is not the order to solve in — the class docstring gives that. Run

	python3 -m doctest deeplearning/notes/integers.py

from the repository root as the progress bar: silence means done. It has to be the repository
root, and the import above has to be absolute rather than relative, so that the same command
works whether this module is loaded as a file or as part of its package — the scorer loads it
the second way.

References:

* https://docs.python.org/3/reference/datamodel.html#object.__hash__
* https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
* https://docs.python.org/3/reference/datamodel.html#basic-customization
"""


from __future__ import annotations


from typing import Self

from deeplearning.notes.naturals import N


class Z:
	"""An integer, represented by two naturals standing for their difference.

	The members `a` and `b` mean a − b, and two integers with the same difference are the
	same integer: `Z(1, 2)` and `Z(4, 5)` are both −1. An instance holds one representative;
	`canonical` produces the one representative that every class has exactly one of.

	The definitions being transcribed, all from §1.5.2:

		[(a, b)] = [(c, d)]  <=>  a + d = c + b
		[(a, b)] ≤ [(c, d)]  <=>  a + d ≤ c + b

		[(a, b)] + [(c, d)] = [(a + c, b + d)]
		[(a, b)] · [(c, d)] = [(ac + bd, ad + bc)]
		          −[(a, b)] = [(b, a)]
		[(a, b)] − [(c, d)] = [(a + d, b + c)]

	Ten members to implement. The file groups them by what they are; this is the order to
	*write* them in, chosen so that each is solvable by the time you reach it:

		1. `__eq__`     — what makes this a quotient. It will break hashing on the spot.
		2. `canonical`  — the one that takes real thought.
		3. `__hash__`   — repairs what `__eq__` broke, using `canonical`.
		4. `__repr__`   — the debugger for everything after, also using `canonical`.
		5. `__neg__`    — the reason ℤ exists at all.
		6. `__add__`, then `__sub__`, then `__mul__`.
		7. `__bool__`, then `__le__`.

	Until `__repr__` exists there is nothing to look at, so check the earlier members with
	`vars(z)`, which shows the two naturals `__init__` stored.

	>>> Z(3), Z(0, 3), Z(-3), Z()
	(3, -3, -3, 0)
	>>> Z(1, 2) == Z(4, 5)                    # one integer, two representatives
	True
	>>> Z(1, 2).canonical, Z(4, 5).canonical  # ... reducing to the same canonical pair
	((0, 1), (0, 1))
	>>> len({Z(1, 2), Z(4, 5), Z(0, 1)})      # ... so a set collapses them
	1
	>>> Z(2) + Z(3), Z(2) * Z(3), Z(2) - Z(3)
	(5, 6, -1)
	>>> -Z(3), -Z(0, 3)
	(-3, 3)
	>>> Z(2) + 3, Z(2) * 3, Z(2) - 3          # ints and naturals coerce on the right
	(5, 6, -1)
	>>> Z(0, 1) < Z(1, 0), Z(0, 0) < Z(1, 5)  # -1 < 1, and 0 < -4 is false
	(True, False)
	>>> Z(1, 2).equivalence_class             # the object really is a class of pairs
	'{(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), ...}'
	"""

	# ---------------------------------------------------------------------------------------
	# REPRESENTATIONS — how an integer is built, reduced to the form that names its class,
	#                   and shown.
	# ---------------------------------------------------------------------------------------

	# ............................................................................ given

	def __init__(self,
		a: N | int = 0,
		b: N | int = 0,
	) -> None:
		"""Store the two naturals `a` and `b` that represent the integer a − b.

		Given; not part of the exercise.

		Four design decisions are settled here, and the rest of the class is written against
		them, so they are worth reading before anything else:

		* **Two members, and they are naturals.** §1.5.2 says an integer is an ordered pair
		  of naturals; `self.a` and `self.b` are that pair. They are `N` instances, never
		  python ints, which is what puts ℕ's `+`, `*`, `==`, `<=`, `.next` and `.prev` at
		  the disposal of every method below. Storing ints instead would make the definitions
		  in §1.5.2 impossible to transcribe honestly.
		* **Defaults make the common spellings short.** `Z()` is zero and `Z(3)` is the class
		  [(3, 0)], which §1.5.2 identifies with +3.
		* **`__init__`, not `__new__`.** `naturals.py` needed `__new__` because its base class
		  was immutable and the contents had to exist before the object did. There is no base
		  class here, so the ordinary hook is the right one — and immutability stops being a
		  guarantee and becomes a promise the class makes. Do not reassign `self.a` or
		  `self.b` anywhere else; see the module docstring.
		* **A negative int is not rejected, it is repositioned.** ℕ cannot hold −3, so the 3
		  goes into the other member and `Z(-3)` is (0, 3). Generally, a − b with a negative
		  part on either side is the same integer as (a⁺ + b⁻) − (a⁻ + b⁺), which is what the
		  two lines below compute. This is the construction's own trick applied to its own
		  front door: negativity here is positional, not a sign.

		>>> Z(3), Z(1, 2), Z()
		(3, -1, 0)
		>>> Z(-3), Z(1, -2), Z(-1, -2)            # the negative part crosses to the other side
		(-3, 3, 1)
		>>> Z(N(1), N(2))                         # naturals pass straight through
		-1
		>>> Z(3).canonical                        # ... and ints are read as naturals
		(3, 0)
		>>> [type(m).__name__ for m in Z(3).canonical]
		['N', 'N']
		>>> sorted(vars(Z(1, 2)))                 # the two members, before __repr__ exists
		['a', 'b']
		"""
		# ℕ holds no negatives, so each argument is split into the part it can take and the
		# part that has to cross over: a − b = (a⁺ + b⁻) − (a⁻ + b⁺).
		p_a, m_a = (a, 0) if not isinstance(a, int) or a >= 0 else (0, -a)
		p_b, m_b = (b, 0) if not isinstance(b, int) or b >= 0 else (0, -b)

		self.a = N(p_a) + m_b
		self.b = N(m_a) + p_b

	@property
	def equivalence_class(self) -> str:
		"""The first few members of this integer's class, e.g. `'{(0, 1), (1, 2), ...}'`.

		Given; not part of the exercise.

		`__repr__` shows an integer the way people write integers, which is what makes this
		module readable — and also what hides the object being studied. This property does the
		opposite: it displays the integer as what §1.5.2 says it is, a class of pairs, by
		listing the first five of them from the canonical one upward. The class is infinite,
		hence the trailing ellipsis; it has no last member, and no first one either except by
		the convention that the canonical representative comes first.

		The counterpart of `naturals.py`'s `set`, pointing at the same thing: the notation on
		the page is a summary, and underneath it there is a construction.

		>>> Z(1, 2).equivalence_class
		'{(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), ...}'
		>>> Z(2).equivalence_class
		'{(2, 0), (3, 1), (4, 2), (5, 3), (6, 4), ...}'
		>>> Z().equivalence_class
		'{(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), ...}'
		>>> Z(4, 5).equivalence_class == Z(1, 2).equivalence_class    # same integer, same class
		True
		"""
		a, b = self.canonical
		members = ", ".join(f"({a + k}, {b + k})" for k in range(5))

		return "{" + members + ", ...}"

	# ..................................................................... to implement

	@property
	def canonical(self) -> tuple[N, N]:
		"""Return the unique representative of this integer of the form (n, 0) or (0, n).

		§1.5.2 closes by asserting that every class has exactly one such member; this property
		produces it. It is the handle on the class as a whole — `__hash__` and `__repr__` both
		need something that comes out the same for every representative, and this is the only
		thing on offer.

		Returns a plain pair of naturals rather than a `Z`, deliberately: the caller wants the
		two numbers themselves, and a `Z` would compare equal to every other representative,
		which makes it useless for the two jobs above.

		This is the brain-stretcher, the counterpart of `naturals.py`'s `prev`. Scaffolding, in
		the order worth thinking about:

		* Take (4, 6) and ask what the answer must be: (0, 2). Now ask what was done to get
		  there — one was taken off *both* members, twice, until one of them ran out.
		* Why is that allowed? Check it against the defining relation. Is (a, b) ∼ (σa, σb)?
		  Substitute into a + d = b + c and see what cancels. If it holds, then peeling a
		  successor off both members at once never leaves the class, which is exactly the
		  licence this method needs.
		* Which ℕ operation is "take one off"? `naturals.py` wrote it, and it is the only member
		  of ℕ that goes downwards.
		* When do you stop? When one of them has run out — and that same operation raises at
		  zero, so the loop condition has to stop before it does rather than catch it after.
		  Zero in ℕ is the empty set, so asking whether a natural is zero needs no comparison.
		* Both run out together exactly when the integer is zero, and (0, 0) is the canonical
		  form of zero, so that case needs no special handling. Check that your loop gets it
		  right rather than adding a branch for it.
		* Work on local names, never on `self.a` and `self.b` themselves. Reducing in place
		  would leave the object no longer the one you started with — see the module docstring
		  on immutability being a convention here.
		* Sanity check: the result must still be equal, as an integer, to what you started
		  with, and applying it twice must change nothing.

		>>> Z(4, 6).canonical, Z(6, 4).canonical
		((0, 2), (2, 0))
		>>> Z(7, 7).canonical, Z().canonical
		((0, 0), (0, 0))
		>>> all(Z(*Z(a, b).canonical) == Z(a, b) for a in range(5) for b in range(5))
		True
		>>> Z(9, 4).canonical == Z(*Z(9, 4).canonical).canonical      # idempotent
		True
		"""
		...

	def __repr__(self) -> str:
		"""Return the familiar signed decimal spelling, e.g. `'-3'` for the class of (0, 3).

		Without this there is nothing to look at: `object`'s repr prints a class name and an
		address. Write it as soon as `canonical` works — it is the debugger for everything
		after.

		Hints:

		* A representative such as (4, 6) cannot be printed directly; (0, 2) can. So this
		  method starts by asking `canonical` for the representative that *is* printable, and
		  the shape of that answer is what makes the rest easy.
		* Of the two naturals in a canonical pair, at least one is zero. Which one tells you
		  the sign, and the other is the magnitude — already carrying ℕ's own decimal repr
		  from `naturals.py`, so there is nothing to compute here.
		* Zero is the case where *both* are zero, and it must print `'0'`, not `'-0'`. Pick
		  the branch order that gets that for free rather than special-casing it.
		* No arithmetic. If `len` or `-` appears in this body, `canonical` was not used.

		>>> repr(Z(3)), repr(Z(0, 3)), repr(Z())
		('3', '-3', '0')
		>>> Z(4, 6), Z(6, 4)                      # non-canonical representatives print fine
		(-2, 2)
		>>> print(Z(7) * Z(0, 3))
		-21
		"""
		...

	def __hash__(self) -> int:
		"""Hash the integer, consistently with `__eq__`.

		Equal objects must hash equally, or a `set` or `dict` holding them quietly misbehaves.
		`Z(1, 2)` and `Z(4, 5)` are equal, so whatever this method hashes has to come out the
		same for both — which rules out hashing `self.a` and `self.b` as they are, and points
		at the one thing every representative of a class shares.

		Hints:

		* §1.5.2's last paragraph is the whole hint: every class has exactly one member of
		  the form (n, 0) or (0, n). That member is an invariant of the class, so hashing it
		  hashes the integer rather than the representative.
		* `canonical` hands you that member, as a plain pair. Hash it directly and this is one
		  line — and note that hashing a `Z` instead would call this method again.
		* Nothing needs inventing: the members are naturals, and `frozenset` gave them a
		  working hash back in `naturals.py`.
		* This method exists at all because `__eq__` set `__hash__` to `None`. That is python
		  refusing to let a changed equality keep an inherited hash, and it is right to.

		>>> hash(Z(1, 2)) == hash(Z(4, 5)) == hash(Z(0, 1))
		True
		>>> len({Z(2), Z(3, 1), Z(9, 7)})         # one integer, three representatives
		1
		>>> {Z(0, 1): "minus one"}[Z(3, 4)]       # usable as a dict key
		'minus one'
		"""
		...

	# ---------------------------------------------------------------------------------------
	# OPERATIONS — the arithmetic of §1.5.2, each one a transcription and nothing more.
	# ---------------------------------------------------------------------------------------

	# ............................................................................ given

	def __radd__(self, other: N | int) -> Self:
		"""Return `other + self`, so that an int or a natural works on the left of `+`.

		Given; not part of the exercise.

		Python reaches here only after the left operand has declined, so `Z(2) + 3` never
		comes through. As in `naturals.py`, the one line pays for its convenience by computing
		`other + self` as `self + other`, which is legitimate only because addition is
		commutative — a theorem in this development rather than one of §1.5.2's definitions.
		Given rather than assigned, so that the exercise is not invited to prove it by
		assuming it.

		>>> 3 + Z(2), N(3) + Z(2), 0 + Z(0, 3)
		(5, 5, -3)
		"""
		return self + other

	def __rsub__(self, other: N | int) -> Self:
		"""Return `other - self`, so that an int or a natural works on the left of `-`.

		Given; not part of the exercise.

		Worth reading beside `__radd__`, because subtraction does *not* commute and the line
		is correspondingly different: `other - self` is computed as `(-self) + other`, which
		reorders an addition rather than a subtraction. So the debt is the same one —
		commutativity of `+` — and not a second, false claim that `-` commutes.

		>>> 5 - Z(3), 3 - Z(5), 0 - Z(3)
		(2, -2, -3)
		>>> N(5) - Z(3)
		2
		"""
		return -self + other

	def __rmul__(self, other: N | int) -> Self:
		"""Return `other * self`, so that an int or a natural works on the left of `*`.

		Given; not part of the exercise. The mirror of `__radd__`, carrying the same debt one
		rung up: it assumes multiplication commutes.

		>>> 3 * Z(2), N(3) * Z(0, 2)
		(6, -6)
		"""
		return self * other

	# ..................................................................... to implement

	def __neg__(self) -> Self:
		"""Return the additive inverse, per §1.5.2:

			−[(a, b)] = [(b, a)]

		The single reason ℤ exists: ℕ has addition but no inverses, and the pair construction
		supplies them. Every other definition in this class adapts something ℕ already had;
		this one is genuinely new.

		Hint: read the rule literally. Nothing is computed — the two members change places,
		which is what it means for (a, b) to stand for a − b and (b, a) for b − a.

		>>> -Z(3), -Z(0, 3), -Z()
		(-3, 3, 0)
		>>> -(-Z(5)) == Z(5)                      # an involution
		True
		>>> Z(5) + -Z(5) == Z()                   # ... and an inverse
		True
		"""
		...

	def __add__(self, other: Self | N | int) -> Self:
		"""Return `self + other`, per §1.5.2:

			[(a, b)] + [(c, d)] = [(a + c, b + d)]

		Memberwise addition, using ℕ's `+` on each. The result is a new integer, so it is
		built the same way any other is.

		`other` may be a `Z`, an `N`, or a python int, the last two read through the
		constructor on the way in. Any other type returns `NotImplemented`, which lets python
		raise its ordinary `TypeError`. Only the right operand is this method's business —
		`3 + Z(2)` is routed to the given `__radd__`.

		Hints:

		* Screen the type before touching either operand, exactly as `naturals.py`'s `__add__`
		  did, and for the same reason.
		* Then coerce. An `N` or an int is *not* already a pair of naturals, so it has to go
		  through the constructor before its members can be read; a `Z` already is one and
		  must not be rebuilt. Two lines, and they will read the same in `__sub__` and
		  `__mul__`.
		* The `+` signs in the rule are ℕ's, which already works. Nothing recurses here — the
		  recursion happened one exercise ago.

		>>> Z(2) + Z(3), Z(2) + Z(0, 3)
		(5, -1)
		>>> Z(1, 2) + Z(4, 5) == Z(0, 2)          # representatives do not matter
		True
		>>> Z(2) + 3, Z(2) + N(3), Z(2) + 0, Z(2) + -3
		(5, 5, 2, -1)
		>>> Z(2) + "x"
		Traceback (most recent call last):
			...
		TypeError: unsupported operand type(s) for +: 'Z' and 'str'
		"""
		...

	def __sub__(self, other: Self | N | int) -> Self:
		"""Return `self - other`, per §1.5.2:

			[(a, b)] − [(c, d)] = [(a + d, b + c)]

		The operation ℕ could not have: there, `a - b` is undefined whenever b exceeds a,
		which is exactly the gap this whole construction was built to close.

		Same operand handling as `__add__`.

		Hints:

		* There are two routes, and it is worth writing both down before choosing. One is to
		  transcribe the rule above directly. The other is the sentence in §1.5.2 that
		  precedes it: subtraction is the addition of the additive inverse, which makes this
		  method one line in terms of `__add__` and `__neg__`.
		* Check that the two agree, on paper, by expanding the one-liner: which pair does
		  `self + (-other)` build? If it is not literally (a + d, b + c), one of the two is
		  wrong.
		* The one-liner is the better answer here, and for a reason worth naming: it leans
		  only on definitions this class has already made, so it cannot drift from them.

		>>> Z(3) - Z(5), Z(5) - Z(3), Z(3) - Z(3)
		(-2, 2, 0)
		>>> Z(0, 3) - Z(0, 5) == Z(2)             # (-3) - (-5) = 2
		True
		>>> Z(2) - 3, Z(2) - N(3), Z(2) - -3
		(-1, -1, 5)
		"""
		...

	def __mul__(self, other: Self | N | int) -> Self:
		"""Return `self * other`, per §1.5.2:

			[(a, b)] · [(c, d)] = [(ac + bd, ad + bc)]

		Same operand handling as `__add__`.

		Hints:

		* Transcribe the rule. Four products and two sums, all of them ℕ's, all of them
		  already working.
		* It is worth seeing *why* that rule is the right one before writing it. Expand
		  (a − b)(c − d) as if subtraction were available: ac − ad − bc + bd. Collect the
		  terms that survive with a plus into the first member and those with a minus into the
		  second, and the rule is what falls out. That is the whole trick of this
		  construction — a definition mentioning no subtraction, derived from one that does.
		* Check that the sign rule falls out rather than being imposed: a negative times a
		  negative is (0, b)·(0, d) = (bd, 0), positive. Nobody wrote that down; it is a
		  consequence.

		>>> Z(2) * Z(3), Z(2) * Z(0, 3), Z(0, 2) * Z(0, 3)
		(6, -6, 6)
		>>> Z(7) * Z(), Z(0, 7) * Z()
		(0, 0)
		>>> Z(1, 2) * Z(4, 5) == Z(1)             # (-1)(-1) = 1, whatever the representatives
		True
		>>> Z(2) * 3, Z(2) * N(3), Z(2) * -3
		(6, 6, -6)
		"""
		...

	# ---------------------------------------------------------------------------------------
	# RELATIONS — when two integers are the same, when one is below another, and when one
	#             is zero.
	# ---------------------------------------------------------------------------------------

	# ............................................................................ given

	def __lt__(self, other: Self) -> bool:
		"""Return whether `self` < `other`, as `≤` and not `=`.

		Given; not part of the exercise. Derivable boilerplate around `__le__` and `__eq__`,
		both of which carry the actual definitions.

		`functools.total_ordering` would in fact work on this class and fill in all three of
		the derived comparisons, precisely because a bare class starts with none of them —
		the decorator only supplies what is missing, so it would have been no help against a
		`tuple` base, which arrives with all four already wrong. They are written out here
		anyway, because three short methods make the derivation visible where a decorator
		would hide it.

		>>> Z(1) < Z(2), Z(2) < Z(2), Z(2) < Z(1)
		(True, False, False)
		>>> Z(1, 2) < Z(4, 4)
		True
		"""
		below = self.__le__(other)

		return below if below is NotImplemented else below and self != other

	def __ge__(self, other: Self) -> bool:
		"""Return whether `self` ≥ `other`, by asking `other` whether it is ≤ `self`.

		Given; not part of the exercise.

		>>> Z(2) >= Z(1), Z(2) >= Z(2), Z(1) >= Z(2)
		(True, True, False)
		"""
		if not isinstance(other, Z):
			return NotImplemented

		return other.__le__(self)

	def __gt__(self, other: Self) -> bool:
		"""Return whether `self` > `other`, by asking `other` whether it is < `self`.

		Given; not part of the exercise.

		>>> Z(2) > Z(1), Z(2) > Z(2), Z(1) > Z(2)
		(True, False, False)
		"""
		if not isinstance(other, Z):
			return NotImplemented

		return other.__lt__(self)

	# ..................................................................... to implement

	def __eq__(self, other: object) -> bool:
		"""Return whether `self` and `other` are the same integer, per §1.5.2:

			[(a, b)] = [(c, d)]  <=>  a + d = c + b

		This is the method that makes the class a quotient rather than a pair of numbers.
		`object`'s inherited `==` compares identity, so two separately built representatives
		of −1 would be different integers — and even two identical ones would be, which is
		the distinction the equivalence relation of §1.4 exists to erase.

		Returns `NotImplemented` for anything that is not a `Z`, which lets python fall back
		to identity and answer `False` rather than raise. Note the deliberate asymmetry with
		the arithmetic operators, which *do* coerce ints: making `Z(3) == 3` true would oblige
		`__hash__` to agree with `hash(3)` — python's numeric-tower contract — and that is a
		different exercise. Reach for `Z(3) == Z(3)` instead.

		Hints:

		* Transcribe the relation. Both sides need ℕ's `+` and ℕ's `==`, and nothing else —
		  in particular no subtraction, which is the reason §1.5.2 states the rule this way.
		* There is a second, equally correct route: compare `canonical` forms. It is worth
		  seeing that it agrees, but transcribing the relation is the definition, and it does
		  not need a loop.
		* Once this method exists, `hash(Z(3))` raises `TypeError: unhashable type`. That is
		  not a bug you introduced — see `__hash__`.
		* You do **not** need `__ne__`. `object` derives `!=` by negating this method, and
		  passes `NotImplemented` through untouched. Check that `Z(1, 2) != Z(4, 5)` is False
		  once this works, and then leave it alone.

		>>> Z(1, 2) == Z(4, 5), Z(1, 2) == Z(2, 1)
		(True, False)
		>>> Z(0, 0) == Z(7, 7), Z(3) == Z(3, 0)
		(True, True)
		>>> Z(1, 2) != Z(4, 5), Z(1, 2) != Z(2, 1)      # free, from object
		(False, True)
		>>> Z(3) == 3, Z(3) == "x"                      # not a Z: NotImplemented, hence False
		(False, False)
		"""
		...

	def __bool__(self) -> bool:
		"""Return whether this integer is non-zero.

		`object` calls every instance true, zero included. That is wrong on its own, and
		quietly wrong in any code that writes `if z:`.

		Hint: zero is the class [(0, 0)], and §1.5.2's equality says which *other* pairs are
		in it. Read that rule with c = d and see what it collapses to; the answer is a
		one-line comparison of the two members, and it needs no arithmetic at all.

		>>> bool(Z(3)), bool(Z(0, 3))
		(True, True)
		>>> bool(Z()), bool(Z(7, 7)), bool(Z(2) - Z(2))
		(False, False, False)
		"""
		...

	def __le__(self, other: Self) -> bool:
		"""Return whether `self` ≤ `other`, per §1.5.2:

			[(a, b)] ≤ [(c, d)]  <=>  a + d ≤ c + b

		`object` supplies no ordering at all, so until this exists `<` raises TypeError.

		Takes a `Z` only and returns `NotImplemented` otherwise, matching `__eq__` rather than
		the arithmetic operators. `__lt__`, `__gt__` and `__ge__` are derived from this one
		and given.

		Hints:

		* Transcribe the rule. It is `__eq__` with `≤` in place of `=`, on the very same two
		  sums — which is the sense in which ℤ inherits its order from ℕ rather than inventing
		  one.
		* The `<=` on the right-hand side is ℕ's, which `naturals.py` got for free from
		  `frozenset`: on von Neumann naturals, ⊆ *is* ≤. So this method bottoms out in a
		  subset test, two constructions down.
		* Check it against a case that a naive memberwise comparison gets wrong — (0, 0)
		  against (1, 5), which is 0 against −4 — before trusting it.

		>>> Z(0, 1) <= Z(1, 0), Z(1, 0) <= Z(0, 1)
		(True, False)
		>>> Z(0, 0) <= Z(1, 5), Z(1, 5) <= Z(0, 0)      # 0 ≤ -4 is false; -4 ≤ 0 is true
		(False, True)
		>>> Z(3) <= Z(3), Z(1, 2) <= Z(4, 5)            # ≤ is reflexive, up to ∼
		(True, True)
		>>> sorted([Z(2), Z(0, 3), Z(), Z(1)])
		[-3, 0, 1, 2]
		"""
		...
