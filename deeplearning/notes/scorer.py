"""Axiom scorer for the number systems constructed in chapter 1.

Each type built in this chapter (ℕ now; ℤ, ℚ, ... to follow) claims to satisfy a specific
list of axioms from the notes. This script turns that claim into a score: it quantifies
every applicable law over a small sample of values and reports what holds, what fails, and
with which counterexample.

Three kinds of law are scored separately, because failing them means different things:

* **axioms** — the definitions the type is *built* from (§1.4.1). A failure here means the
  implementation does not define the object it claims to define.
* **theorems** — consequences that must follow if the axioms are right (commutativity,
  associativity, distributivity). A failure here means the axioms were transcribed in a
  way that does not entail what it should.
* **contracts** — the python-level promises of the class: which operands it accepts, which
  errors it owes the caller. Independent of the mathematics, and the part most easily lost
  while refactoring.

Usage:

	python3 scorer.py              # score every registered type
	python3 scorer.py N            # score one
	python3 scorer.py -v           # show a counterexample for each failure

Exit status is 0 only when every law passes.

Registering a new type: write a factory returning a `Spec`, decorate it with
`@register("Z")`, and list the law families that apply in `features`. Everything else —
quantification, reporting, scoring — is shared. The `same` hook exists because structural
`==` on these constructions can be exponential (see the `numbers` module docstring), so a
canonical `repr` is used as the equality oracle and then *validated* against `==` by the
`oracle` family.
"""


from __future__ import annotations


import sys


# `numbers.py` in this directory shadows the standard library's `numbers` module, which
# `fractions`, `decimal` and `statistics` all import. Drop the script's own directory from
# the search path and load targets by explicit file path instead, so that this scorer keeps
# working once ℚ needs `fractions`.
from pathlib import Path


ROOT = Path(__file__).resolve().parent

for _entry in ("", ".", str(ROOT)):
	while _entry in sys.path:
		sys.path.remove(_entry)


import importlib.util
import operator

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any


def load(filename: str, name: str | None = None) -> Any:
	"""Import a module from this directory by path, bypassing the shadowed name."""
	path = ROOT / filename
	spec = importlib.util.spec_from_file_location(name or f"chapter_{path.stem}", path)

	if spec is None or spec.loader is None:
		raise ImportError(f"cannot load {path}")

	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)

	return module


@dataclass(frozen=True)
class Result:
	family: str
	law: str
	ok: bool
	kind: str = "axiom"          # axiom | theorem | contract
	detail: str = ""


@dataclass
class Spec:
	"""Everything the shared law families need in order to score one type."""

	name: str                                   # "ℕ  the natural numbers"
	origin: str                                 # "numbers.N"
	source: str                                 # the file under test, for leak detection
	cls: type
	features: frozenset[str]
	embed: Callable[[int], Any]                 # int -> the type under test
	unembed: Callable[[Any], int]               # the type -> int, for agreement checks
	same: Callable[[Any, Any], bool]            # equality oracle
	sample: Sequence[int]                       # values for unary/binary laws
	small: Sequence[int]                        # values for ternary laws
	succ: Callable[[Any], Any] | None = None
	pred: Callable[[Any], Any] | None = None
	normalize: Callable[[Any], Any] | None = None    # operand reader; defaults to `cls`
	bad_operands: Sequence[Any] = ()            # values the operators must refuse
	domain_errors: Sequence[tuple[str, Callable[[], Any], type[BaseException]]] = ()
	order: Callable[[Any, Any], bool] = operator.le

	def values(self) -> list[Any]:
		return [self.embed(k) for k in self.sample]


REGISTRY: dict[str, Callable[[], Spec]] = {}


def register(key: str) -> Callable[[Callable[[], Spec]], Callable[[], Spec]]:
	def decorate(factory: Callable[[], Spec]) -> Callable[[], Spec]:
		REGISTRY[key] = factory
		return factory

	return decorate


# --------------------------------------------------------------------------- probing


def check(family: str, law: str, probe: Callable[[], Any], kind: str = "axiom") -> Result:
	"""Run one law, turning any exception into a failure rather than a crash."""
	try:
		outcome = probe()
	except Exception as exc:                     # a law that explodes has not held
		return Result(family, law, False, kind, f"{type(exc).__name__}: {exc}")

	ok, detail = outcome if isinstance(outcome, tuple) else (bool(outcome), "")

	return Result(family, law, bool(ok), kind, "" if ok else detail)


def forall(cases: Iterable[tuple], predicate: Callable[..., bool]) -> tuple[bool, str]:
	"""True iff `predicate` holds for every case; otherwise the first counterexample."""
	for case in cases:
		try:
			if not predicate(*case):
				return False, f"first counterexample: {', '.join(map(repr, case))}"
		except Exception as exc:
			return False, f"{type(exc).__name__} at {', '.join(map(repr, case))}: {exc}"

	return True, ""


def pairs(seq: Sequence[Any]) -> Iterator[tuple[Any, Any]]:
	return product(seq, repeat=2)


def triples(seq: Sequence[Any]) -> Iterator[tuple[Any, Any, Any]]:
	return product(seq, repeat=3)


# ------------------------------------------------------------------------ law families


def oracle(s: Spec) -> Iterator[Result]:
	"""The equality oracle must agree with the type's own `==` where that is affordable."""
	yield check("oracle", "canonical repr agrees with ==", lambda: forall(
		pairs(s.values()), lambda a, b: s.same(a, b) == (a == b)))
	yield check("oracle", "embedding is injective", lambda: forall(
		pairs(s.sample), lambda j, k: s.same(s.embed(j), s.embed(k)) == (j == k)))
	yield check("oracle", "unembed inverts embed", lambda: forall(
		((k,) for k in s.sample), lambda k: s.unembed(s.embed(k)) == k))


def peano(s: Spec) -> Iterator[Result]:
	"""The five Peano axioms of §1.4.1."""
	zero, succ = s.embed(0), s.succ
	assert succ is not None

	yield check("peano", "1. 0 is a natural number", lambda: (
		isinstance(zero, s.cls), f"type is {type(zero).__name__}"))
	yield check("peano", "2. σ is closed on the type", lambda: forall(
		((v,) for v in s.values()), lambda v: isinstance(succ(v), s.cls)))
	yield check("peano", "3. 0 is no one's successor", lambda: forall(
		((v,) for v in s.values()), lambda v: not s.same(succ(v), zero)))
	yield check("peano", "4. σ is injective", lambda: forall(
		pairs(s.values()), lambda a, b: s.same(succ(a), succ(b)) == s.same(a, b)))
	yield check("peano", "5. σ-chain from 0 enumerates the sample (surrogate)", lambda: (
		[s.unembed(v) for v in walk(zero, succ, len(s.sample))] == list(range(len(s.sample))),
		"iterating σ from 0 does not reproduce 0, 1, 2, ..."))

	if (pred := s.pred) is not None:
		yield check("peano", "σ and its inverse round-trip", lambda: forall(
			((v,) for v in s.values()), lambda v: s.same(pred(succ(v)), v)))


def walk(start: Any, step: Callable[[Any], Any], count: int) -> list[Any]:
	out, current = [], start

	for _ in range(count):
		out.append(current)
		current = step(current)

	return out


def cardinality(s: Spec) -> Iterator[Result]:
	"""§1.3.2 and §1.4.1: in this construction n is a set of exactly n elements."""
	yield check("cardinality", "n has exactly n elements", lambda: forall(
		((k,) for k in s.sample), lambda k: len(s.embed(k)) == k))
	yield check("cardinality", "the elements of n are 0 .. n-1", lambda: forall(
		((k,) for k in s.sample),
		lambda k: sorted(map(s.unembed, s.embed(k))) == list(range(k))))
	yield check("cardinality", "the elements of n form a ⊆-chain", lambda: forall(
		((k,) for k in s.sample),
		lambda k: all(a <= b or b <= a for a, b in pairs(list(s.embed(k))))))


def addition(s: Spec) -> Iterator[Result]:
	"""The addition axioms of §1.4.1, plus agreement with python's own +."""
	succ = s.succ
	assert succ is not None

	yield check("addition", "1. a + 0 = a", lambda: forall(
		((v,) for v in s.values()), lambda v: s.same(v + s.embed(0), v)))
	yield check("addition", "2. a + σ(b) = σ(a + b)", lambda: forall(
		pairs(s.values()), lambda a, b: s.same(a + succ(b), succ(a + b))))
	yield check("addition", "agrees with python's +", lambda: forall(
		pairs(s.sample),
		lambda j, k: s.unembed(s.embed(j) + s.embed(k)) == j + k))
	yield check("addition", "0 + a = a  (needs its own induction)", lambda: forall(
		((v,) for v in s.values()), lambda v: s.same(s.embed(0) + v, v)), "theorem")


def multiplication(s: Spec) -> Iterator[Result]:
	"""The multiplication axioms of §1.4.1, plus agreement with python's own *."""
	succ = s.succ
	assert succ is not None

	yield check("multiplication", "1. a · 0 = 0", lambda: forall(
		((v,) for v in s.values()), lambda v: s.same(v * s.embed(0), s.embed(0))))
	yield check("multiplication", "2. a · σ(b) = a + a · b", lambda: forall(
		pairs(s.values()), lambda a, b: s.same(a * succ(b), a + a * b)))
	yield check("multiplication", "agrees with python's *", lambda: forall(
		pairs(s.sample),
		lambda j, k: s.unembed(s.embed(j) * s.embed(k)) == j * k))
	yield check("multiplication", "a · 1 = a  (unfolds from the axioms)", lambda: forall(
		((v,) for v in s.values()), lambda v: s.same(v * s.embed(1), v)))
	yield check("multiplication", "1 · a = a  (needs its own induction)", lambda: forall(
		((v,) for v in s.values()), lambda v: s.same(s.embed(1) * v, v)), "theorem")


def algebra(s: Spec) -> Iterator[Result]:
	"""Consequences that must follow if the axioms above were transcribed correctly."""
	small = [s.embed(k) for k in s.small]

	yield check("algebra", "+ is associative", lambda: forall(
		triples(small), lambda a, b, c: s.same((a + b) + c, a + (b + c))), "theorem")
	yield check("algebra", "+ is commutative", lambda: forall(
		pairs(s.values()), lambda a, b: s.same(a + b, b + a)), "theorem")
	yield check("algebra", "· is associative", lambda: forall(
		triples(small), lambda a, b, c: s.same((a * b) * c, a * (b * c))), "theorem")
	yield check("algebra", "· is commutative", lambda: forall(
		pairs(s.values()), lambda a, b: s.same(a * b, b * a)), "theorem")
	yield check("algebra", "· distributes over +", lambda: forall(
		triples(small), lambda a, b, c: s.same(a * (b + c), a * b + a * c)), "theorem")
	yield check("algebra", "a · 0 = 0 annihilates", lambda: forall(
		((v,) for v in s.values()), lambda v: s.same(v * s.embed(0), s.embed(0))), "theorem")


def order(s: Spec) -> Iterator[Result]:
	"""§1.4.1: a ≤ b iff some c has a + c = b, and that relation is a total order."""
	le, values = s.order, s.values()

	yield check("order", "reflexive", lambda: forall(
		((v,) for v in values), lambda v: le(v, v)))
	yield check("order", "antisymmetric", lambda: forall(
		pairs(values), lambda a, b: not (le(a, b) and le(b, a)) or s.same(a, b)))
	yield check("order", "transitive", lambda: forall(
		triples([s.embed(k) for k in s.small]),
		lambda a, b, c: not (le(a, b) and le(b, c)) or le(a, c)))
	yield check("order", "total", lambda: forall(
		pairs(values), lambda a, b: le(a, b) or le(b, a)))
	yield check("order", "a ≤ b iff ∃c, a + c = b", lambda: forall(
		pairs(values),
		lambda a, b: le(a, b) == any(s.same(a + c, b) for c in values)))

	if (succ := s.succ) is not None:
		yield check("order", "σ is strictly increasing", lambda: forall(
			((v,) for v in values), lambda v: le(v, succ(v)) and not s.same(v, succ(v))))


def coercion(s: Spec) -> Iterator[Result]:
	"""A python int on the right of an operator must behave as its embedding."""
	yield check("coercion", "a + int(k) = a + embed(k)", lambda: forall(
		pairs(s.sample),
		lambda j, k: s.same(s.embed(j) + k, s.embed(j) + s.embed(k))), "contract")
	yield check("coercion", "a · int(k) = a · embed(k)", lambda: forall(
		pairs(s.sample),
		lambda j, k: s.same(s.embed(j) * k, s.embed(j) * s.embed(k))), "contract")
	read = s.normalize or s.cls

	yield check("coercion", "reading an existing value is the identity", lambda: forall(
		((v,) for v in s.values()), lambda v: s.same(read(v), v)), "contract")
	yield check("coercion", "reading is idempotent", lambda: forall(
		((v,) for v in s.values()), lambda v: s.same(read(read(v)), read(v))), "contract")


def contracts(s: Spec) -> Iterator[Result]:
	"""Operand typing and declared errors: the promises the class makes to callers."""
	one = s.embed(1)

	for bad in s.bad_operands:
		for label, op in (("+", operator.add), ("·", operator.mul)):
			yield check(
				"contracts",
				f"a {label} {bad!r} is refused",
				lambda bad=bad, op=op: refuses(op, one, bad, s.source),
				"contract",
			)

	for label, thunk, expected in s.domain_errors:
		yield check(
			"contracts",
			label,
			lambda thunk=thunk, expected=expected: raises(thunk, expected),
			"contract",
		)


def refuses(op: Callable[[Any, Any], Any], value: Any, bad: Any, source: str) -> tuple[bool, str]:
	"""An unsupported operand must be refused by python's dispatch, not by a leaked error.

	Returning `NotImplemented` hands the decision back to the interpreter, which then either
	tries the reflected operation or raises "unsupported operand type(s)". Either outcome is
	correct, and the wording differs: `N * "x"` legitimately ends up in `str.__rmul__` and
	complains about multiplying a sequence. What must *not* happen is an error escaping from
	inside the type's own code, so the test is where the traceback has been rather than what
	it says.
	"""
	try:
		outcome = op(value, bad)
	except TypeError as exc:
		trace = exc.__traceback__

		while trace is not None:
			if trace.tb_frame.f_code.co_filename == source:
				return False, f"error leaked from {source.rsplit('/', 1)[-1]}: {exc}"

			trace = trace.tb_next

		return True, ""
	except Exception as exc:
		return False, f"{type(exc).__name__} instead of TypeError: {exc}"

	return False, f"no error at all, returned {outcome!r}"


def raises(thunk: Callable[[], Any], expected: type[BaseException]) -> tuple[bool, str]:
	try:
		outcome = thunk()
	except expected:
		return True, ""
	except Exception as exc:
		return False, f"{type(exc).__name__} instead of {expected.__name__}: {exc}"

	return False, f"no {expected.__name__}, returned {outcome!r}"


FAMILIES: dict[str, Callable[[Spec], Iterator[Result]]] = {
	"oracle": oracle,
	"peano": peano,
	"cardinality": cardinality,
	"addition": addition,
	"multiplication": multiplication,
	"algebra": algebra,
	"order": order,
	"coercion": coercion,
	"contracts": contracts,
}


# ------------------------------------------------------------------------------ types


@register("N")
def naturals() -> Spec:
	module = load("numbers.py")
	N = module.N

	return Spec(
		name="ℕ  the natural numbers",
		origin="numbers.N",
		source=module.__file__,
		cls=N,
		features=frozenset(FAMILIES),
		embed=N,
		unembed=len,
		same=lambda a, b: repr(a) == repr(b),
		sample=range(7),
		small=range(4),
		succ=lambda v: v.next,
		pred=lambda v: v.prev,
		bad_operands=("x", 2.0, 0.0, None, []),
		domain_errors=(
			("N(-1) rejects a negative int", lambda: N(-1), ValueError),
			("N(0).prev rejects zero", lambda: N().prev, ValueError),
		),
	)


# ---------------------------------------------------------------------------- scoring


def score(spec: Spec) -> list[Result]:
	results: list[Result] = []

	for key, family in FAMILIES.items():
		if key in spec.features:
			results.extend(family(spec))

	return results


def report(spec: Spec, results: list[Result], verbose: bool) -> bool:
	width = max(len(r.law) for r in results) + 2

	print(f"\n\033[1m{spec.name}\033[0m   ({spec.origin})")
	print(f"  sample: {list(spec.sample)}   ternary sample: {list(spec.small)}")

	shown = ""
	for result in results:
		if result.family != shown:
			shown = result.family
			print(f"\n  \033[2m{shown}\033[0m")

		mark = "\033[32m✓\033[0m" if result.ok else "\033[31m✗\033[0m"
		tag = "" if result.kind == "axiom" else f"\033[2m[{result.kind}]\033[0m"
		print(f"    {mark} {result.law:<{width}} {tag}")

		if not result.ok and (verbose or result.detail):
			print(f"        \033[31m{result.detail}\033[0m")

	print()
	total_ok = 0
	for kind in ("axiom", "theorem", "contract"):
		group = [r for r in results if r.kind == kind]

		if not group:
			continue

		ok = sum(r.ok for r in group)
		total_ok += ok
		mark = "\033[32m✓\033[0m" if ok == len(group) else "\033[31m✗\033[0m"
		print(f"  {kind + 's':<12} {ok:>3}/{len(group):<3} {100 * ok // len(group):>3}%  {mark}")

	pct = 100 * total_ok // len(results)
	verdict = "\033[32mvalid\033[0m" if total_ok == len(results) else "\033[31mincomplete\033[0m"
	print(f"  {'─' * 30}\n  {'total':<12} {total_ok:>3}/{len(results):<3} {pct:>3}%  {verdict}")

	return total_ok == len(results)


def main(argv: Sequence[str]) -> int:
	verbose = any(a in ("-v", "--verbose") for a in argv)
	wanted = [a for a in argv if not a.startswith("-")] or list(REGISTRY)
	unknown = [key for key in wanted if key not in REGISTRY]

	if unknown:
		print(f"unknown type(s): {', '.join(unknown)}", file=sys.stderr)
		print(f"registered: {', '.join(REGISTRY)}", file=sys.stderr)
		return 2

	perfect = True
	for key in wanted:
		spec = REGISTRY[key]()
		perfect &= report(spec, score(spec), verbose)

	return 0 if perfect else 1


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
