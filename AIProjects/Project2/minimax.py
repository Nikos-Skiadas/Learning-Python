from __future__ import annotations


import collections
import enum
import json
import math
import typing



class Action:

	...


class State:

	def __init__(self, node: str, *children: State,
		terminal_utility: float = 0,
	):
		self.name = node
		self.children = list(children)
		self.terminal_utility = terminal_utility

	@classmethod
	def max(cls, state: State) -> float:
		if state.terminal:
			return state.utility

		return max(cls.min(child) for child in state.children)


	@classmethod
	def min(cls, state: State) -> float:
		if state.terminal:
			return state.utility

		return min(cls.max(child) for child in state.children)


	@property
	def minmax(self) -> State | None:
		if self.terminal:
			return None

		return max(self.children,
			key = State.min,
		)

	@property
	def terminal(self) -> bool:
		return not self.children

	@property
	def utility(self) -> float:
		return self.terminal_utility if self.terminal else State.max(self)

	@property
	def json(self) -> dict:
		dict = {}

		if self.terminal:
			dict[self.name] = self.utility

		else:
			for state in self.children:
				dict[state.name] = state.json

		return dict



if __name__ == "__main__":
	game = State("a",
		State("aa",
			State("aaa",
				State("aaaa", terminal_utility = +4),
				State("aaab", terminal_utility = +8),
			),
			State("aab",
				State("aaba", terminal_utility = +9),
				State("aabb", terminal_utility = +3),
			),
		),
		State("ab",
			State("aba",
				State("abaa", terminal_utility = +2),
				State("abab", terminal_utility = -2),
			),
			State("abb",
				State("abba", terminal_utility = +9),
				State("abbb", terminal_utility = -1),
			),
			State("abc",
				State("abca", terminal_utility = +8),
				State("abcb", terminal_utility = +4),
			),
		),
		State("ac",
			State("aca",
				State("acaa", terminal_utility = +3),
				State("acab", terminal_utility = +6),
				State("acac", terminal_utility = +5),
			),
			State("acb",
				State("acba", terminal_utility = +7),
				State("acbb", terminal_utility = +1),
			),
		),
	)

	print(
		json.dumps(game.json,
			indent = 4,
		)
	)
