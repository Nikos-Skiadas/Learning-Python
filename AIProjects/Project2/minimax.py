from __future__ import annotations


import collections
import enum
import json
import math
import typing


class Game[State: typing.Hashable]:

	def __init__(self, state: State, *children: Game,
		terminal_utility: float = 0,
	):
		self.state = state
		self.children = list(children)
		self.terminal_utility = terminal_utility

	@classmethod
	def max(cls, state: Game) -> float:
		if state.terminal:
			return state.utility

		return max(cls.min(child) for child in state.children)


	@classmethod
	def min(cls, state: Game) -> float:
		if state.terminal:
			return state.utility

		return min(cls.max(child) for child in state.children)


	@property
	def minmax(self) -> Game | None:
		if self.terminal:
			return None

		return max(self.children,
			key = Game.min,
		)

	@property
	def terminal(self) -> bool:
		return not self.children

	@property
	def utility(self) -> float:
		return self.terminal_utility if self.terminal else Game.max(self)


	@classmethod
	def from_json(cls, start: State, next: dict | float) -> Game:
		if isinstance(next, dict):
			return Game(start, *(Game.from_json(turn, children) for turn, children in next.items()))

		return Game(start,
			terminal_utility = next
		)



	@property
	def json(self) -> dict:
		dictionary = {}

		if self.terminal:
			dictionary[self.state] = self.utility

		else:
			for state in self.children:
				dictionary[state.state] = state.json

		return dictionary



if __name__ == "__main__":
	states = {
		"aa": {
			"aaa": {
				"aaaa": +4,
				"aaab": +8,
			},
			"aab": {
				"aaba": +9,
				"aabb": +3,
			},
		},
		"ab": {
			"aba": {
				"abaa": +2,
				"abab": -2,
			},
			"abb": {
				"abba": +9,
				"abbb": -1,
			},
			"abc": {
				"abca": +8,
				"abcb": +4,
			},
		},
		"ac": {
			"aca": {
				"acaa": +3,
				"acab": +6,
				"acac": +5,
			},
			"acb": {
				"acba": +7,
				"acbb": +1,
			},
		},
	}

	print(
		json.dumps(Game.from_json("a", states).json,
			indent = 4,
		)
	)
