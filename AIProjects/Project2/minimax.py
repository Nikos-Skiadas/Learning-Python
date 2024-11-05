from __future__ import annotations


import enum
import typing





class MinimaxGame:
    def __init__(self, terminal_states: dict):
        """
        Initialize the game with given terminal states.

        Args:
            terminal_states (dict): A dictionary where keys are terminal state names and
                                    values are the utility scores for Player 1 (maximizing player).
        """
        self.terminal_states = terminal_states
        self.moves = {}  # Dictionary to store possible moves for each state

    def add_moves(self, state, next_states):
        """
        Define possible moves for a given state.

        Args:
            state (str): The current state.
            next_states (list): List of possible next states from the current state.
        """
        self.moves[state] = next_states

    def minimax(self, state, maximizing_player=True):
        """
        Minimax algorithm to evaluate the utility of a state.

        Args:
            state (str): The current state.
            maximizing_player (bool): True if maximizing, False if minimizing.

        Returns:
            int: The utility score of the best move from the given state.
        """
        if state in self.terminal_states:  # Check if it's a terminal state
            return self.terminal_states[state]

        if maximizing_player:
            max_eval = float('-inf')
            for next_state in self.moves[state]:
                eval = self.minimax(next_state, maximizing_player=False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for next_state in self.moves[state]:
                eval = self.minimax(next_state, maximizing_player=True)
                min_eval = min(min_eval, eval)
            return min_eval

    def best_move(self, initial_state):
        """
        Determine the best move for the maximizing player from the initial state.

        Args:
            initial_state (str): The starting state of the game.

        Returns:
            str: The best next state for the maximizing player.
        """
        best_value = float('-inf')
        best_state = None
        for next_state in self.moves[initial_state]:
            move_value = self.minimax(next_state, maximizing_player=False)
            if move_value > best_value:
                best_value = move_value
                best_state = next_state
        return best_state

# Example Usage:
# Define terminal states and their utility scores
terminal_states = {
    "Win": 1,
    "Loss": -1,
    "Draw": 0
}


if __name__ == "__main__":
	# Initialize game with terminal state utilities
	game = MinimaxGame(terminal_states)

	# Define the game's move structure
	game.add_moves("Start", ["Move1", "Move2"])
	game.add_moves("Move1", ["Win", "Draw"])
	game.add_moves("Move2", ["Loss", "Draw"])

	# Determine the best move for Player 1 from the initial state
	initial_state = "Start"
	best_next_state = game.best_move(initial_state)
	print(f"The best move for Player 1 from '{initial_state}' is '{best_next_state}'.")



















class Action:

	...


class State:

	def __init__(self):
		self.player = True
        self.actions


	@property
	def actions(self) -> typing.Iterable[Action]:
		...



class Game:

	def __init__(self, initial: State):
		self._initial = initial
		self._max = True


	@property
	def is_max(self) -> bool:
		return self._max


	def result(self, state: State, action: Action) -> State:
		...

	def actions(self, state: State) -> typing.Iterable[Action]:
		...

	def is_terminal(self, state: State) -> bool:
		...

	def utility(self, state: State, player: Player) -> float:
		...



