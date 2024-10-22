# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import game
import typing


class SearchProblem[State]:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self) -> State:
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state: State):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state) -> typing.Iterable[State]:
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions: typing.Iterable[game.Directions]) -> float:
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem: SearchProblem) -> list[game.Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = game.Directions.SOUTH
    w = game.Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> list[game.Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    frontier = util.Stack()
    start_state = problem.getStartState()

    frontier.push((start_state, [], []))  # (current state, path to state, visited states)

    while not frontier.isEmpty():
        # Pop the state from the frontier
        current_state, actions, visited = frontier.pop()

        # Check if current state is the goal state
        if problem.isGoalState(current_state):
            return actions

        # Avoid revisiting already explored nodes
        for successor, action, _ in problem.getSuccessors(current_state):
            if successor not in visited:
                new_actions = actions + [action]
                frontier.push((successor, new_actions, visited + [current_state]))

    return []  # If no solution found

def breadthFirstSearch(problem: SearchProblem) -> list[game.Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    start_state = problem.getStartState()

    frontier.push((start_state, [], [start_state]))  # (current state, path to state, visited states)

    while not frontier.isEmpty():
        # Pop the state from the frontier
        current_state, actions, visited = frontier.pop()

        # Check if current state is the goal state
        if problem.isGoalState(current_state):
            return actions

        # Avoid revisiting already explored nodes
        for successor, action, _ in problem.getSuccessors(current_state):
            if successor not in visited:
                new_actions = actions + [action]
                frontier.push((successor, new_actions, visited + [current_state]))

    return []  # If no solution found


def uniformCostSearch(problem: SearchProblem) -> list[game.Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> list[game.Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
