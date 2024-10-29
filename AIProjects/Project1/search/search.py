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
    # Initialize the frontier with a Stack (LIFO), since DFS explores deeper nodes first.
    # Each element in the frontier is a tuple: (current state, path to reach the state)
    frontier = util.Stack()
    start_state = problem.getStartState()

    # Push the start state onto the frontier with an empty path, since no actions are taken yet.
    frontier.push((start_state, []))

    # Initialize a visited list to track explored states, preventing revisits and infinite loops.
    visited = []

    # Continue exploring until there are no nodes left in the frontier (LIFO-based exploration).
    while not frontier.isEmpty():
        current_state, actions = frontier.pop()

        # Check if the current state is the goal. If yes, return the path of actions to this state.
        if problem.isGoalState(current_state):
            return actions

        # Mark it as visited.
        if current_state not in visited:
            visited.append(current_state)

            # Get all successors (next states) of the current state.
            # For each successor, push it onto the frontier with the updated path of actions.
            for next_state, action, _ in problem.getSuccessors(current_state):
                frontier.push((next_state, actions + [action]))

    # If no solution is found after exploring all possible paths, return an empty list.
    return []

def breadthFirstSearch(problem: SearchProblem) -> list[game.Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Initialize the frontier with a Queue (FIFO) because BFS explores nodes level by level.
    # Each item in the frontier is a tuple: (current state, path to reach that state)
    frontier = util.Queue()
    start_state = problem.getStartState()

    # Push the start state into the frontier with an empty path, as no actions are needed to reach the start.
    frontier.push((start_state, []))

    # Initialize a visited list with the start state to prevent revisiting it.
    visited = [start_state]

    # Continue exploring until there are no nodes left in the frontier (FIFO-based exploration).
    while not frontier.isEmpty():
        current_state, actions = frontier.pop()

        # Check if the current state is the goal. If yes, return the path of actions to reach this state.
        if problem.isGoalState(current_state):
            return actions

        # Get all successors (next states) of the current state.
        for next_state, action, _ in problem.getSuccessors(current_state):
            if next_state not in visited:
                visited.append(next_state)  # Track it in the visited list

                # Push the successor into the frontier with the updated path of actions.
                frontier.push((next_state, actions + [action]))

    # If no solution is found after exploring all possible paths, return an empty list.
    return []


def uniformCostSearch(problem: SearchProblem) -> list[game.Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Initialize the frontier with a Priority Queue. Each item in the frontier
    # is a tuple: (total path cost to state, current state, path to reach that state).
    frontier = util.PriorityQueue()
    start_state = problem.getStartState()

    # Push the start state into the frontier with zero cost and an empty path
    frontier.push((start_state, [], 0), 0)  # (state, path, total cost), priority=0

    # Initialize a dictionary to store the lowest cost found for each state to avoid revisiting with higher cost
    visited_costs = {start_state: 0}

    while not frontier.isEmpty():
        # Pop the state with the lowest path cost from the frontier
        current_state, actions, current_cost = frontier.pop()

        # Check if the current state is the goal. If yes, return the path of actions to reach this state.
        if problem.isGoalState(current_state):
            return actions

        # Only proceed if the current cost is the lowest known for this state to prevent revisits
        if current_cost <= visited_costs.get(current_state, float('inf')):
            # Explore each successor (state, action, step cost) of the current state.
            for next_state, action, step_cost in problem.getSuccessors(current_state):
                new_cost = current_cost + step_cost  # Calculate the new path cost to reach the successor

                # Only add the successor if it hasn't been visited at a lower cost
                if next_state not in visited_costs or new_cost < visited_costs[next_state]:
                    visited_costs[next_state] = new_cost  # Update the cost to reach this state
                    new_actions = actions + [action]  # Update the path of actions
                    frontier.push((next_state, new_actions, new_cost), new_cost)  # Push to frontier with new cost

    # If no solution is found after exploring all possible paths, return an empty list.
    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> list[game.Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # Initialize the frontier with a Priority Queue, using (cost + heuristic) as priority.
    # Each item in the frontier is a tuple: (current state, path to state, path cost).
    frontier = util.PriorityQueue()
    start_state = problem.getStartState()

    # Initial cost is 0, and heuristic is applied from the start
    start_heuristic = heuristic(start_state, problem)
    frontier.push((start_state, [], 0), start_heuristic)  # (state, path, cost), priority=0+heuristic

    visited_costs = {start_state: 0}

    while not frontier.isEmpty():
        # Pop the state with the lowest (cost + heuristic) from the frontier
        current_state, actions, current_cost = frontier.pop()

        # Check if the current state is the goal. If yes, return the path of actions to reach this state.
        if problem.isGoalState(current_state):
            return actions

        # Proceed only if the current cost is the lowest known for this state
        if current_cost <= visited_costs.get(current_state, float('inf')):
            # Explore each successor (state, action, step cost) of the current state
            for next_state, action, step_cost in problem.getSuccessors(current_state):
                new_cost = current_cost + step_cost  # Calculate the new path cost to reach successor
                heuristic_cost = new_cost + heuristic(next_state, problem)  # Add heuristic for A* priority

                # Add successor if it's unvisited or can be reached at a lower cost
                if next_state not in visited_costs or new_cost < visited_costs[next_state]:
                    visited_costs[next_state] = new_cost
                    new_actions = actions + [action]
                    frontier.push((next_state, new_actions, new_cost), heuristic_cost)

    # If no solution is found after exploring all paths, return an empty list.
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
