# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions, Actions
import random, util, math, typing

from game import Agent, AgentState
from pacman import GameState


def mean(numbers: typing.Collection[float]) -> float:
    """Helper function to calculate the mean of a collection of numbers."""
    return sum(numbers) / len(numbers)


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index, score in enumerate(scores) if score == bestScore]
        # bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: Actions):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()

        score = 0
        currentFood = currentGameState.getFood().asList()  # type: ignore

        for newGhostState in newGhostStates:
            newGhostPos = newGhostState.getPosition()
            movesAway = manhattanDistance(newPos, newGhostPos)

            # Food is good:
            if newPos in currentFood:
                score += 1

            # Walls should always be avoided:
            if currentGameState.hasWall(*newPos):
                score -= math.inf

            # Eat ghost:
            if movesAway <= newGhostState.scaredTimer:
                score += movesAway

            # Run away from ghost but compete with eating ghost above:
            if movesAway < 2:
                score -= 2

            # Add 1 / minimum distance to nearest food:
            score -= .1 * min(manhattanDistance(newPos, foodPos) for foodPos in currentFood)

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    # NOTE: pacman index has been moved to a class variable as is the same across all pacmans
    index = 0 # Pacman is always agent index 0


    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.evaluationFunction: typing.Callable[[GameState], float] = util.lookup(evalFn, globals())
        self.depth = int(depth)


    def better(self,
        min: float,
        max: float,
        new: float, agentIndex: int | None = None
    ) -> bool:
        """
        """
        if agentIndex is None: agentIndex = self.index

        return max < new and agentIndex == self.index \
            or min > new and agentIndex != self.index

    def opt(self, gameState: GameState,
        agentIndex: int | None = None,
        depth: int | None = None,
        prune: bool = False,
        a: float = -math.inf,
        b: float = +math.inf,
    ) -> tuple[float, Actions | None]:
        """Helper function replacing min and max helper functions usually found in minimax algorithms.

        I believe it is simpler to check `agentIndex` and act accordingly instead of creating two functions.
        Here we do not have a one-to-one game, we have one-to-many, so we get a max-min-min-...-max-min-min-...-gameover.

        This is made to expand the game tree at fixed depth, given as an attribute in the class.

        Finally, this is made a local function for better recursion.
        That way I can use argument default values as syntactic sugar.

        The method optionally supports alpha-beta pruning with set pruning bounds.
        If `prune` is false, `a` and `b` will be ignored.
        """
        if agentIndex is None: agentIndex = self.index
        if depth is None: depth = self.depth

        if not depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # Cycle to the next agent first:
        agentIndexNext = agentIndex + 1

        # Reset agent index on full cycle:
        if agentIndexNext == gameState.getNumAgents():
            agentIndexNext = self.index  # cycle back to pacman
            depth -= 1  # and write out one depth level

        # Initialize the best action to None:
        best_value = -math.inf if agentIndex == self.index else +math.inf  # either max or min logic depending on who plays
        best_action = None

        # Get actions and results for the enxt agent (presumably a ghost):
        for action in gameState.getLegalActions(agentIndex):
            state = gameState.generateSuccessor(agentIndex, action)  # query succerssors one by one
            value, _ = self.opt(state, agentIndexNext, depth, prune, a,  b)  # evaluate successor and get the value of it

            # Track the best value together with the action it corresponds to:
            if self.better(
                best_value,
                best_value, value, agentIndex
            ):
                best_value = value  # keep track of the optimum value depending on who plays
                best_action = action  # keep track of the corresponding action depending on who plays

            # If pruning is necessary cut the loop here:
            if prune:
                if self.better(
                    a,
                    b, value, agentIndex
                ):
                    break

                # Update pruning bounds otherwise:
                a = max(a, best_value) if agentIndex == self.index else a
                b = min(b, best_value) if agentIndex != self.index else b

        # Return both optimum plus the action it corresponds to:
        return best_value, best_action


class MinimaxAgent(MultiAgentSearchAgent):

    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        _, action = self.opt(gameState)

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        _, action = self.opt(gameState,
            prune = True,
        )

        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def opt(self, gameState: GameState,
        agentIndex: int | None = None,
        depth: int | None = None,
    ) -> tuple[float, Actions | None]:
        """Helper function replacing min and max helper functions usually found in minimax algorithms.

        I believe it is simpler to check `agentIndex` and act accordingly instead of creating two functions.
        Here we do not have a one-to-one game, we have one-to-many, so we get a max-min-min-...-max-min-min-...-gameover.

        This is made to expand the game tree at fixed depth, given as an attribute in the class.

        Finally, this is made a local function for better recursion.
        That way I can use argument default values as syntactic sugar.

        This variant of the method calculates expectation values instead of min values for all non-pacman agents.
        """
        if agentIndex is None: agentIndex = self.index
        if depth is None: depth = self.depth

        if not depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # Cycle to the next agent first:
        agentIndexNext = agentIndex + 1

        # Reset agent index on full cycle:
        if agentIndexNext == gameState.getNumAgents():
            agentIndexNext = self.index  # cycle back to pacman
            depth -= 1  # and write out one depth level

        # Initialize the best action to None:
        best_value = -math.inf if agentIndex == self.index else +math.inf  # either max or min logic depending on who plays
        best_action = None

        # Collect all values and actions if a ghost:
        all_values = []
        all_actions = []

        # Get actions and results for the enxt agent (presumably a ghost):
        for action in gameState.getLegalActions(agentIndex):
            state = gameState.generateSuccessor(agentIndex, action)  # query succerssors one by one
            value, _ = self.opt(state, agentIndexNext, depth)  # evaluate successor and get the value of it

            # Track the best value together with the action it corresponds to:
            if agentIndex == self.index:
                if best_value < value:
                    best_value = value  # keep track of the optimum value depending on who plays
                    best_action = action  # keep track of the corresponding action depending on who plays

            else:
                all_values.append(value)
                all_actions.append(action)

        # If a ghost, the best value is the expectation of all values and the best action is a random one:
        if agentIndex != self.index:
            best_value = mean(all_values)
            best_action = random.choice(all_actions)

        # Return both optimum plus the action it corresponds to:
        return best_value, best_action

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        _, action = self.opt(gameState)

        return action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:

    We start with a score of 0, but that does not mean it cannot be negative.

    If a wall is hit for a move, we drop the score all the way, walls should always be avoided.

    For every ghost out there, we see how far it is and if it is scared.
    -   If it is scared, we should approach it.
    -   Otherwise move away from the nearest one.

    Finally be attracted by the nearest food
    """
    position = currentGameState.getPacmanPosition()
    ghostStates: list[AgentState] = currentGameState.getGhostStates()

    # Initial value:
    score = 0

    # Walls should always be avoided:
    if currentGameState.hasWall(*position):
        score -= math.inf

    # For each ghost:
    for ghostState in ghostStates:
        ghostPosition = ghostState.getPosition()
        movesAway = manhattanDistance(position, ghostPosition)

        # Eat ghost:
        if movesAway <= ghostState.scaredTimer:
            score += movesAway

        else:
            score -= movesAway

    score -= currentGameState.getNumFood()

    return score


# Abbreviation
better = betterEvaluationFunction
