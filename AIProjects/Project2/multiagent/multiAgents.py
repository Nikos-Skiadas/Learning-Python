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

from game import Agent
from pacman import GameState

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

        score = float(0)
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

            # Add 1/minimum distance to nearest food:
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

    @classmethod
    def max(cls, results: typing.Iterable[float]) -> float:
        value = -math.inf

        for result in results:
            if value < result:
                value = result

        return value

    @classmethod
    def min(cls, results: typing.Iterable[float]) -> float:
        value = +math.inf

        for result in results:
            if value > result:
                value = result

        return value

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
        """
        # Set default values for optional parameters:
        agentIndex = agentIndex if agentIndex is not None else self.index
        depth = depth if depth is not None else self.depth

        # Terminate on leaf nodes:
        if not depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # Cycle to the next agent first:
        agentIndexNext = agentIndex + 1

        # Reset agent index on full cycle:
        if agentIndexNext == gameState.getNumAgents():
            agentIndexNext = self.index
            depth -= 1

        # Get actions and results for the enxt agent (presumably a ghost):
        actions = gameState.getLegalActions(agentIndex)
        states = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        results = [self.opt(state, agentIndexNext, depth)[0] for state in states]

        # get corresponding max or min depending on whose turn it is:
        # Get max for pacman, min for everyone (ghosts) else:
        opt_result = self.max(results) if agentIndex == self.index else self.min(results)

        # Return both optimum plus the action it corresponds to:
        return opt_result, actions[results.index(opt_result)]

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction: typing.Callable[[GameState], float] = util.lookup(evalFn, globals())
        self.depth = int(depth)

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
        return self.opt(gameState)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def opt(gameState: GameState, alpha: float = -math.inf, beta: float = math.inf,
            agentIndex: int = self.index,
            depth: int = self.depth,
        ) -> tuple[float, Actions | None]:
            """Helper function replacing min and max helper functions usually found in minimax algorithms.

            I believe it is simpler to check `agentIndex` and act accordingly instead of creating two functions.
            Here we do not have a one-to-one game, we have one-to-many, so we get a max-min-min-...-max-min-min-...-gameover.

            This is made to expand the game tree at fixed depth, given as an attribute in the class.

            Finally, this is made a local function for better recursion.
            That way I can use argument default values as syntactic sugar.

            This implementation additionaly has an `alpha` and `beta` parameter to perform alpha-beta-pruning.
            """
            if not depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None

            # Cycle to the next agent first:
            agentIndexNext = agentIndex + 1

            # Reset agent index on full cycle:
            if agentIndexNext == gameState.getNumAgents():
                agentIndexNext = self.index
                depth -= 1

            # Get actions and results for the enxt agent (presumably a ghost):
            actions = gameState.getLegalActions(agentIndex)
            states = [gameState.generateSuccessor(agentIndex, action) for action in actions]
            results = [opt(state, agentIndexNext, depth)[0] for state in states]

            # get corresponding max or min depending on whose turn it is:
            # Get max for pacman, min for everyone (ghosts) else:
            opt_result = max(results) if agentIndex == self.index else min(results)

            # Return both optimum plus the action it corresponds to:
            return opt_result, actions[results.index(opt_result)]

        return opt(gameState)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
