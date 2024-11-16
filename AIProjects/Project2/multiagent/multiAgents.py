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
        def opt(gameState: GameState,
            agentIndex: int = self.index,
            depth: int = self.depth,
        ) -> tuple[float, Actions | None]:
            """Helper function replacing min and max helper functions usually found in minimax algorithms.

            I believe it is simpler to check `agentIndex` and act accordingly instead of creating two functions.
            Here we do not have a one-to-one game, we have one-to-many, so we get a max-min-min-...-max-min-min-...-gameover.

            This is made to expand the game tree at fixed depth, given as an attribute in the class.

            Finally, this is made a local function for better recursion.
            That way I can use argument default values as syntactic sugar.
            """
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
                value, _ = opt(state, agentIndexNext, depth)  # evaluate successor and get the value of it

                # Track the best value together with the action it corresponds to:
                if (best_value < value and agentIndex == self.index) \
                or (best_value > value and agentIndex != self.index):
                    best_value = value  # keep track of the optimum value depending on who plays
                    best_action = action  # keep track of the corresponding action depending on who plays

            # Return both optimum plus the action it corresponds to:
            return best_value, best_action

        _, action = opt(gameState)

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def better(
            x: float,
            y: float,
            z: float, agentIndex: int = self.index
        ) -> bool:
            return x < z and agentIndex == self.index \
                or y > z and agentIndex != self.index

        def opt(gameState: GameState, agentIndex: int = self.index, depth: int = self.depth,
            a: float = -math.inf,
            b: float = +math.inf,
        ) -> tuple[float, Actions | None]:
            """Helper function replacing min and max helper functions usually found in minimax algorithms.

            I believe it is simpler to check `agentIndex` and act accordingly instead of creating two functions.
            Here we do not have a one-to-one game, we have one-to-many, so we get a max-min-min-...-max-min-min-...-gameover.

            This is made to expand the game tree at fixed depth, given as an attribute in the class.

            Finally, this is made a local function for better recursion.
            That way I can use argument default values as syntactic sugar.

            This implementation has extra alpha and beta parameters for pruning.
            """
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
                value, _ = opt(state, agentIndexNext, depth,
                    a,
                    b,
                )  # evaluate successor and get the value of it

                # Track the best value together with the action it corresponds to:
                if better(
                    best_value,
                    best_value, value, agentIndex
                ):
                    best_value = value  # keep track of the optimum value depending on who plays
                    best_action = action  # keep track of the corresponding action depending on who plays

                # If pruning is necessary cut the loop here:
                if better(
                    b,
                    a, value, agentIndex
                ):
                    break

                # Update pruning bounds otherwise:
                a = max(a, best_value) if agentIndex == self.index else a
                b = min(b, best_value) if agentIndex != self.index else b

            # Return both optimum plus the action it corresponds to:
            return best_value, best_action

        _, action = opt(gameState)

        return action


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
