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
from game import Directions
import random, util

from game import Agent

PACMAN_INDEX = 0

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print "The score for successor state is: " + str(successorGameState.getScore())
        newPos = successorGameState.getPacmanPosition()
        #print newPos
        newFood = successorGameState.getFood()
        #print "New food are:\n"
        #print newFood
        #print "new food as list\n"
        #print newFood.asList()
        newFoodCoordinates = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        #print "New ghostStates are:\n"
        #print newGhostStates
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print "New ScaredTimes are:\n"
        #print newScaredTimes
        score = successorGameState.getScore()
        distanceToFood = [manhattanDistance(newPos, xy2) for xy2 in newFoodCoordinates]
        xy2 = newGhostStates[0].getPosition()
        distanceToGhost = manhattanDistance(newPos, xy2) 

        if len(distanceToFood) :
          #reciprocal = [float(1/x) for x in distanceToFood]
          #score += max(reciprocal) * 10
          score += 10.0 / min(distanceToFood)

        if distanceToGhost > 0:
          #score -=float(1/distanceToGhost)*10
          score -= 50.0 / distanceToGhost

        "*** YOUR CODE HERE ***"
        return score

def scoreEvaluationFunction(currentGameState):
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
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        "*** YOUR CODE HERE ***"
        """Have defined two helper functions called maximizer and minimizer"""
        return self.maximizer(gameState, 1)

    def maximizer(self, gameState, depth):
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

        alpha = float("-infinity")
        pacmanAction = Directions.STOP
        pacmanLegalActions = gameState.getLegalActions(PACMAN_INDEX)
        evals = []
        counter = 0 
        for action in pacmanLegalActions:
          
          successor = gameState.generateSuccessor(PACMAN_INDEX, action)
          evals += [self.minimizer(successor, depth, 1)]
          if evals[counter] > alpha:
            alpha = evals[counter]
            pacmanAction = action
          counter+=1

        if depth > 1:
          return max(evals)
        return pacmanAction

    def minimizer(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

        beta = float("infinity")
        agentAction = Directions.STOP
        agentLegalActions = gameState.getLegalActions(agentIndex)

        successorGameStates = []
        for action in agentLegalActions:
          successorGameStates+= [gameState.generateSuccessor(agentIndex, action)]

        evals = []
        for successor in successorGameStates:
          if agentIndex == gameState.getNumAgents() - 1:
            if depth < self.depth:
              evals += [self.maximizer(successor, depth + 1)]
            else:
              evals += [self.evaluationFunction(successor)]
          else:
            evals += [self.minimizer(successor, depth, agentIndex + 1)]
        return min(evals)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-infinity")
        beta = float("infinity")
        return self.maximizer(gameState, 1, alpha, beta)

        #util.raiseNotDefined()

    def maximizer(self, gameState, depth, alpha, beta):
      """this function was a pain to implement srsly"""
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      currentScore = float("-infinity")
      pacmanAction = Directions.STOP
      pacmanLegalActions = gameState.getLegalActions(PACMAN_INDEX)
      # successorGameStates = []
      #------cut-------
      for action in pacmanLegalActions:
        successor = gameState.generateSuccessor(PACMAN_INDEX, action)
        prune = self.minimizer(successor, depth, 1, alpha, beta)
        if prune > currentScore:
          currentScore = prune
          pacmanAction = action

        #You must not prune on equality in order 
        #to match the set of states explored by our autograder
        if currentScore > beta:
          return currentScore

        alpha = max(alpha, currentScore)

      if depth > 1:
        return currentScore
      return pacmanAction

    def minimizer(self, gameState, depth, agentIndex, alpha, beta):
      """this function was a pain to implement srsly"""
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      currentScore = float("infinity")
      agentAction = Directions.STOP
      agentLegalActions = gameState.getLegalActions(agentIndex)

      #successorGameStates = []
      for action in agentLegalActions:
        successor = gameState.generateSuccessor(agentIndex, action)
      #for successor in successorGameStates:
        if (agentIndex == gameState.getNumAgents() - 1):
          if (depth < self.depth):
            prune = self.maximizer(successor, depth + 1, alpha, beta)
          else:
            prune = self.evaluationFunction(successor)
        else:
          prune = self.minimizer(successor, depth, agentIndex + 1, alpha, beta)
        if prune < currentScore:
          currentScore = prune

        #You must not prune on equality in order 
        #to match the set of states explored by our autograder
        if currentScore < alpha:
          return currentScore
        beta = min(beta, currentScore)
      return currentScore

    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return self.maximizer(gameState, 1)
    def maximizer(self, gameState, depth):
      """This is the exact same function as the Minimax maximizer, since the maximizing player pacman,
      has no changes in this algorithm. He always chooses optimal paths."""
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      alpha = float("-infinity")
      pacmanAction = Directions.STOP
      pacmanLegalActions = gameState.getLegalActions(PACMAN_INDEX)
      evals = []
      counter = 0 
      for action in pacmanLegalActions:
          
        successor = gameState.generateSuccessor(PACMAN_INDEX, action)
        evals += [self.minimizer(successor, depth, 1)]
        if evals[counter] > alpha:
          alpha = evals[counter]
          pacmanAction = action
        counter+=1

      if depth > 1:
        return max(evals)
      return pacmanAction

    def minimizer(self, gameState, depth, agentIndex):
      """Similar function as minimax minimizer, except that now, instead of calculating min,
      we have to calculate expectation based on equal probability."""
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      beta = float("infinity")
      agentAction = Directions.STOP
      agentLegalActions = gameState.getLegalActions(agentIndex)

      successorGameStates = []
      for action in agentLegalActions:
        successorGameStates+= [gameState.generateSuccessor(agentIndex, action)]

      expectation = 0.0
      chance = float(len(agentLegalActions))

      for successor in successorGameStates:
        if agentIndex == gameState.getNumAgents() - 1:
          if depth < self.depth:
            #beta = min(beta, self.maximizer(successor, depth + 1))
            expectation += self.maximizer(successor, depth + 1)/chance
          else:
            expectation += self.evaluationFunction(successor)/chance
        else:
          expectation += self.minimizer(successor, depth, agentIndex + 1)/chance
      return expectation

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: I am going to model this on a very similar approach as the
      evaluation function in reflexAgent, except now the successorState is the currentState
      and we also have scared ghosts.
      Through Trial and error, I found that we need to impart maximum importance to 
      dealing with scared ghosts, then be aware of nonscared ghosts, and give least 
      importance to food pellets. The values 150, 50 and 10 are a result of pure luck lol,
      since I tried a bunch of different options, but this one gives the highest average score.
    """
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    #print "The score for successor state is: " + str(successorGameState.getScore())
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newFoodCoordinates = newFood.asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()

    #----------This part accounts for food----------------
    distanceToFood = [manhattanDistance(newPos, xy2) for xy2 in newFoodCoordinates]

    if len(distanceToFood) :
      #reciprocal = [float(1/x) for x in distanceToFood]
      #score += max(reciprocal) * 10
      score += 10.0 / min(distanceToFood)

    #----------This part accounts for ghosts---------------

    xy2 = newGhostStates[0].getPosition()
    distanceToGhost = manhattanDistance(newPos, xy2) 
    if distanceToGhost > 0:
      for ghost in newGhostStates:
        if ghost.scaredTimer > 0:
          score += 150.0/distanceToGhost
        else:
          score -= 50.0/distanceToGhost



    # if distanceToGhost > 0:
    #   #score -=float(1/distanceToGhost)*10
    #   score -= 20.0 / distanceToGhost

    "*** YOUR CODE HERE ***"
    return score

# Abbreviation
better = betterEvaluationFunction

