# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
from layout import Layout

from game import Agent

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
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"

    if successorGameState.isWin():
      return float("inf") - 20

    pos=currentGameState.getPacmanPosition()
    ghostPos=currentGameState.getGhostPosition(1)
    ghostDistance=util.manhattanDistance(ghostPos,pos)
    
    foodList=newFood.asList(key=True)
    closestfooddistance=10000000
    for i in foodList:
      if util.manhattanDistance(i,newPos)<closestfooddistance:
        closestfooddistance=util.manhattanDistance(i,newPos)

    powerBalls=currentGameState.getCapsules()
    extraPoints=0
    if newPos in powerBalls:
      extraPoints=100
 
    score=successorGameState.getScore()+ghostDistance+1/closestfooddistance+extraPoints

    if currentGameState.getNumFood()>successorGameState.getNumFood():
      score+=100

    if action==Directions.STOP:
      score-=100

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

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth='3'):
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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    
    ghostNum=gameState.getNumAgents()-1
    

    def minPlay(gameState,depth,agentIndex,ghostNum,evalFunc):
      
      if gameState.isWin() or gameState.isLose() or depth==0:
        #print "self.evaluationFunction(gameState) in minPlay="+str(self.evaluationFunction(gameState))
        return evalFunc(gameState)

      legalMoves=gameState.getLegalActions(agentIndex)
      if Directions.STOP in legalMoves:
        legalMoves.remove(Directions.STOP)
      minValue=100000000
      for move in legalMoves:
        if agentIndex==ghostNum:
          minValue=min(maxPlay(gameState.generateSuccessor(agentIndex,move),depth-1,ghostNum,evalFunc),minValue)
        
        else:
          minValue=min(minPlay(gameState.generateSuccessor(agentIndex,move),depth,agentIndex+1,ghostNum,evalFunc),minValue)
  
      #print "minValue="+str(minValue)
      return minValue
    
    def maxPlay(gameState, depth, ghostNum,evalFunc):
      if gameState.isWin() or gameState.isLose() or depth==0:
        #print "gamsState="+str(gameState)
        #print  "self.evaluationFunction(gameState) in maxPlay="+str(self.evaluationFunction(gameState))
        return evalFunc(gameState)
      legalMoves=gameState.getLegalActions(0)
      if Directions.STOP in legalMoves:
        legalMoves.remove(Directions.STOP)
      maxValue=-100000000
      for move in legalMoves:
        maxValue=max(minPlay(gameState.generateSuccessor(0,move),depth,1,ghostNum,evalFunc),maxValue)
        
      #print "maxValue="+str(maxValue)
      return maxValue
 
    legalMoves=gameState.getLegalActions()
    if Directions.STOP in legalMoves:
        legalMoves.remove(Directions.STOP)
    bestMove=Directions.RIGHT
    bestScore=-1000000000
    for move in legalMoves:
      Successor=gameState.generateSuccessor(0,move)
      newScore=minPlay(Successor,self.depth,1,ghostNum,self.evaluationFunction)
      if newScore>bestScore:
        bestMove=move
        bestScore=newScore
    print "bestScore="+str(bestScore)
    return bestMove
    util.raiseNotDefined()
      
    
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    ghostNum=gameState.getNumAgents()-1
    

    def minPlay(gameState,depth,agentIndex,ghostNum,evalFunc,alpha,beta):
      
      if gameState.isWin() or gameState.isLose() or depth==0:
        #print "self.evaluationFunction(gameState) in minPlay="+str(self.evaluationFunction(gameState))
        return evalFunc(gameState)

      legalMoves=gameState.getLegalActions(agentIndex)
      if Directions.STOP in legalMoves:
        legalMoves.remove(Directions.STOP)
      minValue=100000000
      for move in legalMoves:
        if agentIndex==ghostNum:
          minValue=min(maxPlay(gameState.generateSuccessor(agentIndex,move),depth-1,ghostNum,evalFunc,alpha,beta),minValue)
          if minValue<=alpha:
            return minValue
          beta=min(beta,minValue)
        else:
          minValue=min(minPlay(gameState.generateSuccessor(agentIndex,move),depth,agentIndex+1,ghostNum,evalFunc,alpha,beta),minValue)
          if minValue<=alpha:
            return minValue
          beta=min(beta,minValue)
      #print "minValue="+str(minValue)
      return minValue
    
    def maxPlay(gameState, depth, ghostNum,evalFunc,alpha,beta):
      if gameState.isWin() or gameState.isLose() or depth==0:
        #print "gamsState="+str(gameState)
        #print  "self.evaluationFunction(gameState) in maxPlay="+str(self.evaluationFunction(gameState))
        return evalFunc(gameState)
      legalMoves=gameState.getLegalActions(0)
      if Directions.STOP in legalMoves:
        legalMoves.remove(Directions.STOP)
      maxValue=-100000000
      for move in legalMoves:
        maxValue=max(minPlay(gameState.generateSuccessor(0,move),depth,1,ghostNum,evalFunc,alpha,beta),maxValue)
        if maxValue>=beta:
          return maxValue
        alpha=max(alpha,maxValue)
      #print "maxValue="+str(maxValue)
      return maxValue
 
    legalMoves=gameState.getLegalActions()
    if Directions.STOP in legalMoves:
        legalMoves.remove(Directions.STOP)
    bestMove=Directions.RIGHT
    bestScore=-1000000000
    alpha=-100000000
    beta=1000000000
    for move in legalMoves:
      Successor=gameState.generateSuccessor(0,move)
      newScore=minPlay(Successor,self.depth,1,ghostNum,self.evaluationFunction,alpha,beta)
      if newScore>bestScore:
        bestMove=move
        bestScore=newScore
      if bestScore>=beta:
        return bestMove
      alpha=max(alpha,bestScore)
    print "bestScore="+str(bestScore)
    return bestMove
    util.raiseNotDefined()
           

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
    ghostNum=gameState.getNumAgents()-1

    def expectationPlay(gameState,depth,agentIndex,numghosts=ghostNum):
      if gameState.isWin() or gameState.isLose() or depth==0:
        return self.evaluationFunction(gameState)
      legalMoves=gameState.getLegalActions(agentIndex)
      if Directions.STOP in legalMoves:
        legalMoves.remove(Directions.STOP)
      expectValue=0
      if agentIndex==numghosts:
        for move in legalMoves:
          expectValue+=maxPlay(gameState.generateSuccessor(agentIndex,move),depth-1,numghosts)
      else:
        for move in legalMoves:
          expectValue+=expectationPlay(gameState.generateSuccessor(agentIndex,move),depth,agentIndex+1,numghosts)   
      return expectValue/len(legalMoves)
      
    def maxPlay(gameState,depth,agentIndex=0,numghosts=ghostNum):
      if gameState.isWin() or gameState.isLose() or depth==0:
        return self.evaluationFunction(gameState)
      legalMoves=gameState.getLegalActions()
      if Directions.STOP in legalMoves:
        legalMoves.remove(Directions.STOP)
      maxValue=-100000000
      for move in legalMoves:
        print "legalmove="+str(move)
        maxValue=max(expectationPlay(gameState.generateSuccessor(0,move),depth,1),maxValue)
      return maxValue

   
    legalMoves=gameState.getLegalActions()
    if Directions.STOP in legalMoves:
      legalMoves.remove(Directions.STOP)
    bestMove=Directions.RIGHT
    bestScore=-10000000
    for move in legalMoves:
      Successor=gameState.generateSuccessor(0,move)
      newScore=expectationPlay(Successor,self.depth,1)
      if newScore>bestScore:
        bestMove=move
        bestScore=newScore
    return bestMove
    #print "bestScore="+str(bestScore)
    #return bestMove
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    My evaluation function return score which is better if higher.
    My score consists of 5 parts:
    1. Evaluation score of current state, I call scoreEvaluationFunction here. Higher evaluation scores, higher scores.
    2. The closest ghost distance, which is the closest manhattan distance between current pacman position and all ghost positions. 
       When closest distance under 5, higher score if higher distance. When closest distance above 5, add a constant to the score, 
       which means that ghost distance weigh less in Pacman's consideration when closest ghost distance is above 5.
    3. The closest distance between Pacman and all foods. Here score is higher is distance is less.
    4. A coefficient related to the number of foods, smaller the number of food is, higher the score is.
    5. A coefficient related to the capsule positions. More capusles left, lower the score is.

  """
  "*** YOUR CODE HERE ***"
  state=currentGameState
  score=scoreEvaluationFunction(state)
  if state.isWin():
    score+=float("inf")
  if state.isLose():
    score-=float("-inf")
  
  ghostNum=state.getNumFood()-1
  ghostDis=10000000
  for i in range(ghostNum):
    if util.manhattanDistance(state.getPacmanPosition(), state.getGhostPosition(i+1))<ghostDis:
      ghostDis=util.manhattanDistance(state.getPacmanPosition(), state.getGhostPosition(i+1))
  score+=max(ghostDis,5)*3

  food=state.getFood()
  foodList=food.asList()
  foodDis=10000000
  for i in foodList:
    if foodDis>util.manhattanDistance(i,state.getPacmanPosition()):
      foodDis=util.manhattanDistance(i,state.getPacmanPosition())
  score-=foodDis*3
  
  score-=3*len(foodList)
  
  capsules=state.getCapsules()
  score-=3*len(capsules)
  
  return score
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

