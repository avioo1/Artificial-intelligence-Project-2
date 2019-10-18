# myTeam.py
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

import distanceCalculator
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################
print("Running myteam.py")



# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).




class ValueIterationAgent:
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        #print self.values
        state = self.mdp.getStates()[2]
        print("State is", state)
        #print mdp.getPossibleActions(state)
        print("posible actions are", mdp.getPossibleActions(state))
        nextState = mdp.getTransitionStatesAndProbs(state, mdp.getPossibleActions(state)[0])
        #print nextState
        #print "printed next state"
        #print mdp.getReward(state, mdp.getPossibleActions(state)[0] ,nextState)

        states = self.mdp.getStates()

        #print self.mdp.getStartState()

        for i in range(iterations):
          valuesCopy = self.values.copy()
          for state in states:
            finalValue = None
            for action in self.mdp.getPossibleActions(state):
              currentValue = self.computeQValueFromValues(state,action)
              if finalValue == None or finalValue < currentValue:
                finalValue = currentValue
            if finalValue == None:
              finalValue = 0
            valuesCopy[state] = finalValue
          
          self.values = valuesCopy



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
      """
        Compute the Q-value of action in state from the
        value function stored in self.values.
      """
      "*** YOUR CODE HERE ***"
      value = 0
      transitionFunction = self.mdp.getTransitionStatesAndProbs(state,action)
      for nextState, probability in transitionFunction:
        #print("next state and probability are", nextState, probability)
        value += probability * (self.mdp.getReward(state, action, nextState) 
                  + (self.discount * self.values[nextState]))

      return value
          
    def computeActionFromValues(self, state):
      """
        The policy is the best action in the given state
        according to the values currently stored in self.values.
        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
      """
      "*** YOUR CODE HERE ***"
      possibleActions = self.mdp.getPossibleActions(state)
     
      if len(possibleActions) == 0:
        return None

      value = None
      result = None
      for action in possibleActions:
        temp = self.computeQValueFromValues(state, action)
        if value == None or temp > value:
          value = temp
          result = action

      return result
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class IMDP:


  def __init__(self, initial_state, states):
    self.initial_state = initial_state
    self.states = states
    possibleActions = util.Counter()
    self.rewards = util.Counter()
    print("states are ", states)
    for s in states:
      print(s)
      x,y = s
      if (x-1,y) in states:
        possibleActions[((x,y), Directions.WEST)] = (x-1,y)
      if (x + 1, y) in states:
        possibleActions[((x,y), Directions.EAST)] = (x + 1, y)
      if (x, y - 1) in states:
        possibleActions[((x,y), Directions.SOUTH)] = (x, y - 1)
      if (x, y + 1) in states:
        possibleActions[((x,y), Directions.NORTH)] = (x, y + 1)
    self.possibleActions = possibleActions



  def addReward(self, state, reward):
    self.rewards[state] = self.rewards[state] + reward

  def getStates(self):
    return list(self.states)

  def getPossibleActions(self, state):
    possible = []
    for i in self.possibleActions.keys():
      if i[0] == state:
        possible.append(i[1])

    return possible 
    
  def getReward(self, state, action, nextState):
    if action == Directions.STOP:
      return -10
    return min(self.rewards[state], self.rewards[nextState])


  def isTerminal(self):
    return False 

  def getTransitionStatesAndProbs(self, state, actions):
    action = self.possibleActions[(state, actions)]
    probability = 1
    return [(self.possibleActions[(state, actions)], 1), ]


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def gameGrid(self, initial_state, gameState, length_search_square):
    search_grid = []
    game_length, game_breadth = gameState.getWalls().asList()[-1]
    #print("position_x and position_y are", initial_state)
    position_x, position_y = initial_state
    search_x_start = max(1, position_x-length_search_square)
    search_x_end = min(game_length, position_x+length_search_square)
    search_y_start = max(1, position_y-length_search_square)
    search_y_end = min(game_breadth, position_y+length_search_square)
    #print(search_x_start, search_x_end, search_y_start, search_y_end)
    for x in range(search_x_start, search_y_end):
      for y in range(search_y_start, search_y_end):
        if (x,y) not in gameState.getWalls().asList():
          search_grid.append((x,y))
        #else:
        #  print("walls were",(x,y))

    return search_grid

  def values(self, initial_state, gameState, length_search_square):

    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()
    print("self.distancer is", self.distancer)
    states = self.gameGrid(initial_state, gameState, length_search_square)
    states = {cell for cell in states if abs(myPos[0]+cell[0]) + abs(myPos[1]+cell[1]) <= 7}
    print("states are", states)
    print(states)
    mdp = IMDP(initial_state, states)
    
    for i in states:
      if i == (1,8):
        mdp.addReward(i,1)
      if i in gameState.getRedFood().asList():
        print("reward added")
        mdp.addReward(i,1)
      else:
        mdp.addReward(i,-0.04)
    mdp.addReward((1,5),0.5)
    print("reward for 1,8 is", mdp.getReward((1,8), 'North', (1,8)))
    print("rewards are", mdp.rewards)
    values = ValueIterationAgent(mdp).values
    return values 
  
  def chooseAction(self, gameState, cc=0):
    s = OffensiveReflexAgent(self.index)
    agent_position = gameState.getAgentPosition(self.index)
    Qvalues = s.values(agent_position,gameState, 10)
    actions = gameState.getLegalActions(self.index)
    imdp = IMDP(agent_position, self.gameGrid(agent_position,gameState, 10))
    possibleActions = imdp.getPossibleActions(agent_position)
    #possible_states = [gameState.getSuccessor(gameState,a) for a in actions]
    #print("possible states are" ,possible_states)
    final = {} 
    #print("possible actions are", possibleActions)
    x,y = agent_position
    #print("possible aactions are", possibleActions)
    print("agent position is", agent_position)
    print("Qvalues are", Qvalues)
    for i in possibleActions:
      print("i is", i)
      if i == Directions.NORTH:

        final[Directions.NORTH] = Qvalues[(x,y+1)]
      if i is Directions.SOUTH:
        final[Directions.SOUTH] = Qvalues[(x,y-1)]
      if i is Directions.EAST:
        final[Directions.EAST] = Qvalues[(x+1,y)]
      if i is Directions.WEST:
        final[Directions.WEST] = Qvalues[(x-1,y)]
    print("FINAL IS", final)
    temp = max(final.values()) 
    res = [key for key in final if final[key] == temp] 
    print("returned value is", res)
    return res[0]


  

  '''
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction
    
    return random.choice(bestActions)
  '''
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    print("nearest point pos is ", nearestPoint(pos))
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    
  

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()   
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
gamma = 0.9 


class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    print("registerInitialState is ", CaptureAgent.registerInitialState(self, gameState))

    '''
    Your initialization code goes here, if you need any.
    '''

  
  
  def chooseAction(self, gameState):
    
    """
    Picks among actions randomly.
    
    actions = gameState.getLegalActions(self.index)
    goal = CaptureAgent.getFood(self,gameState).asList()
    print("goal is", goal)
    agent_index = CaptureAgent.getTeam(self,gameState)[1]
    print("agent index is", agent_index)
    print("locations is", gameState.getAgentPosition(agent_index))
    priority_wastar = util.PriorityQueue()
    priority_wastar.push(())

    """

    '''
    You should change this in your own agent.
    '''
    stack = util.Stack()
    goal = (CaptureAgent.getFood(self,gameState).asList())
    s = OffensiveReflexAgent(self.index)
    #print("search grid is ", s.gameGrid(gameState.getAgentPosition(self.index),gameState, 5))
    #print("values are", s.values(gameState.getAgentPosition(self.index),gameState, 5))
    print("actions are",  s.chooseAction(gameState))
    nodes_visited = []
    final_goal = (len(goal)== 0)
    #print("the exploring grid is", self.value_iteration(gameState))
    stack.push((gameState.getAgentPosition(self.index),[]))
    #print("walls are", gameState.getWalls().asList())
    #print("agent location is ", gameState.getAgentPosition(self.index))
    #print("successors are", gameState.generateSuccessor(self.index, 'North').getAgentPosition(self.index))
   
    #print("agent distance is ", gameState.getAgentDistances())
    #print("goal is", goal)
    #xprint("game dimensions are", gameState.getWalls().asList()[-1])
    actions = gameState.getLegalActions(self.index)
    print("actions are", actions)
    return random.choice(actions)






'''
class MonteCarloTreeSearch():

  def __init__(self):
    self.visited_count = defaultdict(int)
    self.children = dict()
    self.reward = defaultdict(int)

  def _select(self, node):
    if node not in self.children:
      return node.find_random_child()

    def score(n):
      if node.get_visited_count() == 0:
        return float("-inf")
      return node.reward()/node.get_visited_count()

    return max(self.children[node], key=score)

  def rollout(self, node):
    path = self._select(node)
    leaf = path[-1]
    self._expand(leaf)
    reward = self._simulate(leaf)
    self._backpropagate(path, reward)

  def _select(self, node):
    path = []
    while True:
      path.append(node)
      if node not in self.children or not self.children[node]:
        return path
      unexplored = self.children[node] - self.children.keys()
    if unexplored:
      n = unexplored.pop()
      path.append(n)
      return path
    node = self._uct_select(node)

    def _expand(self, node):
      if node in self.children:
        return
      self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct) 


class Node():

  def __init__(self, gameState, agent_index) : 
    self.position = gameState.getAgentPosition(self.index)
    self.index = agent_index
    self.visited_count = 0
    self.V = 0
    
  def get_visited_count(self):
    return self.visited_count
  
  def find_children(self):
    return gameState.generateSuccessor

  def updateVisitedCount(self):
    self.visited_count = self.visited_count + 1
    return self.visited_count

  def find_random_child(self):
    return random.choose(gameState.getLegalActions(agent_index))

  def reward(self):
    if self.position in CaptureAgent.getFood(self,gameState).asList():
      return 1
    return -0.04

  def is_terminal(self):
    return True 
  
  
  def grid(gameState, index, grid_length):
      grid = Grid(2*grid_length, 2*grid_length, initial_value=False)
      grid_x, grid_y = gameState.getWalls().asList()
      initial_x, initial_y = gameState.getAgentPosition(self.index)
      final_x = min(grid_x, initial_x+grid_length)
      starting_x = max(1,grid_x-grid_length)
      final_y = min(grid_y, initial_y+grid_length)
      starting_y = max(1,grid_y-grid_length)

      return 
  

  
class MDP:

    
    def __init__(self, gameState, agent_index, range, captureAgent):
      self.index = agent_index
      self.position = gameState.getAgentPosition(self.index)
      self.food = captureAgent.getFood(self, gameState)[]
    

    def __init__(self, gameState, agent_index):
      location_x, location_y = gameState.getAgentPosition(agent_index)
      game_grid_x, game_grid_y = gameState.getWalls().asList()[-1]
      grid_x_start = max(1,location_x-10)
      grid_x_final = min(location_x+10, game_grid_x)
      grid_y_start = max(1,location_y-10)
      grid_y_final = min(location_y+10, game_grid_y)
      #self.exploring_grid = [[x for x in range(self.grid_x_start, self.grid_x_final)][y for y in range(grid_y_start, grid_y_final)]]
      self.food = gameState.getRedFood()[grid_x_start:grid_x_final][grid_y_start:grid_y_final]
     #self.reward = []
      #for i in self.food:
       # if i == False:
        #  self.reward = {i:-0.04}
        #else:
         # self.reward = {i:1}

    

    def __init__(self, reward={}, gamma=0.9):
      self.reward = reward
      self.gamma = gamma

    def actions(self, state):


  '''
