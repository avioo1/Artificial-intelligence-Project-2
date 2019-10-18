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


from game import Actions, Agent
from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
from game import Directions
import game

from distanceCalculator import Distancer 

import graphicsDisplay as gd







def createTeam(firstIndex, secondIndex, isRed,
               first = 'AstarDefensiveAgent', second = 'ReflexCaptureAgent'):
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




class DummyAgent(CaptureAgent):
    def chooseAction(self, gameState):
        return Directions.STOP   



class ValueIteration:
    """
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.7, iterations=15):
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
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        for i in range(0, iterations):
            iteration_values = util.Counter()
            for state in mdp.getStates():
                best_action = self.computeActionFromValues(state)
                if best_action:
                    iteration_values[state] = round(self.computeQValueFromValues(state, best_action),3)

            self.values = iteration_values.copy()

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
        # print(state, action)
        qvalue = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            #print(state, action)
            reward = self.mdp.getReward(state, action, next_state)
            # print prob, reward, self.discount
            qvalue += prob * (reward + self.discount * self.values[next_state])
        return qvalue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
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

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class MDP:

    def __init__(self, initial_state, states):
        self.states = states
        self.start = initial_state
        self.rewards = util.Counter()
        possibleActions = util.Counter()
        for state in states:
            x, y = state
            # legalMoves[(state, Directions.STOP)] = state
            if (x - 1, y) in states:
                possibleActions[(state, Directions.WEST)] = (x - 1, y)
            if (x + 1, y) in states:
                possibleActions[(state, Directions.EAST)] = (x + 1, y)
            if (x, y - 1) in states:
                possibleActions[(state, Directions.SOUTH)] = (x, y - 1)
            if (x, y + 1) in states:
                possibleActions[(state, Directions.NORTH)] = (x, y + 1)

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

    def getTransitionStatesAndProbs(self, state, action):
        return [(self.possibleActions[(state, action)], 1), ]

    def isTerminal(self):
        return False 

    def getReward(self, state, action, nextState):
        if action == Directions.STOP:
            return -10
        return self.rewards[state]

class ReflexCaptureAgent(CaptureAgent):


    def makeMap(self, initial_state, gameState, length_search_square, all_stats = False):

        search_grid = []
        game_length, game_breadth = gameState.getWalls().asList()[-1]
        #print("position_x and position_y are", initial_state)
        position_x, position_y = initial_state
        search_x_start = max(1, position_x-int(length_search_square))
        search_x_end = min(game_length, position_x+int(length_search_square))
        search_y_start = max(1, position_y-int(length_search_square))
        search_y_end = min(game_breadth, position_y+int(length_search_square))
        #print(search_x_start, search_x_end, search_y_start, search_y_end)
        x = list(range(search_x_start,search_y_end))
        y = list(range(search_y_start, search_y_end))
        

        for x in range(search_x_start, search_x_end):
            for y in range(search_y_start,search_y_end):
                #print("(x,y) is", x,y)
                if (x,y) not in gameState.getWalls().asList():
                    search_grid.append((x,y))
        #print("search grid is", search_grid)
        if all_stats == False:
            return search_grid
        else:
            return (search_x_start, search_x_end, search_y_start, search_y_end)
    

 
    def registerInitialState(self, gameState):

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.lefthomeever = False
        self.walls = set(gameState.data.layout.walls.asList())
        self.max_x = max([i[0] for i in self.walls])
        self.max_y = max([i[1] for i in self.walls])
        self.sign = 1 if gameState.isOnRedTeam(self.index) else -1
        self.length_of_grid = 7
        self.usingMDP=True

        # Determining home boundary
        self.goal_index= 0
        self.homeXBoundary = self.start[0] + ((self.max_x // 2 - 1) * self.sign)
        cells = [(self.homeXBoundary, y) for y in range(1, self.max_y)]
        self.homeBoundaryCells = [item for item in cells if item not in self.walls]
        self.states_visited = util.Counter()
    def values(self, initial_state, gameState, length_search_square):
        agentpos = gameState.getAgentPosition(self.index)
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        #print("self.distancer is", self.distancer)
        states = self.makeMap(initial_state, gameState, length_search_square)
        #states = {cell for cell in states if abs(myPos[0]+cell[0]) + abs(myPos[1]+cell[1]) <= 7}
        #print("states are", states)
        #print(states)
        mdp = MDP(initial_state, states)


    #def isHome(self, current_state):
    
    def legalSuccessors(self, gameState, cell):
        x,y = cell
        actions = []
        if (x+1,y) not in  gameState.getWalls().asList():
            actions.append(Directions.EAST)
        if (x-1,y) not in  gameState.getWalls().asList():
            actions.append(Directions.WEST)
        if (x,y+1) not in  gameState.getWalls().asList():
            actions.append(Directions.NORTH)
        if (x,y-1) not in  gameState.getWalls().asList():
            actions.append(Directions.SOUTH)
        return actions 
    def isHome(self, gameState):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        #if CaptureAgent.
        if not self.red:
            return myPos[0] < self.homeBoundaryCells[0][0]+1
        else:
            return myPos[0] > self.homeBoundaryCells[0][0]-1
    
    def distanceHome(self, current_position, gameState):
        x,y = current_position
        l = []
        if self.red and x < self.homeBoundaryCells[0][0] :
            return 0
        if not self.red and x > self.homeBoundaryCells[0][0]:
            return 0
        for i in self.homeBoundaryCells:
            l.append(abs(i[0] - x) + abs(i[1] - y))
        return min(l)


    def nearestFood(self, gameState, position):
        distances = {}
        for i in CaptureAgent.getFood(self,gameState).asList():
            distances[i] = self.distancer.getDistance(position, i)
        minimum = min(distances, key=distances.get)
        return minimum
    def nearestfoodtoeat(self, gameState,position):
        distances = {}
        for i in CaptureAgent.getFood(self,gameState).asList():
            distances[i] = self.distancer.getDistance(position, i)
        minimum = sorted(distances, key=distances.get)
        #print("distances are " ,minimum)
        return minimum

    def checkisenemynearby(self, gameState, current_postion, distance=False):
        enemy_idx = self.getOpponents(gameState)
        enemy_count = 0
        dis = []
        for i in enemy_idx:
            if gameState.getAgentPosition(i) is not None:
                enemy_count += 1
                dis.append(gameState.getAgentDistances()[i])
        if distance == False:
            return enemy_count > 0
        else:
            return(min(dis))

    def chooseAction(self, gameState):
    
        
        actions = gameState.getLegalActions(self.index)
        agentpos = gameState.getAgentPosition(self.index)
        #opponents_index = 
        
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        
        x, y = myPos
        
        #r =  self.chooseActionAStar(gameState)
        food_carrying = myState.numCarrying
        food_in_grid = 0
        foodlist = self.nearestfoodtoeat(gameState, myPos)
        if not self.isHome(gameState) and not self.lefthomeever: #and not self.checkisenemynearby(gameState,myPos):
            #f = list(foodlist[0])
            r = self.pathToGoalAStar(gameState, myPos, self.homeBoundaryCells)
            
            return r[0]
        
        foodleft = len(CaptureAgent.getFood(self, gameState).asList())
        
        if food_carrying > 9 or gameState.data.timeleft < 40 or foodleft < 3:
            r = self.pathToGoalAStar(gameState, myPos, self.homeBoundaryCells)
            return r[0]

        
        else:
            grid = self.makeMap(agentpos, gameState, self.length_of_grid)
            mdp = MDP(agentpos,grid)
            foodPositions = CaptureAgent.getFood(self, gameState).asList()
            foodLeft = len(foodPositions)
            def getKey(item):
                return item[0]
            sorted_grid = sorted(grid, key=getKey)
            if self.red:
                leftmost = sorted_grid[0][0]
            else:
               # print("sorted grid last emelent is", sorted_grid[-1], sorted_grid[0])
                leftmost = sorted_grid[-1][0]
            try:
                self.states_visited[myPos] +=1
            except:
                self.states_visited[myPos] = 0



            for i in sorted_grid:
                enemy_distance = [gameState.getAgentDistances()[l] for l in CaptureAgent.getOpponents(self, gameState)]
                enemy_distance = min(enemy_distance)
                if gameState.getAgentState(self.index).numCarrying > 3:
                    mdp.addReward(i,1/(self.distanceHome(i,gameState)+2))
                if self.red:
                    if i[0] ==  self.homeBoundaryCells[0][0]+1:
                        mdp.addReward(i, 0.0002)
                        mdp.addReward((i[0]-1,i[1]),0.0001)
                        mdp.addReward((i[0]-2,i[1]),0.0001)

                if not self.red:
                    if i[0] == self.homeBoundaryCells[0][0]-1:
                        mdp.addReward(i,0.0002)
                        mdp.addReward((i[0]+1,i[1]),0.0001)
                        mdp.addReward((i[0]+2,i[1]),0.0001)
                
                if i in foodPositions:
                    food_in_grid += 1
                    mdp.addReward(i, (0.8*enemy_distance+2+0.5*self.distancer.getDistance(myPos,i))/(self.distanceHome(i,gameState)))
                    
        
                oppo1 = gameState.getAgentPosition(self.getOpponents(gameState)[0])
                oppo2 = gameState.getAgentPosition(self.getOpponents(gameState)[1])
                
                
                if oppo1 is not None or oppo2 is not None:
                   # print("opposition locations are ", oppo1, oppo2)

                    if oppo1 is not None:
                        x,y = oppo1
                        if i in gameState.getCapsules():
                            mdp.addReward(i,3)
                        mdp.addReward(oppo1, -6)
                        mdp.addReward((x-1,y),-3)
                        mdp.addReward((x+1,y),-3)
                        mdp.addReward((x,y-1),-3)
                        #if (x-2,y)  not in gameState.getWalls().asList():
                        #    mdp.addReward((x-2,y),-4/2)
                        
                        #if (x+2,y)  not in gameState.getWalls().asList():
                        #    mdp.addReward((x+2,y),-4/2)
                        
                        
                        #if (x,y-2)  not in gameState.getWalls().asList():
                        #    mdp.addReward((x,y-2),-4/2)
                        
                        mdp.addReward((x,y+1),-4)
                        #if (x,y+2)  not in gameState.getWalls().asList():
                        #    mdp.addReward((x,y+2),-4/2)
                        
                    if oppo2 is not None:
                        x,y = oppo2
                        if i in gameState.getCapsules():
                            mdp.addReward(i,3)
                        mdp.addReward(oppo2, -6)
                            
                        mdp.addReward((x-1,y),-4)
                        #if (x-2,y) not in gameState.getWalls().asList():
                        #    mdp.addReward((x-2,y),-4/2)
                        mdp.addReward((x+1,y),-4)
                        #if (x+2,y) not in gameState.getWalls().asList():
                        #    mdp.addReward((x+2,y),-4/2)
                        mdp.addReward((x,y-1),-4)
                        #if (x,y-2) not in gameState.getWalls().asList():
                        #    mdp.addReward((x,y-2),-4/2)
                        mdp.addReward((x,y+1),-4)
                        #if (x,y+2) not in gameState.getWalls().asList():
                        #    mdp.addReward((x,y+2),-4/2)
                    if gameState.getAgentState(self.index).numCarrying > 5:
                        for k in self.homeBoundaryCells:
                            mdp.addReward(k,1)
                  #  print("leftmost is ", leftmost)
                    
               
                
           # print("number of elements in foodgrid are ", food_in_grid)
          #  print("number of food carrying are ", gameState.getAgentState(self.index).numCarrying)
            if gameState.getAgentState(self.index).numCarrying > 10:
                for j in self.homeBoundaryCells:
                    mdp.addReward(j,100)
           
            if food_in_grid <= 1 and myPos[0] < self.homeBoundaryCells[0][0]:
                while food_in_grid >3:
                    self.length_of_grid += 1
                    grid = self.makeMap(agentpos, gameState, self.length_of_grid)
                    for i in grid:
                        if i in CaptureAgent.getFood(self,gameState).asList():
                            food_in_grid += 1

                
            Qvalue = ValueIteration(mdp, 0.8)
            
            return Qvalue.getAction(myPos)
    
        
            
        
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
          # Only half a grid position was covered
          return successor.generateSuccessor(self.index, action)
        else:
          return successor
    def chooseActionAStar(self, gameState):
        foodGrid = CaptureAgent.getFood(self, gameState).asList()
        myState = gameState.getAgentState(self.index)
        position = myState.getPosition()

        def getSuccessors(x):
            ini_x, ini_y = x
            succ = []
            if (ini_x, ini_y+1) not in gameState.getWalls().asList():
                succ.append([(ini_x, ini_y+1), Directions.NORTH])
            if (ini_x, ini_y-1) not in gameState.getWalls().asList():
                succ.append([(ini_x, ini_y-1), Directions.SOUTH])
            if (ini_x+1, ini_y) not in gameState.getWalls().asList():
                succ.append([(ini_x+1, ini_y), Directions.EAST])
            if (ini_x-1, ini_y) not in gameState.getWalls().asList():
                succ.append([(ini_x-1, ini_y), Directions.WEST])
            #print("succ", succ)
            return succ

        def isGoalState(state):
            #print("len foodGrid is", len(foodGrid))
            if not self.red:
                return state[0] < self.homeBoundaryCells[0][0]
            else:
                return state[0] > self.homeBoundaryCells[0][0]
        def heuristic(gameState, foodGrid, position):
            distances = []
            indices = []
            initial_x,initial_y = position[0],position[1]

            if isGoalState(x):
                return 0   
            for i in foodGrid:
                distances.append(abs(initial_x - i[0]) + abs(initial_y - i[1]))
                indices.append(i)
            #print("heuristic returned is", max(distances))
            return max(distances)
        priority_wastar = util.PriorityQueue()
        current_node = position
        nodes_visited = []
        
        priority_wastar.push((current_node,[]),0)
        food_eat = 0
        
        while not priority_wastar.isEmpty() :
            
            x,y = priority_wastar.pop()
            
            if isGoalState(x):
                return y
            
            if x not in nodes_visited: 
                next = getSuccessors(x)
                for i in next:
                    current_node = i[0]
                    if current_node not in nodes_visited:
                        #print("next is ", i, heuristic(gameState, foodGrid,current_node))
                        direction = i[1]
                        priority_wastar.push((current_node,y+[direction]),heuristic(gameState, foodGrid, current_node))
            nodes_visited.append(x)
            
        return y+[direction]

    def pathToGoalAStar(self, gameState, current_state, goal):
        def heuristic(position):
            distances = []
            indices = []
            initial_x,initial_y = position[0],position[1]
            for i in goal:
                distances.append((initial_x - i[0]) + abs(initial_y - i[1]))
            return min(distances)
        def getSuccessors(x):
            ini_x, ini_y = x
            succ = []
            if (ini_x, ini_y+1) not in gameState.getWalls().asList():
                succ.append([(ini_x, ini_y+1), Directions.NORTH])
            if (ini_x, ini_y-1) not in gameState.getWalls().asList():
                succ.append([(ini_x, ini_y-1), Directions.SOUTH])
            if (ini_x+1, ini_y) not in gameState.getWalls().asList():
                succ.append([(ini_x+1, ini_y), Directions.EAST])
            if (ini_x-1, ini_y) not in gameState.getWalls().asList():
                succ.append([(ini_x-1, ini_y), Directions.WEST])
            #print("succ", succ)
            return succ
        priority_wastar = util.PriorityQueue()
        current_node = current_state
        nodes_visited = []
       
        priority_wastar.push((current_node,[]),0)
        
        
        while not priority_wastar.isEmpty() :
            
            x,y = priority_wastar.pop()
            
            if x in goal:
                return y
            if x not in nodes_visited: 
                next = getSuccessors(x)
                for i in next:
                    current_node = i[0]
                    
                    if current_node not in nodes_visited:
                        
                        direction = i[1]
                        priority_wastar.push((current_node,y+[direction]),heuristic(current_node))
            nodes_visited.append(x)
            
        return y+[direction]

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

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman:    features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
    features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

class AstarDefensiveAgent(CaptureAgent):
    '''
    This is an offensive agent to eat opponent's foods,
    which applies A* with heuristic function.
    '''
    def makeMap(self, initial_state, gameState, length_search_square, all_stats = False):

        search_grid = []
        game_length, game_breadth = gameState.getWalls().asList()[-1]
        #print("position_x and position_y are", initial_state)
        position_x, position_y = initial_state
        position_x = int(position_x)
        position_y = int(position_y)
        search_x_start = max(1, position_x-int(length_search_square))
        search_x_end = min(game_length, position_x+int(length_search_square))
        search_y_start = max(1, position_y-int(length_search_square))
        search_y_end = min(game_breadth, position_y+int(length_search_square))
        print(search_x_start, search_x_end, search_y_start, search_y_end)
        x = list(range(search_x_start,search_y_end))
        y = list(range(search_y_start, search_y_end))
        

        for x in range(search_x_start, search_x_end):
            for y in range(search_y_start,search_y_end):
                #print("(x,y) is", x,y)
                if (x,y) not in gameState.getWalls().asList():
                    search_grid.append((x,y))
        #print("search grid is", search_grid)
        if all_stats == False:
            return search_grid
        else:
            return (search_x_start, search_x_end, search_y_start, search_y_end)
    def registerInitialState(self, gameState):
        '''
        This is a function to initialize arguments.

        '''
        CaptureAgent.registerInitialState(self, gameState)
        self.width = gameState.data.layout.width
        self.height = gameState.data.layout.height
        midPointLeft = (int(self.width / 2.0)-1, int(self.height / 2.0))
        midPointRight = (int(self.width / 2.0)+1, int(self.height / 2.0))
        if gameState.isOnRedTeam(self.index):
            self.midPoint = midPointLeft
        else:
            self.midPoint = midPointRight

        self.start = gameState.getAgentPosition(self.index)
        self.counter = 0
        self.goal = []
        self.lastPacmanPosition = {}
        self.counter = {}
        CaptureAgent.registerInitialState(self, gameState)
        for opponentIndex in CaptureAgent.getOpponents(self, gameState):
            self.counter[opponentIndex] = 0

    def isGoal(self, gameState):
        '''
        This function is used in A* search to judge the goal state
        '''
        x,y = gameState.getAgentState(self.index).getPosition()
        #print("self.goal is ")
        if (int(x),int(y)) in self.goal:
           # print("yessss it is true!!!!!", gameState.getAgentState(self.index).getPosition())
            return True
        else:
           # print("NOOOOO IT IS FALSE!!!!!", gameState.getAgentState(self.index).getPosition())
            return False

    def chooseAction(self, gameState):
        '''
        Return the best action currently to move according to the state.
        '''
       # print("runnnning choooseaction")
        myPos = gameState.getAgentState(self.index).getPosition()
        # record the positions of the opponent ghosts when they appear in the last time
        
        # calculate the minimum distance to one ghost
        distancesToGhost = []
        distancesToPacman = []
        if len(self.lastPacmanPosition) != 0:
            for opponentIndex in self.getOpponents(gameState):
                if opponentIndex in self.lastPacmanPosition:
                    distancesToGhost.append(self.getMazeDistance(myPos, self.lastPacmanPositionhostsPos[opponentIndex]))
            distanceToGhost = min(distancesToGhost)
        else:
            distanceToGhost = 999
        
        # record the maximum time left for opponent to be scared
        scaredTimer = 0
        opponentIndex = self.getOpponents(gameState)
        for oppo in opponentIndex:
            currentTimer = gameState.getAgentState(oppo).scaredTimer
            if currentTimer > 0:
                scaredTimer = currentTimer

        # how many foods left to eat
        foodLeft = len(self.getFood(gameState).asList())

        # distance to the nearest food
        distance_to_latest_food_eaten = 999 
        if len(self.latest_food_eaten(gameState)) > 0:
            distance_to_latest_food_eaten = self.distancer.getDistance(list(self.latest_food_eaten(gameState))[0],myPos)
        distance_to_pacman = 999
        if not util.flipCoin(0.05) :
            if  gameState.getAgentPosition(opponentIndex[0]) is not None and gameState.getAgentPosition(opponentIndex[1]) is not None:
                distance_to_pacman = min(self.distancer.getDistance(gameState.getAgentPosition(opponentIndex[0]),myPos),self.distancer.getDistance(gameState.getAgentPosition(opponentIndex[1]),myPos))
                self.goal = []
                x,y = gameState.getAgentPosition(opponentIndex[1])
                grid = self.makeMap(myPos, gameState, 5)
                mdp = MDP(myPos, grid)
                
                mdp.addReward((x,y),6)
                mdp.addReward((x-1,y),3)
                mdp.addReward((x,y-1),3)
                mdp.addReward((x+1,y),3)
                mdp.addReward((x,y+1),3)
                return ValueIteration(mdp, 0.8).getAction(myPos)
                #u,v = gameState.getAgentPosition(opponentIndex[1])
                #self.goal.append([(x,y),(u,v)])
                #bestAction = self.aStarSearch(gameState)
                #return bestAction
            if gameState.getAgentPosition(opponentIndex[1]) is not None:
               # print("running mdp on the distances to the one opponent")
                distance_to_pacman = self.distancer.getDistance(gameState.getAgentPosition(opponentIndex[1]),myPos)
                self.goal = []
                #x,y = gameState.getAgentPosition(opponentIndex[1])
                x,y = gameState.getAgentPosition(opponentIndex[1])
                grid = self.makeMap(myPos, gameState, 5)
                mdp = MDP(myPos, grid)
                mdp.addReward((x,y),6)
                mdp.addReward((x-1,y),3)
                mdp.addReward((x,y-1),3)
                mdp.addReward((x+1,y),3)
                mdp.addReward((x,y+1),3)
                return ValueIteration(mdp, 0.8).getAction(myPos)
                
            if gameState.getAgentPosition(opponentIndex[0]) is not None:
              #  print("running mdp on the distances to the one opponent")
                distance_to_pacman = self.distancer.getDistance(gameState.getAgentPosition(opponentIndex[0]),myPos)
                self.goal = []
                x,y = gameState.getAgentPosition(opponentIndex[0])
                #x,y = gameState.getAgentPosition(opponentIndex[1])
                grid = self.makeMap(myPos, gameState, 8)
                mdp = MDP(myPos, grid)
                mdp.addReward((x,y),6)
                mdp.addReward((x-1,y),3)
                mdp.addReward((x,y-1),3)
                mdp.addReward((x+1,y),3)
                mdp.addReward((x,y+1),3)
                return ValueIteration(mdp, 0.8).getAction(myPos)

                self.goal.append(gameState.getAgentPosition(opponentIndex[0]))
                if (x+1,y) not in gameState.getWalls():
                    self.goal.append((x+1,y))
                if (x-1,y) not in gameState.getWalls():
                    self.goal.append((x-1,y))
                if (x,y+1) not in gameState.getWalls():
                    self.goal.append((x,y+1))
                if (x,y-1) not in gameState.getWalls():
                    self.goal.append((x,y-1))

                bestAction = self.aStarSearch(gameState)
                return bestAction
            if distance_to_pacman > distance_to_latest_food_eaten:
              # print("running a* on the distances to the nearest food")
                self.goal = []
                self.goal.append(list(self.latest_food_eaten(gameState))[0])
              #  print("goal is ", self.goal)
                bestAction = self.aStarSearch(gameState)
                return bestAction

            self.goal.append(self.midPoint)
          #  print("goal is ", self.goal)
            bestAction = self.aStarSearch(gameState)

#            print("actions is ", bestAction)
            return bestAction
        else:
         #   print("returning a random action")
            return random.choice(gameState.getLegalActions(self.index))



        
    def latest_food_eaten(self, gameState):
        try:
            prevfoodGamestate = self.getPreviousObservation()
            #print("prevfoodgamestate is ", prevfoodGamestate)
            prevfood = set(CaptureAgent.getFoodYouAreDefending(self, prevfoodGamestate).asList())
            currentfood = set(CaptureAgent.getFoodYouAreDefending(self, gameState).asList())
            return prevfood-currentfood
        except:
            return []
    
    def aStarSearch(self, gameState):
        '''
        A* search with heuristic function
        Return the first action in the best path to goal
        '''
      #  print("running a*")
        closed = [] # visited positions list
        priorityQueue = util.PriorityQueue()
        initialState = gameState
        priorityQueue.push([initialState, [], 0], 0)
        while not priorityQueue.isEmpty():
            currentState, actionsToCurrent, costToCurrent = priorityQueue.pop()
          #  print("states are", currentState, actionsToCurrent, costToCurrent)
            if not currentState.getAgentState(self.index).getPosition() in closed: # expand unvisited state
                if self.isGoal(currentState): # whether this is goal state
                    if len(actionsToCurrent) == 0:  # if no action returned, then return the nearest action to home.
                        legalActions = currentState.getLegalActions(self.index)
                        result = legalActions[0]
                        minDistance = 999
                        for action in legalActions:
                            successor = gameState.generateSuccessor(self.index,action)
                            position = successor.getAgentState(self.index).getPosition()
                            distanceToStart = self.distancer.getDistance(position,self.start)
                            if distanceToStart < minDistance:
                                minDistance = distanceToStart
                                result = action
                   #    print("returining result",result)
                        return result
                    else:
                    #    print("returnin action ",actionsToCurrent[0])
                        return actionsToCurrent[0] # first action to home
                closed.append(currentState.getAgentState(self.index).getPosition())
                actions = currentState.getLegalActions(self.index)
                for action in actions:
                    nextState = currentState.generateSuccessor(self.index, action)
                    if not nextState.getAgentState(self.index).getPosition() in closed: # push unvisited state
                        actionsToNext = actionsToCurrent + [action]
                        costToNext = costToCurrent + 1 # steps from initial position to current position
                        costWithHeuristic = costToNext + self.heuristic(nextState, costToNext) # this value is the priority
                        priorityQueue.push([nextState, actionsToNext, costToNext], costWithHeuristic) # The nearer from ghost, it has lower priority to be expanded

        return Directions.STOP
    def heuristic(self, gameState, costToNext):
        '''
        heuristic function used in A* search
        Return a value that has a negative correlation with the distance between pacman and opponent ghost.
        The shorter the distance between pacman and ghost, the biger the value it returnself.
        As a result, it will cause a lower priority to be expanded
        '''
        thisPos = gameState.getAgentState(self.index).getPosition()

        # calculate the distance to the nearest ghost
        distancesToGhost = []
        value = 0
        if len(self.lastPacmanPosition) != 0:
            for opponentIndex in self.getOpponents(gameState):
                if opponentIndex in self.lastPacmanPosition:
                    if self.getMazeDistance(thisPos, self.lastPacmanPosition[opponentIndex]) <= 5:
                        value += (6 - self.getMazeDistance(thisPos, self.lastPacmanPosition[opponentIndex])) * 1000

        # return a heuristic value: the longer the distance, the larger the value
        return value

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
    '''
    def final(self, gameState):
        #print("*****running final******")
        #print(self.weights)
        file = open('weighted.txt', 'w')
        file.write(str(self.weights))
    '''



