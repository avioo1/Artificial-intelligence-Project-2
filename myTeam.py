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

#import sys 

#sys.path.append(’teams/Power-Rangerz/’)



#print("runngin FINAL ")
import distanceCalculator 
import math
from game import Actions, Agent
from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
from game import Directions
import game
import datetime

from distanceCalculator import Distancer 


#print("running astarmyteam")


def createTeam(firstIndex, secondIndex, isRed,
               first = 'AStarDefender', second = 'MCTAgents'):
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

class ReflexCaptureAgents(CaptureAgent):

    def registerInitialState(self, gameState):

        CaptureAgent.registerInitialState(self, gameState)
        self.player1_in_danger = False
        self.player2_in_danger = False
        self.code_starting_time = 0
        self.pos_count = {}
        self.any_member_in_danger = False 
        self.food_eaten_by_opponents = []
        self.food_list = []
        self.att = False
        self.is_other_player_going = False
        self.coming_back_team = False
        self.start = gameState.getAgentPosition(self.index)
        self.lefthomeever = False
        self.walls = gameState.getWalls().asList()
        self.max_x = max([i[0] for i in self.walls])
        self.max_y = max([i[1] for i in self.walls])
        self.sign = 1 if gameState.isOnRedTeam(self.index) else -1
        self.positions_visited = []
        self.positions_visited_team = []
        self.homeXBoundary =self.start[0]  + ((self.max_x // 2 - 1) * self.sign)
        cells = [(self.homeXBoundary, y) for y in range(1, self.max_y)]
        self.homeBoundaryCells = [item for item in cells if item not in self.walls]
        self.states_visited = util.Counter()
        self.upper_level_food = [i for i in self.getFood(gameState).asList() if i[1]>self.max_y/2]
        self.lower_level_food = [i for i in self.getFood(gameState).asList() if i[1] <= self.max_y/2]
        self.counter = 0
        self.lastGhostsPos = {}
        self.counter = {}
        self.goals = []
        self.goals_not_food = []
        self.stuck_in_a_loop=False
        self.stuck_in_a_loop_team = False
        self.attack = True

    def nearest_home(self, gameState):
        mypos = gameState.getAgentState(self.index).getPosition()
        min_dist = 999
        min_home = self.homeBoundaryCells[0][0]
        for i in self.homeBoundaryCells:
            if self.getDistance(i,mypos) < min_dist:
                min_dist = self.getDistance(i,mypos)
                min_home = i
        return i,min_dist
        
    
    
    
    def stuck(self, gameState, is_teammate=False):
        team = [i for i in self.getTeam(gameState) if i != self.index][0]
        if is_teammate == False:
            self.positions_visited.append(gameState.getAgentState(self.index).getPosition())
            if len(self.positions_visited) >= 4:
                if len(set(self.positions_visited)) == 2:
                    self.stuck_in_a_loop =True
                    self.positions_visited = list(set(self.positions_visited))
                    return True 
                    
            if len(self.positions_visited) >= 4 and len(set(self.positions_visited)) != 2:
                self.stuck_in_a_loop = False
                self.positions_visited.pop(0)
            else:
                self.stuck_in_a_loop =False
            return False
        if is_teammate == True:
            self.positions_visited_team.append(gameState.getAgentState(team).getPosition())
            if len(self.positions_visited_team) >= 4 and len(set(self.positions_visited_team)) == 2:
                self.stuck_in_a_loop_team =True
                self.positions_visited_team = list(set(self.positions_visited_team))
            if len(self.positions_visited_team) >= 4 and len(set(self.positions_visited_team)) != 2:
                self.stuck_in_a_loop = False
                self.positions_visited_team.pop(0)
            else:
                self.stuck_in_a_loop_team =False
    
    
    
    
    
    def getDistance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) +  abs(pos1[1] - pos2[1])
    
    
    
    def main_defending_positions(self, gameState):
    
        patrol_positions=[]
        breadth = gameState.data.layout.width
        length = gameState.data.layout.height
        
        if self.red:
            for i in range(int(breadth/2-2),int(breadth/2)):
                for j in range(int(length/2) -3,int(length/2)+4):
                    if (i,j) not in self.walls:
                        patrol_positions.append((i,j))
        if not self.red:
            for i in range(int(breadth/2)+1,int(breadth/2)+3):
                for j in range(int(length/2)-3,int(length/2)+4):
                    if (i,j) not in self.walls:
                        patrol_positions.append((i,j))
        return patrol_positions

    
    
    
    def nearest_food(self, gameState):
        food = self.getFood(gameState).asList()
        mypos = gameState.getAgentState(self.index).getPosition()
        minimum_food = (100,100)
        min_dist = 999
        for i in food:
            if self.getDistance(i,mypos) < min_dist:
                min_dist = self.getDistance(i,mypos)
                minimum_food = i
        return minimum_food,min_dist

    
    
    def costOfAction(self, gameState, nextState):
    
        
        foodlist = self.getFood(gameState).asList()
        nextPos = nextState.getAgentState(self.index).getPosition()
        myPos = gameState.getAgentState(self.index).getPosition()
        team = [i for i in self.getTeam(gameState) if i != self.index][0]
        team_pos = gameState.getAgentState(team).getPosition()
        
        opponents = self.getOpponents(gameState)
        opponent_index_nearest = []
        distance_of_the_nearest_ghost = 0

        
        for idx in opponents:
            if nextState.getAgentState(idx).getPosition() is not None  :
                oppo_pos = nextState.getAgentState(idx).getPosition()
                if  gameState.getAgentState(self.index).isPacman and nextState.getAgentState(idx).scaredTimer  < 4 and self.distancer.getDistance(oppo_pos,nextPos) < 2:
                    #print("adding cost ", 999)
                    return 999
                if nextPos in gameState.getCapsules():
                    return -1
                
                
        
        #if  not self.any_member_in_danger and nextState.getAgentState(self.index).getPosition() in gameState.getCapsules() and not nextState.getAgentState(self.index).getPosition() in self.getCapsulesYouAreDefending(gameState):
        #    return 1200
        
        
        if nextPos in self.pos_count:
            if self.pos_count[nextPos] > 7:
                return 1.3*self.pos_count[nextPos]
        
        if self.distancer.getDistance(team_pos,myPos) < 4:
            return 10*self.distancer.getDistance(team_pos,myPos)

        return 10


    
    def nearest_capsule(self,gameState):
        capsules = self.getCapsules(gameState)
        mypos = gameState.getAgentState(self.index).getPosition()
        nearest_capsule = ()
        dist = 9999
        if len(capsules) > 0:
            for i in capsules:
                if self.getDistance(i,mypos) < dist:
                    dist = self.getDistance(i,mypos)
                    nearest_capsule=i
            return nearest_capsule,dist
        return None
    
    
    
    def choosing_attack_or_defend(self, gameState):
        food = self.getFood(gameState).asList()
        opponents = self.getOpponents(gameState)
        myteam = [i for i in self.getTeam(gameState) if i != self.index][0]
        mypos = gameState.getAgentState(self.index).getPosition()
        can_switch = min(self.distancer.getDistance(mypos,i) for i in self.homeBoundaryCells) < 6
        distance_between_teammate_ghost = 9999
        number_of_opponents = 0
        oppo_idx = -1

        if (not gameState.getAgentState(opponents[0]).isPacman and gameState.getAgentState(opponents[0]).scaredTimer> 2) and (not gameState.getAgentState(opponents[1]).isPacman and gameState.getAgentState(opponents[1]).scaredTimer> 2):
            return True 
        
        for i in opponents:
            if gameState.getAgentState(i).getPosition() is not None :
                ghost_pos = gameState.getAgentState(i).getPosition()
                team_pos = gameState.getAgentState(myteam).getPosition()
                if self.distancer.getDistance(team_pos,ghost_pos) < distance_between_teammate_ghost:
                    distance_between_teammate_ghost = self.distancer.getDistance(team_pos,ghost_pos)
                    oppo_idx = i
            if gameState.getAgentState(i).isPacman:
                number_of_opponents += 1
        
        
        if len(food) < 4 and gameState.getAgentState(self.index).isPacman:
            return False
        
        
        
        if gameState.getAgentState(self.index).scaredTimer >=5 :
            return True 
        '''
        ##CHECK THIS BEFORE SUBMITTING !!!!
        if (oppo_idx > 0 and not gameState.getAgentState(self.index).isPacman and not gameState.getAgentState(oppo_idx).isPacman and gameState.getAgentState(oppo_idx).getPosition() is not None):
            return False 
        '''
        if number_of_opponents > 0 :
            return False


        return self.attack

    def pathToGoalAStar(self, gameState, current_state, goal, depth):
        d = 0
        states_visited = [] 
        priorityQueue = util.PriorityQueue()
        initialState = gameState
        priorityQueue.push([initialState, [], 0], 0)
        while not priorityQueue.isEmpty():
            d =d+1
            #print("value of d is ", d)
            initialState, actionsToGoal, cost = priorityQueue.pop()
            if d == depth:
                return False 
            #print("cost is ",cost)
            if not initialState.getAgentState(self.index).getPosition() in states_visited: 
                if initialState.getAgentState(self.index).getPosition() == goal: 
                    return True 
                    if len(actionsToGoal) == 0:  
                        legalActions = initialState.getLegalActions(self.index)
                        result = legalActions[0]
                        minDistance = 999
                        for action in legalActions:
                            successor = self.getSuccessor(gameState,action)
                            position = successor.getAgentState(self.index).getPosition()
                            distanceToStart = self.getMazeDistance(position,self.start)
                            if distanceToStart < minDistance:
                                minDistance = distanceToStart
                                result = action
                        #print("returning result is ",result)
                        return result
                
                    
                
                states_visited.append(initialState.getAgentState(self.index).getPosition())
                actions = initialState.getLegalActions(self.index)
                for action in actions:
                    nextState = self.getSuccessor(initialState, action)
                    if not nextState.getAgentState(self.index).getPosition() in states_visited: 
                        actionsToNext = actionsToGoal + [action]
                        costToNext = self.costOfAction(initialState,nextState)
                        costWithHeuristic = self.costOfAction(initialState,nextState) #+ self.heuristic(nextState, costToNext)
                        priorityQueue.push([nextState, actionsToNext, costToNext], costWithHeuristic)
        #print("queue is empty")
        return Directions.STOP
    
    def aStarSearch(self, gameState):
        
        states_visited = [] 
        priorityQueue = util.PriorityQueue()
        initialState = gameState
        priorityQueue.push([initialState, [], 0], 0)
        while not priorityQueue.isEmpty():
            
            initialState, actionsToGoal, cost = priorityQueue.pop()
            
            if not initialState.getAgentState(self.index).getPosition() in states_visited: 
                if self.isGoal(initialState): 
                    if len(actionsToGoal) == 0:  
                        legalActions = initialState.getLegalActions(self.index)
                        result = legalActions[0]
                        minDistance = 999
                        for action in legalActions:
                            successor = self.getSuccessor(gameState,action)
                            position = successor.getAgentState(self.index).getPosition()
                            distanceToStart = self.getMazeDistance(position,self.start)
                            if distanceToStart < minDistance:
                                minDistance = distanceToStart
                                result = action
                        
                        return result
                    else:
                        #print("time taken is ",time.time() - self.code_starting_time)
                        #if time.time() - self.code_starting_time    > 0.8:
                           # print("astar")
                        return actionsToGoal[0] 
                
                states_visited.append(initialState.getAgentState(self.index).getPosition())
                actions = initialState.getLegalActions(self.index)
                for action in actions:
                    nextState = self.getSuccessor(initialState, action)
                    if not nextState.getAgentState(self.index).getPosition() in states_visited: 
                        actionsToNext = actionsToGoal + [action]
                        costToNext = self.costOfAction(initialState,nextState)
                        costWithHeuristic = self.costOfAction(initialState,nextState) + self.heuristic(nextState, costToNext)
                        priorityQueue.push([nextState, actionsToNext, costToNext], costWithHeuristic)
        
        return Directions.STOP


    
    
    def getSuccessor(self, gameState, action): 
        successor = gameState.generateSuccessor(self.index,action)
        position = successor.getAgentState(self.index).getPosition()
        if position != nearestPoint(position):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    
    
    def isGoal(self, state):
        x,y = state.getAgentState(self.index).getPosition()
        #print("x and y are ", x,y)
        #print("goals are ",self.goals)
        if (int(x),int(y)) in self.goals:
            #print("returned True")
            return True
        else:
            return False
        
    
    def latest_food_eaten(self, gameState, return_list = False):
        
        try:
            prevfoodGamestate = self.getPreviousObservation()
            prevfood = set(CaptureAgent.getFoodYouAreDefending(self, prevfoodGamestate).asList())
        except :
            prevfood = []
            return []
        currentfood = set(CaptureAgent.getFoodYouAreDefending(self, gameState).asList())
        if not return_list:
            if len(prevfood - currentfood) > 0:
                #print(":::::",list(prevfood-currentfood)[-1])
                self.food_list.append(list(prevfood-currentfood)[-1])
            if len(self.food_list) > 0:
                return self.food_list[-1]
            else:
                return []

        else:
            return self.food_eaten_by_opponents

    def upper_level(self, gameState):
        upper_level_food = []
        for i in self.getFood(gameState).asList():
            if 1.0*i[1]/gameState.data.layout.height > 0.5:
                upper_level_food.append(i)
        return upper_level_food

    def lower_level(self, gameState):
        upper_level_food = []
        for i in self.getFood(gameState).asList():
            if 1.0*i[1]/gameState.data.layout.height <= 0.5:
                upper_level_food.append(i)
        return upper_level_food
    
    def heuristic(self, gameState, cost):
        myPos = gameState.getAgentState(self.index).getPosition()
        if len(self.goals) == 1:
           # print("heuristic value is ", self.distancer.getDistance(self.goals[0],myPos))
            return self.distancer.getDistance(self.goals[0],myPos)
        dist = 999
        for i in self.goals: 
            if self.distancer.getDistance(myPos,i) < dist:
                dist = self.getDistance(myPos,i)
        return dist 

class AStarDefender(ReflexCaptureAgents):


    def defendingActions(self, gameState):
        self.code_starting_time = time.time()
        opponents = self.getOpponents(gameState)
        mypos = gameState.getAgentState(self.index).getPosition()
        team = [i for i in self.getTeam(gameState) if i != self.index][0]
        if mypos in self.pos_count:
            self.pos_count[mypos] +=1
        else:
            self.pos_count[mypos] = 0
        oppo_dist = 999
        oppo_index = -1
        number_of_opponents = 0
        self.goals = []

        
        
        
        ##FINDING THE NEAREST PACMAN TO EAT AND GOING FOR IT 
        for i in opponents:
            if gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).isPacman:
                number_of_opponents +=1
                idx = gameState.getAgentState(i).getPosition()
                if self.distancer.getDistance(idx,mypos) < oppo_dist:
                    oppo_dist = self.distancer.getDistance(idx,mypos)
                    oppo_index = i
        
        if oppo_index >= 0:
            self.goals = []
            self.goals.append(gameState.getAgentState(oppo_index).getPosition())
           # print("460")
            #print("goals are ",self.goals)
            return self.aStarSearch(gameState)
       
        
        
    
        ###IF NEAREST ENEMY IS A GHOST AND WE ARE PACMAN 
        if oppo_index >= 0 and gameState.getAgentState(self.index).isPacman:
            if gameState.getAgentState(oppo_index).getPosition() is not None:
                self.goals = []
                self.goals.append(gameState.getAgentState(oppo_index).getPosition())
                #print("471")
                return self.aStarSearch(gameState)
        
        
        ##INTERCEPT THE GHOSTS LOCATION USING ITS LATEST FOOD EATING POSTIION
        if len(self.latest_food_eaten(gameState)) > 0:
            self.goals.append(self.latest_food_eaten(gameState))
            #print("478")
            return self.aStarSearch(gameState)
        
        
        else:
            
            
            self.goals = []
            for i in self.main_defending_positions(gameState):
                self.goals.append(i)
           # print("488")
            return self.aStarSearch(gameState)
        
        
        
        
    def chooseAction(self, gameState):
        self.code_starting_time = time.time()
        team = [i for i in self.getTeam(gameState) if i != self.index][0]
        team_pos = gameState.getAgentState(team).getPosition()
        myPos = gameState.getAgentState(self.index).getPosition()
        if myPos in self.pos_count:
            self.pos_count[myPos] +=1
        else:
            self.pos_count[myPos] = 0
        ##IF AGENT IS IN HOME AND SPOTS AN INVADER THEN IT WILL GO FOR IT 
        
        

        if not self.choosing_attack_or_defend(gameState) :#or (gameState.getAgentState(i).getPosition() is not None and gameState.getAgentState(i).isPacman or (gameState.getAgentState(i).isPacman and gameState.getAgentState(i).numCarrying > 2)):#or self.getDistance(team_pos,myPos)<3:
            return self.defendingActions(gameState)
        else:
            opponents = self.getOpponents(gameState)
            team = [i for i in self.getTeam(gameState) if i != self.index][0]
            team_pos = gameState.getAgentState(team).getPosition()
            myPos = gameState.getAgentState(self.index).getPosition()
            foodlist = self.getFood(gameState).asList()
            moving_in_upper = 1.0*myPos[1]/gameState.data.layout.height <= 0.5
            mate_in_upper = 1.0*team_pos[1]/gameState.data.layout.height <= 0.5

            min_dist = 999
            oppo_idx = -1
            
            for idx in opponents:

                ##IF YOU SPOT A ENEMY WHILE COMING BACK THEN GO AND EAT IT !!!
                if not gameState.getAgentState(self.index).isPacman and  gameState.getAgentState(idx).isPacman and gameState.getAgentState(idx).getPosition() is not None and gameState.getAgentState(self.index).scaredTimer < 1:
                    self.goals = []
                    self.goals.append(gameState.getAgentState(idx).getPosition())
                    #print("527")
                    return self.aStarSearch(gameState)

                ##IF THERE IS A SCARED GHOST THEN GO AND EAT IT !!!
                if gameState.getAgentState(idx).scaredTimer > 2:
                    if gameState.getAgentState(idx).getPosition() is not None:
                        oppo_pos = gameState.getAgentState(idx).getPosition()
                        if self.getDistance(oppo_pos,myPos) < 4:
                            self.goals = []
                            self.goals.append(gameState.getAgentState(idx).getPosition())
                           # print("537")
                            return self.aStarSearch(gameState)
                    
                ##FINDING THE NEAREST GHOST NEARBY 
                if gameState.getAgentState(idx).getPosition() is not None :
                    oppo_pos = gameState.getAgentState(idx).getPosition()
                    if self.getDistance(oppo_pos,myPos) < min_dist:
                        min_dist = self.getDistance(oppo_pos,myPos)
                        oppo_idx = idx
            
            ##IF IN DANGER THEN RETURN HOME 
            for idx in opponents:
                if gameState.getAgentState(idx).getPosition() is not None :
                    oppo_pos = gameState.getAgentState(idx).getPosition()
                    if  (self.distancer.getDistance(myPos,oppo_pos) <= 3 and gameState.getAgentState(idx).scaredTimer < 4) or (gameState.data.timeleft < 40):
                        if self.nearest_capsule(gameState) is not None:
                            self.goals = []
                            self.goals.append(self.nearest_capsule(gameState)[0])
                          #  print("555")
                            return self.aStarSearch(gameState)
                        self.goals = []
                        self.goals.append(self.nearest_home(gameState)[0])
                        
                       # print("560")
                        return self.aStarSearch(gameState)
            
            
            ##IF THE PACMAN IS SAFE THEN EAT EVERYTHING GREEDILY 
            if oppo_idx >= 0:
                oppo_pos = gameState.getAgentState(oppo_idx).getPosition()
                if  (gameState.getAgentState(self.index).isPacman and self.getDistance(myPos,oppo_pos) > 5) or (gameState.getAgentState(self.index).isPacman and gameState.getAgentState(idx).scaredTimer > 4) :     
                    self.goals = []
                    self.goals.append(self.nearest_food(gameState)[0])
                  #  print("570")
                    return self.aStarSearch(gameState)
            
            
            
            
            if  gameState.getAgentState(self.index).numCarrying > 4 or gameState.data.timeleft < 50:
                self.goals = []
                for i in self.main_defending_positions(gameState):
                    self.goals.append(i)
               # print("580")
                return self.aStarSearch(gameState)
            
            
            if len(self.upper_level(gameState)) == 0 or len(self.lower_level(gameState)) == 0:
                self.goals=[]
                for i in foodlist:
                    self.goals.append(i)
                #print("588")
                return self.aStarSearch(gameState)

            
            self.goals = []
            for i in foodlist:
                self.goals.append(i)
           # print("595")
            return self.aStarSearch(gameState)



class MCTAgents(AStarDefender):

    


    def registerInitialState(self, gameState):
        ReflexCaptureAgents.registerInitialState(self,gameState)
        #self.index = index
        self.states = [gameState]
        self.wins_in_states = util.Counter()
        self.plays_in_states = util.Counter()
        self.allGameStates = list().append(gameState)
        
        

    def getDistance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) +  abs(pos1[1] - pos2[1])
    
    
    ####ADD GHOSTS CONDITION TO ITTTTT!!!!!!!!!!!!
    def MctsAction(self, gameState):
        state = gameState#self.states[-1]
        #print("self.index is ", self.index, state)
        actions = state.getLegalActions(self.index)
        #print("legal actions are ", actions )
        actions.remove(Directions.STOP)

        if len(actions) == 1:
            #print("returning")
            return actions, 0.0
        number_of_games = 0
        start = datetime.datetime.utcnow()
        #print("start time is ", start )
        
        
        
        while datetime.datetime.utcnow() -start < datetime.timedelta(seconds=0.6):
            #print("entering ")
            copy = self.states[:]
            #print("done copy ",copy)
            state = copy[-1]
            path = [state]
            mypos = gameState.getAgentState(self.index).getPosition()
            score = gameState.getScore()
            expand = True
            nextstates = []

            for i in range(1,21):


                state = gameState
                legal_actions = state.getLegalActions(self.index)
                #print("legal actions are ", legal_actions)

                if len(legal_actions) == 1:
                    if legal_actions[0] !=  Directions.STOP:
                        #print("returning from 1", legal_actions[0])
                        return legal_actions[0]

                
                nextstates = []
                for i in legal_actions:
                    nextstates.append(   (i, gameState.generateSuccessor(self.index,i)) ) 


                if all(self.plays_in_states.get(s.getAgentState(self.index).getPosition()) for i, s in nextstates):

                    if self.plays_in_states[state.getAgentState(self.index).getPosition()] == 0.0:
                        log = 0.5
                    
                    else:
                        log = float (   2.0 * math.log (self.plays_in_states [ state.getAgentState(self.index).getPosition()])   )
                    value = -9999
                    action=Directions.STOP
                    final_state = gameState
                    #print("log is ",log)
                    for i,s in nextstates:
                        curr_pos = s.getAgentState(self.index).getPosition()
                        if (float(self.wins_in_states[curr_pos]) / float (   self.plays_in_states[curr_pos])) + 2.0*math.sqrt(  log / float (  self.plays_in_states[curr_pos]   )  ) > value:
                            value = (float(self.wins_in_states[curr_pos]) / float (   self.plays_in_states[curr_pos])) + 2.0*math.sqrt(  log / float (  self.plays_in_states[curr_pos]   )  )
                            action = i
                            final_state = s
                else:

                    action,final_state = random.choice(nextstates)

                copy.append(final_state)
                path.append(final_state)
                #print("final state is ", final_state)
                if expand == True and final_state.getAgentState(self.index).getPosition() not in self.plays_in_states:
                    expand = False
                    self.plays_in_states[final_state.getAgentState(self.index).getPosition()] = 0.0
                    self.wins_in_states[final_state.getAgentState(self.index).getPosition()] = 0.0

                new_x,new_y = final_state.getAgentState(self.index).getPosition()

                ghosts = [i for i in self.getOpponents(gameState) if not gameState.getAgentState(i).isPacman]
                invaders = [i for i in self.getOpponents(gameState) if  gameState.getAgentState(i).isPacman]


                if len(self.getCapsules(gameState)) > 0:
                    min_dist = 999
                    pos = final_state.getAgentState(self.index).getPosition()
                    nearest_capsule = ()
                    for i in self.getCapsules(gameState):
                        #print("locations of capsule is ", i,pos)
                        if self.getDistance(pos,i) < min_dist:
                            min_dist = self.getDistance(pos,i)
                            nearest_capsule = i
                    
                    if final_state.getAgentState(self.index).getPosition() == i:
                        for i in path:
                            if i.getAgentState(self.index).getPosition() in self.plays_in_states:
                                curr_pos = i.getAgentState(self.index).getPosition()
                                self.wins_in_states[curr_pos] += 0.002
                        
                if final_state.getScore() > score + 3:
                    for i in path:
                        if i.getAgentState(self.index).getPosition() in self.plays_in_states:
                            curr_pos = i.getAgentState(self.index).getPosition()
                            self.wins_in_states[curr_pos] += 0.4
                    
            for i in path:
                if i.getAgentState(self.index).getPosition() in self.plays_in_states:
                    curr_pos = i.getAgentState(self.index).getPosition()
                    self.plays_in_states[curr_pos] += 1.0
            
            number_of_games += 1

        for i in legal_actions:
            nextstates.append (   (i, gameState.generateSuccessor(self.index,i))   )

        winningRatio = -9999
        final_action = Directions.STOP
        for i,s in nextstates:
            
            
            a =  float (  self.wins_in_states.get(s.getAgentState(self.index).getPosition(), 0)  ) / float (    self.plays_in_states.get(s.getAgentState(self.index).getPosition(), 1)    ) 
            if a > winningRatio:

                winningRatio = a

                final_action = i
        #print("printing from 2", final_action)
        return final_action, winningRatio
    
    
    def final_mcts_action(self, gameState):
        #print("finalmsctsaction")
        actions, win = self.MctsAction(gameState)
        nextstates = []
        for i in gameState.getLegalActions(self.index):
            nextstates.append((i, gameState.generateSuccessor(self.index,i)))

        if win == 0.0:

            #print("entering winning ratio columns")
            return_action = Directions.STOP
            min_dist = 9999
            #print("number carrying are ", gameState.getAgentState(self.index).numCarrying)
            if gameState.getAgentState(self.index).numCarrying > 0:
                goal = self.nearest_home(gameState)[0]
                for i,s in nextstates:
                    mypos = s.getAgentState(self.index).getPosition()
                    if  self.getDistance(mypos, goal ) < min_dist:
                        min_dist = self.getDistance(mypos, goal )
                        return_action = i
                #print("actions are 1",actions)
                actions=return_action
            
            else:
                min_dist = 999
                for i, s in nextstates:
                    mypos = s.getAgentState(self.index).getPosition()
                    dist = 999
                    for k in self.getFood(gameState).asList():
                        if self.getDistance(k,mypos) < dist:
                            dist = self.getDistance(k,mypos)
                    if dist < min_dist:
                        min_dist = dist
                        return_action = i
                actions = return_action

					
        #print("Actions is  ",actions)
        return actions
                
    
    def chooseAction(self, gameState):
        self.code_starting_time = time.time()
        opponents = self.getOpponents(gameState)
        team = [i for i in self.getTeam(gameState) if i != self.index][0]
        team_pos = gameState.getAgentState(team).getPosition()
        myPos = gameState.getAgentState(self.index).getPosition()
        if myPos in self.pos_count:
            self.pos_count[myPos] +=1
        else:
            self.pos_count[myPos] = 0
        

        foodlist = self.getFood(gameState).asList()
        moving_in_upper = 1.0*myPos[1]/gameState.data.layout.height <= 0.5
        mate_in_upper = 1.0*team_pos[1]/gameState.data.layout.height <= 0.5
        min_dist = 999
        oppo_idx = -1
        
        for idx in opponents:

            ##IF YOU SPOT A ENEMY WHILE COMING BACK THEN GO AND EAT IT !!!
            if not gameState.getAgentState(self.index).isPacman and  gameState.getAgentState(idx).isPacman and gameState.getAgentState(idx).getPosition() is not None :
                distance_opponent  = gameState.getAgentState(idx).getPosition()
                if gameState.getAgentState(self.index).scaredTimer < 1 and self.distancer.getDistance(distance_opponent, myPos) <= 6 and gameState.getAgentState(idx).numCarrying:
                    self.goals = []
                    self.goals.append(gameState.getAgentState(idx).getPosition())
                   # print("814")
                    return self.aStarSearch(gameState)



            if gameState.getAgentState(idx).getPosition() is not None :
                oppo_pos = gameState.getAgentState(idx).getPosition()
                if  (self.distancer.getDistance(myPos,oppo_pos) <= 3 and gameState.getAgentState(idx).scaredTimer < 4) or (gameState.data.timeleft < 40):
                    if self.nearest_capsule(gameState) is not None:
                        self.goals = []
                        self.goals.append(self.nearest_capsule(gameState)[0])
                        #print("825")
                        return self.aStarSearch(gameState)
                    self.goals = []
                    self.goals.append(self.nearest_home(gameState)[0])
                    self.player1_in_danger = True
                    #print("830")
                    return self.aStarSearch(gameState)
        


        for idx in opponents:

            ##EATING WHEN THE ENEMY IS SCARED 
            if gameState.getAgentState(idx).scaredTimer > 2 and gameState.getAgentState(self.index).isPacman:
                if gameState.getAgentState(idx).getPosition() is not None :
                    if self.getDistance(gameState.getAgentState(idx).getPosition(),myPos) < 4:
                        self.goals = []
                        self.goals.append(gameState.getAgentState(idx).getPosition())
                        #print("843")
                        return self.aStarSearch(gameState)


            
            ##FINDING THE NEAREST ENEMIES 
            if gameState.getAgentState(idx).getPosition() is not None :
                oppo_pos = gameState.getAgentState(idx).getPosition()
                if self.getDistance(oppo_pos,myPos) < min_dist:
                    min_dist = self.getDistance(oppo_pos,myPos)
                    oppo_idx = idx
        
        
        
        ###IF SAFE THEN EAT FOOD GREEDILLY 
        if oppo_idx > 0:
            oppo_pos = gameState.getAgentState(oppo_idx).getPosition()
            if  (gameState.getAgentState(self.index).isPacman and self.getDistance(myPos,oppo_pos) > 5) or (gameState.getAgentState(self.index).isPacman and gameState.getAgentState(idx).scaredTimer > 4) :    
                self.goals = []
                self.goals.append(self.nearest_food(gameState)[0])
                #print("eating greedily goals are 2", self.goals )
                #print("864")
                return self.aStarSearch(gameState)
        
        
        
        
        
        if  gameState.getAgentState(self.index).numCarrying > 4 or gameState.data.timeleft < 50:
            self.goals = []
            for i in self.homeBoundaryCells:
                self.goals.append(i)
           # print("875")
            return self.aStarSearch(gameState)
        
        
        
        if len(self.upper_level(gameState)) == 0 or len(self.lower_level(gameState)) == 0:
            self.goals=[]
            for i in foodlist:
                self.goals.append(i)
           # print("884")
            return self.aStarSearch(gameState)
        
        

        if mate_in_upper == moving_in_upper:
            if gameState.getAgentState(team).isPacman and gameState.getAgentState(self.index).isPacman:
                if mate_in_upper:
                    f = self.lower_level(gameState)
                    if len(f) > 0:
                        self.goals = []
                        for i in f:
                            
                            self.goals.append(i)
                      #  print("898")
                        return self.aStarSearch(gameState)
                else:
                    g = self.upper_level(gameState)
                    self.goals = []
                    if len(g) > 0:
                        for i in g:
                            self.goals.append(i)
                      #  print("906")
                        return self.aStarSearch(gameState)

        
        
        self.goals = []
        for i in foodlist:
            self.goals.append(i)
       # print("914")
        return self.aStarSearch(gameState)
