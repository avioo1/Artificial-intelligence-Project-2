from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint
def createTeam(firstIndex, secondIndex, isRed,
							 first = 'DefensiveLearningAgent', second = 'DummyAgent'):
	"""
	This function should return a list of two agents that will form the
	team, initialized using firstIndex and secondIndex as their agent
	index numbers.
	"""
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class DummyAgent(CaptureAgent):
	def chooseAction(self, gameState):
		return Directions.STOP



class DefensiveLearningAgent(CaptureAgent):
	#print("running defensive agent")
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


	def registerInitialState(self, gameState):
		#print("running register initail state")
		self.states_visited = {}
		CaptureAgent.registerInitialState(self, gameState)
		self.epsilon = 0.1 #exploration prob
		self.alpha = 0.3 #learning rate
		self.discountRate = 0.8
		
		
		self.food = CaptureAgent.getFoodYouAreDefending(self, gameState)
		try:
			with open('weights.txt', "r") as file:
				self.weights = eval(file.read())
		except:
			self.weights = {'closestFood':0,'nearbyOpponents':0, 'nearbyOppoDistance':0, 'number_of_food_left':0, 'opponent_score':0, 'capsule_left':0, 'opponent_distance':0, 'bias':0, "reverse":0, "stop":0}



	def getFeatures(self, gameState, action):
		#print("running features")
		feature = util.Counter()
		succ = self.getSuccessor(gameState, action)
		mystate = succ.getAgentState(self.index)
		mypos = mystate.getPosition()
		opponents = self.getOpponents(succ)
		nearbyOpponents = [succ.getAgentDistances()[i] for i in opponents if succ.getAgentDistances()[i] is not None]


		feature["nearbyOpponents"] = len(nearbyOpponents)
		if len(nearbyOpponents) == 0:
			feature['opponent_distance'] = min([succ.getAgentDistances()[i] for i in opponents])
		else:
			feature['opponent_distance'] = min(nearbyOpponents)

		#feature["nearbyOppoDistance"] = min(nearbyOpponents)
		
		feature["number_of_food_left"] = len(CaptureAgent.getFoodYouAreDefending(self,succ).asList())
		feature["opponent_score"] = self.getScore(succ)
		feature["capsule_left"] = len(succ.getCapsules())
		feature["bias"] = 1.0
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		
		if action == Directions.STOP:
			feature["stop"] = 1.0
	    
		if action == rev: 
			feature['reverse'] = 1.0
		def closestFood(pos, food, walls):
			"""
			closestFood -- this is similar to the function that we have
			worked on in the search project; here its all in one place
			"""
			#print("pos is ", pos)
			fringe = [(pos[0], pos[1], 0)]
			expanded = set()
			while fringe:
				pos_x, pos_y, dist = fringe.pop(0)
				if (pos_x, pos_y) in expanded:
					continue
				expanded.add((pos_x, pos_y))
				# if we find a food at this location then exit
				if food[pos_x][pos_y]:
					return dist
				# otherwise spread out from the location to its neighbours
				nbrs = []
				for i in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
					if (pos_x+1,pos_y) not in walls:
						nbrs.append((pos_x+1,pos_y))
					if (pos_x-1,pos_y) not in walls:
						nbrs.append((pos_x-1,pos_y))
					if (pos_x,pos_y+1) not in walls:
						nbrs.append((pos_x,pos_y+1))
					if (pos_x,pos_y-1) not in walls:
						nbrs.append((pos_x,pos_y-1))

				for nbr_x, nbr_y in nbrs:
					fringe.append((nbr_x, nbr_y, dist+1))
		  # no food found
				#return None
		
		#minDistance = min([self.getMazeDistance(myPos, food) for food in self.food.asList()])
		feature['closestFood'] = closestFood(gameState.getAgentPosition(self.index), self.food, gameState.getWalls().asList())
		print("featuers are ", feature)
		feature.divideAll(10.0)
		

		
		return feature

	def getQValue(self, gameState, action):
		#print("running qvalue")
		features = self.getFeatures(gameState,action)
		#print("qvalue is ", features*self.weights, " actions is", action)
		return features*self.weights	

	def getValue(self, gameState):
		#print("running getvalue")
		value = []
		legal = gameState.getLegalActions(self.index)
		if len(legal) == 0:
			return 0.0
		else:
			for action in legal:
				value.append(self.getQValue(gameState, action))
			return max(value)

	def getPolicy(self, gameState):
		#print("running getaction")
		values = []
		legal = gameState.getLegalActions(self.index)
		legal.remove(Directions.STOP)

		for i in legal:
			self.update(gameState, i)
			values.append((i, self.getQValue(gameState,i)))

		return max(values)[0]

	def chooseAction(self, gameState):
		#print("running chooseaction")
		#print("runngin choOSEaXTON")
		legal = gameState.getLegalActions(self.index)
		legal.remove(Directions.STOP)
		action = None
		if util.flipCoin(self.epsilon):

			action = random.choice(legal)
			print("random action is", action)
		else:
			action = self.getPolicy(gameState)
			print("nottttt qrandom action is", action)

		return action

	def update(self, gameState, action):
		
		features = self.getFeatures(gameState,action)
		succ = self.getSuccessor(gameState,action)
		#features_next = self.getFeatures(succ, action)

		mystate = succ.getAgentState(self.index)
		mypos = mystate.getPosition()
		
		#print(states_visited[mypos] in states_visited)
		#if states_visited[mypos] not in states_visited: 
		try:

			self.states_visited[mypos] += 1
		except:
			self.states_visited[mypos] = 0
		
		#print("states visited is ", self.states_visited)
		rewards = 0
		#rewards = {"closestFood":-1,"nearbyOpponents":1,"opponent_distance": -2, "number_of_food_left":2, "capsule_left":0.5, "nearbyOppoDistance":0.5, "bias":1, "opponent_score":2,"stop":-0.5,"reverse":1 }
		#if features['closestFood'] > features_next['closestFood']:
		#	rewards += 10
		if mypos in self.food.asList():
			rewards += 10
		if mypos in succ.getCapsules():
			rewards += 15
		else  :
			rewards += 30/(self.states_visited[mypos]+0.1)

		reward = succ.getScore() - gameState.getScore()

		for i in features:
			upd = ((rewards+reward) + self.discountRate*self.getValue(gameState)) - self.getQValue(gameState, action)
			self.weights[i] += self.alpha*upd*features[i]
		


	def final(self, gameState):
		#print("*****running final******")
		#print(self.weights)
		file = open('weights.txt', 'w')
		file.write(str(self.weights))








		









