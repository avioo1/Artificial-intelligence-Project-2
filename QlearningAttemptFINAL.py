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
		self.start = gameState.getAgentPosition(self.index)
		self.walls = gameState.getWalls().asList()
		self.sign = 1 if gameState.isOnRedTeam(self.index) else -1
		self.max_x = max([i[0] for i in gameState.getWalls().asList()])
		self.max_y = max([i[1] for i in gameState.getWalls().asList()])
		self.homeXBoundary = self.start[0] + ((self.max_x // 2 - 1) * self.sign)
		cells = [(self.homeXBoundary, y) for y in range(1, self.max_y)]
		self.homeBoundaryCells = [item for item in cells if item not in self.walls]


		self.mid = (int(self.max_x/4), int(self.max_y/2))
		self.moves=0
		self.states_visited = {}
		CaptureAgent.registerInitialState(self, gameState)
		self.epsilon = 0.7 #exploration prob
		self.alpha = 0.1 #learning rate
		self.discountRate = 0.8
		self.food = CaptureAgent.getFoodYouAreDefending(self, gameState)
		try:
			with open('weights.txt', "r") as file:
				self.weights = eval(file.read())
		except:
			#self.weights = {'closestFood':-10,'nearbyOpponents':100, 'nearbyOppoDistance':-100, 'number_of_food_left':-20, 'opponent_score':-30, 'capsule_left':-10, 'opponent_distance':-100, 'midDistance':80,'bias':-10, "reverse":-10, "stop":-10}
			#self.weights = {'closestFood':1, "opponent_distance":1, "number_of_food_left":1, "number_states_visited":1}#, 'opponent_distance':-100}
			#self.weights = {'closertoboundary':0,'foodPosition': 0, 'capsulePosition': 0, 'closertofood':0, "opponent_distance":0, "times_visited":0} #, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
			self.weights = {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
	
	'''
	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)

		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()

		# Computes whether we're on defense (1) or offense (0)
		features['onDefense'] = 1
		if myState.isPacman: features['onDefense'] = 0

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
	'''
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
		values = {}
		legal = gameState.getLegalActions(self.index)
		legal.remove(Directions.STOP)

		for i in legal:
			self.update(gameState, i)
			values[i] = self.getQValue(gameState,i)
		
		maximum = max(values, key=values.get)
		#print("qvalues are ", values, maximum)
		return maximum

	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)

		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()

		# Computes whether we're on defense (1) or offense (0)
		features['onDefense'] = 1
		if myState.isPacman: features['onDefense'] = 0

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
	def chooseAction(self, gameState):
		#print("running chooseaction")
		#print("runngin choOSEaXTON")
		legal = gameState.getLegalActions(self.index)
		legal.remove(Directions.STOP)
		action = None
		#while self.moves<12:
		#	self.moves += 1
		#	return Directions.NORTH

		if util.flipCoin(self.epsilon):

			action = random.choice(legal)
			#print("random action is", action)
		else:
			action = self.getPolicy(gameState)
			#print("nottttt qrandom action is", action)
		#print("current and next states are ", gameState.getAgentState(self.index).getPosition(), self.getSuccessor(gameState,action).getAgentState(self.index).getPosition())
		return action

	def update(self, gameState, action):
		
		feature = self.getFeatures(gameState,action)
		succ = self.getSuccessor(gameState,action)
		#features_next = self.getFeatures(succ, action)

		mystate = gameState.getAgentState(self.index)
		mypos = mystate.getPosition()
		
		
		for i in feature:
			upd = (succ.getScore() + self.discountRate*self.getValue(gameState)) - self.getQValue(gameState, action)
			self.weights[i] += self.alpha*upd*feature[i]
		


	def final(self, gameState):
		#print("*****running final******")
		#print(self.weights)
		file = open('weights.txt', 'w')
		file.write(str(self.weights))








		









