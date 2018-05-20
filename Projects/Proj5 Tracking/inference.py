# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        #QUESTION 0
        #For an empty distribution or a distribution where all of the values are zero, 
        #do nothing
        #sum is a keyword in python!
        #Use the total method to find the sum of the values in the distribution.
        sUM = float(self.total())
        if sUM: 
            for key in self.keys():
                self[key] = self[key] / sUM
        else:
            #For an empty distribution or a distribution where all of the values are zero, do nothing
            return

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        #QUESTION 0
        #where the probability that a key is sampled is proportional to its 
        #corresponding value.
        #Note that the distribution does not necessarily have to be normalized 
        #prior to calling this method
        #Python's built-in random.random() function useful for this question.
        sUM = float(self.total())
        if sUM != 1.0:
            self.normalize()
        key_lst = list(self.keys())
        #print key_lst
        val_lst = list(self.values())
        #print val_lst
        p = random.random()
        counter = 0
        p_sum = val_lst[0]
        #print val_lst[0]
        while p > p_sum:
            counter += 1
            p_sum += val_lst[counter]
        return key_lst[counter]

class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        #Question 1
        #we want to return P(noisyDistance | pacmanPosition, ghostPosition).
        #probability distribution over distance readings given the true distance 
        #from Pacman to the ghost. This distribution is modeled by the function 
        #busters.getObservationProbability(noisyDistance, trueDistance), which 
        #returns P(noisyDistance | trueDistance)

        #if the ghost's position is the jail position, then the observation is 
        #None with probability 1, and everything else with probability 0
        if ghostPosition == jailPosition:
            #If the distance reading is None, then the ghost is in jail with probability 1
            if noisyDistance == None:
                return 1.0
            #If the distance reading is not None, then the ghost is in jail with probability 0.
            return 0.0
        elif noisyDistance == None:
                return 0.0

        # and use the provided manhattanDistance function to find the distance 
        #between Pacman's location and the ghost's location.
        dist_btw_ghost_pac = manhattanDistance(pacmanPosition, ghostPosition)
        return busters.getObservationProbability(noisyDistance, dist_btw_ghost_pac)
         
        
    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        #Question 2
        #The observe method should, for this problem, update the belief at every position 
        #on the map after receiving a sensor reading. 
        #Beliefs represent the probability that the ghost is at a particular location, 
        #and are stored as a DiscreteDistribution object in a field called self.beliefs, 
        #which you should update.
        #You should use the function self.getObservationProb that you wrote in the last 
        #question, which returns the probability of an observation given Pacman's position, 
        #a potential ghost position, and the jail position. 
        #You can obtain Pacman's position using gameState.getPacmanPosition(), 
        pacmanPosition = gameState.getPacmanPosition()
        #and the jail position using self.getJailPosition().
        jailPosition = self.getJailPosition()

        diDi = DiscreteDistribution()
        #getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition)
        ##You should iterate your updates over the variable self.allPositions which 
        #includes all legal positions plus the special jail position. 
        #print self.allPositions
        for ghostPosition in self.allPositions:
            diDi[ghostPosition] = self.getObservationProb(observation, pacmanPosition, ghostPosition, jailPosition) * self.beliefs[ghostPosition]
        self.beliefs = diDi
        self.beliefs.normalize()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        #Question 3
        #The elapseTime step should, for this problem, update the belief at every position 
        #on the map after one time step elapsing. Your agent has access to the action 
        #distribution for the ghost through self.getPositionDistribution. 
        diDi = DiscreteDistribution()
        for oldPos in self.allPositions:
            #In order to obtain the distribution over new positions for the ghost, 
            #given its previous position, use this line of code:
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            prev = self.beliefs[oldPos]
            for newPos in newPosDist.keys():
                #each position p in self.allPositions, newPosDist[p] is the probability that the ghost is at position p at time t + 1, given that the ghost is at position oldPos at time t. 
                diDi[newPos]+= (prev * newPosDist[newPos])
        self.beliefs = diDi
        self.beliefs.normalize()

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent);
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        #Question 5
        size = 0
        #print self.numParticles
        while size < self.numParticles:
            #the variable you store your particles in must be a list.
            temp = [pos for pos in self.legalPositions if size < self.numParticles]
            self.particles += temp
            size += len(temp)
    
    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        #Question 6
        diDi = DiscreteDistribution()
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()
        for ghostPosition in self.particles:
            #use the function self.getObservationProb to find the probability of an observation given Pacman's position, a potential ghost position, and the jail position
            diDi[ghostPosition] += self.getObservationProb(observation, pacmanPosition, ghostPosition, jailPosition)
        if diDi.total():
            diDi.normalize()
            self.beliefs = diDi 
            self.particles = [diDi.sample() for i in range(self.numParticles)] 
        else:
            #When all particles receive zero weight, the list of particles should be reinitialized by calling initializeUniformly.
            self.initializeUniformly(gameState)

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        #Question 7
        # newParticles = {} or discrete distribution
        # for i in range(self.numParticles):
        #     ghostPosition = self.particles[i]
        #     if ghostPosition in newParticles:
        #         #do something
        #     else:
        #         newPosDist = self.getPositionDistribution(gameState, ghostPosition)
        #         #need to take sample and put it in newGhosts

        belief = self.getBeliefDistribution()
        #print belief[0]
        newGhosts = DiscreteDistribution()

        for ghost in belief:
            newPosDist = self.getPositionDistribution(gameState, ghost)
            #print newPosDist
            for k in newPosDist:
                newPosDist[k] *= belief[ghost]
                if k in newGhosts:
                    newGhosts[k] += newPosDist[k]
                else :
                    newGhosts[k] = newPosDist[k]
        if newGhosts.total():
            #construct a new list of particles that corresponds to each existing particle in self.particles advancing a time step
            newGhosts.normalize()
            #then assign this new list back to self.particles. 
            self.particles = [newGhosts.sample() for i in range(self.numParticles)]
        else:
            self.initializeUniformly(gameState)
            #python autograder.py -t test_cases/q1/4-ParticlePredict

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        """
        "*** YOUR CODE HERE ***"
        #Question 5
        diDi = DiscreteDistribution()
        #Use self.particles for the list of particles.
        for ghostPosition in self.particles:
            diDi[ghostPosition] += 1
        diDi.normalize()
        return diDi

class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        #QuesTION 8
        legalPositions = self.legalPositions
        #look at itertools.product to get an implementation of the Cartesian product
        cartesianProduct = list(itertools.product(legalPositions,legalPositions))
        #you must then shuffle the list of permutations in order to ensure even placement of particles across the board.
        random.shuffle(cartesianProduct)
        size = 0
        while size < self.numParticles:
            #the variable you store your particles in must be a list.
            temp = [pos for pos in cartesianProduct if size < self.numParticles]
            self.particles += temp
            size += len(temp)

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1);

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        #Question 9
        diDi = DiscreteDistribution()
        pacmanPosition = gameState.getPacmanPosition()
        # jailPosition = self.getJailPosition()
        for ghostPosition in self.particles:
            weight = 1
            #To loop over all the ghosts, use:
            for i in range(self.numGhosts):
                weight *= self.getObservationProb(observation[i], pacmanPosition, ghostPosition[i], self.getJailPosition(i))
            diDi[ghostPosition] += weight

        self.beliefs = diDi
        if diDi.total():
            self.beliefs.normalize()
            self.particles = [self.beliefs.sample() for i in range(self.numParticles)]   
        else:
            #In this case, self.particles should be recreated from the prior distribution by calling initializeUniformly.
            self.initializeUniformly(gameState)

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for prevGhostPositions in self.particles:
            newParticle = list(prevGhostPositions)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            #you can loop over the ghosts using
            for i in range(self.numGhosts):
                newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])
                newParticle[i] = newPosDist.sample()
            newParticleT = tuple(newParticle)
            """*** END YOUR CODE HERE ***"""
            newParticles.append(newParticleT)
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist