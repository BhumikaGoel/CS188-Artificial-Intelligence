# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
          valuesCopy = self.values.copy()
          mdpStates = self.mdp.getStates()
          for state in mdpStates:
            possibleActions = self.mdp.getPossibleActions(state)
            tempLst = [self.computeQValueFromValues(state, action) for action in possibleActions]
            if tempLst:
              update = max(tempLst)
              valuesCopy[state] = update
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
        #util.raiseNotDefined()
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
          qValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount*self.values[nextState])
        return qValue
        
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        actionDict = util.Counter()
        possibleActions = self.mdp.getPossibleActions(state)
        if not possibleActions:
          return None
        for action in possibleActions:
          actionDict[action] = self.computeQValueFromValues(state, action)
        return actionDict.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdpStates = self.mdp.getStates()
        numStates = len(self.mdp.getStates())
        quotient = self.iterations//numStates
        remainder = self.iterations % numStates
        
        # for i in range(1,quotient+1):
        #   valuesCopy = self.values.copy()
        #   for state in mdpStates:
        #     if not self.mdp.isTerminal(state):
        #       possibleActions = self.mdp.getPossibleActions(state)
        #       tempLst = [self.computeQValueFromValues(state, action) for action in possibleActions]
        #       if tempLst:
        #         update = max(tempLst)
        #         valuesCopy[state] = update
        #   self.values = valuesCopy
 
        for i in range(self.iterations):
          valuesCopy = self.values.copy()
          state = mdpStates[i % numStates]
          possibleActions = self.mdp.getPossibleActions(state)
          tempLst = [self.computeQValueFromValues(state, action) for action in possibleActions]
          if tempLst:
              update = max(tempLst)
              valuesCopy[state] = update
          self.values = valuesCopy


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        #Compute predecessors of all states.
        prevStates = {}
        mdpStates = self.mdp.getStates()
        mdpStatesNonTerminal = [state for state in mdpStates if not self.mdp.isTerminal(state)]
        for state in mdpStatesNonTerminal:  
          possibleActions = self.mdp.getPossibleActions(state)
          for action in self.mdp.getPossibleActions(state):
            for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
              if nextState in prevStates:
                # print "nextState " + str(nextState)
                # print prevStates
                # print prevStates[nextState]
                prevStates[nextState].add(state)
              else:
                # When you compute predecessors of a state, make sure to store them in a set, not a list, to avoid duplicates.
                prevStates[nextState] = {state}

        # Initialize an empty priority queue. Please use util.PriorityQueue in your implementation. 
        priorityQ = util.PriorityQueue()

        # For each non-terminal state s, do:
        # (note: to make the autograder work for this question, you must iterate over states in the order returned by self.mdp.getStates())
          # Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this max Q-value represents what the value of s should be); call this number diff. Do NOT update self.values[s] in this step.
          # Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
          #computeQValueFromValues(self, state, action):
        
        for state in mdpStatesNonTerminal:
          highestQVal = float("-inf")
          stateValue = self.values[state]
          possibleActions = self.mdp.getPossibleActions(state)   
          for action in self.mdp.getPossibleActions(state):
            highestQVal = max(highestQVal, self.computeQValueFromValues(state, action))
            #print highestQVal
          diff = abs(highestQVal - stateValue)
          #print "diff = " + str(diff) 
          priorityQ.update(state, -diff)

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for i in range(self.iterations):
          # If the priority queue is empty, then terminate.
          if priorityQ.isEmpty():
            break
          else:
            # Pop a state s off the priority queue.
            prevState = priorityQ.pop()
            if not self.mdp.isTerminal(prevState):
              prevHighestQVal = float("-inf")
              #self.values[state] = highestQVal
              for action in self.mdp.getPossibleActions(prevState):
                prevHighestQVal = max(prevHighestQVal,self.computeQValueFromValues(prevState, action))
              # Update s's value (if it is not a terminal state) in self.values.
              self.values[prevState] = prevHighestQVal

            if prevStates[prevState]:
              # For each predecessor p of s, do:
              for predecessor in prevStates[prevState]:
                if not self.mdp.isTerminal(predecessor):
                  prevHighestQVal = float("-inf")
                  prevValue = self.values[predecessor]
                  possibleActions = self.mdp.getPossibleActions(predecessor)
                  for action in possibleActions: 
                    prevHighestQVal = max(prevHighestQVal, self.computeQValueFromValues(predecessor, action))
                    #print prevHighestQVal

                  # Find the absolute value of the difference between the current value of p in self.values 
                  #and the highest Q-value across all possible actions from p (this max Q-value represents 
                  #what the value of p should be); call this number diff. Do NOT update self.values[p] in this step.
                  diff = abs(prevValue - prevHighestQVal)
                  #print "diff = " + str(diff) 

                  #If diff > theta, push p into the priority queue with priority -diff 
                  #(note that this is negative), as long as it does not already exist in the 
                  #priority queue with equal or lower priority. As before, we use a negative
                  # because the priority queue is a min heap, but we want to prioritize updating 
                  #states that have a higher error.

                  if diff > self.theta:
                    priorityQ.update(predecessor, -diff)















