# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # initialize q values to be an empty dictionary (key: state, value: action)
        self.qValues = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # if the state is seen, return the Q node value else return 0.0
        if (state, action) in self.qValues:
            return self.qValues[(state, action)]
        else:
            return 0.0
        # util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # get list of all legal actions available
        actions = self.getLegalActions(state)
        # set value to be an arbitrarily large negative number
        maxAction = -99999
        # if there are legal actions available, i.e. the state is not terminal, find the value from Q value
        if len(actions) > 0:
            # iterate over the actions to find max action
            for action in actions:
                # use getQValue() defined above to get q value and update the max action based on the maximum value
                qValue = self.getQValue(state, action)
                maxAction = max(maxAction, qValue)
        else:
            # set max action to 0.0 is its a terminal state / there are no legal actions available
            maxAction = 0.0

        return maxAction
        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # get the list of all possible actions
        actions = self.getLegalActions(state)
        # initialize best action to be empty list
        bestActions = []
        #  while there are actions to perform, i.e. the state is not terminal
        if len(actions) > 0:
            # # iterate over the list of all available actions to
            for action in actions:
                # compare the qValue with the maxAction qValue, and if qValue = maxAction qValue,
                # add that particular action to the bestAction list
                if self.getQValue(state, action) == self.computeValueFromQValues(state):
                    bestActions.append(action)
            # used to break ties randomly for better behavior. Suggested in the question prompt
            return random.choice(bestActions)
        else:
            # return None if no actions available / its a terminal state
            return None
        # util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # if there are legal actions available
        if len(legalActions) > 0:
            # if there is a probability of epsilon happening a fraction of the time
            # set action as a random action from the list of random actions
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            # if there isn't a probability of epsilon, then set action to be the best action
            else:
                action = self.computeActionFromQValues(state)
        # redundant code since action is anyway initialized to None
        # else:
        #     action = None
        return action
        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # From slides: Update V(s) each time we experience a transition (s, a, s’, r)
        # Based on formula -
        # sample = R(s, pi(s), s') + discount factor(V_pi (s'))
        # V_pi(s) <- (1 - alpha)V_pi(s) + alpha * sample
        s = reward + (self.discount * self.getValue(nextState))
        a = self.alpha
        self.qValues[(state, action)] = ((1 - a) * self.getQValue(state, action)) + (a * s)
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # initialize q value to be 0
        qValue = 0
        # hint from question prompt. Get the feature vector from the function in featExtractor.py
        features = self.featExtractor.getFeatures(state, action)
        # iterate over every feature in the vector to update the q value
        for feature in features.keys():
            # based on formula : Q(s, a) = sigma(1 -> n)feature_i(s,a) * w_i
            qValue += features[feature] * self.weights[feature]
        return qValue
        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Based on formula:
        # w <- w_i + a * difference * feature_i(s, a), where
        # difference = (R(s, pi(s), s') + discount factor * max Q(s', a')) - Q(s, a)
        # V_pi(s) <- (1 - alpha)V_pi(s) + alpha * sample
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        # Get the feature vector from the function in featExtractor.py
        features = self.featExtractor.getFeatures(state, action)
        # iterate over every feature in the vector to update the weights
        for feature in features.keys():
            # based on formula above
            self.weights[feature] += (self.alpha * difference * features[feature])
        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)           # for debugging
