import collections
from hashlib import new
from sre_constants import NOT_LITERAL
import util
import math
import random
from collections import defaultdict
from util import ValueIteration


############################################################
# Problem 1a: BlackjackMDP

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        # total, next card (if any), multiplicity for each card
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None.
    # When the probability is 0 for a particular transition, don't include that
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        (totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts) = state

        if deckCardCounts == None:
            return []

        elif action == 'Take' and nextCardIndexIfPeeked == None:
            newStates = []
            newRewards = []
            for i in range(len(deckCardCounts)):
                totalCardValue = totalCardValueInHand
                newReward = 0
                if deckCardCounts[i] > 0:
                    totalCardValue += self.cardValues[i]
                    newReward = 0

                    if totalCardValue <= self.threshold:
                        newDeck = list(deckCardCounts)
                        newDeck[i] -= 1

                        newState = (totalCardValue, None, tuple(newDeck)) if sum(
                            newDeck) > 0 else (totalCardValue, None, None)
                        newReward = 0 if sum(
                            newDeck) > 0 else totalCardValue

                    else:
                        newState = (totalCardValue, None, None)
                        newReward = 0

                    newStates.append(newState)
                    newRewards.append(newReward)

            return [(newState, 1.0/float(len(newStates)), newRewards[i]) for i, newState in enumerate(newStates)]

        elif action == 'Take' and nextCardIndexIfPeeked != None:
            totalCardValue = totalCardValueInHand + \
                self.cardValues[nextCardIndexIfPeeked]

            if totalCardValueInHand <= self.threshold:
                newDeck = list(deckCardCounts)
                newDeck[nextCardIndexIfPeeked] -= 1

                newState = (totalCardValue, None, tuple(newDeck)) if sum(
                    newDeck) > 0 else (totalCardValue, None, None)
                newReward = 0 if sum(
                    newDeck) > 0 else totalCardValue

            else:
                newState = (totalCardValue, None, None)
                newReward = 0

            return [(newState, 1, newReward)]

        elif action == 'Peek' and nextCardIndexIfPeeked == None:
            newStates = []
            for i in range(len(deckCardCounts)):
                if deckCardCounts[i] > 0:
                    newStates.append((totalCardValueInHand, i, deckCardCounts))
            return [(newState, 1.0/float(len(newStates)), -self.peekCost) for newState in newStates]

        elif action == 'Peek' and nextCardIndexIfPeeked != None:
            return []

        elif action == 'Quit':
            return [((totalCardValueInHand, None, None), 1, totalCardValueInHand if totalCardValueInHand <= self.threshold else 0)]

        else:
            return []

        # END_YOUR_ANSWER

    def discount(self):
        return 1

############################################################
# Problem 1b: ValueIterationDP


class ValueIterationDP(ValueIteration):
    '''
    Solve the MDP using value iteration with dynamic programming.
    '''

    def solve(self, mdp):
        V = {}  # state -> value of state

        # BEGIN_YOUR_ANSWER (our solution is 13 lines of code, but don't worry if you deviate from this)
        def execVI(currState):
            temps = []
            for action in mdp.actions(currState):
                temp = 0
                for (newState, newProb, newReward) in mdp.succAndProbReward(currState, action):
                    if newState not in V:
                        V[newState] = execVI(newState)

                    temp += (newReward + V[newState]) * newProb

                temps.append(temp)
            V[currState] = max(temps)
            return V[currState]

        execVI(mdp.startState())
        # END_YOUR_ANSWER

        # Compute the optimal policy now
        pi = self.computeOptimalPolicy(mdp, V)
        self.pi = pi
        self.V = V

############################################################
# Problem 2a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action


class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        currQ = self.getQ(state, action)

        Qs = [0]
        newQ = reward
        for newAction in self.actions(newState):
            Qs.append(self.getQ(newState, newAction))
        newQ += self.discount * max(Qs)

        alpha = self.getStepSize()

        for (key, value) in self.featureExtractor(state, action):
            self.weights[key] += alpha * (newQ - currQ) * value
        # END_YOUR_ANSWER

############################################################
# Problem 2b: Q SARSA


class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2]
        else:
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        currQ = self.getQ(state, action)
        newQ = reward + self.discount * self.getQ(newState, newAction)
        alpha = self.getStepSize()

        for (key, value) in self.featureExtractor(state, action):
            self.weights[key] += alpha * (newQ - currQ) * value
        # END_YOUR_ANSWER

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.


def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 2c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None


def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
    featureValue = 1

    result = [((total, action), featureValue)]

    if(counts != None):
        temp = []
        for idx, val in enumerate(counts):
            if val > 0:
                temp.append(1)
            else:
                temp.append(0)
        result.append(((tuple(temp), action), featureValue))

        for idx, val in enumerate(counts):
            result.append(((idx, val, action), featureValue))

    return result
    # END_YOUR_ANSWERs
