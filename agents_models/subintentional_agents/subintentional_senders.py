from agents_models.abstract_agents import *
import numpy as np


class RandomSubIntentionalSender(SubIntentionalAgent):

    def update_nested_models(self, action=None, observation=None, iteration_number=None):
        pass

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self.name = "DoM(-1)_RA"
        self.low = None
        self.high = None

    def utility_function(self, action, observation):
        reward = (1 - action)
        reward[reward < self.threshold] = 0.0
        return reward

    def random_forward(self):
        q_values = self.potential_actions
        probabilities = np.repeat(1 / len(self.potential_actions), len(self.potential_actions))
        return self.potential_actions, q_values, probabilities

    def forward(self, action: Action, observation: Action, iteration_number=None):
        q_values = np.repeat(1 / len(self.potential_actions), len(self.potential_actions))
        probabilities = np.repeat(1 / len(self.potential_actions), len(self.potential_actions))
        return self.potential_actions, q_values, probabilities

    def update_seed(self, seed, number):
        return seed + number

    def update_bounds(self, action: Action, observation: Action, iteration_number):
        if action.value is None or observation.value is None:
            self.low = 0.0
            self.high = 1.0
            return None
        low = action.value * (1-observation.value) + observation.value * self.low
        high = action.value * observation.value + (1-observation.value) * self.high
        # If the opponent plays tricks with us
        if high < low:
            temp = high
            high = low
            low = temp
        self.low = low
        self.high = high


class SoftMaxRationalRandomSubIntentionalSender(RandomSubIntentionalSender):

    def __init__(self, actions, softmax_temp: float, penalty: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self._name = "DoM(-1)_RRA"
        self.penalty = penalty

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, threshold):
        if threshold == 0.0:
            self._name = "DoM(-1)_Random"
        else:
            self._name = "DoM(-1)_Rational"

    def _compute_weights(self, offers, iteration_number, low_bound, up_bound):
        w = np.logical_not(np.logical_and(low_bound < offers, offers <= up_bound))
        # If the low and high are the same:
        if low_bound == up_bound:
            w = np.logical_not(np.logical_and(low_bound <= offers, offers <= up_bound))
        # If both bounds are equal to the threshold we do not penalize it
        if low_bound >= np.round(1 - self.threshold, 3) or iteration_number == 0:
            w = np.logical_not(np.logical_and(low_bound <= offers, offers <= up_bound))
        w_prime = self.penalty * w
        return w_prime

    def rational_forward(self, iteration_number: int,low: Optional[float] = None, high: Optional[float] = None):
        """
        This method computes an interval of positive reward offers and returns a uniform distribution over them

        :return:
        """
        high = self.high if high is None else high
        low = min(self.low if low is None else low, 1 - self.threshold)
        upper_bound = np.round(high, 3)
        lower_bound = np.round(low, 3)
        weights = self._compute_weights(self.potential_actions, iteration_number, lower_bound, upper_bound)
        q_values, probabilities = self._compute_q_values_and_probabilities(self.potential_actions, weights)
        return self.potential_actions, q_values, probabilities

    def forward(self, action: Action, observation: Action, iteration_number=None, *args):
        # Random agents act fully random
        if self.threshold == 0.0:
            potential_actions, q_values, probabilities = self.random_forward()
        # Rational random use different policies
        else:
            self.update_bounds(action, observation, iteration_number)
            if len(args) == 0:
                potential_actions, q_values, probabilities = self.rational_forward(iteration_number)
            else:
                potential_actions, q_values, probabilities = self.rational_forward(iteration_number,
                                                                                   args[0][0], args[0][1])
        return potential_actions, q_values, probabilities

    def _compute_q_values_and_probabilities(self, relevant_actions, weights):
        q_values = self.utility_function(relevant_actions, Action(None, False)) + weights
        probabilities = self.softmax_transformation(q_values)
        return q_values, probabilities

    def update_seed(self, seed, number):
        if self.threshold > 0:
            return seed
        return seed + number


class UniformRationalRandomSubIntentionalSender(SoftMaxRationalRandomSubIntentionalSender):

    def _compute_q_values_and_probabilities(self, filtered_actions, weights):
        q_values = self.utility_function(filtered_actions, None)
        relevant_actions = filtered_actions[np.where(q_values >= 0)]
        probabilities = np.repeat(1 / len(relevant_actions), len(relevant_actions))
        probabilities = np.pad(probabilities, (filtered_actions.size - probabilities.size,0), 'constant',
                               constant_values=0)
        return q_values, probabilities

