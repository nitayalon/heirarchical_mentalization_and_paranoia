from agents_models.abstract_agents import *
import numpy as np


class RandomSubIntentionalSender(SubIntentionalAgent):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self.name = "DoM(-1)_RA"

    def utility_function(self, action, observation):
        return (1 - action) - self.threshold

    def random_forward(self, action: Action, observation: Action):
        q_values = self.potential_actions
        probabilities = np.repeat(1 / len(self.potential_actions), len(self.potential_actions))
        return self.potential_actions, q_values, probabilities

    def forward(self, action: Action, observation: Action, iteration_number=None):
        q_values = self.potential_actions
        probabilities = np.repeat(1 / len(self.potential_actions), len(self.potential_actions))
        return self.potential_actions, q_values, probabilities

    def update_seed(self, seed, number):
        return seed + number

    def update_bounds(self, action: Action, observation: Action):
        if action.value is None or observation.value is None:
            return None
        # If the subject accepted the offer the upper bound is updated
        if observation.value:
            self.high = min(action.value, 1.0-self.threshold)
        # If the offer is rejected the upper bound is updated
        else:
            self.low = action.value
        # If the opponent plays tricks with us
        if self.high < self.low:
            temp = self.high
            self.high = self.low
            self.low = temp


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

    def _compute_weights(self, offers, low_bound, up_bound):
        w = np.logical_not(np.logical_and(low_bound < offers, offers <= up_bound))
        w_prime = self.penalty * w
        return w_prime

    def rational_forward(self, action: Action, observation: Action):
        """
        This method computes an interval of positive reward offers and returns a uniform distribution over them
        :param action:
        :param observation:
        :return:
        """
        upper_bound = np.round(self.high, 3)
        lower_bound = np.round(self.low, 3)
        weights = self._compute_weights(self.potential_actions, lower_bound, upper_bound)
        q_values, probabilities = self._compute_q_values_and_probabilities(self.potential_actions, weights)
        return self.potential_actions, q_values, probabilities

    def forward(self, action: Action, observation: Action, iteration_number=None):
        self.update_bounds(action, observation)
        # Random agents act fully random
        if self.threshold == 0.0:
            potential_actions, q_values, probabilities = self.random_forward(action, observation)
        # Rational random use different policies
        else:
            potential_actions, q_values, probabilities = self.rational_forward(action, observation)
        return potential_actions, q_values, probabilities

    def _compute_q_values_and_probabilities(self, relevant_actions, weights):
        q_values = self.utility_function(relevant_actions, Action(None, False)) + weights
        probabilities = self.softmax_transformation(q_values)
        return q_values, probabilities


class UniformRationalRandomSubIntentionalSender(SoftMaxRationalRandomSubIntentionalSender):

    def _compute_q_values_and_probabilities(self, filtered_actions, weights):
        q_values = self.utility_function(filtered_actions, None)
        relevant_actions = filtered_actions[np.where(q_values >= 0)]
        probabilities = np.repeat(1 / len(relevant_actions), len(relevant_actions))
        probabilities = np.pad(probabilities, (filtered_actions.size - probabilities.size,0), 'constant',
                               constant_values=0)
        return q_values, probabilities

