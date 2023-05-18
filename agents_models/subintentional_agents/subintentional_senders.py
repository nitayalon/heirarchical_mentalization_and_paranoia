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
        return (1 - action) - self.threshold

    def random_forward(self):
        q_values = self.potential_actions
        probabilities = np.repeat(1 / len(self.potential_actions), len(self.potential_actions))
        return self.potential_actions, q_values, probabilities

    def forward(self, action: Action, observation: Action, iteration_number=None):
        q_values = self.potential_actions
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
        # # If the subject accepted the offer the upper bound is updated
        # high = self.upper_bounds[iteration_number-1]
        # low = self.lower_bounds[iteration_number-1]
        # # Protection against missing data:
        # if low is None:
        #     low = list(filter(lambda entry: entry is not None, self.lower_bounds))[-1]
        # if high is None:
        #     high = list(filter(lambda entry: entry is not None, self.upper_bounds))[-1]
        # if observation.value:
        #     high = min(action.value, 1.0)
        # # If the offer is rejected the upper bound is updated
        # else:
        #     low = max(action.value, low)
        # If the opponent plays tricks with us
        if high < low:
            temp = high
            high = low
            low = temp
        # if iteration_number < self.config.task_duration:
        #     self.upper_bounds[iteration_number] = high
        #     self.lower_bounds[iteration_number] = low
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

    def _compute_weights(self, offers, low_bound, up_bound):
        if low_bound < up_bound:
            w = np.logical_not(np.logical_and(low_bound < offers, offers <= up_bound))
        # If both bounds are equal to the threshold we do not penalize it
        else:
            w = np.logical_not(np.logical_and(low_bound <= offers, offers <= up_bound))
        w_prime = self.penalty * w
        return w_prime

    def rational_forward(self, low: Optional[float] = None, high: Optional[float] = None):
        """
        This method computes an interval of positive reward offers and returns a uniform distribution over them

        :return:
        """
        high = self.high if high is None else high
        low = self.low if low is None else low
        upper_bound = np.round(high, 3)
        lower_bound = np.round(low, 3)
        weights = self._compute_weights(self.potential_actions, lower_bound, upper_bound)
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
                potential_actions, q_values, probabilities = self.rational_forward()
            else:
                potential_actions, q_values, probabilities = self.rational_forward(args[0][0], args[0][1])
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

