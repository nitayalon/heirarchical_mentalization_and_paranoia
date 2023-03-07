from agents_models.abstract_agents import *
import numpy as np
from scipy.stats import norm


class RandomBasicModel(BasicModel):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self.name = "DoM(-1)_RA"

    def utility_function(self, action, observation):
        return action - self.threshold

    def forward(self, action: Action, observation: Action):
        q_values = self.potential_actions * 0.5
        probabilities = np.repeat(1 / len(self.potential_actions), len(self.potential_actions))
        return self.potential_actions, q_values, probabilities

    def update_seed(self, seed, number):
        return seed + number


class IntentionalAgentSubIntentionalModel(RandomBasicModel):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self._name = "DoM(-1)_IA"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, threshold):
        if threshold == 0.0:
            self._name = "DoM(-1)_RA"
        else:
            self._name = "DoM(-1)_IA"

    def _random_forward(self, action, observation):
        q_values = self.potential_actions * 0.5
        probabilities = np.repeat(1 / len(self.potential_actions), len(self.potential_actions))
        return self.potential_actions, q_values, probabilities

    def intentional_forward(self, action: Action, observation: Action):
        upper_bound = np.round(self.high, 3)
        lower_bound = np.round(self.low, 3)
        if lower_bound >= upper_bound:
            lower_bound = np.round(upper_bound - 0.1, 3)
        if upper_bound <= self.threshold:
            relevant_actions = self.potential_actions[np.where(np.logical_and(self.potential_actions >= lower_bound,
                                                                              self.potential_actions <= self.threshold))]
        else:
            relevant_actions = self.potential_actions[np.where(np.logical_and(self.potential_actions >= lower_bound,
                                                                              self.potential_actions < upper_bound))]
        q_values = self.utility_function(relevant_actions, observation.value)
        probabilities = self.softmax_transformation(q_values)
        return relevant_actions, q_values, probabilities

    def forward(self, action: Action, observation: Action):
        if self.threshold == 0.0:
            relevant_actions, q_values, probabilities = self._random_forward(action, observation)
        else:
            relevant_actions, q_values, probabilities = self.intentional_forward(action, observation)
        return relevant_actions, q_values, probabilities

    def update_bounds(self, action: Action, observation: Action):
        if action.value is None or observation.value is None:
            return None
        # If the subject accepted the offer the lower bound is updated
        if observation.value:
            self.low = action.value
        # If the offer is rejected the upper bound is updated
        else:
            self.high = action.value


class SemiRandomAgent(IntentionalAgentSubIntentionalModel):
    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self.name = "DoM(-1)_SRA"

    def _random_forward(self, action: Action, observation: Action):
        upper_bound = np.round(self.high, 3)
        lower_bound = np.round(self.low, 3)
        if lower_bound >= upper_bound:
            lower_bound = np.round(upper_bound - 0.1, 3)
        if upper_bound <= self.threshold:
            relevant_actions = self.potential_actions[np.where(np.logical_and(self.potential_actions >= lower_bound,
                                                                              self.potential_actions <= self.threshold))]
        else:
            relevant_actions = self.potential_actions[np.where(np.logical_and(self.potential_actions >= lower_bound,
                                                                              self.potential_actions < upper_bound))]
        q_values = relevant_actions
        if action.value is None:
            action = Action(0.5, False)
        probabilities = norm.pdf(relevant_actions, loc=action.value, scale=action.value/10)
        probabilities = probabilities / probabilities.sum()
        return relevant_actions, q_values, probabilities









