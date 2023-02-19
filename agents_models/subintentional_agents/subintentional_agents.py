from agents_models.abstract_agents import *
import numpy as np


class RandomSubIntentionalModel(SubIntentionalModel):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self.belief = SubIntentionalBelief()
        self.name = "DoM(-1)_RA"

    def utility_function(self, action, observation):
        return action - self.threshold

    def forward(self, action=None, observation=None):
        q_values = self.potential_actions * 0.5
        probabilities = np.repeat(1 / len(self.potential_actions),len(self.potential_actions))
        return self.potential_actions, q_values, probabilities


class SubIntentionalBelief(BeliefDistribution):

    def __init__(self):
        super().__init__(None, None)

    def get_current_belief(self):
        return None

    def update_distribution(self, action, observation, first_move):
        return None

    def sample(self, rng_key, n_samples):
        return None

    def update_history(self, action, observation):
        self.history.update_actions(action)
        self.history.update_observations(observation)


class IntentionalAgentSubIntentionalModel(RandomSubIntentionalModel):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self.belief = SubIntentionalBelief()
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

    def _random_forward(self):
        q_values = self.potential_actions * 0.5
        probabilities = np.repeat(1 / len(self.potential_actions), len(self.potential_actions))
        return self.potential_actions, q_values, probabilities

    def forward(self, action=None, observation=None):
        if self.threshold == 0.0:
            relevant_actions, q_values, probabilities = self._random_forward()
        else:
            upper_bound = np.round(self.high, 3)
            lower_bound = np.round(self.low, 3)
            if lower_bound >= upper_bound:
                lower_bound = np.round(upper_bound - 0.1, 3)
            if upper_bound <= self.threshold:
                relevant_actions = self.potential_actions[np.where(np.logical_and(self.potential_actions >= lower_bound, self.potential_actions <= self.threshold))]
            else:
                relevant_actions = self.potential_actions[np.where(np.logical_and(self.potential_actions >= lower_bound, self.potential_actions < upper_bound))]
            q_values = self.utility_function(relevant_actions, observation)
            probabilities = self.softmax_transformation(q_values)
        return relevant_actions, q_values, probabilities

    def update_bounds(self, action, observation):
        if action is None or observation is None:
            return None
        # If the subject accepted the offer the lower bound is updated
        if observation:
            self.low = action
        # If the offer is rejected the upper bound is updated
        else:
            self.high = action











