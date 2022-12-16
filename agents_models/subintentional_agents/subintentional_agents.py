from agents_models.abstract_agents import *
import numpy as np


class RandomSubIntentionalModel(SubIntentionalModel):

    def forward(self, action=None, observation=None):
        pass

    def act(self, seed, action=None, observation=None):
        random_number_generator = np.random.default_rng(seed)
        offer = random_number_generator.choice(self.actions)
        return offer


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
        return None


class IntentionalAgentSubIntentionalModel(SubIntentionalModel):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self.high = [1.0]
        self.low = [0.0]
        self.belief = SubIntentionalBelief()

    def act(self, seed, action=None, observation=None) -> float:
        relevant_actions, q_values, probabilities = self.forward(action, observation)
        random_number_generator = np.random.default_rng(seed)
        optimal_offer = random_number_generator.choice(relevant_actions, p=probabilities)
        return optimal_offer

    def forward(self, action=None, observation=None):
        self.update_bounds(action, observation)
        upper_bound = self.high[-1]
        lower_bound = self.low[-1]
        if upper_bound <= self.threshold:
            relevant_actions = self.actions[np.where(np.logical_and(self.actions >= lower_bound, self.actions <= self.threshold))]
        else:
            relevant_actions = self.actions[np.where(np.logical_and(self.actions >= lower_bound, self.actions < upper_bound))]
        q_values = self.utility_function(relevant_actions, observation)
        probabilities = self.softmax_transformation(q_values)
        return relevant_actions, q_values, probabilities

    def update_bounds(self, action, observation):
        if action is None or observation is None:
            return None
        # If the subject accepted the offer the lower bound is updated
        if observation:
            self.low.append(action)
        # If the offer is rejected the upper bound is updated
        else:
            self.high.append(action)











