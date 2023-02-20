import numpy as np
from IPOMCP_solver.Solver.abstract_classes import *
from typing import Optional


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


class SubIntentionalModel(ABC):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        self.potential_actions = actions
        self._threshold = threshold
        self.softmax_temp = softmax_temp
        self.observations = []
        self.actions = []
        self.rewards = []
        self.high = 1.0
        self.low = 0.0
        self.name = None
        self.belief = SubIntentionalBelief()
        self._alpha = None

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, gamma):
        self._threshold = gamma

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    def softmax_transformation(self, q_values):
        softmax_transformation = np.exp(q_values / self.softmax_temp)
        return softmax_transformation / softmax_transformation.sum()

    def utility_function(self, action, observation):
        pass

    def act(self, seed, action=None, observation=None, iteration_number=None) -> [float, np.array]:
        self.update_bounds(action, observation)
        relevant_actions, q_values, probabilities = self.forward(action, observation)
        random_number_generator = np.random.default_rng(seed)
        optimal_offer = random_number_generator.choice(relevant_actions, p=probabilities)
        return optimal_offer, np.array([relevant_actions, q_values]).T

    @abstractmethod
    def forward(self, action=None, observation=None):
        pass

    def update_bounds(self, action, observation):
        pass

    def update_history(self, action, observation):
        self.actions.append(action)
        self.observations.append(observation)


class DoMZeroBelief(BeliefDistribution):

    def __init__(self, intentional_threshold_belief: np.array, opponent_model:SubIntentionalModel):
        """
        :param intentional_threshold_belief: np.array - represents the prior belief about the agent_parameters
        :param opponent_model:
        """
        super().__init__(intentional_threshold_belief, opponent_model)
        self.opponent_belief = None

    def compute_likelihood(self, action, observation, prior):
        pass

    def update_distribution(self, action, observation, first_move):
        """
        Update the belief based on the last action and observation (IRL)
        :param action:
        :param observation:
        :param first_move:
        :return:
        """
        if first_move < 1:
            return None
        prior = np.copy(self.belief_distribution[:, -1])
        probabilities = self.compute_likelihood(action.value, observation.value, prior)
        posterior = probabilities * prior
        self.belief_distribution = np.c_[self.belief_distribution, posterior / posterior.sum()]

    def sample(self, rng_key, n_samples):
        probabilities = self.belief_distribution[:, -1]
        rng_generator = np.random.default_rng(rng_key)
        particles = rng_generator.choice(self.belief_distribution[:, 0], size=n_samples, p=probabilities)
        return particles

    def reset_belief(self, history_length):
        self.belief_distribution = self.belief_distribution[:, 0:history_length+3]
        self.history.reset(history_length+2)

    def update_history(self, action, observation):
        """
        Method helper for history update - append the last action and observation
        :param action:
        :param observation:
        :return:
        """
        self.history.update_history(action, observation)
        self.opponent_model.belief.update_history(observation, action)


class DoMZeroModel(SubIntentionalModel):

    def __init__(self, actions,
                 softmax_temp: float,
                 prior_belief: np.array,
                 opponent_model: SubIntentionalModel):
        super().__init__(actions, softmax_temp, None)
        self.opponent_model = opponent_model
        self.belief = DoMZeroBelief(prior_belief, self.opponent_model)  # type: DoMZeroBelief

    def act(self, seed, action=None, observation=None, iteration_number=None) -> [float, np.array]:
        pass

    def forward(self, action=None, observation=None, iteration_number=None):
        pass
