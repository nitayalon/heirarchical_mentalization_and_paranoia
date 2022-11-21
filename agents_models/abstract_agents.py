from abc import ABC, abstractmethod
import numpy as np


class SubIntentionalModel(ABC):

    def __init__(self, actions, history, threshold: float, softmax_temp: float, endowment=1.0):
        self.history = history
        self.actions = actions
        self.endowment = endowment
        self.threshold = threshold
        self.softmax_temp = softmax_temp
        self.rewards = []

    def softmax_transformation(self, q_values):
        softmax_transformation = np.exp(q_values / self.softmax_temp)
        return softmax_transformation/softmax_transformation.sum()

    def utility_function(self, action):
        return action - self.threshold

    @abstractmethod
    def act(self, seed, action=None, observation=None):
        pass

    @abstractmethod
    def forward(self, action=None, observation=None):
        pass


class History:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.history = []

    def reset(self, length):
        self.history = self.history[0:length]
        self.actions = self.actions[0:(length-1)]
        self.observations = self.observations[0:(length-1)]

    def get_last_observation(self):
        last_observation = self.observations[len(self.observations)-2]
        return last_observation

    def length(self):
        return len(self.history)

    def update_history(self, action, observation):
        self.update_actions(action)
        self.update_observations(observation)

    def update_actions(self, action):
        self.actions.append(action)
        self.history.append(action)

    def update_observations(self, observation):
        self.observations.append(observation)
        self.history.append(observation)


class BeliefDistribution(ABC):
    """
    Samples root particles from the current history
    """

    def __init__(self, prior_belief, opponent_model):
        self.opponent_model = opponent_model
        self.prior_belief = prior_belief
        self.belief = self.prior_belief
        self.history = History()

    def reset_prior(self):
        self.belief = self.prior_belief

    @abstractmethod
    def update_distribution(self, action, observation, first_move):
        pass

    @abstractmethod
    def sample(self, rng_key, n_samples):
        pass

    def get_current_belief(self):
        return self.belief[:, -1]

    @abstractmethod
    def update_history(self, action, observation):
        pass


class DoMZeroModel(SubIntentionalModel):

    def __init__(self, actions, history, threshold: float, softmax_temp: float,
                 prior_belief: np.array,
                 opponent_model: SubIntentionalModel):
        super().__init__(actions, history, threshold, softmax_temp)
        self.opponent_model = opponent_model
        self.belief = BeliefDistribution(prior_belief, opponent_model)  # type: BeliefDistribution

    def act(self, seed, action=None, observation=None):
        pass

    def forward(self, action=None, observation=None):
        pass
