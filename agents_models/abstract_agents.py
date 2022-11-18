from abc import ABC, abstractmethod
import numpy as np


class SubIntentionalAgent(ABC):

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


class DoMZeroAgent(SubIntentionalAgent):
    def __init__(self, actions, history, threshold: float, softmax_temp: float, belief_distribution: np.array,
                 opponent_model: SubIntentionalAgent):
        super().__init__(actions, history, threshold, softmax_temp)
        self.opponent_model = opponent_model
        self.belief = belief_distribution

    @abstractmethod
    def update_belief(self, action, observation):
        pass
