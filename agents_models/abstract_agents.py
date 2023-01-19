import numpy as np
from IPOMCP_solver.Solver.abstract_classes import *
from typing import Optional


class SubIntentionalModel(ABC):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None, endowment=1.0):
        self.actions = actions
        self.endowment = endowment
        self.threshold = threshold
        self.history = []
        self.softmax_temp = softmax_temp
        self.rewards = []
        self.high = 1.0
        self.low = 0.0
        self.name = None

    def softmax_transformation(self, q_values):
        softmax_transformation = np.exp(q_values / self.softmax_temp)
        return softmax_transformation / softmax_transformation.sum()

    def utility_function(self, action, observation):
        return action - self.threshold

    @abstractmethod
    def act(self, seed, action=None, observation=None) -> [float, np.array]:
        pass

    @abstractmethod
    def forward(self, action=None, observation=None):
        pass

    def update_bounds(self, action, observation):
        pass


class DoMZeroBelief(BeliefDistribution):

    def __init__(self, prior_belief, opponent_model):
        super().__init__(prior_belief, opponent_model)
        self.opponent_belief = None

    def update_distribution(self, action, observation, first_move):
        pass

    def sample(self, rng_key, n_samples):
        pass

    def update_history(self, action, observation):
        pass


class DoMZeroModel(SubIntentionalModel):

    def __init__(self, actions,
                 softmax_temp: float,
                 prior_belief: np.array,
                 opponent_model: SubIntentionalModel,
                 alpha: float):
        super().__init__(actions, softmax_temp, None)
        self.alpha = alpha
        self.opponent_model = opponent_model
        self.belief = DoMZeroBelief(prior_belief, self.opponent_model)  # type: DoMZeroBelief

    def act(self, seed, action=None, observation=None, iteration_number=None) -> [float, np.array]:
        pass

    def forward(self, action=None, observation=None, iteration_number=None):
        pass
