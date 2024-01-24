from dom_one_agent import *
from typing import Optional


class DoMTwoPlayer:

    def __init__(self, game_duration: int, softmax_temperature: float, discount_factor: float,
                 prior_zero_beliefs: np.array, aleph_ipomdp: bool, delta: float) -> None:
        self.game_duration = game_duration
        self.softmax_temperature = softmax_temperature
        self.discount_factor = discount_factor
        self.aleph_ipomdp = aleph_ipomdp
        self.opponent = DoMOnePlayer(game_duration, softmax_temperature, discount_factor, aleph_ipomdp, delta)
        self.beliefs = [prior_zero_beliefs]
        self.nested_nested_beliefs = [prior_zero_beliefs]
        self.actions = self.opponent.observations = []
        self.opponent_policies = self.opponent_policy_helper_method(0, None)
        self.dom_level = 2

    def opponent_policy_helper_method(self, iteration, observation: Optional[int] = None):
        if iteration > 0:
            dom_zero_beliefs = self.opponent.opponent.irl(self.nested_nested_beliefs[-1], observation, iteration)
            self.nested_nested_beliefs.append(dom_zero_beliefs)
            self.opponent.nested_beliefs = self.nested_nested_beliefs
        else:
            dom_zero_beliefs = self.nested_nested_beliefs[-1]
        uninformed_payoff = np.array([1/2, 1/2])
        p_1 = self.opponent.act(dom_zero_beliefs, game_1, iteration, False)
        p_2 = self.opponent.act(dom_zero_beliefs, game_2, iteration, False, False)
        self.opponent_policies = np.array([uninformed_payoff, p_1, p_2])
        return self.opponent_policies

    def irl(self, prior: np.array, observation: int, iteration: int):
        opponent_policies = self.opponent_policy_helper_method(iteration, observation)
        likelihood = opponent_policies[:, observation]
        unnormalized_posterior = prior * likelihood
        normalized_posterior = unnormalized_posterior / np.sum(unnormalized_posterior)
        return normalized_posterior

    def act(self, belief: np.array) -> np.array:
        u_1 = -1 * np.matmul(belief, self.opponent.opponent.dom_zero_utility_subroutine(0, self.opponent_policies))
        u_2 = -1 * np.matmul(belief, self.opponent.opponent.dom_zero_utility_subroutine(1, self.opponent_policies))
        u_3 = -1 * np.matmul(belief, self.opponent.opponent.dom_zero_utility_subroutine(2, self.opponent_policies))
        return softmax_transformation(np.array([u_1, u_2, u_3]), self.softmax_temperature)
