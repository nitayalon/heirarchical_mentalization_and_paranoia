import numpy as np

from agents_models.intentional_agents.tom_zero_agents.tom_zero_sender import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_receiver import *
from typing import Optional, Union


class DoMOneBelief(DoMZeroBelief):
    def __init__(self, zero_level_belief, opponent_model: Optional[Union[DoMZeroSender, SubIntentionalAgent]],
                 history: History):
        super().__init__(zero_level_belief, opponent_model, history)
        self.nested_belief = opponent_model.belief.belief_distribution
        # Because there's no observation uncertainty the DoM(1) belief about the DoM(0) belief is its belief
        self.prior_belief = opponent_model.belief.belief_distribution
        self.belief_distribution = self.prior_belief

    def update_distribution(self, action, observation, iteration_number):
        """
        Update the belief based on the last action and observation (IRL)
        :param action:
        :param observation:
        :param iteration_number:
        :return:
        """
        if iteration_number <= 1:
            return None
        prior = np.copy(self.belief_distribution[:, -1])
        # Compute P(observation|action, history)
        likelihood = self.compute_likelihood(action, observation, prior, iteration_number)
        posterior = likelihood * prior
        self.belief_distribution = np.c_[self.belief_distribution, posterior / posterior.sum()]
        # Store nested belief
        self.nested_belief = self.opponent_model.belief.belief_distribution

    def compute_likelihood(self, action: Action, observation: Action, prior, iteration_number=None):
        """
        Compute observation likelihood given opponent's type and last action
        :param iteration_number:
        :param action:
        :param observation:
        :param prior:
        :return:
        """
        last_observation = self.history.get_last_observation()
        offer_likelihood = np.empty_like(prior)
        original_threshold = self.opponent_model.threshold
        # update nested belief
        self.opponent_model.belief.update_distribution(last_observation, action, iteration_number-1)
        for i in range(len(self.prior_belief[:, 0])):
            theta = self.prior_belief[:, 0][i]
            if theta == 0.0:
                offer_likelihood[i] = 1 / len(self.opponent_model.potential_actions)
                continue
            self.opponent_model.threshold = theta
            actions, q_values, softmax_transformation, mcts_tree = \
                self.opponent_model.forward(last_observation, action, iteration_number-1, False)
            # If the observation is not in the feasible action set then it singles theta hat:
            observation_probability = softmax_transformation[np.where(self.opponent_model.potential_actions == observation.value)]
            offer_likelihood[i] = observation_probability
        self.opponent_model.threshold = original_threshold
        return offer_likelihood

    def sample(self, rng_key, n_samples):
        """
        Sample nested beliefs
        :param rng_key:
        :param n_samples:
        :return:
        """
        probabilities = 1.0
        rng_generator = np.random.default_rng(rng_key)
        particles = rng_generator.choice(self.belief_distribution, size=n_samples, p=probabilities)
        return self.opponent_model.belief.belief_distribution


class DoMOneEnvironmentModel(DoMZeroEnvironmentModel):
    def __init__(self, opponent_model: DoMZeroSender, reward_function, belief_distribution: DoMOneBelief):
        super().__init__(opponent_model, reward_function, belief_distribution)

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs):
        self.opponent_model.threshold = persona
        if action_length == 0 and observation_length == 0:
            return None
        self.opponent_model.reset(self.high, self.low, observation_length, action_length, False)
        self.opponent_model.belief.belief_distribution = nested_beliefs


class DoMOneSenderEnvironmentModel(DoMOneEnvironmentModel):
    def __init__(self, opponent_model: DoMZeroSender, reward_function, belief_distribution: DoMOneBelief):
        super().__init__(opponent_model, reward_function, belief_distribution)

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs):
        self.opponent_model.threshold = persona
        self.opponent_model.reset(self.high, self.low, observation_length, action_length, False)
        self.opponent_model.belief.belief_distribution = nested_beliefs


class DoMOneReceiver(DoMZeroReceiver):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: Optional[Union[DoMZeroSender, SubIntentionalAgent]],
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.environment_model = DoMOneEnvironmentModel(self.opponent_model, self.utility_function, self.belief)
        self.belief = DoMOneBelief(prior_belief, self.opponent_model, self.history)
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)
        self.name = "DoM(1)_receiver"


class DoMOneSender(DoMZeroSender):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: Optional[Union[DoMZeroReceiver, SubIntentionalAgent]],
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.environment_model = DoMOneSenderEnvironmentModel(self.opponent_model, self.utility_function, self.belief)
        self.belief = DoMOneBelief(None, self.opponent_model, self.history)
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)
        self.name = "DoM(1)_sender"


