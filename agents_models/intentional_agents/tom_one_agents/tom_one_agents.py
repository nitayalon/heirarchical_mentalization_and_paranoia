import numpy as np

from agents_models.intentional_agents.tom_zero_agents.tom_zero_sender import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_receiver import *
from agents_models.intentional_agents.tom_one_agents.dom_one_memoization import *
from typing import Optional, Union


class DoMOneBelief(DoMZeroBelief):
    def __init__(self, belief_distribution_support, zero_level_belief, include_persona_inference: bool,
                 opponent_model: Optional[Union[DoMZeroSender, SubIntentionalAgent]],
                 history: History):
        super().__init__(belief_distribution_support, zero_level_belief, opponent_model, history)
        self.nested_belief = opponent_model.belief.belief_distribution
        # Because there's no observation uncertainty the DoM(1) belief about the DoM(0) belief is its belief
        self.prior_belief = opponent_model.belief.belief_distribution
        self.belief_distribution = self.prior_belief
        self.include_persona_inference = include_persona_inference

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
        prior = np.copy(self.belief_distribution[-1, :])
        likelihood = self.compute_likelihood(action, observation, prior, iteration_number)
        if self.include_persona_inference:
            # Compute P(observation|action, history)
            posterior = likelihood * prior
            self.belief_distribution = np.vstack([self.belief_distribution, posterior / posterior.sum()])
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
        self.opponent_model.belief.update_distribution(last_observation, action, iteration_number)
        if self.include_persona_inference:
            for i in range(len(self.support)):
                theta = self.support[i]
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
        return None

    def sample(self, rng_key, n_samples):
        """
        Sample nested beliefs
        :param rng_key:
        :param n_samples:
        :return:
        """
        probabilities = 1.0
        rng_generator = np.random.default_rng(rng_key)
        idx = rng_generator.choice(self.belief_distribution.shape[0], size=n_samples, p=np.array([probabilities]))
        particles = self.opponent_model.belief.belief_distribution[idx, :]
        return np.repeat(0.0, n_samples)


class DoMOneEnvironmentModel(DoMZeroEnvironmentModel):
    def __init__(self, opponent_model: DoMZeroSender, reward_function,
                 actions: np.array,
                 belief_distribution: DoMOneBelief):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs):
        self.opponent_model.threshold = persona
        if action_length == 0 and observation_length == 0:
            return None
        self.opponent_model.reset(self.high, self.low, observation_length, action_length, False)
        self.opponent_model.belief.belief_distribution = nested_beliefs


class DoMOneSenderEnvironmentModel(DoMOneEnvironmentModel):
    def __init__(self, opponent_model: DoMZeroSender, reward_function, actions: np.array,
                 belief_distribution: DoMOneBelief):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs):
        self.opponent_model.threshold = persona
        self.opponent_model.reset(self.high, self.low, observation_length, action_length, False)
        self.opponent_model.belief.belief_distribution = nested_beliefs

    def step(self, interactive_state: InteractiveState, action: Action, observation: Action, seed: int,
             iteration_number: int):
        counter_offer, observation_probability, q_values, opponent_policy = self.opponent_model.act(seed, observation,
                                                                                                    action, iteration_number)
        reward = self.reward_function(action.value, observation.value, counter_offer.value) * observation_probability + \
                 self.reward_function(action.value, observation.value, not counter_offer.value) * (1-observation_probability)
        interactive_state.state.terminal = interactive_state.state.name == 10
        interactive_state.state.name = str(int(interactive_state.state.name) + 1)
        return interactive_state, counter_offer, reward, observation_probability


class DoMOneReceiver(DoMZeroReceiver):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: Optional[Union[DoMZeroSender, SubIntentionalAgent]],
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.environment_model = DoMOneEnvironmentModel(self.opponent_model, self.utility_function, self.belief)
        self.belief = DoMOneBelief(self.opponent_model.belief.support, self.opponent_model.belief.belief_distribution,
                                   True, self.opponent_model, self.history)
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)
        self.name = "DoM(1)_receiver"


class DoMOneSender(DoMZeroSender):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float],
                 memoization_table: DoMOneMemoization,
                 prior_belief: np.array,
                 opponent_model: Optional[Union[DoMZeroReceiver, SubIntentionalAgent]],
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.memoization_table = memoization_table
        self.belief = DoMOneBelief(self.opponent_model.belief.support, self.opponent_model.belief.belief_distribution,
                                   False, self.opponent_model, self.history)
        self.environment_model = DoMOneSenderEnvironmentModel(self.opponent_model, self.utility_function,
                                                              actions,
                                                              self.belief)
        self.exploration_policy = DoMZeroSenderExplorationPolicy(self.potential_actions, self.utility_function,
                                                                 self.config.get_from_env("rollout_rejecting_bonus"),
                                                                 self.belief.belief_distribution,
                                                                 self.belief.support)
        self.solver = IPOMCP(self.belief, self.environment_model, self.memoization_table,
                             self.exploration_policy, self.utility_function, {"threshold": self._threshold}, seed)
        self.name = "DoM(1)_sender"

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, gamma):
        self._threshold = gamma
        self._high = 1 - gamma if gamma is not None else 1.0
        self.solver.planning_parameters = {"threshold": self._threshold}