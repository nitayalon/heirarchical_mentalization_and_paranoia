import numpy as np

from agents_models.intentional_agents.tom_one_agents.tom_one_agents import *


class DoMTwoBelief(DoMOneBelief):

    def __init__(self, self_belief_distribution_support,
                 other_belief_distribution_support,
                 second_level_belief, zero_level_belief, include_persona_inference: bool,
                 opponent_model: Optional[Union[DoMOneSender, SubIntentionalAgent]],
                 history: History):
        """

        :param self_belief_distribution_support: this is the support of the ego agent - in this case the DoM(2) receiver
        [random, 0.1, 0.5]
        :param other_belief_distribution_support: this is the support of the sender - [random, 0.1, 0.5]
        :param second_level_belief:
        .. math::
            P_2(P_1(P_0(\hat \theta))) -
        what my opponent thinks I think

        :param zero_level_belief: P_0(\hat \theta) - what is the type of my opponent
        :param include_persona_inference: do we infer about the persona?
        :param opponent_model: DoM(1) sender object
        :param history: history object
        """
        super().__init__(other_belief_distribution_support, second_level_belief, include_persona_inference,
                         opponent_model, history)
        # What I think about the receiver's type
        self.self_belief_distribution_support = self_belief_distribution_support
        self.zero_level_belief = zero_level_belief
        # What I think the sender thinks I think
        self.nested_belief = opponent_model.belief.belief_distribution

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
        second_order_prior = np.copy(self.belief_distribution[-1, :])
        type_likelihood = self.compute_likelihood(action, observation, second_order_prior, iteration_number)
        if self.include_persona_inference:
            # Compute P(observation|action, history)
            posterior = type_likelihood * second_order_prior
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
        self.opponent_model.belief.update_distribution(last_observation, action, iteration_number-1)
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


class DoMTwoEnvironmentModel(DoMOneEnvironmentModel):
    def __init__(self, opponent_model: DoMOneSender, reward_function, belief_distribution: DoMTwoBelief):
        super().__init__(opponent_model, reward_function, belief_distribution)

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs):
        self.opponent_model.threshold = persona
        self.opponent_model.reset(self.high, self.low, observation_length, action_length, False)
        self.opponent_model.belief.belief_distribution = nested_beliefs


class DoMTwoReceiverExplorationPolicy(DoMZeroExplorationPolicy):
    def __init__(self, actions: np.array, reward_function, exploration_bonus: float, belief: np.array,
                 type_support: np.array):
        super().__init__(actions, reward_function, exploration_bonus, belief, type_support)

    def init_q_values(self, observation: Action):
        reward_from_accept = self.reward_function(True, observation.value)
        reward_from_reject = self.exploration_bonus
        return np.array([reward_from_accept, reward_from_reject])


class DoMTwoReceiver(DoMZeroReceiver):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: Optional[Union[DoMOneSender, DoMZeroSender, SubIntentionalAgent]],
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.threshold = 0.0
        self.belief = DoMTwoBelief(self.opponent_model.belief.support, self.opponent_model.belief.support,
                                   None,
                                   self.opponent_model.belief.belief_distribution,
                                   True, self.opponent_model, self.history)
        self.environment_model = DoMTwoEnvironmentModel(self.opponent_model, self.utility_function, self.belief)
        self.exploration_policy = DoMTwoReceiverExplorationPolicy(self.potential_actions, self.utility_function,
                                                                  self.config.get_from_env("rollout_rejecting_bonus"),
                                                                  self.belief.belief_distribution,
                                                                  self.belief.support)
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)
        self.name = "DoM(2)_receiver"
