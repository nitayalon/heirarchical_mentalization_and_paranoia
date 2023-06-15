from agents_models.intentional_agents.tom_zero_agents.tom_zero_sender import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_receiver import *
from agents_models.intentional_agents.tom_one_agents.dom_one_memoization import *
from typing import Optional, Union


class DoMOneBelief(DoMZeroBelief):
    def __init__(self, belief_distribution_support, zero_level_belief, include_persona_inference: bool,
                 opponent_model: Optional[Union[DoMZeroSender, SubIntentionalAgent]],
                 history: History):
        super().__init__(belief_distribution_support[:, 0],
                         belief_distribution_support[:, 1], opponent_model, history)
        self.type_belief = self.prior_belief
        # Because there's no observation uncertainty the DoM(1) belief about the DoM(0) beliefs are exact
        self.nested_belief = opponent_model.belief.belief_distribution
        self.belief_distribution = {"type_belief": self.type_belief, "nested_beliefs": self.nested_belief}
        self.include_persona_inference = include_persona_inference
        self.nested_mental_state = False

    def update_distribution(self, action, observation, iteration_number):
        """
        Update the belief based on the last action and observation (IRL)
        :param action:
        :param observation:
        :param iteration_number:
        :return:
        """
        if iteration_number < 1:
            return None
        self.nested_mental_state = not self.opponent_model.detection_mechanism.verify_random_behaviour(iteration_number)
        prior = np.copy(self.belief_distribution["type_belief"][-1, :])
        likelihood = self.compute_likelihood(action, observation, prior, iteration_number)
        if self.include_persona_inference:
            # Compute P(observation|action, history)
            posterior = likelihood * prior
            self.belief_distribution['type_belief'] = np.vstack([self.belief_distribution['type_belief'], posterior / posterior.sum()])
        # Store nested belief
        self.nested_belief = self.opponent_model.belief.belief_distribution
        self.belief_distribution["nested_beliefs"] = self.nested_belief
        self.opponent_model.opponent_model.update_bounds(action, observation, iteration_number)

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
        probabilities = self.belief_distribution['type_belief'][-1, :]
        rng_generator = np.random.default_rng(rng_key)
        idx = rng_generator.choice(self.belief_distribution['type_belief'].shape[1], size=n_samples, p=probabilities)
        particles = self.support[idx]
        mental_state = [False] * n_samples
        return list(zip(particles, mental_state))


class DoMOneEnvironmentModel(DoMZeroEnvironmentModel):
    def compute_future_values(self, value, value1, iteration_number, duration):
        pass

    def __init__(self, opponent_model: DoMZeroReceiver, reward_function,
                 actions: np.array,
                 belief_distribution: DoMOneBelief):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs):
        self.opponent_model.threshold = persona
        if action_length == 0 and observation_length == 0:
            return None
        self.opponent_model.reset(self.high, self.low, observation_length, action_length, False)
        self.opponent_model.belief.belief_distribution = nested_beliefs

    def update_persona(self, observation, action, iteration_number):
        self.opponent_model.opponent_model.update_bounds(action, observation, iteration_number+1)

    def update_parameters(self):
        pass


class DoMOneSenderEnvironmentModel(DoMOneEnvironmentModel):

    def __init__(self, opponent_model: DoMZeroReceiver, reward_function, actions: np.array,
                 belief_distribution: DoMOneBelief):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)
        self.upper_bounds = opponent_model.opponent_model.upper_bounds
        self.lower_bounds = opponent_model.opponent_model.lower_bounds

    def rollout_step(self, interactive_state: InteractiveState, action: Action, observation: Action, seed: int,
                     iteration_number: int, *args):
        counter_offer, observation_probability, q_values, opponent_policy = self.opponent_model.act(seed, observation,
                                                                                                    action,
                                                                                                    iteration_number)
        mental_model = self.opponent_model.get_mental_state()
        reward = self.reward_function(action.value, observation.value, counter_offer.value) * observation_probability + \
                 self.reward_function(action.value, observation.value, not counter_offer.value) * (
                         1 - observation_probability)
        interactive_state.state.terminal = interactive_state.state.name == 10
        interactive_state.state.name = str(int(interactive_state.state.name) + 1)
        interactive_state.persona = [interactive_state.persona[0], mental_model]
        interactive_state.opponent_belief = self.opponent_model.belief.belief_distribution[-1, :]
        return interactive_state, counter_offer, reward, observation_probability

    def compute_future_values(self, observation, action, iteration_number, duration):
        current_reward = self.reward_function(action, observation)
        # We can expect to get this reward if the opponent isn't angry with us
        reward = current_reward * (1 - self.opponent_model.get_mental_state())
        total_reward = reward * max(duration - iteration_number, 1)
        return total_reward

    def get_persona(self):
        return [self.opponent_model.threshold, self.opponent_model.get_mental_state()]

    def update_parameters(self):
        self.upper_bounds = self.opponent_model.opponent_model.upper_bounds
        self.lower_bounds = self.opponent_model.opponent_model.lower_bounds

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs):
        self.opponent_model.threshold = persona[0]
        self.opponent_model.reset(self.high, self.low, observation_length, action_length, False)
        self.opponent_model.belief.belief_distribution = nested_beliefs
        iteration_number = action_length
        if iteration_number >= 1 and len(self.belief_distribution.history.observations) > 0:
            action = self.belief_distribution.history.actions[observation_length-1]
            observation = self.belief_distribution.history.observations[observation_length-1]
            self.opponent_model.opponent_model.update_bounds(action, observation, iteration_number)

    def recall_opponents_actions_from_memory(self, key, iteration_number, action, observation,
                                             action_node: ActionNode, seed):
        # update history
        if iteration_number > 0:
            self.opponent_model.history.update_observations(action)
            self.opponent_model.opponent_model.history.update_actions(action)
        # update distribution
        self.opponent_model.belief.update_distribution(observation, action,  iteration_number)
        # update persona
        if self.opponent_model.solver.x_ipomdp_model:
            mental_model = self.opponent_model.solver.detection_mechanism.nonrandom_sender_detection(iteration_number,
                                                                                                     self.opponent_model.belief.belief_distribution)
            self.opponent_model.solver.detection_mechanism.mental_state.append(mental_model)
        # sample previous Q-values
        q_values, opponent_policy = action_node.opponent_response[key]
        prng = np.random.default_rng(seed + iteration_number)
        best_action_idx = prng.choice(a=len(q_values), p=opponent_policy)
        counter_offer, observation_probability = Action(self.opponent_model.potential_actions[best_action_idx], False), \
                                                 opponent_policy[best_action_idx]
        self.opponent_model.environment_model.update_persona(action, counter_offer, iteration_number)
        self.opponent_model.history.update_actions(counter_offer)
        self.opponent_model.environment_model.opponent_model.history.update_observations(counter_offer)
        # In case we're in the XIPOMDP env:
        self.opponent_model.environment_model.update_parameters()
        return counter_offer, observation_probability, q_values, opponent_policy

    def step(self, history_node: HistoryNode, action_node: ActionNode, interactive_state: InteractiveState,
             seed: int, iteration_number: int, *args):
        action = action_node.action
        observation = history_node.observation
        nested_beliefs = np.round(interactive_state.get_nested_belief, 3)
        key = f'{nested_beliefs}-{interactive_state.persona}-{observation.value}-{action.value}-{iteration_number}'
        # If we already visited this node
        if key in action_node.opponent_response.keys():
            counter_offer, observation_probability, q_values, opponent_policy = \
                self.recall_opponents_actions_from_memory(key, iteration_number, action, observation, action_node, seed)
        else:
            counter_offer, observation_probability, q_values, opponent_policy = \
                self.opponent_model.act(seed, observation, action, iteration_number)
            action_node.add_opponent_response(key, q_values, opponent_policy)
        mental_model = self.opponent_model.get_mental_state()
        opponent_reward = counter_offer.value * action.value
        self.opponent_model.history.update_rewards(opponent_reward)
        expected_reward = self.reward_function(action.value, observation.value, counter_offer.value) * observation_probability + \
                 self.reward_function(action.value, observation.value, not counter_offer.value) * (1-observation_probability)
        interactive_state.state.terminal = interactive_state.state.name == 10
        interactive_state.state.name = str(int(interactive_state.state.name) + 1)
        interactive_state.persona = [interactive_state.persona[0], mental_model]
        interactive_state.opponent_belief = self.opponent_model.belief.belief_distribution[-1, :]
        return interactive_state, counter_offer, expected_reward, observation_probability


class DoMOneSenderExplorationPolicy(DoMZeroSenderExplorationPolicy):

    def __init__(self, actions, reward_function, exploration_bonus, belief: np.array, type_support: np.array,
                 nested_type_support:np.array):
        super().__init__(actions, reward_function, exploration_bonus, belief, type_support)
        self.nested_type_support = nested_type_support

    def sample(self, interactive_state: InteractiveState, last_action: float, observation: bool, iteration_number: int):
        acceptance_odds = np.array([x >= 1-self.actions for x in self.nested_type_support[1:]]).T
        acceptance_odds = np.c_[np.repeat(True, len(self.actions)), acceptance_odds]
        current_beliefs = interactive_state.get_nested_belief
        acceptance_probability = np.multiply(current_beliefs, acceptance_odds).sum(axis=1) * 1.0
        opponents_reward = self.actions - interactive_state.persona[0]
        weights = (opponents_reward > 0.0) * 0.5
        expected_reward_from_offer = self.reward_function(self.actions, True) * acceptance_probability + weights
        optimal_action_idx = np.argmax(expected_reward_from_offer)
        optimal_action = self.actions[optimal_action_idx]
        q_value = expected_reward_from_offer[optimal_action_idx]
        return Action(optimal_action, False), q_value


class DoMOneSender(DoMZeroSender):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float],
                 memoization_table: DoMOneMemoization,
                 prior_belief: np.array,
                 opponent_model: DoMZeroReceiver,
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self._planning_parameters = dict(seed=seed, threshold=self._threshold)
        self.memoization_table = memoization_table
        self.belief = DoMOneBelief(prior_belief,
                                   self.opponent_model.belief.belief_distribution,
                                   True, self.opponent_model, self.history)
        self.environment_model = DoMOneSenderEnvironmentModel(self.opponent_model, self.utility_function,
                                                              actions,
                                                              self.belief)
        self.exploration_policy = DoMOneSenderExplorationPolicy(self.potential_actions, self.utility_function,
                                                                self.config.get_from_env("rollout_rejecting_bonus"),
                                                                self.belief.belief_distribution,
                                                                self.belief.support,
                                                                self.opponent_model.belief.support)
        self.solver = IPOMCP(1, self.belief, self.environment_model, self.memoization_table,
                             self.exploration_policy, self.utility_function, self._planning_parameters, seed)
        self.name = "DoM(1)_sender"

    def update_nested_models(self, action=None, observation=None, iteration_number=None):
        self.opponent_model.history.rewards.append(action.value * observation.value)

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, gamma):
        self._threshold = gamma
        self._high = 1 - gamma if gamma is not None else 1.0
        self.solver.planning_parameters["threshold"] = self._threshold
