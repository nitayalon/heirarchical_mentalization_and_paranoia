from agents_models.intentional_agents.tom_zero_agents.tom_zero_sender import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_receiver import *
from agents_models.intentional_agents.tom_one_agents.dom_one_memoization import *
from typing import Optional, Union


class DoMOneBelief(DoMZeroBelief):
    def __init__(self, zero_order_belief_distribution_support,
                 opponent_model: Optional[Union[DoMZeroSender, SubIntentionalAgent]],
                 history: History, include_persona_inference: bool):
        """

        :param zero_order_belief_distribution_support: np.array, representing the DoM(1) zero order beliefs about theta
        :param opponent_model: DoMZeroSender agent
        :param history:
        :param include_persona_inference: bool, should the DoM(1) infer about the persona of the DoM(0) agent
        """
        super().__init__(zero_order_belief_distribution_support[:, 0],
                         zero_order_belief_distribution_support[:, 1], opponent_model, history)
        self.zero_order_belief = self.prior_belief
        self.prior_nested_belief = self.opponent_model.belief.prior_belief
        # Because there's no observation uncertainty the DoM(1) belief about the DoM(0) beliefs are exact
        self.nested_belief = opponent_model.belief.belief_distribution
        self.include_persona_inference = include_persona_inference
        self.supports = {"zero_order_belief": zero_order_belief_distribution_support[:, 0],
                         "nested_beliefs": opponent_model.belief.support}
        self.belief_distribution = {"zero_order_belief": self.zero_order_belief, "nested_beliefs": self.nested_belief}
        self.nested_mental_state = False

    def reset(self, size: int = 1):
        self.belief_distribution['zero_order_belief'] = self.belief_distribution['zero_order_belief'][0:size,]
        self.belief_distribution['nested_beliefs'] = self.belief_distribution['nested_beliefs'][0:size,]

    def update_distribution(self, action, observation, iteration_number, nested=False):
        """
        Update the belief based on the last action and observation (IRL)
        :param action:
        :param observation:
        :param iteration_number:
        :param nested:
        :return:
        """
        if iteration_number < 1:
            return None
        self.nested_mental_state = not self.opponent_model.detection_mechanism.verify_random_behaviour(iteration_number)
        prior = np.copy(self.belief_distribution["zero_order_belief"][-1, :])
        # Compute P_0(a_t|theta, h^{t-1})
        likelihood = self.compute_likelihood(action, observation, prior, iteration_number, nested)
        if self.include_persona_inference:
            # Compute P(observation|action, history)
            posterior = likelihood * prior
            self.belief_distribution['zero_order_belief'] = np.vstack(
                [self.belief_distribution['zero_order_belief'], posterior / posterior.sum()])
        # Store nested belief
        self.nested_belief = self.opponent_model.belief.belief_distribution
        self.belief_distribution["nested_beliefs"] = self.nested_belief
        self.opponent_model.opponent_model.update_bounds(action, observation, iteration_number)

    def compute_likelihood(self, action: Action, observation: Action, prior, iteration_number=None,
                           nested=False):
        """
        Compute observation likelihood given opponent's type and last action
        :param action:
        :param observation:
        :param prior:
        :param iteration_number:
        :param nested:
        :return:
        """
        last_observation = self.history.get_last_observation(nested)
        offer_likelihood = np.empty_like(prior)
        original_threshold = self.opponent_model.threshold
        # update nested belief
        self.opponent_model.belief.update_distribution(last_observation, action, iteration_number, nested)
        if self.include_persona_inference:
            for i in range(len(self.support)):
                theta = self.support[i]
                self.opponent_model.threshold = theta
                actions, q_values, softmax_transformation, mcts_tree = \
                    self.opponent_model.forward(last_observation, action, iteration_number - 1, False)
                # If the observation is not in the feasible action set then it singles theta hat:
                observation_probability = softmax_transformation[
                    np.where(self.opponent_model.potential_actions == observation.value)]
                offer_likelihood[i] = observation_probability
            self.opponent_model.threshold = original_threshold
            # Round up to account for numerical stability issues
            offer_likelihood = np.round_(offer_likelihood, 4) + 1e-4
            return offer_likelihood
        return None

    def sample(self, rng_key, n_samples):
        """
        Sample nested beliefs
        :param rng_key:
        :param n_samples:
        :return:
        """
        probabilities = self.belief_distribution['zero_order_belief'][-1, :]
        rng_generator = np.random.default_rng(rng_key)
        idx = rng_generator.choice(self.belief_distribution['zero_order_belief'].shape[1], size=n_samples,
                                   p=probabilities)
        particles = self.support[idx]
        mental_state = [False] * n_samples
        return list(zip(particles, mental_state))

    def get_current_belief(self):
        values = [x[-1] for x in self.belief_distribution.values()]
        keys = [x for x in self.belief_distribution.keys()]
        return dict(zip(keys, values))

    def update_distribution_from_particles(self, particles: dict, action, observation, iteration_number):
        persona = [x for x in particles.keys()]
        thresholds = [float(x.split("-")[0]) for x in persona]
        # Validate that we have representation of all the types
        all_types_represented = np.sort(self.support) == np.sort(thresholds)
        interactive_states_per_persona = [x[0] for x in particles.values()]
        likelihood = [x[1] for x in interactive_states_per_persona]
        full_likelihood = likelihood * all_types_represented + 0.001 * (1-all_types_represented)
        prior_distribution = np.copy(self.belief_distribution["zero_order_belief"][-1, :])
        _, sorted_likelihood = zip(*sorted(zip(self.support, full_likelihood)))
        posterior_distribution = prior_distribution * sorted_likelihood / np.sum(prior_distribution * sorted_likelihood)
        nested_beliefs = [x[0].get_nested_belief for x in interactive_states_per_persona]
        # Update beliefs
        self.belief_distribution['zero_order_belief'] = np.vstack(
            [self.belief_distribution['zero_order_belief'], posterior_distribution])
        # Store nested belief
        # Note! since the nested belief is single - we average those
        average_nested_beliefs = np.mean(nested_beliefs, axis=0)
        self.nested_belief = np.vstack([self.nested_belief, average_nested_beliefs])
        # Update nested model beliefs
        self.opponent_model.belief.belief_distribution = np.vstack(
            [self.opponent_model.belief.belief_distribution, average_nested_beliefs])
        self.belief_distribution["nested_beliefs"] = np.copy(self.opponent_model.belief.belief_distribution)
        self.opponent_model.opponent_model.update_bounds(action, observation, iteration_number)


class DoMOneEnvironmentModel(DoMZeroEnvironmentModel):

    def __init__(self, opponent_model: Union[DoMZeroReceiver, SubIntentionalAgent], reward_function,
                 actions: np.array,
                 belief_distribution: DoMOneBelief):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs, iteration_number):
        nested_beliefs = nested_beliefs[:iteration_number + 1, :]
        self.opponent_model.threshold = persona
        if action_length == 0 and observation_length == 0:
            return None
        self.opponent_model.reset(self.high, self.low, observation_length, action_length, False)
        self.opponent_model.belief.belief_distribution = nested_beliefs

    def update_persona(self, observation, action, iteration_number):
        self.opponent_model.opponent_model.update_bounds(action, observation, iteration_number + 1)

    def update_parameters(self):
        pass

    def compute_future_values(self, value, value1, iteration_number, duration):
        pass


class DoMOneSenderEnvironmentModel(DoMOneEnvironmentModel):

    def __init__(self, opponent_model: DoMZeroReceiver, reward_function, actions: np.array,
                 belief_distribution: DoMOneBelief):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)
        self.upper_bounds = opponent_model.opponent_model.upper_bounds
        self.lower_bounds = opponent_model.opponent_model.lower_bounds

    @staticmethod
    def compute_iteration(iteration_number):
        return iteration_number + 1

    def compute_expected_reward(self, action, observation, counter_offer, observation_probability):
        expected_reward = self.reward_function(action.value, observation.value, counter_offer.value) * observation_probability + \
                 self.reward_function(action.value, observation.value, not counter_offer.value) * (
                         1 - observation_probability)
        return expected_reward

    def compute_future_values(self, observation, action, iteration_number, duration):
        current_reward = self.reward_function(action, observation)
        # We can expect to get this reward if the opponent isn't angry with us
        reward = current_reward * (1 - self.opponent_model.get_aleph_mechanism_status())
        total_reward = reward * max(duration - iteration_number, 1)
        return total_reward

    def get_persona(self):
        return Persona([self.opponent_model.threshold,False], self.opponent_model.get_aleph_mechanism_status())

    def update_parameters(self):
        self.upper_bounds = self.opponent_model.opponent_model.upper_bounds
        self.lower_bounds = self.opponent_model.opponent_model.lower_bounds

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs, iteration_number):
        nested_beliefs = nested_beliefs[:iteration_number + 1, ]
        try:
            nested_likelihood = self.opponent_model.belief.likelihood[:, :iteration_number + 1]
        except IndexError:
            nested_likelihood = self.opponent_model.belief.likelihood
        self.opponent_model.threshold = persona.persona[0]
        self.opponent_model.reset(self.high, self.low, observation_length, action_length, False)
        self.opponent_model.belief.belief_distribution = nested_beliefs
        self.opponent_model.belief.likelihood = nested_likelihood
        iteration_number = action_length
        if iteration_number >= 1 and len(self.belief_distribution.history.observations) > 0:
            action = self.belief_distribution.history.actions[observation_length - 1]
            observation = self.belief_distribution.history.observations[observation_length - 1]
            self.opponent_model.opponent_model.update_bounds(action, observation, iteration_number)

    def recall_opponents_actions_from_memory(self, key, iteration_number, action, observation,
                                             action_node: ActionNode, seed):
        # update history
        if iteration_number > 0:
            self.opponent_model.history.update_observations(action)
            self.opponent_model.opponent_model.history.update_actions(action)
        # update distribution
        self.opponent_model.belief.update_distribution(observation, action, iteration_number)
        # update persona
        if self.opponent_model.solver.aleph_ipomdp_model:
            mental_model = self.opponent_model.solver.aleph_mechanism.nonrandom_sender_detection(iteration_number,
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

    def update_interactive_state(self, interactive_state, mental_model, updated_nested_beliefs, q_values,
                                 updated_nested_likelihood=None):
        new_state_name = int(interactive_state.state.name) + 1
        new_state = State(str(new_state_name), new_state_name == 10)
        new_persona = Persona([interactive_state.persona.persona[0], mental_model], q_values)
        new_interactive_state = InteractiveState(new_state, new_persona, updated_nested_beliefs, updated_nested_likelihood)
        return new_interactive_state

    @staticmethod
    def _represent_nested_beliefs_as_table(interactive_state):
        return np.round(interactive_state.get_nested_belief, 3)

    def step_from_is(self, new_interactive_state: InteractiveState, previous_observation: Action, action: Action,
                     seed: int, iteration_number):
        # update nested history:
        if int(new_interactive_state.get_state.name) > 0:
            self.opponent_model.history.update_observations(action)
            self.opponent_model.opponent_model.history.update_actions(action)
        observation_probabilities = self.opponent_model.softmax_transformation(new_interactive_state.persona.q_values[:,1])
        random_number_generator = np.random.default_rng(seed)
        optimal_action_idx = random_number_generator.choice(new_interactive_state.persona.q_values.shape[0],
                                                            p=observation_probabilities)
        new_observation = Action(bool(new_interactive_state.persona.q_values[optimal_action_idx, 0]), False)
        observation_probability = observation_probabilities[optimal_action_idx]
        expected_reward = self.compute_expected_reward(action, previous_observation, new_observation,
                                                       observation_probability)
        # update nested model:
        self.opponent_model.history.update_actions(new_observation)
        self.opponent_model.environment_model.opponent_model.history.update_observations(new_observation)
        self.opponent_model.belief.belief_distribution = np.vstack(
            [self.opponent_model.belief.belief_distribution, new_interactive_state.get_nested_belief])
        self.opponent_model.belief.likelihood = new_interactive_state.get_nested_likelihood
        return new_observation, expected_reward, observation_probability

    def step(self, history_node: HistoryNode, action_node: ActionNode, interactive_state: InteractiveState,
             seed: int, iteration_number: int, *args):
        # a_t
        action = action_node.action
        # o_{t-1}
        observation = history_node.observation
        nested_beliefs = self._represent_nested_beliefs_as_table(interactive_state)
        key = f'{nested_beliefs}-{str(interactive_state)}-{observation.value}-{action.value}-{iteration_number}'
        # If we already visited this node
        if key in action_node.opponent_response.keys():
            counter_offer, observation_probability, q_values, opponent_policy = \
                self.recall_opponents_actions_from_memory(key, iteration_number, action, observation, action_node, seed)
        else:
            counter_offer, observation_probability, q_values, opponent_policy = \
                self.opponent_model.act(seed, observation, action, iteration_number)
            action_node.add_opponent_response(key, q_values, opponent_policy)
        mental_model = self.opponent_model.get_aleph_mechanism_status()
        updated_nested_beliefs = self.opponent_model.belief.get_current_belief()
        updated_nested_likelihood = self.opponent_model.belief.get_current_likelihood()
        opponent_reward = counter_offer.value * action.value
        self.opponent_model.history.update_rewards(opponent_reward)
        expected_reward = self.compute_expected_reward(action, observation, counter_offer, observation_probability)
        new_interactive_state = self.update_interactive_state(interactive_state, mental_model, updated_nested_beliefs,
                                                              q_values, updated_nested_likelihood)
        return new_interactive_state, counter_offer, expected_reward, observation_probability

    def rollout_step(self, interactive_state: InteractiveState, action: Action, observation: Action, seed: int,
                     iteration_number: int, *args):
        counter_offer, observation_probability, q_values, opponent_policy = self.opponent_model.act(seed + iteration_number,
                                                                                                    observation,
                                                                                                    action,
                                                                                                    iteration_number)
        mental_model = self.opponent_model.get_aleph_mechanism_status()
        reward = self.compute_expected_reward(action, observation, counter_offer, observation_probability)
        updated_nested_beliefs = self.opponent_model.belief.get_current_belief()
        new_interactive_state = self.update_interactive_state(interactive_state, mental_model, updated_nested_beliefs,
                                                              q_values)
        return new_interactive_state, counter_offer, reward, observation_probability


class DoMOneSenderExplorationPolicy(DoMZeroSenderExplorationPolicy):

    def __init__(self, actions, reward_function, exploration_bonus, belief: np.array, type_support: np.array,
                 nested_type_support: np.array):
        super().__init__(actions, reward_function, exploration_bonus, belief, type_support)
        self.nested_type_support = nested_type_support

    def sample(self, interactive_state: InteractiveState, last_action: float, observation: bool, iteration_number: int):
        acceptance_odds = np.array([x >= 1 - self.actions for x in self.nested_type_support[1:]]).T
        acceptance_odds = np.c_[np.repeat(True, len(self.actions)), acceptance_odds]
        current_beliefs = interactive_state.get_nested_belief
        acceptance_probability = np.multiply(current_beliefs, acceptance_odds).sum(axis=1) * 1.0
        opponents_reward = self.actions - interactive_state.persona.persona[0]
        weights = (opponents_reward > 0.0) * 0.5
        expected_reward_from_offer = self.reward_function(self.actions, True) * acceptance_probability + weights
        optimal_action_idx = np.argmax(expected_reward_from_offer)
        optimal_action = self.actions[optimal_action_idx]
        q_value = expected_reward_from_offer[optimal_action_idx]
        return Action(optimal_action, False), q_value

    def compute_final_round_q_values(self, observation: Action) -> np.array:
        final_q_values = self.init_q_values(observation)
        return final_q_values


class DoMOneSender(DoMZeroSender):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float],
                 memoization_table: DoMOneMemoization,
                 prior_belief: np.array,
                 opponent_model: DoMZeroReceiver,
                 seed: int, nested_model=False):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self._planning_parameters = dict(seed=seed, threshold=self._threshold)
        self.memoization_table = memoization_table
        self.belief = DoMOneBelief(prior_belief, self.opponent_model, self.history, True)
        self.environment_model = DoMOneSenderEnvironmentModel(self.opponent_model, self.utility_function,
                                                              actions,
                                                              self.belief)
        self.exploration_policy = DoMOneSenderExplorationPolicy(self.potential_actions,
                                                                self.utility_function,
                                                                self.config.get_from_env("rollout_rejecting_bonus"),
                                                                self.belief.belief_distribution,
                                                                self.belief.support,
                                                                self.opponent_model.belief.support)
        self.solver = IPOMCP(self.belief, self.environment_model, self.memoization_table,
                             self.exploration_policy, self.utility_function, self._planning_parameters, seed, 1,
                             nested_model)
        self.name = "DoM(1)_sender"
    
    @staticmethod
    def get_aleph_mechanism_status():
        return False

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

    def reset(self, high: Optional[float] = None, low: Optional[float] = None,
              action_length: Optional[float] = 0, observation_length: Optional[float] = 0,
              terminal: Optional[bool] = False):
        self.high = 1.0
        self.low = 0.0
        self.history.reset(action_length, observation_length)
        self.opponent_model.reset(1.0, 0.0, observation_length, action_length, terminal=terminal)
        self.environment_model.reset(action_length)
        self.reset_belief(action_length + 1 * terminal)
        self.reset_solver(action_length)


