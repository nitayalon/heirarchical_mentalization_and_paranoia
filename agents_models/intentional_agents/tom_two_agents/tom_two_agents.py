from agents_models.intentional_agents.tom_one_agents.tom_one_agents import *
from agents_models.intentional_agents.tom_two_agents.dom_two_memoization import *
from agents_models.subintentional_agents.subintentional_senders import *


class DoMTwoBelief(DoMOneBelief):

    def __init__(self,
                 zero_order_belief_distribution_support,
                 opponent_model: Optional[Union[DoMOneSender, SubIntentionalAgent]],
                 history: History,
                 include_persona_inference: bool):
        """
        :param zero_order_belief_distribution_support: np.array, representing the DoM(1) zero order beliefs about theta
        :param opponent_model: DoMZeroSender agent
        :param history:
        :param include_persona_inference: bool, should the DoM(1) infer about the persona of the DoM(0) agent
        """
        super().__init__(zero_order_belief_distribution_support,
                         opponent_model, history,
                         include_persona_inference)
        self.prior_nested_belief = opponent_model.belief.belief_distribution
        self.nested_belief = opponent_model.belief.belief_distribution
        # These are nested dictionaries
        self.supports = {"zero_order_belief": zero_order_belief_distribution_support[:, 0],
                         "nested_beliefs": opponent_model.belief.supports}
        self.belief_distribution = {"zero_order_belief": self.zero_order_belief,
                                    "nested_beliefs": self.nested_belief}

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
        self.nested_mental_state = False
        prior = np.copy(self.belief_distribution["zero_order_belief"][-1, :])
        likelihood = self.compute_likelihood(action, observation, prior, iteration_number)
        if self.include_persona_inference:
            # Compute P(observation|action, history)
            posterior = likelihood * prior
            self.belief_distribution['zero_order_belief'] = np.vstack(
                [self.belief_distribution['zero_order_belief'], posterior / posterior.sum()])
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
        probabilities = self.belief_distribution['zero_order_belief'][-1, :]
        rng_generator = np.random.default_rng(rng_key)
        idx = rng_generator.choice(self.belief_distribution['zero_order_belief'].shape[1], size=n_samples,
                                   p=probabilities)
        particles = self.support[idx]
        mental_state = [False] * n_samples
        return list(zip(particles, mental_state))

    def reset(self):
        self.belief_distribution['zero_order_belief'] = self.prior_belief
        self.belief_distribution['nested_beliefs'] = self.prior_nested_belief


class DoMTwoEnvironmentModel(DoMOneSenderEnvironmentModel):

    def __init__(self, intentional_opponent_model: Union[DoMOneSender], reward_function, actions,
                 belief_distribution: DoMTwoBelief):
        super().__init__(intentional_opponent_model, reward_function, actions, belief_distribution)
        self.random_sender = RandomSubIntentionalSender(
            intentional_opponent_model.opponent_model.opponent_model.potential_actions,
            intentional_opponent_model.opponent_model.opponent_model.softmax_temp, 0.0)

    @staticmethod
    def compute_iteration(iteration_number):
        return iteration_number

    def compute_expected_reward(self, action, observation, counter_offer, observation_probability):
        expected_reward = self.reward_function(action.value, observation.value,
                                               counter_offer.value)
        return expected_reward

    def update_interactive_state(self, interactive_state, mental_model, updated_nested_beliefs, q_values):
        new_state_name = int(interactive_state.state.name)
        new_state = State(str(new_state_name), new_state_name == 10)
        new_persona = Persona(interactive_state.persona.persona, q_values)
        new_interactive_state = InteractiveState(new_state, new_persona, updated_nested_beliefs)
        return new_interactive_state

    def step_from_is(self, new_interactive_state: InteractiveState, previous_observation: Action, action: Action,
                     seed: int):
        # update nested history:
        if int(new_interactive_state.get_state.name) > 0:
            self.opponent_model.history.update_observations(action)
            self.opponent_model.opponent_model.history.update_actions(action)
        observation_probabilities = self.opponent_model.softmax_transformation(new_interactive_state.persona.q_values[:,1])
        random_number_generator = np.random.default_rng(seed)
        optimal_action_idx = random_number_generator.choice(new_interactive_state.persona.q_values.shape[0],
                                                            p=observation_probabilities)
        new_observation = Action(new_interactive_state.persona.q_values[optimal_action_idx, 0], False)
        observation_probability = observation_probabilities[optimal_action_idx]
        expected_reward = self.compute_expected_reward(action, previous_observation, new_observation,
                                                       observation_probability)
        # update nested model:
        self.opponent_model.history.update_actions(new_observation)
        self.opponent_model.environment_model.opponent_model.history.update_observations(new_observation)
        updated_opponent_beliefs = new_interactive_state.get_nested_belief
        updated_zero_order_belief = updated_opponent_beliefs['zero_order_belief']
        updated_first_order_beliefs = updated_opponent_beliefs['nested_beliefs']
        self.opponent_model.belief.belief_distribution['zero_order_belief'] = np.vstack(
            [self.opponent_model.belief.belief_distribution['zero_order_belief'], updated_zero_order_belief])
        self.opponent_model.belief.belief_distribution['nested_beliefs'] = np.vstack(
            [self.opponent_model.belief.belief_distribution['nested_beliefs'], updated_first_order_beliefs])
        return new_observation, expected_reward, observation_probability

    def get_persona(self):
        return Persona([self.opponent_model.threshold, False], self.opponent_model.get_mental_state())

    def _simulate_opponent_response(self, seed, observation, action, iteration_number):
        if self.opponent_model.threshold == 0.0:
            counter_offer, observation_probability, q_values, opponent_policy = \
                self.random_sender.act(seed, observation, action, iteration_number)
        else:
            counter_offer, observation_probability, q_values, opponent_policy = \
                self.opponent_model.act(seed, observation, action, iteration_number-1)
        return counter_offer, observation_probability, q_values, opponent_policy

    @staticmethod
    def _represent_nested_beliefs_as_table(interactive_state):
        beliefs = [np.round(x, 3) for x in interactive_state.get_nested_belief.values()]
        return str(beliefs[0]) + str(beliefs[1])

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs, iteration_number):
        nested_beliefs_values = [x[:iteration_number, :] for x in nested_beliefs.values()]
        nested_beliefs_keys = [x for x in nested_beliefs.keys()]
        current_nested_beliefs = dict(zip(nested_beliefs_keys, nested_beliefs_values))
        self.opponent_model.threshold = persona.persona[0]
        self.opponent_model.reset(self.high, self.low, observation_length, action_length, False)
        self.opponent_model.belief.belief_distribution = current_nested_beliefs

    def reset(self, iteration_number):
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high
        self.random_sender.reset()


class DoMTwoReceiverExplorationPolicy(DoMZeroExplorationPolicy):

    def __init__(self, actions: np.array, reward_function, exploration_bonus: float, belief: np.array,
                 type_support: np.array):
        super().__init__(actions, reward_function, exploration_bonus, belief, type_support)

    def init_q_values(self, observation: Action, *args):
        if observation.value is None:
            return np.array([0.5, 0.5])
        reward_from_accept = self.reward_function(True, observation.value)
        reward_from_reject = self.exploration_bonus
        return np.array([reward_from_accept, reward_from_reject])

    def sample(self, interactive_state: InteractiveState, last_action: bool, observation: float, iteration_number: int):
        expected_reward_from_offer = np.array([self.reward_function(True, observation), self.exploration_bonus])
        optimal_action_idx = np.argmax(expected_reward_from_offer)
        optimal_action = self.actions[optimal_action_idx]
        q_value = expected_reward_from_offer[optimal_action_idx]
        return Action(optimal_action, False), q_value


class DoMTwoReceiver(DoMZeroReceiver):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float],
                 memoization_table: DoMTwoMemoization,
                 prior_belief: np.array,
                 opponent_model: Optional[Union[DoMOneSender, DoMZeroSender, SubIntentionalAgent]],
                 seed: int, task_duration):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed, task_duration)
        self._planning_parameters = dict(seed=seed, threshold=self._threshold)
        self.memoization_table = memoization_table
        self.belief = DoMTwoBelief(prior_belief, self.opponent_model, self.history, True)
        self.environment_model = DoMTwoEnvironmentModel(self.opponent_model, self.utility_function, actions,
                                                        self.belief)
        self.exploration_policy = DoMTwoReceiverExplorationPolicy(self.potential_actions, self.utility_function,
                                                                  self.config.get_from_env("rollout_rejecting_bonus"),
                                                                  self.belief.belief_distribution,
                                                                  self.belief.support)
        self.solver = IPOMCP(self.belief, self.environment_model, self.memoization_table,
                             self.exploration_policy, self.utility_function, self._planning_parameters, seed, False)
        self.name = "DoM(2)_receiver"

    def update_nested_models(self, action=None, observation=None, iteration_number=None):
        # update history for nested models
        # update nested DoM(0) observations
        self.opponent_model.opponent_model.history.observations.append(observation)
        # update nested DoM(-1) observations
        self.opponent_model.opponent_model.opponent_model.history.actions.append(observation)
