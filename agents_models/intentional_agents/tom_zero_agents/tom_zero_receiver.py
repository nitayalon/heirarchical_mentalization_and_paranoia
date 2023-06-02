import functools
from agents_models.abstract_agents import *


class DomZeroReceiverBelief(DoMZeroBelief):

    def __init__(self, support, zero_level_belief, opponent_model, history: History):
        super().__init__(support, zero_level_belief, opponent_model, history)

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
        for i in range(len(self.support)):
            theta = self.support[i]
            if theta == 0.0:
                offer_likelihood[i] = 1 / len(self.opponent_model.potential_actions)
                continue
            self.opponent_model.threshold = theta
            possible_opponent_actions, opponent_q_values, probabilities = \
                self.opponent_model.forward(last_observation, action, iteration_number-1)
            # If the observation is not in the feasible action set then it singles theta hat:
            observation_in_feasible_set = np.any(possible_opponent_actions == observation.value)
            if not observation_in_feasible_set:
                observation_probability = 1e-4
            else:
                observation_probability = probabilities[np.where(possible_opponent_actions == observation.value)]
            offer_likelihood[i] = observation_probability
        self.opponent_model.threshold = original_threshold
        return offer_likelihood


class DoMZeroReceiverEnvironmentModel(DoMZeroEnvironmentModel):

    def __init__(self, opponent_model: SubIntentionalAgent, reward_function, actions, belief_distribution: DomZeroReceiverBelief):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)

    def update_persona(self, observation, action, iteration_number):
        self.opponent_model.update_bounds(observation, action, iteration_number)
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high


class DoMZeroDetectionMechanism:

    def __init__(self, history: History, actions, duration):
        self.mental_state = [False]
        self.history = history
        self.actions = actions
        self.duration = duration

    def reset(self, iteration_number: int, terminal=False):
        if terminal:
            self.mental_state = [False]
        else:
            self.mental_state = self.mental_state[0:iteration_number+1]

    def update_mental_state(self, iteration_number, mental_state):
        self.mental_state.append(mental_state)

    def strong_typicality(self, trial_number):
        observations = [x.value for x in self.history.observations]
        observations, number_of_appearance = np.unique(observations, return_counts=True)
        observed_frequency = number_of_appearance / trial_number
        expected_frequency = 1 / len(self.actions)
        distance = np.absolute(observed_frequency - expected_frequency)
        delta = (1 / trial_number) / expected_frequency
        typical_set = distance <= delta * expected_frequency
        return typical_set

    def expected_reward(self, trial_number):
        average_offer = np.mean([x.value for x in self.history.observations])
        expected_variance = (np.power(2, 2) - 1) / 12
        lower_bound = 0.5 - np.sqrt(expected_variance)/np.sqrt(trial_number)
        upper_bound = 0.5 + np.sqrt(expected_variance)/np.sqrt(trial_number)
        return lower_bound <= average_offer <= upper_bound

    def verify_random_behaviour(self, trial_number):
        strong_typicality = self.strong_typicality(trial_number)
        average_reward = self.expected_reward(trial_number)
        return np.all(strong_typicality) and average_reward

    def nonrandom_sender_detection(self, iteration_number, belief_distribution):
        detection_mechanism = self.verify_random_behaviour(iteration_number)
        p_random = belief_distribution[-1][0] > 0.95
        decision = p_random and not detection_mechanism
        return decision


class DoMZeroReceiverSolver(DoMZeroEnvironmentModel):
    def __init__(self, actions, belief_distribution: DoMZeroBelief, opponent_model,
                 detection_mechanism: DoMZeroDetectionMechanism,
                 active_detection: bool, breakdown_policy: bool,
                 reward_function, planning_horizon, discount_factor, task_duration):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)
        self.actions = actions
        self.belief = belief_distribution
        self.opponent_model = opponent_model
        self.detection_mechanism = detection_mechanism
        self.x_ipomdp_model = active_detection
        self.breakdown_policy = breakdown_policy
        self.utility_function = reward_function
        self.planning_horizon = planning_horizon
        self.discount_factor = discount_factor
        self.task_duration = task_duration
        self.action_node = None
        self.surrogate_actions = [Action(value, False) for value in self.actions]
        self.name = "tree_search"
        self.low = 0.0
        self.high = 1.0
        self.planning_tree = []
        self.q_values = []

    def get_mental_state(self, sequence: bool = False) -> Union[bool, list]:
        if sequence:
            return self.detection_mechanism.mental_state
        return self.detection_mechanism.mental_state[-1]

    def set_mental_state(self, mental_state: bool):
        return self.detection_mechanism.mental_state.append(mental_state)

    def xipomdp_mechanism(self, iteration_number):
        n_visits = np.repeat(self.planning_horizon, self.actions.size)
        # If the Flip Flop mechanism is on
        if self.get_mental_state():
            self.detection_mechanism.update_mental_state(iteration_number, True)
            weighted_q_values = [-1, 1]
            return True, {str(a.value): a for a in self.surrogate_actions}, None, \
                   np.c_[self.actions, weighted_q_values, n_visits]
        non_random_behaviour = self.detection_mechanism.nonrandom_sender_detection(iteration_number,
                                                                                   self.belief_distribution.belief_distribution)
        # If we detect a deviation from random behaviour:
        if non_random_behaviour:
            self.detection_mechanism.update_mental_state(iteration_number, True)
            if self.breakdown_policy:
                weighted_q_values = [-1, 1]
            else:
                weighted_q_values = [-1 / 10, 1 / 10]
            return True, {str(a.value): a for a in self.surrogate_actions}, None, \
                   np.c_[self.actions, weighted_q_values, n_visits]
        else:
            self.detection_mechanism.update_mental_state(iteration_number, False)
        return False, None, None, None

    def x_ipomdp_planning(self, iteration_number):
        am_i_being_fooled, actions, mcts_tree, q_values = self.xipomdp_mechanism(iteration_number)
        if am_i_being_fooled:
            return am_i_being_fooled, actions, mcts_tree, q_values
        return False, None, None, None

    def expectimax_planning(self, observation, action_length, observation_length, iteration_number):
        q_values_array = []
        self.q_values = []
        for threshold in self.belief.support:
            # Reset nested model
            self.reset_persona(threshold, action_length, observation_length,
                               self.opponent_model.belief)
            future_values = functools.partial(self.recursive_tree_spanning,
                                              observation=observation,
                                              current_low=self.low,
                                              current_high=self.high,
                                              opponent_model=self.opponent_model,
                                              iteration_number=iteration_number,
                                              planning_step=0)
            q_values = list(map(future_values, self.surrogate_actions))
            q_values_array.append(q_values)
        dt = pd.DataFrame(self.q_values)
        self.reset_persona(None, action_length, observation_length,
                           self.opponent_model.belief)
        weighted_q_values = self.belief.belief_distribution[-1, :] @ np.array(q_values_array)
        n_visits = np.repeat(self.planning_horizon, self.actions.size)
        return {str(a.value): a for a in self.surrogate_actions}, None, np.c_[self.actions, weighted_q_values, n_visits]

    def plan(self, action, observation, iteration_number, update_belief):
        # Update history
        action_length = len(self.belief.history.actions)
        observation_length = len(self.belief.history.observations)
        # Update belief
        if update_belief:
            self.belief.update_distribution(action, observation, iteration_number)
        # Update nested opponent model
        self.update_low_and_high(self.belief.history.observations[-2] if iteration_number > 1 else Action(None, False),
                                 self.belief.history.actions[-1] if iteration_number > 1 else Action(None, False)
                                 , iteration_number)
        # planning phase
        if self.x_ipomdp_model:
            irritation_mechanism_activated, actions, mcts_tree, q_values = self.x_ipomdp_planning(iteration_number)
            # If the mismatch mechanism activated:
            if irritation_mechanism_activated:
                return actions, mcts_tree, q_values
        actions, mcts_tree, q_values = self.expectimax_planning(observation, action_length, observation_length,
                                                                iteration_number)
        return actions, mcts_tree, q_values

    def recursive_tree_spanning(self, action, observation,
                                current_low, current_high,
                                opponent_model, iteration_number,
                                planning_step):
        # Compute trial reward
        reward = self.utility_function(action.value, observation.value)
        if planning_step >= self.planning_horizon or iteration_number >= self.task_duration:
            remaining_time = max(self.task_duration - iteration_number, 1)
            return self.utility_function(action.value, observation.value) * remaining_time
        # Update bounds:
        current_low = observation.value * (1-action.value) + current_low * action.value
        current_high = observation.value * action.value + current_high * (1-action.value)
        # compute offers and probs given previous history
        potential_actions, _, probabilities = opponent_model.forward(observation, action, iteration_number,
                                                                     [current_low, current_high])
        average_counter_offer_value = np.round(np.dot(potential_actions, probabilities).item() / 0.05) * 0.05
        average_counter_offer = Action(average_counter_offer_value, False)
        # compute offers and probs given previous history
        future_values = functools.partial(self.recursive_tree_spanning,
                                          observation=average_counter_offer,
                                          current_low=current_low,
                                          current_high=current_high,
                                          opponent_model=self.opponent_model,
                                          iteration_number=iteration_number + 1,
                                          planning_step=planning_step + 1)
        self.planning_tree.append([self.opponent_model.threshold, iteration_number, action.value, observation.value,
                                   average_counter_offer_value])
        future_q_values = list(map(future_values, self.surrogate_actions))
        q_value = reward + self.discount_factor * max(future_q_values)
        self.q_values.append([self.opponent_model.threshold, iteration_number, action.value, observation.value, reward,
                              q_value, future_q_values])
        return q_value
    
    def reset(self, iteration_number):
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high
        self.detection_mechanism.reset(iteration_number)


class DoMZeroReceiver(DoMZeroModel):

    def __init__(self,
                 actions,
                 softmax_temp: float,
                 threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: SubIntentionalAgent,
                 seed: int,
                 task_duration: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        # Add flipflop mechanism
        self.detection_mechanism = DoMZeroDetectionMechanism(self.history,
                                                             self.opponent_model.potential_actions,
                                                             task_duration)
        self.belief = DomZeroReceiverBelief(prior_belief[:, 0], prior_belief[:, 1], self.opponent_model, self.history)
        self.environment_model = DoMZeroReceiverEnvironmentModel(self.opponent_model, self.utility_function,
                                                                 actions,
                                                                 self.belief)
        self.solver = DoMZeroReceiverSolver(self.potential_actions, self.belief, self.opponent_model,
                                            self.detection_mechanism,
                                            self.config.get_from_env("active_detection_mechanism"),
                                            self.config.get_from_env("break_down_policy"),
                                            self.utility_function,
                                            float(self.config.get_from_env("planning_depth")),
                                            float(self.config.get_from_env("discount_factor")),
                                            task_duration)
        self.name = "DoM(0)_receiver"

    def get_mental_state(self, sequence: bool = False):
        return self.solver.get_mental_state(sequence)

    def set_mental_state(self, mental_state: bool):
        return self.solver.set_mental_state(mental_state)

    def utility_function(self, observation, action, *args):
        """

        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        if observation is None:
            return 0.0
        game_reward = (action - self.threshold) * observation
        return game_reward

    def update_belief(self, action, observation):
        observation_likelihood_per_type = np.zeros_like(self.belief.belief_distribution[:, 0])
        i = 0
        opponent_threshold = self.opponent_model.threshold
        for gamma in self.belief.belief_distribution[:, 0]:
            self.opponent_model.threshold = gamma
            relevant_actions, q_values, probabilities = self.opponent_model.forward(observation, action)
            observation_likelihood = probabilities[np.where(relevant_actions == observation)]
            observation_likelihood_per_type[i] = observation_likelihood
            i += 1
        self.opponent_model.threshold = opponent_threshold
        prior = self.belief.belief_distribution[:, -1]
        posterior = observation_likelihood_per_type * prior
        self.belief.belief_distribution = np.c_[self.belief.belief_distribution, posterior / posterior.sum()]
