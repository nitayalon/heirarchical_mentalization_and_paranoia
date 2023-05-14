import numpy as np
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


class DoMZeroReceiverExplorationPolicy(DoMZeroExplorationPolicy):

    def __init__(self, actions, reward_function, exploration_bonus, belief: np.array, type_support: np.array):
        super().__init__(actions, reward_function, exploration_bonus, belief, type_support)

    def sample(self, interactive_state: InteractiveState, last_action: bool, observation: float,
               iteration_number: int):
        reward_from_acceptance = self.reward_function(True, observation)
        rejection_bonus = self.exploration_bonus * 1 / iteration_number
        reward_from_rejection = self.reward_function(False, observation) + rejection_bonus
        optimal_action = [True, False][np.argmax([reward_from_acceptance, reward_from_rejection])]
        q_value = reward_from_acceptance * optimal_action + reward_from_rejection * (1-optimal_action)
        return Action(optimal_action, False), q_value

    def init_q_values(self, observation: Action):
        if observation.value is None:
            initial_qvalues = np.repeat(0.0, len(self.actions))
        else:
            initial_qvalues = self.reward_function(self.actions, observation.value)
        return initial_qvalues


class DoMZeroReceiverSolver(DoMZeroEnvironmentModel):
    def __init__(self, actions, belief_distribution: DoMZeroBelief, opponent_model,
                 reward_function, planning_horizon, discount_factor, task_duration):
        super().__init__(opponent_model, reward_function, actions, belief_distribution)
        self.actions = actions
        self.belief = belief_distribution
        self.opponent_model = opponent_model
        self.utility_function = reward_function
        self.planning_horizon = planning_horizon
        self.discount_factor = discount_factor
        self.task_duration = task_duration
        self.action_node = None
        self.surrogate_actions = [Action(value, False) for value in self.actions]
        self.name = "tree_search"
        self.planning_tree = []
        self.q_values = []

    def plan(self, action, observation, iteration_number, update_belief):
        # Belief update via IRL
        action_length = len(self.belief.history.actions)
        observation_length = len(self.belief.history.observations)
        if update_belief:
            self.belief.update_distribution(action, observation, iteration_number)
        # Update rational opponent bounds
        self.update_low_and_high(observation, action, iteration_number)
        # Recursive planning_tree spanning
        q_values_array = []
        self.q_values = []
        for threshold in self.belief.support:
            # Reset nested model
            self.reset_persona(threshold, action_length, observation_length,
                               self.opponent_model.belief)
            future_values = functools.partial(self.recursive_tree_spanning, observation=observation,
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

    def recursive_tree_spanning(self, action, observation, opponent_model, iteration_number,
                                planning_step):
        # Compute trial reward
        reward = self.utility_function(action.value, observation.value)
        if planning_step >= self.planning_horizon or iteration_number >= self.task_duration:
            remaining_time = max(self.task_duration - iteration_number, 0)
            return self.utility_function(action.value, observation.value) * remaining_time
        # compute offers and probs given previous history
        potential_actions, _, probabilities = opponent_model.forward(observation, action, iteration_number)
        average_counter_offer_value = np.dot(potential_actions, probabilities).item()
        average_counter_offer = Action(average_counter_offer_value, False)
        # compute offers and probs given previous history
        future_values = functools.partial(self.recursive_tree_spanning, observation=average_counter_offer,
                                          opponent_model=self.opponent_model,
                                          iteration_number=iteration_number + 1,
                                          planning_step=planning_step + 1)
        self.planning_tree.append([self.opponent_model.threshold, iteration_number, action.value, observation.value,
                                   average_counter_offer_value])
        q_values = list(map(future_values, self.surrogate_actions))
        q_values = reward + self.discount_factor * max(q_values)
        self.q_values.append([self.opponent_model.threshold, iteration_number, action.value, observation.value, q_values])
        return q_values


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
        self.belief = DomZeroReceiverBelief(prior_belief[:, 0], prior_belief[:, 1], self.opponent_model, self.history)
        self.environment_model = DoMZeroReceiverEnvironmentModel(self.opponent_model, self.utility_function,
                                                                 actions,
                                                                 self.belief)
        self.solver = DoMZeroReceiverSolver(self.potential_actions, self.belief, self.opponent_model,
                                            self.utility_function,
                                            float(self.config.get_from_env("planning_depth")),
                                            float(self.config.get_from_env("discount_factor")),
                                            task_duration)
        self.name = "DoM(0)_receiver"

    def utility_function(self, observation, action, *args):
        """

        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        if observation is None:
            return 0.0
        game_reward = (observation - self.threshold) * action
        self.history.rewards.append(game_reward)
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
