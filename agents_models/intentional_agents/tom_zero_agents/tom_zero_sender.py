from agents_models.subintentional_agents.subintentional_receiver import *
import functools


class TomZeroAgentBelief(DoMZeroBelief):

    def __init__(self, intentional_threshold_belief, opponent_model: SubIntentionalAgent, history: History):
        super().__init__(intentional_threshold_belief, opponent_model, history)

    def compute_likelihood(self, action, observation, prior, iteration_number=None):
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
        for i in range(len(self.prior_belief[:, 0])):
            theta = self.prior_belief[:, 0][i]
            self.opponent_model.threshold = theta
            possible_opponent_actions, opponent_q_values, probabilities = \
                self.opponent_model.forward(last_observation, action, iteration_number)
            # If the observation is not in the feasible action set then it singles theta hat:
            observation_probability = probabilities[np.where(possible_opponent_actions == observation.value)]
            offer_likelihood[i] = observation_probability
        self.opponent_model.threshold = original_threshold
        return offer_likelihood


class ToMZeroAgentEnvironmentModel(DoMZeroEnvironmentModel):

    def __init__(self, opponent_model: SubIntentionalAgent, reward_function,
                 belief_distribution: TomZeroAgentBelief):
        super().__init__(opponent_model, reward_function, belief_distribution)


class ToMZeroAgentExplorationPolicy(DoMZeroExplorationPolicy):

    def __init__(self, actions, reward_function, exploration_bonus, belief: np.array):
        super().__init__(actions, reward_function, exploration_bonus, belief)

    def sample(self, interactive_state: InteractiveState, last_action: float, observation: bool, iteration_number: int):
        # if the last offer was rejected - we should narrow down the search space
        potential_actions = self.actions
        if not observation and not np.all(False == (self.actions < last_action)):
            potential_actions = self.actions[self.actions < last_action]
        expected_reward_from_offer = self.reward_function(potential_actions, True) * \
                                     (interactive_state.persona < (1 - potential_actions))
        optimal_action_idx = np.argmax(expected_reward_from_offer)
        optimal_action = potential_actions[optimal_action_idx]
        q_value = expected_reward_from_offer[optimal_action_idx]
        return Action(optimal_action, False), q_value

    def init_q_values(self, observation: Action):
        reward_from_action = self.reward_function(self.actions, True)
        acceptance_probability = np.dot((self.belief[:, 0][:, np.newaxis] <= (1-self.actions)).T, self.belief[:, 1])
        initial_qvalues = np.multiply(reward_from_action, acceptance_probability)
        return initial_qvalues


class DoMZeroSenderSolver(DoMZeroEnvironmentModel):
    def __init__(self, actions, belief_distribution: DoMZeroBelief, opponent_model: SubIntentionalAgent,
                 reward_function, planning_horizon, discount_factor):
        super().__init__(opponent_model, reward_function, belief_distribution)
        self.actions = actions
        self.belief = belief_distribution
        self.opponent_model = opponent_model
        self.utility_function = reward_function
        self.planning_horizon = planning_horizon
        self.discount_factor = discount_factor
        self.action_node = None
        self.surrogate_actions = [Action(value, False) for value in self.actions]
        self.name = "tree_search"
        self.tree = []

    def plan(self, action, observation, iteration_number, update_belief):
        # Belief update via IRL
        action_length = len(self.belief.history.actions)
        observation_length = len(self.belief.history.observations)
        if update_belief:
            self.belief.update_distribution(action, observation, iteration_number)
        # Recursive tree spanning
        q_values_array = []
        for threshold in self.belief.belief_distribution[:, 0]:
            # Reset nested model
            self.reset_persona(threshold, action_length, observation_length,
                               self.opponent_model.belief)
            future_values = functools.partial(self.compute_expected_value_from_offer, observation=observation,
                                              opponent_model=self.opponent_model,
                                              iteration_number=iteration_number)
            q_values = list(map(future_values, self.surrogate_actions))
            q_values_array.append(q_values)
        weighted_q_values = self.belief.belief_distribution[:, -1] @ np.array(q_values_array)
        return {str(a.value): a for a in self.surrogate_actions}, None, np.c_[self.actions, weighted_q_values]

    def compute_expected_value_from_offer(self, action, observation, opponent_model, iteration_number):
        # Compute trial reward
        response, _, probabilities = opponent_model.forward(observation, action)
        reward = self.utility_function(action.value, response) @ probabilities
        return reward


class DoMZeroSender(DoMZeroModel):

    def __init__(self,
                 actions,
                 softmax_temp: float,
                 threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: SubIntentionalAgent,
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.config = get_config()
        self.belief = TomZeroAgentBelief(prior_belief, self.opponent_model, self.history)
        self.environment_model = ToMZeroAgentEnvironmentModel(self.opponent_model, self.utility_function, self.belief)
        self.exploration_policy = ToMZeroAgentExplorationPolicy(self.potential_actions, self.utility_function,
                                                                self.config.get_from_env("rollout_rejecting_bonus"),
                                                                self.belief.belief_distribution[:, :2])
        # self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)
        self.solver = DoMZeroSenderSolver(self.potential_actions, self.belief, self.opponent_model,
                                          self.utility_function,
                                          float(self.config.get_from_env("planning_depth")),
                                          float(self.config.get_from_env("discount_factor")))
        self.name = "DoM(0)_agent"
        self.alpha = 0.0

    def utility_function(self, action, observation, *args):
        """

        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        if len(args) == 0:
            receiver_counter_action = observation
        else:
            receiver_counter_action = args[0]
        game_reward = (1 - action - self.threshold) * receiver_counter_action
        self.history.rewards.append(game_reward)
        return game_reward

