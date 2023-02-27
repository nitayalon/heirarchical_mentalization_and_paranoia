from agents_models.abstract_agents import *
from IPOMCP_solver.Solver.ipomcp_solver import *


class TomZeroAgentBelief(DoMZeroBelief):

    def __init__(self, intentional_threshold_belief, opponent_model: BasicModel, history: History):
        super().__init__(intentional_threshold_belief, opponent_model, history)

    def compute_likelihood(self, action, observation, prior):
        """
        Compute observation likelihood given opponent's type and last action
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
                self.opponent_model.forward(last_observation, action)
            # If the observation is not in the feasible action set then it singles theta hat:
            observation_probability = probabilities[np.where(possible_opponent_actions == observation.value)]
            offer_likelihood[i] = observation_probability
        self.opponent_model.threshold = original_threshold
        return offer_likelihood


class ToMZeroAgentEnvironmentModel(DoMZeroEnvironmentModel):

    def __init__(self, opponent_model: BasicModel, reward_function,
                 belief_distribution: TomZeroAgentBelief):
        super().__init__(opponent_model, reward_function, belief_distribution)


class ToMZeroAgentExplorationPolicy:

    def __init__(self, actions, reward_function, exploration_bonus):
        self.reward_function = reward_function
        self.actions = actions
        self.exploration_bonus = exploration_bonus

    def sample(self, interactive_state: InteractiveState, last_action: float, observation: bool, iteration_number: int):
        # if the last offer was rejected - we should narrow down the search space
        potential_actions = self.actions
        if not observation and not np.all(False == (self.actions < last_action)):
            potential_actions = self.actions[self.actions < last_action]
        expected_reward_from_offer = self.reward_function(potential_actions, True) * \
                                     (interactive_state.persona <= (1 - potential_actions))
        optimal_action_idx = np.argmax(expected_reward_from_offer)
        optimal_action = potential_actions[optimal_action_idx]
        q_value = expected_reward_from_offer[optimal_action_idx]
        return Action(optimal_action, False), q_value

    def init_q_values(self, observation: Action):
        initial_qvalues = self.reward_function(self.actions, True)
        return initial_qvalues


class DoMZeroAgent(DoMZeroModel):

    def __init__(self,
                 actions,
                 softmax_temp: float,
                 threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: BasicModel,
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.config = get_config()
        self.belief = TomZeroAgentBelief(prior_belief, self.opponent_model, self.history)
        self.environment_model = ToMZeroAgentEnvironmentModel(self.opponent_model, self.utility_function, self.belief)
        self.exploration_policy = ToMZeroAgentExplorationPolicy(self.potential_actions, self.utility_function,
                                                                self.config.get_from_env("rollout_rejecting_bonus"))
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)
        self.name = "DoM(0)_agent"
        self.alpha = 0.0

    def utility_function(self, action, observation, *args, **kwargs):
        """

        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        game_reward = (action - self.threshold) * observation
        return game_reward

