from agents_models.abstract_agents import *


class TomZeroSubjectBelief(DoMZeroBelief):

    def __init__(self, intentional_threshold_belief, opponent_model, history: History):
        super().__init__(intentional_threshold_belief, opponent_model, history)

    def compute_likelihood(self, action: Action, observation: Action, prior):
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
            # if theta == 0.0:
            #     offer_likelihood[i] = 1 / len(self.opponent_model.potential_actions)
            #     continue
            self.opponent_model.threshold = theta
            possible_opponent_actions, opponent_q_values, probabilities = \
                self.opponent_model.forward(last_observation, action)
            # If the observation is not in the feasible action set then it singles theta hat:
            observation_in_feasible_set = np.any(possible_opponent_actions == observation.value)
            if not observation_in_feasible_set:
                observation_probability = 1e-4
            else:
                observation_probability = probabilities[np.where(possible_opponent_actions == observation.value)]
            offer_likelihood[i] = observation_probability
        self.opponent_model.threshold = original_threshold
        return offer_likelihood


class ToMZeroSubjectEnvironmentModel(DoMZeroEnvironmentModel):

    def __init__(self, opponent_model: SubIntentionalAgent, reward_function, belief_distribution: TomZeroSubjectBelief):
        super().__init__(opponent_model, reward_function, belief_distribution)

    def update_persona(self, observation, action):
        self.opponent_model.low = self.low
        self.opponent_model.high = self.high
        self.opponent_model.update_bounds(observation, action)
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high


class ToMZeroSubjectExplorationPolicy(DoMZeroExplorationPolicy):

    def __init__(self, actions, reward_function, exploration_bonus, belief: np.array):
        super().__init__(actions, reward_function, exploration_bonus, belief)

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


class DoMZeroReceiver(DoMZeroModel):

    def __init__(self,
                 actions,
                 softmax_temp: float,
                 threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: SubIntentionalAgent,
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.belief = TomZeroSubjectBelief(prior_belief, self.opponent_model, self.history)
        self.environment_model = ToMZeroSubjectEnvironmentModel(self.opponent_model, self.utility_function,
                                                                self.belief)
        self.exploration_policy = ToMZeroSubjectExplorationPolicy(self.potential_actions, self.utility_function,
                                                                  self.config.get_from_env("rollout_rejecting_bonus"),
                                                                  self.belief.belief_distribution[:, :2])
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)
        self.name = "DoM(0)_subject"

    def utility_function(self, action, observation, *args):
        """

        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        game_reward = (1 - observation - self.threshold) * action
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
