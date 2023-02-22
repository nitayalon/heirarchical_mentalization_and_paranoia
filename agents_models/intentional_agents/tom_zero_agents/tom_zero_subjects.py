from agents_models.abstract_agents import *


class TomZeroSubjectBelief(DoMZeroBelief):

    def __init__(self, intentional_threshold_belief, opponent_model):
        super().__init__(intentional_threshold_belief, opponent_model)

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
            if theta == 0.0:
                offer_likelihood[i] = 1 / len(self.opponent_model.potential_actions)
                continue
            self.opponent_model.threshold = theta
            possible_opponent_actions, opponent_q_values, probabilities = \
                self.opponent_model.forward(last_observation, action)
            # If the observation is not in the feasible action set then it singles theta hat:
            observation_in_feasible_set = np.any(possible_opponent_actions == observation)
            if not observation_in_feasible_set:
                observation_probability = 1e-4
            else:
                observation_probability = probabilities[np.where(possible_opponent_actions == observation)]
            offer_likelihood[i] = observation_probability
        self.opponent_model.threshold = original_threshold
        return offer_likelihood


class ToMZeroSubjectEnvironmentModel(EnvironmentModel):

    def __init__(self, opponent_model: BasicModel, reward_function, low, high,
                 belief_distribution: TomZeroSubjectBelief):
        super().__init__(opponent_model, belief_distribution)
        self.reward_function = reward_function
        self.opponent_model = opponent_model
        self.low = low
        self.high = high

    def update_low_and_high(self, observation, action):
        self.opponent_model.update_bounds(observation, action)
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high

    def reset_persona(self, persona, history_length, nested_beliefs):
        self.opponent_model.threshold = persona
        observation = self.opponent_model.belief.history.observations[history_length-1]
        action = self.opponent_model.belief.history.actions[history_length-1]
        self.opponent_model.update_bounds(action, observation)

    def step(self, interactive_state: InteractiveState, action: Action, observation: Action, seed: int,
             iteration_number: int):
        counter_offer, q_values = self.opponent_model.act(seed, observation.value, action.value, iteration_number)
        # Adding belief update here
        self.belief_distribution.update_history(action.value, observation.value)
        self.belief_distribution.update_distribution(action, Action(counter_offer, False), iteration_number)
        reward = self.reward_function(observation.value, action.value, interactive_state.persona)
        interactive_state.state.name = str(int(interactive_state.state.name) + 1)
        interactive_state.state.terminal = interactive_state.state.name == 10
        return interactive_state, Action(counter_offer, False), reward

    def update_persona(self, observation, action):
        response = bool(action.value)
        self.opponent_model.low = self.low
        self.opponent_model.high = self.high
        self.opponent_model.update_bounds(observation, response)
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high


class ToMZeroSubjectExplorationPolicy:

    def __init__(self, actions, reward_function, exploration_bonus):
        self.reward_function = reward_function
        self.actions = actions
        self.exploration_bonus = exploration_bonus

    def sample(self, interactive_state: InteractiveState, last_action: bool, observation: float,
               iteration_number: int):
        reward_from_acceptance = self.reward_function(observation, True, interactive_state.persona)
        rejection_bonus = self.exploration_bonus * 1 / iteration_number
        reward_from_rejection = self.reward_function(observation, False, interactive_state.persona) + rejection_bonus
        optimal_action = [True, False][np.argmax([reward_from_acceptance, reward_from_rejection])]
        q_value = reward_from_acceptance * optimal_action + reward_from_rejection * (1-optimal_action)
        return Action(optimal_action, False), q_value

    def init_q_values(self, observation: Action):
        initial_qvalues = self.reward_function(observation.value, self.actions, None, False)
        return initial_qvalues


class DoMZeroSubject(DoMZeroModel):

    def __init__(self,
                 actions,
                 softmax_temp: float,
                 threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: BasicModel,
                 seed: int,
                 alpha: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.alpha = alpha
        self.belief = TomZeroSubjectBelief(prior_belief, self.opponent_model)
        self.environment_model = ToMZeroSubjectEnvironmentModel(self.opponent_model, self.utility_function,
                                                                self.opponent_model.low, self.opponent_model.high,
                                                                self.belief)
        self.exploration_policy = ToMZeroSubjectExplorationPolicy(self.potential_actions, self.utility_function,
                                                                  self.config.get_from_env("rollout_rejecting_bonus"))
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)
        self.name = "DoM(0)_subject"

    def utility_function(self, action, observation, theta_hat=None, final_trial=True):
        """

        :param theta_hat: float - representing the true persona of the opponent
        :param final_trial: bool - indicate if last trial or not
        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        game_reward = (1 - action - self.threshold) * observation
        recognition_reward = 0.0
        if final_trial:
            true_theta_hat = self.belief.belief_distribution[:, 0] == theta_hat
            theta_hat_distribution = self.belief.belief_distribution[:, -1]
            recognition_reward = np.dot(true_theta_hat, theta_hat_distribution)
        return (1-self.alpha) * recognition_reward + self.alpha * game_reward

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
