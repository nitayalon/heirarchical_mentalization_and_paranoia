from agents_models.abstract_agents import *
from IPOMCP_solver.Solver.ipomcp_solver import *


class TomZeroSubjectBelief(DoMZeroBelief):

    def __init__(self, prior_belief, opponent_model: SubIntentionalModel):
        super().__init__(prior_belief, opponent_model)
        self.rollout_belief = self.belief_distribution

    def update_history(self, action, observation):
        """
        Method helper for history update - append the last action and observation
        :param action:
        :param observation:
        :return:
        """
        self.history.update_history(action, observation)
        self.opponent_model.update_bounds(observation, action)

    def update_distribution(self, action, observation, first_move):
        """
        Update the belief based on the last action and observation (IRL)
        :param action:
        :param observation:
        :param first_move:
        :return:
        """
        prior = np.copy(self.belief_distribution[:, -1])
        policy_based_probabilities = self.compute_likelihood(action.value, observation.value, prior)
        probabilities = policy_based_probabilities
        posterior = probabilities * prior
        self.belief_distribution = np.c_[self.belief_distribution, posterior / posterior.sum()]

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
        for i in range(len(self.prior_belief[:, 0])):
            theta = self.prior_belief[:, 0][i]
            self.opponent_model.threshold = theta
            possible_opponent_actions, opponent_q_values, probabilities = self.opponent_model.forward(last_observation, action)
            # If the observation is not in the feasible action set then it singles theta hat:
            observation_in_feasible_set = np.any(possible_opponent_actions == observation)
            if not observation_in_feasible_set:
                observation_probability = 1e-4
            else:
                observation_probability = probabilities[np.where(possible_opponent_actions == observation)]
            offer_likelihood[i] = observation_probability
        return offer_likelihood

    def sample(self, rng_key, n_samples):
        probabilities = self.belief_distribution[:, -1]
        rng_generator = np.random.default_rng(rng_key)
        particles = rng_generator.choice(self.belief_distribution[:, 0], size=n_samples, p=probabilities)
        return particles

    def reset_belief(self, history_length):
        self.belief_distribution = self.rollout_belief
        self.history = self.history.reset(history_length+2)


class ToMZeroSubjectEnvironmentModel(EnvironmentModel):

    def __init__(self, opponent_model: SubIntentionalModel, reward_function, low, high, 
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

    def reset_persona(self, persona, action, observation, nested_beliefs):
        self.opponent_model.threshold = persona
        self.opponent_model.update_bounds(observation, action)

    def step(self, interactive_state: InteractiveState, action: Action, observation: Action, seed: int,
             iteration_number: int) -> tuple[InteractiveState, Action, float]:
        counter_offer = self.opponent_model.act(seed, observation.value, action.value)
        # Adding belief update here
        self.belief_distribution.update_history(action.value, observation.value)
        self.belief_distribution.update_distribution(action, Action(counter_offer, False), iteration_number)
        reward = self.reward_function(observation.value, action.value, interactive_state.persona)
        interactive_state.state.name = str(int(interactive_state.state.name) + 1)
        interactive_state.state.terminal = interactive_state.state.name == 10
        return interactive_state, Action(counter_offer, False), reward

    def update_persona(self, observation, action):
        self.opponent_model.low = self.low
        self.opponent_model.high = self.high
        self.opponent_model.update_bounds(observation, action)
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high


class ToMZeroSubjectExplorationPolicy:

    def __init__(self, actions, reward_function, exploration_bonus):
        self.reward_function = reward_function
        self.actions = actions
        self.exploration_bonus = exploration_bonus

    def sample(self, interactive_state: InteractiveState, last_action: bool, observation: float,
               rng_key: int, iteration_number):
        reward_from_acceptance = self.reward_function(observation, True, interactive_state.persona)
        rejection_bonus = self.exploration_bonus * 1 / iteration_number
        reward_from_rejection = self.reward_function(observation, False, interactive_state.persona) + rejection_bonus
        optimal_action = [True, False][np.argmax([reward_from_acceptance, reward_from_rejection])]
        q_value = reward_from_acceptance * optimal_action + reward_from_rejection * (1-optimal_action)
        return Action(optimal_action, False), q_value

    def init_q_values(self, observation: Action):
        initial_qvalues = self.reward_function(observation.value, self.actions)
        return initial_qvalues


class ToMZeroSubject(DoMZeroModel):

    def __init__(self,
                 actions,
                 softmax_temp: float,
                 prior_belief: np.array,
                 opponent_model: SubIntentionalModel,
                 seed: int,
                 alpha: float):
        super().__init__(actions, softmax_temp, prior_belief, opponent_model, alpha)
        self.config = get_config()
        self.belief = TomZeroSubjectBelief(prior_belief, self.opponent_model)
        self.environment_model = ToMZeroSubjectEnvironmentModel(self.opponent_model, self.utility_function,
                                                                self.opponent_model.low, self.opponent_model.high,
                                                                self.belief)
        self.exploration_policy = ToMZeroSubjectExplorationPolicy(self.actions, self.utility_function, self.config.get_from_env("rollout_rejecting_bonus"))
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)

    def utility_function(self, action, observation, theta_hat=None, final_trial=True):
        """

        :param theta_hat: float - representing the true persona of the opponent
        :param final_trial: bool - indicate if last trial or not
        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        game_reward = (1 - action) * observation
        recognition_reward = 0.0
        if final_trial:
            cross_entropy = np.log(self.belief.belief_distribution[:, -1]) * (self.belief.belief_distribution[:, 0] == theta_hat)
            recognition_reward = np.sum(cross_entropy)
        return self.alpha * game_reward + (1-self.alpha) * recognition_reward

    def update_belief(self, action, observation):
        observation_likelihood_per_type = np.zeros_like(self.belief.belief_distribution[:, 0])
        i = 0
        for gamma in self.belief.belief_distribution[:, 0]:
            self.opponent_model.threshold = gamma
            relevant_actions, q_values, probabilities = self.opponent_model.forward(observation, action)
            observation_likelihood = probabilities[np.where(relevant_actions == observation)]
            observation_likelihood_per_type[i] = observation_likelihood
            i += 1
        prior = self.belief.belief_distribution[:, -1]
        posterior = observation_likelihood_per_type * prior
        self.belief.belief_distribution = np.c_[self.belief.belief_distribution, posterior / posterior.sum()]

    def act(self, seed, action=None, observation=None, iteration_number=None):
        self.belief.history.update_observations(observation)
        action_nodes, q_values = self.forward(action, observation, iteration_number)
        softmax_transformation = np.exp(q_values[:, 1] / self.softmax_temp) / np.exp(q_values[:, 1] / self.softmax_temp).sum()
        prng = np.random.default_rng(seed)
        best_action_idx = prng.choice(a=len(action_nodes), p=softmax_transformation)
        actions = list(action_nodes.keys())
        best_action = action_nodes[actions[best_action_idx]].action
        self.belief.history.update_actions(best_action)
        self.environment_model.update_persona(observation, bool(best_action.value))
        if action_nodes is not None:
            self.solver.action_node = action_nodes[str(best_action.value)]
        return best_action.value #, q_values[best_action_idx, 1], softmax_transformation[best_action_idx]

    def forward(self, action=None, observation=None, iteration_number=None):
        actions, q_values = self.solver.plan(action, observation, iteration_number)
        return actions, q_values
