from typing import Tuple, Any
from agents_models.abstract_agents import *
from agents_models.subintentional_agents.subintentional_agents import IntentionalAgentSubIntentionalModel
from IPOMCP_solver.Solver.ipomcp_solver import *


class TomZeroSubjectBelief(DoMZeroBelief):

    def __init__(self, prior_belief, opponent_model: IntentionalAgentSubIntentionalModel):
        super().__init__(prior_belief, opponent_model)

    def update_history(self, action, observation):
        """
        Method helper for history update - append the last action and observation
        :param action:
        :param observation:
        :return:
        """
        self.history.update_history(action, observation)

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
            observation_probability = probabilities[np.where(possible_opponent_actions == observation)]
            offer_likelihood[i] = observation_probability
        return offer_likelihood

    def sample(self, rng_key, n_samples):
        probabilities = self.belief_distribution[:, -1]
        rng_generator = np.random.default_rng(rng_key)
        particles = rng_generator.choice(self.belief_distribution[:, 0], size=n_samples, p=probabilities)
        return particles


class ToMZeroSubjectEnvironmentModel(EnvironmentModel):

    def __init__(self, opponent_model: IntentionalAgentSubIntentionalModel, reward_function):
        super().__init__(opponent_model)
        self.reward_function = reward_function
        self.opponent_model = opponent_model

    def reset_persona(self, persona, history_length, nested_beliefs):
        self.opponent_model.threshold = persona

    def step(self, interactive_state: InteractiveState, action: Action, observation: Action, seed: int,
             iteration_number: int) -> tuple[InteractiveState, Action, float | Any]:
        counter_offer = self.opponent_model.act(seed, observation.value, action.value)
        reward = self.reward_function(observation.value, action.value)
        interactive_state.state.name = str(int(interactive_state.state.name) + 1)
        interactive_state.state.terminal = interactive_state.state.name == 10
        return interactive_state, Action(counter_offer, False), reward


class ToMZeroSubjectExplorationPolicy:

    def __init__(self, actions, reward_function, exploration_bonus):
        self.reward_function = reward_function
        self.actions = actions
        self.exploration_bonus = exploration_bonus

    def sample(self, interactive_state: InteractiveState, last_cation: bool, observation: float, rng_key: int):
        reward_from_acceptance = self.reward_function(observation, True)
        reward_from_rejection = 0.0 + self.exploration_bonus
        optimal_action = [True, False][np.argmax([reward_from_acceptance, reward_from_rejection])]
        q_value = reward_from_acceptance * optimal_action + reward_from_rejection * (1-optimal_action)
        return Action(optimal_action, False), q_value


class ToMZeroSubject(DoMZeroModel):

    def __init__(self,
                 actions,
                 softmax_temp: float,
                 prior_belief: np.array,
                 opponent_model: IntentionalAgentSubIntentionalModel,
                 seed: int):
        super().__init__(actions, softmax_temp, prior_belief, opponent_model)
        self.belief = TomZeroSubjectBelief(prior_belief, opponent_model)
        self.environment_model = ToMZeroSubjectEnvironmentModel(opponent_model, self.utility_function)
        self.exploration_policy = ToMZeroSubjectExplorationPolicy(self.actions, self.utility_function, 0.3)
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)

    def utility_function(self, action, observation):
        """

        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        return (1 - action) * observation

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
        self.forward(action, observation, iteration_number)

    def forward(self, action=None, observation=None, iteration_number=None):
        actions, q_values = self.solver.plan(action, observation, iteration_number)
        return actions, q_values
