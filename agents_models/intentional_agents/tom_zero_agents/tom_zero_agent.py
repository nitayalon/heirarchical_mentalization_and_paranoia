from agents_models.abstract_agents import *
from IPOMCP_solver.Solver.ipomcp_solver import *
import os


class TomZeroAgentBelief(DoMZeroBelief):
    def __init__(self, intentional_threshold_belief, opponent_model: SubIntentionalModel):
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
            self.opponent_model.threshold = theta
            possible_opponent_actions, opponent_q_values, probabilities = \
                self.opponent_model.forward(last_observation, action)
            # If the observation is not in the feasible action set then it singles theta hat:
            observation_probability = probabilities[np.where(possible_opponent_actions == observation)]
            offer_likelihood[i] = observation_probability
        self.opponent_model.threshold = original_threshold
        return offer_likelihood


class ToMZeroAgentEnvironmentModel(EnvironmentModel):

    def __init__(self, opponent_model: SubIntentionalModel, reward_function,
                 belief_distribution: TomZeroAgentBelief):
        super().__init__(opponent_model, belief_distribution)
        self.reward_function = reward_function
        self.opponent_model = opponent_model

    def reset_persona(self, persona, history_length, nested_beliefs):
        self.opponent_model.threshold = persona

    def step(self, interactive_state: InteractiveState, action: Action, observation: Action, seed: int,
             iteration_number: int):
        counter_offer, q_values = self.opponent_model.act(seed, observation.value, action.value)
        reward = self.reward_function(observation.value, action.value, interactive_state.persona)
        interactive_state.state.name = str(int(interactive_state.state.name) + 1)
        interactive_state.state.terminal = interactive_state.state.name == 10
        return interactive_state, Action(counter_offer, False), reward


class ToMZeroAgentExplorationPolicy:

    def __init__(self, actions, reward_function, exploration_bonus):
        self.reward_function = reward_function
        self.actions = actions
        self.exploration_bonus = exploration_bonus

    def sample(self, interactive_state: InteractiveState, last_action: bool, observation: float,
               rng_key: int, iteration_number):
        # if the last offer was rejected - we should narrow down the search space
        potential_actions = self.actions
        if not observation:
            potential_actions = self.actions[self.actions < last_action]
        expected_reward_from_offer = self.reward_function(potential_actions) * \
                                     (interactive_state.persona >= self.actions)
        optimal_action_idx = np.argmax(expected_reward_from_offer)
        optimal_action = potential_actions[optimal_action_idx]
        q_value = expected_reward_from_offer[optimal_action_idx]
        return Action(optimal_action, False), q_value

    def init_q_values(self, observation: Action):
        initial_qvalues = self.reward_function(observation.value, self.actions, None, False)
        return initial_qvalues


class DoMZeroAgent(DoMZeroModel):

    def __init__(self,
                 actions,
                 softmax_temp: float,
                 prior_belief: np.array,
                 opponent_model: SubIntentionalModel,
                 seed: int,
                 alpha: float):
        super().__init__(actions, softmax_temp, prior_belief, opponent_model, alpha)
        self.config = get_config()
        self.belief = TomZeroAgentBelief(prior_belief, self.opponent_model)
        self.environment_model = ToMZeroAgentEnvironmentModel(self.opponent_model, self.utility_function, self.belief)
        self.exploration_policy = ToMZeroAgentExplorationPolicy(self.potential_actions, self.utility_function,
                                                                self.config.get_from_env("rollout_rejecting_bonus"))
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)
        self.name = "DoM(0)_agent"

    def utility_function(self, action, observation, theta_hat=None, final_trial=True):
        """
        :param theta_hat: float - representing the true persona of the opponent
        :param final_trial: bool - indicate if last trial or not
        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        game_reward = (1 - action) * observation
        return game_reward

    def act(self, seed, action=None, observation=None, iteration_number=None) -> [float, np.array]:
        self.belief.history.update_observations(observation)
        action_nodes, q_values, mcts_tree = self.forward(action, observation, iteration_number)
        mcts_tree["alpha"] = self.alpha
        mcts_tree["softmax_temp"] = self.softmax_temp
        mcts_tree["agent_type"] = self.name
        mcts_tree_output_name = os.path.join(self.config.planning_results_dir,
                                             self.config.experiment_name)
        mcts_tree.to_csv(mcts_tree_output_name + f'_iteration_number_{iteration_number}_seed_{self.config.seed}.csv',
                         index=False)
        softmax_transformation = np.exp(q_values[:, 1] / self.softmax_temp) / np.exp(q_values[:, 1] / self.softmax_temp).sum()
        prng = np.random.default_rng(seed)
        best_action_idx = prng.choice(a=len(action_nodes), p=softmax_transformation)
        actions = list(action_nodes.keys())
        best_action = action_nodes[actions[best_action_idx]].action
        self.belief.history.update_actions(best_action.value)
        if action_nodes is not None:
            self.solver.action_node = action_nodes[str(best_action.value)]
        return best_action.value, q_values[:, :-1]

    def forward(self, action=None, observation=None, iteration_number=None):
        actions, mcts_tree, q_values = self.solver.plan(action, observation, iteration_number)
        return actions, q_values, mcts_tree
