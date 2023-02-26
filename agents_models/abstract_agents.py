from IPOMCP_solver.Solver.ipomcp_solver import *
import os


class SubIntentionalBelief(BeliefDistribution):

    def __init__(self, history: History):
        super().__init__(None, None, history)

    def get_current_belief(self):
        return None

    def update_distribution(self, action, observation, first_move):
        return None

    def sample(self, rng_key, n_samples):
        return None

    def update_history(self, action, observation, reward):
        self.history.update_history(action, observation, reward)


class BasicModel(ABC):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        self.config = get_config()
        self.potential_actions = actions
        self._threshold = threshold
        self.softmax_temp = softmax_temp
        self.high = 1.0
        self.low = 0.0
        self.name = None
        self.history = History()
        self.belief = SubIntentionalBelief(self.history)
        self._alpha = None

    def reset(self):
        self.high = 1.0
        self.low = 0.0
        self.reset_belief()
        self.reset_solver()

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, gamma):
        self._threshold = gamma

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    def softmax_transformation(self, q_values):
        softmax_transformation = np.exp(q_values / self.softmax_temp)
        return softmax_transformation / softmax_transformation.sum()

    def utility_function(self, action, observation):
        pass

    def act(self, seed, action=None, observation=None, iteration_number=None) -> [float, np.array]:
        self.update_bounds(action, observation)
        seed = self.update_seed(seed, iteration_number)
        relevant_actions, q_values, probabilities = self.forward(action, observation)
        random_number_generator = np.random.default_rng(seed)
        optimal_offer = random_number_generator.choice(relevant_actions, p=probabilities)
        return Action(optimal_offer, False), np.array([relevant_actions, q_values]).T

    @abstractmethod
    def forward(self, action=None, observation=None):
        pass

    def update_bounds(self, action, observation):
        pass

    def update_history(self, action: Action, observation: Action, reward: float):
        self.history.update_history(action, observation, reward)

    def update_seed(self, seed, number):
        return seed

    def reset_solver(self):
        pass

    def reset_belief(self):
        pass


class DoMZeroBelief(BeliefDistribution):

    def __init__(self, intentional_threshold_belief: np.array, opponent_model: Optional[BasicModel],
                 history: History):
        """
        :param intentional_threshold_belief: np.array - represents the prior belief about the agent_parameters
        :param opponent_model:
        """
        super().__init__(intentional_threshold_belief, opponent_model, history)
        self.opponent_belief = None

    def compute_likelihood(self, action, observation, prior):
        pass

    def update_distribution(self, action, observation, first_move):
        """
        Update the belief based on the last action and observation (IRL)
        :param action:
        :param observation:
        :param first_move:
        :return:
        """
        if first_move < 1:
            return None
        prior = np.copy(self.belief_distribution[:, -1])
        probabilities = self.compute_likelihood(action.value, observation.value, prior)
        posterior = probabilities * prior
        self.belief_distribution = np.c_[self.belief_distribution, posterior / posterior.sum()]

    def sample(self, rng_key, n_samples):
        probabilities = self.belief_distribution[:, -1]
        rng_generator = np.random.default_rng(rng_key)
        particles = rng_generator.choice(self.belief_distribution[:, 0], size=n_samples, p=probabilities)
        return particles

    def reset_belief(self, history_length):
        self.belief_distribution = self.belief_distribution[:, 0:history_length+3]
        self.history.reset(history_length+2)


class DoMZeroEnvironmentModel(EnvironmentModel):

    def __init__(self, opponent_model: BasicModel,
                 reward_function, belief_distribution: DoMZeroBelief,
                 low=0.0, high=1.0):
        super().__init__(opponent_model, belief_distribution)
        self.reward_function = reward_function
        self.opponent_model = opponent_model
        self.low = low
        self.high = high

    def reset(self):
        self.opponent_model.reset()
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high

    def update_low_and_high(self, observation, action):
        self.opponent_model.update_bounds(observation, action)
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high

    def reset_persona(self, persona, history_length, nested_beliefs):
        self.opponent_model.threshold = persona
        observation = self.opponent_model.history.observations[history_length-1]
        action = self.opponent_model.history.actions[history_length-1]
        self.opponent_model.update_bounds(action, observation)

    def step(self, interactive_state: InteractiveState, action: Action, observation: Action, seed: int,
             iteration_number: int):
        counter_offer, q_values = self.opponent_model.act(seed, observation.value, action.value, iteration_number)
        interactive_state.state.terminal = interactive_state.state.name == 10
        reward = self.reward_function(observation.value, action.value,
                                      **{"final_trial": True,
                                         "theta_hat": interactive_state.persona,
                                         "counter_offer": counter_offer.value,
                                         "iteration_number": iteration_number})
        interactive_state.state.name = str(int(interactive_state.state.name) + 1)
        return interactive_state, counter_offer, reward

    def update_persona(self, observation, action):
        response = bool(action.value)
        self.opponent_model.low = self.low
        self.opponent_model.high = self.high
        self.opponent_model.update_bounds(observation, response)
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high


class DoMZeroModel(BasicModel):

    def __init__(self, actions,
                 softmax_temp: float,
                 threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: BasicModel,
                 seed: int):
        super().__init__(actions, softmax_temp, threshold)
        self.history = History()
        self.opponent_model = opponent_model
        self.belief = DoMZeroBelief(prior_belief, self.opponent_model, self.history)  # type: DoMZeroBelief
        self.environment_model = DoMZeroEnvironmentModel(self.opponent_model, self.utility_function, self.belief)
        self.solver = IPOMCP(self.belief, self.environment_model, None, self.utility_function, seed)

    def act(self, seed, action=None, observation=None, iteration_number=None) -> [float, np.array]:
        if iteration_number > 0:
            self.history.update_observations(observation)
        action_nodes, q_values, mcts_tree = self.forward(action, observation, iteration_number)
        mcts_tree["alpha"] = self.alpha
        mcts_tree["softmax_temp"] = self.softmax_temp
        mcts_tree["agent_type"] = self.name
        mcts_tree_output_name = os.path.join(self.config.planning_results_dir,
                                             self.config.experiment_name)
        mcts_tree.to_csv(mcts_tree_output_name + f'_iteration_number_{iteration_number}_seed_{self.config.seed}.csv',
                         index=False)
        softmax_transformation = np.exp(q_values[:, 1] / self.softmax_temp) / np.exp(
            q_values[:, 1] / self.softmax_temp).sum()
        prng = np.random.default_rng(seed)
        best_action_idx = prng.choice(a=len(action_nodes), p=softmax_transformation)
        actions = list(action_nodes.keys())
        best_action = action_nodes[actions[best_action_idx]].action
        self.environment_model.update_persona(observation, best_action)
        if action_nodes is not None:
            self.solver.action_node = action_nodes[str(best_action.value)]
        return best_action, q_values[:, :-1]

    def forward(self, action=None, observation=None, iteration_number=None):
        actions, mcts_tree, q_values = self.solver.plan(action, observation, iteration_number)
        return actions, q_values, mcts_tree

    def reset_belief(self):
        self.belief.reset()

    def reset_solver(self):
        self.solver.reset()

    def update_history(self, action, observation, reward):
        """
        Method helper for history update - append the last action and observation
        :param reward:
        :param action:
        :param observation:
        :return:
        """
        self.history.update_history(action, observation, reward)
        self.opponent_model.history.update_history(observation, action, None)

