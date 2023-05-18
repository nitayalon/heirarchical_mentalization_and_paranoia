import numpy as np

from IPOMCP_solver.Solver.ipomcp_solver import *
import os


class SubIntentionalBelief(BeliefDistribution):

    def __init__(self, history: History):
        super().__init__(None, None, None, history)

    def get_current_belief(self):
        return None

    def update_distribution(self, action, observation, first_move):
        return None

    def sample(self, rng_key, n_samples):
        return None

    def update_history(self, action, observation, reward):
        self.history.update_history(action, observation, reward)


class SubIntentionalAgent(ABC):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        self.config = get_config()
        self.potential_actions = actions
        self._threshold = threshold
        self._duration = self.config.task_duration
        self.softmax_temp = softmax_temp
        self.upper_bounds = [1 - self.threshold if threshold is not None else 1.0] + ([None] * (self._duration - 1))
        self.lower_bounds = [0.0] + [None] * (self._duration - 1)
        self.low = self.lower_bounds[0]
        self.high = self.upper_bounds[0]
        self.name = None
        self.history = History()
        self.belief = SubIntentionalBelief(self.history)
        self._alpha = None

    def reset(self, high: Optional[float] = 1.0, low: Optional[float] = 0.0,
              iteration: Optional[int] = 1, terminal: Optional[bool] = False):
        self.low = low
        self.high = high
        # self.upper_bounds = self.upper_bounds[0:iteration] + ([None] * (self._duration - iteration))
        # self.lower_bounds = self.lower_bounds[0:iteration] + ([None] * (self._duration - iteration))
        self.reset_belief()
        self.reset_solver()
        if terminal:
            self.upper_bounds = [1 - self.threshold if self.threshold is not None else 1.0] + ([None] * (self._duration - 1))
            self.lower_bounds = [0.0] + [None] * (self._duration - 1)
            self.history.reset(0, 0)

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, gamma):
        self._threshold = gamma

    def softmax_transformation(self, q_values):
        softmax_transformation = np.exp(q_values / self.softmax_temp)
        return softmax_transformation / softmax_transformation.sum()

    def utility_function(self, action, observation):
        pass

    def act(self, seed, action: Optional[Action] = None, observation: Optional[Action] = None,
            iteration_number: Optional[int] = None) -> [float, np.array]:
        seed = self.update_seed(seed, iteration_number)
        relevant_actions, q_values, probabilities = self.forward(action, observation, iteration_number)
        random_number_generator = np.random.default_rng(seed)
        optimal_action_idx = random_number_generator.choice(len(relevant_actions), p=probabilities)
        optimal_offer = relevant_actions[optimal_action_idx]
        probability = probabilities[optimal_action_idx]
        return Action(optimal_offer, False), probability, np.array([relevant_actions, q_values]).T, probabilities

    @abstractmethod
    def forward(self, action: Action, observation: Action, iteration_number=None):
        pass

    def update_bounds(self, action: Action, observation: Action, iteration_number: Optional[int]):
        pass

    def update_history(self, action: Action, observation: Action, reward: float):
        self.history.update_history(action, observation, reward)

    def update_seed(self, seed, number):
        return seed

    def reset_solver(self):
        pass

    def reset_belief(self):
        pass

    @abstractmethod
    def update_nested_models(self, action=None, observation=None, iteration_number=None):
        pass


class DoMZeroBelief(BeliefDistribution):

    def __init__(self, support, zero_level_belief: Optional[np.array],
                 opponent_model: Optional[SubIntentionalAgent],
                 history: History):
        """
        :param zero_level_belief: np.array - represents the prior belief about the sender_parameters
        :param opponent_model:
        """
        super().__init__(support, zero_level_belief, opponent_model, history)
        self.opponent_belief = None

    def compute_likelihood(self, action, observation, prior, iteration_number=None):
        pass

    def update_distribution(self, action, observation, iteration_number):
        """
        Update the belief based on the last action and observation (IRL)
        :param action:
        :param observation:
        :param iteration_number:
        :return:
        """
        if iteration_number <= 0:
            return None
        prior = np.copy(self.belief_distribution[-1, :])
        probabilities = self.compute_likelihood(action, observation, prior, iteration_number)
        posterior = probabilities * prior
        self.belief_distribution = np.vstack([self.belief_distribution, posterior / posterior.sum()])

    def sample(self, rng_key, n_samples):
        probabilities = self.belief_distribution[:, -1]
        rng_generator = np.random.default_rng(rng_key)
        particles = rng_generator.choice(self.belief_distribution[:, 0], size=n_samples, p=probabilities)
        return particles

    def reset_belief(self, iteration_number, action_length, observation_length):
        self.belief_distribution = self.belief_distribution[:, 0:iteration_number+1]
        self.history.reset(action_length, observation_length)


class DoMZeroEnvironmentModel(EnvironmentModel):

    def __init__(self, opponent_model: SubIntentionalAgent,
                 reward_function,
                 actions: np.array,
                 belief_distribution: DoMZeroBelief,
                 low=0.0, high=1.0):
        super().__init__(opponent_model, belief_distribution)
        self.reward_function = reward_function
        self.actions = actions
        self.surrogate_actions = [Action(value, False) for value in self.actions]
        self.opponent_model = opponent_model
        self.low = [low]
        self.high = [high]

    def update_parameters(self):
        pass

    def reset(self):
        self.low = self.opponent_model.low
        self.high = self.opponent_model.high

    def update_low_and_high(self, observation, action, iteration_number):
        if action.value is None:
            return None
        self.low = observation.value * (1 - action.value) + self.low * action.value
        self.high = observation.value * action.value + self.high * (1 - action.value)

    def reset_persona(self, persona, action_length, observation_length, nested_beliefs):
        self.opponent_model.threshold = persona
        if action_length == 0 and observation_length == 0:
            return None
        self.opponent_model.reset(self.high, self.low, observation_length, False)
        self.opponent_model.history.reset(observation_length, action_length)
        self.opponent_model.belief.belief_distribution = nested_beliefs

    @staticmethod
    def _get_last_from_list(l, location):
        return l[location - 1] if len(l) > 0 else Action(None, False)

    def _simulate_opponent_response(self, seed, observation, action, iteration_number):
        counter_offer, observation_probability, q_values, opponent_policy = self.opponent_model.act(seed, observation,
                                                                                                    action,
                                                                                                    iteration_number)
        return counter_offer, observation_probability, q_values, opponent_policy

    def step(self, interactive_state: InteractiveState, action: Action, observation: Action, seed: int,
             iteration_number: int):
        counter_offer, observation_probability, q_values, opponent_policy = \
            self._simulate_opponent_response(seed, observation, action, iteration_number)
        reward = self.reward_function(action.value, observation.value, counter_offer.value)
        interactive_state.state.terminal = interactive_state.state.name == 10
        interactive_state.state.name = str(int(interactive_state.state.name) + 1)
        return interactive_state, counter_offer, reward, observation_probability

    def update_persona(self, observation, action, iteration_number):
        pass


class DoMZeroExplorationPolicy:
    def __init__(self, actions: np.array, reward_function, exploration_bonus: float, belief: np.array,
                 type_support: np.array):
        self.actions = actions
        self.reward_function = reward_function
        self.exploration_bonus = exploration_bonus
        self.belief = belief
        self.support = type_support

    def update_belief(self, belief: np.array):
        if belief.sum() != 1:
            return None
        self.belief = belief

    def sample(self, interactive_state: InteractiveState, last_action: float, observation: bool, iteration_number: int):
        pass

    def init_q_values(self, observation: Action):
        pass


class DoMZeroModel(SubIntentionalAgent):

    def __init__(self, actions,
                 softmax_temp: float,
                 threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: SubIntentionalAgent,
                 seed: int):
        super().__init__(actions, softmax_temp, threshold)
        self.opponent_model = opponent_model
        self.belief = DoMZeroBelief(prior_belief[:, 0], prior_belief[:, 1], self.opponent_model, self.history)  # type: DoMZeroBelief
        self.environment_model = DoMZeroEnvironmentModel(self.opponent_model, self.utility_function, actions, self.belief)
        self.solver = IPOMCP(self.belief, self.environment_model, None, None, self.utility_function, {}, seed)

    def reset(self, high: Optional[float] = None, low: Optional[float] = None,
              action_length: Optional[float] = 0, observation_length: Optional[float] = 0,
              terminal: Optional[bool] = False):
        self.high = 1.0
        self.low = 0.0
        self.history.reset(action_length, observation_length)
        self.opponent_model.reset(1.0, 0.0, terminal=terminal)
        self.environment_model.reset()
        self.reset_belief()
        self.reset_solver()

    def act(self, seed, action=None, observation=None, iteration_number=None) -> [float, np.array]:
        if iteration_number > 0:
            self.history.update_observations(observation)
            self.opponent_model.history.update_actions(observation)
            self.update_nested_models(action, observation, iteration_number)
        action_nodes, q_values, softmax_transformation, mcts_tree = self.forward(action, observation, iteration_number)
        if mcts_tree is not None:
            mcts_tree["softmax_temp"] = self.softmax_temp
            mcts_tree["agent_type"] = self.name
            mcts_tree_output_name = os.path.join(self.config.planning_results_dir,
                                                 self.config.experiment_name)
            mcts_tree.to_csv(mcts_tree_output_name + f'_iteration_number_{iteration_number}_seed_{self.config.seed}.csv',
                             index=False)
        prng = np.random.default_rng(seed)
        best_action_idx = prng.choice(a=len(action_nodes), p=softmax_transformation)
        actions = list(action_nodes.keys())
        action_probability = softmax_transformation[best_action_idx]
        if self.solver.name == "IPOMCP":
            best_action = action_nodes[actions[best_action_idx]].action
        else:
            best_action = action_nodes[actions[best_action_idx]]
        self.environment_model.update_persona(observation, best_action, iteration_number)
        self.history.update_actions(best_action)
        self.environment_model.opponent_model.history.update_observations(best_action)
        self.environment_model.update_parameters()
        if action_nodes is not None:
            self.solver.action_node = action_nodes[str(best_action.value)]
        return best_action, action_probability, q_values[:, :-1], softmax_transformation

    def forward(self, action=None, observation=None, iteration_number=None, update_belief=True):
        actions, mcts_tree, q_values = self.solver.plan(action, observation, iteration_number, update_belief)
        softmax_transformation = np.exp(q_values[:, 1] / self.softmax_temp) / np.exp(
            q_values[:, 1] / self.softmax_temp).sum()
        return actions, q_values, softmax_transformation, mcts_tree

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

    def update_nested_models(self, action=None, observation=None, iteration_number=None):
        pass
