from typing import Callable, Union

import numpy as np
import numpy.random as npr

from IPOMCP_solver.Solver.ipomcp_config import get_config
from IPOMCP_solver.Solver.abstract_classes import *


class TreeNode(object):

    def __init__(self, parent):
        self.parent = parent
        self.children = {}
        self.config = get_config()

    def export_tree_to_json(self):
        pass


class ActionNode(TreeNode):

    def __init__(self, parent: TreeNode, action, is_terminal=False,
                 deterministic_q_value=0.0):
        self.action = action
        self.particles = []
        self.is_terminal = is_terminal
        self.deterministic_q_value = deterministic_q_value
        super().__init__(parent)

    def append_particle(self, interactive_state: InteractiveState):
        self.particles.append(interactive_state.persona)

    @property
    def q_value(self):
        if self.is_terminal:
            return self.deterministic_q_value
        idx = np.where(self.parent.children_visited[:, 0] == self.action.value)
        return self.parent.children_qvalues[idx, 1].item()

    @property
    def visit_counter(self):
        idx = np.where(self.parent.children_visited[:, 0] == self.action.value)
        return self.parent.children_visited[idx, 1].item()

    def update_q_value(self, reward):
        idx = np.where(self.parent.children_visited[:, 0] == self.action.value)
        self.parent.children_qvalues[idx, 1] = self.q_value + (reward - self.q_value) / self.visit_counter

    def increment_visited(self):
        idx = np.where(self.parent.children_visited[:, 0] == self.action.value)
        self.parent.children_visited[idx, 1] += 1

    def add_history_node(self, observation,
                         action_exploration_policy,
                         is_terminal: bool = False):
        history_node = HistoryNode(self, observation, action_exploration_policy,
                                   is_terminal=is_terminal)
        self.children[str(history_node.observation)] = history_node
        return history_node

    def __str__(self):
        return str(self.action)


class HistoryNode(TreeNode):

    def __init__(self,
                 parent: Union[TreeNode, None],
                 observation: Action,
                 exploration_policy,
                 is_terminal=False):
        super().__init__(parent)

        self.exploration_policy = exploration_policy
        self.observation = observation
        self.exploration_bonus = self.config.get_from_env("exploration_bonus")
        self.is_terminal = is_terminal
        self.init_node()

    def init_node(self):
        self.particles = []
        self.children_values = []
        self.children_visited = np.vstack((self.exploration_policy.actions,
                                           np.repeat(0, self.exploration_policy.actions.shape[0]))).T
        self.children_qvalues = np.vstack((self.exploration_policy.actions, self.init_q_value())).T
        self.visited_counter = 1
        for child in list(self.exploration_policy.actions):
            idx = np.where(list(self.exploration_policy.actions) == child)
            self.add_action_node(Action(child, False), False, self.children_qvalues[idx, 1])

    def init_q_value(self):
        exploration_reward = self.exploration_policy.reward_function(self.exploration_policy.actions, self.observation.value)
        if self.parent is None:
            return exploration_reward
        lower_bound = np.minimum(self.parent.action.value, self.observation.value)
        upper_bound = np.maximum(self.parent.action.value, self.observation.value)
        exploration_reward[np.where(np.logical_or(self.exploration_policy.actions < lower_bound,
                                                  self.exploration_policy.actions > upper_bound))] = -np.inf
        return exploration_reward

    @property
    def previous_observation(self) -> Action:
        if self.parent:
            if self.parent.parent:
                return self.parent.parent.observation

    def compute_deterministic_actions_reward(self, reward_func: Callable):
        if str(-2.0) not in self.children_values:
            reward = reward_func(self.observation.value)
            # Adding accept node
            self.add_action_node(Action(-2.0, True), True, reward)
            self.children_qvalues = np.r_[self.children_qvalues, np.array([[-2, reward]])]
            self.children_visited = np.r_[self.children_visited, np.array([[-2, 0]])]
            # Adding quit node
            self.add_action_node(Action(-1.0, True), True, 0.0)
            self.children_qvalues = np.r_[self.children_qvalues, np.array([[-1, 0.0]])]
            self.children_visited = np.r_[self.children_visited, np.array([[-1, 0]])]

    def add_action_node(self, action: Action, is_terminal: bool = False, default_q_value=0.0):
        if str(action.value) in self.children_values:
            new_action_node = self.children[str(action.value)]
        else:
            new_action_node = ActionNode(self, action, is_terminal=is_terminal,
                                         deterministic_q_value=default_q_value)
            self.children[str(action)] = new_action_node
            self.children_values.append(str(new_action_node))
        return new_action_node

    def select_action(self, interactive_state, last_cation, observation, in_tree=True) -> ActionNode:
        rng_key = npr.default_rng(get_config().seed)
        if in_tree:
            action = self.uct()
        else:
            action, q_value = self.exploration_policy.sample(interactive_state, last_cation.value, observation.value,
                                                             rng_key)
            action = self.add_action_node(action, action.is_terminal, q_value)
        return action

    def increment_visited(self):
        self.visited_counter += 1

    def uct(self):
        best_action = np.argmax(self.children_qvalues[:, 1] +
                                self.exploration_bonus * np.sqrt(
            np.log(self.visited_counter) / (self.children_visited[:, 1] + 1)))
        return self.children[str(self.children_values[best_action])]

    def rollout_policy(self, interactive_state, last_cation, observation, depth) -> ActionNode:
        rng = npr.default_rng(self.config.seed + depth)
        action, terminal = self.exploration_policy.sample(interactive_state, last_cation.value, observation.value, rng,
                                                          True)
        action = Action(action, terminal)
        new_action_node = self.add_action_node(action, is_terminal=terminal)
        return new_action_node

    def sample_interactive_state(self, seed):
        particles_probabilities = np.array([x[1] for x in self.particles])
        particles_probabilities = particles_probabilities[~np.isnan(particles_probabilities)]
        particles = [x[0] for x in self.particles]
        probs = np.array(particles_probabilities) / np.array(particles_probabilities).sum()
        rng = npr.default_rng(seed)
        particle_index = rng.choice(len(particles), 1, True, probs)
        return particles[particle_index[0]]

    def __str__(self):
        return str(self.observation)

    def compute_node_value(self):
        return np.mean(self.children_qvalues[:, 1])
