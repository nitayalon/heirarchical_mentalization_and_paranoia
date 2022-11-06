import pandas as pd
from utils.logger import *
from enviroment.nodes import *
from enviroment.abstract_classes import *
from ipomcp_config import get_config
import time
# import networkx as nx
# import matplotlib.pyplot as plt


class IPOMCP:

    def __init__(self,
                 root_sampling,
                 environment_simulator,
                 exploration_policy,
                 reward_function,
                 seed: int):
        """

        :param root_sampling: Generative model for sampling root IS particles
        :param environment_simulator: Generative model for simulating environment dynamics
        :param exploration_policy: An exploration policy class
        :param reward_function: reward function for computation of accept value
        """
        self.root_sampling = root_sampling
        self.environment_simulator = environment_simulator
        self.action_exploration_policy = exploration_policy
        self.reward_function = reward_function
        self.config = get_config()
        self.seed = seed
        self.tree = dict()
        self.history_node = None
        self.action_node = None
        self.exploration_bonus = float(self.config.get_from_env("exploration_bonus"))
        self.depth = float(self.config.get_from_env("planning_depth"))
        self.n_iterations = int(self.config.get_from_env("mcts_number_of_iterations"))
        self.softmax_temperature = float(self.config.softmax_temperature)
        self.pruning_epsilon = 0.01

    def plan(self, offer, counter_offer,
             iteration_number):
        """
        
        :param iteration_number:
        :param offer:  action_{t-1}
        :param counter_offer: observation_{t}
        :return: action_node
        """
        previous_counter_offer = self.root_sampling.history.get_last_observation()
        current_history_length = self.root_sampling.history.length()
        base_node = HistoryNode(None, Action(previous_counter_offer), self.action_exploration_policy)
        offer_node = base_node.add_action_node(Action(offer))
        if self.action_node is None or str(counter_offer) not in self.action_node.children:
            self.history_node = offer_node.add_history_node(Action(counter_offer), self.action_exploration_policy)
        else:
            self.history_node = self.action_node.children[str(counter_offer)]
        self.root_sampling.update_distribution(Action(offer), Action(counter_offer), iteration_number)
        root_samples = self.root_sampling.sample(self.seed, n_samples=self.n_iterations)
        iteration_times = []
        for i in range(self.n_iterations):
            persona = root_samples[i]
            self.environment_simulator.reset_persona(persona, current_history_length,
                                                     self.root_sampling.opponent_belief)
            nested_belief = self.environment_simulator.opponent_model.belief_distribution.get_current_belief()
            interactive_state = InteractiveState(None, persona, nested_belief)
            self.history_node.particles.append(interactive_state)
            start_time = time.time()
            self.simulate(i, interactive_state, self.history_node, self.depth, self.seed, True, iteration_number)
            end_time = time.time()
            iteration_time = end_time - start_time
            iteration_times.append([persona, iteration_time])
        iteration_time_for_logging = pd.DataFrame(iteration_times)
        iteration_time_for_logging.columns = ["persona", "time"]
        get_logger().info(iteration_time_for_logging.groupby("persona").describe().to_string())
        self.environment_simulator.reset_persona(current_history_length)
        # self.plot_max_value_trajectory(self.history_node)
        return self.history_node.children, \
               np.c_[self.history_node.children_qvalues, self.history_node.children_visited[:, 1]]

    def simulate(self, trail_number, interactive_state: InteractiveState,
                 history_node: HistoryNode, depth,
                 seed: int, tree: bool, iteration_number):
        if depth <= 0:
            return self._compute_terminal_tree_reward(interactive_state.persona, interactive_state.get_belief), True
        history_node.compute_deterministic_actions_reward(self.reward_function)
        action_node = history_node.select_action(interactive_state,
                                                 history_node.parent.action,
                                                 history_node.observation,
                                                 tree)
        action_node.append_particle(interactive_state)
        # If the selected action is terminal
        if action_node.action.is_terminal:
            history_node.increment_visited()
            action_node.increment_visited()
            return self._halting_action_reward(action_node.action, history_node.observation.value), True

        new_interactive_state, observation, q_value, reward, log_prob = \
            self.environment_simulator.act(interactive_state,
                                           action_node.action,
                                           history_node.observation, seed, iteration_number)
        new_observation_flag = True
        if str(observation.value) in action_node.children:
            new_observation_flag = False
            new_history_node = action_node.children[str(observation.value)]
        else:
            new_history_node = action_node.add_history_node(observation, self.action_exploration_policy,
                                                            is_terminal=observation.is_terminal)
        new_history_node.particles.append(interactive_state)
        if observation.is_terminal:
            history_node.increment_visited()
            action_node.increment_visited()
            new_history_node.increment_visited()
            action_node.update_q_value(reward)
            return reward, observation.is_terminal

        if new_observation_flag:
            action_node.children[str(new_history_node.observation)] = new_history_node
            future_reward, is_terminal = self.simulate(trail_number, new_interactive_state, new_history_node, depth - 1,
                                                       seed, False, iteration_number + 1)
            total = reward + future_reward
        else:
            future_reward, is_terminal = self.simulate(trail_number, new_interactive_state, new_history_node, depth - 1,
                                                       seed, True, iteration_number + 1)
            total = reward + future_reward
        history_node.increment_visited()
        action_node.increment_visited()
        action_node.update_q_value(total)
        return total, observation.is_terminal

    def _halting_action_reward(self, action, observation):
        reward = 0.0
        if action.value == -2:
            reward = self.reward_function(observation)
        return reward

    def _compute_terminal_tree_reward(self, persona, nested_belief):
        average_nested_persona = np.sum(nested_belief[:, 0] * nested_belief[:, 1])
        if self.action_exploration_policy.agent_type == "worker":
            split_pot = (persona - average_nested_persona) / 2
            final_offer = persona - split_pot
        else:
            split_pot = (average_nested_persona - persona) / 2
            final_offer = split_pot + persona
        reward = self.reward_function(final_offer)
        return reward

    # def plot_max_value_trajectory(self, root_node: HistoryNode):
    #     tree = self.extract_max_q_value_trajectory(root_node)
    #     g = nx.DiGraph()
    #     g.add_nodes_from(list(tree.keys()))
    #     g.add_edges_from(zip(tree.keys(), [x[0] for x in tree.values()]))
    #     nx.draw(g, with_labels=True)
    #     plt.draw()
    #     plt.show()
    #     return tree
    #
    # def extract_max_q_value_trajectory(self, root_node: HistoryNode, trajectory=None):
    #     if trajectory is None:
    #         tree = {'root': [root_node.observation.value, root_node.compute_node_value()]}
    #     else:
    #         tree = trajectory
    #     max_q_value_action = np.argmax(root_node.children_qvalues[:, 1])
    #     optimal_child = root_node.children[str(root_node.children_values[max_q_value_action])]
    #     tree[str(root_node.observation.value)] = [optimal_child.action.value, optimal_child.q_value]
    #     tree = self.extract_max_value_trajectory(optimal_child, tree)
    #     return tree
    #
    # def extract_max_value_trajectory(self, root_node: ActionNode, trajectory):
    #     for potential_observation in root_node.children:
    #         child = root_node.children[potential_observation]
    #         node = [child.observation.value,
    #                 child.compute_node_value()]
    #         trajectory[str(root_node.action.value)] = node
    #         trajectory = self.extract_max_q_value_trajectory(child, trajectory)
    #     return trajectory




