from IPOMCP_solver.Solver.nodes import *
from IPOMCP_solver.Solver.abstract_classes import *
from IPOMCP_solver.Solver.ipomcp_config import get_config
import time
# import networkx as nx
# import matplotlib.pyplot as plt
# from IPOMCP_solver.utils.logger import get_logger


class IPOMCP:

    def __init__(self,
                 root_sampling: BeliefDistribution,
                 environment_simulator: EnvironmentModel,
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
        self.exploration_bonus = float(self.config.get_from_env("uct_exploration_bonus"))
        self.depth = float(self.config.get_from_env("planning_depth"))
        self.n_iterations = int(self.config.get_from_env("mcts_number_of_iterations"))
        self.discount_factor = float(self.config.get_from_env("discount_factor"))
        self.softmax_temperature = float(self.config.softmax_temperature)
        self.name = "IPOMCP"

    def reset(self):
        self.history_node = None
        self.action_node = None

    def plan(self, offer: Action, counter_offer: Action,
             iteration_number, update_belief):
        """
        
        :param iteration_number:
        :param offer:  action_{t-1}
        :param counter_offer: observation_{t}
        :return: action_node
        :param update_belief:
        """
        action_length = len(self.root_sampling.history.actions)
        observation_length = len(self.root_sampling.history.observations)
        if self.action_node is None or str(counter_offer) not in self.action_node.children:
            previous_counter_offer = self.root_sampling.history.get_last_observation()
            base_node = HistoryNode(None, previous_counter_offer, self.action_exploration_policy)
            offer_node = base_node.add_action_node(offer)
            self.history_node = offer_node.add_history_node(counter_offer, self.action_exploration_policy)
        else:
            self.history_node = self.action_node.children[str(counter_offer)]
        self.root_sampling.update_distribution(offer, counter_offer, iteration_number)
        root_samples = self.root_sampling.sample(self.seed, n_samples=self.n_iterations)
        self.action_exploration_policy.update_belief(self.root_sampling.belief_distribution)
        iteration_times = []
        depth_statistics = []
        for i in range(self.n_iterations):
            persona = root_samples[i]
            self.environment_simulator.reset_persona(persona, action_length, observation_length,
                                                     self.root_sampling.opponent_model.belief.belief_distribution[:iteration_number, :])
            nested_belief = self.environment_simulator.opponent_model.belief.get_current_belief()
            interactive_state = InteractiveState(State(str(i), False), persona, nested_belief)
            self.history_node.particles.append(interactive_state)
            start_time = time.time()
            _, _, depth = self.simulate(i, interactive_state, self.history_node, 0, self.seed, iteration_number)
            end_time = time.time()
            iteration_time = end_time - start_time
            iteration_times.append([persona, iteration_time])
            depth_statistics.append([persona, depth])
        self.environment_simulator.reset_persona(None, action_length, observation_length,
                                                 self.root_sampling.opponent_model.belief.belief_distribution[:iteration_number, :])
        # Reporting iteration time
        if self.config.report_ipocmp_statistics:
            iteration_time_for_logging = pd.DataFrame(iteration_times)
            iteration_time_for_logging.columns = ["persona", "time"]
        if self.config.output_planning_tree:
            optimal_tree, optimal_tree_beliefs = self.extract_max_q_value_trajectory(self.history_node)
            optimal_tree_table = pd.DataFrame(optimal_tree, columns=['node_type', 'parent_id', 'self_id', 'parent_value',
                                                                     'self_value', 'q_value'])
        else:
            optimal_tree_table = None
        return self.history_node.children, optimal_tree_table, \
               np.c_[self.history_node.children_qvalues, self.history_node.children_visited[:, 1]]

    def simulate(self, trail_number, interactive_state: InteractiveState,
                 history_node: HistoryNode, depth,
                 seed: int, iteration_number):
        action_node = history_node.select_action(interactive_state,
                                                 history_node.parent.action,
                                                 history_node.observation,
                                                 True, iteration_number)
        if depth >= self.depth:
            reward = self.environment_simulator.reward_function(history_node.observation.value,
                                                                action_node.action.value)
            return reward, True, depth
        action_node.append_particle(interactive_state)
        # If the selected action is terminal
        if action_node.action.is_terminal:
            history_node.increment_visited()
            action_node.increment_visited()
            return self._halting_action_reward(action_node.action, history_node.observation.value), True, depth

        new_interactive_state, observation, reward = \
            self.environment_simulator.step(interactive_state,
                                            action_node.action,
                                            history_node.observation,
                                            seed, iteration_number + 1)
        history_node.update_reward(action_node.action, reward)
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
            return reward, observation.is_terminal, depth

        if new_observation_flag:
            action_node.children[str(new_history_node.observation)] = new_history_node
            future_reward, is_terminal, depth = self.rollout(trail_number, new_interactive_state,
                                                             action_node.action, observation, depth + 1,
                                                             seed, iteration_number + 1)
        else:
            future_reward, is_terminal, depth = self.simulate(trail_number, new_interactive_state, new_history_node,
                                                              depth + 1, seed, iteration_number + 1)
        total = reward + self.discount_factor * future_reward
        history_node.increment_visited()
        action_node.increment_visited()
        action_node.update_q_value(total)
        return total, observation.is_terminal, depth

    def rollout(self, trail_number, interactive_state: InteractiveState, last_action: Action, observation: Action,
                depth, seed: int,
                iteration_number) -> [float, bool, int]:
        if depth >= self.depth:
            reward = self.environment_simulator.reward_function(observation.value,
                                                                last_action.value)
            return reward, True, depth
        action, _ = self.action_exploration_policy.sample(interactive_state,
                                                          last_action.value, observation.value,
                                                          iteration_number)
        if action.is_terminal:
            reward = self._halting_action_reward(action, observation.value)
            return reward, True, depth
        new_interactive_state, observation, reward = \
            self.environment_simulator.step(interactive_state, action, observation, seed, iteration_number + 1)
        if observation.is_terminal:
            return reward, observation.is_terminal, depth
        else:
            future_reward, is_terminal, depth = self.rollout(trail_number, new_interactive_state, action, observation, depth + 1,
                                                             seed, iteration_number + 1)
        total = reward + self.discount_factor * future_reward
        return total, observation.is_terminal, depth

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
    def extract_max_q_value_trajectory(self, root_node: HistoryNode, planning_tree=None, belief_tree=None):
        if planning_tree is None or belief_tree is None:
            tree = [["root", None, root_node.id, root_node.parent.action.value, root_node.observation.value, 0.0]]
            beliefs = [["root", None, root_node.id,  root_node.summarize_particles_distribution()]]
        else:
            tree = planning_tree
            beliefs = belief_tree
        max_q_value_action = np.argmax(root_node.children_qvalues[:, 1])
        optimal_child = root_node.children[str(root_node.children_values[max_q_value_action])]
        tree.append(["action", root_node.id, optimal_child.id, optimal_child.parent.observation.value,
                     optimal_child.action.value, optimal_child.q_value])
        beliefs.append(["action", root_node.id, optimal_child.id, optimal_child.parent.observation.value,
                     optimal_child.action.value, optimal_child.summarize_particles_distribution()])
        tree, beliefs = self.extract_max_value_trajectory(optimal_child, tree, beliefs)
        return tree, beliefs

    def extract_max_value_trajectory(self, root_node: ActionNode, planning_tree, beliefs_tree):
        for potential_observation in root_node.children:
            child = root_node.children[potential_observation]
            node = ["observation",
                    root_node.id, child.id,
                    child.parent.action.value,
                    child.observation.value,
                    child.compute_node_value()]
            beliefs = ["observation", root_node.id, child.id, child.parent.action.value,
                       child.observation.value, root_node.summarize_particles_distribution()]
            planning_tree.append(node)
            beliefs_tree.append(beliefs)
            planning_tree, beliefs_tree = self.extract_max_q_value_trajectory(child, planning_tree, beliefs_tree)
        return planning_tree, beliefs_tree




