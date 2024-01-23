from dom_zero_agent import *
import functools


class DoMOnePlayer:

    def __init__(self, game_duration: int, softmax_temperature: float, discount_factor: float) -> None:
        self.game_duration = game_duration
        self.softmax_temperature = softmax_temperature
        self.discount_factor = discount_factor
        self.opponent = DoMZeroPlayer(game_duration, softmax_temperature, discount_factor)
        self.planning_tree = []
        self.nested_beliefs = []

    def dom_1_expected_utility(self, action: int, beliefs: np.array, payout_matrix: np.array):
        dom_zero_policy = softmax_transformation(self.opponent.act(beliefs), self.softmax_temperature)
        expected_reward = np.matmul(payout_matrix[action,], dom_zero_policy)
        return expected_reward

    def act(self, prior_beliefs, payout_matrix, iteration):
        q_values_array = np.zeros(2)
        for action in np.array([0, 1]):
            q_values_array[action] = self.recursive_tree_span(action, prior_beliefs, payout_matrix, iteration)
        policy = softmax_transformation(q_values_array / np.min(q_values_array), self.softmax_temperature)
        return policy

    def recursive_tree_span(self, action, beliefs, payout_matrix, iteration, depth=12):
        reward = self.dom_1_expected_utility(action, beliefs, payout_matrix)
        updated_belief = np.round(self.opponent.irl(beliefs, action, iteration), 3)
        # halting condition
        if iteration >= depth:
            # self.planning_tree.append(np.array([action, reward, reward, iteration]))
            # self.nested_beliefs.append(np.array([action, iteration, updated_belief[0], updated_belief[1],updated_belief[2]]))
            return reward
        actions = np.array([0, 1])
        expectimax_tree = functools.partial(self.recursive_tree_span,
                                            beliefs=updated_belief,
                                            payout_matrix=payout_matrix,
                                            iteration=iteration + 1)
        future_q_values = list(map(expectimax_tree, actions))
        q_value = reward + self.discount_factor * np.max(future_q_values)
        # self.planning_tree.append(np.array([action, reward, reward, iteration]))
        # self.nested_beliefs.append(np.array([action, iteration, updated_belief[0], updated_belief[1],updated_belief[2]]))        
        return q_value
