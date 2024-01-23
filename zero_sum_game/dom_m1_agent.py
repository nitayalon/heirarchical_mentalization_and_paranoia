import numpy as np
import argparse


class DoMM1Player:

    def __init__(self, softmax_temperature: float, discount_factor: float) -> None:
        self.softmax_temperature = softmax_temperature
        self.discount_factor = discount_factor

    def act(self, prior_beliefs, payout_matrix, iteration):
        q_values = dom_m1_utility(payout_matrix)
        return softmax_transformation(q_values, self.softmax_temperature)


class RandomPlayer:

    def __init__(self, softmax_temperature: float, discount_factor: float) -> None:
        self.softmax_temperature = softmax_temperature
        self.discount_factor = discount_factor

    def act(self, prior_beliefs, payout_matrix, iteration):
        q_values = np.repeat(1, 2)
        return softmax_transformation(q_values, self.softmax_temperature)


def softmax_transformation(q_values: np.array, temperature=0.01) -> np.array:
    softmax_transformation = np.exp(q_values / temperature)
    return softmax_transformation / np.sum(softmax_transformation)


def dom_m1_utility(game: np.array):
    return np.matmul(game, np.array([1 / 2, 1 / 2, 1 / 2]))


game_1 = np.array([[4, 0, 2], [4, 0, -2]])
game_2 = np.array([[0, 4, -2], [0, 4, 2]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cognitive hierarchy task')
    parser.add_argument('--payout_matrix', type=str, default='G1', metavar='N',
                        help='payout matrix (default: G1)')
    parser.add_argument('--duration', type=int, default='12', metavar='N',
                        help='task duration (default: 12)')
    parser.add_argument('--seed', type=int, default='6431', metavar='N',
                        help='set simulation seed (default: 6431)')
    parser.add_argument('--softmax_temp', type=float, default='0.05', metavar='N',
                        help='set softmax temp (default: 0.05)')
    parser.add_argument('--save_results', type=str, default='True', metavar='N',
                        help='save simulation results (default: True)')
    args = parser.parse_args()
    duration = args.duration
    seed = args.seed
    payout_matrix_name = args.payout_matrix
    payout_game = game_1 if payout_matrix_name == 'G1' else game_2
    initial_beliefs = np.repeat(1 / 2, 2)
    softmax_temp = args.softmax_temp
    save_results = args.save_results
    dom_m1_agent = DoMM1Player(softmax_temp, 0.99)
    random_agent = RandomPlayer(softmax_temp, 0.99)
    print(dom_m1_agent.act(None, payout_game, 0))
    print(random_agent.act(None, payout_game, 0))
