import pandas as pd
import itertools
import os
from dom_two_agent import *
import argparse
from collections import namedtuple

TaskSetting = namedtuple("TaskSetting", ["Task_duration", "Seed", "Game", "Temp", "ExportResults", "AlephIPOMDP"])


def export_zero_sum_game_results(path_to_data_dir, beliefs, actions, payoffs, row_player_dom_level,
                                 column_player_dom_level):
    os.makedirs(f"{path_to_data_dir}/payoffs/", exist_ok=True)
    os.makedirs(f"{path_to_data_dir}/actions/", exist_ok=True)
    os.makedirs(f"{path_to_data_dir}/beliefs/", exist_ok=True)
    payoff_df = pd.DataFrame(payoffs, columns=["iteration", "row_reward", "column_reward"])
    actions_df = pd.DataFrame(actions, columns=["iteration", "row_action", "column_action"])
    beliefs_df = pd.DataFrame(beliefs, columns=["p_uninformed", "p_game_1", "p_game_2"])
    beliefs_df["iteration"] = np.arange(0, duration + 1)
    beliefs_df["seed"] = seed
    actions_df["seed"] = seed
    payoff_df["seed"] = seed
    beliefs_df["row_player_dom_level"] = row_player_dom_level
    actions_df["row_player_dom_level"] = row_player_dom_level
    payoff_df["row_player_dom_level"] = row_player_dom_level
    beliefs_df["column_player_dom_level"] = column_player_dom_level
    actions_df["column_player_dom_level"] = column_player_dom_level
    payoff_df["column_player_dom_level"] = column_player_dom_level
    beliefs_df.to_csv(
        f"{path_to_data_dir}/beliefs/dom_{row_player_dom_level}_vs_dom_{column_player_dom_level}_seed_{seed}.csv",
        index=False)
    payoff_df.to_csv(
        f"{path_to_data_dir}/payoffs/dom_{row_player_dom_level}_vs_dom_{column_player_dom_level}_seed_{seed}.csv",
        index=False)
    actions_df.to_csv(
        f"{path_to_data_dir}/actions/dom_{row_player_dom_level}_vs_dom_{column_player_dom_level}_seed_{seed}.csv",
        index=False)


def simulate_row_column_task(path_to_data_dir, column_player_updated_beliefs, dom_levels,
                             row_player, column_player,
                             task_configuration: TaskSetting):
    row_player_dom_level = dom_levels[0]
    column_player_dom_level = dom_levels[1]
    payoffs = np.array([0, 0, 0])
    actions = np.array([0, 0, 0])
    column_player_beliefs = column_player_updated_beliefs
    nested_row_player_beliefs = column_player_updated_beliefs
    payout_matrix = task_configuration.Game
    task_duration = task_configuration.Task_duration
    rng_seed = task_configuration.Seed
    for i in np.arange(0, task_duration):
        row_player_policy = row_player.act(nested_row_player_beliefs, payout_matrix, i)
        column_player_policy = column_player.act(column_player_updated_beliefs)
        prng = np.random.default_rng(rng_seed + i)
        column_player_action = prng.choice(a=3, p=column_player_policy)
        row_player_action = prng.choice(a=2, p=row_player_policy)
        reward = payout_matrix[row_player_action, column_player_action]
        payoffs = np.vstack([payoffs, np.array([i, reward, -reward])])
        nested_row_player_beliefs = dom_zero_agent.irl(observation=row_player_action,
                                                       prior=nested_row_player_beliefs, iteration=i)
        column_player_updated_beliefs = column_player.irl(observation=row_player_action,
                                                          prior=column_player_updated_beliefs, iteration=i)
        column_player.beliefs.append(column_player_updated_beliefs)
        if task_configuration.AlephIPOMDP:
            row_player.observations.append(column_player_action)
            column_player.actions.append(column_player_action)
        column_player_beliefs = np.vstack([column_player_beliefs, column_player_updated_beliefs])
        actions = np.vstack([actions, np.array([i, column_player_action, row_player_action])])
    if task_configuration.ExportResults == "True":
        export_zero_sum_game_results(path_to_data_dir, column_player_beliefs, actions, payoffs, row_player_dom_level,
                                     column_player_dom_level)


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
    parser.add_argument('--aleph_ipomdp', type=str, default='True', metavar='N',
                        help='Aleph-IPOMDP env (default: True)')
    parser.add_argument('--strong_typicality_delta', type=float, default='0.8', metavar='N',
                        help='Strong typicality delta (default: True)')
    args = parser.parse_args()
    duration = args.duration
    seed = args.seed
    payout_matrix_name = args.payout_matrix
    softmax_temp = args.softmax_temp
    save_results = args.save_results
    is_aleph_ipomdp = args.aleph_ipomdp == "True"
    strong_typicality_delta = args.strong_typicality_delta
    payout_game = game_1 if payout_matrix_name == 'G_1' else game_2
    task_setting = TaskSetting(duration, seed, payout_game, softmax_temp, save_results, is_aleph_ipomdp)
    random_agent = RandomPlayer(softmax_temp, 0.99)
    dom_m1_agent = DoMM1Player(softmax_temp, 0.99)
    dom_zero_agent = DoMZeroPlayer(duration, softmax_temp, 0.99)
    dom_one_agent = DoMOnePlayer(duration, softmax_temp, 0.99, is_aleph_ipomdp, strong_typicality_delta)
    initial_beliefs = np.repeat(1 / 3, 3)
    dom_two_agent = DoMTwoPlayer(duration, softmax_temp, 0.99, initial_beliefs, is_aleph_ipomdp,
                                 strong_typicality_delta)
    agents_dictionary = {"-2": random_agent, "0": dom_zero_agent, "1": dom_one_agent, "2": dom_two_agent}
    path_to_results_dir = f"data/aleph_ipomdp_{is_aleph_ipomdp}/{payout_matrix_name}"
    os.makedirs(path_to_results_dir, exist_ok=True)
    agents_list = [["1"], ["0", "2"]]
    for dyad in itertools.product(*agents_list):
        simulate_row_column_task(path_to_results_dir, initial_beliefs, dyad, agents_dictionary[dyad[0]],
                                 agents_dictionary[dyad[1]], task_setting)
