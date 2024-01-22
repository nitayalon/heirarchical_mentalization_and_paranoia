import numpy as np
import pandas as pd
import argparse
import itertools
import os
from dom_m1_agent import *
from dom_zero_agent import *
from dom_one_agent import * 
from dom_two_agent import * 

def simulate_row_column_task(path_to_data_dir, duration, updated_beliefs, dom_levels, row_player, column_player, seed: int):
    row_player_dom_level = dom_levels[0]
    column_player_dom_level = dom_levels[1]
    payoffs = np.array([0,0,0])
    actions = np.array([0,0,0])
    column_player_beliefs = updated_beliefs
    for i in np.arange(0, duration):                
        column_player_policy = column_player.act(updated_beliefs)
        row_player_policy = row_player.act(updated_beliefs, game_1, 0)
        prng = np.random.default_rng(seed)
        column_player_action = prng.choice(a=3, p=column_player_policy)
        row_player_action = prng.choice(a=2, p=row_player_policy)
        reward = game_1[row_player_action,column_player_action]
        payoffs = np.vstack([payoffs, np.array([i, reward, -reward])])
        updated_beliefs = column_player.irl(observation=row_player_action,prior=updated_beliefs, iteration=i)
        column_player_beliefs = np.vstack([column_player_beliefs, updated_beliefs])       
        actions = np.vstack([actions, np.array([i, column_player_action, row_player_action])])   
    os.makedirs(f"{path_to_data_dir}/payoffs/", exist_ok=True)
    os.makedirs(f"{path_to_data_dir}/actions/", exist_ok=True)
    os.makedirs(f"{path_to_data_dir}/beliefs/", exist_ok=True)
    payoff_df = pd.DataFrame(payoffs, columns=["iteration","row_reward","column_reward"])
    actions_df = pd.DataFrame(actions, columns=["iteration","row_action","column_action"])
    beliefs_df = pd.DataFrame(column_player_beliefs, columns=["p_random","p_game_1","p_game_2"])
    beliefs_df["iteration"] = np.arange(0, duration+1)
    beliefs_df["seed"] = seed
    actions_df["seed"] = seed
    payoff_df["seed"] = seed
    beliefs_df["row_player_dom_level"] = row_player_dom_level
    actions_df["row_player_dom_level"] = row_player_dom_level
    payoff_df["row_player_dom_level"] = row_player_dom_level
    beliefs_df["column_player_dom_level"] = column_player_dom_level
    actions_df["column_player_dom_level"] = column_player_dom_level
    payoff_df["column_player_dom_level"] = column_player_dom_level
    beliefs_df.to_csv(f"{path_to_data_dir}/beliefs/dom_{row_player_dom_level}_vs_dom_{column_player_dom_level}_seed_{seed}.csv", index=False)
    payoff_df.to_csv(f"{path_to_data_dir}/payoffs/dom_{row_player_dom_level}_vs_dom_{column_player_dom_level}_seed_{seed}.csv", index=False)
    actions_df.to_csv(f"{path_to_data_dir}/actions/dom_{row_player_dom_level}_vs_dom_{column_player_dom_level}_seed_{seed}.csv", index=False)    


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
    payout_game = args.payout_matrix           
    initial_beliefs = np.repeat(1/3,3)    
    softmax_temp = args.softmax_temp       

    dom_m1_agent = DoMM1Player(softmax_temp, 0.99)
    dom_zero_agent = DoMZeroPlayer(duration, softmax_temp, 0.99)
    dom_one_agent = DoMOnePlayer(duration, softmax_temp, 0.99)
    dom_two_agent = DoMTwoPlayer(duration, softmax_temp, 0.99, initial_beliefs)
    agents_dictionary = {"-1": dom_m1_agent, "0": dom_zero_agent, "1": dom_one_agent, "2": dom_two_agent}
    path_to_data_dir = f"data/{payout_game}"
    os.makedirs(path_to_data_dir, exist_ok=True)        
    agents_list = [["-1","1"],["0","2"]]    
    for dyad in itertools.product(*agents_list):
        row_player = agents_dictionary[dyad[0]]
        column_player = agents_dictionary[dyad[1]]        
        simulate_row_column_task(path_to_data_dir, duration, initial_beliefs, dyad, row_player, column_player, seed)
    
    
         
 

