import numpy as np
from dom_m1_agent import *
from dom_zero_agent import *
from dom_one_agent import * 

if __name__ == "__main__":    
    softmax_transformation(dom_m1_utility(game_1))
    softmax_transformation(dom_m1_utility(game_2))
    dom_zero_agent = DoMZeroPlayer(12, 0.01, 0.99)
    dom_one_agent = DoMOnePlayer(12, 0.01, 0.99)
    updated_beliefs = np.repeat(1/3,3)
    dom_zero_beliefs = np.repeat(1/3,3)
    payoffs = np.array([0,0])
    seed = 6431
    for i in np.arange(0,12):                
        dom_zero_policy = softmax_transformation(dom_zero_agent.dom_0_utility(updated_beliefs))
        dom_one_policy = dom_one_agent.dom_1_expectimax(updated_beliefs, game_1, 0)
        prng = np.random.default_rng(seed)
        dom_zero_action = prng.choice(a=3, p=dom_zero_policy)
        dom_one_action = prng.choice(a=2, p=dom_one_policy)
        reward = game_1[dom_one_action,dom_zero_action]
        payoffs = np.vstack[(payoffs, np.array([reward, -reward]))]
        updated_beliefs = dom_zero_agent.dom_0_irl(observation=dom_one_action,prior=updated_beliefs)
        dom_zero_beliefs = np.vstack([dom_zero_beliefs, updated_beliefs])
    np.savetxt("dom_1_vs_dom_0_zero_sum_game.csv", payoffs, delimiter=",")
    np.savetxt("dom_1_vs_dom_0_zero_sum_game_beliefs.csv", dom_zero_beliefs, delimiter=",")
    
         
 

