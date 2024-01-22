import numpy as np
from dom_m1_agent import *

class DoMZeroPlayer:
    def __init__(self, game_duration:int, softmax_temperature:float, discount_factor:float) -> None:
        self.game_duration = game_duration
        self.softmax_temperature = softmax_temperature
        self.discount_factor = discount_factor
    
    def irl(self, prior:np.array, observation:int, iteration: int) -> np.array:
        p_0 = 1/2
        p_1 = softmax_transformation(dom_m1_utility(game_1))[observation]
        p_2 = softmax_transformation(dom_m1_utility(game_2))[observation]
        unnormalized_posterior = prior * np.array([p_0, p_1, p_2])
        normalized_posterior = unnormalized_posterior / np.sum(unnormalized_posterior) 
        return(normalized_posterior)
    
    def dom_zero_utility_subroutine(self, action:int , policies_matrix:np.array):
        random_payoff = 1/2 * np.matmul(game_1[:,action],policies_matrix[0,:]) + 1/2 * np.matmul(game_2[:,action],policies_matrix[0,:])
        p_1_payoff = np.matmul(game_1[:,action], policies_matrix[1,:])
        p_2_payoff = np.matmul(game_2[:,action], policies_matrix[2,:])
        return random_payoff,p_1_payoff,p_2_payoff

    def act(self,belief):
        # Optimal policies per type
        pi_0 = np.array([1/2, 1/2])
        pi_1 = softmax_transformation(dom_m1_utility(game_1))
        pi_2 = softmax_transformation(dom_m1_utility(game_2))
        policies_matrix = np.array([pi_0, pi_1, pi_2])
        u_1 = -1 * np.matmul(belief, self.dom_zero_utility_subroutine(0, policies_matrix))
        u_2 = -1 * np.matmul(belief, self.dom_zero_utility_subroutine(1, policies_matrix))
        u_3 = -1 * np.matmul(belief, self.dom_zero_utility_subroutine(2, policies_matrix))
        return(softmax_transformation(np.array([u_1, u_2, u_3])))
