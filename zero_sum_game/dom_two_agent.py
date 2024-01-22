from dom_one_agent import *
from typing import Optional

class DoMTwoPlayer:

    def __init__(self, game_duration:int, softmax_temperature:float, discount_factor:float, prior_zero_beliefs:np.array) -> None:
        self.game_duration = game_duration
        self.softmax_temperature = softmax_temperature
        self.discount_factor = discount_factor
        self.opponent = DoMOnePlayer(game_duration, softmax_temperature, discount_factor)
        self.nested_nested_beliefs = [prior_zero_beliefs]
        self.opponent_policies = self.oppnent_policy_helper_method(0, None)

    def oppnent_policy_helper_method(self, iteration, observation:Optional[int]=None):
        if iteration > 0:
            dom_zero_beliefs = self.opponent.opponent.irl(self.nested_nested_beliefs[-1], observation, iteration)
        else:
            dom_zero_beliefs = self.nested_nested_beliefs[-1]
        p_0 = np.repeat(1/2, 2)
        p_1 = self.opponent.act(dom_zero_beliefs, game_1, iteration)
        p_2 = self.opponent.act(dom_zero_beliefs, game_2, iteration)
        self.opponent_policies = np.array([p_0, p_1, p_2])
        return self.opponent_policies

    def irl(self, prior:np.array, observation:int, iteration: int):
        opponent_policies = self.oppnent_policy_helper_method(iteration, observation)        
        likelihood = opponent_policies[:,observation]        
        unnormalized_posterior = prior * likelihood
        normalized_posterior = unnormalized_posterior / np.sum(unnormalized_posterior) 
        return(normalized_posterior)
    
    def act(self, belief:np.array)-> np.array:         
        u_1 = -1 * np.matmul(belief, self.opponent.opponent.dom_zero_utility_subroutine(0, self.opponent_policies))
        u_2 = -1 * np.matmul(belief, self.opponent.opponent.dom_zero_utility_subroutine(1, self.opponent_policies))
        u_3 = -1 * np.matmul(belief, self.opponent.opponent.dom_zero_utility_subroutine(2, self.opponent_policies))
        return(softmax_transformation(np.array([u_1, u_2, u_3])))
    
    
    