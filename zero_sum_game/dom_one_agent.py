from dom_zero_column_player import *

class DoMOnePlayer:
    
    def __init__(self, game_duration:int, softmax_temperature:float, discount_factor:float) -> None:
        self.game_duration = game_duration
        self.softmax_temperature = softmax_temperature
        self.discount_factor = discount_factor
        self.opponent = DoMZeroPlayer(game_duration, softmax_temperature, discount_factor)

    def dom_1_expected_utility(self, action: int, beliefs:np.array, payout_matrix:np.array):
        dom_zero_policy = softmax_transformation(self.opponent.dom_0_utility(beliefs))
        expected_reward = np.matmul(payout_matrix[action,],dom_zero_policy)
        return expected_reward

    def dom_1_expectimax(self,prior_beliefs, payout_matrix, iteration):
        planning = lapply(1:2, function(x){recursive_tree_span(x,prior_beliefs, payout_matrix, iteration)}) 
        q_values = sapply(planning, function(x){x$q_value})
        policy = softmax(q_values - mean(q_values))
        return(list(planning = planning, policy = policy))

    def recursive_tree_span(action, beliefs, payout_matrix, iteration, depth=12, discount_factor=0.99):
        reward = dom_1_expected_utility(action, beliefs, payout_matrix)
        updated_belief = round(dom_0_irl(beliefs, action),3)
        # halting condition
        if iteration >= depth:
        return ExpectimaxPlanning(reward, np.array(action, reward, reward, iteration),
                    np.array([action, iteration, updated_belief[1], updated_belief[2],updated_belief[3]]))
        actions = np.array([0,1])
        expectimax_tree = functools.partial(recursive_tree_span,
                                            beliefs=updated_belief,
                                            payout_matrix=payout_matrix,
                                            iteration=iteration)
        q_values = list(map(expectimax_tree, actions))    
        q_value = reward + discount_factor * np.max(sapply(future_q_values, function(x){x$q_value}))
        branch = data.frame(action = action, q_value = q_value, reward = reward, iteration = iteration)
        beliefs = beliefs = c(action = action,
                            iteration = iteration,
                            p_random = updated_belief[1],
                            p_1 = updated_belief[2],
                            p_2 = updated_belief[3])
        sub_tree = rbind(branch, bind_rows(lapply(future_q_values,function(x){x$branch})))
        belief_dynamics = rbind(beliefs, bind_rows(lapply(future_q_values,function(x){x$beliefs})))
        return(list(q_value = q_value, branch = sub_tree, beliefs = belief_dynamics))   