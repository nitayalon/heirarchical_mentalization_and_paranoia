dom_1_expectimax <- function(prior_beliefs, payout_matrix, iteration)
{
  planning = lapply(1:2, function(x){recursive_tree_span(x,prior_beliefs, payout_matrix, iteration)}) 
  q_values = sapply(planning, function(x){x$q_value})
  policy = softmax(q_values - mean(q_values))
  return(list(planning = planning, policy = policy))
}

recursive_tree_span <- function(action, beliefs, payout_matrix, iteration)
{
  reward = dom_1_expected_utility(action, beliefs, payout_matrix)
  updated_belief = round(dom_0_irl(beliefs, action),3)
  # halting condition
  if(iteration >= 12)
  {
    return(list(q_value = reward, branch = data.frame(action = action, 
                                                      q_value = reward,
                                                      reward = reward,
                                                      iteration = iteration),
                beliefs = c(action = action,
                            iteration = iteration,
                            p_random = updated_belief[1],
                            p_1 = updated_belief[2],
                            p_2 = updated_belief[3])
    ))    
  }
  actions <- c(1,2)
  future_q_values = lapply(1:2,function(x){recursive_tree_span(actions[x],updated_belief, payout_matrix, iteration+1)})
  q_value = reward + max(sapply(future_q_values, function(x){x$q_value}))
  branch = data.frame(action = action, q_value = q_value, reward = reward, iteration = iteration)
  beliefs = beliefs = c(action = action,
                        iteration = iteration,
                        p_random = updated_belief[1],
                        p_1 = updated_belief[2],
                        p_2 = updated_belief[3])
  sub_tree = rbind(branch, bind_rows(lapply(future_q_values,function(x){x$branch})))
  belief_dynamics = rbind(beliefs, bind_rows(lapply(future_q_values,function(x){x$beliefs})))
  return(list(q_value = q_value, branch = sub_tree, beliefs = belief_dynamics))    
}

dom_1_expected_utility <- function(action, beliefs, payout_matrix)
{
  dom_zero_policy = softmax(dom_0_utility(beliefs))
  expected_reward <- payout_matrix[action,] %*% dom_zero_policy
  return(expected_reward)
}
