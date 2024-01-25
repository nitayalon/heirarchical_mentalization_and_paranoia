dom_0_irl <- function(prior, observation)
{
  p_0 = 1/2
  p_1 = softmax(dom_m1_utility(game_1))[observation]
  p_2 = softmax(dom_m1_utility(game_2))[observation]
  unnormalized_posterior = prior * c(p_0, p_1, p_2)
  normalized_posterior = unnormalized_posterior / sum(unnormalized_posterior) 
  return(normalized_posterior)
}

dom_o_utility_subroutine <- function(action, policies_matrix)
{
  random_payoff = 1/2 * game_1[,action] %*% policies_matrix[1,] + 1/2 * game_2[,action] %*% policies_matrix[1,]
  p_1_payoff = game_1[,action] %*% policies_matrix[2,]
  p_2_payoff = game_2[,action] %*% policies_matrix[3,]
  
  return(c(random_payoff,p_1_payoff,p_2_payoff))
}

dom_0_utility <- function(belief)
{
  # Optimal policies per type
  pi_0 = t(c(1/2, 1/2))
  pi_1 = softmax(dom_m1_utility(game_1))
  pi_2 = softmax(dom_m1_utility(game_2))
  
  policies_matrix <- matrix(c(pi_0, pi_1, pi_2), nrow = 3, byrow = T)
  
  u_1 = -1 * sum(belief %*% dom_o_utility_subroutine(1, policies_matrix))
  u_2 = -1 * sum(belief %*% dom_o_utility_subroutine(2, policies_matrix))
  u_3 = -1 * sum(belief %*% dom_o_utility_subroutine(3, policies_matrix))
  
  return(c(u_1, u_2, u_3))
}
