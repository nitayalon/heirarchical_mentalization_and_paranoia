observations <- seq(0,1,0.05)
actions <- c(T,F)
low = 0.0
high = 1.0
eta = 0.1

q_values <- function(eta)
{
  return(1 - observations - eta)
}

weigths <- function(low, high)
{
  w = !(observations > low & observations <= high)
  return(w * -0.95)
}

policy <- function(eta, low, high)
{
  updated_qv <- q_values(eta) + weigths(low, high)
  policy <- exp(updated_qv/0.1) / sum(exp(updated_qv/0.1))
  return(sum(policy * observations))
}

policy(eta, low, high)
policy(eta, 0.4, high)
policy(eta, 0.5, high)
policy(eta, 0.5, 0.6)

expectimax_tree <- function(depth, low, high)
{
      
}

recursive_span <- function(action, observation, previous_low, previous_high,depth)
{
  reward = observation * action
  if(depth >= 10)
  {
    return(reward)  
  }
  new_low = observation * (1-action) + previous_low*action  
  new_high = observation * (action) + previous_high*(1-action)
  new_observation = policy(0.1, new_low, new_high)
  future_values = sapply(actions, function(x){recursive_span(x,new_observation,new_low,new_high,
                                                             depth + 1)})
  return(reward + max(future_values))
}

recursive_span(T, 0.15, 0.0, 1.0, 0)
recursive_span(F, 0.15, 0.0, 1.0, 0)
