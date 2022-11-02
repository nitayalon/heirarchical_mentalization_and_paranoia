library(tidyverse)
library(dplyr)
library(ggplot2)

utility_function <- function(action, gamma)
{
  # Max(a_t - threshold, 0)
  return(max(c(action-gamma, 0)))
}

policy <- function(low, high, gamma, temp=0.1, seed=6431) {
  # Simple SoftMax policy
  filtered_actions = actions[actions >= low & actions < high]
  q_values <- sapply(filtered_actions, function(x){utility_function(x, gamma)})
  probability <- exp(q_values / temp) / sum(exp(q_values / temp))
  optimal_action = sample(filtered_actions, 1, prob = probability)
  return(list(probability=probability, q_values=q_values, action = optimal_action))
}

inverse_rl <- function(history, offers, temp=0.1)
{
  # Tom(0) IRL solver
  prior <- data.frame(p = rep(1 / length(thresholds), length(thresholds)))
  for (i in 1:length(offers))
  {
    observation_probability = c()
    low = history[i, ]$low
    high = history[i, ]$high
    for (threshold in thresholds)
    {
      filtered_actions = actions[actions >= low & actions < high]
      observation_q_value = utility_function(offers[i], threshold)
      q_values <- sapply(filtered_actions, function(x){utility_function(x, threshold)})
      observation_probability <- c(observation_probability, exp(observation_q_value / temp) / sum(exp(q_values / temp)))
    }
    posterior = observation_probability * prior[, i]
    posterior = posterior / sum(posterior)
    prior = cbind(prior, posterior)
  }
  posterior_belief = as.data.frame(t(prior)) %>% rename(low=V1, medium=V2, high=V3) %>% 
    mutate(trial = 0:length(offers))
  return(posterior_belief)
}
