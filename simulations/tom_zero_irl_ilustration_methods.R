library(tidyverse)
library(dplyr)
library(ggplot2)

utility_function <- function(action, gamma)
{
  # Max(a_t - threshold) - linear utility
  return(action-gamma)
}

policy <- function(low, high, gamma, actions, temp=0.05, seed=6431) {
  # Simple SoftMax policy
  upper_limit = actions < high
  if(high < gamma){upper_limit = actions <= gamma}
  filtered_actions = actions[actions >= low & upper_limit]
  q_values <- sapply(filtered_actions, function(x){utility_function(x, gamma)})
  probability <- exp(q_values / temp) / sum(exp(q_values / temp))
  optimal_action = sample(filtered_actions, 1, prob = probability)
  return(list(probability=probability, q_values=q_values, action = optimal_action))
}

inverse_rl <- function(history, actions, offers, temp=0.05)
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
      upper_limit = actions < high
      if(high < threshold){upper_limit = actions <= threshold}
      filtered_actions = actions[actions >= low & actions < upper_limit]
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

tom_minus_one <- function(agent_thresholds, subject_thresholds, actions, trials, plot = F)
{
  outcomes_subject <- list()
  outcomes_agent   <- list()
  for(k in 1:length(agent_thresholds))
  {
    for(j in 1:length(subject_thresholds))
    {
      ## This corresponds to the minimal amount each party wants
      agents_threshold   <- agent_thresholds[k]   # changed the name of the variable for interpretability
      subjects_threshold <- subject_thresholds[j] # changed the name of the variable for interpretability
      history <- data.frame(low = 0, high = 1,
                            offers = 0, reward = 0,
                            agent_threshold = agents_threshold,
                            subject_threshold = subjects_threshold,
                            trial = 1)
      for (i in 2:trials)
      {
        offers <- c()
        reward <- offers
        low = history[i-1, ]$low
        high = history[i-1, ]$high
        offer <- policy(low, high, agents_threshold, actions)
        offers <- c(offers, offer$action)
        if((1 - offer$action) > subjects_threshold)
        {
          reward = c(reward, offer$action)
          low = offer$action
        }
        if((1-offer$action) <= subjects_threshold)
        {
          reward = c(reward, 0)
          high = offer$action
        }
        history[i, ] = c(low, high, offers, reward, agents_threshold, subjects_threshold, i)
      }
      if(plot)
      {
        plot(history$offers, type='b')
        abline(h = history$agents_threshold, col = "blue")
        plot(history$reward)
      }
      outcomes_subject[[j]] <- history
    }
    outcomes_agent[[k]] <- do.call(rbind, outcomes_subject)
  }
  return(do.call(rbind, outcomes_agent))
}
