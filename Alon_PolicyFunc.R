library(tidyverse)
library(dplyr)
library(ggplot2)
library(patchwork)

utility_function <- function(action, gamma)
{
  # Max(a_t - threshold, 0)
  return(max(c(action-gamma, 0)))
}

# actions are now directly input into the policy function to avoid environment clashes.
policy <- function(low, high, gamma, actions, temp=0.1, seed=6431) {
  # Simple SoftMax policy
  filtered_actions = actions[actions >= low & actions < high]
  q_values <- sapply(filtered_actions, function(x){utility_function(x, gamma)})
  probability <- exp(q_values / temp) / sum(exp(q_values / temp))
  optimal_action = sample(filtered_actions, 1, prob = probability)
  return(list(probability=probability, q_values=q_values, action = optimal_action))
}

# actions are now directly input into the inverse_rl function to avoid environment clashes.
inverse_rl <- function(history, actions, agent_thresholds, temp=0.1)
{
  # Tom(0) IRL solver
  prior <- data.frame(p = rep(1 / length(agent_thresholds), length(agent_thresholds)))
  offer_n <- length(history$offers)
  offers  <- history$offers
  for (i in 1:offer_n)
  {
    observation_probability = c()
    low = history[i, ]$low
    high = history[i, ]$high
    for (threshold in agent_thresholds)
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
    mutate(trial = 0:offer_n)
  return(posterior_belief)
}

#ToM (-1)
ToM_minus_one <- function(agent_thresholds, subject_thresholds, actions, trials, plot = 0)
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
     if(plot == 1)
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
