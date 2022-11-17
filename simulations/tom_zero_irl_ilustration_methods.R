library(tidyverse)
library(dplyr)
library(ggplot2)

utility_function <- function(action, gamma)
{
  # u(a_t, gamma) = a_t - gamma: linear utility
  return(action-gamma)
}

policy <- function(low, high, gamma, actions, temp=0.05, seed=6431) 
  {
  # If the last offer was rejected
  upper_limit = actions < high
  # If the last offer was rejected but equal to the threshold
  if(high < gamma){upper_limit = actions <= gamma}
  filtered_actions = actions[actions >= low & upper_limit]
  q_values <- sapply(filtered_actions, function(x){utility_function(x, gamma)})
  # SoftMax policy
  probability <- exp(q_values / temp) / sum(exp(q_values / temp))
  optimal_action = sample(filtered_actions, 1, prob = probability)
  return(list(probability=probability, q_values=q_values, 
              action = optimal_action, action_probability = probability[filtered_actions == optimal_action]))
}

inverse_rl <- function(history, actions, agent_thresholds, temp=0.05)
{
  offers = history$offers
  # Tom(0) IRL solver
  belief <- data.frame(p = rep(1 / length(agent_thresholds), length(agent_thresholds)))
  offer_probabilities <- data.frame(p = rep(0, length(agent_thresholds)))
  for (i in 2:length(offers))
  {
    observation_probability = c()
    low = history[i-1, ]$low
    high = history[i-1, ]$high
    observation = offers[i]
    for (threshold in agent_thresholds)
    {
      upper_limit = actions < high
      if(high < threshold){upper_limit = actions <= threshold}
      filtered_actions = actions[actions >= low & actions < upper_limit]
      # If the offer exceeds the limits then we nullify the probability
      observation_q_value = utility_function(observation, threshold)
      q_values <- sapply(filtered_actions, function(x){utility_function(x, threshold)})
      likelihood = min(exp(observation_q_value / temp) / sum(exp(q_values / temp)), 1.0)
      observation_probability <- c(observation_probability, likelihood)
    }
    posterior = observation_probability * belief[, i-1]
    posterior = posterior / sum(posterior)
    belief = cbind(belief, posterior)
    offer_probabilities = cbind(offer_probabilities, observation_probability)
  }
  posterior_belief = as.data.frame(t(belief)) %>% rename(low=V1, medium=V2, high=V3) %>% 
    mutate(trial = 0:(ncol(belief)-1))
  offer_probabilities = as.data.frame(t(offer_probabilities)) %>% rename(low=V1, medium=V2, high=V3) %>% 
    mutate(trial = 0:(ncol(offer_probabilities)-1))
  return(list(posterior_belief = posterior_belief,offer_probabilities = offer_probabilities))
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
                            offers = 0, 
                            offer_probability = 0.0,
                            reward = 0,
                            agent_threshold = agents_threshold,
                            subject_threshold = subjects_threshold,
                            trial = 1)
      for (i in 2:trials)
      {
        offers <- c()
        action_probabilities <- c()
        reward <- offers
        low = history[i-1, ]$low
        high = history[i-1, ]$high
        offer <- policy(low, high, agents_threshold, actions)
        offers <- c(offers, offer$action)
        action_probabilities <- c(action_probabilities, offer$action_probability)
        if((1 - offer$action) >= subjects_threshold)
        {
          reward = c(reward, offer$action)
          low = offer$action
        }
        if((1-offer$action) < subjects_threshold)
        {
          reward = c(reward, 0)
          high = offer$action
        }
        history[i, ] = c(low, high, offers, action_probabilities, reward, agents_threshold, subjects_threshold, i)
      }
      if(plot)
      {
        p = history %>% 
        ggplot(aes(trial, offers, colour=reward)) + 
          geom_point(size = 1.5) + 
          geom_hline(yintercept = agents_threshold, colour="blue", size = 1.0) + 
          geom_hline(yintercept = (1-subjects_threshold), colour="red", size = 1.0) + 
          scale_color_viridis_c()+
          ggtitle(sprintf("Agent theshold %s, subject threshold %s",agents_threshold, subjects_threshold)) + 
          theme_classic()
        print(p)
      }
      outcomes_subject[[j]] <- history
    }
    outcomes_agent[[k]] <- do.call(rbind, outcomes_subject)
  }
  return(do.call(rbind, outcomes_agent))
}
