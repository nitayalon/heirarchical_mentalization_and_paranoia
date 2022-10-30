set.seed(6431)

thresholds <- c(0.2, 0.5, 0.7)

utility_function <- function(action, gamma)
{
  return(max(c(action-gamma, 0)))
}

policy <- function(low, high, gamma, temp=0.1, seed=6431) {
  filtered_actions = actions[actions >= low & actions < high]
  q_values <- sapply(filtered_actions, function(x){utility_function(x, gamma)})
  probability <- exp(q_values / temp) / sum(exp(q_values / temp))
  optimal_action = sample(filtered_actions, 1, prob = probability)
  return(list(probability=probability, q_values=q_values, action = optimal_action))
}


# This illustrates the ToM(-1) IA agent's policy --------------------------


actions <- seq(0.0, 1.0, 0.01)
opponents_threshold = 0.7
history <- data.frame(low = 0, high = 1)
reward = c()
offers <- c()
for (i in 2:25)
{
  low = history[i-1, ]$low
  high = history[i-1, ]$high
  offer <- policy(low, high, thresholds[3])
  offers <- c(offers, offer$action)
  if((1 - offer$action) > opponents_threshold) 
  {
    reward = c(reward, offer$action)
    low = offer$action
  }
  if((1-offer$action) <= opponents_threshold)
  {
    reward = c(reward, 0)
    high = offer$action
  }
  history[i, ] = c(low, high)
}
plot(offers, type='b')
plot(reward)


# This illustrates the ToM(0) inference over a grid  ----------------------

inverse_rl <- function(history, offers, temp=0.1)
{
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
}

