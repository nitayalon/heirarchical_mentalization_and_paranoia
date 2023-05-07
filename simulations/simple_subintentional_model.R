library(ggplot2)
library(tidyverse)

offers <- seq(0,1,0.05)
bounds <- c(0,1)
thresholds <- c(0.0,0.1,0.5)
temp <- 0.1

update_bounds <- function(action, observation, threshold)
{
  high = 1.0-threshold
  low = 0.0
  if(observation)
  {
    high = min(action, high)
  }
  else
  {
    low = action
  }
  # If the opponent plays tricks with us
  if (high < low)
  {
    temp = high
    high = low
    low = temp
  }
  return(c(low,high))
}

softmax_policy <- function(action, observation, threshold)
{
  if (threshold == 0)
  {
    return(rep(1/length(offers), length(offers)))
  }
  bounds <- update_bounds(action, observation, threshold)
  weights <- -0.8 * !(offers > bounds[1] & offers <= bounds[2])
  utility <- (1-offers - threshold)
  q_values <- utility + weights
  probabilities <- exp(q_values / temp) / sum(exp(q_values / temp))
  return(probabilities)
}

probs <- sapply(thresholds, function(x){softmax_policy(0.5, F, x)})

probs %>% as.data.frame() %>% 
  mutate(actions = offers) %>% 
  ggplot() + 
  geom_line(aes(x = actions, y = V1, colour='V1'),
            size=1.0) + 
  geom_line(aes(x = actions, y = V2, colour='V2'),
            size=1.0) + 
  geom_line(aes(x = actions, y = V3, colour='V3'),
            size=1.0) + 
  scale_color_viridis_d(name = expression(paste('P(',gamma, ')')))
