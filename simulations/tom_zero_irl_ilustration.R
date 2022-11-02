set.seed(6431)

thresholds <- c(0.2, 0.5, 0.7)

# This illustrates the ToM(-1) IA agent's policy --------------------------

actions <- seq(0.0, 1.0, 0.01)

## This corresponds to the minimal amount the subject has to get
opponents_threshold = 0.7
history <- data.frame(low = 0, high = 1)
reward = c()
offers <- c()
## This corresponds to the minimal amount the agent has to get
persona <- thresholds[3]

for (i in 2:25)
{
  low = history[i-1, ]$low
  high = history[i-1, ]$high
  offer <- policy(low, high, persona)
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
abline(h = persona, col = "blue")
plot(reward)


# This illustrates the ToM(0) inference over a grid  ----------------------

posterior <- inverse_rl(history, offers)
posterior %>% 
  pivot_longer(cols = !c(trial), names_to = "Type", values_to = "Probability") %>% 
ggplot(aes(x = trial, y = Probability, colour = Type)) + 
  geom_point() + 
  geom_line() + 
  theme_bw()
