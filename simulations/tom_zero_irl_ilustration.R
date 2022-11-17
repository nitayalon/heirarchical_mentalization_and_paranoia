source('simulations/tom_zero_irl_ilustration_methods.R')
set.seed(6431)

ai_thresholds <- c(0.2, 0.5, 0.9) # we can use this as a span of both agent and subject for multiple simulation
human_thresholds <- c(0.2, 0.5, 0.7) # we can use this as a span of both agent and subject for multiple simulation
# the above can also be specified separately


# This illustrates the ToM(-1) IA agent's policy --------------------------

actions <- seq(0.0, 1.0, 0.01)
trials  <- 25

history_sum <- tom_minus_one(agent_thresholds = ai_thresholds, 
                             subject_thresholds = human_thresholds,
                             actions, trials,
                             plot = T)

ggplot(history_sum, aes(x = trial, y = offers, color = factor(agent_threshold)))+
  geom_line() +
  geom_point(aes(y = ifelse(reward == 0, 0, 1)), color = 'black', alpha = 0.2) +
  geom_point() +
  labs(y = 'Split Kept By Agent', x = 'Trial')+
  scale_color_brewer(name = 'Agent\nThreshold', palette = 'Dark2')+
  scale_alpha(breaks = c(0,1), labels = c('No', 'Yes'), name = 'Rewarded?')+
  facet_wrap(~subject_threshold, labeller = label_both)+
  annotate(geom = 'text', x = trials - 5, y = 0.98, label = 'Rewarded')+
  annotate(geom = 'text', x = trials - 5, y = 0.02, label = 'Unrewarded')+
  theme_bw()+
  theme(text = element_text(size = 14))

# This illustrates the ToM(0) inference over a grid  ----------------------

#adjust the filtering based on the values in 'thresholds'
history   <- history_sum %>% filter(agent_threshold == ai_thresholds[1], subject_threshold == human_thresholds[3])

# actions are now directly input into the inverse_rl function to avoid environment clashes.
irl_inference <- inverse_rl(history, actions, agent_thresholds = thresholds)

posterior = irl_inference$posterior_belief
irl_inference$offer_probabilities
posterior %>%
  pivot_longer(cols = !c(trial), names_to = "Type", values_to = "Probability") %>%
  ggplot(aes(x = trial, y = Probability, colour = Type)) +
  geom_point() +
  geom_line() +
  labs(y = 'p(Type)')+
  theme_bw()
