source("simulations/dom_1_sender_policy_compression.R")

# Noisy DoM(1) q-values

q_values %>% 
  filter(agent %in% c("DoM(1)_sender"),
         seed == 110, trial_number == 1)
  

q_values %>% 
  filter(agent %in% c("DoM(1)_sender")) %>% 
  mutate(sender_dom_level = factor(sender_dom_level),
         sender_threshold = factor(sender_threshold),
         receiver_dom_level = factor(receiver_dom_level),
         receiver_threshold = factor(receiver_threshold)) %>% 
  ggplot(aes(action, q_value, colour = factor(seed))) + 
  geom_point() +
  geom_line() +
  facet_grid(factor(trial_number)~factor(sender_threshold))

q_values %>% 
  filter(agent %in% c("DoM(1)_sender")) %>% 
  mutate(sender_dom_level = factor(sender_dom_level),
         sender_threshold = factor(sender_threshold),
         receiver_dom_level = factor(receiver_dom_level),
         receiver_threshold = factor(receiver_threshold)) %>% 
  group_by(trial_number, agent, action , sender_dom_level, receiver_dom_level,
           receiver_threshold, sender_threshold) %>% 
  summarize(avg_q_value = mean(q_value),
            std_q_value = sd(q_value)) %>% 
  mutate(snr = std_q_value / avg_q_value) %>% 
  ggplot(aes(action, avg_q_value, colour = factor(trial_number))) + 
  geom_line(aes(action, avg_q_value, linetype = factor(sender_threshold)))

# join DoM(1) Q-values with DoM(0) beliefs
data_for_model <- dom_one_aggregated_q_values %>%
  ungroup() %>% 
  select(trial_number , action , sender_threshold, q_value) %>% 
  inner_join(aggregated_beliefs %>% 
               ungroup() %>% select(trial_number , `P(0.0)` , `P(0.1)`,
                                    sender_threshold), 
             by=c("trial_number","sender_threshold")) 

## understand the DoM(1) q-values

data_for_model %>% 
  ggplot(aes(action, q_value)) + 
  geom_line(aes(linetype=sender_threshold, color=`P(0.0)`)) + 
  facet_grid(.~trial_number)+
  scale_color_viridis_c()

## we are interested in the action probability prediction
set.seed(6431)
random_trials <- sample(unique(data_for_model$trial_number), 3)
train <- data_for_model %>% filter(!(trial_number %in% random_trials),
                                   sender_threshold == 0.1) 
test <- data_for_model %>% filter(trial_number %in% random_trials,
                                  sender_threshold == 0.1)

linear_model <- lm(q_value ~ ., data = train)
summary(linear_model)
test$predicted_q_values <- predict(linear_model, test)
test_loss <- sqrt(mean((test$predicted_q_values - test$q_value) ** 2))

# predicted behaviour

test %>% 
  ggplot() +
  geom_line(aes(action, q_value, colour = "true")) + 
  geom_line(aes(action, predicted_q_values, colour = "predicted")) + 
  facet_wrap(vars(trial_number))


test %>% 
  mutate(exp_qv = exp(q_value/softmax_temp),
         exp_hat_qv = exp(predicted_q_values/softmax_temp)) %>% 
  group_by(trial_number, sender_threshold) %>% 
  mutate(true_policy = exp_qv / sum(exp_qv),
         predicted_policy = exp_hat_qv / sum(exp_hat_qv)) %>% 
  arrange(trial_number, sender_threshold) %>% 
  ggplot() +
  geom_smooth(aes(action, true_policy)) + 
  geom_smooth(aes(action, predicted_policy))
