library(tidyverse)

# Plotting settings for aliasing on windows
# Enable anti-aliasing on Windows
if(Sys.info()['sysname'] == "Windows"){
  
  trace(grDevices::png, quote({
    if (missing(type) && missing(antialias)) {
      type <- "cairo-png"
      antialias <- "subpixel"
    }
  }), print = FALSE)
  
  
  # Enable anti-aliasing on Windows
  trace(grDevices:::png, quote({
    if (missing(type) && missing(antialias)) {
      type <- "cairo-png"
      antialias <- "subpixel"
    }
  }), print = FALSE)
  
  
}

# Define general plot style and style
base_size = 15
theme_set(theme_classic(base_size = base_size))


# Define path to simulation results
results_path = "~/Max_Planck/Hierarchical-modelling/data/agent_subject/basic_task/softmax/"
dom_levels = list(
  c("0","1"))

agents_beliefs <- c("receiver_beliefs", "sender_beliefs")
directory_pattern <- "DoM%s_receiver_DoM%s_sender_softmax_temp_0.1"
path_to_game_results <- paste0(results_path, directory_pattern)

# Load game results

game_results <- c()
for(dom_level in dom_levels)
{
  receiver_dom_level = dom_level[1]
  sender_dom_level = dom_level[2]
  path_to_experiments_results <- sprintf(path_to_game_results, receiver_dom_level, sender_dom_level)
  path_to_dir <- paste(path_to_experiments_results, "simulation_results" ,sep="/")
  temp = list.files(path=path_to_dir, pattern="*.csv")
  if (length(temp) == 0)
  {
    next
  }
  myfiles = lapply(paste(path_to_dir, temp, sep="/"), read.csv)
  interim_results <- bind_rows(myfiles)
  game_results <- rbind(game_results, cbind(interim_results, receiver_dom_level, sender_dom_level))
}

## Aggregate game results
aggregated_game_results <- game_results %>% 
  mutate(sender_dom_level = factor(sender_dom_level),
         sender_threshold = factor(sender_threshold),
         receiver_dom_level = factor(receiver_dom_level),
         receiver_threshold = factor(receiver_threshold)) %>% 
  group_by(trial_number, sender_dom_level, receiver_dom_level,  receiver_threshold, sender_threshold) %>% 
  summarise(average_sender_reward = mean(sender_reward),
            average_offer = mean(offer),
            average_receiver_reward = mean(receiver_reward),
            response_probability = mean(response))


# Load beliefs

beliefs <- c()
for(dom_level in dom_levels)
{
  receiver_dom_level = dom_level[1]
  sender_dom_level = dom_level[2]
  for(agent_name in agents_beliefs)
  {
    path_to_experiments_results <- sprintf(path_to_game_results, receiver_dom_level, sender_dom_level)
    path_to_dir <- paste(path_to_experiments_results, "beliefs" ,sep="/")
    updated_path = paste(path_to_dir, agent_name, sep="/")
    temp = list.files(path=updated_path, pattern="*.csv")
    if (length(temp) == 0)
    {
      next
    }
    myfiles = lapply(paste(updated_path, temp, sep="/"), read.csv)
    interim_results <- bind_rows(myfiles)
    beliefs <- rbind(beliefs, cbind(interim_results, receiver_dom_level, sender_dom_level))
  }
}

aggregated_beliefs <- beliefs %>% 
  filter(agent_name == "DoM(0)_receiver",
         sender_threshold > 0) %>% 
  mutate(sender_dom_level = factor(sender_dom_level),
         sender_threshold = factor(sender_threshold),
         receiver_dom_level = factor(receiver_dom_level),
         receiver_threshold = factor(receiver_threshold)) %>% 
  group_by(trial_number, agent_name , sender_dom_level, receiver_dom_level,
           receiver_threshold, sender_threshold) %>% 
  summarize(`P(0.0)` = mean(`X0.0`),
         `P(0.1)` = mean(`X0.1`),
         `P(0.5)` = mean(`X0.5`)) 

# Load q-values
dom_levels = list(c("0","1"))

q_values <- c()
for(dom_level in dom_levels)
{
  receiver_dom_level = dom_level[1]
  sender_dom_level = dom_level[2]
  path_to_experiments_results <- sprintf(path_to_game_results, receiver_dom_level, sender_dom_level)
  path_to_dir <- paste(path_to_experiments_results, "q_values" ,sep="/")
  temp = list.files(path=path_to_dir, pattern="*.csv")
  if (length(temp) == 0)
  {
    next
  }
  myfiles = lapply(paste(path_to_dir, temp, sep="/"), read.csv)
  interim_results <- bind_rows(myfiles)
  issue_list <- sapply(myfiles, function(x){class(x$q_value)})
  temp[issue_list == "character"][1]
  myfiles[issue_list == "character"][1]
  q_values <- rbind(q_values, cbind(interim_results, receiver_dom_level, sender_dom_level))
}

## join game results with belief

softmax_temp = 0.1

dom_one_aggregated_q_values <- q_values %>% 
  filter(agent %in% c("DoM(1)_sender")) %>% 
  mutate(sender_dom_level = factor(sender_dom_level),
         sender_threshold = factor(sender_threshold),
         receiver_dom_level = factor(receiver_dom_level),
         receiver_threshold = factor(receiver_threshold)) %>% 
  group_by(trial_number, agent, action , sender_dom_level, receiver_dom_level,
           receiver_threshold, sender_threshold) %>% 
  summarize(q_value = mean(q_value)) %>% 
  mutate(exp_q_value = exp(q_value / softmax_temp)) %>% 
  group_by(trial_number, agent, sender_dom_level, receiver_dom_level,
           receiver_threshold, sender_threshold) %>% 
  mutate(probability = exp_q_value / sum(exp_q_value)) %>% 
  ungroup() %>% 
  select(trial_number, agent, action, sender_threshold, probability)

subintentional_aggregated_q_values <- q_values %>% 
  filter(agent %in% c("DoM(-1)_RRA"), sender_threshold > 0) %>% 
  mutate(sender_dom_level = factor(sender_dom_level),
         sender_threshold = factor(sender_threshold),
         receiver_dom_level = factor(receiver_dom_level),
         receiver_threshold = factor(receiver_threshold)) %>% 
  group_by(trial_number, agent, action, sender_threshold) %>% 
  summarize(q_value = mean(q_value)) %>% 
  mutate(exp_q_value = exp(q_value / softmax_temp)) %>% 
  group_by(trial_number, agent, sender_threshold) %>% 
  mutate(probability = exp_q_value / sum(exp_q_value)) %>% 
  ungroup() %>% 
  select(trial_number, agent, action, sender_threshold, probability)
  
  
aggregated_game_results %>% 
  inner_join(aggregated_beliefs, by = c("trial_number",
                                        "sender_dom_level", "receiver_dom_level",
                                        "receiver_threshold", "sender_threshold")) %>% 
  ggplot() + 
  geom_line(aes(x = trial_number, y = `P(0.0)`, colour = "`P(0.0)`")) + 
  geom_line(aes(x = trial_number, y = `P(0.1)`, colour = "`P(0.1)`")) + 
  geom_line(aes(x = trial_number, y = `P(0.5)`, colour = "`P(0.5)`")) + 
  geom_point(aes(x = trial_number, y = average_offer, colour = "average_offer")) + 
  facet_wrap(vars(factor(sender_threshold)))

random_agent <- q_values %>% 
  filter(agent %in% c("DoM(-1)_RRA")) %>% 
  mutate(sender_threshold = factor(0.0)) %>%  
  group_by(trial_number, sender_threshold, agent) %>% 
  summarize(probability = 1/n(),
            action = seq(0,1,0.05)) %>% 
  mutate(type = "Random")

  
aggregated_q_values %>% 
  rbind(random_agent[,c(1,3,5,2,4)]) %>% 
  ggplot(aes(x=action , y=probability, colour=factor(sender_threshold))) + 
  geom_line() + 
  facet_wrap(vars(factor(trial_number)))

### IRL illustration

dom_one_aggregated_q_values %>% 
  mutate(type = "Intentional") %>% 
  rbind(random_agent[,c(1,3,5,2,4,6)]) %>% 
  filter(trial_number == 1) %>% 
  group_by(trial_number, factor(action)) %>% 
  mutate(posterior_probability = probability / sum(probability)) %>% 
  mutate(name = "DoM(2)") %>% 
  rbind(
  subintentional_aggregated_q_values %>% 
    mutate(type = "Subintentional") %>% 
    rbind(random_agent[,c(1,3,5,2,4,6)]) %>% 
    filter(trial_number == 1) %>% 
    group_by(trial_number, factor(action)) %>% 
    mutate(posterior_probability = probability / sum(probability)) %>% 
    mutate(name = "DoM(0)")) %>% 
  ggplot(aes(x=action , y=posterior_probability, colour=factor(name))) + 
  scale_color_viridis_d(name = expression(widehat(paste('P(',gamma,')'))))+
  geom_smooth(se=FALSE) + 
  facet_wrap(vars(factor(sender_threshold)))

# compressed belief

dom_one_aggregated_q_values %>% 
  mutate(type = "Intentional",
         action = factor(action)) %>% 
  group_by(trial_number, action, type) %>% 
  summarise(probability = sum(probability)) %>% 
  rbind(random_agent %>% 
          ungroup() %>% 
          mutate(action = factor(action)) %>% 
          select(trial_number, action, type, probability)) %>% 
  filter(trial_number == 1) %>% 
  group_by(trial_number, action) %>% 
  mutate(posterior_probability = probability / sum(probability)) %>% 
  mutate(name = "DoM(2)") %>% 
  rbind(
  subintentional_aggregated_q_values %>% 
    mutate(type = "Intentional",
           action = factor(action)) %>% 
    select(trial_number, action, type, probability) %>% 
    rbind(random_agent %>% 
            ungroup() %>% 
            mutate(action = factor(action)) %>% 
            select(trial_number, action, type, probability)) %>% 
    filter(trial_number == 1) %>% 
    group_by(trial_number, action) %>% 
    mutate(posterior_probability = probability / sum(probability)) %>% 
    mutate(name = "DoM(0)")) %>% 
  mutate(action = as.numeric(as.character(action))) %>% 
  ggplot(aes(x=action , y=posterior_probability, colour=factor(name))) + 
  scale_color_viridis_d(name = expression(widehat(paste('P(',gamma,')'))))+
  geom_smooth(se=FALSE) + 
  facet_wrap(vars(type))

# Entropy

dom_one_aggregated_q_values %>% 
  rbind(random_agent[,c(1,3,5,2,4)]) %>% 
  filter(trial_number == 1) %>% 
  group_by(trial_number, factor(action)) %>% 
  mutate(posterior_probability = probability / sum(probability)) %>% 
  summarise(entropy = sum( -posterior_probability * log2(posterior_probability))) %>% 
  mutate(name = "DoM(2)") %>% 
  rbind(
  subintentional_aggregated_q_values %>% 
    rbind(random_agent[,c(1,3,5,2,4)]) %>% 
    filter(trial_number == 1) %>% 
    group_by(trial_number, factor(action)) %>% 
    mutate(posterior_probability = probability / sum(probability)) %>% 
    summarise(entropy = sum( -posterior_probability * log2(posterior_probability))) %>% 
    mutate(name = "DoM(0)")) %>% 
  ggplot(aes(x=action , y=entropy, colour=factor(name))) + 
  scale_color_viridis_d(name = expression(widehat(paste('P(',gamma,')'))))+
  geom_smooth(se=FALSE)


dom_one_aggregated_q_values %>% 
  rbind(random_agent[,c(1,3,5,2,4)]) %>% 
  filter(trial_number == 1) %>% 
  group_by(trial_number, factor(action)) %>% 
  mutate(posterior_probability = probability / sum(probability)) %>% 
  ggplot(aes(x=action , y=posterior_probability)) + 
  scale_color_viridis_d() +
  geom_smooth(se=FALSE) + 
  facet_wrap(vars(sender_threshold))
