
# Analysis of simulations --------------------------------------------------

library(tidyverse)

# Load Data ---------------------------------------------------------------

alpha_seq = c('0.1', '0.3', '0.5', '0.7', '0.9')  # parameters to control exploration
threshold = c('0.2', '0.5', '0.8')  # parameters to control threshold of agent
softmax   = c('0.1', '0.5', '1.0')

for (i in alpha_seq){
  for(j in threshold){
    for (k in softmax){
      
    file_path_dat <- paste('/Users/josephbarnby/DataspellProjects/Hierarchical-modelling/data/agent_subject/env_2/',
                           'tom0_subject_tom-1_agent_softmax_',k,
                           '/simulation_results/seed_4356_alpha_',i, 
                           '_threshold_',j,
                           '.csv', 
                           sep = "")  
    x <- read.csv(file_path_dat)  
    
    y1 <- x %>% 
      rename(Trial = X,
             Offer = X0,
             Accept = X1,
             SUB_ret = X2,
             AGT_ret = X3) %>%
      mutate(alpha = as.numeric(i),
             threshold = 1-as.numeric(j),
             softmax = as.numeric(k),
             Trial = Trial + 1)
    
      if(i == 0.1 & j == 0.2 & k == 0.1){
        y2 <- y1
      } else{
        y2 <- rbind(y1, y2)
      }
    }
  }
}

y2 %>%
  ggplot(aes(Trial, 1-Offer, color = factor(alpha), group = factor(alpha), shape = factor(alpha))) +
  geom_jitter(aes(Trial, Accept), color = 'black', height = 0.05, alpha = 0.5)+
  geom_line(linewidth = 1, aes(group = factor(alpha))) +
  facet_grid(threshold ~ softmax, margins = F, labeller = label_both)+
  scale_color_viridis_d(name = expression(paste('Exploration: ',alpha)))+
  scale_shape(name = expression(paste('Exploration: ',alpha)))+
  labs(title = '[Agent DoM = -1 | Subject DoM = 0]', y = 'Agents Offer To Subject')+
  guides(color = guide_legend(override.aes = list(size = 5)))+
  theme_bw()+
  theme(text = element_text(size = 18, family = 'Helvetica Light'),
        legend.direction = 'horizontal',
        legend.position = 'bottom',
        legend.key.size = unit(1, 'cm'))

y3 <- y2 %>%
  group_by(alpha, threshold, softmax) %>%
  mutate(SumAccept = sum(Accept),
         Rew = ifelse(Accept == 1, Offer, 0),
         TotalRew = sum(Rew)) %>%
  dplyr::select(alpha, threshold, softmax, SumAccept, TotalRew) %>%
  distinct()

ggplot(y3, aes(factor(softmax), SumAccept, fill = factor(alpha)))+
  geom_col(position = 'dodge')+
  scale_fill_viridis_d(name = expression(paste('Exploration: ',alpha)))+
  labs(y = 'Total Accepted Offer', x = expression(paste('Decision Temperature: ',tau)))+
  facet_wrap(~threshold)+
    theme_bw()+
  theme(text = element_text(size = 18, family = 'Helvetica Light'),
        legend.direction = 'horizontal',
        legend.position = 'bottom')
