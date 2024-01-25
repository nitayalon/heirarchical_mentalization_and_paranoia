load_simulations_results <- function(path_to_results_dir, 
                                     task_name,
                                     experiment_name, 
                                     directory_pattern,
                                     dom_levels,
                                     include_mental_state = F)
{
  game_results <- c()
  path_to_game_results <- paste(path_to_results_dir, experiment_name, 
                                directory_pattern, sep="/") 
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
  game_results$task_name = task_name
  
  sender_beliefs <- c()
  dom_0_receiver_beliefs <- c()
  dom_2_receiver_beliefs <- c()
  
  for(dom_level in dom_levels)
  {
    receiver_dom_level = dom_level[1]
    sender_dom_level = dom_level[2]
    for(agent_name in agents_beliefs)
    {
      path_to_experiments_results <- sprintf(path_to_game_results, receiver_dom_level, sender_dom_level)
      path_to_dir <- paste(path_to_experiments_results, "beliefs" ,sep="/")
      updated_path = paste(path_to_dir, agent_name, sep="/")
      if(agent_name == "sender_beliefs" & sender_dom_level == "1")
      {
        temp = list.files(path=updated_path, pattern="*.csv")
      }
      else
      {
        temp = list.files(path=updated_path, pattern="*.csv")
      }
      if (length(temp) == 0)
      {
        next
      }
      myfiles = lapply(paste(updated_path, temp, sep="/"), read.csv)
      interim_results <- bind_rows(myfiles)
      if(agent_name == "sender_beliefs")
      {
        sender_beliefs <- rbind(sender_beliefs, cbind(interim_results, receiver_dom_level, sender_dom_level))
      }
      else
      {
        if(receiver_dom_level == "0")
        {
          dom_0_receiver_beliefs <- rbind(dom_0_receiver_beliefs, cbind(interim_results, receiver_dom_level, sender_dom_level))
        }
        else
        {
          dom_2_receiver_beliefs <- rbind(dom_2_receiver_beliefs, cbind(interim_results, receiver_dom_level, sender_dom_level))
        }
      }
    }
  }
  dom_0_receiver_beliefs$task_name = task_name
  dom_2_receiver_beliefs$task_name = task_name
  sender_beliefs$task_name = task_name
  mental_state <- c()
  
  if(include_mental_state)
  {
    for(dom_level in list(c("0","1")))
    {
      receiver_dom_level = dom_level[1]
      sender_dom_level = dom_level[2]
      for(agent_name in agents_beliefs)
      {
        path_to_experiments_results <- sprintf(path_to_game_results, receiver_dom_level, sender_dom_level)
        path_to_dir <- paste(path_to_experiments_results, "beliefs/receiver_mental_state" ,sep="/")
        temp = list.files(path=path_to_dir, pattern="*.csv")
        if (length(temp) == 0)
        {
          next
        }
        myfiles = lapply(paste(path_to_dir, temp, sep="/"), read.csv)
        interim_results <- bind_rows(myfiles)
        mental_state <- rbind(mental_state, cbind(interim_results, receiver_dom_level, sender_dom_level))
      }
    }
  }
  
  return(list(game_results = game_results, 
              sender_beliefs = sender_beliefs, 
              dom_0_receiver_beliefs = dom_0_receiver_beliefs, 
              dom_2_receiver_beliefs = dom_2_receiver_beliefs, 
              mental_state=mental_state))
}

load_nested_beliefs <- function(updated_path)
{
  path_to_nested_beliefs <- paste(updated_path, "nested_beliefs", sep="/")
  nested_beliefs = list.files(path=path_to_nested_beliefs, pattern="*.csv")
  
  path_to_zero_order_beliefs <- paste(updated_path, "type_belief", sep="/")
  zero_order_beliefs = list.files(path=path_to_zero_order_beliefs, pattern="*.csv")
  
  all_files = list(nested_beliefs, zero_order_beliefs) 
  return(all_files)
}