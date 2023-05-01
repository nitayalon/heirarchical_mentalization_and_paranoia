
# DoM(0) receiver methods -------------------------------------------------

ObservationLikelihood <- function(previous_observation, previous_response)
{
  likelihood <- c()
  for(i in 1:length(gamma))
  {
    g=gamma[i]
    if (g == 0)
    {
      lk = 1/length(offers)
    }
    else
    {
      lk = RationalRandom(g, previous_observation, previous_response)[2] 
    }
    likelihood[i] = lk    
  }
  return(likelihood)
}

UpdateBelief <- function(belief, previous_observation, previous_response, iteration)
{
  current_belief = belief[iteration - 1, ]
  likelihood <- ObservationLikelihood(previous_observation, previous_response) 
  updated_belief = current_belief * likelihood
  belief[iteration, ] = updated_belief / sum(updated_belief)
  return(belief)
}

FixedOpponentQV <- function(g, current_offer, total_time, iteration)
{
  immediate_reward <- current_offer * responses
  trajectories <- 
    for(t in seq(1,(total_time-iteration)))
    {
      offer_a = RationalRandomExpectation(g, offer, T)
      offer_r = RationalRandomExpectation(g, offer, F)
    }
}

Planning <- function(previous_response, offer, belief, iteration)
{
  updated_belief <- UpdateBelief(belief, offer, previous_response, iteration)
  reward <- offer * responses
  future_rewards_a <- c()
  future_rewards_r <- c()
  for(i in 1:length(gamma))
  {
    g=gamma[i]
    if (g == 0)
    {
      efr_a = 0.5
      efr_r = 0.5
    }
    else
    {
      efr_a = RationalRandomExpectation(g, offer, T)
      efr_r = RationalRandomExpectation(g, offer, F)
    }
    future_rewards_a[i] = efr_a
    future_rewards_r[i] = efr_r
  }
  q_a = reward[1] + future_rewards_a  %*% updated_belief[iteration, ]
  q_r = reward[2] + future_rewards_r  %*% updated_belief[iteration, ]
  return(c(q_a, q_r))
}


RecursiveSplit <- function(action, self_thresold, other_thresold, offer, 
                           iteration, 
                           max_depth)
{
  if(iteration == max_depth)
  {
    q_max = max(offer - self_thresold, 0.0)
    return((c(q_max, offer - self_thresold, 0.0)))
  }
  reward = (offer - self_thresold) * action
  counter_offer = RationalRandomExpectation(other_thresold, offer, action)
  return(reward + max(RecursiveSplit(T,self_thresold, other_thresold,
                                     counter_offer, iteration+1,
                                     max_depth),
                      RecursiveSplit(F,self_thresold, other_thresold,
                                     counter_offer, iteration+1,
                                     max_depth)))
}

RecursiveSplit(T, 0.1, 0.2, 0.6, 1, 10)
RecursiveSplit(F, 0.1, 0.2, 0.6, 1, 10)
