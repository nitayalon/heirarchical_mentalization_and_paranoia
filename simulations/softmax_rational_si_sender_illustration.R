FullyRandomIRL <- function(offer, response, threshold, offers)
{
  utility = ((1-offers) - threshold)
  probability = return(1/len(offers))
  return(list(acceptence_weigths = rep(0,length(offers)),
              probability = probability,
              q_values = utility))
}

RationalRandom <- function(offer, response, threshold, offers, temperature=0.05)
{
 acceptence_weigths = Filteroffers(offers, UpdateBounds(offer, threshold, response))
 utility = ((1-offers) - threshold) * acceptence_weigths
 probs = exp(utility / temperature) / sum(exp(utility / temperature)) 
 return(list(acceptence_weigths = acceptence_weigths,
             probability = probs / sum(probs),
             q_values = utility))
}

UpdateBounds <- function(offer, gamma, response)
{
  UL = 1.0 - gamma
  LL = 0.0
  if(response)
  {
    UL = min(offer, 1.0 - gamma)
  }
  else
  {
    LL = offer
  }
  return(c(LL,UL))
}

Filteroffers <- function(offers, bounds)
{
  w <- offers >= bounds[1] & offers <= bounds[2]
  w_prime = w + 0.1
  return(w_prime / sum(w_prime))
}


results <- lapply(c(0.1,0.2,0.4), function(x){RationalRandom(0.15, T, x, offers)})
policies <- sapply(results, function(x){x$probability})
acceptence_weigths <- sapply(results, function(x){x$acceptence_weigths})
matplot(offers, policies, ylab = "Posterior Probability", xlab = "Offer")
matplot(offers, policies / rowSums(policies), ylab = "Posterior Probability", xlab = "Offer")
