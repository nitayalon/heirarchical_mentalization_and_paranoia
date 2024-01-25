softmax <- function(q_values, temp=0.01)
{
  softmax_transformation <- exp(q_values / temp)
  return(softmax_transformation / sum(softmax_transformation))
}