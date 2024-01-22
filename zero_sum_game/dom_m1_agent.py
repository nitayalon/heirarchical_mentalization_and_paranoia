import numpy as np

class DoMM1Player:

  def __init__(self, softmax_temperature:float, discount_factor:float) -> None:
    self.softmax_temperature = softmax_temperature
    self.discount_factor = discount_factor

  def act(self, prior_beliefs, payout_matrix, iteration):
    q_values = dom_m1_utility(payout_matrix)
    return softmax_transformation(q_values, self.softmax_temperature)
      
def softmax_transformation(q_values: np.array, temperature=0.01) -> np.array:
    softmax_transformation = np.exp(q_values / temperature)
    return softmax_transformation / np.sum(softmax_transformation)

def dom_m1_utility(game:np.array):
  return np.matmul(game,np.array([1/2, 1/2, 1/2]))

game_1 = np.array([[4,-1,2],[4,-1,-2]])
game_2 = np.array([[-1,4,-2],[-1,4,2]])