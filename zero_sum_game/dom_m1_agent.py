import numpy as np

class ExpectimaxPlanning:
    def __init__(self, q_values: np.array, branch: np.array, beliefs: np.array) -> None:        
        self.q_values = q_values
        self.branch = branch
        self.beliefs = beliefs


# DoM(-1) row player model
def softmax_transformation(q_values: np.array, temperature=0.01):
    softmax_transformation = np.exp(q_values / temperature)
    return softmax_transformation / np.sum(softmax_transformation)

def dom_m1_utility(game:np.array):
  return np.matmul(game,np.array([1/2, 1/2, 1/2]))

game_1 = np.array([[4,0,2],[4,0,-2]])
game_2 = np.array([[0,4,-2],[0,4,2]])
