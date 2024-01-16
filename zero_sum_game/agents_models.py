import numpy as np
import functools
from dom_m1_agent import *
from dom_zero_column_player import *


if __name__ == "__main__":    
    softmax_transformation(dom_m1_utility(game_1))
    softmax_transformation(dom_m1_utility(game_2))

    b1_1 = dom_0_irl(np.repeat(1/3,3), 0)
    print(softmax_transformation(dom_0_utility(b1_1)))


 

