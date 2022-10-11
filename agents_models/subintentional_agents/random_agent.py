from agents_models.agent import *


class RandomAgent(Agent):

    def act(self, seed):
        random_number_generator = np.random.default_rng(seed)
        offer = random_number_generator.random()
        return offer



