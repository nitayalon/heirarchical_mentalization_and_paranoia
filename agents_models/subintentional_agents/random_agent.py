from agents_models.agent import *


class RandomAgent(Agent):

    def act(self, seed):
        random_number_generator = np.random.default_rng(seed)
        offer = random_number_generator.random() * 13 - 6.5
        transformed_offer = 1 / (1 + np.exp(-offer))
        return transformed_offer



