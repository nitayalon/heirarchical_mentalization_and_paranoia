from agents_models.agent import *


class IntentionalAgent(Agent):

    def __init__(self, preference, endowment):
        super().__init__(preference, endowment)

    def act(self, seed):
        random_number_generator = np.random.default_rng(seed)
        offer = random_number_generator.normal(loc=self.preference, scale=1.0)
        transformed_offer = 1 / (1 + np.exp(-offer))
        return transformed_offer




