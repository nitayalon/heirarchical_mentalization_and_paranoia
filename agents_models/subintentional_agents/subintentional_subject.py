from agents_models.abstract_agents import *


class Subject(SubIntentionalAgent):

    def act(self, seed, action=None, observation=None):
        if action >= self.threshold:
            return True
        return False
