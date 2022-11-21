from agents_models.abstract_agents import *


class Subject(SubIntentionalModel):

    def forward(self, action=None, observation=None):
        pass

    def act(self, seed, action=None, observation=None):
        if action >= self.threshold:
            return True
        return False
