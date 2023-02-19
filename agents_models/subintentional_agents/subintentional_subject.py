from agents_models.abstract_agents import *


class SubIntentionalSubject(SubIntentionalModel):

    def __init__(self, actions, softmax_temp: float, threshold: Optional[float] = None):
        super().__init__(actions, softmax_temp, threshold)
        self._name = "DoM(-1)_Subject"

    def utility_function(self, action, observation):
        return (observation - self.threshold) * action

    def forward(self, action=None, observation=None):
        q_values = self.utility_function(self.actions, observation)
        probabilities = self.softmax_transformation(q_values)
        return self.actions, q_values, probabilities

