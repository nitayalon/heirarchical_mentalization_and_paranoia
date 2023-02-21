from agents_models.intentional_agents.tom_zero_agents.tom_zero_agent import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_subjects import *
from typing import Optional


class TomOneSubjectBelief(TomZeroSubjectBelief):
    def __init__(self, intentional_threshold_belief, opponent_model: Optional[DoMZeroAgent, BasicModel]):
        super().__init__(intentional_threshold_belief, opponent_model)


class DoMOneSubject(DoMZeroSubject):
    def __init__(self, actions, softmax_temp: float, prior_belief: np.array,
                 opponent_model: Optional[DoMZeroAgent, BasicModel],
                 seed: int, alpha: Optional[float] = None):
        super().__init__(actions, softmax_temp, prior_belief, opponent_model, seed, alpha)
        self.belief = TomOneSubjectBelief(prior_belief, self.opponent_model)

