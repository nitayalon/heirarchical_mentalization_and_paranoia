from agents_models.intentional_agents.tom_zero_agents.tom_zero_sender import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_receiver import *
from typing import Optional, Union


class TomOneSubjectBelief(TomZeroSubjectBelief):
    def __init__(self, intentional_threshold_belief, opponent_model: Optional[Union[DoMZeroSender, SubIntentionalAgent]],
                 history: History):
        super().__init__(intentional_threshold_belief, opponent_model, history)


class DoMOneReceiver(DoMZeroReceiver):
    def __init__(self, actions, softmax_temp: float, threshold: Optional[float],
                 prior_belief: np.array,
                 opponent_model: Optional[Union[DoMZeroSender, SubIntentionalAgent]],
                 seed: int):
        super().__init__(actions, softmax_temp, threshold, prior_belief, opponent_model, seed)
        self.belief = TomOneSubjectBelief(prior_belief, self.opponent_model, self.history)
        self.solver = IPOMCP(self.belief, self.environment_model, self.exploration_policy, self.utility_function, seed)
