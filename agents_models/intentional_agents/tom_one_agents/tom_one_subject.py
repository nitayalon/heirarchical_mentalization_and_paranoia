from agents_models.intentional_agents.tom_zero_agents.tom_zero_sender import *
from agents_models.intentional_agents.tom_zero_agents.tom_zero_receiver import *
from typing import Optional


class TomOneSubjectBelief(TomZeroSubjectBelief):
    def __init__(self, intentional_threshold_belief, opponent_model: Optional[DoMZeroSender, SubIntentionalAgent]):
        super().__init__(intentional_threshold_belief, opponent_model)


class DoMOneReceiver(DoMZeroReceiver):
    def __init__(self, actions, softmax_temp: float, prior_belief: np.array,
                 opponent_model: Optional[DoMZeroSender, SubIntentionalAgent],
                 seed: int, alpha: Optional[float] = None):
        super().__init__(actions, softmax_temp, prior_belief, opponent_model, seed, alpha)
        self.belief = TomOneSubjectBelief(prior_belief, self.opponent_model)

    def utility_function(self, action, observation, theta_hat=None, final_trial=True):
        """

        :param theta_hat: float - representing the true persona of the opponent
        :param final_trial: bool - indicate if last trial or not
        :param action: bool - either True for accepting the offer or False for rejecting it
        :param observation: float - representing the current offer
        :return:
        """
        game_reward = (1 - action) * observation
        return game_reward

