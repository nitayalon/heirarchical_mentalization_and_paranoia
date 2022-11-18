from agents_models.abstract_agents import *


class ToMZeroSubject(DoMZeroAgent):

    def update_belief(self, action, observation):
        observation_likelihood_per_type = np.zeros_like(self.belief)
        i = 0
        for gamma in self.belief[:, 0]:
            self.opponent_model.threshold = gamma
            relevant_actions, q_values, probabilities = self.opponent_model.forward(observation, action)
            observation_likelihood = probabilities[np.where(relevant_actions == observation)]
            observation_likelihood_per_type[i] = observation_likelihood
            i += 1
        prior = self.belief[:, -1]
        posterior = observation_likelihood_per_type * prior
        self.belief = np.c_[self.belief, posterior / posterior.sum()]

    def act(self, seed, action=None, observation=None):
        self.update_belief(action, observation)
        self.forward(action, observation)

    def forward(self, action=None, observation=None):
        pass