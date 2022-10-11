import numpy as np

from agents_models.agent import *
from scipy.stats import norm


class ToMZeroSubject(Subject):

    def __init__(self, preference, prior_weights: np.array, endowment, softmax_temp):
        super().__init__(preference, endowment, softmax_temp)
        self.prior_weights = prior_weights
        self.prior_mu = 0.0
        self.prior_sigma = 1.0
        self.posterior_mu = 0.0
        self.posterior_sigma = 1.0
        self.uniform_probability = 1 / 1000
        self.posterior_beliefs = prior_weights
        self.observations = []

    @property
    def prior_weights(self):
        return self._prior_weights

    @prior_weights.setter
    def prior_weights(self, prior_weights):
        if prior_weights.sum() != 1:
            raise Exception("prior weights must sum to one")
        self._prior_weights = prior_weights

    @staticmethod
    def _inverse_offer_computation(offer):
        return np.log(offer / (1-offer))

    def belief_update(self, observation):
        self.observations.append(observation)
        intentional_agent_loglikelihood = self.intentional_agent_likelihood()
        intentional_agent_posterior_probability = np.exp(intentional_agent_loglikelihood) * self.posterior_beliefs[0]
        random_agent_posterior_probability = self.uniform_probability * self.posterior_beliefs[1]
        posterior_belief = intentional_agent_posterior_probability / \
                           (intentional_agent_posterior_probability + random_agent_posterior_probability)
        self.posterior_beliefs = np.array([posterior_belief, 1-posterior_belief])

    def intentional_agent_likelihood(self):
        variance_ratio_coefficient = self.prior_sigma + self.prior_sigma/len(self.observations)
        updated_mean = self.prior_sigma / variance_ratio_coefficient * self.prior_mu + \
                       self.prior_sigma / variance_ratio_coefficient * np.mean(self.observations)
        updated_variance = 1 / (1 / self.prior_sigma + len(self.observations) / self.prior_sigma)
        self.posterior_mu = updated_mean
        self.posterior_sigma = updated_variance
        return norm(self.posterior_mu, np.sqrt(1.0)).logpdf(self.observations).sum()

    def act(self, seed, offer):
        self.belief_update(self._inverse_offer_computation(offer))
        random_number_generator = np.random.default_rng(seed)
        accept_offer = offer * self.endowment - self.preference
        reject_offer = 0.0
        response_probabilities = np.exp(np.array([reject_offer, accept_offer]) / self.softmax_temp) / \
                                 np.exp(np.array([accept_offer, reject_offer]) / self.softmax_temp).sum()
        response = random_number_generator.choice(2, p=response_probabilities)
        return response

