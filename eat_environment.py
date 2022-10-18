import pandas as pd
from agents_models.agent import *


class EAT:

    def __init__(self, n_trails, seed, endowment):
        self.endowment = endowment
        self.n_trails = n_trails
        self.seed = seed
        self.trail_results = []
        self.subject_posterior_beliefs = []

    def simulate_task(self, subject: Subject, agent: Agent):
        seed = self.seed
        for trial in range(self.n_trails):
            trial_results = self.trial(subject, agent, seed)
            self.trail_results.append(trial_results)
            self.subject_posterior_beliefs.append(subject.posterior_beliefs)
            seed += 1
            print(f'The updated belief of the subject is {subject.posterior_beliefs}')
        experiment_results = pd.DataFrame(self.trail_results)
        subject_posterior_belief = pd.DataFrame(self.subject_posterior_beliefs)
        return experiment_results

    @staticmethod
    def trial(subject, agent, seed):
        offer = agent.act(seed)
        response = subject.act(seed, offer)
        return np.array([offer, response, subject.posterior_mu])

